/**
 * EyeRenderer — physically-based forward eye shader.
 *
 * Models a stylised but physically-motivated eye using a single sphere mesh.
 * The fragment shader partitions the sphere surface into three zones based on
 * the angle from the "gaze axis":
 *
 *   Cornea  (front cap)   — Fresnel specular + iris parallax/depth effect
 *   Limbus  (iris ring)   — coloured iris with parallax offset
 *   Sclera  (white shell) — warm white + subsurface scatter contribution
 *
 * Integration
 * ───────────
 * EyeRenderer is a forward pass rendered to the canvas AFTER the deferred
 * pipeline has already tone-mapped to the default framebuffer.  A fresh depth
 * test is performed against a per-frame cleared depth buffer so eye geometry
 * correctly occludes itself and hair but not the deferred scene.
 *
 * Controls
 * ────────
 *   irisColor   — base iris pigment (default: [0.18, 0.38, 0.75] blue)
 *   irisDilate  — iris radius fraction (0.15 → narrow pupil, 0.32 → dilated)
 *   corneaIOR   — index of refraction for parallax offset (default: 1.376)
 *   specPower   — specular shininess of cornea (default: 256)
 */
import { Shader } from '../core/Shader.js';
import { Mat4 }   from '../math/Mat4.js';
import { Mat3 }   from '../math/Mat3.js';

// ─────────────────────────────────────────────────────────────────────────────
// Eye vertex shader
// ─────────────────────────────────────────────────────────────────────────────
const EYE_VERT = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;
in vec3 a_Normal;

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;
uniform mat3 u_NormalMatrix;

out vec3 v_WorldPos;
out vec3 v_Normal;

void main() {
  vec4 wPos  = u_ModelMatrix * vec4(a_Position, 1.0);
  v_WorldPos = wPos.xyz;
  v_Normal   = normalize(u_NormalMatrix * a_Normal);
  gl_Position = u_ProjectionMatrix * u_ViewMatrix * wPos;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Eye fragment shader
// ─────────────────────────────────────────────────────────────────────────────
const EYE_FRAG = /* glsl */`#version 300 es
precision highp float;

in vec3 v_WorldPos;
in vec3 v_Normal;

// ── Eye parameters
uniform vec3  u_EyeCenter;      // world-space centre of the eye sphere
uniform vec3  u_GazeAxis;       // normalised gaze direction (forward = +Z)
uniform vec3  u_IrisColor;
uniform float u_IrisRadius;     // angular radius of iris (radians), ~0.30
uniform float u_CorneaIOR;      // index of refraction ~1.376
uniform float u_SpecPower;      // cornea specular shininess

// ── Lighting
uniform vec3  u_LightDir;       // normalised world-space toward-light direction
uniform vec3  u_LightColor;
uniform vec3  u_CameraPos;

out vec4 fragColor;

const float PI = 3.14159265358979;

// Simple Schlick Fresnel
vec3 schlick(vec3 F0, float cosTheta) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
  vec3 N = normalize(v_Normal);
  vec3 V = normalize(u_CameraPos - v_WorldPos);
  vec3 L = normalize(u_LightDir);
  vec3 H = normalize(V + L);

  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.0001);
  float NdotH = max(dot(N, H), 0.0);

  // ── Determine which zone we are in based on angle from gaze axis
  // cosine of the angle between the outward sphere normal and the gaze axis
  float cosGaze = dot(N, u_GazeAxis);   // 1 at front, -1 at back

  // Zone thresholds (cosine space)
  float corneaThresh = cos(u_IrisRadius * 1.2);   // slightly wider than iris
  float irisThresh   = cos(u_IrisRadius);

  // 0 = sclera, 1 = iris/limbus, 2 = cornea
  float inIris   = smoothstep(corneaThresh - 0.03, corneaThresh + 0.03, cosGaze);
  float inCornea = smoothstep(irisThresh   - 0.03, irisThresh   + 0.03, cosGaze);

  // ── Sclera shading (white + warm SSS tint from light wrap-around)
  vec3 scleraAlbedo = vec3(0.93, 0.91, 0.88);
  // Light wrap-around for sub-surface scatter illusion
  float wrapLight = (dot(N, L) + 0.4) / 1.4;  // wrapped Lambert
  vec3  scleraDiff = scleraAlbedo * max(0.0, wrapLight) * u_LightColor;
  // Ambient
  vec3  scleraAmb  = scleraAlbedo * 0.12;
  vec3  scleraCol  = scleraDiff + scleraAmb;

  // ── Iris shading
  // Parallax / depth offset: iris appears recessed behind cornea
  // Compute refracted view direction at the cornea surface using Snell's law approx.
  float eta    = 1.0 / u_CorneaIOR;
  vec3  refDir = refract(-V, N, eta);          // refracted into eye medium
  // Virtual iris position: project along refDir by iris recess depth
  float irisDepth = 0.12;                      // fraction of sphere radius
  vec3  irisPoint = v_WorldPos + refDir * irisDepth;
  // Use the projected point as iris UV for slight parallax variation
  vec2  irisUV   = irisPoint.xy - u_EyeCenter.xy;
  // Simple radial pattern for iris detail (ring bands)
  float r        = length(irisUV) * 6.0;
  float pattern  = 0.85 + 0.15 * sin(r * PI);
  vec3  irisCol  = u_IrisColor * pattern * max(0.4, NdotL);

  // Pupil: dark centre of iris
  float pupilR   = u_IrisRadius * 0.35;          // pupil as fraction of iris
  float pupilCos = cos(pupilR);
  float inPupil  = smoothstep(pupilCos - 0.02, pupilCos + 0.02, cosGaze);
  irisCol        = mix(vec3(0.01), irisCol, inPupil);

  // ── Cornea specular (Phong + Fresnel)
  vec3 F0cornea   = vec3(0.04);                  // IOR ~1.376 → R0 ≈ 0.023; use 0.04 for visibility
  vec3 fresnelC   = schlick(F0cornea, NdotV);
  float spec      = pow(NdotH, u_SpecPower) * NdotL;
  vec3 corneaSpec = fresnelC * spec * u_LightColor * 2.5;

  // ── Blend zones
  vec3 color = scleraCol;
  color = mix(color, irisCol, inIris);
  // Add cornea specular across the whole front hemisphere (peaks at inCornea)
  color = color + corneaSpec * inCornea;

  fragColor = vec4(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Procedural eye sphere
// ─────────────────────────────────────────────────────────────────────────────
function generateSphere(radius, stacks, slices) {
  const pos = [], nrm = [], idx = [];
  for (let i = 0; i <= stacks; i++) {
    const phi = (i / stacks) * Math.PI;
    for (let j = 0; j <= slices; j++) {
      const theta = (j / slices) * 2 * Math.PI;
      const x = Math.sin(phi) * Math.sin(theta);
      const y = Math.cos(phi);
      const z = Math.sin(phi) * Math.cos(theta);
      pos.push(x * radius, y * radius, z * radius);
      nrm.push(x, y, z);
    }
  }
  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = i * (slices + 1) + j;
      const b = a + slices + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return {
    positions: new Float32Array(pos),
    normals:   new Float32Array(nrm),
    indices:   new Uint16Array(idx),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
export class EyeRenderer {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   */
  constructor(glContext, stateCache) {
    this.ctx   = glContext;
    this.gl    = glContext.gl;
    this.cache = stateCache;

    this._shader = new Shader(this.gl, EYE_VERT, EYE_FRAG);

    // Build procedural eye sphere VAO
    this._buildMesh();

    // Pre-allocated matrices
    this._modelMat  = new Mat4();
    this._normalMat = new Mat3();

    // Eye shader parameters
    this.irisColor  = [0.18, 0.38, 0.75];
    this.irisRadius = 0.30;                // radians
    this.corneaIOR  = 1.376;
    this.specPower  = 256.0;
  }

  // ─────────────────────────────────────────── public API

  /** @param {number[]} rgb */
  setIrisColor(r, g, b) { this.irisColor = [r, g, b]; }
  /** @param {number} v  0.15 = narrow, 0.32 = dilated */
  setIrisDilate(v)       { this.irisRadius = Math.max(0.1, Math.min(0.5, v)); }
  /** @param {number} v  index of refraction, ~1.376 */
  setCorneaIOR(v)        { this.corneaIOR = v; }
  /** @param {number} v  specular shininess */
  setSpecPower(v)        { this.specPower = v; }

  /**
   * Render all eye nodes to the current framebuffer.
   *
   * @param {Array<{worldMatrix: Float32Array|import('../math/Mat4.js').Mat4, mesh: object}>} nodes
   * @param {import('../scene/Camera.js').Camera} camera
   * @param {import('../scene/Light.js').Light[]} lights
   */
  render(nodes, camera, lights) {
    if (!nodes || nodes.length === 0) return;

    const gl    = this.gl;
    const cache = this.cache;

    camera.updateProjection();
    camera.updateView();

    cache.setDepthTest(true);
    cache.setDepthWrite(true);
    cache.setBlend(false);
    cache.setCullFace(true, gl.BACK);

    cache.useProgram(this._shader.program);

    const dirLight = lights?.find(l => l.type === 'directional') ?? null;
    const lightDir = dirLight
      ? [dirLight.direction.x, dirLight.direction.y, dirLight.direction.z]
      : [0.5, -1.0, 0.5];
    const lightCol = dirLight
      ? [dirLight.color.x * dirLight.intensity, dirLight.color.y * dirLight.intensity, dirLight.color.z * dirLight.intensity]
      : [1.0, 1.0, 1.0];

    this._shader.setVec3('u_LightDir',   -lightDir[0], -lightDir[1], -lightDir[2]);
    this._shader.setVec3('u_LightColor', lightCol[0], lightCol[1], lightCol[2]);
    const cp = camera.position.e;
    this._shader.setVec3('u_CameraPos', cp[0], cp[1], cp[2]);

    // Set view and projection once
    this._shader.setMat4('u_ViewMatrix',       camera.viewMatrix.e);
    this._shader.setMat4('u_ProjectionMatrix', camera.projectionMatrix.e);

    this._shader.setVec3('u_IrisColor',  this.irisColor[0], this.irisColor[1], this.irisColor[2]);
    this._shader.setFloat('u_IrisRadius', this.irisRadius);
    this._shader.setFloat('u_CorneaIOR',  this.corneaIOR);
    this._shader.setFloat('u_SpecPower',  this.specPower);

    for (const node of nodes) {
      if (!node.mesh) continue;
      this._renderEyeNode(node, camera);
    }
  }

  resize(_w, _h) { /* no internal render targets */ }

  destroy() {
    const gl = this.gl;
    this._shader?.destroy();
    if (this._vao) gl.deleteVertexArray(this._vao);
    if (this._posBuf) gl.deleteBuffer(this._posBuf);
    if (this._nrmBuf) gl.deleteBuffer(this._nrmBuf);
    if (this._idxBuf) gl.deleteBuffer(this._idxBuf);
    this._shader = null;
    this._vao = null;
  }

  // ─────────────────────────────────────────── private

  _buildMesh() {
    const gl   = this.gl;
    const mesh = generateSphere(0.5, 24, 24);

    this._vao = gl.createVertexArray();
    gl.bindVertexArray(this._vao);

    this._posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    this._nrmBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._nrmBuf);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.normals, gl.STATIC_DRAW);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(1);

    this._idxBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);

    this._indexCount = mesh.indices.length;

    gl.bindVertexArray(null);
  }

  _renderEyeNode(node, camera) {
    const gl    = this.gl;
    const sh    = this._shader;

    // Model matrix from node
    const mxArr = node.worldMatrix?.e ?? node.worldMatrix;
    sh.setMat4('u_ModelMatrix', mxArr);

    // Normal matrix = transpose(inverse(mat3(model)))
    const m = mxArr;
    const nm = this._normalMat.e;
    nm[0] = m[0]; nm[1] = m[1]; nm[2] = m[2];
    nm[3] = m[4]; nm[4] = m[5]; nm[5] = m[6];
    nm[6] = m[8]; nm[7] = m[9]; nm[8] = m[10];
    sh.setMat3('u_NormalMatrix', nm);

    // Eye centre = node translation (from world matrix column 3)
    sh.setVec3('u_EyeCenter', m[12], m[13], m[14]);
    // Gaze axis = model +Z transformed to world space (3rd column of model rotation)
    sh.setVec3('u_GazeAxis', m[8], m[9], m[10]);

    gl.bindVertexArray(this._vao);
    gl.drawElements(gl.TRIANGLES, this._indexCount, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
  }
}
