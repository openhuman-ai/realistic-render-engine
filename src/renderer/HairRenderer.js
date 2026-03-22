/**
 * HairRenderer — Kajiya-Kay anisotropic hair shader.
 *
 * Implements a card-based hair renderer where each hair lock is represented
 * as a pair of triangles (a quad) with the hair flowing direction encoded as
 * the vertex tangent.  The fragment shader uses the Kajiya-Kay reflectance
 * model with two anisotropic lobes:
 *
 *   Primary lobe   — specular highlight at the expected hair angle
 *   Secondary lobe — shifted specular highlight (simulates cuticle layer)
 *
 * The card edges are faded with an alpha mask derived from the V texture
 * coordinate, so cards appear as loose strands rather than flat ribbons.
 * Fragments below the alpha threshold are discarded.
 *
 * Controls
 * ────────
 *   hairColor       — base hair pigment [R, G, B]
 *   specColor1      — primary  specular tint (usually light/pale)
 *   specColor2      — secondary specular tint (warm highlight)
 *   specPower1      — primary  lobe exponent  (default: 80)
 *   specPower2      — secondary lobe exponent (default: 32)
 *   specShift       — secondary lobe tangent shift (default: 0.1)
 *   alphaThreshold  — card edge discard threshold (default: 0.1)
 *
 * Geometry helper
 * ───────────────
 * HairRenderer.buildHairCards(gl, opts) returns a ready-to-render mesh object
 * compatible with the Node.mesh convention used elsewhere in the engine.
 */
import { Shader } from '../core/Shader.js';
import { Mat4 }   from '../math/Mat4.js';
import { Mat3 }   from '../math/Mat3.js';

// ─────────────────────────────────────────────────────────────────────────────
// Hair vertex shader
// ─────────────────────────────────────────────────────────────────────────────
const HAIR_VERT = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;
in vec3 a_Normal;
in vec3 a_Tangent;   // along-hair direction (world-aligned at rest)
in vec2 a_TexCoord;  // u = along-hair (0 root → 1 tip), v = across card (0/1 edges)

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;
uniform mat3 u_NormalMatrix;

out vec3 v_WorldPos;
out vec3 v_WorldTangent;
out vec3 v_Normal;
out vec2 v_TexCoord;

void main() {
  vec4 wPos       = u_ModelMatrix * vec4(a_Position, 1.0);
  v_WorldPos      = wPos.xyz;
  v_WorldTangent  = normalize(u_NormalMatrix * a_Tangent);
  v_Normal        = normalize(u_NormalMatrix * a_Normal);
  v_TexCoord      = a_TexCoord;
  gl_Position     = u_ProjectionMatrix * u_ViewMatrix * wPos;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Hair fragment shader — Kajiya-Kay + two-lobe anisotropic specular
// ─────────────────────────────────────────────────────────────────────────────
const HAIR_FRAG = /* glsl */`#version 300 es
precision highp float;

in vec3 v_WorldPos;
in vec3 v_WorldTangent;
in vec3 v_Normal;
in vec2 v_TexCoord;

// ── Hair material
uniform vec3  u_HairColor;
uniform vec3  u_SpecColor1;     // primary specular
uniform vec3  u_SpecColor2;     // secondary specular
uniform float u_SpecPower1;
uniform float u_SpecPower2;
uniform float u_SpecShift;      // secondary lobe tangent shift
uniform float u_AlphaThreshold;

// ── Lighting
uniform vec3 u_LightDir;        // normalised toward-light
uniform vec3 u_LightColor;
uniform vec3 u_CameraPos;
uniform vec3 u_AmbientColor;

out vec4 fragColor;

// Kajiya-Kay diffuse term: sqrt(1 - (T·L)²)
float kkDiffuse(vec3 T, vec3 L) {
  float TdotL = dot(T, L);
  return sqrt(max(0.0, 1.0 - TdotL * TdotL));
}

// Kajiya-Kay specular term: -(T·L)*(T·V) + sqrt(...) approach
// Using the sin-based form: spec = (T·H is "tangent space highlight")
float kkSpecular(vec3 T, vec3 L, vec3 V, float exponent) {
  vec3  H    = normalize(L + V);
  float TdotH = dot(T, H);
  float sinTH = sqrt(max(0.0, 1.0 - TdotH * TdotH));
  return pow(sinTH, exponent);
}

// Shift tangent along normal by a scalar (secondary lobe)
vec3 shiftTangent(vec3 T, vec3 N, float shift) {
  return normalize(T + shift * N);
}

void main() {
  // ── Alpha mask — soft fade at card edges (v = 0 or v = 1)
  float edgeMask = smoothstep(0.0, 0.12, v_TexCoord.y) *
                   smoothstep(0.0, 0.12, 1.0 - v_TexCoord.y);
  if (edgeMask < u_AlphaThreshold) discard;

  // Root-to-tip transparency: slightly more transparent at tip
  float tipFade = mix(1.0, 0.6, v_TexCoord.x);
  if (edgeMask * tipFade < u_AlphaThreshold * 0.5) discard;

  vec3 T = normalize(v_WorldTangent);
  vec3 N = normalize(v_Normal);
  vec3 L = normalize(u_LightDir);
  vec3 V = normalize(u_CameraPos - v_WorldPos);

  // ── Kajiya-Kay diffuse
  float diff = kkDiffuse(T, L);
  vec3  diffuse = u_HairColor * diff * u_LightColor;

  // ── Primary specular lobe
  float spec1 = kkSpecular(T, L, V, u_SpecPower1);
  vec3  primary = u_SpecColor1 * spec1 * u_LightColor;

  // ── Secondary specular lobe (shifted tangent)
  vec3  T2   = shiftTangent(T, N, u_SpecShift);
  float spec2 = kkSpecular(T2, L, V, u_SpecPower2);
  // Tint second lobe by hair color (subsurface-ish behaviour)
  vec3  secondary = u_SpecColor2 * u_HairColor * spec2 * u_LightColor;

  // ── Ambient
  vec3 ambient = u_HairColor * u_AmbientColor;

  // ── Combine
  vec3 color = ambient + diffuse + primary + secondary;
  // Apply edge+tip alpha for soft card appearance
  float alpha = edgeMask * tipFade;

  fragColor = vec4(color, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Procedural hair card geometry builder
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a set of hair cards arranged in a crown/cap configuration.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {{
 *   cardCount?:  number,   // number of hair cards  (default: 32)
 *   cardLength?: number,   // card length            (default: 0.9)
 *   cardWidth?:  number,   // card width             (default: 0.06)
 *   radius?:     number,   // scalp dome radius      (default: 0.52)
 *   droop?:      number,   // downward droop factor  (default: 0.55)
 * }} [opts]
 * @returns {{ vao, indexBuffer, indexCount, indexType, material, skinned: false }}
 */
export function buildHairCards(gl, opts = {}) {
  const COUNT  = opts.cardCount  ?? 32;
  const LEN    = opts.cardLength ?? 0.9;
  const WIDTH  = opts.cardWidth  ?? 0.06;
  const RADIUS = opts.radius     ?? 0.52;
  const DROOP  = opts.droop      ?? 0.55;

  const positions = [];
  const normals   = [];
  const tangents  = [];
  const uvs       = [];
  const indices   = [];

  for (let c = 0; c < COUNT; c++) {
    const angle    = (c / COUNT) * 2 * Math.PI;
    const cosA     = Math.cos(angle);
    const sinA     = Math.sin(angle);

    // Root position on the scalp dome (slightly above a sphere at y=0 centre)
    const rx  = cosA * RADIUS * 0.92;
    const ry  = RADIUS * 0.6;
    const rz  = sinA * RADIUS * 0.92;

    // Hair growth direction: outward + downward (gravity droop)
    const gx  = cosA * Math.cos(Math.PI * 0.22);
    const gy  = -DROOP;
    const gz  = sinA * Math.cos(Math.PI * 0.22);
    const glen = Math.sqrt(gx * gx + gy * gy + gz * gz);
    const tx  = gx / glen, ty = gy / glen, tz = gz / glen;

    // Perpendicular to tangent for card width (cross(tangent, up))
    const wx  = ty * 0 - tz * 1;   // cross(T, Y) ≈ width direction
    const wy  = tz * 0 - tx * 0;
    const wz  = tx * 1 - ty * 0;
    const wlen = Math.sqrt(wx * wx + wy * wy + wz * wz) || 1;
    const bx  = wx / wlen * WIDTH * 0.5;
    const by  = wy / wlen * WIDTH * 0.5;
    const bz  = wz / wlen * WIDTH * 0.5;

    // Normal: cross(width, tangent) → outward from scalp
    const nx  = (by * tz - bz * ty);
    const ny  = (bz * tx - bx * tz);
    const nz  = (bx * ty - by * tx);
    const nlen = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    const nnx = nx / nlen, nny = ny / nlen, nnz = nz / nlen;

    // Slight random variation per card
    const wobble = (Math.sin(c * 7.3) * 0.04);

    // 4 verts per card: root-left, root-right, tip-left, tip-right
    const vBase = positions.length / 3;

    // Root left
    positions.push(rx - bx + nnx * wobble, ry - by, rz - bz + nnz * wobble);
    normals.push(nnx, nny, nnz);
    tangents.push(tx, ty, tz);
    uvs.push(0, 0);

    // Root right
    positions.push(rx + bx + nnx * wobble, ry + by, rz + bz + nnz * wobble);
    normals.push(nnx, nny, nnz);
    tangents.push(tx, ty, tz);
    uvs.push(0, 1);

    // Tip left
    const tipWobble = Math.sin(c * 3.7) * 0.06;
    positions.push(
      rx - bx * 0.6 + tx * LEN + tipWobble,
      ry - by * 0.6 + ty * LEN,
      rz - bz * 0.6 + tz * LEN + tipWobble
    );
    normals.push(nnx, nny, nnz);
    tangents.push(tx, ty, tz);
    uvs.push(1, 0);

    // Tip right
    positions.push(
      rx + bx * 0.6 + tx * LEN + tipWobble,
      ry + by * 0.6 + ty * LEN,
      rz + bz * 0.6 + tz * LEN + tipWobble
    );
    normals.push(nnx, nny, nnz);
    tangents.push(tx, ty, tz);
    uvs.push(1, 1);

    // Two triangles: 0,1,2 and 1,3,2
    indices.push(vBase, vBase + 1, vBase + 2);
    indices.push(vBase + 1, vBase + 3, vBase + 2);
  }

  const posArr = new Float32Array(positions);
  const nrmArr = new Float32Array(normals);
  const tanArr = new Float32Array(tangents);
  const uvArr  = new Float32Array(uvs);
  const idxArr = new Uint16Array(indices);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const upload = (data, loc, size) => {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc);
    return buf;
  };

  const posBuf = upload(posArr, 0, 3);
  const nrmBuf = upload(nrmArr, 1, 3);
  const tanBuf = upload(tanArr, 2, 3);
  const uvBuf  = upload(uvArr,  3, 2);

  const idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idxArr, gl.STATIC_DRAW);

  gl.bindVertexArray(null);

  return {
    vao,
    _bufs:      [posBuf, nrmBuf, tanBuf, uvBuf, idxBuf],
    indexBuffer: { _buf: idxBuf },
    indexCount:  idxArr.length,
    indexType:   gl.UNSIGNED_SHORT,
    skinned:     false,
    isHair:      true,
    material: {
      baseColorFactor: [0.08, 0.05, 0.02, 1.0],   // dark brown
      roughnessFactor:  0.6,
      metallicFactor:   0.0,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
export class HairRenderer {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   */
  constructor(glContext, stateCache) {
    this.ctx   = glContext;
    this.gl    = glContext.gl;
    this.cache = stateCache;

    this._shader     = new Shader(this.gl, HAIR_VERT, HAIR_FRAG);
    this._normalMat  = new Mat3();

    // Hair material defaults
    this.hairColor      = [0.08, 0.05, 0.02];
    this.specColor1     = [0.9,  0.85, 0.75];
    this.specColor2     = [0.6,  0.45, 0.25];
    this.specPower1     = 80.0;
    this.specPower2     = 32.0;
    this.specShift      = 0.1;
    this.alphaThreshold = 0.05;
  }

  // ─────────────────────────────────────────── public API

  /** @param {number[]} rgb  base hair pigment */
  setHairColor(r, g, b)    { this.hairColor  = [r, g, b]; }
  /** @param {number[]} rgb  primary specular tint */
  setSpecColor1(r, g, b)   { this.specColor1 = [r, g, b]; }
  /** @param {number[]} rgb  secondary specular tint */
  setSpecColor2(r, g, b)   { this.specColor2 = [r, g, b]; }
  /** @param {number} v  primary lobe exponent */
  setSpecPower1(v)         { this.specPower1 = v; }
  /** @param {number} v  secondary lobe exponent */
  setSpecPower2(v)         { this.specPower2 = v; }
  /** @param {number} v  secondary tangent shift */
  setSpecShift(v)          { this.specShift  = v; }

  /**
   * Render all hair nodes to the current framebuffer.
   *
   * @param {Array} nodes
   * @param {import('../scene/Camera.js').Camera} camera
   * @param {import('../scene/Light.js').Light[]} lights
   */
  render(nodes, camera, lights) {
    if (!nodes || nodes.length === 0) return;

    const gl    = this.gl;
    const cache = this.cache;

    camera.updateProjection();
    camera.updateView();

    // Two-sided rendering + alpha blend for hair transparency
    cache.setDepthTest(true);
    cache.setDepthWrite(false);         // hair cards are semi-transparent
    cache.setCullFace(false, gl.BACK);  // render both sides
    cache.setBlend(true);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    cache.useProgram(this._shader.program);

    const dirLight = lights?.find(l => l.type === 'directional') ?? null;
    const lx = dirLight ? -dirLight.direction.x : -0.5;
    const ly = dirLight ? -dirLight.direction.y :  1.0;
    const lz = dirLight ? -dirLight.direction.z : -0.3;
    const li = dirLight ? dirLight.intensity : 1.0;
    const lcx = dirLight ? dirLight.color.x * li : 1.0;
    const lcy = dirLight ? dirLight.color.y * li : 1.0;
    const lcz = dirLight ? dirLight.color.z * li : 1.0;

    this._shader.setVec3('u_LightDir',   lx, ly, lz);
    this._shader.setVec3('u_LightColor', lcx, lcy, lcz);
    this._shader.setVec3('u_AmbientColor', 0.12, 0.10, 0.09);
    const cp = camera.position.e;
    this._shader.setVec3('u_CameraPos', cp[0], cp[1], cp[2]);

    this._shader.setMat4('u_ViewMatrix',       camera.viewMatrix.e);
    this._shader.setMat4('u_ProjectionMatrix', camera.projectionMatrix.e);

    this._shader.setVec3('u_HairColor',  this.hairColor[0],  this.hairColor[1],  this.hairColor[2]);
    this._shader.setVec3('u_SpecColor1', this.specColor1[0], this.specColor1[1], this.specColor1[2]);
    this._shader.setVec3('u_SpecColor2', this.specColor2[0], this.specColor2[1], this.specColor2[2]);
    this._shader.setFloat('u_SpecPower1',     this.specPower1);
    this._shader.setFloat('u_SpecPower2',     this.specPower2);
    this._shader.setFloat('u_SpecShift',      this.specShift);
    this._shader.setFloat('u_AlphaThreshold', this.alphaThreshold);

    for (const node of nodes) {
      if (!node.mesh) continue;
      this._renderNode(node);
    }

    // Restore depth write
    cache.setDepthWrite(true);
    cache.setBlend(false);
    cache.setCullFace(true, gl.BACK);
  }

  resize(_w, _h) { /* no internal render targets */ }

  destroy() {
    this._shader?.destroy();
    this._shader = null;
  }

  // ─────────────────────────────────────────── private

  _renderNode(node) {
    const gl = this.gl;
    const sh = this._shader;

    const mxArr = node.worldMatrix?.e ?? node.worldMatrix;
    sh.setMat4('u_ModelMatrix', mxArr);

    // Normal matrix (upper-left 3×3 of model matrix, non-uniform scale not handled for simplicity)
    const m  = mxArr;
    const nm = this._normalMat.e;
    nm[0] = m[0]; nm[1] = m[1]; nm[2] = m[2];
    nm[3] = m[4]; nm[4] = m[5]; nm[5] = m[6];
    nm[6] = m[8]; nm[7] = m[9]; nm[8] = m[10];
    sh.setMat3('u_NormalMatrix', nm);

    const mesh = node.mesh;
    gl.bindVertexArray(mesh.vao);
    gl.drawElements(gl.TRIANGLES, mesh.indexCount, mesh.indexType ?? gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
  }
}
