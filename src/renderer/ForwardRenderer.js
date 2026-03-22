/**
 * ForwardRenderer — single-pass Phong/PBR-lite forward renderer.
 *
 * Renders a scene of Node → mesh objects using a built-in GLSL 300 es shader
 * with directional lighting, diffuse + specular Phong shading.
 *
 * Skinned meshes (mesh.skinned === true) are rendered with a second shader
 * variant that samples dual-quaternion joint data from an RGBA32F texture.
 */
import { Shader }     from '../core/Shader.js';
import { Mat4 }       from '../math/Mat4.js';
import { Mat3 }       from '../math/Mat3.js';

// ─────────────────────────────────────────────────────────────────────────────
// Inline GLSL — static (non-skinned) path
// ─────────────────────────────────────────────────────────────────────────────
const VERT_SRC = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;
uniform mat3 u_NormalMatrix;

out vec3 v_Normal;
out vec3 v_WorldPos;
out vec2 v_TexCoord;

void main() {
  vec4 worldPos = u_ModelMatrix * vec4(a_Position, 1.0);
  v_WorldPos = worldPos.xyz;
  v_Normal = normalize(u_NormalMatrix * a_Normal);
  v_TexCoord = a_TexCoord;
  gl_Position = u_ProjectionMatrix * u_ViewMatrix * worldPos;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Inline GLSL — skinned (dual-quaternion) vertex shader
// ─────────────────────────────────────────────────────────────────────────────
const VERT_SKINNED_SRC = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;
in vec4 a_Joints;   // joint indices as float (0–255), cast to int in shader
in vec4 a_Weights;

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;
// Joint texture: RGBA32F, width=2, height=jointCount
// texel(0,j) = real quaternion,  texel(1,j) = dual quaternion
uniform highp sampler2D u_JointTexture;

out vec3 v_Normal;
out vec3 v_WorldPos;
out vec2 v_TexCoord;

// Rotate vector v by unit quaternion q (q.xyz = imag, q.w = real).
vec3 quatRotate(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void applySkinning(inout vec3 pos, inout vec3 nrm) {
  vec4 blendReal = vec4(0.0);
  vec4 blendDual = vec4(0.0);

  // Sample first influence for antipodality reference
  vec4 ref = texelFetch(u_JointTexture, ivec2(0, int(a_Joints.x)), 0);

  for (int i = 0; i < 4; i++) {
    float w = a_Weights[i];
    if (w < 0.0001) continue;
    int ji = int(a_Joints[i]);
    vec4 re = texelFetch(u_JointTexture, ivec2(0, ji), 0);
    vec4 de = texelFetch(u_JointTexture, ivec2(1, ji), 0);
    float sgn = (dot(re, ref) < 0.0) ? -1.0 : 1.0;
    blendReal += sgn * w * re;
    blendDual += sgn * w * de;
  }

  // Normalise blended dual quaternion
  float len = length(blendReal);
  if (len > 0.0001) { blendReal /= len; blendDual /= len; }

  // Extract translation from dual part:
  //   t = 2 * (rw*d.xyz - dw*r.xyz + cross(r.xyz, d.xyz))
  vec3 t = 2.0 * (blendReal.w * blendDual.xyz
                - blendDual.w * blendReal.xyz
                + cross(blendReal.xyz, blendDual.xyz));

  // Apply rotation then translation
  pos = quatRotate(blendReal, pos) + t;
  nrm = quatRotate(blendReal, nrm);
}

void main() {
  vec3 pos = a_Position;
  vec3 nrm = a_Normal;
  applySkinning(pos, nrm);

  vec4 worldPos = u_ModelMatrix * vec4(pos, 1.0);
  v_WorldPos = worldPos.xyz;
  // Normal: use upper-left 3×3 of model matrix (valid for uniform/no scale)
  v_Normal   = normalize(mat3(u_ModelMatrix) * nrm);
  v_TexCoord = a_TexCoord;
  gl_Position = u_ProjectionMatrix * u_ViewMatrix * worldPos;
}
`;

const FRAG_SRC = /* glsl */`#version 300 es
precision mediump float;

in vec3 v_Normal;
in vec3 v_WorldPos;
in vec2 v_TexCoord;

uniform vec3  u_LightDir;
uniform vec3  u_LightColor;
uniform vec3  u_BaseColor;
uniform float u_Roughness;
uniform float u_Metallic;
uniform vec3  u_CameraPos;

out vec4 fragColor;

void main() {
  vec3 N = normalize(v_Normal);
  vec3 L = normalize(-u_LightDir);
  vec3 V = normalize(u_CameraPos - v_WorldPos);
  vec3 H = normalize(L + V);

  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(N, H), 0.0), 32.0 * (1.0 - u_Roughness) + 1.0);

  vec3 ambient  = u_BaseColor * 0.15;
  vec3 diffuse  = u_BaseColor * u_LightColor * diff;
  vec3 specular = mix(vec3(0.04), u_BaseColor, u_Metallic) * u_LightColor * spec * (1.0 - u_Roughness);

  fragColor = vec4(ambient + diffuse + specular, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
export class ForwardRenderer {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   */
  constructor(glContext, stateCache) {
    this.ctx   = glContext;
    this.gl    = glContext.gl;
    this.cache = stateCache;

    this.shader        = new Shader(this.gl, VERT_SRC, FRAG_SRC);
    this.skinnedShader = new Shader(this.gl, VERT_SKINNED_SRC, FRAG_SRC);

    // Pre-allocated matrices to avoid per-frame heap allocation
    this._mvp       = new Mat4();
    this._normalMat = new Mat3();
  }

  /**
   * Render a scene from a camera's perspective.
   * @param {import('../scene/Node.js').Node} scene   — root node (or array of nodes)
   * @param {import('../scene/Camera.js').Camera} camera
   * @param {import('../scene/Light.js').Light[]} [lights]
   * @param {import('../animation/GPUSkinning.js').GPUSkinning} [gpuSkinning] — optional skinning state
   */
  render(scene, camera, lights = [], gpuSkinning = null) {
    const gl    = this.gl;
    const cache = this.cache;

    // Update matrices
    camera.updateProjection();
    camera.updateView();

    gl.clearColor(0.1, 0.1, 0.12, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    cache.setDepthTest(true);
    cache.setDepthWrite(true);
    cache.setCullFace(true, gl.BACK);
    cache.setBlend(false);

    // Light uniforms (shared by both shader variants)
    const dirLight = lights.find(l => l.type === 'directional');
    const lightDir   = dirLight ? dirLight.direction.e : [0.5, -1.0, 0.5];
    const lightColor = dirLight
      ? [dirLight.color.e[0] * dirLight.intensity,
         dirLight.color.e[1] * dirLight.intensity,
         dirLight.color.e[2] * dirLight.intensity]
      : [1, 1, 1];

    // Collect renderables
    const nodes = Array.isArray(scene) ? scene : this._collectNodes(scene);

    // Separate into skinned and static draw lists
    const staticNodes  = [];
    const skinnedNodes = [];
    for (const node of nodes) {
      if (!node.mesh) continue;
      if (node.mesh.skinned && gpuSkinning) skinnedNodes.push(node);
      else staticNodes.push(node);
    }

    // ─── Static pass
    if (staticNodes.length) {
      cache.useProgram(this.shader.program);
      this._setSharedUniforms(this.shader, camera, lightDir, lightColor);
      for (const node of staticNodes) this._drawNode(node, this.shader, false);
    }

    // ─── Skinned pass
    if (skinnedNodes.length && gpuSkinning) {
      cache.useProgram(this.skinnedShader.program);
      this._setSharedUniforms(this.skinnedShader, camera, lightDir, lightColor);
      gpuSkinning.bind(this.skinnedShader, 0);
      for (const node of skinnedNodes) this._drawNode(node, this.skinnedShader, true);
    }
  }

  /** Upload camera + light uniforms to a shader. @private */
  _setSharedUniforms(shader, cam, lightDir, lightColor) {
    shader.setMat4('u_ViewMatrix',       cam.viewMatrix.e);
    shader.setMat4('u_ProjectionMatrix', cam.projectionMatrix.e);
    shader.setVec3('u_CameraPos',
      cam.position.e[0], cam.position.e[1], cam.position.e[2]);
    shader.setVec3('u_LightDir',    lightDir[0],   lightDir[1],   lightDir[2]);
    shader.setVec3('u_LightColor',  lightColor[0], lightColor[1], lightColor[2]);
  }

  /** @private */
  _drawNode(node, shader, skinned = false) {
    const mesh  = node.mesh;
    const gl    = this.gl;
    const cache = this.cache;

    // Model matrix
    node.updateWorldMatrix(node.parent ? node.parent.worldMatrix : null);
    shader.setMat4('u_ModelMatrix', node.worldMatrix.e);

    if (!skinned) {
      // Normal matrix: inverse-transpose of upper-left 3×3 of model matrix
      Mat3.normalMatrix(node.worldMatrix, this._normalMat);
      shader.setMat3('u_NormalMatrix', this._normalMat.e);
    }

    // Material uniforms
    const mat = mesh.material || {};
    const bc  = mat.baseColorFactor ?? [0.8, 0.6, 0.4, 1.0];
    shader.setVec3('u_BaseColor',  bc[0], bc[1], bc[2]);
    shader.setFloat('u_Roughness', mat.roughnessFactor ?? 0.5);
    shader.setFloat('u_Metallic',  mat.metallicFactor  ?? 0.0);

    cache.bindVAO(mesh.vao);

    if (mesh.indexBuffer) {
      gl.drawElements(gl.TRIANGLES, mesh.indexCount, mesh.indexType, 0);
    } else {
      gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
    }

    cache.bindVAO(null);
  }

  /** Depth-first collect all scene nodes into a flat list. @private */
  _collectNodes(root) {
    const result = [];
    const stack  = [root];
    while (stack.length) {
      const node = stack.pop();
      result.push(node);
      if (node.children) {
        for (let i = node.children.length - 1; i >= 0; i--) {
          stack.push(node.children[i]);
        }
      }
    }
    return result;
  }

  resize(w, h) {
    this.cache.setViewport(0, 0, w, h);
  }

  destroy() {
    this.shader.destroy();
    this.skinnedShader.destroy();
    this.shader        = null;
    this.skinnedShader = null;
  }
}
