/**
 * ShadowMap — single directional-light depth-map shadow renderer.
 *
 * Creates a dedicated depth-texture FBO, renders the scene geometry from the
 * light's orthographic perspective, and exposes the resulting texture + light-
 * space matrix for use in the deferred lighting pass.
 *
 * PCF filtering is done in the consumer shader via manual 3×3 kernel samples.
 */
import { Shader } from '../core/Shader.js';
import { Mat4 }   from '../math/Mat4.js';
import { Vec3 }   from '../math/Vec3.js';
import { Mat3 }   from '../math/Mat3.js';

// ─────────────────────────────────────────────────────────────────────────────
// Depth-only vertex shader — static geometry
// ─────────────────────────────────────────────────────────────────────────────
const SHADOW_VERT = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;

uniform mat4 u_ModelMatrix;
uniform mat4 u_LightSpaceMatrix;

void main() {
  gl_Position = u_LightSpaceMatrix * u_ModelMatrix * vec4(a_Position, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Depth-only vertex shader — skinned (GPU DQ) geometry
// ─────────────────────────────────────────────────────────────────────────────
const SHADOW_VERT_SKINNED = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;
in vec4 a_Joints;
in vec4 a_Weights;

uniform mat4 u_ModelMatrix;
uniform mat4 u_LightSpaceMatrix;
uniform highp sampler2D u_JointTexture;

vec3 quatRotate(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void applySkinning(inout vec3 pos) {
  vec4 blendReal = vec4(0.0);
  vec4 blendDual = vec4(0.0);
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
  float len = length(blendReal);
  if (len > 0.0001) { blendReal /= len; blendDual /= len; }
  vec3 t = 2.0 * (blendReal.w * blendDual.xyz
                - blendDual.w * blendReal.xyz
                + cross(blendReal.xyz, blendDual.xyz));
  pos = quatRotate(blendReal, pos) + t;
}

void main() {
  vec3 pos = a_Position;
  applySkinning(pos);
  gl_Position = u_LightSpaceMatrix * u_ModelMatrix * vec4(pos, 1.0);
}
`;

// Depth-only fragment shader (WebGL2 writes depth automatically)
const SHADOW_FRAG = /* glsl */`#version 300 es
precision highp float;
void main() {}
`;

// ─────────────────────────────────────────────────────────────────────────────
export class ShadowMap {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   * @param {{ width?: number, height?: number, near?: number, far?: number,
   *           orthoSize?: number, bias?: number }} [opts]
   */
  constructor(glContext, stateCache, opts = {}) {
    this.ctx    = glContext;
    this.gl     = glContext.gl;
    this.cache  = stateCache;
    this.width  = opts.width     ?? 2048;
    this.height = opts.height    ?? 2048;
    this.near   = opts.near      ?? 0.5;
    this.far    = opts.far       ?? 100;
    this.orthoSize = opts.orthoSize ?? 20;
    /** Depth bias applied in the lighting pass to avoid self-shadowing acne. */
    this.bias   = opts.bias      ?? 0.002;

    this._lightSpaceMatrix = new Mat4();
    this._lightView        = new Mat4();
    this._lightProj        = new Mat4();

    // Pre-alloc scratch vecs for light-matrix computation
    this._lightPosVec  = new Vec3();
    this._lightDirVec  = new Vec3();

    this._depthTex = null;
    this._fbo      = null;
    this._shader         = new Shader(this.gl, SHADOW_VERT, SHADOW_FRAG);
    this._skinnedShader  = new Shader(this.gl, SHADOW_VERT_SKINNED, SHADOW_FRAG);

    this._normalMat = new Mat3(); // unused in shadow pass but kept for symmetry

    this._buildDepthFBO();
  }

  // ─────────────────────────────────────── internal FBO with depth texture
  _buildDepthFBO() {
    const gl = this.gl;

    this._depthTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._depthTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT32F, this.width, this.height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this._fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this._depthTex, 0
    );
    // No colour attachment — depth only
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('[ShadowMap] Depth FBO incomplete, status:', status);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─────────────────────────────────────── light-space matrix computation
  /**
   * Compute and cache the orthographic light-space view-projection matrix.
   * @param {import('../scene/Light.js').Light} light
   */
  _updateLightMatrix(light) {
    const d = light.direction.e;
    // Normalise direction
    const len = Math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) || 1;
    const dx = d[0]/len, dy = d[1]/len, dz = d[2]/len;

    // Light position: step back along the light direction from scene centre
    const dist = (this.far - this.near) * 0.5 + this.near;
    this._lightPosVec.set(-dx * dist, -dy * dist, -dz * dist);

    // Choose an up vector that isn't parallel to the light direction
    const up = (Math.abs(dy) < 0.99)
      ? new Vec3(0, 1, 0)
      : new Vec3(1, 0, 0);

    // View matrix: looking from light position toward origin
    Mat4.lookAt(this._lightPosVec, new Vec3(0, 0, 0), up, this._lightView);

    // Symmetric orthographic projection
    const s = this.orthoSize;
    Mat4.orthographic(-s, s, -s, s, this.near, this.far, this._lightProj);

    // Final light-space matrix
    Mat4.multiply(this._lightProj, this._lightView, this._lightSpaceMatrix);
  }

  // ─────────────────────────────────────── public API
  /**
   * Render the shadow depth map from the given light's perspective.
   * Call this before the main render pass each frame.
   *
   * @param {Array} nodes  — flat list of scene nodes with .mesh
   * @param {import('../scene/Light.js').Light} light
   * @param {import('../animation/GPUSkinning.js').GPUSkinning|null} [gpuSkinning]
   */
  render(nodes, light, gpuSkinning = null) {
    const gl    = this.gl;
    const cache = this.cache;

    this._updateLightMatrix(light);

    // ── Bind shadow FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
    gl.viewport(0, 0, this.width, this.height);
    gl.clear(gl.DEPTH_BUFFER_BIT);

    cache.setDepthTest(true);
    cache.setDepthWrite(true);
    // Front-face culling reduces peter-panning artefacts on back faces
    cache.setCullFace(true, gl.FRONT);
    cache.setBlend(false);

    const lsm = this._lightSpaceMatrix.e;

    // Separate static vs skinned
    const staticNodes  = [];
    const skinnedNodes = [];
    for (const node of nodes) {
      if (!node.mesh) continue;
      if (node.mesh.skinned && gpuSkinning) skinnedNodes.push(node);
      else staticNodes.push(node);
    }

    // ── Static shadow pass
    if (staticNodes.length) {
      cache.useProgram(this._shader.program);
      this._shader.setMat4('u_LightSpaceMatrix', lsm);
      for (const node of staticNodes) {
        node.updateWorldMatrix(node.parent ? node.parent.worldMatrix : null);
        this._shader.setMat4('u_ModelMatrix', node.worldMatrix.e);
        cache.bindVAO(node.mesh.vao);
        if (node.mesh.indexBuffer) {
          gl.drawElements(gl.TRIANGLES, node.mesh.indexCount, node.mesh.indexType, 0);
        } else {
          gl.drawArrays(gl.TRIANGLES, 0, node.mesh.vertexCount);
        }
      }
    }

    // ── Skinned shadow pass
    if (skinnedNodes.length && gpuSkinning) {
      cache.useProgram(this._skinnedShader.program);
      this._skinnedShader.setMat4('u_LightSpaceMatrix', lsm);
      gpuSkinning.bind(this._skinnedShader, 0);
      for (const node of skinnedNodes) {
        node.updateWorldMatrix(node.parent ? node.parent.worldMatrix : null);
        this._skinnedShader.setMat4('u_ModelMatrix', node.worldMatrix.e);
        cache.bindVAO(node.mesh.vao);
        if (node.mesh.indexBuffer) {
          gl.drawElements(gl.TRIANGLES, node.mesh.indexCount, node.mesh.indexType, 0);
        } else {
          gl.drawArrays(gl.TRIANGLES, 0, node.mesh.vertexCount);
        }
      }
    }

    cache.bindVAO(null);
    cache.setCullFace(true, gl.BACK);   // restore
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * The light-space view-projection matrix, updated each time render() is called.
   * Pass this to the deferred lighting shader as u_LightSpaceMatrix.
   * @returns {Mat4}
   */
  getLightSpaceMatrix() {
    return this._lightSpaceMatrix;
  }

  /**
   * The shadow depth texture (DEPTH_COMPONENT32F).
   * @returns {WebGLTexture|null}
   */
  getTexture() {
    return this._depthTex;
  }

  destroy() {
    const gl = this.gl;
    if (this._fbo)      { gl.deleteFramebuffer(this._fbo);   this._fbo = null; }
    if (this._depthTex) { gl.deleteTexture(this._depthTex);  this._depthTex = null; }
    this._shader?.destroy();
    this._skinnedShader?.destroy();
    this._shader = null;
    this._skinnedShader = null;
  }
}
