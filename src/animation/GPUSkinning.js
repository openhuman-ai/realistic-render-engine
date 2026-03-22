/**
 * GPUSkinning — uploads per-joint dual-quaternion data to a WebGL2 texture
 * for consumption by the skinned vertex shader.
 *
 * Texture layout:
 *   Format : RGBA32F (4 × float32 per texel)
 *   Width  : 2  (texel 0 = real part, texel 1 = dual part)
 *   Height : maxJoints
 *
 *   texelFetch(u_JointTexture, ivec2(0, jointIndex), 0) → real quaternion
 *   texelFetch(u_JointTexture, ivec2(1, jointIndex), 0) → dual quaternion
 *
 * CPU data array (Float32Array, 8 floats per row / joint):
 *   [re.x, re.y, re.z, re.w,  de.x, de.y, de.z, de.w, ...]
 */
import { DualQuat } from '../math/DualQuat.js';

export class GPUSkinning {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {number} [maxJoints=256]
   */
  constructor(gl, maxJoints = 256) {
    this.gl         = gl;
    this.maxJoints  = maxJoints;

    // Flat CPU buffer — 2 texels × 4 channels = 8 floats per joint
    this._data = new Float32Array(maxJoints * 8);

    // Pre-allocate DualQuat scratch array for skeleton evaluation
    this._dqScratch = [];
    for (let i = 0; i < maxJoints; i++) {
      this._dqScratch.push(new DualQuat());
    }

    // Create the RGBA32F texture
    this._tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 2, maxJoints);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // Upload identity DQs so the shader never reads uninitialised data
    this._uploadIdentity(maxJoints);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ───────────────────────────────────────────────────────────── update
  /**
   * Evaluate the skeleton's skinning dual-quaternions and stream them to GPU.
   * Call once per frame after the animation system has set the joint local poses.
   *
   * @param {import('../animation/Skeleton.js').Skeleton} skeleton
   */
  update(skeleton) {
    const joints = skeleton.joints;
    const n      = joints.length;
    if (n === 0) return;

    // Reuse the skeleton's own pre-allocated DQ scratch if it has it,
    // otherwise fall back to our own.
    const dqs = skeleton._skinDQs ?? this._dqScratch;

    skeleton.updateWorldMatrices();
    skeleton.computeSkinningDQs(dqs);

    const data = this._data;
    for (let i = 0; i < n; i++) {
      const dq   = dqs[i];
      const base = i * 8;
      data[base]     = dq.real.e[0];
      data[base + 1] = dq.real.e[1];
      data[base + 2] = dq.real.e[2];
      data[base + 3] = dq.real.e[3];
      data[base + 4] = dq.dual.e[0];
      data[base + 5] = dq.dual.e[1];
      data[base + 6] = dq.dual.e[2];
      data[base + 7] = dq.dual.e[3];
    }

    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this._tex);
    // Update only the rows that correspond to active joints
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 2, n, gl.RGBA, gl.FLOAT,
                     data.subarray(0, n * 8));
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ──────────────────────────────────────────────────────────────── bind
  /**
   * Bind the joint texture to a texture unit and set the sampler uniform.
   * Call before issuing the skinned draw call.
   *
   * @param {import('../core/Shader.js').Shader} shader
   * @param {number} textureUnit — GL texture unit index (e.g. 0)
   */
  bind(shader, textureUnit) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, this._tex);
    shader.setInt('u_JointTexture', textureUnit);
  }

  // ──────────────────────────────────────────────────────────── destroy
  destroy() {
    this.gl.deleteTexture(this._tex);
    this._tex  = null;
    this._data = null;
  }

  // ──────────────────────────────────────────────────── private helpers
  /** Upload identity dual quaternions for all joints. @private */
  _uploadIdentity(count) {
    const data = this._data;
    for (let i = 0; i < count; i++) {
      const base = i * 8;
      // Identity DQ: real = (0,0,0,1), dual = (0,0,0,0)
      data[base]     = 0; data[base + 1] = 0; data[base + 2] = 0; data[base + 3] = 1;
      data[base + 4] = 0; data[base + 5] = 0; data[base + 6] = 0; data[base + 7] = 0;
    }
    this.gl.texSubImage2D(
      this.gl.TEXTURE_2D, 0, 0, 0, 2, count,
      this.gl.RGBA, this.gl.FLOAT, data.subarray(0, count * 8));
  }
}
