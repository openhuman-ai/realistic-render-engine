/**
 * ShadowMap — depth-map based shadow rendering.
 * TODO: implement shadow-map pass and PCF filtering.
 */
export class ShadowMap {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   * @param {{ width?: number, height?: number }} [opts]
   */
  constructor(glContext, stateCache, opts = {}) {
    this.ctx    = glContext;
    this.gl     = glContext.gl;
    this.cache  = stateCache;
    this.width  = opts.width  ?? 1024;
    this.height = opts.height ?? 1024;
    // TODO: create depth-only framebuffer (RenderTarget with depthAttachment only)
    // TODO: compile depth-pass vertex/fragment shaders
    // TODO: store light-space matrix
    console.warn('[ShadowMap] Not yet implemented.');
  }

  /**
   * Render scene from light's perspective into the shadow map.
   * @param {*} scene
   * @param {import('../scene/Light.js').Light} light
   */
  render(scene, light) {
    // TODO: bind shadow FBO
    // TODO: render scene with depth shader from light's viewProjection
    // TODO: unbind shadow FBO
  }

  /**
   * @returns {WebGLTexture|null} depth texture
   */
  getTexture() {
    // TODO: return depth texture from shadow render target
    return null;
  }

  destroy() {
    // TODO: destroy depth FBO, shaders
  }
}
