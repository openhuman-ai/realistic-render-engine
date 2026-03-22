/**
 * DeferredRenderer — G-buffer based deferred shading pipeline.
 * TODO: implement G-buffer pass, lighting pass, and composition.
 */
export class DeferredRenderer {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   */
  constructor(glContext, stateCache) {
    this.ctx   = glContext;
    this.gl    = glContext.gl;
    this.cache = stateCache;
    // TODO: create G-buffer (position, normal, albedo/specular) render targets
    // TODO: create lighting accumulation render target
    // TODO: compile geometry pass and lighting pass shaders
    console.warn('[DeferredRenderer] Not yet implemented — use ForwardRenderer.');
  }

  /**
   * Render scene using deferred shading.
   * @param {*} scene
   * @param {*} camera
   * @param {*} lights
   */
  render(scene, camera, lights = []) {
    // TODO: geometry pass — fill G-buffer
    // TODO: lighting pass — accumulate contributions per light
    // TODO: composition / tone-mapping pass
  }

  destroy() {
    // TODO: destroy G-buffer textures, shaders, render targets
  }
}
