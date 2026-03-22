/**
 * PostProcessStack — chain of full-screen post-processing passes.
 * TODO: implement ping-pong render targets and pass execution.
 */
export class PostProcessStack {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   */
  constructor(glContext, stateCache) {
    this.ctx    = glContext;
    this.gl     = glContext.gl;
    this.cache  = stateCache;
    /** @type {Array<{name: string, shader: *, enabled: boolean}>} */
    this.passes = [];
    // TODO: create ping-pong render targets for pass chaining
    // TODO: create full-screen quad VAO
  }

  /**
   * Add a post-process pass.
   * @param {{ name: string, shader: *, uniforms?: Record<string,*>, enabled?: boolean }} pass
   */
  addPass(pass) {
    this.passes.push({ enabled: true, ...pass });
    // TODO: initialise pass resources
  }

  /**
   * Execute all enabled passes on the provided input texture.
   * @param {WebGLTexture} inputTexture
   */
  render(inputTexture) {
    // TODO: iterate passes, ping-pong between render targets
    // TODO: blit final result to default framebuffer
  }

  destroy() {
    // TODO: destroy ping-pong render targets, quad VAO
    this.passes = [];
  }
}
