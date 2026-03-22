/**
 * GLContext — WebGL 2.0 context wrapper with extension discovery.
 * Falls back to WebGL1 with a console warning if WebGL2 is unavailable.
 */
export class GLContext {
  /** @type {WebGL2RenderingContext|WebGLRenderingContext} */
  gl = null;
  /** @type {boolean} */
  isWebGL2 = false;
  /** @type {Record<string, *>} */
  ext = {};

  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas = canvas;
    this._init();
    this._queryExtensions();
    this._attachContextHandlers();
  }

  _init() {
    const attrs = {
      alpha: true,
      antialias: true,
      depth: true,
      stencil: true,
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: false,
    };

    this.gl = this.canvas.getContext('webgl2', attrs);
    if (this.gl) {
      this.isWebGL2 = true;
      return;
    }

    console.warn('[GLContext] WebGL2 not available, falling back to WebGL1.');
    this.gl = this.canvas.getContext('webgl', attrs)
           || this.canvas.getContext('experimental-webgl', attrs);

    if (!this.gl) {
      throw new Error('[GLContext] WebGL is not supported in this browser.');
    }
  }

  _queryExtensions() {
    const gl = this.gl;
    const names = [
      'EXT_color_buffer_float',
      'OES_texture_float_linear',
      'EXT_texture_filter_anisotropic',
      'WEBGL_compressed_texture_s3tc',
      'WEBGL_compressed_texture_etc',
      'WEBGL_compressed_texture_bptc',
    ];
    for (const name of names) {
      const e = gl.getExtension(name);
      this.ext[name] = e || null;
      if (!e) console.info(`[GLContext] Extension not available: ${name}`);
    }
  }

  _attachContextHandlers() {
    this.canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      console.warn('[GLContext] Context lost.');
    });
    this.canvas.addEventListener('webglcontextrestored', () => {
      console.info('[GLContext] Context restored.');
      this._queryExtensions();
    });
  }

  /**
   * Resize the canvas and update the GL viewport.
   * @param {number} w
   * @param {number} h
   */
  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.gl.viewport(0, 0, w, h);
  }

  destroy() {
    const ext = this.gl.getExtension('WEBGL_lose_context');
    if (ext) ext.loseContext();
    this.gl = null;
  }
}
