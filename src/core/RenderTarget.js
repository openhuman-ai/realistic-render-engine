/**
 * RenderTarget — off-screen framebuffer with MRT colour attachments and an
 * optional depth/stencil renderbuffer or depth texture.  Supports MSAA via
 * the `samples` option.
 *
 * Colour format options: 'RGBA8' | 'RGBA16F' | 'RGBA32F' | 'R32F'
 *
 * Depth options:
 *   depthAttachment: true   → depth+stencil renderbuffer (default)
 *   depthTexture: true      → DEPTH_COMPONENT32F texture (sampleable; mutually
 *                              exclusive with depthAttachment renderbuffer path)
 */
export class RenderTarget {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {{
   *   width: number,
   *   height: number,
   *   colorAttachments?: Array<'RGBA8'|'RGBA16F'|'RGBA32F'|'R32F'>,
   *   depthAttachment?: boolean,
   *   depthTexture?: boolean,
   *   samples?: number
   * }} opts
   */
  constructor(gl, opts = {}) {
    this.gl = gl;
    this.width  = opts.width  ?? 1;
    this.height = opts.height ?? 1;
    this.samples = opts.samples ?? 0;
    this._colorFmts  = opts.colorAttachments ?? ['RGBA8'];
    this._wantDepth  = opts.depthAttachment  ?? true;
    this._wantDepthTex = opts.depthTexture   ?? false;
    // depthTexture takes priority over depthAttachment renderbuffer
    if (this._wantDepthTex) this._wantDepth = false;

    this._colorTextures = [];
    this._depthTexture  = null;
    this._depthRBO = null;
    this._fbo = null;

    this._build();
  }

  // ------------------------------------------------------------------- build
  _build() {
    const gl = this.gl;
    this._fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);

    const drawBuffers = [];

    for (let i = 0; i < Math.min(this._colorFmts.length, 4); i++) {
      const tex = this._createColorTexture(this._colorFmts[i]);
      this._colorTextures.push(tex);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0 + i,
        gl.TEXTURE_2D,
        tex,
        0
      );
      drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
    }

    if (drawBuffers.length > 1) {
      gl.drawBuffers(drawBuffers);
    }

    // ── Depth texture (sampleable, used by shadow maps and depth pre-pass)
    if (this._wantDepthTex) {
      this._depthTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this._depthTexture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT32F, this.width, this.height);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this._depthTexture, 0
      );
    }

    // ── Depth renderbuffer (non-sampleable, used for G-buffer and HDR targets)
    if (this._wantDepth) {
      this._depthRBO = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, this._depthRBO);
      if (this.samples > 1) {
        gl.renderbufferStorageMultisample(
          gl.RENDERBUFFER, this.samples, gl.DEPTH24_STENCIL8,
          this.width, this.height
        );
      } else {
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, this.width, this.height);
      }
      gl.framebufferRenderbuffer(
        gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, this._depthRBO
      );
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    }

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('[RenderTarget] Framebuffer incomplete, status:', status);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** @private */
  _internalFormat(fmt) {
    const gl = this.gl;
    switch (fmt) {
      case 'RGBA16F': return gl.RGBA16F;
      case 'RGBA32F': return gl.RGBA32F;
      case 'R32F':    return gl.R32F;
      default:        return gl.RGBA8;
    }
  }

  /** @private */
  _baseFormat(fmt) {
    const gl = this.gl;
    switch (fmt) {
      case 'R32F': return gl.RED;
      default:     return gl.RGBA;
    }
  }

  /** @private */
  _createColorTexture(fmt) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, this._internalFormat(fmt), this.width, this.height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  // ------------------------------------------------------------------ public
  bind() {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this._fbo);
    this.gl.viewport(0, 0, this.width, this.height);
  }

  unbind() {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  /**
   * @param {number} index — colour attachment index (0-based)
   * @returns {WebGLTexture}
   */
  getColorTexture(index = 0) {
    return this._colorTextures[index] ?? null;
  }

  /**
   * Returns the depth texture, if this target was created with depthTexture:true.
   * @returns {WebGLTexture|null}
   */
  getDepthTexture() {
    return this._depthTexture;
  }

  /**
   * Resize all attachments.
   * @param {number} w
   * @param {number} h
   */
  resize(w, h) {
    if (this.width === w && this.height === h) return;
    this.width  = w;
    this.height = h;
    this.destroy();
    this._colorTextures = [];
    this._depthTexture  = null;
    this._depthRBO = null;
    this._fbo = null;
    this._build();
  }

  destroy() {
    const gl = this.gl;
    for (const t of this._colorTextures) gl.deleteTexture(t);
    if (this._depthTexture) gl.deleteTexture(this._depthTexture);
    if (this._depthRBO) gl.deleteRenderbuffer(this._depthRBO);
    if (this._fbo)      gl.deleteFramebuffer(this._fbo);
    this._colorTextures = [];
    this._depthTexture  = null;
    this._depthRBO = null;
    this._fbo = null;
  }
}
