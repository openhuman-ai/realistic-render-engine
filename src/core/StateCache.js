/**
 * StateCache — Centralised WebGL state tracker.
 * Every setter checks current state before issuing a GL call to minimise
 * redundant driver state changes in hot render loops.
 */
export class StateCache {
  /** @param {WebGL2RenderingContext} gl */
  constructor(gl) {
    this.gl = gl;
    this.reset();
  }

  reset() {
    this._program      = null;
    this._vao          = null;
    this._arrayBuffer  = null;
    this._elementBuffer = null;
    this._readFramebuffer  = null;
    this._drawFramebuffer  = null;
    // 32 texture units is a safe upper bound for all WebGL2 implementations.
    this._textures     = new Array(32).fill(null).map(() => ({ target: 0, tex: null }));
    this._blendEnabled = false;
    this._depthTest    = false;
    this._depthWrite   = true;
    this._cullFace     = false;
    this._cullFaceMode = 0;
    this._viewport     = { x: -1, y: -1, w: -1, h: -1 };
  }

  // ------------------------------------------------------------------ program
  useProgram(program) {
    if (this._program === program) return;
    this._program = program;
    this.gl.useProgram(program);
  }

  // ----------------------------------------------------------------------- VAO
  bindVAO(vao) {
    if (this._vao === vao) return;
    this._vao = vao;
    this.gl.bindVertexArray(vao);
  }

  // ---------------------------------------------------------------------- buffers
  bindBuffer(target, buf) {
    const gl = this.gl;
    if (target === gl.ARRAY_BUFFER) {
      if (this._arrayBuffer === buf) return;
      this._arrayBuffer = buf;
    } else if (target === gl.ELEMENT_ARRAY_BUFFER) {
      if (this._elementBuffer === buf) return;
      this._elementBuffer = buf;
    }
    gl.bindBuffer(target, buf);
  }

  // ---------------------------------------------------------------- framebuffer
  bindFramebuffer(target, fbo) {
    const gl = this.gl;
    if (target === gl.READ_FRAMEBUFFER) {
      if (this._readFramebuffer === fbo) return;
      this._readFramebuffer = fbo;
    } else if (target === gl.DRAW_FRAMEBUFFER) {
      if (this._drawFramebuffer === fbo) return;
      this._drawFramebuffer = fbo;
    } else {
      // gl.FRAMEBUFFER — sets both
      if (this._readFramebuffer === fbo && this._drawFramebuffer === fbo) return;
      this._readFramebuffer = fbo;
      this._drawFramebuffer = fbo;
    }
    gl.bindFramebuffer(target, fbo);
  }

  // ------------------------------------------------------------------ textures
  /**
   * @param {number} unit  — texture unit index (0-based)
   * @param {number} target — e.g. gl.TEXTURE_2D
   * @param {WebGLTexture|null} tex
   */
  bindTexture(unit, target, tex) {
    const slot = this._textures[unit];
    if (slot.target === target && slot.tex === tex) return;
    slot.target = target;
    slot.tex    = tex;
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(target, tex);
  }

  // ------------------------------------------------------------------- blend
  setBlend(enabled) {
    if (this._blendEnabled === enabled) return;
    this._blendEnabled = enabled;
    enabled ? this.gl.enable(this.gl.BLEND) : this.gl.disable(this.gl.BLEND);
  }

  // ---------------------------------------------------------------- depth test
  setDepthTest(enabled) {
    if (this._depthTest === enabled) return;
    this._depthTest = enabled;
    enabled ? this.gl.enable(this.gl.DEPTH_TEST) : this.gl.disable(this.gl.DEPTH_TEST);
  }

  // --------------------------------------------------------------- depth write
  setDepthWrite(enabled) {
    if (this._depthWrite === enabled) return;
    this._depthWrite = enabled;
    this.gl.depthMask(enabled);
  }

  // ---------------------------------------------------------------- cull face
  /**
   * @param {boolean} enabled
   * @param {number}  face — gl.BACK | gl.FRONT | gl.FRONT_AND_BACK
   */
  setCullFace(enabled, face) {
    if (this._cullFace !== enabled) {
      this._cullFace = enabled;
      enabled ? this.gl.enable(this.gl.CULL_FACE) : this.gl.disable(this.gl.CULL_FACE);
    }
    if (enabled && this._cullFaceMode !== face) {
      this._cullFaceMode = face;
      this.gl.cullFace(face);
    }
  }

  // ----------------------------------------------------------------- viewport
  setViewport(x, y, w, h) {
    const vp = this._viewport;
    if (vp.x === x && vp.y === y && vp.w === w && vp.h === h) return;
    vp.x = x; vp.y = y; vp.w = w; vp.h = h;
    this.gl.viewport(x, y, w, h);
  }
}
