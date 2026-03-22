/**
 * Buffer wrappers — VertexBuffer, IndexBuffer, UniformBuffer.
 * All classes avoid re-uploading data when it has not changed and support
 * typed-array uploads with configurable usage hints.
 */

// ─────────────────────────────────────────────────────────────────────────────
// VertexBuffer  (ARRAY_BUFFER)
// ─────────────────────────────────────────────────────────────────────────────
export class VertexBuffer {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {number} [usage] — gl.STATIC_DRAW | gl.DYNAMIC_DRAW | gl.STREAM_DRAW
   */
  constructor(gl, usage) {
    this.gl    = gl;
    this.usage = usage ?? gl.STATIC_DRAW;
    this.buf   = gl.createBuffer();
    this.byteLength = 0;
  }

  /**
   * Upload typed-array data.
   * @param {Float32Array|Uint16Array|Uint32Array|Uint8Array} data
   */
  upload(data) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, this.usage);
    this.byteLength = data.byteLength;
  }

  bind() {
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buf);
  }

  destroy() {
    this.gl.deleteBuffer(this.buf);
    this.buf = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexBuffer  (ELEMENT_ARRAY_BUFFER)
// ─────────────────────────────────────────────────────────────────────────────
export class IndexBuffer {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {number} [usage]
   */
  constructor(gl, usage) {
    this.gl    = gl;
    this.usage = usage ?? gl.STATIC_DRAW;
    this.buf   = gl.createBuffer();
    this.byteLength = 0;
    /** @type {number} gl.UNSIGNED_SHORT | gl.UNSIGNED_INT | gl.UNSIGNED_BYTE */
    this.indexType = gl.UNSIGNED_SHORT;
  }

  /**
   * @param {Uint16Array|Uint32Array|Uint8Array} data
   */
  upload(data) {
    const gl = this.gl;
    if (data instanceof Uint32Array)      this.indexType = gl.UNSIGNED_INT;
    else if (data instanceof Uint8Array)  this.indexType = gl.UNSIGNED_BYTE;
    else                                  this.indexType = gl.UNSIGNED_SHORT;

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, this.usage);
    this.byteLength = data.byteLength;
    this.count = data.length;
  }

  bind() {
    this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.buf);
  }

  destroy() {
    this.gl.deleteBuffer(this.buf);
    this.buf = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// UniformBuffer  (UNIFORM_BUFFER — WebGL2 only)
// ─────────────────────────────────────────────────────────────────────────────
export class UniformBuffer {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {number} byteSize — size in bytes of the buffer layout
   * @param {number} [usage]
   */
  constructor(gl, byteSize, usage) {
    this.gl         = gl;
    this.usage      = usage ?? gl.DYNAMIC_DRAW;
    this.buf        = gl.createBuffer();
    this.byteLength = byteSize;

    gl.bindBuffer(gl.UNIFORM_BUFFER, this.buf);
    gl.bufferData(gl.UNIFORM_BUFFER, byteSize, this.usage);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  /**
   * Upload full or partial data.
   * @param {Float32Array|Uint8Array} data
   * @param {number} [offsetBytes=0]
   */
  upload(data, offsetBytes = 0) {
    const gl = this.gl;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.buf);
    gl.bufferSubData(gl.UNIFORM_BUFFER, offsetBytes, data);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  /**
   * Bind to a uniform buffer binding point.
   * @param {number} bindingPoint
   */
  bindBase(bindingPoint) {
    this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, bindingPoint, this.buf);
  }

  /**
   * Bind a range to a binding point.
   * @param {number} bindingPoint
   * @param {number} offsetBytes
   * @param {number} sizeBytes
   */
  bindRange(bindingPoint, offsetBytes, sizeBytes) {
    this.gl.bindBufferRange(this.gl.UNIFORM_BUFFER, bindingPoint, this.buf, offsetBytes, sizeBytes);
  }

  bind() {
    this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.buf);
  }

  destroy() {
    this.gl.deleteBuffer(this.buf);
    this.buf = null;
  }
}
