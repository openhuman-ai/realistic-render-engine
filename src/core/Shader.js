/**
 * Shader — compile, link, cache uniform locations, set uniforms.
 */
export class Shader {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {string} vertexSrc
   * @param {string} fragmentSrc
   * @param {Record<string,string|number>} [defines]
   */
  constructor(gl, vertexSrc, fragmentSrc, defines = {}) {
    this.gl = gl;
    this._uniformCache = new Map();
    this.program = this._build(vertexSrc, fragmentSrc, defines);
  }

  // ------------------------------------------------------------------ compile
  _injectDefines(src, defines) {
    const lines = Object.entries(defines)
      .map(([k, v]) => `#define ${k} ${v}`)
      .join('\n');
    // Insert defines after the first #version directive if present.
    return src.replace(/(#version\s+\S+[^\n]*\n)/, `$1${lines}\n`);
  }

  _compileShader(type, src) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      const typeName = type === gl.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT';
      console.error(`[Shader] ${typeName} compile error:\n${info}`);
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  _build(vertSrc, fragSrc, defines) {
    const gl = this.gl;
    const vSrc = this._injectDefines(vertSrc, defines);
    const fSrc = this._injectDefines(fragSrc, defines);

    const vert = this._compileShader(gl.VERTEX_SHADER,   vSrc);
    const frag = this._compileShader(gl.FRAGMENT_SHADER, fSrc);
    if (!vert || !frag) throw new Error('[Shader] Shader compilation failed.');

    const prog = gl.createProgram();
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);

    gl.deleteShader(vert);
    gl.deleteShader(frag);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error(`[Shader] Program link error:\n${info}`);
    }
    return prog;
  }

  // ---------------------------------------------------------------- uniforms
  getUniformLocation(name) {
    if (this._uniformCache.has(name)) return this._uniformCache.get(name);
    const loc = this.gl.getUniformLocation(this.program, name);
    this._uniformCache.set(name, loc);
    return loc;
  }

  use() {
    this.gl.useProgram(this.program);
  }

  setFloat(name, v) {
    this.gl.uniform1f(this.getUniformLocation(name), v);
  }

  setInt(name, v) {
    this.gl.uniform1i(this.getUniformLocation(name), v);
  }

  setVec2(name, x, y) {
    this.gl.uniform2f(this.getUniformLocation(name), x, y);
  }

  setVec3(name, x, y, z) {
    this.gl.uniform3f(this.getUniformLocation(name), x, y, z);
  }

  setVec4(name, x, y, z, w) {
    this.gl.uniform4f(this.getUniformLocation(name), x, y, z, w);
  }

  /** @param {Float32Array} mat3array — 9 elements, column-major */
  setMat3(name, mat3array) {
    this.gl.uniformMatrix3fv(this.getUniformLocation(name), false, mat3array);
  }

  /** @param {Float32Array} mat4array — 16 elements, column-major */
  setMat4(name, mat4array) {
    this.gl.uniformMatrix4fv(this.getUniformLocation(name), false, mat4array);
  }

  /**
   * Bind a texture to a uniform sampler.
   * @param {string}       name
   * @param {number}       unit    — texture unit index
   * @param {WebGLTexture} texture
   * @param {number}       [target] — default gl.TEXTURE_2D
   */
  setTexture(name, unit, texture, target) {
    const gl = this.gl;
    const tgt = target ?? gl.TEXTURE_2D;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(tgt, texture);
    this.gl.uniform1i(this.getUniformLocation(name), unit);
  }

  destroy() {
    this.gl.deleteProgram(this.program);
    this.program = null;
    this._uniformCache.clear();
  }
}
