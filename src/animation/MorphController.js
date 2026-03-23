/**
 * MorphController — GPU-accelerated texture-based morph target / blendshape system.
 *
 * Supports up to MAX_MORPH_TARGETS (52) morph targets, matching the ARKit/FACS 52-shape set.
 *
 * GPU texture layout (two RGBA32F textures):
 *   u_MorphPosDeltaTex — width=vertexCount, height=numMorphs
 *     texelFetch(tex, ivec2(vertexIndex, morphIndex), 0) → (dx, dy, dz, 0)
 *   u_MorphNrmDeltaTex — width=vertexCount, height=numMorphs
 *     texelFetch(tex, ivec2(vertexIndex, morphIndex), 0) → (dnx, dny, dnz, 0)
 *
 * Shader uniforms set via bind():
 *   uniform int             u_NumMorphs;
 *   uniform float           u_MorphWeights[52];
 *   uniform highp sampler2D u_MorphPosDeltaTex;
 *   uniform highp sampler2D u_MorphNrmDeltaTex;
 */

export const MAX_MORPH_TARGETS = 52;

/**
 * ARKit / Apple FACS 52 blendshape names — OpenHuman canonical order.
 * @type {readonly string[]}
 */
export const FACS_NAMES = Object.freeze([
  'browDownLeft',     'browDownRight',     'browInnerUp',
  'browOuterUpLeft',  'browOuterUpRight',
  'cheekPuff',        'cheekSquintLeft',   'cheekSquintRight',
  'eyeBlinkLeft',     'eyeBlinkRight',
  'eyeLookDownLeft',  'eyeLookDownRight',
  'eyeLookInLeft',    'eyeLookInRight',
  'eyeLookOutLeft',   'eyeLookOutRight',
  'eyeLookUpLeft',    'eyeLookUpRight',
  'eyeSquintLeft',    'eyeSquintRight',
  'eyeWideLeft',      'eyeWideRight',
  'jawForward',       'jawLeft',           'jawRight',     'jawOpen',
  'mouthClose',
  'mouthDimpleLeft',  'mouthDimpleRight',
  'mouthFrownLeft',   'mouthFrownRight',
  'mouthFunnel',
  'mouthLeft',        'mouthRight',
  'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft',   'mouthPressRight',
  'mouthPucker',
  'mouthRollLower',   'mouthRollUpper',
  'mouthShrugLower',  'mouthShrugUpper',
  'mouthSmileLeft',   'mouthSmileRight',
  'mouthStretchLeft', 'mouthStretchRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight',
  'noseSneerLeft',    'noseSneerRight',
  'tongueOut',
]);

export class MorphController {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {readonly string[]} [names]   — morph target names; defaults to FACS_NAMES
   * @param {number}            [vertexCount] — vertex count for GPU texture allocation (0 = deferred)
   */
  constructor(gl, names = FACS_NAMES, vertexCount = 0) {
    this.gl          = gl;
    this._names      = Array.from(names).slice(0, MAX_MORPH_TARGETS);
    this.numMorphs   = this._names.length;
    this.vertexCount = 0;

    // CPU weight array — padded to MAX_MORPH_TARGETS so uniform upload is always the same size
    this._weights = new Float32Array(MAX_MORPH_TARGETS);

    // GPU delta textures — allocated when vertexCount is known
    this._posTex = null;
    this._nrmTex = null;

    if (vertexCount > 0) {
      this._initTextures(vertexCount);
    }
  }

  // ──────────────────────────────────────────────────────────── name / weight API

  /** @returns {readonly string[]} */
  get names() { return this._names; }

  /**
   * Set a morph weight by name.
   * @param {string} name
   * @param {number} weight — clamped to [0, 1]
   */
  set(name, weight) {
    const idx = this._names.indexOf(name);
    if (idx >= 0) {
      this._weights[idx] = Math.max(0, Math.min(1, weight));
    } else {
      console.warn(`[MorphController] Unknown morph target: "${name}"`);
    }
  }

  /**
   * Set multiple weights from a { name: weight } map.
   * @param {Record<string, number>} map
   */
  setMany(map) {
    for (const [name, weight] of Object.entries(map)) {
      this.set(name, weight);
    }
  }

  /**
   * Set a morph weight by index.
   * @param {number} index
   * @param {number} weight — clamped to [0, 1]
   */
  setByIndex(index, weight) {
    if (index >= 0 && index < MAX_MORPH_TARGETS) {
      this._weights[index] = Math.max(0, Math.min(1, weight));
    }
  }

  /**
   * Get the current weight for a named morph target.
   * @param {string} name
   * @returns {number}
   */
  getWeight(name) {
    const idx = this._names.indexOf(name);
    return idx >= 0 ? this._weights[idx] : 0;
  }

  // ──────────────────────────────────────────────────────────── GPU data

  /**
   * Allocate or re-allocate GPU delta textures for a given vertex count.
   * Must be called before uploadMorphDeltas() if vertexCount was 0 at construction.
   * @param {number} vertexCount
   */
  setVertexCount(vertexCount) {
    const gl = this.gl;
    if (this._posTex) { gl.deleteTexture(this._posTex); this._posTex = null; }
    if (this._nrmTex) { gl.deleteTexture(this._nrmTex); this._nrmTex = null; }
    this.vertexCount = 0;
    if (vertexCount > 0) {
      this._initTextures(vertexCount);
    }
  }

  /**
   * Upload position and normal deltas for one morph target.
   *
   * @param {number}            morphIndex — index in the names array [0, numMorphs)
   * @param {Float32Array|null} posDeltas  — 3 * vertexCount floats (dx,dy,dz per vertex)
   * @param {Float32Array|null} nrmDeltas  — 3 * vertexCount floats (dnx,dny,dnz per vertex)
   */
  uploadMorphDeltas(morphIndex, posDeltas, nrmDeltas) {
    if (!this._posTex || !this._nrmTex) {
      console.warn('[MorphController] Textures not initialised — call setVertexCount() first.');
      return;
    }
    if (morphIndex < 0 || morphIndex >= this.numMorphs) {
      console.warn(`[MorphController] morphIndex ${morphIndex} out of range (0..${this.numMorphs - 1}).`);
      return;
    }

    const gl  = this.gl;
    const vc  = this.vertexCount;
    const row = this._rowBuf;   // reuse preallocated scratch row buffer

    // Position deltas → RGBA row
    if (posDeltas) {
      for (let v = 0; v < vc; v++) {
        row[v * 4]     = posDeltas[v * 3];
        row[v * 4 + 1] = posDeltas[v * 3 + 1];
        row[v * 4 + 2] = posDeltas[v * 3 + 2];
        row[v * 4 + 3] = 0;
      }
    } else {
      row.fill(0);
    }
    gl.bindTexture(gl.TEXTURE_2D, this._posTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, morphIndex, vc, 1, gl.RGBA, gl.FLOAT, row);

    // Normal deltas → RGBA row
    if (nrmDeltas) {
      for (let v = 0; v < vc; v++) {
        row[v * 4]     = nrmDeltas[v * 3];
        row[v * 4 + 1] = nrmDeltas[v * 3 + 1];
        row[v * 4 + 2] = nrmDeltas[v * 3 + 2];
        row[v * 4 + 3] = 0;
      }
    } else {
      row.fill(0);
    }
    gl.bindTexture(gl.TEXTURE_2D, this._nrmTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, morphIndex, vc, 1, gl.RGBA, gl.FLOAT, row);

    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ──────────────────────────────────────────────────────────── shader binding

  /**
   * Bind morph textures and upload weights/count uniforms to a geometry shader.
   *
   * @param {import('../core/Shader.js').Shader} shader
   * @param {number}       posTexUnit  — texture unit for position delta texture
   * @param {number}       nrmTexUnit  — texture unit for normal delta texture
   * @param {WebGLTexture} fallbackTex — dummy 1×1 texture used when GPU textures are not ready
   */
  bind(shader, posTexUnit, nrmTexUnit, fallbackTex) {
    const gl     = this.gl;
    const posTex = this._posTex ?? fallbackTex;
    const nrmTex = this._nrmTex ?? fallbackTex;

    gl.activeTexture(gl.TEXTURE0 + posTexUnit);
    gl.bindTexture(gl.TEXTURE_2D, posTex);
    shader.setInt('u_MorphPosDeltaTex', posTexUnit);

    gl.activeTexture(gl.TEXTURE0 + nrmTexUnit);
    gl.bindTexture(gl.TEXTURE_2D, nrmTex);
    shader.setInt('u_MorphNrmDeltaTex', nrmTexUnit);

    // Only activate morphs when GPU textures are ready
    const activeMorphs = this._posTex ? this.numMorphs : 0;
    shader.setInt('u_NumMorphs', activeMorphs);
    gl.uniform1fv(shader.getUniformLocation('u_MorphWeights'), this._weights);
  }

  // ──────────────────────────────────────────────────────────── destroy

  destroy() {
    const gl = this.gl;
    if (this._posTex) { gl.deleteTexture(this._posTex); this._posTex = null; }
    if (this._nrmTex) { gl.deleteTexture(this._nrmTex); this._nrmTex = null; }
    this._weights = null;
    this._rowBuf  = null;
  }

  // ──────────────────────────────────────────────────────────── private

  /** @private */
  _initTextures(vertexCount) {
    const gl     = this.gl;
    const n      = this.numMorphs;
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);

    if (vertexCount > maxTex) {
      console.warn(
        `[MorphController] vertexCount (${vertexCount}) exceeds MAX_TEXTURE_SIZE (${maxTex}).` +
        ' Morph deltas will be truncated to the first ' + maxTex + ' vertices.'
      );
      vertexCount = maxTex;
    }

    // Reusable scratch buffer (RGBA per vertex for one morph row)
    this._rowBuf = new Float32Array(vertexCount * 4);

    const makeTex = () => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, vertexCount, n);
      // Zero-fill row by row to avoid undefined initial values
      const zero = this._rowBuf;
      for (let m = 0; m < n; m++) {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, m, vertexCount, 1, gl.RGBA, gl.FLOAT, zero);
      }
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return tex;
    };

    this._posTex     = makeTex();
    this._nrmTex     = makeTex();
    this.vertexCount = vertexCount;
  }
}
