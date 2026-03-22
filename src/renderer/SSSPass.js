/**
 * SSSPass — Jorge Jimenez-inspired separable screen-space subsurface scattering.
 *
 * Applies a two-pass (horizontal + vertical) edge-stopped Gaussian blur to the
 * HDR lighting buffer.  The blurred "scattered" result is additively composited
 * back onto the original with a configurable per-channel color tint (the warm
 * red-dominant skin scatter profile) and a global strength scalar.
 *
 * Pass order
 * ──────────
 *   1. Horizontal Gaussian blur   (HDR → _tmpRT)
 *   2. Vertical   Gaussian blur + composite (HDR + _tmpRT → output target / canvas)
 *
 * Both passes use the G-buffer normal texture for edge-stopping: samples whose
 * surface normals deviate too far from the centre pixel are down-weighted,
 * preventing colour bleeding across hard geometry boundaries.
 *
 * Controls
 * ────────
 *   enabled  — on/off toggle  (default: false)
 *   strength — blend factor 0–2  (default: 0.5)
 *   width    — kernel step size in pixels  (default: 2.0)
 *   color    — vec3 RGB scatter tint, e.g. [1, 0.45, 0.18] for warm skin
 */
import { Shader }       from '../core/Shader.js';
import { RenderTarget } from '../core/RenderTarget.js';

// ─────────────────────────────────────────────────────────────────────────────
// Shared fullscreen-triangle vertex shader
// ─────────────────────────────────────────────────────────────────────────────
const FULLSCREEN_VERT = /* glsl */`#version 300 es
precision highp float;
void main() {
  vec2 pos;
  pos.x = (gl_VertexID == 1) ? 3.0 : -1.0;
  pos.y = (gl_VertexID == 2) ? 3.0 : -1.0;
  gl_Position = vec4(pos, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Horizontal Gaussian blur pass
// ─────────────────────────────────────────────────────────────────────────────
const SSS_H_FRAG = /* glsl */`#version 300 es
precision highp float;

uniform sampler2D u_HDR;           // linear HDR lighting buffer
uniform sampler2D u_GNormalRough;  // G-buffer RT1: worldNormal.xyz (encoded 0..1)
uniform float     u_Width;         // kernel step multiplier (pixels)

out vec4 fragColor;

// 7-tap symmetric Gaussian (sigma ≈ 1.9)
const int   TAPS    = 7;
const float OFF[7]  = float[](-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0);
const float WGT[7]  = float[](0.03, 0.09, 0.19, 0.38, 0.19, 0.09, 0.03);

void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);

  // Centre normal for edge-stopping
  vec3 cN = texelFetch(u_GNormalRough, fc, 0).xyz * 2.0 - 1.0;

  vec3  result = vec3(0.0);
  float totalW = 0.0;

  for (int i = 0; i < TAPS; i++) {
    ivec2 sc  = fc + ivec2(int(OFF[i] * u_Width), 0);
    vec3  sN  = texelFetch(u_GNormalRough, sc, 0).xyz * 2.0 - 1.0;
    // Edge-stop: reduce weight when normals diverge
    float edgeW = pow(max(0.0, dot(cN, sN)), 12.0);
    float w     = WGT[i] * edgeW;
    result += texelFetch(u_HDR, sc, 0).rgb * w;
    totalW += w;
  }

  if (totalW > 0.001) result /= totalW;
  fragColor = vec4(result, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Vertical Gaussian blur pass + composite with original HDR
// ─────────────────────────────────────────────────────────────────────────────
const SSS_V_COMPOSITE_FRAG = /* glsl */`#version 300 es
precision highp float;

uniform sampler2D u_HDROriginal;   // original (un-blurred) HDR
uniform sampler2D u_HDRHBlurred;   // horizontally blurred HDR
uniform sampler2D u_GNormalRough;  // G-buffer RT1
uniform sampler2D u_AlbedoAO;      // G-buffer RT0 (for skin mask)
uniform float     u_Width;         // kernel step multiplier
uniform float     u_Strength;      // SSS blend strength (0..1+)
uniform vec3      u_SSSColor;      // per-channel scatter tint

out vec4 fragColor;

const int   TAPS   = 7;
const float OFF[7] = float[](-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0);
const float WGT[7] = float[](0.03, 0.09, 0.19, 0.38, 0.19, 0.09, 0.03);

void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);

  // Centre normal for edge-stopping
  vec3 cN = texelFetch(u_GNormalRough, fc, 0).xyz * 2.0 - 1.0;

  // Vertical Gaussian over the horizontal-blurred buffer
  vec3  blurred = vec3(0.0);
  float totalW  = 0.0;

  for (int i = 0; i < TAPS; i++) {
    ivec2 sc  = fc + ivec2(0, int(OFF[i] * u_Width));
    vec3  sN  = texelFetch(u_GNormalRough, sc, 0).xyz * 2.0 - 1.0;
    float edgeW = pow(max(0.0, dot(cN, sN)), 12.0);
    float w     = WGT[i] * edgeW;
    blurred += texelFetch(u_HDRHBlurred, sc, 0).rgb * w;
    totalW  += w;
  }
  if (totalW > 0.001) blurred /= totalW;

  // Composite: add color-tinted scattered component onto original
  vec3 original = texelFetch(u_HDROriginal, fc, 0).rgb;
  vec3 albedo   = texelFetch(u_AlbedoAO, fc, 0).rgb;

  // Cheap skin mask from albedo hue (warm/red-biased surfaces get more SSS)
  float redDom  = albedo.r - max(albedo.g, albedo.b);
  // 3.0 boosts weak red-dominance into a usable mask range for typical skin albedo.
  // 0.35 is a base threshold so neutral warm tones still receive subtle SSS.
  float skinMask = clamp(redDom * 3.0 + 0.35, 0.0, 1.0);

  // Scattered = tinted blurred - original, represents extra scattering
  vec3 scattered = blurred * u_SSSColor;
  // Additive blend: original + scatter contribution
  vec3 result = original + (scattered - original) * clamp(u_Strength, 0.0, 1.0) * skinMask;

  fragColor = vec4(result, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
export class SSSPass {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   * @param {{ width?: number, height?: number }} [opts]
   */
  constructor(glContext, stateCache, opts = {}) {
    this.ctx   = glContext;
    this.gl    = glContext.gl;
    this.cache = stateCache;

    this._width  = opts.width  ?? glContext.canvas.width  ?? 1;
    this._height = opts.height ?? glContext.canvas.height ?? 1;

    // Shader for horizontal pass
    this._hShader = new Shader(this.gl, FULLSCREEN_VERT, SSS_H_FRAG);
    // Shader for vertical pass + composite
    this._vShader = new Shader(this.gl, FULLSCREEN_VERT, SSS_V_COMPOSITE_FRAG);

    // Temporary render target for horizontal blur output
    this._tmpRT = new RenderTarget(this.gl, {
      width: this._width, height: this._height,
      colorAttachments: ['RGBA16F'],
      depthAttachment:  false,
    });

    // SSS parameters
    this.enabled  = false;
    this.strength = 0.5;
    this.width    = 2.0;
    this.color    = [1.0, 0.45, 0.18];  // warm skin scatter profile
  }

  // ─────────────────────────────────────────── public API

  /** @param {boolean} v */
  setEnabled(v)  { this.enabled = !!v; }
  /** @param {number} v   0 → no SSS, 1 → full SSS */
  setStrength(v) { this.strength = Math.max(0, v); }
  /** @param {number} v   blur kernel step size in pixels */
  setWidth(v)    { this.width = Math.max(0.5, v); }
  /** @param {number} r @param {number} g @param {number} b */
  setColor(r, g, b) { this.color = [r, g, b]; }

  /**
   * Execute the SSS pass.
   *
   * Reads from hdrTex + gNormalTex, writes result to the currently-bound
   * framebuffer (or canvas if null).
   *
   * @param {WebGLTexture} hdrTex        — HDR lighting result
   * @param {WebGLTexture} gNormalTex    — G-buffer RT1 (worldNormal * 0.5 + 0.5)
   * @param {WebGLTexture} gAlbedoTex    — G-buffer RT0 (albedo + ao)
   * @param {number} w
   * @param {number} h
   */
  render(hdrTex, gNormalTex, gAlbedoTex, w, h) {
    if (!this.enabled) return;

    const gl    = this.gl;
    const cache = this.cache;

    this._ensureTmp(w, h);

    cache.setDepthTest(false);
    cache.setDepthWrite(false);
    cache.setBlend(false);
    cache.setCullFace(false, gl.BACK);

    // ── Pass 1: horizontal blur → _tmpRT
    this._tmpRT.bind();
    cache.setViewport(0, 0, w, h);
    cache.useProgram(this._hShader.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, hdrTex);
    this._hShader.setInt('u_HDR', 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, gNormalTex);
    this._hShader.setInt('u_GNormalRough', 1);
    this._hShader.setFloat('u_Width', this.width);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this._tmpRT.unbind();

    // ── Pass 2: vertical blur + composite → caller's bound framebuffer
    cache.setViewport(0, 0, w, h);
    cache.useProgram(this._vShader.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, hdrTex);
    this._vShader.setInt('u_HDROriginal', 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this._tmpRT.getColorTexture(0));
    this._vShader.setInt('u_HDRHBlurred', 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, gNormalTex);
    this._vShader.setInt('u_GNormalRough', 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, gAlbedoTex);
    this._vShader.setInt('u_AlbedoAO', 3);
    this._vShader.setFloat('u_Width',    this.width);
    this._vShader.setFloat('u_Strength', this.strength);
    this._vShader.setVec3('u_SSSColor', this.color[0], this.color[1], this.color[2]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Handle canvas resize. */
  resize(w, h) {
    if (this._width === w && this._height === h) return;
    this._width  = w;
    this._height = h;
    this._tmpRT?.resize(w, h);
  }

  destroy() {
    this._hShader?.destroy();
    this._vShader?.destroy();
    this._tmpRT?.destroy();
    this._hShader = this._vShader = this._tmpRT = null;
  }

  // ─────────────────────────────────────────── private

  /** Lazily resize or create the temporary horizontal-blur render target. */
  _ensureTmp(w, h) {
    if (!this._tmpRT) {
      this._tmpRT = new RenderTarget(this.gl, {
        width: w, height: h,
        colorAttachments: ['RGBA16F'],
        depthAttachment:  false,
      });
    } else {
      this._tmpRT.resize(w, h);
    }
  }
}
