/**
 * PostProcessStack — extensible chain of full-screen post-processing passes.
 *
 * Each pass receives the previous pass's output texture and renders to either
 * an internal ping-pong render target or the default canvas framebuffer (final
 * pass).
 *
 * Built-in pass: ACES filmic tone-mapping with exposure control.
 * Additional passes can be injected via addPass().
 */
import { Shader }       from '../core/Shader.js';
import { RenderTarget } from '../core/RenderTarget.js';

// ─────────────────────────────────────────────────────────────────────────────
// Shared fullscreen-triangle vertex shader (used by all passes)
// ─────────────────────────────────────────────────────────────────────────────
const FULLSCREEN_VERT = /* glsl */`#version 300 es
precision highp float;

// Oversized triangle covering the entire NDC clip space — no VBO needed.
// gl_VertexID: 0→(-1,-1), 1→(3,-1), 2→(-1,3)
void main() {
  vec2 pos;
  pos.x = (gl_VertexID == 1) ? 3.0 : -1.0;
  pos.y = (gl_VertexID == 2) ? 3.0 : -1.0;
  gl_Position = vec4(pos, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ACES filmic tone-mapping + gamma correction fragment shader
// ─────────────────────────────────────────────────────────────────────────────
const TONEMAP_FRAG = /* glsl */`#version 300 es
precision mediump float;

uniform sampler2D u_HDRBuffer;
uniform float     u_Exposure;

out vec4 fragColor;

// ACES fitted curve by Krzysztof Narkowicz
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 ACESFilmic(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec3  hdr   = texelFetch(u_HDRBuffer, coord, 0).rgb;

  // Apply exposure
  hdr *= u_Exposure;

  // ACES filmic tone-map
  vec3 ldr = ACESFilmic(hdr);

  // Gamma correction (sRGB γ≈2.2)
  ldr = pow(ldr, vec3(1.0 / 2.2));

  fragColor = vec4(ldr, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
export class PostProcessStack {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   * @param {{ width: number, height: number }} [opts]
   */
  constructor(glContext, stateCache, opts = {}) {
    this.ctx    = glContext;
    this.gl     = glContext.gl;
    this.cache  = stateCache;

    /** @type {Array<{name:string, shader:Shader, uniforms:Record<string,*>, enabled:boolean}>} */
    this.passes = [];

    this._width  = opts.width  ?? glContext.canvas.width  ?? 1;
    this._height = opts.height ?? glContext.canvas.height ?? 1;

    // Ping-pong targets — allocated lazily on first render or on resize
    this._pingPong = [null, null];

    // Built-in ACES tone-map shader
    this._tonemapShader = new Shader(this.gl, FULLSCREEN_VERT, TONEMAP_FRAG);
    this._exposure = 1.0;

    // Register the ACES pass as the default (and currently only) pass
    this.passes.push({
      name:    'tonemap_aces',
      shader:  this._tonemapShader,
      uniforms: {},
      enabled: true,
    });
  }

  // ─────────────────────────────────── public API

  /**
   * Set exposure value applied before tone-mapping.
   * @param {number} value
   */
  setExposure(value) {
    this._exposure = Math.max(0.001, value);
  }

  /**
   * Append a custom post-process pass.
   * @param {{ name: string, shader: Shader, uniforms?: Record<string,*>, enabled?: boolean }} pass
   */
  addPass(pass) {
    this.passes.push({ enabled: true, uniforms: {}, ...pass });
  }

  /**
   * Execute all enabled passes and blit the final result to the canvas.
   *
   * @param {WebGLTexture} inputTexture — HDR source (e.g. lighting pass output)
   * @param {number} width
   * @param {number} height
   */
  render(inputTexture, width, height) {
    const gl    = this.gl;
    const cache = this.cache;

    const enabled = this.passes.filter(p => p.enabled);
    if (!enabled.length) return;

    this._ensurePingPong(width, height);

    cache.setDepthTest(false);
    cache.setDepthWrite(false);
    cache.setBlend(false);
    cache.setCullFace(false, gl.BACK);

    let currentInput = inputTexture;

    for (let i = 0; i < enabled.length; i++) {
      const pass     = enabled[i];
      const isLast   = i === enabled.length - 1;
      const outputRT = isLast ? null : this._pingPong[i % 2];

      if (isLast) {
        // Blit to canvas default framebuffer
        cache.bindFramebuffer(gl.FRAMEBUFFER, null);
        cache.setViewport(0, 0, width, height);
      } else {
        outputRT.bind();
      }

      cache.useProgram(pass.shader.program);

      // Bind input texture to unit 0 — use gl directly to avoid StateCache texture skip
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, currentInput);

      // ACES tone-map specific uniforms
      if (pass.name === 'tonemap_aces') {
        pass.shader.setInt('u_HDRBuffer', 0);
        pass.shader.setFloat('u_Exposure', this._exposure);
      }

      // Custom pass uniforms
      for (const [k, v] of Object.entries(pass.uniforms)) {
        if (typeof v === 'number') pass.shader.setFloat(k, v);
      }

      // Draw oversized triangle (3 vertices, no VAO needed)
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      if (!isLast) {
        outputRT.unbind();
        currentInput = outputRT.getColorTexture(0);
      }
    }
  }

  /**
   * Handle canvas resize — reallocate ping-pong targets.
   * @param {number} w
   * @param {number} h
   */
  resize(w, h) {
    if (this._width === w && this._height === h) return;
    this._width  = w;
    this._height = h;
    for (let i = 0; i < 2; i++) {
      if (this._pingPong[i]) {
        this._pingPong[i].resize(w, h);
      }
    }
  }

  destroy() {
    for (const rt of this._pingPong) rt?.destroy();
    this._pingPong = [null, null];
    this._tonemapShader?.destroy();
    this._tonemapShader = null;
    this.passes = [];
  }

  // ─────────────────────────────────── private

  /** Lazily create ping-pong render targets (RGBA8, no depth). */
  _ensurePingPong(w, h) {
    for (let i = 0; i < 2; i++) {
      if (!this._pingPong[i]) {
        this._pingPong[i] = new RenderTarget(this.gl, {
          width: w, height: h,
          colorAttachments: ['RGBA8'],
          depthAttachment: false,
        });
      } else {
        this._pingPong[i].resize(w, h);
      }
    }
  }
}
