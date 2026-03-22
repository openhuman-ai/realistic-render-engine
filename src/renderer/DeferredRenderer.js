/**
 * DeferredRenderer — WebGL 2.0 deferred shading pipeline.
 *
 * Pipeline stages
 * ───────────────
 * 1. [optional] Shadow pass  — depth from directional-light POV (via ShadowMap)
 * 2. Geometry   pass         — fill 3-RT G-buffer
 * 3. Lighting   pass         — fullscreen PBR + IBL + PCF shadows → HDR target
 * 4. Tone-map   pass         — ACES filmic via PostProcessStack → canvas
 *
 * G-buffer layout
 * ───────────────
 *   RT0  RGBA8   : albedo.rgb + ao
 *   RT1  RGBA16F : worldNormal.xyz + roughness
 *   RT2  RGBA16F : worldPos.xyz    + metallic
 *   depth renderbuffer (DEPTH24_STENCIL8)
 *
 * IBL
 * ───
 * Full split-sum IBL is supported when an irradiance and environment texture
 * are supplied via setEnvironment().  Without them the renderer uses built-in
 * hemisphere ambient + a BRDF-LUT-only specular contribution so the PBR
 * metallic-roughness response is always visible.
 *
 * A 128×128 BRDF look-up table is computed once at construction time using
 * a GGX importance-sampling render pass.
 */
import { Shader }          from '../core/Shader.js';
import { RenderTarget }    from '../core/RenderTarget.js';
import { PostProcessStack } from './PostProcessStack.js';
import { Mat4 }            from '../math/Mat4.js';
import { Mat3 }            from '../math/Mat3.js';

// ─────────────────────────────────────────────────────────────────────────────
// Shared fullscreen-triangle vertex shader (no VBO required)
// ─────────────────────────────────────────────────────────────────────────────
const FULLSCREEN_VERT = /* glsl */`#version 300 es
precision highp float;
void main() {
  vec2 pos;
  pos.x = (gl_VertexID == 1) ? 3.0 : -1.0;
  pos.y = (gl_VertexID == 2) ? 3.0 : -1.0;
  gl_Position = vec4(pos, 0.0, 1.0);
  // Derive UV from position for use in fragment shader if needed
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Geometry pass — static vertex shader
// ─────────────────────────────────────────────────────────────────────────────
const GBUF_VERT = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;
uniform mat3 u_NormalMatrix;

out vec3 v_WorldPos;
out vec3 v_Normal;
out vec2 v_TexCoord;

void main() {
  vec4 worldPos  = u_ModelMatrix * vec4(a_Position, 1.0);
  v_WorldPos     = worldPos.xyz;
  v_Normal       = normalize(u_NormalMatrix * a_Normal);
  v_TexCoord     = a_TexCoord;
  gl_Position    = u_ProjectionMatrix * u_ViewMatrix * worldPos;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Geometry pass — skinned (GPU dual-quaternion) vertex shader
// ─────────────────────────────────────────────────────────────────────────────
const GBUF_VERT_SKINNED = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;
in vec4 a_Joints;
in vec4 a_Weights;

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;
uniform highp sampler2D u_JointTexture;

out vec3 v_WorldPos;
out vec3 v_Normal;
out vec2 v_TexCoord;

vec3 quatRotate(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void applySkinning(inout vec3 pos, inout vec3 nrm) {
  vec4 blendReal = vec4(0.0);
  vec4 blendDual = vec4(0.0);
  vec4 ref = texelFetch(u_JointTexture, ivec2(0, int(a_Joints.x)), 0);
  for (int i = 0; i < 4; i++) {
    float w = a_Weights[i];
    if (w < 0.0001) continue;
    int ji = int(a_Joints[i]);
    vec4 re = texelFetch(u_JointTexture, ivec2(0, ji), 0);
    vec4 de = texelFetch(u_JointTexture, ivec2(1, ji), 0);
    float sgn = (dot(re, ref) < 0.0) ? -1.0 : 1.0;
    blendReal += sgn * w * re;
    blendDual += sgn * w * de;
  }
  float len = length(blendReal);
  if (len > 0.0001) { blendReal /= len; blendDual /= len; }
  vec3 t = 2.0 * (blendReal.w * blendDual.xyz
                - blendDual.w * blendReal.xyz
                + cross(blendReal.xyz, blendDual.xyz));
  pos = quatRotate(blendReal, pos) + t;
  nrm = quatRotate(blendReal, nrm);
}

void main() {
  vec3 pos = a_Position;
  vec3 nrm = a_Normal;
  applySkinning(pos, nrm);
  vec4 worldPos  = u_ModelMatrix * vec4(pos, 1.0);
  v_WorldPos     = worldPos.xyz;
  v_Normal       = normalize(mat3(u_ModelMatrix) * nrm);
  v_TexCoord     = a_TexCoord;
  gl_Position    = u_ProjectionMatrix * u_ViewMatrix * worldPos;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Geometry pass — fragment shader (writes to 3 MRT)
// ─────────────────────────────────────────────────────────────────────────────
const GBUF_FRAG = /* glsl */`#version 300 es
precision highp float;

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec2 v_TexCoord;

uniform vec3  u_BaseColor;
uniform float u_Roughness;
uniform float u_Metallic;
uniform float u_AO;

layout(location = 0) out vec4 gAlbedoAO;       // RGBA8  : albedo.rgb + ao
layout(location = 1) out vec4 gNormalRoughness; // RGBA16F: normalWS.xyz + roughness
layout(location = 2) out vec4 gWorldPosMetallic;// RGBA16F: worldPos.xyz + metallic

void main() {
  vec3 N = normalize(v_Normal);

  gAlbedoAO        = vec4(u_BaseColor, u_AO);
  // Encode world-space normal to [0,1] for RGBA16F storage
  gNormalRoughness = vec4(N * 0.5 + 0.5, u_Roughness);
  gWorldPosMetallic = vec4(v_WorldPos, u_Metallic);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Lighting pass — fullscreen deferred PBR shader
// ─────────────────────────────────────────────────────────────────────────────
const LIGHTING_FRAG = /* glsl */`#version 300 es
precision highp float;

// ── G-buffer inputs
uniform sampler2D u_gAlbedoAO;
uniform sampler2D u_gNormalRoughness;
uniform sampler2D u_gWorldPosMetallic;

// ── Directional light
uniform vec3  u_LightDir;     // normalised world-space direction TOWARD the light
uniform vec3  u_LightColor;   // pre-multiplied by intensity

// ── Shadow
uniform sampler2D u_ShadowMap;        // DEPTH_COMPONENT32F
uniform mat4      u_LightSpaceMatrix;
uniform float     u_ShadowBias;
uniform int       u_ShadowEnabled;

// ── IBL
uniform sampler2D u_IrradianceMap;  // equirectangular irradiance (or 1×1 fallback)
uniform sampler2D u_EnvMap;         // equirectangular prefiltered env (or 1×1 fallback)
uniform sampler2D u_BrdfLUT;        // 128×128 GGX split-sum LUT
uniform float     u_IBLIntensity;
uniform int       u_IBLEnabled;
// Hemisphere ambient fallback (used when IBL textures not supplied)
uniform vec3      u_AmbientSky;
uniform vec3      u_AmbientGround;

// ── Camera
uniform vec3 u_CameraPos;

out vec4 fragColor;

const float PI = 3.14159265358979;

// ── GGX Distribution (Trowbridge-Reitz)
float DistributionGGX(float NdotH, float roughness) {
  float a  = roughness * roughness;
  float a2 = a * a;
  float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d + 0.0001);
}

// ── Smith joint masking-shadowing (direct lighting k)
float GeometrySmithDirect(float NdotV, float NdotL, float roughness) {
  float r = roughness + 1.0;
  float k = (r * r) / 8.0;
  float gV = NdotV / (NdotV * (1.0 - k) + k);
  float gL = NdotL / (NdotL * (1.0 - k) + k);
  return gV * gL;
}

// ── Schlick Fresnel
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── Schlick Fresnel with roughness (for IBL)
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
  return F0 + (max(vec3(1.0 - roughness), F0) - F0) *
         pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── Equirectangular UV from direction
vec2 dirToEquirect(vec3 dir) {
  float phi   = atan(dir.z, dir.x);
  float theta = asin(clamp(dir.y, -1.0, 1.0));
  return vec2(phi / (2.0 * PI) + 0.5, theta / PI + 0.5);
}

// ── PCF shadow (3×3 kernel, manual depth comparison)
float ShadowPCF(sampler2D shadowMap, vec4 lightSpacePos, float bias) {
  vec3 proj = lightSpacePos.xyz / lightSpacePos.w;
  proj = proj * 0.5 + 0.5;
  if (proj.z >= 1.0) return 1.0;

  vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
  float shadow = 0.0;
  float refDepth = proj.z - bias;

  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      float closestDepth = texture(shadowMap, proj.xy + vec2(float(x), float(y)) * texelSize).r;
      shadow += (refDepth <= closestDepth) ? 1.0 : 0.0;
    }
  }
  return shadow / 9.0;
}

void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);

  // ── Read G-buffer
  vec4 albedoAO       = texelFetch(u_gAlbedoAO,        fc, 0);
  vec4 normalRough    = texelFetch(u_gNormalRoughness,  fc, 0);
  vec4 worldPosMetal  = texelFetch(u_gWorldPosMetallic, fc, 0);

  vec3  albedo    = albedoAO.rgb;
  float ao        = albedoAO.a;
  vec3  N         = normalize(normalRough.xyz * 2.0 - 1.0);
  float roughness = normalRough.w;
  vec3  worldPos  = worldPosMetal.xyz;
  float metallic  = worldPosMetal.w;

  vec3 V = normalize(u_CameraPos - worldPos);
  vec3 L = normalize(u_LightDir);
  vec3 H = normalize(V + L);

  float NdotV = max(dot(N, V), 0.0001);
  float NdotL = max(dot(N, L), 0.0);
  float NdotH = max(dot(N, H), 0.0);
  float VdotH = max(dot(V, H), 0.0);

  // ── Base reflectance F0 (metallic-roughness workflow)
  vec3 F0 = mix(vec3(0.04), albedo, metallic);

  // ── Direct lighting (GGX specular + Lambertian diffuse)
  vec3 directLight = vec3(0.0);
  if (NdotL > 0.0) {
    float D = DistributionGGX(NdotH, roughness);
    float G = GeometrySmithDirect(NdotV, NdotL, roughness);
    vec3  F = FresnelSchlick(VdotH, F0);

    vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    // Energy-conserving diffuse (dielectrics only; metals absorb diffuse)
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    directLight = (diffuse + specular) * u_LightColor * NdotL;
  }

  // ── PCF shadow
  float shadowFactor = 1.0;
  if (u_ShadowEnabled != 0) {
    vec4 lsPos = u_LightSpaceMatrix * vec4(worldPos, 1.0);
    shadowFactor = ShadowPCF(u_ShadowMap, lsPos, u_ShadowBias);
  }
  directLight *= shadowFactor;

  // ── IBL ambient
  vec3 ambient = vec3(0.0);
  vec3 F_ibl   = FresnelSchlickRoughness(NdotV, F0, roughness);

  if (u_IBLEnabled != 0) {
    // Diffuse irradiance from equirectangular map
    vec2 irradUV   = dirToEquirect(N);
    vec3 irradiance = texture(u_IrradianceMap, irradUV).rgb;
    vec3 kD_ibl     = (1.0 - F_ibl) * (1.0 - metallic);
    vec3 diffIBL    = kD_ibl * irradiance * albedo;

    // Specular via split-sum approximation
    vec3  R        = reflect(-V, N);
    float maxMip   = float(textureSize(u_EnvMap, 0).x > 1 ? 4 : 0);
    vec2  envUV    = dirToEquirect(R);
    // Approximate roughness mip lookup by biasing the UV (equirect doesn't have mips here)
    vec3  envColor = texture(u_EnvMap, envUV).rgb;
    vec2  brdf     = texture(u_BrdfLUT, vec2(NdotV, roughness)).rg;
    vec3  specIBL  = envColor * (F_ibl * brdf.x + brdf.y);

    ambient = (diffIBL + specIBL) * ao * u_IBLIntensity;
  } else {
    // Hemisphere ambient fallback
    float hemi     = N.y * 0.5 + 0.5;
    vec3  envColor  = mix(u_AmbientGround, u_AmbientSky, hemi);
    vec3  kD_hemi   = (1.0 - F_ibl) * (1.0 - metallic);
    vec3  diffHemi  = kD_hemi * albedo * envColor;

    // Specular contribution from BRDF LUT + hemisphere color
    vec2  brdf      = texture(u_BrdfLUT, vec2(NdotV, roughness)).rg;
    vec3  specHemi  = envColor * (F_ibl * brdf.x + brdf.y);

    ambient = (diffHemi + specHemi) * ao * u_IBLIntensity;
  }

  vec3 color = ambient + directLight;
  fragColor  = vec4(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// BRDF LUT generation shader (GGX importance sampling, 1024 samples)
// ─────────────────────────────────────────────────────────────────────────────
const BRDF_LUT_FRAG = /* glsl */`#version 300 es
precision highp float;

out vec4 fragColor;

const float PI           = 3.14159265358979;
const uint  SAMPLE_COUNT = 1024u;

float RadicalInverse_VdC(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10;
}

vec2 Hammersley(uint i, uint N) {
  return vec2(float(i) / float(N), RadicalInverse_VdC(i));
}

// GGX importance-sampling in tangent space
vec3 ImportanceSampleGGX(vec2 Xi, float roughness) {
  float a        = roughness * roughness;
  float phi      = 2.0 * PI * Xi.x;
  float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// Smith G₁ for IBL (k = α²/2)
float GeometrySchlickIBL(float NdotX, float roughness) {
  float a = roughness;
  float k = (a * a) / 2.0;
  return NdotX / (NdotX * (1.0 - k) + k + 0.0001);
}

float GeometrySmithIBL(float NdotV, float NdotL, float roughness) {
  return GeometrySchlickIBL(NdotV, roughness) *
         GeometrySchlickIBL(NdotL, roughness);
}

vec2 IntegrateBRDF(float NdotV, float roughness) {
  vec3  V = vec3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
  float A = 0.0;
  float B = 0.0;

  for (uint i = 0u; i < SAMPLE_COUNT; i++) {
    vec2 Xi    = Hammersley(i, SAMPLE_COUNT);
    vec3 H     = ImportanceSampleGGX(Xi, roughness);
    vec3 L     = normalize(2.0 * dot(V, H) * H - V);
    float NdotL = max(L.z, 0.0);
    float NdotH = max(H.z, 0.0);
    float VdotH = max(dot(V, H), 0.0);
    if (NdotL > 0.0) {
      float G     = GeometrySmithIBL(NdotV, NdotL, roughness);
      float G_Vis = (G * VdotH) / (NdotH * NdotV + 0.0001);
      float Fc    = pow(1.0 - VdotH, 5.0);
      A += (1.0 - Fc) * G_Vis;
      B += Fc          * G_Vis;
    }
  }
  return vec2(A, B) / float(SAMPLE_COUNT);
}

void main() {
  // gl_FragCoord pixel centre → NdotV and roughness in [0,1]
  // LUT is rendered at 128×128; gl_FragCoord ∈ [0.5, 127.5]
  float NdotV    = (gl_FragCoord.x) / float(128);
  float roughness = (gl_FragCoord.y) / float(128);
  roughness = max(roughness, 0.001);
  NdotV     = max(NdotV,     0.001);

  vec2 lut = IntegrateBRDF(NdotV, roughness);
  fragColor = vec4(lut, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// DeferredRenderer
// ─────────────────────────────────────────────────────────────────────────────
export class DeferredRenderer {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   * @param {{ width?: number, height?: number }} [opts]
   */
  constructor(glContext, stateCache, opts = {}) {
    this.ctx   = glContext;
    this.gl    = glContext.gl;
    this.cache = stateCache;

    const w = opts.width  ?? glContext.canvas.width  ?? 1;
    const h = opts.height ?? glContext.canvas.height ?? 1;

    // ── Require EXT_color_buffer_float for RGBA16F render targets
    const gl = this.gl;
    if (!gl.getExtension('EXT_color_buffer_float')) {
      console.warn('[DeferredRenderer] EXT_color_buffer_float unavailable — RGBA16F RT may not be supported.');
    }

    // ── G-buffer (3 MRT)
    this._gBuffer = new RenderTarget(gl, {
      width:  w, height: h,
      colorAttachments: ['RGBA8', 'RGBA16F', 'RGBA16F'],
      depthAttachment:  true,
    });

    // ── HDR lighting accumulation target
    this._hdrTarget = new RenderTarget(gl, {
      width: w, height: h,
      colorAttachments: ['RGBA16F'],
      depthAttachment: false,
    });

    // ── Post-process stack (ACES tone-map)
    this._postStack = new PostProcessStack(glContext, stateCache, { width: w, height: h });

    // ── Shaders
    this._geoShader        = new Shader(gl, GBUF_VERT,         GBUF_FRAG);
    this._geoSkinnedShader = new Shader(gl, GBUF_VERT_SKINNED, GBUF_FRAG);
    this._lightShader      = new Shader(gl, FULLSCREEN_VERT,   LIGHTING_FRAG);

    // ── Generate BRDF LUT (128×128, one-time render)
    this._brdfLUT = this._generateBrdfLUT();

    // ── Default 1×1 IBL textures (plain white irradiance / env)
    const { irr, env } = this._createDefaultIBL();
    this._irradianceTex = irr;
    this._envTex        = env;
    this._iblEnabled    = false;   // use hemisphere fallback by default

    // ── Exposure and IBL intensity
    this._exposure    = 1.0;
    this._iblIntensity = 0.5;

    // ── Pre-allocated matrices
    this._normalMat = new Mat3();
  }

  // ─────────────────────────────────────────── public API

  /**
   * Render scene using the deferred pipeline.
   * Drop-in compatible with ForwardRenderer.render() signature.
   *
   * @param {import('../scene/Node.js').Node|Array} scene — root node or flat array of nodes
   * @param {import('../scene/Camera.js').Camera} camera
   * @param {import('../scene/Light.js').Light[]} [lights]
   * @param {import('../animation/GPUSkinning.js').GPUSkinning|null} [gpuSkinning]
   * @param {import('./ShadowMap.js').ShadowMap|null} [shadowMap]
   */
  render(scene, camera, lights = [], gpuSkinning = null, shadowMap = null) {
    camera.updateProjection();
    camera.updateView();

    const nodes = Array.isArray(scene) ? scene : this._collectNodes(scene);
    const dirLight = lights.find(l => l.type === 'directional') ?? null;

    // 1 ── Shadow pass (handled externally by caller; shadowMap.render() called before this)

    // 2 ── Geometry pass → fill G-buffer
    this._geometryPass(nodes, camera, gpuSkinning);

    // 3 ── Lighting pass → HDR target
    this._lightingPass(camera, dirLight, shadowMap);

    // 4 ── Tone-map pass → canvas
    const gl = this.gl;
    this._postStack.render(
      this._hdrTarget.getColorTexture(0),
      this._gBuffer.width,
      this._gBuffer.height
    );
  }

  /** @param {number} value */
  setExposure(value) {
    this._exposure = Math.max(0.001, value);
    this._postStack.setExposure(value);
  }

  /**
   * Supply pre-baked IBL textures (equirectangular format).
   * Pass null to revert to the hemisphere ambient fallback.
   * @param {WebGLTexture|null} irradianceTex — diffuse irradiance map
   * @param {WebGLTexture|null} envTex        — prefiltered specular env map
   */
  setEnvironment(irradianceTex, envTex) {
    if (irradianceTex) { this._irradianceTex = irradianceTex; this._iblEnabled = true; }
    if (envTex)        { this._envTex        = envTex;         this._iblEnabled = true; }
    if (!irradianceTex && !envTex) {
      // Revert to fallback
      const def = this._createDefaultIBL();
      this._irradianceTex = def.irr;
      this._envTex        = def.env;
      this._iblEnabled    = false;
    }
  }

  /** @param {number} v */
  setIBLIntensity(v) { this._iblIntensity = Math.max(0, v); }

  /** Handle canvas resize. */
  resize(w, h) {
    const cache = this.cache;
    this._gBuffer.resize(w, h);
    this._hdrTarget.resize(w, h);
    this._postStack.resize(w, h);
    cache.setViewport(0, 0, w, h);
  }

  destroy() {
    this._gBuffer?.destroy();
    this._hdrTarget?.destroy();
    this._postStack?.destroy();
    this._geoShader?.destroy();
    this._geoSkinnedShader?.destroy();
    this._lightShader?.destroy();
    const gl = this.gl;
    if (this._brdfLUT)      gl.deleteTexture(this._brdfLUT);
    if (this._irradianceTex) gl.deleteTexture(this._irradianceTex);
    if (this._envTex)        gl.deleteTexture(this._envTex);
    this._gBuffer = this._hdrTarget = this._postStack = null;
    this._geoShader = this._geoSkinnedShader = this._lightShader = null;
    this._brdfLUT = this._irradianceTex = this._envTex = null;
  }

  // ─────────────────────────────────────────── private passes

  /** @private — geometry pass: fill G-buffer MRT */
  _geometryPass(nodes, camera, gpuSkinning) {
    const gl    = this.gl;
    const cache = this.cache;

    this._gBuffer.bind();
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    cache.setDepthTest(true);
    cache.setDepthWrite(true);
    cache.setCullFace(true, gl.BACK);
    cache.setBlend(false);

    const staticNodes  = [];
    const skinnedNodes = [];
    for (const node of nodes) {
      if (!node.mesh) continue;
      if (node.mesh.skinned && gpuSkinning) skinnedNodes.push(node);
      else staticNodes.push(node);
    }

    // Static geometry
    if (staticNodes.length) {
      cache.useProgram(this._geoShader.program);
      this._geoShader.setMat4('u_ViewMatrix',       camera.viewMatrix.e);
      this._geoShader.setMat4('u_ProjectionMatrix', camera.projectionMatrix.e);
      for (const node of staticNodes) {
        this._drawNodeGeo(node, this._geoShader, false);
      }
    }

    // Skinned geometry
    if (skinnedNodes.length && gpuSkinning) {
      cache.useProgram(this._geoSkinnedShader.program);
      this._geoSkinnedShader.setMat4('u_ViewMatrix',       camera.viewMatrix.e);
      this._geoSkinnedShader.setMat4('u_ProjectionMatrix', camera.projectionMatrix.e);
      gpuSkinning.bind(this._geoSkinnedShader, 0);
      for (const node of skinnedNodes) {
        this._drawNodeGeo(node, this._geoSkinnedShader, true);
      }
    }

    cache.bindVAO(null);
    this._gBuffer.unbind();
  }

  /** @private — per-node geometry draw */
  _drawNodeGeo(node, shader, skinned) {
    const mesh  = node.mesh;
    const gl    = this.gl;
    const cache = this.cache;

    node.updateWorldMatrix(node.parent ? node.parent.worldMatrix : null);
    shader.setMat4('u_ModelMatrix', node.worldMatrix.e);

    if (!skinned) {
      Mat3.normalMatrix(node.worldMatrix, this._normalMat);
      shader.setMat3('u_NormalMatrix', this._normalMat.e);
    }

    const mat = mesh.material ?? {};
    const bc  = mat.baseColorFactor ?? [0.8, 0.6, 0.4, 1.0];
    shader.setVec3('u_BaseColor',  bc[0], bc[1], bc[2]);
    shader.setFloat('u_Roughness', mat.roughnessFactor ?? 0.5);
    shader.setFloat('u_Metallic',  mat.metallicFactor  ?? 0.0);
    shader.setFloat('u_AO',        mat.aoFactor        ?? 1.0);

    cache.bindVAO(mesh.vao);
    if (mesh.indexBuffer) {
      gl.drawElements(gl.TRIANGLES, mesh.indexCount, mesh.indexType, 0);
    } else {
      gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
    }
  }

  /** @private — fullscreen lighting pass → HDR target */
  _lightingPass(camera, dirLight, shadowMap) {
    const gl    = this.gl;
    const cache = this.cache;

    this._hdrTarget.bind();
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    cache.setDepthTest(false);
    cache.setDepthWrite(false);
    cache.setCullFace(false, gl.BACK);
    cache.setBlend(false);

    cache.useProgram(this._lightShader.program);

    // ── G-buffer textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._gBuffer.getColorTexture(0));
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this._gBuffer.getColorTexture(1));
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this._gBuffer.getColorTexture(2));

    this._lightShader.setInt('u_gAlbedoAO',        0);
    this._lightShader.setInt('u_gNormalRoughness',  1);
    this._lightShader.setInt('u_gWorldPosMetallic', 2);

    // ── IBL textures
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this._irradianceTex);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this._envTex);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this._brdfLUT);

    this._lightShader.setInt('u_IrradianceMap', 3);
    this._lightShader.setInt('u_EnvMap',        4);
    this._lightShader.setInt('u_BrdfLUT',       5);
    this._lightShader.setFloat('u_IBLIntensity', this._iblIntensity);
    this._lightShader.setInt('u_IBLEnabled', this._iblEnabled ? 1 : 0);

    // Hemisphere ambient fallback colours
    this._lightShader.setVec3('u_AmbientSky',    0.35, 0.45, 0.6);
    this._lightShader.setVec3('u_AmbientGround', 0.15, 0.12, 0.1);

    // ── Directional light
    const lightDir = dirLight ? dirLight.direction.e : [0.5, -1.0, 0.5];
    const ld = lightDir;
    const len = Math.sqrt(ld[0]*ld[0]+ld[1]*ld[1]+ld[2]*ld[2]) || 1;
    // LightDir in shader = direction *toward* light source
    this._lightShader.setVec3('u_LightDir',
      -ld[0]/len, -ld[1]/len, -ld[2]/len);

    const lc = dirLight ? dirLight.getColorArray() : [1, 1, 1];
    this._lightShader.setVec3('u_LightColor', lc[0], lc[1], lc[2]);

    // ── Camera
    const cp = camera.position.e;
    this._lightShader.setVec3('u_CameraPos', cp[0], cp[1], cp[2]);

    // ── Shadows
    const shadowEnabled = (shadowMap && shadowMap.getTexture()) ? 1 : 0;
    this._lightShader.setInt('u_ShadowEnabled', shadowEnabled);
    if (shadowEnabled) {
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, shadowMap.getTexture());
      this._lightShader.setInt('u_ShadowMap', 6);
      this._lightShader.setMat4('u_LightSpaceMatrix', shadowMap.getLightSpaceMatrix().e);
      this._lightShader.setFloat('u_ShadowBias', shadowMap.bias ?? 0.002);
    } else {
      // Bind a dummy texture so the sampler is not unbound
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, this._brdfLUT); // harmless reuse
      this._lightShader.setInt('u_ShadowMap', 6);
      this._lightShader.setFloat('u_ShadowBias', 0.002);
    }

    // Draw fullscreen triangle (no VAO, vertex IDs only)
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    this._hdrTarget.unbind();
  }

  // ─────────────────────────────────────────── startup helpers

  /**
   * One-time GGX BRDF LUT generation.
   * Renders to a 128×128 RGBA16F texture using the BRDF_LUT_FRAG shader.
   * @private
   * @returns {WebGLTexture}
   */
  _generateBrdfLUT() {
    const gl   = this.gl;
    const SIZE = 128;

    const lut_shader = new Shader(gl, FULLSCREEN_VERT, BRDF_LUT_FRAG);

    // Allocate RGBA16F LUT texture
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, SIZE, SIZE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    gl.viewport(0, 0, SIZE, SIZE);
    gl.useProgram(lut_shader.program);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    lut_shader.destroy();

    return tex;
  }

  /**
   * Create 1×1 default IBL textures (white irradiance, white env).
   * @private
   * @returns {{ irr: WebGLTexture, env: WebGLTexture }}
   */
  _createDefaultIBL() {
    const gl = this.gl;

    const make1x1 = (r, g, b) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      // Use RGBA8 (1×1 pixel)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array([Math.round(r*255), Math.round(g*255), Math.round(b*255), 255]));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return t;
    };

    return {
      irr: make1x1(1, 1, 1),
      env: make1x1(1, 1, 1),
    };
  }

  /** @private */
  _collectNodes(root) {
    const result = [];
    const stack  = [root];
    while (stack.length) {
      const node = stack.pop();
      result.push(node);
      if (node.children) {
        for (let i = node.children.length - 1; i >= 0; i--) {
          stack.push(node.children[i]);
        }
      }
    }
    return result;
  }
}
