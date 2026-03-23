Build a **pure WebGL 2.0 digital human render engine** (zero dependencies — no Three.js, no Babylon.js, no external math libraries) capable of rendering photorealistic human characters in real-time on the browser, with full animation control and streaming animation pipeline.

This engine will serve as the core rendering layer for **OpenHuman**, an open store for realistic digital humans. It must be embeddable as an SDK into third-party websites and apps.

---

## 🏗️ Architecture Overview

```
OpenHuman Engine
├── Core (WebGL 2.0 context management)
├── Renderer
│   ├── ForwardRenderer (opaque pass)
│   ├── DeferredRenderer (lighting pass)
│   └── PostProcessPass (tonemap, bloom, SSS, DOF)
├── ShaderLibrary
│   ├── SkinShader (subsurface scattering)
│   ├── HairShader (Kajiya-Kay / Marschner)
│   ├── EyeShader (cornea, iris, sclera)
│   └── ClothShader (anisotropic)
├── AssetPipeline
│   ├── GLTFLoader (glTF 2.0 + extensions)
│   ├── TextureManager (KTX2 / basis compression)
│   └── MeshProcessor (morph targets, skinning)
├── AnimationSystem
│   ├── SkeletalAnimation (GPU skinning)
│   ├── MorphTargetController (facial blendshapes)
│   ├── AnimationGraph (state machine)
│   └── StreamingAnimationPlayer (WebSocket / fetch streaming)
├── SceneGraph
│   ├── Node / Transform
│   ├── Camera (perspective, frustum culling)
│   └── Light (directional, point, IBL probe)
└── SDK
    ├── EmbedAPI (iframe / web component)
    ├── EventBus
    └── StreamingClient
```

---

## 📋 Technical Requirements

### Platform Target

- **Primary**: WebGL 2.0 (Chrome 60+, Firefox 55+, Edge 79+, Safari 15+)
- **Fallback**: WebGL 1.0 với reduced quality mode
- **Mobile**: Chrome Android (WebGL 2.0), Safari iOS 15+ (WebGL 2.0)
- **No native dependencies**: chạy thuần trong browser tab, không cần plugin

### Performance Budget (60 fps target)

- Frame budget: **16.6ms** tổng
  - CPU (JS): ≤ 4ms (scene update, animation sampling, draw call prep)
  - GPU: ≤ 12ms (geometry + lighting + post-process)
- Draw calls: ≤ 50 per frame cho 1 character
- Triangle budget: 50k–150k tris cho LOD0 character
- Texture memory: ≤ 256MB per character set
- JS heap: ≤ 64MB engine core

### Rendering Quality Targets

- **Skin**: Physically-based subsurface scattering (screen-space SSS)
- **Hair**: Multi-layer strand rendering hoặc card-based với anisotropic highlight
- **Eyes**: Cornea refraction, iris parallax depth, sclera scattering
- **Lighting**: Image-based lighting (IBL) + 1 directional key light + fill
- **Resolution**: Native up to 4K, dynamic resolution scaling cho mobile

---

## 🔧 Core Systems — Implementation Detail

### 1. WebGL Context Setup

```javascript
// Yêu cầu implement: Context Manager
class GLContext {
  constructor(canvas) {
    // Khởi tạo WebGL2 với các extension cần thiết
    // Extensions bắt buộc:
    // - EXT_color_buffer_float (HDR framebuffer)
    // - OES_texture_float_linear (linear filtering trên float texture)
    // - WEBGL_compressed_texture_etc, _s3tc, _bptc (texture compression theo platform)
    // - EXT_texture_filter_anisotropic (anisotropic filtering)
    // - OES_vertex_array_object (VAO - native trong WebGL2)
    // State cache để tránh redundant GL calls
    // Double-buffered command queue
  }
}
```

### 2. Shader System

```glsl
-- Yêu cầu implement: GLSL 300 es --

// Skin Shader — Subsurface Scattering (Screen-Space)
// Technique: Separable SSS (Jorge Jimenez 2015)
// 1. Geometry pass: output albedo, normal, depth, curvature vào G-Buffer
// 2. SSS pass: Gaussian blur theo curvature map + depth
// 3. Combine pass: SSS result + specular + IBL

// G-Buffer layout:
// RT0 (RGBA8): albedo.rgb, cavity
// RT1 (RGBA16F): world normal.xyz, roughness
// RT2 (RGBA16F): motion vector.xy, metallic, ao
// RT3 (R32F): linear depth

// Skin BRDF:
// Specular: GGX với dual-lobe (primary + secondary)
// Diffuse: Modified Burley với SSS weight
// Backscatter: thin-slab approximation

// Key shader uniforms:
// u_SSSStrength: float (0.0 = no SSS, 1.0 = full)
// u_SSSWidth: vec3 (RGB scattering widths — red scatters furthest)
// u_SSSColor: vec3 (scattering tint — warm red/orange for skin)
// u_Roughness: float (GGX roughness)
// u_Specular: float (F0 — 0.028 for skin)
```

### 3. GPU Skinning

```glsl
// Implement: Dual Quaternion Skinning (DQS)
// Tốt hơn Linear Blend Skinning cho joint artifacts
//
// Vertex shader:
// Input: joint indices (ivec4), weights (vec4), up to 4 influences
// Uniform: mat2x4[] jointDualQuats — packed dual quaternion array (texture buffer)
//
// Pipeline:
// 1. Upload joint matrices mỗi frame vào Texture Buffer Object
// 2. Sample trong vertex shader bằng texelFetch()
// 3. Blend dual quaternions, convert sang matrix
// 4. Transform position, normal, tangent

// Bone count limit: 256 joints (standard for humanoid)
// Joint data format: RGBA32F texture, 2 texels per joint (dual quat)
```

### 4. Morph Target System (Facial Blendshapes)

```javascript
// Yêu cầu implement: GPU Morph Target Accumulator
//
// Approach: Texture-based morph targets (scalable đến 500+ targets)
// Không dùng vertex attribute per morph (limited by attribute slots)
//
// Data layout:
// - Base mesh: VBO bình thường
// - Morph deltas: packed vào R16F / RGB16F texture
//   Width = vertex count, Height = morph target count
//   Channels: delta position (vec3), delta normal (vec3)
//
// Compute pass (Transform Feedback hoặc manual accumulation):
//   finalPos = basePos + Σ(weight[i] * delta[i])
//
// Facial Action Coding System (FACS) targets phải có (52 targets tối thiểu):
// - browDownLeft, browDownRight, browInnerUp
// - browOuterUpLeft, browOuterUpRight
// - cheekPuff, cheekSquintLeft, cheekSquintRight
// - eyeBlinkLeft, eyeBlinkRight
// - eyeLookDownLeft, eyeLookDownRight, eyeLookInLeft, eyeLookInRight
// - eyeLookOutLeft, eyeLookOutRight, eyeLookUpLeft, eyeLookUpRight
// - eyeSquintLeft, eyeSquintRight, eyeWideLeft, eyeWideRight
// - jawForward, jawLeft, jawOpen, jawRight
// - mouthClose, mouthDimpleLeft, mouthDimpleRight
// - mouthFrownLeft, mouthFrownRight
// - mouthFunnel, mouthLeft, mouthLowerDownLeft, mouthLowerDownRight
// - mouthPressLeft, mouthPressRight, mouthPucker, mouthRight
// - mouthRollLower, mouthRollUpper, mouthShrugLower, mouthShrugUpper
// - mouthSmileLeft, mouthSmileRight
// - mouthStretchLeft, mouthStretchRight
// - mouthUpperUpLeft, mouthUpperUpRight
// - noseSneerLeft, noseSneerRight
// - tongueOut
```

### 5. Animation System

```javascript
// AnimationClip: chuỗi keyframe data
// Format hỗ trợ: glTF animation tracks
// Interpolation: LINEAR, STEP, CUBICSPLINE (Hermite)
//
// AnimationGraph (State Machine):
// - States: idle, talk, gesture, blink, look
// - Transitions: crossfade với blending duration
// - Layers: upper body / lower body / facial (additive blending)
//
// Sampling pipeline:
// 1. Sample joint transforms tại thời điểm t
// 2. Blend giữa 2 clips (normalized lerp cho quaternion rotation)
// 3. Upload kết quả vào GPU skinning texture

class AnimationGraph {
  // addState(name, clip)
  // addTransition(from, to, condition, duration)
  // setFloat(param, value)  // blend tree param
  // setBool(param, value)   // trigger transition
  // update(deltaTime)
  // getCurrentPose() → JointTransform[]
}
```

### 6. Streaming Animation Pipeline

```javascript
// Streaming Animation Player — realtime animation từ server
// Use case: lip sync với AI TTS, mocap stream, remote control
//
// Protocol options (implement đủ cả 2):
//
// Option A — WebSocket binary stream:
//   Frame packet: [timestamp: f32][jointCount: u16][joint data: f32 * 8 * count]
//   Joint data: position(3) + quaternion(4) + scale(1) = 8 floats
//   Compression: quantized 16-bit fixed point (giảm 50% bandwidth)
//   Latency target: < 50ms end-to-end
//
// Option B — HTTP chunked / fetch streaming:
//   Dùng ReadableStream API
//   Format: newline-delimited JSON hoặc binary frames
//   Phù hợp cho pre-recorded streaming (không cần bidirectional)
//
// Jitter buffer:
//   Buffer 3–5 frames để smooth out network jitter
//   Extrapolation nếu frame đến muộn (linear predict từ velocity)
//
// Facial streaming (lip sync):
//   Server gửi FACS weight array [52 floats] mỗi frame
//   Smoothing: low-pass filter (exponential moving average α=0.7)

class StreamingAnimationPlayer {
  // connectWebSocket(url)
  // connectHTTPStream(url)
  // onFrame(callback)  // callback nhận JointTransform[] + FACSWeights[]
  // getInterpolatedPose(currentTime) // jitter buffer output
}
```

---

## 📦 Asset Format & Pipeline

### Character Asset Bundle (.ohb — OpenHuman Bundle)

```
character.ohb (ZIP container)
├── manifest.json          # metadata, LOD info, feature flags
├── mesh/
│   ├── lod0.glb           # 150k tris — full quality
│   ├── lod1.glb           # 80k tris
│   ├── lod2.glb           # 30k tris — mobile
│   └── morphs.bin         # packed morph delta texture
├── textures/
│   ├── albedo.ktx2        # BC7 / ETC2 compressed
│   ├── normal.ktx2
│   ├── roughness.ktx2
│   ├── sss_mask.ktx2      # SSS strength map
│   ├── thickness.ktx2     # SSS thickness (backscatter)
│   ├── hair_flow.ktx2     # anisotropy direction
│   └── eye_iris.ktx2      # high-res iris detail
├── rig/
│   ├── skeleton.json      # joint hierarchy
│   └── bindpose.bin       # inverse bind matrices
└── animations/
    ├── idle.glb
    ├── talk_neutral.glb
    └── index.json         # clip manifest
```

### glTF Extensions cần support

- `KHR_draco_mesh_compression` — geometry compression
- `KHR_texture_basisu` — KTX2 texture compression
- `KHR_materials_unlit` — fallback cho low-end
- `EXT_mesh_gpu_instancing` — multiple characters
- `KHR_animation_pointer` — morph target animation tracks

---

## 🎛️ Animation Control API (Public SDK)

```javascript
// SDK interface mà developer bên ngoài sẽ dùng
const human = await OpenHuman.load("character.ohb", canvas)

// Playback control
human.animation.play("idle")
human.animation.crossFadeTo("talk", 0.3) // 300ms crossfade
human.animation.setLayer("facial", facialClip, { additive: true })

// Blend tree params
human.animation.setFloat("emotionHappy", 0.8)
human.animation.setFloat("emotionSad", 0.0)
human.animation.setFloat("lookX", 0.3) // -1 left, +1 right
human.animation.setFloat("lookY", -0.1) // -1 down, +1 up

// Direct morph control
human.morph.set("mouthSmileLeft", 0.6)
human.morph.set("mouthSmileRight", 0.6)
human.morph.set("eyeBlinkLeft", 1.0)
human.morph.setMany({ jawOpen: 0.3, mouthFunnel: 0.2 })

// Streaming
human.streaming.connect("wss://your-server/stream")
human.streaming.onData((pose) => {
  // pose: { joints: Float32Array, facs: Float32Array }
})

// Events
human.on("animationEnd", (clipName) => {})
human.on("streamConnected", () => {})
human.on("streamDisconnected", () => {})

// Rendering
human.renderer.setExposure(1.2)
human.renderer.setEnvironment("studio_warm") // IBL preset
human.renderer.setDOF({ focalDistance: 1.5, aperture: 0.05 })
human.renderer.setSSSStrength(0.85)

// Camera
human.camera.setPosition([0, 1.65, 0.8])
human.camera.lookAt([0, 1.55, 0])
human.camera.setFOV(35) // portrait FOV
human.camera.enableOrbit(true) // mouse/touch orbit

// Dispose
human.destroy()
```

---

## 🖥️ Render Pipeline — Step by Step

```
Frame N:
│
├── 1. INPUT UPDATE (JS — 0.5ms)
│   ├── Process input events
│   ├── Update streaming buffer (pop jitter queue)
│   └── Camera orbit/pan delta
│
├── 2. ANIMATION UPDATE (JS — 1ms)
│   ├── Sample AnimationGraph → joint local transforms
│   ├── Forward kinematics → world transforms
│   ├── Accumulate morph weights (blend tree + streaming FACS)
│   └── Upload joint dual quats → TBO texture (gl.texSubImage2D)
│
├── 3. MORPH PASS (GPU — 1ms)
│   ├── Compute shader (hoặc vertex shader trick):
│   │   finalDelta = Σ weight[i] * morphDelta[i]
│   └── Output: accumulated position/normal delta texture
│
├── 4. GEOMETRY PASS — Deferred (GPU — 2ms)
│   ├── Bind G-Buffer FBO (4 MRT)
│   ├── Draw character:
│   │   └── VS: skinning (TBO sample) + morph apply + project
│   │   └── FS: albedo, normal unpack, PBR params → G-Buffer
│   └── Draw hair, eyes, cloth (separate shaders, same G-Buffer)
│
├── 5. SHADOW PASS (GPU — 1ms)
│   ├── Shadow map 2048×2048 (directional key light)
│   └── PCF soft shadows
│
├── 6. LIGHTING PASS (GPU — 2ms)
│   ├── Screen-quad shader reads G-Buffer
│   ├── IBL: diffuse irradiance (SH9) + specular (prefiltered cubemap + BRDF LUT)
│   ├── Directional light + shadow
│   └── Output: HDR color buffer
│
├── 7. SSS PASS (GPU — 1.5ms)
│   ├── Horizontal Gaussian blur (skin-masked, depth-aware)
│   ├── Vertical Gaussian blur
│   └── Composite SSS onto lit result
│
├── 8. POST PROCESS (GPU — 1ms)
│   ├── Bloom (dual-kawase blur)
│   ├── Depth of Field (CoC + bokeh blur)
│   ├── Tone mapping (ACES filmic)
│   ├── Color grading (LUT 3D)
│   └── FXAA anti-aliasing
│
└── 9. PRESENT
    └── Blit to canvas (drawArrays fullscreen quad)
```

---

## 📁 File Structure (Codebase)

```
openhuman-engine/
├── src/
│   ├── core/
│   │   ├── GLContext.js         # WebGL2 context + extension management
│   │   ├── StateCache.js        # GL state cache (avoid redundant calls)
│   │   ├── RenderTarget.js      # FBO / MRT wrapper
│   │   ├── Buffer.js            # VBO, IBO, UBO, TBO wrappers
│   │   └── Shader.js            # GLSL compile, link, uniform cache
│   ├── math/
│   │   ├── Vec2.js / Vec3.js / Vec4.js
│   │   ├── Mat3.js / Mat4.js
│   │   ├── Quat.js              # quaternion ops
│   │   └── DualQuat.js          # dual quaternion skinning
│   ├── renderer/
│   │   ├── ForwardRenderer.js
│   │   ├── DeferredRenderer.js
│   │   ├── PostProcessStack.js
│   │   └── ShadowMap.js
│   ├── shaders/
│   │   ├── skin.vert.glsl
│   │   ├── skin.frag.glsl
│   │   ├── hair.vert.glsl
│   │   ├── hair.frag.glsl
│   │   ├── eye.frag.glsl
│   │   ├── lighting.frag.glsl   # deferred lighting pass
│   │   ├── sss.frag.glsl        # SSS blur passes
│   │   ├── bloom.frag.glsl
│   │   ├── dof.frag.glsl
│   │   └── tonemap.frag.glsl
│   ├── animation/
│   │   ├── Skeleton.js          # joint hierarchy
│   │   ├── AnimationClip.js     # keyframe data + sampling
│   │   ├── AnimationGraph.js    # state machine
│   │   ├── MorphController.js   # blendshape accumulation
│   │   ├── GPUSkinning.js       # TBO upload + shader binding
│   │   └── StreamingPlayer.js   # WebSocket + HTTP stream
│   ├── asset/
│   │   ├── GLTFLoader.js        # glTF 2.0 parser
│   │   ├── KTX2Loader.js        # KTX2 / Basis Universal
│   │   ├── BundleLoader.js      # .ohb bundle reader
│   │   └── TextureManager.js    # GPU upload + cache
│   ├── scene/
│   │   ├── Node.js              # transform hierarchy
│   │   ├── Camera.js            # perspective + orbit control
│   │   ├── Light.js             # directional + IBL
│   │   └── Character.js         # assembled human entity
│   └── sdk/
│       ├── OpenHuman.js         # public API entry point
│       ├── EventBus.js
│       └── StreamingClient.js
├── shaders/                     # raw GLSL files (bundled at build)
├── tools/
│   ├── asset-builder/           # CLI: build .ohb từ source assets
│   └── morph-baker/             # bake morph deltas vào texture
├── demo/
│   ├── index.html               # standalone demo
│   └── streaming-demo.html      # streaming demo với mock server
├── tests/
└── package.json                 # build: esbuild / rollup, no runtime deps
```

---

## 🚀 Implementation Order (Sprint Plan)

### Sprint 1 — Tuần 1–2: Foundation

- [ ] GLContext, StateCache, Buffer, RenderTarget wrappers
- [ ] Math library: Vec3, Vec4, Mat4, Quat, DualQuat
- [ ] Shader compile/link system với uniform caching
- [ ] Load và hiển thị 1 static mesh (glTF) với basic Phong shading
- [ ] **Milestone**: Có thể thấy 1 mesh trên màn hình

### Sprint 2 — Tuần 3–4: Skinning & Animation

- [ ] GLTFLoader: skeleton, animation tracks
- [ ] GPU skinning với Dual Quaternion (TBO approach)
- [ ] AnimationClip sampling (linear + cubic interpolation)
- [ ] Basic AnimationGraph (idle → talk state)
- [ ] **Milestone**: Character đi/đứng với skeletal animation

### Sprint 3 — Tuần 5–6: PBR & Deferred Rendering

- [ ] G-Buffer setup (4 MRT)
- [ ] PBR skin shader (GGX specular, modified diffuse)
- [ ] IBL: load HDRI → prefilter cubemap → BRDF LUT (offline hoặc runtime)
- [ ] Directional shadow map + PCF
- [ ] Tone mapping (ACES)
- [ ] **Milestone**: Character trông "real" dưới studio lighting

### Sprint 4 — Tuần 7–8: SSS + Eyes + Hair

- [ ] Separable SSS blur (Jorge Jimenez technique)
- [ ] Eye shader: cornea refraction, iris depth, sclera SSS
- [ ] Hair shader: Kajiya-Kay hoặc card-based với anisotropy
- [ ] **Milestone**: Skin, eyes, hair đều convincing

### Sprint 5 — Tuần 9–10: Morph Targets + Facial

- [ ] Texture-based morph target system (GPU accumulation)
- [ ] 52 FACS targets loaded và blend-able
- [ ] MorphController API
- [ ] **Milestone**: Facial animation đầy đủ biểu cảm

### Sprint 6 — Tuần 11–12: Streaming + SDK

- [ ] StreamingPlayer: WebSocket binary protocol
- [ ] Jitter buffer + interpolation
- [ ] Post-process stack: Bloom, DoF, Color grading
- [ ] Public SDK API (OpenHuman.js)
- [ ] Demo page + embed snippet
- [ ] **Milestone**: Demo hoàn chỉnh, embeddable, streaming được

---

## 🛑 Constraints & Non-Goals (MVP)

**Không làm trong MVP:**

- Ray tracing / path tracing
- Cloth simulation (physics)
- Multi-character scene > 2 nhân vật
- Shadow từ nhiều light source
- LOD streaming (chỉ LOD0 cho desktop MVP)
- Editor / authoring tool
- Backend / store / marketplace
- Android OpenGL ES native (dùng WebGL trên Chrome Android)

**Giới hạn chấp nhận được:**

- Hair: card-based (không phải strand simulation)
- SSS: screen-space (không phải volumetric)
- IK: không có trong MVP (animation clip only)
- Physics hair: không có trong MVP

---

## 📐 Coding Standards

```javascript
// Không dùng bất kỳ runtime dependency nào
// Math: tự implement, không dùng gl-matrix
// Loader: tự parse glTF JSON, không dùng thư viện
// Build tool: esbuild (dev dependency only — không bundle vào output)

// Performance rules:
// - Không allocate object trong hot path (render loop)
// - Pool Vec3/Mat4 objects, tái sử dụng bằng out params
// - Tất cả GL state thay đổi phải đi qua StateCache
// - Không gọi gl.getError() trong production build

// GLSL rules:
// - GLSL 300 es (WebGL 2.0)
// - Precision: mediump cho color, highp cho position/transform
// - Tránh dynamic branching trong fragment shader hot path
// - Dùng preprocessor #define cho shader variants (thay vì if/else)
```

---

## 🔬 Reference Papers & Techniques

- **Skin SSS**: "Separable Subsurface Scattering" — Jorge Jimenez et al. (2015)
- **Dual Quaternion Skinning**: Kavan et al. (2007) — "Skinning with Dual Quaternions"
- **GGX BRDF**: Walter et al. (2007) — "Microfacet Models for Refraction"
- **IBL**: "Real Shading in Unreal Engine 4" — Brian Karis (2013)
- **Hair**: "Light Scattering from Human Hair Fibers" — Marschner et al. (2003)
- **Eye**: "A Practical Model for Subsurface Light Transport" — Jensen et al.
- **ACES Tonemap**: Academy Color Encoding System filmic curve
- **Bloom**: "Dual Kawase Blur" — AMD (2015)

---

## ✅ Definition of Done (MVP Complete)

- [ ] 1 realistic digital human character render được trên Chrome desktop với 60fps
- [ ] Skeletal animation + morph targets hoạt động
- [ ] Skin SSS, eye shader, hair shader đều active
- [ ] WebSocket streaming animation nhận và apply được real-time
- [ ] JavaScript SDK có thể embed vào 1 iframe / web component
- [ ] Chạy được trên Chrome Android ở 30fps+ (WebGL 2.0)
- [ ] Bundle size SDK ≤ 200KB gzipped (không kể assets)
- [ ] Zero runtime dependencies
