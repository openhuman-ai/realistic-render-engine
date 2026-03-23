# OpenHuman Realistic Render Engine

A pure **WebGL 2.0** digital human render engine with **zero runtime dependencies**.

## Quick Start

```bash
npm install
npm run dev
# Open http://localhost:8080/demo/index.html
```

## Development

```bash
npm run dev
# Main demo: http://localhost:8080/demo/index.html
# Streaming demo: http://localhost:8080/demo/streaming.html
```

## Production Build

```bash
npm run build
npm run preview
```

## Architecture

- `src/core/` — WebGL2 context, state cache, buffer/shader/render-target wrappers
- `src/math/` — Vec2/3/4, Mat3/4, Quat, DualQuat (no external math lib)
- `src/renderer/` — ForwardRenderer (functional), Deferred/PostProcess/ShadowMap (scaffolded)
- `src/asset/` — glTF 2.0 loader
- `src/scene/` — Node, Camera, Light, Character
- `src/sdk/` — Public OpenHuman API

## Demo

Open `/demo/index.html` in a browser (served from `public/demo`).

The demos now run through Vite with vanilla JavaScript entry modules:

- `public/demo/index.html` loads `demo/index.js`
- `public/demo/streaming.html` loads `demo/streaming.js`

## Constraints

- Zero runtime dependencies
- WebGL 2.0 primary target
- Performance-first: no allocations in hot loops, all GL state via StateCache
