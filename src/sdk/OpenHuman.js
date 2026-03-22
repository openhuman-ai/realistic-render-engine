/**
 * OpenHuman SDK — Public API
 * Zero runtime dependencies
 */

import { GLContext }           from '../core/GLContext.js';
import { StateCache }          from '../core/StateCache.js';
import { VertexBuffer, IndexBuffer } from '../core/Buffer.js';
import { ForwardRenderer }     from '../renderer/ForwardRenderer.js';
import { DeferredRenderer }    from '../renderer/DeferredRenderer.js';
import { ShadowMap }           from '../renderer/ShadowMap.js';
import { GLTFLoader }          from '../asset/GLTFLoader.js';
import { Camera }              from '../scene/Camera.js';
import { Light }               from '../scene/Light.js';
import { Character }           from '../scene/Character.js';
import { Node }                from '../scene/Node.js';
import { Vec3 }                from '../math/Vec3.js';

// Module-level scratch Vec3 for SDK lookAt calls — avoids per-call allocation.
const _lookAtScratch = new Vec3();

// ─────────────────────────────────────────────────────────────────────────────
// Procedural sphere generator
// ─────────────────────────────────────────────────────────────────────────────
function generateSphere(radius = 1, stacks = 32, slices = 32) {
  const positions = [];
  const normals   = [];
  const uvs       = [];
  const indices   = [];

  for (let i = 0; i <= stacks; i++) {
    const phi = (i / stacks) * Math.PI;
    const cosPhi = Math.cos(phi);
    const sinPhi = Math.sin(phi);

    for (let j = 0; j <= slices; j++) {
      const theta = (j / slices) * 2 * Math.PI;
      const x = sinPhi * Math.sin(theta);
      const y = cosPhi;
      const z = sinPhi * Math.cos(theta);

      positions.push(x * radius, y * radius, z * radius);
      normals.push(x, y, z);
      uvs.push(j / slices, i / stacks);
    }
  }

  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = i * (slices + 1) + j;
      const b = a + slices + 1;
      indices.push(a, b, a + 1);
      indices.push(b, b + 1, a + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals:   new Float32Array(normals),
    uvs:       new Float32Array(uvs),
    indices:   new Uint16Array(indices),
  };
}

// Build a Character from procedural sphere data
function buildDemoCharacter(gl) {
  const sphere = generateSphere(1, 32, 32);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const posBuf = new VertexBuffer(gl, gl.STATIC_DRAW);
  posBuf.upload(sphere.positions);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  const normBuf = new VertexBuffer(gl, gl.STATIC_DRAW);
  normBuf.upload(sphere.normals);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(1);

  const uvBuf = new VertexBuffer(gl, gl.STATIC_DRAW);
  uvBuf.upload(sphere.uvs);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(2);

  const idxBuf = new IndexBuffer(gl, gl.STATIC_DRAW);
  idxBuf.upload(sphere.indices);

  gl.bindVertexArray(null);

  const meshData = {
    vao,
    vertexBuffers: { position: posBuf, normal: normBuf, texCoord0: uvBuf },
    indexBuffer:   idxBuf,
    indexCount:    sphere.indices.length,
    indexType:     gl.UNSIGNED_SHORT,
    material: {
      baseColorFactor:  [0.9, 0.6, 0.3, 1.0],
      roughnessFactor:  0.4,
      metallicFactor:   0.1,
    },
  };

  return new Character({ meshes: [meshData] });
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenHumanInstance
// ─────────────────────────────────────────────────────────────────────────────
class OpenHumanInstance {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {Character} character
   * @param {GLContext} glContext
   * @param {StateCache} stateCache
   * @param {DeferredRenderer|ForwardRenderer} renderer
   * @param {Camera} camera
   * @param {Light[]} lights
   * @param {ShadowMap|null} shadowMap
   */
  constructor(canvas, character, glContext, stateCache, renderer, camera, lights, shadowMap = null) {
    this._canvas    = canvas;
    this._character = character;
    this._glContext = glContext;
    this._cache     = stateCache;
    this._renderer  = renderer;
    this._camera    = camera;
    this._lights    = lights;
    this._shadowMap = shadowMap;
    this._rafId     = null;
    this._events    = {};
    this._exposure  = 1.0;

    // Public namespaces
    const self = this;

    this.animation = {
      play(name)              { character.playAnimation(name); },
      crossFadeTo(name, dur)  { character._animGraph?.crossFadeTo(name, dur); },
      setLayer(name, clip, opts) { /* layer mask support — future PR */ },
      setFloat(param, value)  { character.setFloat(param, value); },
      setBool(param, value)   { character.setBool(param, value); },
    };

    this.morph = {
      set(name, weight)   { character.setMorphWeight(name, weight); },
      setMany(map)        { for (const [k, v] of Object.entries(map)) character.setMorphWeight(k, v); },
    };

    this.streaming = {
      connect(url)   { /* TODO: WebSocket / WebRTC data channel */ },
      disconnect()   { /* TODO */ },
      onData(cb)     { /* TODO */ },
    };

    this.camera = {
      setPosition(x, y, z) { camera.setPosition(x, y, z); },
      lookAt(x, y, z) {
        _lookAtScratch.set(x, y, z);
        camera.lookAt(_lookAtScratch);
      },
      setFOV(deg)          { camera.setFOV(deg); },
      enableOrbit(enable)  { enable ? camera.enableOrbit(canvas) : camera.disableOrbit(); },
      setOrbit(theta, phi, radius) { camera.setOrbit(theta, phi, radius); },
    };

    this.renderer = {
      setExposure(v) {
        self._exposure = v;
        if (self._renderer instanceof DeferredRenderer) {
          self._renderer.setExposure(v);
        }
      },
      setEnvironment(irradianceTex, envTex) {
        if (self._renderer instanceof DeferredRenderer) {
          self._renderer.setEnvironment(irradianceTex ?? null, envTex ?? null);
        }
      },
      setACES(enabled) {
        if (self._renderer instanceof DeferredRenderer) {
          self._renderer.setACES(enabled);
        }
      },
      setIBL(enabled) {
        if (self._renderer instanceof DeferredRenderer) {
          self._renderer.setIBLEnabled(enabled);
        }
      },
      setSSSStrength(v)      { /* TODO: subsurface scattering strength uniform */ },
    };
  }

  on(event, cb) {
    if (!this._events[event]) this._events[event] = [];
    this._events[event].push(cb);
  }

  _emit(event, data) {
    const cbs = this._events[event] ?? [];
    for (const cb of cbs) cb(data);
  }

  _startRenderLoop() {
    const loop = (ts) => {
      this._rafId = requestAnimationFrame(loop);
      this._tick(ts);
    };
    this._rafId = requestAnimationFrame(loop);
  }

  _stopRenderLoop() {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  /** @private */
  _tick(ts) {
    // Handle canvas resize
    const canvas  = this._canvas;
    const dpr     = window.devicePixelRatio ?? 1;
    const displayW = Math.floor(canvas.clientWidth  * dpr);
    const displayH = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width  = displayW;
      canvas.height = displayH;
      this._camera.aspect = displayW / (displayH || 1);
      this._camera.updateProjection();
      this._renderer.resize(displayW, displayH);
    }

    // Compute deltaTime
    const now = performance.now();
    const dt  = Math.min((now - (this._lastTs ?? now)) / 1000, 0.1); // cap at 100ms
    this._lastTs = now;

    this._character.update(dt);

    // Shadow pass (deferred path only)
    if (this._shadowMap && this._renderer instanceof DeferredRenderer) {
      const nodes = this._character.node.children;
      const dirLight = this._lights.find(l => l.type === 'directional');
      if (dirLight) {
        this._shadowMap.render(nodes, dirLight, this._character._gpuSkinning);
      }
    }

    // Main render
    if (this._renderer instanceof DeferredRenderer) {
      this._renderer.render(
        this._character.node.children,
        this._camera,
        this._lights,
        this._character._gpuSkinning,
        this._shadowMap
      );
    } else {
      this._character.render(this._renderer, this._camera, this._lights);
    }

    this._emit('frame', ts);
  }

  destroy() {
    this._stopRenderLoop();
    this._camera.disableOrbit();
    this._character.destroy?.();
    this._renderer.destroy();
    this._shadowMap?.destroy();
    this._glContext.destroy();
    this._emit('destroy', null);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenHuman — static factory
// ─────────────────────────────────────────────────────────────────────────────
export class OpenHuman {
  /**
   * Load a character asset and initialise the engine on the given canvas.
   *
   * @param {string|null} assetUrl — URL to .glb / .gltf, or null / '__demo__' for the built-in demo sphere
   * @param {HTMLCanvasElement} canvas
   * @returns {Promise<OpenHumanInstance>}
   */
  static async load(assetUrl, canvas) {
    const glContext  = new GLContext(canvas);
    const stateCache = new StateCache(glContext.gl);

    const w = canvas.clientWidth  || canvas.width  || 1;
    const h = canvas.clientHeight || canvas.height || 1;

    const renderer  = new DeferredRenderer(glContext, stateCache, { width: w, height: h });
    const shadowMap = new ShadowMap(glContext, stateCache);

    const camera = new Camera({
      fov:    60,
      aspect: canvas.clientWidth / (canvas.clientHeight || 1),
      near:   0.1,
      far:    1000,
    });
    camera.setOrbit(0, Math.PI / 4, 3);
    camera.enableOrbit(canvas);

    const light = new Light('directional');
    light.direction.set(0.5, -1.0, 0.5);
    light.color.set(1, 1, 1);
    light.intensity = 1.0;

    let character;
    const isDemo = !assetUrl || assetUrl === '__demo__';

    if (isDemo) {
      character = buildDemoCharacter(glContext.gl);
    } else {
      const loader = new GLTFLoader(glContext.gl);
      const data   = await loader.load(assetUrl);
      character    = new Character(data, glContext.gl);
    }

    const instance = new OpenHumanInstance(
      canvas, character, glContext, stateCache, renderer, camera, [light], shadowMap
    );
    instance._startRenderLoop();
    return instance;
  }
}

// Re-export core modules for advanced use cases
export { GLContext, StateCache, ForwardRenderer, DeferredRenderer, ShadowMap, GLTFLoader, Camera, Light, Character, Node };
export { VertexBuffer, IndexBuffer };
export { PostProcessStack } from '../renderer/PostProcessStack.js';
export { Skeleton, Joint }    from '../animation/Skeleton.js';
export { AnimationClip, Pose } from '../animation/AnimationClip.js';
export { AnimationGraph }      from '../animation/AnimationGraph.js';
export { GPUSkinning }         from '../animation/GPUSkinning.js';
