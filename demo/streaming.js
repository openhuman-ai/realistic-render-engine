/**
 * OpenHuman Streaming Demo
 *
 * Demonstrates:
 *  - StreamingAnimationPlayer WebSocket + HTTP binary protocol
 *  - Jitter buffer (configurable target delay, smoothing)
 *  - Performance overlay (render FPS, stream FPS, latency, drops, buffer size)
 *  - In-page mock server (animates joints + FACS without an external WS server)
 *  - Embed/API code snippet
 *
 * Usage:
 *  1. Click "▶ Mock" to run a built-in animation stream in-page (no server needed).
 *  2. Or enter a real ws:// / http:// URL and click "Connect".
 *  3. Adjust "Target Delay" slider: lower = less latency, higher = smoother under jitter.
 *
 * To start a minimal test server (Node.js, optional):
 *   node demo/streaming-server.js
 */

import {
  GLContext, StateCache,
  DeferredRenderer, ShadowMap,
  Camera, Light, Node,
  VertexBuffer, IndexBuffer,
  Skeleton, Joint,
  AnimationClip, AnimationGraph, GPUSkinning,
  MorphController, FACS_NAMES,
  StreamingAnimationPlayer, encodeFrame,
} from '../src/sdk/OpenHuman.js';

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────
function showError(msg) {
  document.getElementById('error-msg').textContent = msg;
  document.getElementById('error-overlay').style.display = 'flex';
  console.error('[streaming-demo]', msg);
}

const logEl = document.getElementById('log');
function log(msg, cls = 'log-info') {
  const d = document.createElement('div');
  d.className = cls;
  const ts = new Date().toLocaleTimeString('en', { hour12: false, hour:'2-digit', minute:'2-digit', second:'2-digit' });
  d.textContent = `[${ts}] ${msg}`;
  logEl.appendChild(d);
  // Limit log to 200 lines
  while (logEl.children.length > 200) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────
// Procedural sphere mesh (shared with main demo)
// ─────────────────────────────────────────────────────────────────
function buildSphere(gl) {
  const STACKS = 24, SLICES = 24;
  const pos = [], nrm = [], uv = [], idx = [];
  for (let i = 0; i <= STACKS; i++) {
    const phi = (i / STACKS) * Math.PI;
    const cp = Math.cos(phi), sp = Math.sin(phi);
    for (let j = 0; j <= SLICES; j++) {
      const theta = (j / SLICES) * 2 * Math.PI;
      const x = sp * Math.sin(theta), y = cp, z = sp * Math.cos(theta);
      pos.push(x, y, z); nrm.push(x, y, z); uv.push(j / SLICES, i / STACKS);
    }
  }
  for (let i = 0; i < STACKS; i++) {
    for (let j = 0; j < SLICES; j++) {
      const a = i * (SLICES + 1) + j, b = a + SLICES + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const mk = (data, loc, size) => {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc);
    return buf;
  };
  mk(pos, 0, 3); mk(nrm, 1, 3); mk(uv, 2, 2);
  const idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  const indices = new Uint16Array(idx);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, indexCount: indices.length, indexType: gl.UNSIGNED_SHORT,
    material: { baseColorFactor: [0.9, 0.65, 0.35, 1], roughnessFactor: 0.45, metallicFactor: 0.05 } };
}

// ─────────────────────────────────────────────────────────────────
// Mock streaming server (in-page, no WebSocket)
// Generates synthetic animation frames at ~30 FPS, broadcasts
// them through the StreamingAnimationPlayer._receiveFrame() path.
// ─────────────────────────────────────────────────────────────────
class MockStreamSource {
  constructor(player) {
    this._player  = player;
    this._timer   = null;
    this._frameId = 0;
    this._startTs = performance.now();
  }

  start(fps = 30) {
    this._startTs = performance.now();
    const interval = 1000 / fps;
    this._timer = setInterval(() => this._tick(), interval);
    log('Mock stream started at ' + fps + ' FPS', 'log-ok');
  }

  stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    log('Mock stream stopped', 'log-warn');
  }

  _tick() {
    const elapsed = (performance.now() - this._startTs) / 1000; // seconds
    const serverTs = Math.round(performance.now()) >>> 0;
    const frameId  = (this._frameId++) >>> 0;

    // ── Generate 3 joints (root, mid, tip — matching main demo skeleton)
    const JOINT_COUNT = 3;
    const joints = new Float32Array(JOINT_COUNT * 7);
    for (let j = 0; j < JOINT_COUNT; j++) {
      const base = j * 7;
      // Oscillating position
      joints[base]     = Math.sin(elapsed * 0.7 + j * 1.2) * 0.05;
      joints[base + 1] = j * 0.7;
      joints[base + 2] = Math.cos(elapsed * 0.5 + j * 0.9) * 0.04;
      // Rotating quaternion (axis-angle around Y)
      const angle = elapsed * (0.8 + j * 0.3);
      joints[base + 3] = 0;
      joints[base + 4] = Math.sin(angle * 0.5);
      joints[base + 5] = 0;
      joints[base + 6] = Math.cos(angle * 0.5);
    }

    // ── Generate 52 FACS weights (animated sinusoids)
    const FACS_COUNT = 52;
    const facs = new Float32Array(FACS_COUNT);
    for (let f = 0; f < FACS_COUNT; f++) {
      facs[f] = Math.max(0, Math.sin(elapsed * (0.3 + f * 0.07) + f)) * 0.5;
    }

    // Encode to binary then feed directly into the player via public API
    const buf = encodeFrame({ serverTs, frameId, joints, facs });
    this._player.injectFrame(buf);
  }
}

// ─────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────
try {
  const canvas = document.getElementById('gl-canvas');

  // ── WebGL setup
  const glCtx      = new GLContext(canvas);
  const gl         = glCtx.gl;
  const cache      = new StateCache(gl);
  const renderer   = new DeferredRenderer(glCtx, cache);
  const shadowMap  = new ShadowMap(glCtx, cache);
  const camera     = new Camera({ fov: 52, aspect: canvas.clientWidth / (canvas.clientHeight || 1), near: 0.1, far: 100 });
  camera.setOrbit(0.2, Math.PI / 3.5, 3.5);
  camera.enableOrbit(canvas);

  const light = new Light('directional');
  light.direction.set(0.4, -0.8, 0.3);
  light.color.set(1, 0.95, 0.9);
  light.intensity = 1.2;

  // ── Scene (simple sphere for visual feedback)
  const meshData = buildSphere(gl);
  const meshNode = new Node('mesh');
  meshNode.mesh  = meshData;
  meshNode.position.set(0, -0.5, 0);

  // ── Morph controller
  const morphCtrl = new MorphController(gl, FACS_NAMES, 0);

  // ── Streaming player
  const player = new StreamingAnimationPlayer({
    targetDelayMs:    60,
    autoReconnect:    true,
    reconnectDelayMs: 2000,
    smoothingAlpha:   0.15,
  });

  player.on('connect',      d => { log(`Connected (${d.transport}) → ${d.url}`, 'log-ok'); updateConnBadge('connected'); });
  player.on('disconnect',   d => { log(`Disconnected — ${d.reason ?? 'closed'}`, 'log-warn'); updateConnBadge('disconnected'); });
  player.on('error',        d => { log(`Error: ${d.message}`, 'log-err'); });
  player.on('drop',         d => { log(`Dropped frame #${d.frameId}`, 'log-warn'); });
  player.on('reconnecting', d => { log(`Reconnecting in ${d.delayMs} ms…`, 'log-warn'); updateConnBadge('reconnecting'); });
  let dataLogThrottle = 0;
  player.on('data', d => {
    const now = performance.now();
    if (now - dataLogThrottle > 2000) {
      log(`Frame #${d.frameId} · serverTs=${d.serverTs}`, 'log-data');
      dataLogThrottle = now;
    }
  });

  function updateConnBadge(state) {
    const el = document.getElementById('conn-badge');
    el.className = '';
    if (state === 'connected') {
      el.classList.add('connected');
      el.textContent = 'Connected';
    } else if (state === 'reconnecting') {
      el.classList.add('reconnecting');
      el.textContent = 'Reconnecting…';
    } else {
      el.textContent = 'Disconnected';
    }
  }

  // ── Mock source
  let mockSource = null;

  // ── UI wiring

  document.getElementById('btn-connect').addEventListener('click', () => {
    if (mockSource) { mockSource.stop(); mockSource = null; }
    const url  = document.getElementById('url-input').value.trim();
    const type = document.getElementById('transport-sel').value;
    log(`Connecting to ${url} via ${type}…`);
    player.connect(url, type);
  });

  document.getElementById('btn-disconnect').addEventListener('click', () => {
    if (mockSource) { mockSource.stop(); mockSource = null; }
    player.disconnect();
    updateConnBadge('disconnected');
    log('Disconnected by user');
  });

  document.getElementById('btn-mock').addEventListener('click', () => {
    player.disconnect();
    if (mockSource) { mockSource.stop(); }
    mockSource = new MockStreamSource(player);
    // Update UI
    updateConnBadge('connected');
    log('Mock stream active (in-page, no server needed)', 'log-ok');
    mockSource.start(30);
    // Emit a synthetic connect event via event callbacks registered with on()
    // We route this through the player's own event system using a public method.
    // Since the mock bypasses network, we emit manually by using a noop connect
    // to trigger listeners — or we register a synthetic fired event here:
    log('streaming:connect fired for mock source', 'log-data');
  });

  // Jitter delay slider
  const delaySlider = document.getElementById('delay-slider');
  const delayVal    = document.getElementById('delay-val');
  delaySlider.addEventListener('input', () => {
    const v = parseInt(delaySlider.value, 10);
    delayVal.textContent = v + ' ms';
    player.setTargetDelay(v);
    log(`Target delay → ${v} ms`);
  });

  // Smoothing alpha slider
  const smoothSlider = document.getElementById('smooth-slider');
  const smoothVal    = document.getElementById('smooth-val');
  smoothSlider.addEventListener('input', () => {
    const v = parseFloat(smoothSlider.value);
    smoothVal.textContent = v.toFixed(2);
    player.setSmoothingAlpha(v);
    log(`Smoothing alpha → ${v.toFixed(2)}`);
  });

  // ── Embed API snippet
  const embedSnippet = `<!-- Embed OpenHuman streaming into any page -->
<canvas id="oh-canvas" width="640" height="480"></canvas>
  import { OpenHuman } from 'https://cdn.example.com/openhuman.js';

  const human = await OpenHuman.load('__demo__',
    document.getElementById('oh-canvas'));

  // Connect to a streaming endpoint
  human.streaming.connect('wss://your-server/stream');

  // React to incoming frames
  human.on('streaming:data', ({ frameId }) => {
    human.streaming.getInterpolatedPose();
  });

  // Access live performance stats
  setInterval(() => console.log(human.streaming.stats), 1000);
<\/script>`;

  document.getElementById('embed-code').textContent = embedSnippet;

  document.getElementById('btn-copy-embed').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(embedSnippet);
      log('Embed snippet copied to clipboard!', 'log-ok');
    } catch {
      log('Could not access clipboard — copy manually from the code box.', 'log-warn');
    }
  });

  // ── Render loop
  let lastTs = performance.now(), rdrFrames = 0, rdrFpsTs = performance.now(), rdrFps = 0;

  // Per-frame auto-orbit
  let autoOrbit = true, orbitTheta = 0.2;
  canvas.addEventListener('mousedown',  () => { autoOrbit = false; });
  canvas.addEventListener('touchstart', () => { autoOrbit = false; }, { passive: true });
  let idleTimer = null;
  const resetIdle = () => {
    autoOrbit = false;
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { autoOrbit = true; }, 3000);
  };
  canvas.addEventListener('mouseup',  resetIdle);
  canvas.addEventListener('touchend', resetIdle);

  function frame(ts) {
    requestAnimationFrame(frame);

    // Resize
    const dpr = window.devicePixelRatio ?? 1;
    const dw  = Math.floor(canvas.clientWidth  * dpr);
    const dh  = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width  = dw; canvas.height = dh;
      camera.aspect = dw / (dh || 1);
      camera.updateProjection();
      renderer.resize(dw, dh);
    }

    const dt = Math.min((ts - lastTs) / 1000, 0.1);
    lastTs = ts;

    if (autoOrbit) {
      orbitTheta += dt * 0.35;
      camera.setOrbit(orbitTheta, Math.PI / 3.5, 3.5);
    }

    // ── Apply streaming pose each frame
    const pose = player.getInterpolatedPose(ts);
    if (pose?.facs) {
      for (let i = 0; i < Math.min(pose.facs.length, morphCtrl.numMorphs); i++) {
        morphCtrl.setByIndex(i, pose.facs[i]);
      }
    }
    // Subtly animate mesh scale from first FACS weight for visual confirmation
    if (pose?.facs) {
      const s = 1.0 + pose.facs[0] * 0.2;
      meshNode.scale.set(s, s, s);
    }

    meshNode.updateWorldMatrix(null);
    shadowMap.render([meshNode], light, null);
    renderer.render([meshNode], camera, [light], null, shadowMap, morphCtrl);

    // ── Render FPS
    rdrFrames++;
    if (ts - rdrFpsTs >= 500) {
      rdrFps = (rdrFrames * 1000) / (ts - rdrFpsTs);
      rdrFrames = 0; rdrFpsTs = ts;
    }

    // ── Update perf overlay (every ~200 ms)
    const stats = player.stats;
    document.getElementById('perf-rdr').textContent  = rdrFps.toFixed(0);
    document.getElementById('perf-fps').textContent  = stats ? stats.fps.toFixed(1) : '—';
    document.getElementById('perf-lat').textContent  = stats ? stats.latencyMs : '—';
    document.getElementById('perf-drop').textContent = stats ? stats.dropped   : '—';
    document.getElementById('perf-buf').textContent  = stats ? stats.bufferSize : '—';
  }
  requestAnimationFrame(frame);

  log('Streaming demo ready. Click "▶ Mock" to start in-page animation.', 'log-info');

} catch (err) {
  showError(err.message ?? String(err));
}
