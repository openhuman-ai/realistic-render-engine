/**
 * Deferred PBR Demo — GPU Dual-Quaternion Skinning + G-buffer + ACES
 *
 * Builds a procedural segmented cylinder with 3 joints and a 2-bone rig.
 * Demonstrates:
 *   - Deferred rendering pipeline (G-buffer → PBR lighting → ACES tone-map)
 *   - PCF shadow mapping from a directional light
 *   - IBL toggle (hemisphere ambient fallback vs BRDF-LUT IBL)
 *   - ACES filmic tone-mapping toggle
 *   - Exposure control
 *   - Pipeline toggle (Deferred ↔ Forward)
 *   - GPU dual-quaternion skinning
 *   - AnimationGraph idle→talk crossfade
 */
import {
  GLContext, StateCache,
  ForwardRenderer, DeferredRenderer, ShadowMap,
  EyeRenderer, HairRenderer, buildHairCards,
  Camera, Light, Node,
  VertexBuffer, IndexBuffer,
  Skeleton, Joint,
  AnimationClip,
  AnimationGraph,
  GPUSkinning,
  MorphController, FACS_NAMES,
} from '../src/sdk/OpenHuman.js';

const canvas  = document.getElementById('gl-canvas');
const fpsEl   = document.getElementById('fps');
const infoEl  = document.getElementById('renderer-info');
const stateEl = document.getElementById('anim-state');
const pipeEl  = document.getElementById('pipeline-label');
const pathBadge = document.getElementById('path-badge');

function showError(msg) {
  const overlay = document.getElementById('error-overlay');
  document.getElementById('error-msg').textContent = msg;
  overlay.style.display = 'flex';
  console.error('[demo]', msg);
}

// ─────────────────────────────────────────────────────────────────
// 1. Build a skinned cylinder mesh (3 joints, 2 bones)
//    The cylinder is oriented along Y, split into 3 segments.
//    Joint 0 (root): base at Y=0
//    Joint 1 (mid) : at Y=1
//    Joint 2 (tip) : at Y=2
// ─────────────────────────────────────────────────────────────────
function buildSkinnedCylinder(gl) {
  const RADIUS  = 0.18;
  const HEIGHT  = 2.0;
  const SEGS    = 24;
  const VSTACKS = 12;

  const positions = [], normals = [], uvs = [];
  const joints = [], weights = [], indices = [];

  for (let vi = 0; vi <= VSTACKS; vi++) {
    const y   = (vi / VSTACKS) * HEIGHT;
    const t0  = y;
    const w0  = Math.max(0, 1.0 - Math.max(0, (t0 - 0.6) / 0.8));
    const w1  = Math.max(0, Math.min(1, 1.0 - Math.abs(t0 - 1.0)));
    const w2  = Math.max(0, 1.0 - Math.max(0, (2.0 - t0 - 0.6) / 0.8));
    const inv = 1.0 / (w0 + w1 + w2 + 1e-6);

    for (let si = 0; si <= SEGS; si++) {
      const theta = (si / SEGS) * 2 * Math.PI;
      const nx = Math.cos(theta), nz = Math.sin(theta);
      positions.push(nx * RADIUS, y, nz * RADIUS);
      normals.push(nx, 0, nz);
      uvs.push(si / SEGS, vi / VSTACKS);
      joints.push(0, 1, 2, 0);
      weights.push(w0 * inv, w1 * inv, w2 * inv, 0);
    }
  }

  for (let vi = 0; vi < VSTACKS; vi++) {
    for (let si = 0; si < SEGS; si++) {
      const a = vi * (SEGS + 1) + si, b = a + (SEGS + 1);
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }

  // Bottom cap
  const botC = positions.length / 3;
  positions.push(0, 0, 0); normals.push(0, -1, 0); uvs.push(0.5, 0.5);
  joints.push(0, 1, 2, 0); weights.push(1, 0, 0, 0);
  for (let si = 0; si < SEGS; si++) indices.push(botC, (si + 1) % SEGS, si);

  // Top cap
  const topC = positions.length / 3;
  const lrb  = VSTACKS * (SEGS + 1);
  positions.push(0, HEIGHT, 0); normals.push(0, 1, 0); uvs.push(0.5, 0.5);
  joints.push(1, 2, 0, 0); weights.push(0, 1, 0, 0);
  for (let si = 0; si < SEGS; si++) indices.push(topC, lrb + si, lrb + (si + 1) % SEGS);

  const posArr = new Float32Array(positions);
  const nrmArr = new Float32Array(normals);
  const uvArr  = new Float32Array(uvs);
  const jArr   = new Float32Array(joints);
  const wArr   = new Float32Array(weights);
  const idxArr = new Uint16Array(indices);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const upload = (data, loc, size) => {
    const buf = new VertexBuffer(gl, gl.STATIC_DRAW);
    buf.upload(data);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc);
    return buf;
  };

  const posBuf = upload(posArr, 0, 3);
  const nrmBuf = upload(nrmArr, 1, 3);
  const uvBuf  = upload(uvArr,  2, 2);
  const jBuf   = upload(jArr,   3, 4);
  const wBuf   = upload(wArr,   4, 4);

  const idxBuf = new IndexBuffer(gl, gl.STATIC_DRAW);
  idxBuf.upload(idxArr);

  gl.bindVertexArray(null);

  return {
    vao,
    vertexBuffers: { position: posBuf, normal: nrmBuf, texCoord0: uvBuf,
                     joints: jBuf, weights: wBuf },
    indexBuffer: idxBuf,
    indexCount:  idxArr.length,
    indexType:   gl.UNSIGNED_SHORT,
    skinned:     true,
    material: {
      baseColorFactor: [0.85, 0.55, 0.3, 1.0],
      roughnessFactor: 0.45,
      metallicFactor:  0.05,
    },
    _cpuPositions: posArr,   // retained for morph delta generation
    vertexCount:   posArr.length / 3,
  };
}

// ─────────────────────────────────────────────────────────────────
// 1b. Generate procedural FACS morph deltas for the demo cylinder
//     These create visible deformations so the FACS sliders are live.
// ─────────────────────────────────────────────────────────────────
function buildDemoMorphDeltas(posArr, vertexCount) {
  const RADIUS = 0.18;

  // Each entry: Float32Array of 3*vertexCount floats (dx,dy,dz per vertex)
  const deltas = Array.from({ length: 52 }, () => new Float32Array(vertexCount * 3));

  const ss = (e0, e1, v) => {
    const t = Math.max(0, Math.min(1, (v - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
  };

  for (let v = 0; v < vertexCount; v++) {
    const x = posArr[v * 3], y = posArr[v * 3 + 1], z = posArr[v * 3 + 2];
    const r = Math.sqrt(x * x + z * z);
    const nx = r > 1e-4 ? x / r : 0, nz = r > 1e-4 ? z / r : 0;

    // jawOpen (index 25): lower half stretches down
    const tJaw = ss(1.0, 0.0, y);
    deltas[25][v * 3 + 1] = -tJaw * 0.45;
    deltas[25][v * 3 + 2] =  tJaw * 0.12;

    // mouthSmileLeft (43): left side of mid-band bulges out
    const xL = Math.max(0, -x / (RADIUS + 1e-6));
    const yMid = ss(0.8, 1.0, y) * ss(1.4, 1.2, y);
    deltas[43][v * 3]     = -xL * yMid * 0.12;
    deltas[43][v * 3 + 1] =  xL * yMid * 0.06;

    // mouthSmileRight (44): right side of mid-band bulges out
    const xR = Math.max(0, x / (RADIUS + 1e-6));
    deltas[44][v * 3]     =  xR * yMid * 0.12;
    deltas[44][v * 3 + 1] =  xR * yMid * 0.06;

    // mouthPucker (38): mid-band pinches radially inward
    const yPuck = ss(0.7, 0.9, y) * ss(1.3, 1.1, y);
    deltas[38][v * 3]     = -nx * yPuck * 0.09;
    deltas[38][v * 3 + 2] = -nz * yPuck * 0.09;

    // cheekPuff (5): mid section expands outward
    const yPuff = ss(0.4, 0.7, y) * ss(1.6, 1.3, y);
    deltas[5][v * 3]     = nx * yPuff * 0.14;
    deltas[5][v * 3 + 2] = nz * yPuff * 0.14;

    // browInnerUp (2): top extends upward
    const yBrow = ss(1.6, 2.0, y);
    deltas[2][v * 3 + 1] = yBrow * 0.25;

    // eyeBlinkLeft (8): upper-left collapses inward
    const tBL = Math.max(0, -x / (RADIUS + 1e-6)) * ss(1.4, 2.0, y);
    deltas[8][v * 3]     = -nx * tBL * 0.1;
    deltas[8][v * 3 + 2] = -nz * tBL * 0.1;

    // eyeBlinkRight (9): upper-right collapses inward
    const tBR = Math.max(0, x / (RADIUS + 1e-6)) * ss(1.4, 2.0, y);
    deltas[9][v * 3]     = -nx * tBR * 0.1;
    deltas[9][v * 3 + 2] = -nz * tBR * 0.1;

    // jawLeft (23): lower half slides left
    const tJL = ss(1.0, 0.0, y);
    deltas[23][v * 3] = -tJL * 0.1;

    // jawRight (24): lower half slides right
    deltas[24][v * 3] = tJL * 0.1;

    // noseSneerLeft (49): upper-left wrinkle upward
    const tNL = Math.max(0, -x / (RADIUS + 1e-6)) * ss(1.0, 1.4, y);
    deltas[49][v * 3 + 1] = tNL * 0.08;
    deltas[49][v * 3 + 2] = tNL * 0.05;
  }

  return deltas;
}

// ─────────────────────────────────────────────────────────────────
// 2. Build skeleton (3 joints, parent-child chain)
// ─────────────────────────────────────────────────────────────────
function buildSkeleton() {
  const j0 = new Joint('root', 0, -1); j0.localTranslation.set(0, 0, 0);
  const j1 = new Joint('mid',  1,  0); j1.localTranslation.set(0, 1, 0);
  const j2 = new Joint('tip',  2,  1); j2.localTranslation.set(0, 1, 0);
  const skel = new Skeleton([j0, j1, j2]);
  skel.updateWorldMatrices();
  for (const j of skel.joints) _mat4Invert(j.worldMatrix, j.inverseBindMatrix);
  return skel;
}

function _mat4Invert(src, dst) {
  const m = src.e, o = dst.e;
  const m00=m[0],m01=m[1],m02=m[2],m03=m[3];
  const m10=m[4],m11=m[5],m12=m[6],m13=m[7];
  const m20=m[8],m21=m[9],m22=m[10],m23=m[11];
  const m30=m[12],m31=m[13],m32=m[14],m33=m[15];
  const b00=m00*m11-m01*m10,b01=m00*m12-m02*m10,b02=m00*m13-m03*m10;
  const b03=m01*m12-m02*m11,b04=m01*m13-m03*m11,b05=m02*m13-m03*m12;
  const b06=m20*m31-m21*m30,b07=m20*m32-m22*m30,b08=m20*m33-m23*m30;
  const b09=m21*m32-m22*m31,b10=m21*m33-m23*m31,b11=m22*m33-m23*m32;
  let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (!det) return; det=1/det;
  o[0]=(m11*b11-m12*b10+m13*b09)*det; o[1]=(m02*b10-m01*b11-m03*b09)*det;
  o[2]=(m31*b05-m32*b04+m33*b03)*det; o[3]=(m22*b04-m21*b05-m23*b03)*det;
  o[4]=(m12*b08-m10*b11-m13*b07)*det; o[5]=(m00*b11-m02*b08+m03*b07)*det;
  o[6]=(m32*b02-m30*b05-m33*b01)*det; o[7]=(m20*b05-m22*b02+m23*b01)*det;
  o[8]=(m10*b10-m11*b08+m13*b06)*det; o[9]=(m01*b08-m00*b10-m03*b06)*det;
  o[10]=(m30*b04-m31*b02+m33*b00)*det; o[11]=(m21*b02-m20*b04-m23*b00)*det;
  o[12]=(m11*b07-m10*b09-m12*b06)*det; o[13]=(m00*b09-m01*b07+m02*b06)*det;
  o[14]=(m31*b01-m30*b03-m32*b00)*det; o[15]=(m20*b03-m21*b01+m22*b00)*det;
}

// ─────────────────────────────────────────────────────────────────
// 3. Build animation clips (idle sway + talk nod)
// ─────────────────────────────────────────────────────────────────
function makeRotationChannel(jointIdx, times, angles, axis) {
  const values = new Float32Array(times.length * 4);
  for (let i = 0; i < times.length; i++) {
    const h = angles[i] * 0.5, s = Math.sin(h), c = Math.cos(h);
    const b = i * 4;
    values[b] = axis === 'x' ? s : 0;
    values[b+1] = axis === 'y' ? s : 0;
    values[b+2] = axis === 'z' ? s : 0;
    values[b+3] = c;
  }
  return { targetJointIndex: jointIdx, property: 'rotation', interpolation: 'LINEAR',
           times: new Float32Array(times), values };
}

function buildIdleClip() {
  return new AnimationClip('idle', 2.0, [
    makeRotationChannel(1, [0,0.5,1.0,1.5,2.0], [0,0.26,0,-0.26,0], 'z'),
  ]);
}

function buildTalkClip() {
  const t = [0,0.25,0.5,0.75,1.0];
  return new AnimationClip('talk', 1.0, [
    makeRotationChannel(1, t, [0,0.44,0,-0.44,0], 'z'),
    makeRotationChannel(2, t, [0,-0.26,0,0.26,0], 'z'),
  ]);
}

// ─────────────────────────────────────────────────────────────────
// 4. Main entry point
// ─────────────────────────────────────────────────────────────────
try {
  const glCtx = new GLContext(canvas);
  const gl    = glCtx.gl;
  const cache = new StateCache(gl);

  // ── Create both renderer instances (deferred is default)
  const deferredRenderer = new DeferredRenderer(glCtx, cache);
  const forwardRenderer  = new ForwardRenderer(glCtx, cache);
  const shadowMap        = new ShadowMap(glCtx, cache);
  const eyeRenderer      = new EyeRenderer(glCtx, cache);
  const hairRenderer     = new HairRenderer(glCtx, cache);

  // Active renderer pointer — starts on deferred path
  let renderer     = deferredRenderer;
  let useDeferred  = true;

  // Per-frame toggle flags
  let shadowEnabled = true;
  let iblEnabled    = false;
  let acesEnabled   = true;
  let sssEnabled    = false;
  let eyeEnabled    = true;
  let hairEnabled   = true;

  const camera = new Camera({ fov: 50, aspect: canvas.clientWidth / (canvas.clientHeight || 1), near: 0.1, far: 100 });
  camera.setOrbit(0.2, Math.PI / 3.5, 3.5);
  camera.enableOrbit(canvas);

  const light = new Light('directional');
  light.direction.set(0.4, -0.8, 0.3);
  light.color.set(1, 0.95, 0.9);
  light.intensity = 1.2;

  // ── Scene
  const skeleton = buildSkeleton();
  const meshData = buildSkinnedCylinder(gl);
  const gpuSkin  = new GPUSkinning(gl, skeleton.joints.length);

  // ── Morph controller — 52 FACS slots with procedural demo deltas
  const morphCtrl   = new MorphController(gl, FACS_NAMES, meshData.vertexCount);
  const morphDeltas = buildDemoMorphDeltas(meshData._cpuPositions, meshData.vertexCount);
  for (let i = 0; i < 52; i++) {
    morphCtrl.uploadMorphDeltas(i, morphDeltas[i], null);
  }

  const charNode = new Node('Character');
  const meshNode = new Node('mesh');
  meshNode.mesh  = meshData;
  charNode.addChild(meshNode);
  charNode.position.set(0, -1, 0);

  // Additional demo meshes for PR#4
  const skinNode = new Node('skinSphere');
  skinNode.mesh  = { ...meshData };
  skinNode.position.set(-0.75, 0.05, -0.15);
  skinNode.scale.set(0.62, 0.62, 0.62);
  skinNode.mesh.material = {
    baseColorFactor: [0.92, 0.62, 0.50, 1.0],
    roughnessFactor: 0.58,
    metallicFactor:  0.02,
  };

  const eyeNode = new Node('eye');
  eyeNode.mesh  = { vao: true }; // marker mesh for eye renderer pass
  eyeNode.position.set(0.22, 0.15, 0.16);
  eyeNode.scale.set(0.24, 0.24, 0.24);

  const hairNode = new Node('hairCards');
  hairNode.mesh  = buildHairCards(gl, { cardCount: 48, cardLength: 0.95, cardWidth: 0.06 });
  hairNode.position.set(0.0, 0.48, 0.0);

  const sceneNodes = [meshNode, skinNode];

  // ── Animation graph
  const graph = new AnimationGraph(skeleton);
  graph.addState('idle', buildIdleClip());
  graph.addState('talk', buildTalkClip());
  graph.addTransition('idle', 'talk', 'isTalking', 0.35, true);
  graph.addTransition('talk', 'idle', 'isTalking', 0.35, false);
  graph.play('idle');

  // ── Update path badge + HUD labels
  function updatePathUI() {
    if (useDeferred) {
      const parts = ['Deferred PBR · G-buffer'];
      if (shadowEnabled) parts.push('PCF Shadows');
      if (iblEnabled)    parts.push('IBL');
      if (sssEnabled)    parts.push('SSS');
      if (eyeEnabled)    parts.push('Eye');
      if (hairEnabled)   parts.push('Hair');
      if (acesEnabled)   parts.push('ACES');
      pathBadge.textContent = parts.join(' · ');
      pathBadge.style.background = '#7a3c1c';
      pipeEl.textContent = 'Deferred';
      infoEl.textContent = 'WebGL 2.0 · Deferred PBR · G-buffer MRT';
    } else {
      pathBadge.textContent = 'Forward · Phong';
      pathBadge.style.background = '#2c4a8c';
      pipeEl.textContent = 'Forward';
      infoEl.textContent = 'WebGL 2.0 · Forward Renderer · Phong';
    }
  }
  updatePathUI();
  stateEl.textContent = 'idle';

  // ── Render controls wiring

  // Exposure
  const expSlider = document.getElementById('exposure-slider');
  const expVal    = document.getElementById('exposure-val');
  expSlider.addEventListener('input', () => {
    const v = parseFloat(expSlider.value);
    expVal.textContent = v.toFixed(2);
    deferredRenderer.setExposure(v);
  });

  // Shadow toggle
  const btnShadow = document.getElementById('btn-shadow');
  btnShadow.addEventListener('click', () => {
    shadowEnabled = !shadowEnabled;
    btnShadow.textContent = shadowEnabled ? 'ON' : 'OFF';
    btnShadow.classList.toggle('off', !shadowEnabled);
    updatePathUI();
  });

  // IBL toggle
  const btnIBL = document.getElementById('btn-ibl');
  btnIBL.addEventListener('click', () => {
    iblEnabled = !iblEnabled;
    deferredRenderer.setIBLEnabled(iblEnabled);
    btnIBL.textContent = iblEnabled ? 'ON' : 'OFF';
    btnIBL.classList.toggle('off', !iblEnabled);
    updatePathUI();
  });

  // ACES toggle
  const btnACES = document.getElementById('btn-aces');
  btnACES.addEventListener('click', () => {
    acesEnabled = !acesEnabled;
    deferredRenderer.setACES(acesEnabled);
    btnACES.textContent = acesEnabled ? 'ON' : 'OFF';
    btnACES.classList.toggle('off', !acesEnabled);
    updatePathUI();
  });

  // Pipeline toggle: Deferred ↔ Forward
  const btnPipeline = document.getElementById('btn-pipeline');
  btnPipeline.addEventListener('click', () => {
    useDeferred = !useDeferred;
    renderer    = useDeferred ? deferredRenderer : forwardRenderer;
    btnPipeline.textContent = useDeferred ? 'Deferred' : 'Forward';
    btnPipeline.classList.toggle('off', !useDeferred);
    // Sync canvas size to newly active renderer
    renderer.resize(canvas.width, canvas.height);
    updatePathUI();
  });

  // SSS controls
  const btnSSS = document.getElementById('btn-sss');
  btnSSS.addEventListener('click', () => {
    sssEnabled = !sssEnabled;
    deferredRenderer.setSSSEnabled(sssEnabled);
    btnSSS.textContent = sssEnabled ? 'ON' : 'OFF';
    btnSSS.classList.toggle('off', !sssEnabled);
    updatePathUI();
  });
  const sssStrength = document.getElementById('sss-strength');
  const sssStrengthVal = document.getElementById('sss-strength-val');
  sssStrength.addEventListener('input', () => {
    const v = parseFloat(sssStrength.value);
    sssStrengthVal.textContent = v.toFixed(2);
    deferredRenderer.setSSSStrength(v);
  });
  const sssWidth = document.getElementById('sss-width');
  const sssWidthVal = document.getElementById('sss-width-val');
  sssWidth.addEventListener('input', () => {
    const v = parseFloat(sssWidth.value);
    sssWidthVal.textContent = v.toFixed(1);
    deferredRenderer.setSSSWidth(v);
  });
  const sssR = document.getElementById('sss-r');
  const sssG = document.getElementById('sss-g');
  const sssB = document.getElementById('sss-b');
  const sssRVal = document.getElementById('sss-r-val');
  const sssGVal = document.getElementById('sss-g-val');
  const sssBVal = document.getElementById('sss-b-val');
  function applySSSColorFromUI() {
    const r = parseFloat(sssR.value), g = parseFloat(sssG.value), b = parseFloat(sssB.value);
    sssRVal.textContent = r.toFixed(2);
    sssGVal.textContent = g.toFixed(2);
    sssBVal.textContent = b.toFixed(2);
    deferredRenderer.setSSSColor(r, g, b);
  }
  sssR.addEventListener('input', applySSSColorFromUI);
  sssG.addEventListener('input', applySSSColorFromUI);
  sssB.addEventListener('input', applySSSColorFromUI);
  applySSSColorFromUI();

  // Eye controls
  const btnEye = document.getElementById('btn-eye');
  btnEye.addEventListener('click', () => {
    eyeEnabled = !eyeEnabled;
    btnEye.textContent = eyeEnabled ? 'ON' : 'OFF';
    btnEye.classList.toggle('off', !eyeEnabled);
    updatePathUI();
  });
  const eyeIris = document.getElementById('eye-iris');
  const eyeIrisVal = document.getElementById('eye-iris-val');
  eyeIris.addEventListener('input', () => {
    const v = parseFloat(eyeIris.value);
    eyeIrisVal.textContent = v.toFixed(2);
    eyeRenderer.setIrisDilate(v);
  });

  // Hair controls
  const btnHair = document.getElementById('btn-hair');
  btnHair.addEventListener('click', () => {
    hairEnabled = !hairEnabled;
    btnHair.textContent = hairEnabled ? 'ON' : 'OFF';
    btnHair.classList.toggle('off', !hairEnabled);
    updatePathUI();
  });
  const hairSpec = document.getElementById('hair-spec');
  const hairSpecVal = document.getElementById('hair-spec-val');
  hairSpec.addEventListener('input', () => {
    const v = parseFloat(hairSpec.value);
    hairSpecVal.textContent = v.toFixed(0);
    hairRenderer.setSpecPower1(v);
    // Secondary lobe is wider/softer than primary; clamp to keep it visible at low values.
    hairRenderer.setSpecPower2(Math.max(8, v * 0.4));
  });

  // ── Animation controls
  document.getElementById('btn-idle').addEventListener('click', () => {
    graph.setBool('isTalking', false);
    stateEl.textContent = 'idle (crossfading…)';
    setTimeout(() => { if (stateEl.textContent.startsWith('idle')) stateEl.textContent = 'idle'; }, 380);
  });
  document.getElementById('btn-talk').addEventListener('click', () => {
    graph.setBool('isTalking', true);
    stateEl.textContent = 'talk (crossfading…)';
    setTimeout(() => { if (stateEl.textContent.startsWith('talk')) stateEl.textContent = 'talk'; }, 380);
  });

  // ── FACS Slider Panel (PR #5)
  // Groups matching the canonical FACS_NAMES order
  const FACS_GROUPS = [
    { label: 'Brow', names: ['browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight'] },
    { label: 'Cheek', names: ['cheekPuff','cheekSquintLeft','cheekSquintRight'] },
    { label: 'Eye — Blink & Squint', names: ['eyeBlinkLeft','eyeBlinkRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight'] },
    { label: 'Eye — Look', names: ['eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight'] },
    { label: 'Jaw', names: ['jawForward','jawLeft','jawRight','jawOpen','mouthClose'] },
    { label: 'Mouth — Shape', names: ['mouthFunnel','mouthPucker','mouthLeft','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper'] },
    { label: 'Mouth — Smile / Frown', names: ['mouthSmileLeft','mouthSmileRight','mouthFrownLeft','mouthFrownRight','mouthDimpleLeft','mouthDimpleRight'] },
    { label: 'Mouth — Stretch / Press', names: ['mouthStretchLeft','mouthStretchRight','mouthPressLeft','mouthPressRight'] },
    { label: 'Mouth — Upper / Lower', names: ['mouthUpperUpLeft','mouthUpperUpRight','mouthLowerDownLeft','mouthLowerDownRight'] },
    { label: 'Nose', names: ['noseSneerLeft','noseSneerRight'] },
    { label: 'Tongue', names: ['tongueOut'] },
  ];

  /** Build the FACS panel rows and wire slider events. */
  function buildFACSPanel() {
    const scroll = document.getElementById('facs-scroll');
    scroll.innerHTML = '';
    const sliderMap = new Map(); // name → { slider, valEl }

    for (const group of FACS_GROUPS) {
      const groupEl = document.createElement('div');
      groupEl.className = 'facs-group-label';
      groupEl.textContent = group.label;
      scroll.appendChild(groupEl);

      for (const name of group.names) {
        const row = document.createElement('div');
        row.className = 'facs-row';

        const label = document.createElement('label');
        label.title = name;
        // Shorten camelCase for display: insert spaces before uppercase
        label.textContent = name.replace(/([A-Z])/g, ' $1').trim();

        const slider = document.createElement('input');
        slider.type  = 'range';
        slider.min   = '0';
        slider.max   = '1';
        slider.step  = '0.01';
        slider.value = '0';

        const valEl = document.createElement('span');
        valEl.className = 'facs-val';
        valEl.textContent = '0.00';

        slider.addEventListener('input', () => {
          const w = parseFloat(slider.value);
          valEl.textContent = w.toFixed(2);
          morphCtrl.set(name, w);
        });

        row.appendChild(label);
        row.appendChild(slider);
        row.appendChild(valEl);
        scroll.appendChild(row);
        sliderMap.set(name, { slider, valEl });
      }
    }

    // Reset all
    document.getElementById('btn-facs-reset').addEventListener('click', () => {
      for (const [name, { slider, valEl }] of sliderMap) {
        slider.value = '0';
        valEl.textContent = '0.00';
        morphCtrl.set(name, 0);
      }
    });

    return sliderMap;
  }

  buildFACSPanel();

  // FACS panel open/close toggle
  const facsPanel  = document.getElementById('facs-panel');
  const facsBtnTgl = document.getElementById('btn-facs-toggle');
  facsBtnTgl.addEventListener('click', () => {
    facsPanel.classList.toggle('open');
    facsBtnTgl.textContent = facsPanel.classList.contains('open') ? '✕ Close FACS' : '🎭 FACS';
  });

  // ── Render loop
  let lastTs     = performance.now();
  let frameCount = 0;
  let lastFps    = performance.now();

  let autoOrbit = true, orbitTheta = 0.2;
  canvas.addEventListener('mousedown',  () => { autoOrbit = false; });
  canvas.addEventListener('touchstart', () => { autoOrbit = false; }, { passive: true });
  let idleTimer = null;
  const resetIdle = () => {
    autoOrbit = false;
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { autoOrbit = true; }, 2500);
  };
  canvas.addEventListener('mouseup',  resetIdle);
  canvas.addEventListener('touchend', resetIdle);

  function frame(ts) {
    requestAnimationFrame(frame);

    // ── Resize
    const dpr = window.devicePixelRatio ?? 1;
    const dw  = Math.floor(canvas.clientWidth  * dpr);
    const dh  = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width  = dw; canvas.height = dh;
      camera.aspect = dw / (dh || 1);
      camera.updateProjection();
      deferredRenderer.resize(dw, dh);
      forwardRenderer.resize(dw, dh);
    }

    const dt = Math.min((ts - lastTs) / 1000, 0.1);
    lastTs = ts;

    // ── Auto-orbit
    if (autoOrbit) {
      orbitTheta += dt * 0.4;
      camera.setOrbit(orbitTheta, Math.PI / 3.5, 3.5);
    }

    // ── Animation update
    graph.update(dt);
    gpuSkin.update(skeleton);
    charNode.updateWorldMatrix(null);

    // ── Shadow pass (deferred path only, when shadows enabled)
    const activeShadow = (useDeferred && shadowEnabled) ? shadowMap : null;
    if (activeShadow) {
      shadowMap.render(sceneNodes, light, gpuSkin);
    }

    // ── Main render
    if (useDeferred) {
      deferredRenderer.render(sceneNodes, camera, [light], gpuSkin, activeShadow, morphCtrl);
    } else {
      forwardRenderer.render(sceneNodes, camera, [light], gpuSkin);
    }

    // Forward overlay passes for specialised eye/hair shaders
    if (eyeEnabled || hairEnabled) {
      // Clear depth so eye/hair overlay passes can render on top of deferred scene color.
      gl.clear(gl.DEPTH_BUFFER_BIT);
    }
    if (eyeEnabled) {
      eyeNode.updateWorldMatrix(null);
      eyeRenderer.render([eyeNode], camera, [light]);
    }
    if (hairEnabled) {
      hairNode.updateWorldMatrix(null);
      hairRenderer.render([hairNode], camera, [light]);
    }

    // ── FPS counter
    frameCount++;
    if (ts - lastFps >= 500) {
      fpsEl.textContent = (frameCount / ((ts - lastFps) / 1000)).toFixed(0);
      frameCount = 0; lastFps = ts;
    }
  }
  requestAnimationFrame(frame);

} catch (err) {
  showError(err.message ?? String(err));
}
