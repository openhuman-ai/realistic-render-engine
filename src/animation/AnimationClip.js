/**
 * AnimationClip — keyframe data + sampling.
 *
 * Supports the three glTF interpolation modes:
 *   STEP        — immediate snap at the next keyframe
 *   LINEAR      — component-wise linear interpolation (NLERP for rotations)
 *   CUBICSPLINE — glTF Hermite spline; values stored as
 *                 [inTangent…, value…, outTangent…] per keyframe
 *
 * A Pose object is passed in to receive the sampled values; no objects
 * are allocated during sampling.
 */
import { Quat } from '../math/Quat.js';

// ─────────────────────────────────────────────────────────────────────────────
// Pose — pre-allocated flat-array pose buffer
// ─────────────────────────────────────────────────────────────────────────────
export class Pose {
  /**
   * @param {number} jointCount
   */
  constructor(jointCount) {
    this.jointCount   = jointCount;
    this.translations = new Float32Array(jointCount * 3);
    this.rotations    = new Float32Array(jointCount * 4);
    this.scales       = new Float32Array(jointCount * 3);

    // Initialise to identity pose
    for (let i = 0; i < jointCount; i++) {
      this.rotations[i * 4 + 3] = 1.0;   // w = 1
      this.scales[i * 3]     = 1.0;
      this.scales[i * 3 + 1] = 1.0;
      this.scales[i * 3 + 2] = 1.0;
    }
  }

  /** Copy src into this pose. @param {Pose} src */
  copyFrom(src) {
    this.translations.set(src.translations);
    this.rotations.set(src.rotations);
    this.scales.set(src.scales);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnimationChannel  (internal descriptor)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * @typedef {Object} AnimationChannel
 * @property {number}        targetJointIndex
 * @property {'translation'|'rotation'|'scale'} property
 * @property {'LINEAR'|'STEP'|'CUBICSPLINE'} interpolation
 * @property {Float32Array}  times   — keyframe timestamps (seconds), ascending
 * @property {Float32Array}  values  — packed keyframe values (see above)
 */

// ─────────────────────────────────────────────────────────────────────────────
// AnimationClip
// ─────────────────────────────────────────────────────────────────────────────
export class AnimationClip {
  /**
   * @param {string}             name
   * @param {number}             duration — total clip length in seconds
   * @param {AnimationChannel[]} channels
   */
  constructor(name, duration, channels) {
    this.name     = name;
    this.duration = duration > 0 ? duration : 1;
    this.channels = channels;
    this.loop     = true;
  }

  // ----------------------------------------------------------------- sampling
  /**
   * Sample all channels at time t and write results into pose.
   * Only channels present in the clip are written; missing channels
   * retain whatever was already in pose (useful for blending).
   *
   * @param {number} t    — playback time in seconds
   * @param {Pose}   pose — output pose (must be large enough for all joints)
   */
  sample(t, pose) {
    const time = this.loop
      ? ((t % this.duration) + this.duration) % this.duration
      : Math.min(Math.max(t, 0), this.duration);

    for (const ch of this.channels) {
      _sampleChannel(ch, time, pose);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling helpers (module-private, allocation-free)
// ─────────────────────────────────────────────────────────────────────────────

// Reusable scratch for normalised quaternion
const _qScratch = new Quat();

/**
 * Binary-search for the lower keyframe index at time t.
 * Returns the index of the last keyframe whose time ≤ t.
 * Clamps to [0, n-2] so there is always a next frame.
 *
 * @param {Float32Array} times
 * @param {number}       t
 * @returns {number}
 */
function _findKeyframe(times, t) {
  const n = times.length;
  if (n <= 1) return 0;
  if (t <= times[0])     return 0;
  if (t >= times[n - 1]) return n - 2;

  let lo = 0, hi = n - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/**
 * Cubic Hermite interpolation scalar.
 * @param {number} p0 @param {number} m0 @param {number} p1 @param {number} m1 @param {number} t
 */
function _hermite(p0, m0, p1, m1, t) {
  const t2 = t * t, t3 = t2 * t;
  return (2*t3 - 3*t2 + 1)*p0 + (t3 - 2*t2 + t)*m0 + (-2*t3 + 3*t2)*p1 + (t3 - t2)*m1;
}

/** @param {AnimationChannel} ch @param {number} t @param {Pose} pose */
function _sampleChannel(ch, t, pose) {
  const { targetJointIndex: ji, property: prop, interpolation: interp,
          times, values } = ch;
  const k  = _findKeyframe(times, t);
  const k1 = k + 1;

  // Component count per keyframe value
  const comps = prop === 'rotation' ? 4 : 3;
  const base  = ji * comps;
  const dest  = prop === 'translation' ? pose.translations
              : prop === 'rotation'    ? pose.rotations
              :                          pose.scales;

  if (interp === 'STEP' || times.length === 1) {
    // --- STEP: snap to current keyframe value
    const vBase = k * comps;
    for (let c = 0; c < comps; c++) dest[base + c] = values[vBase + c];
    if (prop === 'rotation') _nlerp(dest, base);
    return;
  }

  const t0 = times[k];
  const t1 = times[k1];
  const dt = t1 - t0;
  const s  = dt > 1e-8 ? (t - t0) / dt : 0;  // local parametric t in [0,1]

  if (interp === 'LINEAR') {
    // --- LINEAR
    const v0 = k  * comps;
    const v1 = k1 * comps;

    if (prop === 'rotation') {
      // NLERP for quaternions (faster than SLERP, adequate for small dt)
      _nlerpBlend(values, v0, v1, s, dest, base, comps);
    } else {
      for (let c = 0; c < comps; c++) {
        dest[base + c] = values[v0 + c] + s * (values[v1 + c] - values[v0 + c]);
      }
    }
    return;
  }

  if (interp === 'CUBICSPLINE') {
    // --- CUBICSPLINE (glTF Hermite)
    // Values are stored as:  [inTangent(comps), value(comps), outTangent(comps)] per keyframe
    const stride = comps * 3;
    const v0  = k  * stride + comps;         // value at k
    const o0  = k  * stride + comps * 2;     // outTangent at k
    const i1  = k1 * stride;                 // inTangent at k+1
    const v1  = k1 * stride + comps;         // value at k+1

    if (prop === 'rotation') {
      // Hermite interpolate each component then NLERP
      for (let c = 0; c < comps; c++) {
        dest[base + c] = _hermite(
          values[v0 + c], dt * values[o0 + c],
          values[v1 + c], dt * values[i1 + c], s);
      }
      _nlerp(dest, base);
    } else {
      for (let c = 0; c < comps; c++) {
        dest[base + c] = _hermite(
          values[v0 + c], dt * values[o0 + c],
          values[v1 + c], dt * values[i1 + c], s);
      }
    }
    return;
  }

  // Fallback: copy value at k
  const vBase = k * comps;
  for (let c = 0; c < comps; c++) dest[base + c] = values[vBase + c];
}

/**
 * Normalise quaternion in-place at flat-array offset.
 * @param {Float32Array} arr @param {number} base
 */
function _nlerp(arr, base) {
  const x = arr[base], y = arr[base+1], z = arr[base+2], w = arr[base+3];
  const len = Math.sqrt(x*x + y*y + z*z + w*w) || 1;
  arr[base]   = x/len; arr[base+1] = y/len;
  arr[base+2] = z/len; arr[base+3] = w/len;
}

/**
 * Normalised linear blend between two quaternions stored in flat arrays,
 * handling antipodality.
 */
function _nlerpBlend(src, i0, i1, t, dst, base, comps) {
  // Dot product to detect antipodal
  const d = src[i0]*src[i1] + src[i0+1]*src[i1+1] + src[i0+2]*src[i1+2] + src[i0+3]*src[i1+3];
  const sign = d < 0 ? -1 : 1;
  const t1 = 1 - t;
  for (let c = 0; c < comps; c++) {
    dst[base + c] = t1 * src[i0 + c] + t * sign * src[i1 + c];
  }
  _nlerp(dst, base);
}
