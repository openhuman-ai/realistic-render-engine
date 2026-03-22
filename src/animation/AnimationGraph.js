/**
 * AnimationGraph — state machine with crossfade blending and basic layer support.
 *
 * Usage:
 *   const graph = new AnimationGraph(skeleton);
 *   graph.addState('idle', idleClip);
 *   graph.addState('talk', talkClip);
 *   graph.addTransition('idle', 'talk', 'isTalking', 0.3);
 *   graph.addTransition('talk', 'idle', 'isTalking', 0.2);
 *   graph.play('idle');
 *
 *   // each frame:
 *   graph.setBool('isTalking', true);
 *   graph.update(deltaTime);
 *   const pose = graph.getCurrentPose();    // Pose ready for GPU upload
 */
import { Pose } from './AnimationClip.js';

// ─────────────────────────────────────────────────────────────────────────────
// AnimationGraph
// ─────────────────────────────────────────────────────────────────────────────
export class AnimationGraph {
  /**
   * @param {import('./Skeleton.js').Skeleton} skeleton
   */
  constructor(skeleton) {
    this._skeleton  = skeleton;
    const n         = skeleton.joints.length;

    /** @type {Map<string, { clip: import('./AnimationClip.js').AnimationClip, time: number }>} */
    this._states = new Map();

    /**
     * @type {Array<{
     *   from: string, to: string,
     *   param: string, conditionValue: boolean|number,
     *   duration: number
     * }>}
     */
    this._transitions = [];

    /** @type {Map<string, number|boolean>} */
    this._params = new Map();

    this._currentState  = null;   // name of active state
    this._targetState   = null;   // name of blend-target state (transition)
    this._blendTime     = 0;
    this._blendDuration = 0;

    // Pre-allocated pose buffers — zero GC in hot path
    this._poseA       = new Pose(n);
    this._poseB       = new Pose(n);
    this._blendedPose = new Pose(n);

    // Capture the skeleton's current bind/rest pose.
    // This is used as the base before each sample so that channels absent
    // from a clip retain their bind-pose values (matching glTF semantics).
    this._restPose = new Pose(n);
    this._captureRestPose();

    // Layer support scaffolding — base layer runs the main graph;
    // additional layers (e.g. facial) can override joints by mask.
    /** @type {Array<{ name: string, clip: import('./AnimationClip.js').AnimationClip|null, time: number, weight: number }>} */
    this._layers = [
      { name: 'base',   clip: null, time: 0, weight: 1 },
      { name: 'facial', clip: null, time: 0, weight: 0 },  // stub — full mask in future PR
    ];
  }

  // ────────────────────────────────────────────────── private helpers
  /** Snapshot the current joint TRS from the skeleton into _restPose. @private */
  _captureRestPose() {
    const joints = this._skeleton.joints;
    const pose   = this._restPose;
    for (let i = 0; i < joints.length; i++) {
      const j  = joints[i];
      const ti = i * 3, ri = i * 4, si = i * 3;
      pose.translations[ti]   = j.localTranslation.e[0];
      pose.translations[ti+1] = j.localTranslation.e[1];
      pose.translations[ti+2] = j.localTranslation.e[2];
      pose.rotations[ri]   = j.localRotation.e[0];
      pose.rotations[ri+1] = j.localRotation.e[1];
      pose.rotations[ri+2] = j.localRotation.e[2];
      pose.rotations[ri+3] = j.localRotation.e[3];
      pose.scales[si]   = j.localScale.e[0];
      pose.scales[si+1] = j.localScale.e[1];
      pose.scales[si+2] = j.localScale.e[2];
    }
  }

  // ──────────────────────────────────────────────── state machine API
  /**
   * Register a named state.
   * @param {string}                                          name
   * @param {import('./AnimationClip.js').AnimationClip}      clip
   */
  addState(name, clip) {
    this._states.set(name, { clip, time: 0 });
  }

  /**
   * Add a conditional transition between two states.
   *
   * @param {string}          from           — source state name
   * @param {string}          to             — target state name
   * @param {string}          param          — parameter name to watch
   * @param {number}          duration       — crossfade duration in seconds
   * @param {boolean|number}  [conditionValue=true] — fire when param equals this value
   */
  addTransition(from, to, param, duration, conditionValue = true) {
    this._transitions.push({ from, to, param, duration, conditionValue });
  }

  /** Start playback immediately on the named state. @param {string} name */
  play(name) {
    if (!this._states.has(name)) {
      console.warn(`[AnimationGraph] Unknown state: "${name}"`);
      return;
    }
    this._currentState = name;
    this._targetState  = null;
    this._blendTime    = 0;
    const s = this._states.get(name);
    s.time  = 0;
    this._layers[0].clip = s.clip;
    this._layers[0].time = 0;
  }

  // ──────────────────────────────────────────────── parameter API
  /** @param {string} name @param {number} value */
  setFloat(param, value) {
    this._params.set(param, value);
    this._evaluateTransitions();
  }

  /** @param {string} name @param {boolean} value */
  setBool(param, value) {
    this._params.set(param, value);
    this._evaluateTransitions();
  }

  // ──────────────────────────────────────────────── update / pose
  /**
   * Advance the graph by deltaTime seconds and update the blended pose.
   * @param {number} deltaTime — seconds since last frame
   */
  update(deltaTime) {
    if (!this._currentState) return;

    const stateA = this._states.get(this._currentState);
    stateA.time += deltaTime;

    if (this._targetState) {
      // ─── Transitioning: blend A → B
      const stateB = this._states.get(this._targetState);
      stateB.time += deltaTime;
      this._blendTime += deltaTime;
      const t = Math.min(this._blendTime / this._blendDuration, 1.0);

      // Start from bind/rest pose so channels absent from a clip keep their
      // bind-pose values (matching glTF animation semantics).
      this._poseA.copyFrom(this._restPose);
      this._poseB.copyFrom(this._restPose);

      stateA.clip.sample(stateA.time, this._poseA);
      stateB.clip.sample(stateB.time, this._poseB);

      _blendPoses(this._poseA, this._poseB, t, this._blendedPose);

      if (t >= 1.0) {
        // Transition complete — commit to target state
        this._currentState  = this._targetState;
        this._targetState   = null;
        this._blendTime     = 0;
        this._layers[0].clip = stateB.clip;
      }

    } else {
      // ─── Single state: start from rest pose and sample over it
      this._blendedPose.copyFrom(this._restPose);
      stateA.clip.sample(stateA.time, this._blendedPose);
    }

    // Apply blended pose to skeleton
    this._applyPoseToSkeleton(this._blendedPose);
  }

  /**
   * Returns the most recently computed blended pose.
   * @returns {Pose}
   */
  getCurrentPose() {
    return this._blendedPose;
  }

  // ──────────────────────────────────────────────── private helpers
  /** @private */
  _evaluateTransitions() {
    if (!this._currentState || this._targetState) return;

    for (const tr of this._transitions) {
      if (tr.from !== this._currentState) continue;
      if (!this._states.has(tr.to)) continue;

      const paramVal = this._params.get(tr.param);
      // Compare param value with expected conditionValue
      // Support both bool triggers and numeric thresholds (truthy comparison for numbers)
      const condVal = tr.conditionValue;
      let matches;
      if (typeof condVal === 'boolean') {
        matches = condVal ? !!paramVal : !paramVal;
      } else {
        matches = paramVal === condVal;
      }
      if (!matches) continue;

      // Trigger transition
      this._targetState   = tr.to;
      this._blendDuration = tr.duration;
      this._blendTime     = 0;
      const targetState   = this._states.get(tr.to);
      targetState.time    = 0;
      return;
    }
  }

  /** Write blended pose values back onto skeleton joints. @private */
  _applyPoseToSkeleton(pose) {
    const joints = this._skeleton.joints;
    for (let i = 0; i < joints.length; i++) {
      const j   = joints[i];
      const ti  = i * 3;
      const ri  = i * 4;
      const si  = i * 3;

      j.localTranslation.e[0] = pose.translations[ti];
      j.localTranslation.e[1] = pose.translations[ti + 1];
      j.localTranslation.e[2] = pose.translations[ti + 2];

      j.localRotation.e[0] = pose.rotations[ri];
      j.localRotation.e[1] = pose.rotations[ri + 1];
      j.localRotation.e[2] = pose.rotations[ri + 2];
      j.localRotation.e[3] = pose.rotations[ri + 3];

      j.localScale.e[0] = pose.scales[si];
      j.localScale.e[1] = pose.scales[si + 1];
      j.localScale.e[2] = pose.scales[si + 2];
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-private utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Blend two poses with weight t (0 = all a, 1 = all b).
 * Quaternion blend uses NLERP with antipodality correction.
 * @param {Pose} a @param {Pose} b @param {number} t @param {Pose} out
 */
function _blendPoses(a, b, t, out) {
  const n  = a.jointCount;
  const t1 = 1 - t;

  for (let i = 0; i < n; i++) {
    const ti = i * 3;
    const ri = i * 4;
    const si = i * 3;

    // Translation — linear
    out.translations[ti]     = t1*a.translations[ti]     + t*b.translations[ti];
    out.translations[ti + 1] = t1*a.translations[ti + 1] + t*b.translations[ti + 1];
    out.translations[ti + 2] = t1*a.translations[ti + 2] + t*b.translations[ti + 2];

    // Scale — linear
    out.scales[si]     = t1*a.scales[si]     + t*b.scales[si];
    out.scales[si + 1] = t1*a.scales[si + 1] + t*b.scales[si + 1];
    out.scales[si + 2] = t1*a.scales[si + 2] + t*b.scales[si + 2];

    // Rotation — NLERP with antipodality
    const ax=a.rotations[ri], ay=a.rotations[ri+1],
          az=a.rotations[ri+2], aw=a.rotations[ri+3];
    const bx=b.rotations[ri], by=b.rotations[ri+1],
          bz=b.rotations[ri+2], bw=b.rotations[ri+3];
    const dot = ax*bx + ay*by + az*bz + aw*bw;
    const s   = dot < 0 ? -1 : 1;
    const rx  = t1*ax + t*s*bx;
    const ry  = t1*ay + t*s*by;
    const rz  = t1*az + t*s*bz;
    const rw  = t1*aw + t*s*bw;
    const len = Math.sqrt(rx*rx + ry*ry + rz*rz + rw*rw) || 1;
    out.rotations[ri]     = rx/len;
    out.rotations[ri + 1] = ry/len;
    out.rotations[ri + 2] = rz/len;
    out.rotations[ri + 3] = rw/len;
  }
}
