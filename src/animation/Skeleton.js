/**
 * Skeleton — joint hierarchy with local/world transform evaluation
 * and GPU skinning dual-quaternion computation.
 */
import { Vec3 }     from '../math/Vec3.js';
import { Quat }     from '../math/Quat.js';
import { Mat4 }     from '../math/Mat4.js';
import { DualQuat } from '../math/DualQuat.js';

// ─────────────────────────────────────────────────────────────────────────────
// Joint
// ─────────────────────────────────────────────────────────────────────────────
export class Joint {
  /**
   * @param {string} name
   * @param {number} index        — index into Skeleton.joints[]
   * @param {number} parentIndex  — -1 for root joints
   */
  constructor(name, index, parentIndex = -1) {
    this.name        = name;
    this.index       = index;
    this.parentIndex = parentIndex;

    // Local pose — set each frame by the animation system
    this.localTranslation = new Vec3(0, 0, 0);
    this.localRotation    = new Quat(0, 0, 0, 1);
    this.localScale       = new Vec3(1, 1, 1);

    // World-space transform (computed from FK traversal)
    this.worldMatrix = new Mat4();

    // Inverse bind-pose matrix (from glTF skin.inverseBindMatrices)
    this.inverseBindMatrix = new Mat4();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Skeleton
// ─────────────────────────────────────────────────────────────────────────────
export class Skeleton {
  /**
   * @param {Joint[]} joints — ordered so parents precede children
   */
  constructor(joints) {
    this.joints = joints;

    // Pre-allocated scratch objects — avoids per-frame heap allocation.
    this._localMat  = new Mat4();
    this._skinMat   = new Mat4();
    this._skinDQs   = joints.map(() => new DualQuat());
  }

  // ----------------------------------------------------------------------- FK
  /**
   * Recompute all joint world matrices using current local pose.
   * Assumes joints[] is topologically ordered (parent before child).
   */
  updateWorldMatrices() {
    const joints    = this.joints;
    const localMat  = this._localMat;

    for (let i = 0; i < joints.length; i++) {
      const j = joints[i];
      Mat4.fromRotationTranslationScale(
        j.localRotation, j.localTranslation, j.localScale, localMat);

      if (j.parentIndex < 0) {
        j.worldMatrix.e.set(localMat.e);
      } else {
        Mat4.multiply(joints[j.parentIndex].worldMatrix, localMat, j.worldMatrix);
      }
    }
  }

  // ----------------------------------------------- skinning dual quaternions
  /**
   * Compute per-joint skinning dual quaternions:
   *   skinDQ[i] = toRigidDQ( worldMatrix[i] * inverseBindMatrix[i] )
   *
   * Must be called after updateWorldMatrices().
   *
   * @param {DualQuat[]} outDQs — pre-allocated array (length ≥ joints.length)
   * @returns {DualQuat[]} outDQs
   */
  computeSkinningDQs(outDQs) {
    const joints   = this.joints;
    const skinMat  = this._skinMat;

    for (let i = 0; i < joints.length; i++) {
      const j = joints[i];
      Mat4.multiply(j.worldMatrix, j.inverseBindMatrix, skinMat);
      _mat4ToRigidDQ(skinMat, outDQs[i]);
    }
    return outDQs;
  }

  /**
   * Convenience: update world matrices then compute skinning DQs.
   * @returns {DualQuat[]}
   */
  evaluate() {
    this.updateWorldMatrices();
    return this.computeSkinningDQs(this._skinDQs);
  }

  // ----------------------------------------------------------------- helpers
  /**
   * Reset all joints to bind pose (identity local transform).
   */
  resetToBindPose() {
    for (const j of this.joints) {
      j.localTranslation.e[0] = 0; j.localTranslation.e[1] = 0; j.localTranslation.e[2] = 0;
      j.localRotation.e[0] = 0; j.localRotation.e[1] = 0; j.localRotation.e[2] = 0; j.localRotation.e[3] = 1;
      j.localScale.e[0] = 1; j.localScale.e[1] = 1; j.localScale.e[2] = 1;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (module-private)
// ─────────────────────────────────────────────────────────────────────────────

// Module-level scratch objects to avoid per-call allocations.
const _rotQ   = new Quat();
const _transV = new Vec3();

/**
 * Extract a rigid-body DualQuat from a column-major Mat4.
 * The matrix is assumed to encode rotation + translation only
 * (scale is stripped by normalising the rotation columns).
 *
 * @param {Mat4}     m
 * @param {DualQuat} out
 */
function _mat4ToRigidDQ(m, out) {
  const e = m.e;

  // Translation
  _transV.e[0] = e[12];
  _transV.e[1] = e[13];
  _transV.e[2] = e[14];

  // Normalise rotation columns to remove any accumulated scale.
  const sx = Math.sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]) || 1;
  const sy = Math.sqrt(e[4]*e[4] + e[5]*e[5] + e[6]*e[6]) || 1;
  const sz = Math.sqrt(e[8]*e[8] + e[9]*e[9] + e[10]*e[10]) || 1;

  const r00 = e[0]/sx;  const r10 = e[1]/sx;  const r20 = e[2]/sx;
  const r01 = e[4]/sy;  const r11 = e[5]/sy;  const r21 = e[6]/sy;
  const r02 = e[8]/sz;  const r12 = e[9]/sz;  const r22 = e[10]/sz;

  // Shepperd's method — rotation matrix → quaternion (column-major convention)
  // Reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
  const trace = r00 + r11 + r22;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    _rotQ.e[3] = 0.25 / s;
    _rotQ.e[0] = (r21 - r12) * s;
    _rotQ.e[1] = (r02 - r20) * s;
    _rotQ.e[2] = (r10 - r01) * s;
  } else if (r00 > r11 && r00 > r22) {
    const s = 2 * Math.sqrt(1 + r00 - r11 - r22);
    _rotQ.e[3] = (r21 - r12) / s;
    _rotQ.e[0] = 0.25 * s;
    _rotQ.e[1] = (r01 + r10) / s;
    _rotQ.e[2] = (r20 + r02) / s;
  } else if (r11 > r22) {
    const s = 2 * Math.sqrt(1 + r11 - r00 - r22);
    _rotQ.e[3] = (r02 - r20) / s;
    _rotQ.e[0] = (r01 + r10) / s;
    _rotQ.e[1] = 0.25 * s;
    _rotQ.e[2] = (r12 + r21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + r22 - r00 - r11);
    _rotQ.e[3] = (r10 - r01) / s;
    _rotQ.e[0] = (r20 + r02) / s;
    _rotQ.e[1] = (r12 + r21) / s;
    _rotQ.e[2] = 0.25 * s;
  }

  DualQuat.fromRotationTranslation(_rotQ, _transV, out);
}
