/**
 * DualQuat — Dual Quaternion for rigid-body skinning (ScLERP blend).
 * A dual quaternion is represented as { real: Quat, dual: Quat }.
 * Convention: real = rotation quat, dual = (1/2) * t * real
 */
import { Quat } from './Quat.js';
import { Vec3 } from './Vec3.js';

export class DualQuat {
  constructor() {
    this.real = new Quat(0, 0, 0, 1); // identity rotation
    this.dual = new Quat(0, 0, 0, 0); // zero translation
  }

  /** @returns {DualQuat} identity */
  static create() {
    return new DualQuat();
  }

  /**
   * Build from a unit quaternion and a translation vector.
   * @param {Quat} q — rotation
   * @param {Vec3} t — translation
   * @param {DualQuat} [out]
   */
  static fromRotationTranslation(q, t, out = new DualQuat()) {
    const qn = Quat.normalize(q, _qScratch);
    out.real.e.set(qn.e);

    // dual = 0.5 * (tx, ty, tz, 0) * q
    const tx=t.e[0]*0.5, ty=t.e[1]*0.5, tz=t.e[2]*0.5;
    const qx=qn.e[0], qy=qn.e[1], qz=qn.e[2], qw=qn.e[3];
    out.dual.e[0]= tx*qw+ty*qz-tz*qy;
    out.dual.e[1]=-tx*qz+ty*qw+tz*qx;
    out.dual.e[2]= tx*qy-ty*qx+tz*qw;
    out.dual.e[3]=-tx*qx-ty*qy-tz*qz;
    return out;
  }

  /**
   * Multiply two dual quaternions: out = a * b
   */
  static multiply(a, b, out = new DualQuat()) {
    Quat.multiply(a.real, b.real, out.real);
    // dual = a.real * b.dual + a.dual * b.real
    Quat.multiply(a.real, b.dual, _dqA);
    Quat.multiply(a.dual, b.real, _dqB);
    const oe=out.dual.e;
    oe[0]=_dqA.e[0]+_dqB.e[0];
    oe[1]=_dqA.e[1]+_dqB.e[1];
    oe[2]=_dqA.e[2]+_dqB.e[2];
    oe[3]=_dqA.e[3]+_dqB.e[3];
    return out;
  }

  /**
   * Normalise so that ||real|| = 1 (keeps dual perpendicular to real).
   * @param {DualQuat} a
   * @param {DualQuat} [out]
   */
  static normalize(a, out = new DualQuat()) {
    const mag = Quat.length(a.real);
    if (mag < 1e-10) { DualQuat.identity(out); return out; }
    const inv = 1/mag;
    const re=a.real.e, de=a.dual.e, ore=out.real.e, ode=out.dual.e;
    ore[0]=re[0]*inv; ore[1]=re[1]*inv; ore[2]=re[2]*inv; ore[3]=re[3]*inv;
    ode[0]=de[0]*inv; ode[1]=de[1]*inv; ode[2]=de[2]*inv; ode[3]=de[3]*inv;
    return out;
  }

  /** Set out to identity dual quaternion */
  static identity(out = new DualQuat()) {
    Quat.identity(out.real);
    out.dual.e[0]=0; out.dual.e[1]=0; out.dual.e[2]=0; out.dual.e[3]=0;
    return out;
  }

  /**
   * Convert to a 4×4 column-major transformation matrix.
   * @param {DualQuat} dq
   * @param {Mat4} out — Mat4-like object with Float32Array .e[16]
   */
  static toMat4(dq, out) {
    const norm = DualQuat.normalize(dq, _dqNorm);
    const re=norm.real.e, de=norm.dual.e;
    const qx=re[0], qy=re[1], qz=re[2], qw=re[3];
    const x2=qx+qx, y2=qy+qy, z2=qz+qz;
    const xx=qx*x2, yx=qy*x2, yy=qy*y2;
    const zx=qz*x2, zy=qz*y2, zz=qz*z2;
    const wx=qw*x2, wy=qw*y2, wz=qw*z2;

    // Rotation part
    const e=out.e;
    e[0]=1-yy-zz; e[1]=yx+wz;    e[2]=zx-wy;    e[3]=0;
    e[4]=yx-wz;   e[5]=1-xx-zz;  e[6]=zy+wx;    e[7]=0;
    e[8]=zx+wy;   e[9]=zy-wx;    e[10]=1-xx-yy; e[11]=0;

    // Translation part: t = 2 * dual * conj(real)
    // t.xyz = 2*(rw*d.xyz - dw*r.xyz + cross(r.xyz, d.xyz))
    const tx=2*(de[0]*re[3]-de[3]*re[0]+re[1]*de[2]-re[2]*de[1]);
    const ty=2*(de[1]*re[3]-de[3]*re[1]+re[2]*de[0]-re[0]*de[2]);
    const tz=2*(de[2]*re[3]-de[3]*re[2]+re[0]*de[1]-re[1]*de[0]);
    e[12]=tx; e[13]=ty; e[14]=tz; e[15]=1;
    return out;
  }

  /**
   * ScLERP — scalar linear blend of dual quaternions (used for skinning).
   * @param {DualQuat} a
   * @param {DualQuat} b
   * @param {number}   t
   * @param {DualQuat} [out]
   */
  static lerp(a, b, t, out = new DualQuat()) {
    // Ensure shortest path by checking dot(a.real, b.real)
    let dot = Quat.dot(a.real, b.real);
    const sign = dot < 0 ? -1 : 1;
    const t1 = 1 - t;

    const ore=out.real.e, ode=out.dual.e;
    const are=a.real.e, ade=a.dual.e, bre=b.real.e, bde=b.dual.e;
    ore[0]=t1*are[0]+sign*t*bre[0];
    ore[1]=t1*are[1]+sign*t*bre[1];
    ore[2]=t1*are[2]+sign*t*bre[2];
    ore[3]=t1*are[3]+sign*t*bre[3];
    ode[0]=t1*ade[0]+sign*t*bde[0];
    ode[1]=t1*ade[1]+sign*t*bde[1];
    ode[2]=t1*ade[2]+sign*t*bde[2];
    ode[3]=t1*ade[3]+sign*t*bde[3];
    return DualQuat.normalize(out, out);
  }

  static clone(a) {
    const out = new DualQuat();
    out.real.e.set(a.real.e);
    out.dual.e.set(a.dual.e);
    return out;
  }
}

// Module-level scratch objects to avoid per-call allocations.
const _qScratch = new Quat();
const _dqA      = new Quat();
const _dqB      = new Quat();
const _dqNorm   = new DualQuat();
