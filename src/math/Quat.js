/**
 * Quat — unit quaternion [x, y, z, w] backed by Float32Array[4].
 */
export class Quat {
  constructor(x = 0, y = 0, z = 0, w = 1) {
    this.e = new Float32Array([x, y, z, w]);
  }
  get x() { return this.e[0]; } set x(v) { this.e[0] = v; }
  get y() { return this.e[1]; } set y(v) { this.e[1] = v; }
  get z() { return this.e[2]; } set z(v) { this.e[2] = v; }
  get w() { return this.e[3]; } set w(v) { this.e[3] = v; }

  static identity(out = new Quat()) {
    out.e[0]=0; out.e[1]=0; out.e[2]=0; out.e[3]=1; return out;
  }

  static multiply(a, b, out = new Quat()) {
    const ae=a.e, be=b.e;
    const ax=ae[0], ay=ae[1], az=ae[2], aw=ae[3];
    const bx=be[0], by=be[1], bz=be[2], bw=be[3];
    out.e[0]=ax*bw+aw*bx+ay*bz-az*by;
    out.e[1]=ay*bw+aw*by+az*bx-ax*bz;
    out.e[2]=az*bw+aw*bz+ax*by-ay*bx;
    out.e[3]=aw*bw-ax*bx-ay*by-az*bz;
    return out;
  }

  static length(a) {
    const e=a.e;
    return Math.sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]+e[3]*e[3]);
  }

  static dot(a, b) {
    return a.e[0]*b.e[0]+a.e[1]*b.e[1]+a.e[2]*b.e[2]+a.e[3]*b.e[3];
  }

  static normalize(a, out = new Quat()) {
    const l = Quat.length(a) || 1;
    const inv = 1/l;
    out.e[0]=a.e[0]*inv; out.e[1]=a.e[1]*inv;
    out.e[2]=a.e[2]*inv; out.e[3]=a.e[3]*inv;
    return out;
  }

  static conjugate(a, out = new Quat()) {
    out.e[0]=-a.e[0]; out.e[1]=-a.e[1]; out.e[2]=-a.e[2]; out.e[3]=a.e[3];
    return out;
  }

  static invert(a, out = new Quat()) {
    const e=a.e;
    const d=e[0]*e[0]+e[1]*e[1]+e[2]*e[2]+e[3]*e[3];
    if (!d) { Quat.identity(out); return out; }
    const inv=1/d;
    out.e[0]=-e[0]*inv; out.e[1]=-e[1]*inv;
    out.e[2]=-e[2]*inv; out.e[3]= e[3]*inv;
    return out;
  }

  /**
   * Spherical linear interpolation.
   * @param {Quat} a
   * @param {Quat} b
   * @param {number} t — [0,1]
   * @param {Quat} [out]
   */
  static slerp(a, b, t, out = new Quat()) {
    const ae=a.e, be=b.e, oe=out.e;
    let bx=be[0], by=be[1], bz=be[2], bw=be[3];
    let d = ae[0]*bx+ae[1]*by+ae[2]*bz+ae[3]*bw;
    // Shortest path
    if (d < 0) { bx=-bx; by=-by; bz=-bz; bw=-bw; d=-d; }
    let scale0, scale1;
    if (d > 0.9995) {
      scale0=1-t; scale1=t;
    } else {
      const theta=Math.acos(d);
      const sinTheta=Math.sin(theta);
      scale0=Math.sin((1-t)*theta)/sinTheta;
      scale1=Math.sin(t*theta)/sinTheta;
    }
    oe[0]=scale0*ae[0]+scale1*bx;
    oe[1]=scale0*ae[1]+scale1*by;
    oe[2]=scale0*ae[2]+scale1*bz;
    oe[3]=scale0*ae[3]+scale1*bw;
    return out;
  }

  /**
   * From Euler angles (XYZ order), angles in radians.
   */
  static fromEuler(x, y, z, out = new Quat()) {
    const hx=x*0.5, hy=y*0.5, hz=z*0.5;
    const cx=Math.cos(hx), cy=Math.cos(hy), cz=Math.cos(hz);
    const sx=Math.sin(hx), sy=Math.sin(hy), sz=Math.sin(hz);
    out.e[0]=sx*cy*cz+cx*sy*sz;
    out.e[1]=cx*sy*cz-sx*cy*sz;
    out.e[2]=cx*cy*sz+sx*sy*cz;
    out.e[3]=cx*cy*cz-sx*sy*sz;
    return out;
  }

  /**
   * From axis-angle. axis must be normalised.
   * @param {Vec3} axis
   * @param {number} angle — radians
   */
  static fromAxisAngle(axis, angle, out = new Quat()) {
    const half=angle*0.5, s=Math.sin(half);
    out.e[0]=axis.e[0]*s;
    out.e[1]=axis.e[1]*s;
    out.e[2]=axis.e[2]*s;
    out.e[3]=Math.cos(half);
    return out;
  }

  /**
   * Build quaternion from a 3×3 rotation matrix (Mat3 column-major).
   */
  static fromMat3(m, out = new Quat()) {
    const me=m.e;
    const m00=me[0],m10=me[1],m20=me[2];
    const m01=me[3],m11=me[4],m21=me[5];
    const m02=me[6],m12=me[7],m22=me[8];
    const trace=m00+m11+m22;

    if (trace > 0) {
      const s=0.5/Math.sqrt(trace+1);
      out.e[3]=0.25/s;
      out.e[0]=(m12-m21)*s;
      out.e[1]=(m20-m02)*s;
      out.e[2]=(m01-m10)*s;
    } else if (m00>m11 && m00>m22) {
      const s=2*Math.sqrt(1+m00-m11-m22);
      out.e[3]=(m12-m21)/s;
      out.e[0]=0.25*s;
      out.e[1]=(m01+m10)/s;
      out.e[2]=(m20+m02)/s;
    } else if (m11>m22) {
      const s=2*Math.sqrt(1+m11-m00-m22);
      out.e[3]=(m20-m02)/s;
      out.e[0]=(m01+m10)/s;
      out.e[1]=0.25*s;
      out.e[2]=(m12+m21)/s;
    } else {
      const s=2*Math.sqrt(1+m22-m00-m11);
      out.e[3]=(m01-m10)/s;
      out.e[0]=(m20+m02)/s;
      out.e[1]=(m12+m21)/s;
      out.e[2]=0.25*s;
    }
    return out;
  }

  /**
   * Convert quaternion to a Mat4 rotation matrix.
   * (Delegates to Mat4.fromQuat to avoid circular imports.)
   */
  static toMat4(q, out) {
    // Inline to avoid circular import
    const e=out.e, qe=q.e;
    const x=qe[0],y=qe[1],z=qe[2],w=qe[3];
    const x2=x+x, y2=y+y, z2=z+z;
    const xx=x*x2, yx=y*x2, yy=y*y2;
    const zx=z*x2, zy=z*y2, zz=z*z2;
    const wx=w*x2, wy=w*y2, wz=w*z2;
    e[0]=1-yy-zz; e[1]=yx+wz;   e[2]=zx-wy;    e[3]=0;
    e[4]=yx-wz;   e[5]=1-xx-zz; e[6]=zy+wx;    e[7]=0;
    e[8]=zx+wy;   e[9]=zy-wx;   e[10]=1-xx-yy; e[11]=0;
    e[12]=0;      e[13]=0;       e[14]=0;        e[15]=1;
    return out;
  }

  static clone(a) { return new Quat(a.e[0], a.e[1], a.e[2], a.e[3]); }
}
