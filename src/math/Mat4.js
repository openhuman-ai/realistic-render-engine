/**
 * Mat4 — column-major 4×4 matrix backed by Float32Array[16].
 * Column layout: e[col*4 + row]
 */
import { Vec3 } from './Vec3.js';

export class Mat4 {
  constructor() { this.e = new Float32Array(16); Mat4.identity(this); }

  static identity(out = new Mat4()) {
    const e = out.e;
    e[0]=1;e[1]=0;e[2]=0;e[3]=0;
    e[4]=0;e[5]=1;e[6]=0;e[7]=0;
    e[8]=0;e[9]=0;e[10]=1;e[11]=0;
    e[12]=0;e[13]=0;e[14]=0;e[15]=1;
    return out;
  }

  static multiply(a, b, out = new Mat4()) {
    const ae=a.e, be=b.e, oe=out.e;
    for (let col=0; col<4; col++) {
      for (let row=0; row<4; row++) {
        oe[col*4+row] =
          ae[0*4+row]*be[col*4+0]+
          ae[1*4+row]*be[col*4+1]+
          ae[2*4+row]*be[col*4+2]+
          ae[3*4+row]*be[col*4+3];
      }
    }
    return out;
  }

  static transpose(a, out = new Mat4()) {
    const ae=a.e, oe=out.e;
    oe[0]=ae[0];  oe[1]=ae[4];  oe[2]=ae[8];  oe[3]=ae[12];
    oe[4]=ae[1];  oe[5]=ae[5];  oe[6]=ae[9];  oe[7]=ae[13];
    oe[8]=ae[2];  oe[9]=ae[6];  oe[10]=ae[10];oe[11]=ae[14];
    oe[12]=ae[3]; oe[13]=ae[7]; oe[14]=ae[11];oe[15]=ae[15];
    return out;
  }

  static invert(a, out = new Mat4()) {
    const m=a.e, o=out.e;
    const m00=m[0],m01=m[1],m02=m[2],m03=m[3];
    const m10=m[4],m11=m[5],m12=m[6],m13=m[7];
    const m20=m[8],m21=m[9],m22=m[10],m23=m[11];
    const m30=m[12],m31=m[13],m32=m[14],m33=m[15];

    const b00=m00*m11-m01*m10, b01=m00*m12-m02*m10;
    const b02=m00*m13-m03*m10, b03=m01*m12-m02*m11;
    const b04=m01*m13-m03*m11, b05=m02*m13-m03*m12;
    const b06=m20*m31-m21*m30, b07=m20*m32-m22*m30;
    const b08=m20*m33-m23*m30, b09=m21*m32-m22*m31;
    const b10=m21*m33-m23*m31, b11=m22*m33-m23*m32;

    let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
    if (!det) { Mat4.identity(out); return out; }
    det=1/det;

    o[0]=(m11*b11-m12*b10+m13*b09)*det;
    o[1]=(m02*b10-m01*b11-m03*b09)*det;
    o[2]=(m31*b05-m32*b04+m33*b03)*det;
    o[3]=(m22*b04-m21*b05-m23*b03)*det;
    o[4]=(m12*b08-m10*b11-m13*b07)*det;
    o[5]=(m00*b11-m02*b08+m03*b07)*det;
    o[6]=(m32*b02-m30*b05-m33*b01)*det;
    o[7]=(m20*b05-m22*b02+m23*b01)*det;
    o[8]=(m10*b10-m11*b08+m13*b06)*det;
    o[9]=(m01*b08-m00*b10-m03*b06)*det;
    o[10]=(m30*b04-m31*b02+m33*b00)*det;
    o[11]=(m21*b02-m20*b04-m23*b00)*det;
    o[12]=(m11*b07-m10*b09-m12*b06)*det;
    o[13]=(m00*b09-m01*b07+m02*b06)*det;
    o[14]=(m31*b01-m30*b03-m32*b00)*det;
    o[15]=(m20*b03-m21*b01+m22*b00)*det;
    return out;
  }

  /**
   * Perspective projection matrix (right-handed, depth [-1,1]).
   * @param {number} fovy   — vertical field of view in radians
   * @param {number} aspect — width / height
   * @param {number} near
   * @param {number} far
   * @param {Mat4}   [out]
   */
  static perspective(fovy, aspect, near, far, out = new Mat4()) {
    const f  = 1 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    const e  = out.e;
    e[0]=f/aspect; e[1]=0; e[2]=0;              e[3]=0;
    e[4]=0;        e[5]=f; e[6]=0;              e[7]=0;
    e[8]=0;        e[9]=0; e[10]=(far+near)*nf; e[11]=-1;
    e[12]=0;       e[13]=0;e[14]=2*far*near*nf; e[15]=0;
    return out;
  }

  /**
   * Orthographic projection matrix.
   */
  static orthographic(left, right, bottom, top, near, far, out = new Mat4()) {
    const lr=1/(left-right), bt=1/(bottom-top), nf=1/(near-far);
    const e=out.e;
    e[0]=-2*lr;  e[1]=0;      e[2]=0;      e[3]=0;
    e[4]=0;      e[5]=-2*bt;  e[6]=0;      e[7]=0;
    e[8]=0;      e[9]=0;      e[10]=2*nf;  e[11]=0;
    e[12]=(left+right)*lr; e[13]=(top+bottom)*bt; e[14]=(far+near)*nf; e[15]=1;
    return out;
  }

  /**
   * View matrix (camera transform, right-handed).
   * @param {Vec3} eye
   * @param {Vec3} center
   * @param {Vec3} up
   * @param {Mat4} [out]
   */
  static lookAt(eye, center, up, out = new Mat4()) {
    const f = Vec3.normalize(Vec3.sub(center, eye, _v0), _v0);
    const s = Vec3.normalize(Vec3.cross(f, up, _v1), _v1);
    const u = Vec3.cross(s, f, _v2);
    const e = out.e;

    e[0]=s.e[0];  e[1]=u.e[0];  e[2]=-f.e[0]; e[3]=0;
    e[4]=s.e[1];  e[5]=u.e[1];  e[6]=-f.e[1]; e[7]=0;
    e[8]=s.e[2];  e[9]=u.e[2];  e[10]=-f.e[2];e[11]=0;
    e[12]=-Vec3.dot(s,eye);
    e[13]=-Vec3.dot(u,eye);
    e[14]= Vec3.dot(f,eye);
    e[15]=1;
    return out;
  }

  /**
   * Apply a translation to a matrix.
   * @param {Mat4} m
   * @param {Vec3} v
   * @param {Mat4} [out]
   */
  static translate(m, v, out = new Mat4()) {
    if (out !== m) out.e.set(m.e);
    const e=out.e, x=v.e[0], y=v.e[1], z=v.e[2];
    e[12]=e[0]*x+e[4]*y+e[8] *z+e[12];
    e[13]=e[1]*x+e[5]*y+e[9] *z+e[13];
    e[14]=e[2]*x+e[6]*y+e[10]*z+e[14];
    e[15]=e[3]*x+e[7]*y+e[11]*z+e[15];
    return out;
  }

  static rotateX(m, angle, out = new Mat4()) {
    const c=Math.cos(angle), s=Math.sin(angle);
    const ae=m.e, oe=out.e;
    if (out !== m) oe.set(ae);
    const a10=ae[4],a11=ae[5],a12=ae[6],a13=ae[7];
    const a20=ae[8],a21=ae[9],a22=ae[10],a23=ae[11];
    oe[4]=a10*c+a20*s; oe[5]=a11*c+a21*s; oe[6]=a12*c+a22*s; oe[7]=a13*c+a23*s;
    oe[8]=a20*c-a10*s; oe[9]=a21*c-a11*s; oe[10]=a22*c-a12*s; oe[11]=a23*c-a13*s;
    return out;
  }

  static rotateY(m, angle, out = new Mat4()) {
    const c=Math.cos(angle), s=Math.sin(angle);
    const ae=m.e, oe=out.e;
    if (out !== m) oe.set(ae);
    const a00=ae[0],a01=ae[1],a02=ae[2],a03=ae[3];
    const a20=ae[8],a21=ae[9],a22=ae[10],a23=ae[11];
    oe[0]=a00*c-a20*s; oe[1]=a01*c-a21*s; oe[2]=a02*c-a22*s; oe[3]=a03*c-a23*s;
    oe[8]=a00*s+a20*c; oe[9]=a01*s+a21*c; oe[10]=a02*s+a22*c; oe[11]=a03*s+a23*c;
    return out;
  }

  static rotateZ(m, angle, out = new Mat4()) {
    const c=Math.cos(angle), s=Math.sin(angle);
    const ae=m.e, oe=out.e;
    if (out !== m) oe.set(ae);
    const a00=ae[0],a01=ae[1],a02=ae[2],a03=ae[3];
    const a10=ae[4],a11=ae[5],a12=ae[6],a13=ae[7];
    oe[0]=a00*c+a10*s; oe[1]=a01*c+a11*s; oe[2]=a02*c+a12*s; oe[3]=a03*c+a13*s;
    oe[4]=a10*c-a00*s; oe[5]=a11*c-a01*s; oe[6]=a12*c-a02*s; oe[7]=a13*c-a03*s;
    return out;
  }

  /**
   * Build a rotation matrix from a quaternion.
   * @param {Quat} q — {e: Float32Array[4]} x,y,z,w
   * @param {Mat4} [out]
   */
  static fromQuat(q, out = new Mat4()) {
    const e=out.e, qe=q.e;
    const x=qe[0],y=qe[1],z=qe[2],w=qe[3];
    const x2=x+x, y2=y+y, z2=z+z;
    const xx=x*x2, yx=y*x2, yy=y*y2;
    const zx=z*x2, zy=z*y2, zz=z*z2;
    const wx=w*x2, wy=w*y2, wz=w*z2;

    e[0]=1-yy-zz; e[1]=yx+wz;   e[2]=zx-wy;   e[3]=0;
    e[4]=yx-wz;   e[5]=1-xx-zz; e[6]=zy+wx;   e[7]=0;
    e[8]=zx+wy;   e[9]=zy-wx;   e[10]=1-xx-yy;e[11]=0;
    e[12]=0;      e[13]=0;       e[14]=0;       e[15]=1;
    return out;
  }

  /**
   * Build a TRS matrix from quaternion rotation, translation Vec3, scale Vec3.
   * @param {Quat} q
   * @param {Vec3} t
   * @param {Vec3} s
   * @param {Mat4} [out]
   */
  static fromRotationTranslationScale(q, t, s, out = new Mat4()) {
    const e=out.e, qe=q.e;
    const x=qe[0],y=qe[1],z=qe[2],w=qe[3];
    const x2=x+x, y2=y+y, z2=z+z;
    const xx=x*x2, yx=y*x2, yy=y*y2;
    const zx=z*x2, zy=z*y2, zz=z*z2;
    const wx=w*x2, wy=w*y2, wz=w*z2;
    const sx=s.e[0], sy=s.e[1], sz=s.e[2];

    e[0]=(1-yy-zz)*sx; e[1]=(yx+wz)*sx;   e[2]=(zx-wy)*sx;   e[3]=0;
    e[4]=(yx-wz)*sy;   e[5]=(1-xx-zz)*sy; e[6]=(zy+wx)*sy;   e[7]=0;
    e[8]=(zx+wy)*sz;   e[9]=(zy-wx)*sz;   e[10]=(1-xx-yy)*sz;e[11]=0;
    e[12]=t.e[0];      e[13]=t.e[1];       e[14]=t.e[2];       e[15]=1;
    return out;
  }

  static clone(a) { const o=new Mat4(); o.e.set(a.e); return o; }
}

// Reusable scratch vectors for lookAt to avoid per-frame allocations.
const _v0 = new Vec3();
const _v1 = new Vec3();
const _v2 = new Vec3();
