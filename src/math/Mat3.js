/**
 * Mat3 — column-major 3×3 matrix backed by Float32Array[9].
 * Column layout: e[col*3 + row]
 */
export class Mat3 {
  constructor() { this.e = new Float32Array(9); Mat3.identity(this); }

  static identity(out = new Mat3()) {
    const e = out.e;
    e[0]=1; e[3]=0; e[6]=0;
    e[1]=0; e[4]=1; e[7]=0;
    e[2]=0; e[5]=0; e[8]=1;
    return out;
  }

  static multiply(a, b, out = new Mat3()) {
    const ae=a.e, be=b.e, oe=out.e;
    const a00=ae[0],a10=ae[1],a20=ae[2];
    const a01=ae[3],a11=ae[4],a21=ae[5];
    const a02=ae[6],a12=ae[7],a22=ae[8];

    const b00=be[0],b10=be[1],b20=be[2];
    const b01=be[3],b11=be[4],b21=be[5];
    const b02=be[6],b12=be[7],b22=be[8];

    oe[0]=a00*b00+a01*b10+a02*b20;
    oe[1]=a10*b00+a11*b10+a12*b20;
    oe[2]=a20*b00+a21*b10+a22*b20;
    oe[3]=a00*b01+a01*b11+a02*b21;
    oe[4]=a10*b01+a11*b11+a12*b21;
    oe[5]=a20*b01+a21*b11+a22*b21;
    oe[6]=a00*b02+a01*b12+a02*b22;
    oe[7]=a10*b02+a11*b12+a12*b22;
    oe[8]=a20*b02+a21*b12+a22*b22;
    return out;
  }

  static transpose(a, out = new Mat3()) {
    const ae=a.e, oe=out.e;
    oe[0]=ae[0]; oe[1]=ae[3]; oe[2]=ae[6];
    oe[3]=ae[1]; oe[4]=ae[4]; oe[5]=ae[7];
    oe[6]=ae[2]; oe[7]=ae[5]; oe[8]=ae[8];
    return out;
  }

  static invert(a, out = new Mat3()) {
    const ae=a.e;
    const a00=ae[0],a01=ae[3],a02=ae[6];
    const a10=ae[1],a11=ae[4],a12=ae[7];
    const a20=ae[2],a21=ae[5],a22=ae[8];

    const b01= a22*a11-a12*a21;
    const b11=-a22*a10+a12*a20;
    const b21= a21*a10-a11*a20;

    let det = a00*b01 + a01*b11 + a02*b21;
    if (!det) { Mat3.identity(out); return out; }
    det = 1/det;

    const oe=out.e;
    oe[0]=b01*det; oe[1]=b11*det; oe[2]=b21*det;
    oe[3]=(-a22*a01+a02*a21)*det;
    oe[4]=(a22*a00-a02*a20)*det;
    oe[5]=(-a21*a00+a01*a20)*det;
    oe[6]=(a12*a01-a02*a11)*det;
    oe[7]=(-a12*a00+a02*a10)*det;
    oe[8]=(a11*a00-a01*a10)*det;
    return out;
  }

  /**
   * Extract the upper-left 3×3 from a Mat4.
   * @param {Mat4} m
   * @param {Mat3} [out]
   */
  static fromMat4(m, out = new Mat3()) {
    const me=m.e, oe=out.e;
    oe[0]=me[0]; oe[1]=me[1]; oe[2]=me[2];
    oe[3]=me[4]; oe[4]=me[5]; oe[5]=me[6];
    oe[6]=me[8]; oe[7]=me[9]; oe[8]=me[10];
    return out;
  }

  /**
   * Normal matrix: transpose of the inverse of the upper-left 3×3 of a Mat4.
   * Used to transform normals into view/world space.
   * @param {Mat4} m
   * @param {Mat3} [out]
   */
  static normalMatrix(m, out = new Mat3()) {
    Mat3.fromMat4(m, out);
    Mat3.invert(out, out);
    Mat3.transpose(out, out);
    return out;
  }

  static clone(a) {
    const out = new Mat3();
    out.e.set(a.e);
    return out;
  }
}
