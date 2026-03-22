// Float32Array-backed Vec3 with out-parameter style static methods.
export class Vec3 {
  constructor(x = 0, y = 0, z = 0) { this.e = new Float32Array([x, y, z]); }
  get x() { return this.e[0]; } set x(v) { this.e[0] = v; }
  get y() { return this.e[1]; } set y(v) { this.e[1] = v; }
  get z() { return this.e[2]; } set z(v) { this.e[2] = v; }
  set(x, y, z) { this.e[0] = x; this.e[1] = y; this.e[2] = z; return this; }

  static add(a, b, out = new Vec3()) {
    out.e[0]=a.e[0]+b.e[0]; out.e[1]=a.e[1]+b.e[1]; out.e[2]=a.e[2]+b.e[2]; return out;
  }
  static sub(a, b, out = new Vec3()) {
    out.e[0]=a.e[0]-b.e[0]; out.e[1]=a.e[1]-b.e[1]; out.e[2]=a.e[2]-b.e[2]; return out;
  }
  static scale(a, s, out = new Vec3()) {
    out.e[0]=a.e[0]*s; out.e[1]=a.e[1]*s; out.e[2]=a.e[2]*s; return out;
  }
  static negate(a, out = new Vec3()) {
    out.e[0]=-a.e[0]; out.e[1]=-a.e[1]; out.e[2]=-a.e[2]; return out;
  }
  static dot(a, b) {
    return a.e[0]*b.e[0] + a.e[1]*b.e[1] + a.e[2]*b.e[2];
  }
  static cross(a, b, out = new Vec3()) {
    const ax=a.e[0], ay=a.e[1], az=a.e[2];
    const bx=b.e[0], by=b.e[1], bz=b.e[2];
    out.e[0]=ay*bz-az*by;
    out.e[1]=az*bx-ax*bz;
    out.e[2]=ax*by-ay*bx;
    return out;
  }
  static len(a) { return Math.sqrt(Vec3.dot(a, a)); }
  static normalize(a, out = new Vec3()) {
    const l = Vec3.len(a) || 1;
    return Vec3.scale(a, 1/l, out);
  }
  static lerp(a, b, t, out = new Vec3()) {
    const e=out.e, ae=a.e, be=b.e;
    e[0]=ae[0]+(be[0]-ae[0])*t;
    e[1]=ae[1]+(be[1]-ae[1])*t;
    e[2]=ae[2]+(be[2]-ae[2])*t;
    return out;
  }
  static distanceTo(a, b) { return Vec3.len(Vec3.sub(a, b, _tmp3)); }
  static clone(a) { return new Vec3(a.e[0], a.e[1], a.e[2]); }
  static fromArray(arr, offset = 0, out = new Vec3()) {
    out.e[0]=arr[offset]; out.e[1]=arr[offset+1]; out.e[2]=arr[offset+2]; return out;
  }
  static toArray(a, arr = [], offset = 0) {
    arr[offset]=a.e[0]; arr[offset+1]=a.e[1]; arr[offset+2]=a.e[2]; return arr;
  }
  /**
   * Transform a Vec3 position by a Mat4 (w=1).
   * @param {Vec3} a
   * @param {Mat4} m
   * @param {Vec3} [out]
   */
  static transformMat4(a, m, out = new Vec3()) {
    const e=m.e, x=a.e[0], y=a.e[1], z=a.e[2];
    const w = e[3]*x + e[7]*y + e[11]*z + e[15] || 1;
    out.e[0]=(e[0]*x + e[4]*y + e[8] *z + e[12])/w;
    out.e[1]=(e[1]*x + e[5]*y + e[9] *z + e[13])/w;
    out.e[2]=(e[2]*x + e[6]*y + e[10]*z + e[14])/w;
    return out;
  }
  /**
   * Transform a Vec3 direction by a Mat3 (no translation).
   * @param {Vec3} a
   * @param {Mat3} m
   * @param {Vec3} [out]
   */
  static transformMat3(a, m, out = new Vec3()) {
    const e=m.e, x=a.e[0], y=a.e[1], z=a.e[2];
    out.e[0]=e[0]*x + e[3]*y + e[6]*z;
    out.e[1]=e[1]*x + e[4]*y + e[7]*z;
    out.e[2]=e[2]*x + e[5]*y + e[8]*z;
    return out;
  }
}

// Scratch vector used inside static methods to avoid allocations.
const _tmp3 = new Vec3();
