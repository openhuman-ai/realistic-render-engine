// Float32Array-backed Vec4 with out-parameter style static methods.
export class Vec4 {
  constructor(x = 0, y = 0, z = 0, w = 0) { this.e = new Float32Array([x, y, z, w]); }
  get x() { return this.e[0]; } set x(v) { this.e[0] = v; }
  get y() { return this.e[1]; } set y(v) { this.e[1] = v; }
  get z() { return this.e[2]; } set z(v) { this.e[2] = v; }
  get w() { return this.e[3]; } set w(v) { this.e[3] = v; }
  set(x, y, z, w) { this.e[0]=x; this.e[1]=y; this.e[2]=z; this.e[3]=w; return this; }

  static add(a, b, out = new Vec4()) {
    out.e[0]=a.e[0]+b.e[0]; out.e[1]=a.e[1]+b.e[1];
    out.e[2]=a.e[2]+b.e[2]; out.e[3]=a.e[3]+b.e[3]; return out;
  }
  static sub(a, b, out = new Vec4()) {
    out.e[0]=a.e[0]-b.e[0]; out.e[1]=a.e[1]-b.e[1];
    out.e[2]=a.e[2]-b.e[2]; out.e[3]=a.e[3]-b.e[3]; return out;
  }
  static scale(a, s, out = new Vec4()) {
    out.e[0]=a.e[0]*s; out.e[1]=a.e[1]*s; out.e[2]=a.e[2]*s; out.e[3]=a.e[3]*s; return out;
  }
  static dot(a, b) {
    return a.e[0]*b.e[0] + a.e[1]*b.e[1] + a.e[2]*b.e[2] + a.e[3]*b.e[3];
  }
  static len(a) { return Math.sqrt(Vec4.dot(a, a)); }
  static normalize(a, out = new Vec4()) {
    const l = Vec4.len(a) || 1;
    return Vec4.scale(a, 1/l, out);
  }
  static lerp(a, b, t, out = new Vec4()) {
    const e=out.e, ae=a.e, be=b.e;
    e[0]=ae[0]+(be[0]-ae[0])*t; e[1]=ae[1]+(be[1]-ae[1])*t;
    e[2]=ae[2]+(be[2]-ae[2])*t; e[3]=ae[3]+(be[3]-ae[3])*t;
    return out;
  }
  static clone(a) { return new Vec4(a.e[0], a.e[1], a.e[2], a.e[3]); }
  static fromArray(arr, offset = 0, out = new Vec4()) {
    out.e[0]=arr[offset]; out.e[1]=arr[offset+1];
    out.e[2]=arr[offset+2]; out.e[3]=arr[offset+3]; return out;
  }
  static toArray(a, arr = [], offset = 0) {
    arr[offset]=a.e[0]; arr[offset+1]=a.e[1];
    arr[offset+2]=a.e[2]; arr[offset+3]=a.e[3]; return arr;
  }
  /**
   * Transform by a Mat4 (a is treated as a full vec4).
   * @param {Vec4} a
   * @param {Mat4} m — column-major Float32Array
   * @param {Vec4} [out]
   */
  static transformMat4(a, m, out = new Vec4()) {
    const e=m.e, x=a.e[0], y=a.e[1], z=a.e[2], w=a.e[3];
    out.e[0]=e[0]*x + e[4]*y + e[8] *z + e[12]*w;
    out.e[1]=e[1]*x + e[5]*y + e[9] *z + e[13]*w;
    out.e[2]=e[2]*x + e[6]*y + e[10]*z + e[14]*w;
    out.e[3]=e[3]*x + e[7]*y + e[11]*z + e[15]*w;
    return out;
  }
}
