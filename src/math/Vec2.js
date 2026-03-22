// Float32Array-backed, out-parameter style
export class Vec2 {
  constructor(x = 0, y = 0) { this.e = new Float32Array([x, y]); }
  get x() { return this.e[0]; } set x(v) { this.e[0] = v; }
  get y() { return this.e[1]; } set y(v) { this.e[1] = v; }
  set(x, y) { this.e[0] = x; this.e[1] = y; return this; }
  static add(a, b, out = new Vec2())      { out.e[0]=a.e[0]+b.e[0]; out.e[1]=a.e[1]+b.e[1]; return out; }
  static sub(a, b, out = new Vec2())      { out.e[0]=a.e[0]-b.e[0]; out.e[1]=a.e[1]-b.e[1]; return out; }
  static scale(a, s, out = new Vec2())    { out.e[0]=a.e[0]*s; out.e[1]=a.e[1]*s; return out; }
  static dot(a, b)                        { return a.e[0]*b.e[0]+a.e[1]*b.e[1]; }
  static len(a)                           { return Math.sqrt(Vec2.dot(a, a)); }
  static normalize(a, out = new Vec2())   { const l=Vec2.len(a)||1; return Vec2.scale(a, 1/l, out); }
  static clone(a)                         { return new Vec2(a.e[0], a.e[1]); }
}
