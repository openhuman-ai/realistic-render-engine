/**
 * Light — scene light source (directional, point, or ambient).
 */
import { Vec3 } from '../math/Vec3.js';

export class Light {
  /**
   * @param {'directional'|'point'|'ambient'} type
   */
  constructor(type = 'directional') {
    this.type      = type;
    this.color     = new Vec3(1, 1, 1);
    this.intensity = 1.0;

    // Directional light
    this.direction = new Vec3(0.5, -1.0, 0.5);

    // Point light
    this.position  = new Vec3(0, 0, 0);
  }

  /** @returns {Float32Array} 3-element direction array (normalised) */
  getDirectionArray() {
    // Normalise on the way out, avoid mutation
    const e = this.direction.e;
    const len = Math.sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]) || 1;
    _dirOut[0] = e[0]/len;
    _dirOut[1] = e[1]/len;
    _dirOut[2] = e[2]/len;
    return _dirOut;
  }

  /** @returns {Float32Array} 3-element RGB array scaled by intensity */
  getColorArray() {
    const e = this.color.e;
    _colOut[0] = e[0] * this.intensity;
    _colOut[1] = e[1] * this.intensity;
    _colOut[2] = e[2] * this.intensity;
    return _colOut;
  }
}

// Reusable output arrays — avoids per-call allocation in hot render loops
const _dirOut = new Float32Array(3);
const _colOut = new Float32Array(3);
