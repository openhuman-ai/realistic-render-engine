/**
 * Camera — perspective camera with optional orbit controls.
 */
import { Node }  from './Node.js';
import { Vec3 }  from '../math/Vec3.js';
import { Mat4 }  from '../math/Mat4.js';

const DEG2RAD = Math.PI / 180;

export class Camera extends Node {
  /**
   * @param {{fov?: number, aspect?: number, near?: number, far?: number}} [opts]
   */
  constructor(opts = {}) {
    super('Camera');
    this.fov    = opts.fov    ?? 60;
    this.aspect = opts.aspect ?? 1;
    this.near   = opts.near   ?? 0.1;
    this.far    = opts.far    ?? 1000;

    this.projectionMatrix = new Mat4();
    this.viewMatrix       = new Mat4();

    // Orbit state
    this._orbitTarget = new Vec3(0, 0, 0);
    this._theta  = 0;           // azimuth (radians)
    this._phi    = Math.PI / 4; // polar   (radians)
    this._radius = 3;
    this._orbitActive = false;
    this._orbitCanvas = null;
    this._orbitHandlers = {};

    this.updateProjection();
    this._updateFromSpherical();
  }

  // -------------------------------------------------------- projection
  updateProjection() {
    Mat4.perspective(this.fov * DEG2RAD, this.aspect, this.near, this.far, this.projectionMatrix);
  }

  /** Recompute view matrix from current world matrix (inverse of world). */
  updateView() {
    Mat4.invert(this.worldMatrix, this.viewMatrix);
  }

  /** Rebuild world matrix from current spherical orbit coords, then recompute view. */
  _updateFromSpherical() {
    const sinPhi = Math.sin(this._phi);
    const cosPhi = Math.cos(this._phi);
    const x = this._orbitTarget.e[0] + this._radius * sinPhi * Math.sin(this._theta);
    const y = this._orbitTarget.e[1] + this._radius * cosPhi;
    const z = this._orbitTarget.e[2] + this._radius * sinPhi * Math.cos(this._theta);
    this.position.set(x, y, z);
    Mat4.lookAt(this.position, this._orbitTarget, _up, this.worldMatrix);
    Mat4.invert(this.worldMatrix, this.viewMatrix);
  }

  // -------------------------------------------------------- controls
  /** Point camera at a world-space target. */
  lookAt(target) {
    this._orbitTarget.e.set(target.e);
    this._updateFromSpherical();
  }

  setPosition(x, y, z) {
    this.position.set(x, y, z);
    this.updateMatrix();
    this.updateView();
  }

  setFOV(deg) {
    this.fov = deg;
    this.updateProjection();
  }

  // --------------------------------------------------- orbit controls
  /**
   * Attach mouse / touch orbit controls to a canvas.
   * @param {HTMLCanvasElement} canvas
   */
  enableOrbit(canvas) {
    if (this._orbitCanvas) this.disableOrbit();
    this._orbitCanvas = canvas;

    let lastX = 0, lastY = 0, dragging = false;

    const onMouseDown = (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    };
    const onMouseMove = (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      this._theta -= dx * 0.005;
      this._phi    = Math.max(0.05, Math.min(Math.PI - 0.05, this._phi + dy * 0.005));
      this._updateFromSpherical();
    };
    const onMouseUp = () => { dragging = false; };
    const onWheel   = (e) => {
      e.preventDefault();
      this._radius = Math.max(0.5, this._radius + e.deltaY * 0.01);
      this._updateFromSpherical();
    };

    // Touch support
    let prevTouchDist = 0;
    const onTouchStart = (e) => {
      if (e.touches.length === 1) {
        dragging = true;
        lastX = e.touches[0].clientX;
        lastY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        dragging = false;
        prevTouchDist = _touchDist(e.touches);
      }
    };
    const onTouchMove = (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && dragging) {
        const dx = e.touches[0].clientX - lastX;
        const dy = e.touches[0].clientY - lastY;
        lastX = e.touches[0].clientX;
        lastY = e.touches[0].clientY;
        this._theta -= dx * 0.005;
        this._phi    = Math.max(0.05, Math.min(Math.PI - 0.05, this._phi + dy * 0.005));
        this._updateFromSpherical();
      } else if (e.touches.length === 2) {
        const dist = _touchDist(e.touches);
        this._radius = Math.max(0.5, this._radius - (dist - prevTouchDist) * 0.01);
        prevTouchDist = dist;
        this._updateFromSpherical();
      }
    };
    const onTouchEnd = () => { dragging = false; };

    canvas.addEventListener('mousedown',  onMouseDown);
    canvas.addEventListener('mousemove',  onMouseMove);
    canvas.addEventListener('mouseup',    onMouseUp);
    canvas.addEventListener('mouseleave', onMouseUp);
    canvas.addEventListener('wheel',      onWheel,      { passive: false });
    canvas.addEventListener('touchstart', onTouchStart, { passive: false });
    canvas.addEventListener('touchmove',  onTouchMove,  { passive: false });
    canvas.addEventListener('touchend',   onTouchEnd);

    this._orbitHandlers = { onMouseDown, onMouseMove, onMouseUp, onWheel,
                            onTouchStart, onTouchMove, onTouchEnd };
  }

  disableOrbit() {
    const c = this._orbitCanvas;
    if (!c) return;
    const h = this._orbitHandlers;
    c.removeEventListener('mousedown',  h.onMouseDown);
    c.removeEventListener('mousemove',  h.onMouseMove);
    c.removeEventListener('mouseup',    h.onMouseUp);
    c.removeEventListener('mouseleave', h.onMouseUp);
    c.removeEventListener('wheel',      h.onWheel);
    c.removeEventListener('touchstart', h.onTouchStart);
    c.removeEventListener('touchmove',  h.onTouchMove);
    c.removeEventListener('touchend',   h.onTouchEnd);
    this._orbitCanvas   = null;
    this._orbitHandlers = {};
  }
}

// Shared up vector
const _up = new Vec3(0, 1, 0);

function _touchDist(touches) {
  const dx = touches[0].clientX - touches[1].clientX;
  const dy = touches[0].clientY - touches[1].clientY;
  return Math.sqrt(dx*dx + dy*dy);
}
