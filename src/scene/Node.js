/**
 * Node — scene graph node with TRS transform.
 */
import { Vec3 } from '../math/Vec3.js';
import { Quat } from '../math/Quat.js';
import { Mat4 } from '../math/Mat4.js';

export class Node {
  constructor(name = '') {
    this.name     = name;
    this.position = new Vec3(0, 0, 0);
    this.rotation = new Quat(0, 0, 0, 1);
    this.scale    = new Vec3(1, 1, 1);

    this.localMatrix = new Mat4();
    this.worldMatrix = new Mat4();

    /** @type {Node|null} */
    this.parent   = null;
    /** @type {Node[]} */
    this.children = [];

    /** Optional mesh attachment { vao, indexBuffer, indexCount, indexType, material, vertexCount } */
    this.mesh = null;
  }

  // ----------------------------------------------------------- hierarchy
  /** @param {Node} node */
  addChild(node) {
    if (node.parent) node.parent.removeChild(node);
    node.parent = this;
    this.children.push(node);
  }

  /** @param {Node} node */
  removeChild(node) {
    const idx = this.children.indexOf(node);
    if (idx !== -1) {
      this.children.splice(idx, 1);
      node.parent = null;
    }
  }

  // ----------------------------------------------------------- matrices
  /** Recompute localMatrix from position / rotation / scale. */
  updateMatrix() {
    Mat4.fromRotationTranslationScale(this.rotation, this.position, this.scale, this.localMatrix);
  }

  /**
   * Recursively update world matrices.
   * @param {Mat4|null} parentMatrix
   */
  updateWorldMatrix(parentMatrix = null) {
    this.updateMatrix();
    if (parentMatrix) {
      Mat4.multiply(parentMatrix, this.localMatrix, this.worldMatrix);
    } else {
      this.worldMatrix.e.set(this.localMatrix.e);
    }
    for (const child of this.children) {
      child.updateWorldMatrix(this.worldMatrix);
    }
  }
}
