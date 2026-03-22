/**
 * Character — wraps loaded mesh data and provides animation/morph stubs.
 */
import { Node } from './Node.js';

export class Character {
  /**
   * @param {{ meshes: Array }} gltfData — result from GLTFLoader.load()
   */
  constructor(gltfData) {
    this.node = new Node('Character');
    this._meshes = gltfData?.meshes ?? [];

    // Attach mesh primitives as children of the root node
    for (const meshData of this._meshes) {
      const child = new Node('mesh');
      child.mesh = meshData;
      this.node.addChild(child);
    }

    // Morph weight map — stub
    this._morphWeights = new Map();
    // Animation state — stub
    this._currentAnimation = null;
  }

  // --------------------------------------------------- animation stubs
  /**
   * Set a blend-shape / morph target weight.
   * @param {string} name   — morph target name
   * @param {number} weight — [0, 1]
   */
  setMorphWeight(name, weight) {
    this._morphWeights.set(name, Math.max(0, Math.min(1, weight)));
    // TODO: update GPU morph-target buffer
  }

  /**
   * Play a named animation clip.
   * @param {string} name
   */
  playAnimation(name) {
    this._currentAnimation = name;
    // TODO: start animation playback via AnimationSystem
  }

  // --------------------------------------------------- rendering
  /**
   * Render this character through the provided renderer.
   * @param {import('../renderer/ForwardRenderer.js').ForwardRenderer} renderer
   * @param {import('./Camera.js').Camera} camera
   * @param {import('./Light.js').Light[]} lights
   */
  render(renderer, camera, lights = []) {
    this.node.updateWorldMatrix(null);
    renderer.render(this.node.children, camera, lights);
  }
}
