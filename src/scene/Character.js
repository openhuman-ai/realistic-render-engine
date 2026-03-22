/**
 * Character — wraps loaded mesh data, skeleton, and animation system.
 */
import { Node }           from './Node.js';
import { GPUSkinning }    from '../animation/GPUSkinning.js';
import { AnimationGraph } from '../animation/AnimationGraph.js';

export class Character {
  /**
   * @param {{ meshes: Array, skeletons?: Array, animations?: Array }} gltfData
   * @param {WebGL2RenderingContext} gl
   */
  constructor(gltfData, gl = null) {
    this.node    = new Node('Character');
    this._meshes = gltfData?.meshes ?? [];
    this._gl     = gl;

    // Attach mesh primitives as children of the root node
    for (const meshData of this._meshes) {
      const child = new Node('mesh');
      child.mesh = meshData;
      this.node.addChild(child);
    }

    // Skeleton — first skin, if any
    this._skeleton     = gltfData?.skeletons?.[0] ?? null;
    this._animations   = gltfData?.animations ?? [];
    this._gpuSkinning  = null;
    this._animGraph    = null;

    if (this._skeleton && gl) {
      this._gpuSkinning = new GPUSkinning(gl, this._skeleton.joints.length);
      this._animGraph   = new AnimationGraph(this._skeleton);
    }

    // Morph weight map — stub for future morph system
    this._morphWeights = new Map();
  }

  // ────────────────────────────────────────────────── animation API
  /**
   * Register an AnimationClip (loaded from glTF or procedurally built).
   * @param {string} name
   * @param {import('../animation/AnimationClip.js').AnimationClip} clip
   */
  addAnimation(name, clip) {
    if (!this._animGraph) {
      console.warn('[Character] No skeleton — cannot add animation.');
      return;
    }
    this._animGraph.addState(name, clip);
  }

  /**
   * Add a transition between two animation states.
   * @param {string} from @param {string} to @param {string} param @param {number} duration
   */
  addTransition(from, to, param, duration) {
    this._animGraph?.addTransition(from, to, param, duration);
  }

  /**
   * Play a named animation clip.
   * @param {string} name
   */
  playAnimation(name) {
    if (!this._animGraph) return;
    this._animGraph.play(name);
  }

  /** @param {string} param @param {boolean} value */
  setBool(param, value) {
    this._animGraph?.setBool(param, value);
  }

  /** @param {string} param @param {number} value */
  setFloat(param, value) {
    this._animGraph?.setFloat(param, value);
  }

  /**
   * Set a blend-shape / morph target weight.
   * @param {string} name   — morph target name
   * @param {number} weight — [0, 1]
   */
  setMorphWeight(name, weight) {
    this._morphWeights.set(name, Math.max(0, Math.min(1, weight)));
    // TODO: update GPU morph-target buffer (next PR)
  }

  // ────────────────────────────────────────────────── update / render
  /**
   * Advance the animation system by deltaTime seconds.
   * Must be called every frame before render().
   * @param {number} deltaTime — seconds
   */
  update(deltaTime) {
    if (!this._animGraph || !this._gpuSkinning) return;
    this._animGraph.update(deltaTime);
    this._gpuSkinning.update(this._skeleton);
  }

  /**
   * Render this character through the provided renderer.
   * @param {import('../renderer/ForwardRenderer.js').ForwardRenderer} renderer
   * @param {import('./Camera.js').Camera} camera
   * @param {import('./Light.js').Light[]} lights
   */
  render(renderer, camera, lights = []) {
    this.node.updateWorldMatrix(null);
    renderer.render(this.node.children, camera, lights, this._gpuSkinning);
  }

  destroy() {
    this._gpuSkinning?.destroy();
    this._gpuSkinning = null;
  }
}
