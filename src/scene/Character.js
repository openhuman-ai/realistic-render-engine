/**
 * Character — wraps loaded mesh data, skeleton, and animation system.
 */
import { Node } from "./Node.js"
import { GPUSkinning } from "../animation/GPUSkinning.js"
import { AnimationGraph } from "../animation/AnimationGraph.js"
import { MorphController, FACS_NAMES } from "../animation/MorphController.js"

export class Character {
  /**
   * @param {{ meshes: Array, skeletons?: Array, animations?: Array }} gltfData
   * @param {WebGL2RenderingContext} gl
   */
  constructor(gltfData, gl = null) {
    this.node = new Node("Character")
    this._meshes = gltfData?.meshes ?? []
    this._gl = gl

    // Attach mesh primitives as children of the root node
    for (const meshData of this._meshes) {
      const child = new Node("mesh")
      child.mesh = meshData
      this.node.addChild(child)
    }

    // Skeleton — first skin, if any
    this._skeleton = gltfData?.skeletons?.[0] ?? null
    this._animations = gltfData?.animations ?? []
    this._gpuSkinning = null
    this._animGraph = null

    if (this._skeleton && gl) {
      this._gpuSkinning = new GPUSkinning(gl, this._skeleton.joints.length)
      this._animGraph = new AnimationGraph(this._skeleton)
    }

    // Morph controller — always created when gl is available
    this._morphController = null
    if (gl) {
      this._morphController = this._initMorphController(gl, this._meshes)
    }
  }

  // ────────────────────────────────────────────────── animation API
  /**
   * Register an AnimationClip (loaded from glTF or procedurally built).
   * @param {string} name
   * @param {import('../animation/AnimationClip.js').AnimationClip} clip
   */
  addAnimation(name, clip) {
    if (!this._animGraph) {
      console.warn("[Character] No skeleton — cannot add animation.")
      return
    }
    this._animGraph.addState(name, clip)
  }

  /**
   * Add a transition between two animation states.
   * @param {string} from @param {string} to @param {string} param @param {number} duration
   */
  addTransition(from, to, param, duration) {
    this._animGraph?.addTransition(from, to, param, duration)
  }

  /**
   * Play a named animation clip.
   * @param {string} name
   */
  playAnimation(name) {
    if (!this._animGraph) return
    this._animGraph.play(name)
  }

  /** @param {string} param @param {boolean} value */
  setBool(param, value) {
    this._animGraph?.setBool(param, value)
  }

  /** @param {string} param @param {number} value */
  setFloat(param, value) {
    this._animGraph?.setFloat(param, value)
  }

  /**
   * Set a blend-shape / morph target weight.
   * @param {string} name   — morph target name
   * @param {number} weight — [0, 1]
   */
  setMorphWeight(name, weight) {
    if (this._morphController) {
      this._morphController.set(name, weight)
    }
  }

  // ────────────────────────────────────────────────── update / render
  /**
   * Advance the animation system by deltaTime seconds.
   * Must be called every frame before render().
   * @param {number} deltaTime — seconds
   */
  update(deltaTime) {
    if (!this._animGraph || !this._gpuSkinning) return
    this._animGraph.update(deltaTime)
    this._gpuSkinning.update(this._skeleton)
  }

  /**
   * Render this character through the provided renderer.
   * @param {import('../renderer/ForwardRenderer.js').ForwardRenderer} renderer
   * @param {import('./Camera.js').Camera} camera
   * @param {import('./Light.js').Light[]} lights
   */
  render(renderer, camera, lights = []) {
    this.node.updateWorldMatrix(null)
    renderer.render(this.node.children, camera, lights, this._gpuSkinning)
  }

  destroy() {
    this._gpuSkinning?.destroy()
    this._gpuSkinning = null
    this._morphController?.destroy()
    this._morphController = null
  }

  // ────────────────────────────────────────────────── private helpers

  /**
   * Initialise the MorphController from loaded mesh data.
   * Uses FACS names by default; overrides with names from the first mesh that has morph targets.
   * @private
   */
  _initMorphController(gl, meshes) {
    // Look for a mesh with morph target data
    let names = FACS_NAMES
    let vertexCount = 0
    let targetMesh = null

    for (const mesh of meshes) {
      if (mesh.morphTargets && mesh.morphTargets.length > 0) {
        targetMesh = mesh
        vertexCount = mesh.vertexCount ?? 0
        // If the asset provided target names and they match FACS, use them;
        // otherwise fall back to FACS_NAMES for the 52-slot controller
        if (mesh.morphTargetNames && mesh.morphTargetNames.length > 0) {
          names = mesh.morphTargetNames
        }
        break
      }
    }

    const ctrl = new MorphController(gl, names, vertexCount)

    // Upload per-morph deltas when mesh data is available
    if (targetMesh) {
      for (let i = 0; i < targetMesh.morphTargets.length; i++) {
        const { position, normal } = targetMesh.morphTargets[i]
        ctrl.uploadMorphDeltas(i, position, normal)
      }
    }

    return ctrl
  }
}
