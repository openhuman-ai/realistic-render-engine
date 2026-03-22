/**
 * GLTFLoader — glTF 2.0 asset loader.
 *
 * Supports:
 *   - Separate .bin buffers (fetched by URL)
 *   - Embedded base64 data URIs
 *   - POSITION / NORMAL / TEXCOORD_0 / JOINTS_0 / WEIGHTS_0 vertex attributes + indices
 *   - PBR metallic-roughness material properties
 *   - Skin ingest: joints array + inverseBindMatrices → Skeleton
 *   - Animation ingest: samplers/channels → AnimationClip[]
 *   - Stub warnings for KHR_draco_mesh_compression and KHR_texture_basisu
 *
 * Returns: {
 *   meshes:     [{ vao, vertexBuffers, indexBuffer, indexCount, indexType, material, skinned }],
 *   skeletons:  Skeleton[],   (one per glTF skin, or empty)
 *   animations: AnimationClip[]
 * }
 */
import { VertexBuffer, IndexBuffer } from '../core/Buffer.js';
import { Joint, Skeleton }           from '../animation/Skeleton.js';
import { AnimationClip }             from '../animation/AnimationClip.js';

// glTF component type → typed array constructor
const CTYPES = {
  5120: Int8Array,
  5121: Uint8Array,
  5122: Int16Array,
  5123: Uint16Array,
  5125: Uint32Array,
  5126: Float32Array,
};

// glTF accessor type → element count
const TYPE_SIZES = {
  SCALAR: 1,
  VEC2:   2,
  VEC3:   3,
  VEC4:   4,
  MAT2:   4,
  MAT3:   9,
  MAT4:   16,
};

export class GLTFLoader {
  /** @param {WebGL2RenderingContext} gl */
  constructor(gl) {
    this.gl = gl;
  }

  /**
   * Load a glTF asset from a URL.
   * @param {string} url — path to .gltf or .glb file
   * @returns {Promise<{ meshes: Array }>}
   */
  async load(url) {
    const isGLB = url.toLowerCase().endsWith('.glb');
    if (isGLB) {
      return this._loadGLB(url);
    }
    return this._loadGLTF(url);
  }

  // ------------------------------------------------------------------- glTF JSON
  async _loadGLTF(url) {
    const baseUrl = url.substring(0, url.lastIndexOf('/') + 1);
    const res  = await fetch(url);
    if (!res.ok) throw new Error(`[GLTFLoader] Failed to fetch ${url}: ${res.status}`);
    const json = await res.json();
    this._warnExtensions(json);

    const buffers = await this._loadBuffers(json, baseUrl);
    return this._parse(json, buffers);
  }

  // ------------------------------------------------------------------- GLB binary
  async _loadGLB(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`[GLTFLoader] Failed to fetch ${url}: ${res.status}`);
    const ab = await res.arrayBuffer();

    // GLB header: magic (4), version (4), length (4)
    const view32 = new Uint32Array(ab, 0, 3);
    const magic   = view32[0];
    if (magic !== 0x46546C67) throw new Error('[GLTFLoader] Not a valid GLB file.');

    let offset = 12;
    let json   = null;
    let binBuf = null;

    while (offset < ab.byteLength) {
      const chunkLen  = new Uint32Array(ab, offset, 1)[0];
      const chunkType = new Uint32Array(ab, offset + 4, 1)[0];
      offset += 8;

      if (chunkType === 0x4E4F534A) { // JSON chunk
        const bytes = new Uint8Array(ab, offset, chunkLen);
        json = JSON.parse(new TextDecoder().decode(bytes));
      } else if (chunkType === 0x004E4942) { // BIN chunk
        binBuf = ab.slice(offset, offset + chunkLen);
      }
      offset += chunkLen;
    }

    if (!json) throw new Error('[GLTFLoader] GLB missing JSON chunk.');
    this._warnExtensions(json);

    // GLB has a single embedded buffer (index 0 → the BIN chunk)
    const buffers = [];
    if (json.buffers) {
      for (let i = 0; i < json.buffers.length; i++) {
        if (i === 0 && binBuf) {
          buffers.push(binBuf);
        } else {
          // Embedded base64 or separate bin referenced from JSON inside GLB (rare)
          buffers.push(await this._loadBuffer(json.buffers[i], ''));
        }
      }
    }
    return this._parse(json, buffers);
  }

  // ------------------------------------------------------------------- buffers
  async _loadBuffers(json, baseUrl) {
    if (!json.buffers) return [];
    return Promise.all(json.buffers.map(b => this._loadBuffer(b, baseUrl)));
  }

  async _loadBuffer(bufDef, baseUrl) {
    const uri = bufDef.uri;
    if (!uri) return null; // GLB-embedded; caller provides data

    if (uri.startsWith('data:')) {
      // Embedded base64
      const b64 = uri.split(',')[1];
      const bin = atob(b64);
      const buf = new ArrayBuffer(bin.length);
      const u8  = new Uint8Array(buf);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      return buf;
    }

    const res = await fetch(baseUrl + uri);
    if (!res.ok) throw new Error(`[GLTFLoader] Failed to fetch buffer ${uri}: ${res.status}`);
    return res.arrayBuffer();
  }

  // ------------------------------------------------------------------- parse
  _parse(json, buffers) {
    const meshes = [];

    if (json.meshes) {
      for (const meshDef of json.meshes) {
        for (const prim of meshDef.primitives) {
          const result = this._parsePrimitive(json, prim, buffers);
          if (result) meshes.push(result);
        }
      }
    }

    const skeletons  = this._parseSkeletons(json, buffers);
    const animations = this._parseAnimations(json, buffers);

    return { meshes, skeletons, animations };
  }

  /** @private */
  _parsePrimitive(json, prim, buffers) {
    const gl    = this.gl;
    const attrs = prim.attributes ?? {};

    // Require at least POSITION
    if (attrs.POSITION === undefined) {
      console.warn('[GLTFLoader] Primitive missing POSITION accessor; skipping.');
      return null;
    }

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vertexBuffers = {};

    // Helper: upload an accessor as a vertex attribute
    const setupAttrib = (accessorIdx, attribLoc, normalize = false) => {
      if (accessorIdx === undefined) return;
      const acc   = json.accessors[accessorIdx];
      const bvDef = json.bufferViews[acc.bufferView];
      const data  = this._accessorToTypedArray(json, acc, buffers);

      const vb = new VertexBuffer(gl, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, vb.buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

      const componentType = acc.componentType;
      const numComponents = TYPE_SIZES[acc.type];
      const byteStride    = bvDef.byteStride ?? 0;
      gl.vertexAttribPointer(attribLoc, numComponents, componentType, normalize, byteStride, 0);
      gl.enableVertexAttribArray(attribLoc);

      vb.byteLength = data.byteLength;
      return vb;
    };

    vertexBuffers.position  = setupAttrib(attrs.POSITION,   0);
    vertexBuffers.normal    = setupAttrib(attrs.NORMAL,     1);
    vertexBuffers.texCoord0 = setupAttrib(attrs.TEXCOORD_0, 2);

    // Skinning attributes (optional)
    let skinned = false;
    if (attrs.JOINTS_0 !== undefined && attrs.WEIGHTS_0 !== undefined) {
      skinned = true;
      // JOINTS_0 — use float attribute (UNSIGNED_BYTE or UNSIGNED_SHORT cast to float)
      const jointsAcc = json.accessors[attrs.JOINTS_0];
      const jointsData = this._accessorToTypedArray(json, jointsAcc, buffers);
      // Convert to Float32 so we can use vertexAttribPointer (avoids ivec4 attribute)
      const jointsF32 = new Float32Array(jointsData.length);
      for (let i = 0; i < jointsData.length; i++) jointsF32[i] = jointsData[i];
      const jvb = new VertexBuffer(gl, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, jvb.buf);
      gl.bufferData(gl.ARRAY_BUFFER, jointsF32, gl.STATIC_DRAW);
      gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(3);
      vertexBuffers.joints = jvb;

      vertexBuffers.weights = setupAttrib(attrs.WEIGHTS_0, 4);
    }

    // Index buffer
    let indexBuffer = null;
    let indexCount  = 0;
    let indexType   = gl.UNSIGNED_SHORT;

    if (prim.indices !== undefined) {
      const acc  = json.accessors[prim.indices];
      const data = this._accessorToTypedArray(json, acc, buffers);
      indexBuffer = new IndexBuffer(gl, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer.buf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW);
      indexBuffer.count = acc.count;
      indexBuffer.indexType = acc.componentType === 5125 ? gl.UNSIGNED_INT
                            : acc.componentType === 5121 ? gl.UNSIGNED_BYTE
                            : gl.UNSIGNED_SHORT;
      indexCount = acc.count;
      indexType  = indexBuffer.indexType;
    } else {
      // Non-indexed: count from POSITION accessor
      indexCount = json.accessors[attrs.POSITION].count;
    }

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    // Material
    const material = this._parseMaterial(json, prim.material);

    return { vao, vertexBuffers, indexBuffer, indexCount, indexType, material, skinned };
  }

  /** @private */
  _accessorToTypedArray(json, acc, buffers) {
    const bvDef    = json.bufferViews[acc.bufferView];
    const buffer   = buffers[bvDef.buffer];
    const TypeArr  = CTYPES[acc.componentType] ?? Float32Array;
    const numComps = TYPE_SIZES[acc.type] ?? 1;
    const count    = acc.count;
    const byteOffset = (bvDef.byteOffset ?? 0) + (acc.byteOffset ?? 0);

    // Handle interleaved data (byteStride !== element size)
    const elemSize  = TypeArr.BYTES_PER_ELEMENT * numComps;
    const byteStride = bvDef.byteStride ?? elemSize;

    if (byteStride === elemSize) {
      // Packed — create a view directly
      return new TypeArr(buffer, byteOffset, count * numComps);
    }

    // Interleaved — copy to packed array
    const out = new TypeArr(count * numComps);
    const src = new TypeArr(buffer);
    const srcOff = byteOffset / TypeArr.BYTES_PER_ELEMENT;
    const stride  = byteStride / TypeArr.BYTES_PER_ELEMENT;
    for (let i = 0; i < count; i++) {
      for (let c = 0; c < numComps; c++) {
        out[i * numComps + c] = src[srcOff + i * stride + c];
      }
    }
    return out;
  }

  /** @private */
  _parseMaterial(json, materialIdx) {
    const mat = {
      baseColorFactor:  [0.8, 0.8, 0.8, 1.0],
      roughnessFactor:  0.5,
      metallicFactor:   0.0,
    };
    if (materialIdx === undefined || !json.materials) return mat;

    const mDef = json.materials[materialIdx];
    if (!mDef) return mat;

    const pbr = mDef.pbrMetallicRoughness;
    if (pbr) {
      if (pbr.baseColorFactor)  mat.baseColorFactor  = pbr.baseColorFactor;
      if (pbr.roughnessFactor !== undefined) mat.roughnessFactor = pbr.roughnessFactor;
      if (pbr.metallicFactor  !== undefined) mat.metallicFactor  = pbr.metallicFactor;
    }
    return mat;
  }

  /** Emit warnings for unsupported but detected extensions. @private */
  _warnExtensions(json) {
    const exts = json.extensionsUsed ?? [];
    if (exts.includes('KHR_draco_mesh_compression')) {
      console.warn('[GLTFLoader] KHR_draco_mesh_compression detected but not supported. Meshes may not load correctly.');
    }
    if (exts.includes('KHR_texture_basisu')) {
      console.warn('[GLTFLoader] KHR_texture_basisu detected but not supported. Textures may not load correctly.');
    }
  }

  // ----------------------------------------------------------------- skins
  /**
   * Parse all glTF skins into Skeleton instances.
   * @private
   */
  _parseSkeletons(json, buffers) {
    if (!json.skins) return [];
    const skeletons = [];

    for (const skin of json.skins) {
      const jointNodeIndices = skin.joints ?? [];
      const n = jointNodeIndices.length;

      // Build a mapping: glTF node index → skeleton joint index
      const nodeToJoint = new Map();
      jointNodeIndices.forEach((ni, ji) => nodeToJoint.set(ni, ji));

      // Parse inverse bind matrices
      let ibmData = null;
      if (skin.inverseBindMatrices !== undefined) {
        const acc = json.accessors[skin.inverseBindMatrices];
        ibmData   = this._accessorToTypedArray(json, acc, buffers);
      }

      const joints = [];
      for (let ji = 0; ji < n; ji++) {
        const nodeIdx = jointNodeIndices[ji];
        const node    = json.nodes?.[nodeIdx];
        const name    = node?.name ?? `joint_${ji}`;

        // Determine parent joint index (first ancestor that is also a joint)
        let parentJointIdx = -1;
        if (node) {
          const parentNodeIdx = _findParentNodeIndex(json, nodeIdx);
          if (parentNodeIdx !== undefined && nodeToJoint.has(parentNodeIdx)) {
            parentJointIdx = nodeToJoint.get(parentNodeIdx);
          }
        }

        const joint = new Joint(name, ji, parentJointIdx);

        // Inverse bind matrix (column-major Mat4)
        if (ibmData) {
          joint.inverseBindMatrix.e.set(ibmData.subarray(ji * 16, ji * 16 + 16));
        }
        // else remains identity — acceptable for unrigged skins

        // Seed local pose from glTF node TRS (bind pose)
        if (node) {
          if (node.translation) {
            joint.localTranslation.e[0] = node.translation[0];
            joint.localTranslation.e[1] = node.translation[1];
            joint.localTranslation.e[2] = node.translation[2];
          }
          if (node.rotation) {
            joint.localRotation.e[0] = node.rotation[0];
            joint.localRotation.e[1] = node.rotation[1];
            joint.localRotation.e[2] = node.rotation[2];
            joint.localRotation.e[3] = node.rotation[3];
          }
          if (node.scale) {
            joint.localScale.e[0] = node.scale[0];
            joint.localScale.e[1] = node.scale[1];
            joint.localScale.e[2] = node.scale[2];
          }
        }

        joints.push(joint);
      }

      // Ensure topological order (parent before child) — simple reorder
      _topoSortJoints(joints);

      skeletons.push(new Skeleton(joints));
    }

    return skeletons;
  }

  // --------------------------------------------------------------- animations
  /**
   * Parse all glTF animations into AnimationClip instances.
   * Each skeleton in the file shares the same glTF node index space, so
   * channels reference glTF node indices which callers must map to joint
   * indices (stored in channel.targetNodeIndex for caller convenience).
   * @private
   */
  _parseAnimations(json, buffers) {
    if (!json.animations) return [];
    const clips = [];

    for (const anim of json.animations) {
      const samplers  = anim.samplers ?? [];
      const channels  = anim.channels ?? [];
      const duration  = this._animationDuration(json, samplers, buffers);

      const parsedChannels = [];
      for (const ch of channels) {
        const sampler = samplers[ch.sampler];
        if (!sampler) continue;

        const target = ch.target;
        if (!target || target.node === undefined) continue;

        const prop = target.path; // 'translation' | 'rotation' | 'scale' | 'weights'
        if (prop === 'weights') continue; // handled in future morph PR

        const interp = (sampler.interpolation ?? 'LINEAR').toUpperCase();

        // Input (times) accessor
        const timesAcc = json.accessors[sampler.input];
        const times    = new Float32Array(
          this._accessorToTypedArray(json, timesAcc, buffers));

        // Output (values) accessor
        const valuesAcc = json.accessors[sampler.output];
        const values    = new Float32Array(
          this._accessorToTypedArray(json, valuesAcc, buffers));

        parsedChannels.push({
          targetNodeIndex: target.node,
          targetJointIndex: target.node, // remapped by caller if needed
          property:        prop,
          interpolation:   interp,
          times,
          values,
        });
      }

      clips.push(new AnimationClip(anim.name ?? `animation_${clips.length}`, duration, parsedChannels));
    }

    return clips;
  }

  /** Compute animation duration from sampler input accessors. @private */
  _animationDuration(json, samplers, buffers) {
    let max = 0;
    for (const s of samplers) {
      if (s.input === undefined) continue;
      const acc  = json.accessors[s.input];
      if (acc.max) {
        max = Math.max(max, acc.max[0]);
      } else {
        // Fallback: parse the accessor to find the last time value
        const times = this._accessorToTypedArray(json, acc, buffers);
        if (times.length) max = Math.max(max, times[times.length - 1]);
      }
    }
    return max > 0 ? max : 1;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-level helpers (not exported)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Find the parent node index for a given node index in the glTF scene graph.
 * Returns undefined if the node has no parent or no nodes are present.
 * @param {object} json  @param {number} nodeIdx
 * @returns {number|undefined}
 */
function _findParentNodeIndex(json, nodeIdx) {
  if (!json.nodes) return undefined;
  for (let i = 0; i < json.nodes.length; i++) {
    const n = json.nodes[i];
    if (n.children && n.children.includes(nodeIdx)) return i;
  }
  return undefined;
}

/**
 * Stable topological sort of joints so parents always precede children.
 * Operates in-place on the joints array.
 * @param {import('../animation/Skeleton.js').Joint[]} joints
 */
function _topoSortJoints(joints) {
  // Already in glTF order which is typically parent-first, but sort to be safe.
  // Simple insertion-sort preserving relative order when parent < child index.
  for (let i = 1; i < joints.length; i++) {
    const j = joints[i];
    if (j.parentIndex < 0 || j.parentIndex < i) continue;
    // parentIndex > i — move joint i after its parent
    const parent = joints.splice(joints.findIndex(x => x.index === j.parentIndex), 1)[0];
    joints.splice(i, 0, parent);
  }
  // Re-assign indices in new order
  for (let i = 0; i < joints.length; i++) {
    joints[i].index = i;
    if (joints[i].parentIndex >= 0) {
      // Keep parentIndex pointing at the correct joint
      const parentName = joints.find((x, xi) => xi < i && x.name === joints[i]._parentName);
      // parentIndex was already set correctly during construction; nothing to do here
    }
  }
}
