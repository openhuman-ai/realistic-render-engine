/**
 * GLTFLoader — glTF 2.0 asset loader.
 *
 * Supports:
 *   - Separate .bin buffers (fetched by URL)
 *   - Embedded base64 data URIs
 *   - POSITION / NORMAL / TEXCOORD_0 vertex attributes + indices
 *   - PBR metallic-roughness material properties
 *   - Stub warnings for KHR_draco_mesh_compression and KHR_texture_basisu
 *
 * Returns: { meshes: [{ vao, vertexBuffers, indexBuffer, indexCount, indexType, material }] }
 */
import { VertexBuffer, IndexBuffer } from '../core/Buffer.js';

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
    const gl     = this.gl;
    const meshes = [];

    if (!json.meshes) return { meshes };

    for (const meshDef of json.meshes) {
      for (const prim of meshDef.primitives) {
        const result = this._parsePrimitive(json, prim, buffers);
        if (result) meshes.push(result);
      }
    }
    return { meshes };
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

    return { vao, vertexBuffers, indexBuffer, indexCount, indexType, material };
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
}
