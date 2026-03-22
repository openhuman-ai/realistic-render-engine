/**
 * ForwardRenderer — single-pass Phong/PBR-lite forward renderer.
 *
 * Renders a scene of Node → mesh objects using a built-in GLSL 300 es shader
 * with directional lighting, diffuse + specular Phong shading.
 */
import { Shader }     from '../core/Shader.js';
import { Mat4 }       from '../math/Mat4.js';
import { Mat3 }       from '../math/Mat3.js';

// ─────────────────────────────────────────────────────────────────────────────
// Inline GLSL
// ─────────────────────────────────────────────────────────────────────────────
const VERT_SRC = /* glsl */`#version 300 es
precision highp float;

in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;
uniform mat4 u_NormalMatrix;

out vec3 v_Normal;
out vec3 v_WorldPos;
out vec2 v_TexCoord;

void main() {
  vec4 worldPos = u_ModelMatrix * vec4(a_Position, 1.0);
  v_WorldPos = worldPos.xyz;
  v_Normal = normalize((u_NormalMatrix * vec4(a_Normal, 0.0)).xyz);
  v_TexCoord = a_TexCoord;
  gl_Position = u_ProjectionMatrix * u_ViewMatrix * worldPos;
}
`;

const FRAG_SRC = /* glsl */`#version 300 es
precision mediump float;

in vec3 v_Normal;
in vec3 v_WorldPos;
in vec2 v_TexCoord;

uniform vec3  u_LightDir;
uniform vec3  u_LightColor;
uniform vec3  u_BaseColor;
uniform float u_Roughness;
uniform float u_Metallic;
uniform vec3  u_CameraPos;

out vec4 fragColor;

void main() {
  vec3 N = normalize(v_Normal);
  vec3 L = normalize(-u_LightDir);
  vec3 V = normalize(u_CameraPos - v_WorldPos);
  vec3 H = normalize(L + V);

  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(N, H), 0.0), 32.0 * (1.0 - u_Roughness) + 1.0);

  vec3 ambient  = u_BaseColor * 0.15;
  vec3 diffuse  = u_BaseColor * u_LightColor * diff;
  vec3 specular = mix(vec3(0.04), u_BaseColor, u_Metallic) * u_LightColor * spec * (1.0 - u_Roughness);

  fragColor = vec4(ambient + diffuse + specular, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
export class ForwardRenderer {
  /**
   * @param {import('../core/GLContext.js').GLContext} glContext
   * @param {import('../core/StateCache.js').StateCache} stateCache
   */
  constructor(glContext, stateCache) {
    this.ctx   = glContext;
    this.gl    = glContext.gl;
    this.cache = stateCache;

    this.shader = new Shader(this.gl, VERT_SRC, FRAG_SRC);

    // Pre-allocated matrices to avoid per-frame heap allocation
    this._mvp         = new Mat4();
    this._normalMat   = new Mat3();
    this._normalMat4  = new Mat4(); // padded mat3 as mat4 for the uniform
  }

  /**
   * Render a scene from a camera's perspective.
   * @param {import('../scene/Node.js').Node} scene   — root node (or array of nodes)
   * @param {import('../scene/Camera.js').Camera} camera
   * @param {import('../scene/Light.js').Light[]} [lights]
   */
  render(scene, camera, lights = []) {
    const gl    = this.gl;
    const cache = this.cache;

    // Update matrices
    camera.updateProjection();
    camera.updateView();

    gl.clearColor(0.1, 0.1, 0.12, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    cache.setDepthTest(true);
    cache.setDepthWrite(true);
    cache.setCullFace(true, gl.BACK);
    cache.setBlend(false);

    cache.useProgram(this.shader.program);

    // Camera uniforms
    const cam = camera;
    this.shader.setMat4('u_ViewMatrix',       cam.viewMatrix.e);
    this.shader.setMat4('u_ProjectionMatrix', cam.projectionMatrix.e);
    this.shader.setVec3('u_CameraPos',
      cam.position.e[0], cam.position.e[1], cam.position.e[2]);

    // Light uniforms — use first directional or a default
    const dirLight = lights.find(l => l.type === 'directional');
    if (dirLight) {
      this.shader.setVec3('u_LightDir',
        dirLight.direction.e[0], dirLight.direction.e[1], dirLight.direction.e[2]);
      this.shader.setVec3('u_LightColor',
        dirLight.color.e[0] * dirLight.intensity,
        dirLight.color.e[1] * dirLight.intensity,
        dirLight.color.e[2] * dirLight.intensity);
    } else {
      this.shader.setVec3('u_LightDir',   0.5, -1.0, 0.5);
      this.shader.setVec3('u_LightColor', 1.0,  1.0, 1.0);
    }

    // Collect and draw renderables
    const nodes = Array.isArray(scene) ? scene : this._collectNodes(scene);
    for (const node of nodes) {
      if (!node.mesh) continue;
      this._drawNode(node);
    }
  }

  /** @private */
  _drawNode(node) {
    const mesh  = node.mesh;
    const gl    = this.gl;
    const cache = this.cache;

    // Model matrix
    node.updateWorldMatrix(node.parent ? node.parent.worldMatrix : null);
    this.shader.setMat4('u_ModelMatrix', node.worldMatrix.e);

    // Normal matrix (inverse transpose of model matrix upper-left 3x3)
    Mat4.invert(node.worldMatrix, this._mvp);
    Mat4.transpose(this._mvp, this._mvp);
    this.shader.setMat4('u_NormalMatrix', this._mvp.e);

    // Material uniforms
    const mat = mesh.material || {};
    const bc  = mat.baseColorFactor ?? [0.8, 0.6, 0.4, 1.0];
    this.shader.setVec3('u_BaseColor',  bc[0], bc[1], bc[2]);
    this.shader.setFloat('u_Roughness', mat.roughnessFactor ?? 0.5);
    this.shader.setFloat('u_Metallic',  mat.metallicFactor  ?? 0.0);

    cache.bindVAO(mesh.vao);

    if (mesh.indexBuffer) {
      gl.drawElements(gl.TRIANGLES, mesh.indexCount, mesh.indexType, 0);
    } else {
      gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
    }

    cache.bindVAO(null);
  }

  /** Depth-first collect all scene nodes into a flat list. @private */
  _collectNodes(root) {
    const result = [];
    const stack  = [root];
    while (stack.length) {
      const node = stack.pop();
      result.push(node);
      if (node.children) {
        for (let i = node.children.length - 1; i >= 0; i--) {
          stack.push(node.children[i]);
        }
      }
    }
    return result;
  }

  resize(w, h) {
    this.cache.setViewport(0, 0, w, h);
  }

  destroy() {
    this.shader.destroy();
    this.shader = null;
  }
}
