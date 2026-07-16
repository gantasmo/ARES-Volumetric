/**
 * WebGL2 fallback renderer (spec §10.4) — same public surface as WebGPURenderer
 * (renderer.ts) so the player can swap it in when `navigator.gpu` is missing.
 *
 * WebGL2 has no storage buffers, so the vertex-pull layout maps onto plain
 * vertex attributes with the SAME byte layouts the player already uploads:
 *   - positions: 3 × UNSIGNED_SHORT, stride 8 (u16 x,y,z,pad — raw, NOT normalized;
 *     the shader dequantizes exactly like the WGSL: world = min + q·invLevels·size)
 *   - uvs:       2 × UNSIGNED_SHORT normalized, stride 4  (u/65535)
 *   - normals:   stride 4, two program variants mirroring the WGSL NRM_DECODE:
 *       0 legacy: 4 × BYTE normalized (i8 snorm, w = pad)
 *       1 oct16:  2 × SHORT normalized + octahedral decode in GLSL
 * Texture uploads go through texImage2D — ImageBitmap directly; VideoFrame
 * directly where supported (modern Chrome), else via a createImageBitmap copy.
 */
import type { Aabb } from "./quant.js";
import { invert, multiply } from "./camera.js";
import { buildOriginTripod, type GridParams } from "./overlay.js";

/**
 * The renderer surface the player drives — implemented by both WebGPURenderer
 * (structurally) and WebGL2Renderer, so the player can hold either.
 */
export interface AresRenderer {
  resize(width: number, height: number): void;
  setNormalEncoding(encoding: number): void;
  /** World-space crop-box preview (mesh editor); null disables. */
  setCrop(crop: { min: [number, number, number]; max: [number, number, number] } | null): void;
  /** Wireframe render mode (mesh editor). */
  setWireframe(on: boolean): void;
  /** Viewport shading: textured (default) vs untextured clay. */
  setTextured(on: boolean): void;
  /** Infinite ground grid + origin tripod overlay. */
  setGrid(on: boolean): void;
  /** Live model transform (column-major mat4), null = identity. Preview only; the bake is encoder-side. */
  setModelMatrix(m: Float32Array | null): void;
  /** Per-frame grid sizing (cell sizes, ground height, fade radius) — see overlay.ts gridLod. */
  setGridParams(p: GridParams): void;
  uploadTopology(indices: Uint32Array): void;
  uploadUVs(uvsQ: Uint16Array): void;
  uploadNormals(normalsQ: Int8Array): void;
  writePositions(positionsQ: Uint16Array): void;
  setTextureFromBitmap(bitmap: ImageBitmap): void;
  setTextureFromVideoFrame(frame: VideoFrame): void;
  render(viewProj: Float32Array, aabb: Aabb, invLevels: number, indexCount: number): void;
  dispose(): void;
}

const A_POS = 0, A_UV = 1, A_NRM = 2;

/** GLSL 300 es vertex shader — mirrors the WGSL dequant + normal unpack exactly. */
const VS = (normalEncoding: number): string => `#version 300 es
precision highp float;
layout(location=${A_POS}) in vec3 aPosQ; // quantized u16 position, raw integer values (not normalized)
layout(location=${A_UV}) in vec2 aUV;    // u16 normalized -> [0,1]
${normalEncoding === 1
    ? `layout(location=${A_NRM}) in vec2 aNrm; // oct16: 2 x i16 snorm`
    : `layout(location=${A_NRM}) in vec4 aNrm; // legacy: i8 x4 snorm (w = pad)`}
uniform mat4 uViewProj;
uniform vec3 uAabbMin;
uniform vec3 uAabbSize;
uniform float uInvLevels;
out vec2 vUV;
out vec3 vWorld;
out vec3 vNormal;
void main() {
  vec3 world = uAabbMin + (aPosQ * uInvLevels) * uAabbSize; // GPU-side dequant, same as the WGSL
  gl_Position = uViewProj * vec4(world, 1.0);
  vUV = aUV;
  vWorld = world;
${normalEncoding === 1
    ? `  vec3 on = vec3(aNrm, 1.0 - abs(aNrm.x) - abs(aNrm.y));
  float ot = clamp(-on.z, 0.0, 1.0);
  on.x += (on.x >= 0.0) ? -ot : ot;
  on.y += (on.y >= 0.0) ? -ot : ot;
  vNormal = normalize(on);`
    : `  vNormal = aNrm.xyz;`}
}
`;

/** Fragment stage: same two-sided lighting as the WGSL — prefer the vertex normal, fall back to a face normal. */
const FS = `#version 300 es
precision highp float;
in vec2 vUV;
in vec3 vWorld;
in vec3 vNormal;
uniform sampler2D uTex;
uniform float uTexMix;                       // 1 = textured (shaded), 0 = untextured clay
out vec4 fragColor;
void main() {
  vec3 faceN = cross(dFdx(vWorld), dFdy(vWorld));
  vec3 n = normalize(dot(vNormal, vNormal) > 0.01 ? vNormal : faceN);
  vec3 L = normalize(vec3(0.35, 0.75, 0.55));
  float diff = abs(dot(n, L));               // two-sided (winding-agnostic)
  // Matches the WebGPU path's clay colour so the two renderers agree.
  vec3 albedo = mix(vec3(0.72, 0.71, 0.68), texture(uTex, vUV).rgb, uTexMix);
  fragColor = vec4(albedo * (0.4 + 0.6 * diff), 1.0);
}
`;

interface ProgramInfo {
  prog: WebGLProgram;
  uViewProj: WebGLUniformLocation | null;
  uAabbMin: WebGLUniformLocation | null;
  uAabbSize: WebGLUniformLocation | null;
  uInvLevels: WebGLUniformLocation | null;
  uTexMix: WebGLUniformLocation | null;
}

const SLOTS = 3;
const roundUp16 = (n: number) => (n + 15) & ~15;

export class WebGL2Renderer implements AresRenderer {
  private programs = new Map<number, ProgramInfo>();
  private normalEncoding = 0;
  private texMix = 1;          // viewport shading: 1 = textured, 0 = untextured clay
  private posBufs: (WebGLBuffer | null)[] = [null, null, null];
  private posCap = 0;
  private uvBuf: WebGLBuffer | null = null;
  private uvCap = 0;
  private nrmBuf: WebGLBuffer | null = null;
  private nrmCap = 0;
  private hasNormals = false;
  private idxBuf: WebGLBuffer | null = null;
  private idxCap = 0;
  private tex: WebGLTexture;
  private slot = 0;
  private texGen = 0;
  /** texImage2D(VideoFrame) works on modern Chrome; flips off after the first throw. */
  private videoFrameDirect = true;
  private disposed = false;
  // Infinite ground grid (analytic, fullscreen triangle) + origin tripod (interleaved pos+color VBO).
  // Default ON, matching the WebGPU path — the floor is a reference, not a tool.
  private gridOn = true;
  private gridParams: GridParams = { step0: 100, step1: 1000, fineFade: 1, floorY: 0, centerX: 0, centerZ: 0, fadeRadius: 1e4, axisLen: 300 };
  private gridProg: { prog: WebGLProgram; uInvViewProj: WebGLUniformLocation | null; uViewProj: WebGLUniformLocation | null;
    uParams: WebGLUniformLocation | null; uFade: WebGLUniformLocation | null } | null = null;
  // Model transform: folded into the MESH's viewProj only — the grid/tripod are world reference.
  private model: Float32Array | null = null;
  private tripodProg: { prog: WebGLProgram; uViewProj: WebGLUniformLocation | null } | null = null;
  private tripodVbo: WebGLBuffer | null = null;
  private tripodCount = 0;

  private constructor(private readonly gl: WebGL2RenderingContext) {
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);            // match copyExternalImageToTexture (no flip)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.disable(gl.CULL_FACE);                                 // cull off, like the WebGPU pipeline
    gl.clearColor(0.05, 0.055, 0.07, 1);

    const tex = gl.createTexture();
    if (!tex) throw new Error("createTexture failed");
    this.tex = tex;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    // 1x1 white placeholder so the first render works even before the atlas/video loads.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));

    this.getProgram(0); // compile the default program eagerly so shader errors surface at create()
  }

  static async create(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<WebGL2Renderer> {
    const gl = (canvas as HTMLCanvasElement).getContext("webgl2", {
      alpha: false, antialias: false, depth: true, stencil: false,
      powerPreference: "high-performance", preserveDrawingBuffer: false,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("WebGL2 unavailable");
    return new WebGL2Renderer(gl);
  }

  private compile(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type);
    if (!sh) throw new Error("createShader failed");
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) ?? "";
      gl.deleteShader(sh);
      throw new Error("GLSL compile failed: " + log);
    }
    return sh;
  }

  private getProgram(encoding: number): ProgramInfo {
    const key = encoding === 1 ? 1 : 0;
    const cached = this.programs.get(key);
    if (cached) return cached;
    const gl = this.gl;
    const vs = this.compile(gl.VERTEX_SHADER, VS(key));
    const fs = this.compile(gl.FRAGMENT_SHADER, FS);
    const prog = gl.createProgram();
    if (!prog) throw new Error("createProgram failed");
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog) ?? "";
      gl.deleteProgram(prog);
      throw new Error("GLSL link failed: " + log);
    }
    gl.useProgram(prog);
    gl.uniform1i(gl.getUniformLocation(prog, "uTex"), 0);
    const info: ProgramInfo = {
      prog,
      uViewProj: gl.getUniformLocation(prog, "uViewProj"),
      uAabbMin: gl.getUniformLocation(prog, "uAabbMin"),
      uAabbSize: gl.getUniformLocation(prog, "uAabbSize"),
      uInvLevels: gl.getUniformLocation(prog, "uInvLevels"),
      uTexMix: gl.getUniformLocation(prog, "uTexMix"),
    };
    this.programs.set(key, info);
    return info;
  }

  /** Select the per-vertex normal unpack from the superblock normal_encoding (0 = i8×4, 1 = oct16). */
  setNormalEncoding(encoding: number): void {
    this.normalEncoding = encoding === 1 ? 1 : 0;
    // Program variants are compiled lazily in render(); the attribute pointer switches with them.
  }

  /** Crop preview is a WebGPU-path feature for now — no-op on the GL2 fallback (bake still works). */
  setCrop(_crop: { min: [number, number, number]; max: [number, number, number] } | null): void { /* not implemented in GL2 */ }

  /** Wireframe is a WebGPU-path editor feature for now — no-op on the GL2 fallback. */
  setWireframe(_on: boolean): void { /* not implemented in GL2 */ }

  /** Viewport shading: textured (default) vs untextured clay. Parity with the WebGPU path. */
  setTextured(on: boolean): void { this.texMix = on ? 1 : 0; }

  setGridParams(p: GridParams): void { this.gridParams = p; }

  setModelMatrix(m: Float32Array | null): void { this.model = m; }

  setGrid(on: boolean): void {
    this.gridOn = on;
    if (!on || this.gridProg) return;
    const gl = this.gl;
    const mk = (t: number, s: string): WebGLShader => {
      const sh = gl.createShader(t)!; gl.shaderSource(sh, s); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error("grid shader: " + gl.getShaderInfoLog(sh));
      return sh;
    };
    const link = (vsSrc: string, fsSrc: string): WebGLProgram => {
      const prog = gl.createProgram()!;
      gl.attachShader(prog, mk(gl.VERTEX_SHADER, vsSrc));
      gl.attachShader(prog, mk(gl.FRAGMENT_SHADER, fsSrc));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("grid link: " + gl.getProgramInfoLog(prog));
      return prog;
    };

    // --- Origin tripod: plain colored lines.
    const tProg = link(`#version 300 es
precision highp float;
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aColor;
uniform mat4 uViewProj; out vec3 vColor;
void main(){ vColor = aColor; gl_Position = uViewProj * vec4(aPos, 1.0); }`, `#version 300 es
precision highp float;
in vec3 vColor; out vec4 frag;
void main(){ frag = vec4(vColor, 1.0); }`);
    this.tripodProg = { prog: tProg, uViewProj: gl.getUniformLocation(tProg, "uViewProj") };
    const { data, count } = buildOriginTripod({ floorY: this.gridParams.floorY, axisLen: this.gridParams.axisLen });
    this.tripodVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tripodVbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    this.tripodCount = count;

    // --- Infinite grid: the exact math of the WGSL in renderer.ts, kept line-for-line comparable so
    // the two backends can't drift. ONE deliberate difference, and it is load-bearing: camera.ts
    // builds WebGPU-convention projections (clip z in [0,1]), but GL's window depth is z*0.5+0.5, so
    // gl_FragDepth needs that remap or the grid would depth-test against the mesh in the wrong space.
    const gp = link(`#version 300 es
precision highp float;
out vec2 vNdc;
void main(){
  vec2 p = vec2(-1.0, -1.0);
  if (gl_VertexID == 1) p = vec2(3.0, -1.0);
  else if (gl_VertexID == 2) p = vec2(-1.0, 3.0);
  vNdc = p;
  gl_Position = vec4(p, 0.0, 1.0);
}`, `#version 300 es
precision highp float;
uniform mat4 uInvViewProj;
uniform mat4 uViewProj;
uniform vec4 uParams;   // step0, step1, fineFade, floorY
uniform vec4 uFade;     // centerX, centerZ, fadeRadius, unused
in vec2 vNdc; out vec4 frag;

vec3 unproj(vec2 ndc, float z){ vec4 p = uInvViewProj * vec4(ndc, z, 1.0); return p.xyz / p.w; }

float gridAA(vec2 xz, float cell){
  vec2 c = xz / cell;
  vec2 d = max(fwidth(c), vec2(1e-8));
  vec2 l = abs(fract(c - 0.5) - 0.5) / d;
  return 1.0 - min(min(l.x, l.y), 1.0);
}

void main(){
  vec3 n = unproj(vNdc, 0.0);
  vec3 f = unproj(vNdc, 1.0);
  vec3 dir = f - n;
  float dy = abs(dir.y) < 1e-9 ? 1e-9 : dir.y;
  float t = (uParams.w - n.y) / dy;
  bool hitsPlane = t >= 0.0 && t <= 1.0 && abs(dir.y) >= 1e-9;
  vec3 p = n + clamp(t, 0.0, 1.0) * dir;

  float fine   = gridAA(p.xz, uParams.x) * uParams.z;
  float coarse = gridAA(p.xz, uParams.y);
  vec2 d = max(fwidth(p.xz), vec2(1e-8));
  float centerLine = max(1.0 - min(abs(p.z) / d.y, 1.0), 1.0 - min(abs(p.x) / d.x, 1.0));

  float a = max(max(fine, coarse), centerLine);
  vec3 col = mix(vec3(0.28, 0.28, 0.31), vec3(0.46, 0.46, 0.50), centerLine);

  float radial  = 1.0 - smoothstep(uFade.z * 0.45, uFade.z, length(p.xz - uFade.xy));
  float grazing = smoothstep(0.0, 0.06, abs(normalize(dir).y));
  a *= radial * grazing * 0.9 * (hitsPlane ? 1.0 : 0.0);
  if (a < 0.002) discard;

  vec4 cp = uViewProj * vec4(p, 1.0);
  gl_FragDepth = clamp(cp.z / cp.w, 0.0, 1.0) * 0.5 + 0.5;
  frag = vec4(col, a);
}`);
    this.gridProg = {
      prog: gp,
      uInvViewProj: gl.getUniformLocation(gp, "uInvViewProj"),
      uViewProj: gl.getUniformLocation(gp, "uViewProj"),
      uParams: gl.getUniformLocation(gp, "uParams"),
      uFade: gl.getUniformLocation(gp, "uFade"),
    };
  }

  resize(_width: number, _height: number): void {
    // The GL drawing buffer tracks canvas.width/height (set by the player before this call);
    // the viewport is (re)set from drawingBufferWidth/Height on every render.
  }

  /** Bind (creating/growing as needed) and upload into a dynamic buffer. Returns [buffer, cap]. */
  private upload(target: number, buf: WebGLBuffer | null, cap: number, data: ArrayBufferView): [WebGLBuffer, number] {
    const gl = this.gl;
    const need = data.byteLength;
    let b = buf;
    if (!b) {
      b = gl.createBuffer();
      if (!b) throw new Error("createBuffer failed");
    }
    gl.bindBuffer(target, b);
    if (cap < need) {
      cap = roundUp16(Math.max(need, Math.ceil(cap * 1.5), 256));
      gl.bufferData(target, cap, gl.DYNAMIC_DRAW);
    }
    gl.bufferSubData(target, 0, data);
    return [b, cap];
  }

  /** Upload the index buffer (persistent per GOP; per frame in intra mode). */
  uploadTopology(indices: Uint32Array): void {
    [this.idxBuf, this.idxCap] = this.upload(this.gl.ELEMENT_ARRAY_BUFFER, this.idxBuf, this.idxCap, indices);
  }

  uploadUVs(uvsQ: Uint16Array): void {
    [this.uvBuf, this.uvCap] = this.upload(this.gl.ARRAY_BUFFER, this.uvBuf, this.uvCap, uvsQ);
  }

  uploadNormals(normalsQ: Int8Array): void {
    [this.nrmBuf, this.nrmCap] = this.upload(this.gl.ARRAY_BUFFER, this.nrmBuf, this.nrmCap, normalsQ);
    this.hasNormals = true;
  }

  /** Write the next frame's quantized positions into the next triple-buffer slot. */
  writePositions(positionsQ: Uint16Array): void {
    const gl = this.gl;
    this.slot = (this.slot + 1) % SLOTS;
    const need = positionsQ.byteLength;
    if (this.posCap < need || !this.posBufs[this.slot]) {
      // grow all slots together so the cap stays uniform (mirrors the WebGPU renderer)
      for (let i = 0; i < SLOTS; i++) {
        const b = this.posBufs[i];
        if (b) gl.deleteBuffer(b);
        this.posBufs[i] = null;
      }
      this.posCap = roundUp16(Math.max(need, Math.ceil(this.posCap * 1.5), 256));
      for (let i = 0; i < SLOTS; i++) {
        const b = gl.createBuffer();
        if (!b) throw new Error("createBuffer failed");
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, this.posCap, gl.DYNAMIC_DRAW);
        this.posBufs[i] = b;
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBufs[this.slot]!);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, positionsQ);
  }

  setTextureFromBitmap(bitmap: ImageBitmap): void {
    this.texGen++;
    this.teximage(bitmap);
  }

  /** Upload a decoded VideoFrame: texImage2D directly where supported, else via ImageBitmap. */
  setTextureFromVideoFrame(frame: VideoFrame): void {
    const gen = ++this.texGen;
    if (this.videoFrameDirect) {
      try {
        this.teximage(frame);
        return;
      } catch {
        this.videoFrameDirect = false;
      }
    }
    // Fallback: snapshot the frame (taken synchronously by createImageBitmap while the
    // frame is still open) and upload only if no newer texture landed meanwhile.
    void createImageBitmap(frame).then((bmp) => {
      if (gen === this.texGen && !this.disposed) this.teximage(bmp);
      bmp.close();
    }).catch(() => { /* frame closed before the snapshot — keep the previous texture */ });
  }

  private teximage(src: TexImageSource): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, src);
  }

  render(viewProj: Float32Array, aabb: Aabb, invLevels: number, indexCount: number): void {
    if (this.disposed) return;
    const gl = this.gl;
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.idxBuf || !this.posBufs[this.slot] || indexCount <= 0) return;

    const p = this.getProgram(this.normalEncoding);
    gl.useProgram(p.prog);
    gl.uniformMatrix4fv(p.uViewProj, false, this.model ? multiply(viewProj, this.model) : viewProj);
    gl.uniform3f(p.uAabbMin, aabb.min[0], aabb.min[1], aabb.min[2]);
    gl.uniform3f(p.uAabbSize, aabb.max[0] - aabb.min[0], aabb.max[1] - aabb.min[1], aabb.max[2] - aabb.min[2]);
    gl.uniform1f(p.uInvLevels, invLevels);
    gl.uniform1f(p.uTexMix, this.texMix);

    // Attribute pointers are re-specified per draw (cheap) so buffer growth/encoding
    // switches never leave a stale binding behind.
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBufs[this.slot]!);
    gl.vertexAttribPointer(A_POS, 3, gl.UNSIGNED_SHORT, false, 8, 0); // raw u16 → float q
    gl.enableVertexAttribArray(A_POS);

    if (this.uvBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
      gl.vertexAttribPointer(A_UV, 2, gl.UNSIGNED_SHORT, true, 4, 0); // normalized → u/65535
      gl.enableVertexAttribArray(A_UV);
    } else {
      gl.disableVertexAttribArray(A_UV);
      gl.vertexAttrib2f(A_UV, 0, 0);
    }

    if (this.nrmBuf && this.hasNormals) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.nrmBuf);
      if (this.normalEncoding === 1) gl.vertexAttribPointer(A_NRM, 2, gl.SHORT, true, 4, 0); // oct16 snorm
      else gl.vertexAttribPointer(A_NRM, 4, gl.BYTE, true, 4, 0);                            // legacy i8 snorm
      gl.enableVertexAttribArray(A_NRM);
    } else {
      gl.disableVertexAttribArray(A_NRM);
      gl.vertexAttrib4f(A_NRM, 0, 0, 0, 0); // zero normal → fragment falls back to the face normal
    }

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);

    // Overlay after the mesh: no depth write (never occludes), but the mesh occludes it.
    if (this.gridOn && this.gridProg) {
      const inv = invert(viewProj);
      if (inv) {
        const gp = this.gridParams;
        gl.useProgram(this.gridProg.prog);
        gl.uniformMatrix4fv(this.gridProg.uInvViewProj, false, inv);
        gl.uniformMatrix4fv(this.gridProg.uViewProj, false, viewProj);
        gl.uniform4f(this.gridProg.uParams, gp.step0, gp.step1, gp.fineFade, gp.floorY);
        gl.uniform4f(this.gridProg.uFade, gp.centerX, gp.centerZ, gp.fadeRadius, 0);
        // The grid shader pulls nothing per-vertex (gl_VertexID only), so the mesh's attribute
        // arrays must be OFF: a still-enabled array whose buffer is shorter than 3 vertices is an
        // INVALID_OPERATION on some drivers even when the program never reads it.
        gl.disableVertexAttribArray(A_POS);
        gl.disableVertexAttribArray(A_UV);
        gl.disableVertexAttribArray(A_NRM);
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
      }
    }
    if (this.gridOn && this.tripodProg && this.tripodVbo && this.tripodCount > 0) {
      gl.useProgram(this.tripodProg.prog);
      gl.uniformMatrix4fv(this.tripodProg.uViewProj, false, viewProj);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tripodVbo);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0); gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12); gl.enableVertexAttribArray(1);
      gl.depthMask(false);
      gl.drawArrays(gl.LINES, 0, this.tripodCount);
      gl.depthMask(true);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    for (const b of this.posBufs) if (b) gl.deleteBuffer(b);
    if (this.uvBuf) gl.deleteBuffer(this.uvBuf);
    if (this.nrmBuf) gl.deleteBuffer(this.nrmBuf);
    if (this.idxBuf) gl.deleteBuffer(this.idxBuf);
    if (this.tripodVbo) gl.deleteBuffer(this.tripodVbo);
    if (this.tripodProg) gl.deleteProgram(this.tripodProg.prog);
    if (this.gridProg) gl.deleteProgram(this.gridProg.prog);
    gl.deleteTexture(this.tex);
    for (const [, p] of this.programs) gl.deleteProgram(p.prog);
    this.programs.clear();
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
