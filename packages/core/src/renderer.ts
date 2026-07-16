/**
 * WebGPU renderer for the mesh profile (spec §10.3, §12.3–§12.4).
 *
 * - Positions are pulled and DEQUANTIZED in the vertex shader from a u16 storage
 *   buffer using the GOP AABB — the CPU never touches float positions (§12.4).
 * - The index buffer is uploaded once per GOP (persistent topology, §12.3); only
 *   the position buffer rotates per frame across THREE slots (triple buffering, §10.2).
 * - The texture atlas is a still imported to a sampled texture; the fragment shader
 *   derives a normal from screen-space derivatives for simple lighting.
 *
 * The VideoDecoder→importExternalTexture path (spec §7.1) is the P1.5 texture-video
 * upgrade; P1 ships the still-atlas path (§7.7).
 */
import type { Aabb } from "./quant.js";
import { invert, multiply } from "./camera.js";
import { buildOriginTripod, type GridParams } from "./overlay.js";

// Origin tripod: world-space colored lines. Reads only viewProj from the shared uniform (first 64
// bytes); vertex buffer is interleaved position(vec3)+color(vec3).
const LINES_WGSL = /* wgsl */ `
struct GU { viewProj : mat4x4<f32> };
@group(0) @binding(0) var<uniform> u : GU;
struct VO { @builtin(position) clip : vec4<f32>, @location(0) color : vec3<f32> };
@vertex fn vs(@location(0) pos : vec3<f32>, @location(1) col : vec3<f32>) -> VO {
  var o : VO; o.clip = u.viewProj * vec4<f32>(pos, 1.0); o.color = col; return o;
}
@fragment fn fs(in : VO) -> @location(0) vec4<f32> { return vec4<f32>(in.color, 1.0); }
`;

// INFINITE ground grid — a fullscreen triangle that raymarches the y = floorY plane per pixel.
// No geometry, so no extent to run out of and no fixed cell size to alias: the fragment knows its
// exact ground hit point, so cells are sized to the live zoom (overlay.ts gridLod), each line is
// antialiased against its own screen-space derivative, and the whole thing fades out with distance.
// Writes frag_depth from the hit point so the mesh occludes the floor correctly (depth WRITE is off
// — the grid must never occlude the mesh).
const GRID_WGSL = /* wgsl */ `
struct GU {
  invViewProj : mat4x4<f32>,
  viewProj    : mat4x4<f32>,
  params      : vec4<f32>,   // step0 (fine cell), step1 (coarse cell), fineFade, floorY
  fade        : vec4<f32>,   // centerX, centerZ, fadeRadius, unused
};
@group(0) @binding(0) var<uniform> g : GU;

struct VO { @builtin(position) clip : vec4<f32>, @location(0) ndc : vec2<f32> };
struct FO { @location(0) color : vec4<f32>, @builtin(frag_depth) depth : f32 };

// One oversized triangle covers the viewport with no vertex buffer and no clipped diagonal seam.
@vertex fn vs(@builtin(vertex_index) vi : u32) -> VO {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VO;
  o.clip = vec4<f32>(p[vi], 0.0, 1.0);
  o.ndc = p[vi];
  return o;
}

// NDC + clip-space depth -> world. z = 0 is the near plane, z = 1 the far plane (camera.ts builds
// WebGPU-convention [0,1] projections). Works for ortho as well as perspective: a true inverse of
// viewProj carries whichever one is in force.
fn unproj(ndc : vec2<f32>, z : f32) -> vec3<f32> {
  let p = g.invViewProj * vec4<f32>(ndc.x, ndc.y, z, 1.0);
  return p.xyz / p.w;
}

// Antialiased line coverage for a grid of the given cell size, using screen-space derivatives: the
// distance to the nearest cell edge, measured in PIXELS, so every line lands ~1px wide at any zoom.
fn gridAA(xz : vec2<f32>, cell : f32) -> f32 {
  let c = xz / cell;
  let d = max(fwidth(c), vec2<f32>(1e-8, 1e-8));
  let l = abs(fract(c - vec2<f32>(0.5, 0.5)) - vec2<f32>(0.5, 0.5)) / d;
  return 1.0 - min(min(l.x, l.y), 1.0);
}

@fragment fn fs(in : VO) -> FO {
  var o : FO;
  o.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  o.depth = 1.0;

  let n = unproj(in.ndc, 0.0);
  let f = unproj(in.ndc, 1.0);
  let dir = f - n;
  let t = (g.params.w - n.y) / select(dir.y, 1e-9, abs(dir.y) < 1e-9);
  // The ground is behind the camera or past the far plane for this pixel. Clamp t BEFORE deriving
  // the hit point regardless: fwidth reads the neighbouring lanes in the quad, and a discarded lane
  // holding an infinity would poison a surviving neighbour's derivative along the horizon.
  let hitsPlane = t >= 0.0 && t <= 1.0 && abs(dir.y) >= 1e-9;
  let p = n + clamp(t, 0.0, 1.0) * dir;

  let fine   = gridAA(p.xz, g.params.x) * g.params.z;
  let coarse = gridAA(p.xz, g.params.y);
  // The X (z = 0) and Z (x = 0) center lines, so the origin stays findable at any zoom.
  let d = max(fwidth(p.xz), vec2<f32>(1e-8, 1e-8));
  let centerLine = max(1.0 - min(abs(p.z) / d.y, 1.0), 1.0 - min(abs(p.x) / d.x, 1.0));

  var a = max(max(fine, coarse), centerLine);
  let col = mix(vec3<f32>(0.28, 0.28, 0.31), vec3<f32>(0.46, 0.46, 0.50), centerLine);

  // "Endlessly extend / fade into the distance": radial fade around the point being looked at, in
  // ground-plane distance — camera-azimuth independent, so it doesn't swim while you orbit.
  let radial = 1.0 - smoothstep(g.fade.z * 0.45, g.fade.z, length(p.xz - g.fade.xy));
  // Near-edge-on, a plane's cells compress below a pixel and any line grid turns to moiré. Fading
  // as the ray flattens is what an infinite grid must do instead of shimmering.
  let grazing = smoothstep(0.0, 0.06, abs(normalize(dir).y));
  a = a * radial * grazing * 0.9 * select(0.0, 1.0, hitsPlane);
  if (a < 0.002) { discard; }

  let cp = g.viewProj * vec4<f32>(p, 1.0);
  o.depth = clamp(cp.z / cp.w, 0.0, 1.0);
  o.color = vec4<f32>(col, a);
  return o;
}
`;

/** Per-vertex normal unpack, chosen by superblock normal_encoding (0 = legacy i8×4, 1 = oct16). */
const NRM_DECODE: Record<number, string> = {
  0: /* wgsl */ `o.normal = vec3<f32>(sx8(np), sx8(np >> 8u), sx8(np >> 16u)) / 127.0;`,
  1: /* wgsl */ `
  let onx = f32(i32(np << 16u) >> 16u) / 32767.0;
  let ony = f32(i32(np) >> 16u) / 32767.0;
  var on = vec3<f32>(onx, ony, 1.0 - abs(onx) - abs(ony));
  let ot = clamp(-on.z, 0.0, 1.0);
  on.x = on.x + select(ot, -ot, on.x >= 0.0);
  on.y = on.y + select(ot, -ot, on.y >= 0.0);
  o.normal = normalize(on);`,
};

const WGSL = (normalEncoding: number) => /* wgsl */ `
struct Uniforms {
  viewProj : mat4x4<f32>,
  aabbMin  : vec3<f32>,
  invLevels: f32,
  aabbSize : vec3<f32>,
  _pad     : f32,
  cropMin  : vec3<f32>,   // world-space crop box (mesh editor preview)
  cropOn   : f32,
  cropMax  : vec3<f32>,
  texMix   : f32,        // 1 = textured (shaded), 0 = untextured clay — the viewport shading mode
};
@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read> qpos : array<u32>; // 2 u32 / vertex (u16 x,y,z,pad)
@group(0) @binding(2) var<storage, read> quv  : array<u32>; // 1 u32 / vertex (u16 u,v)
@group(0) @binding(3) var samp : sampler;
@group(0) @binding(4) var tex  : texture_2d<f32>;
@group(0) @binding(5) var<storage, read> qnrm : array<u32>; // 1 u32 / vertex (i8 x,y,z,pad snorm)

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv         : vec2<f32>,
  @location(1) world      : vec3<f32>,
  @location(2) normal     : vec3<f32>,
};

fn sx8(v : u32) -> f32 { return f32(i32(v << 24u) >> 24u); }

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  let a = qpos[vi * 2u];
  let b = qpos[vi * 2u + 1u];
  let q = vec3<f32>(f32(a & 0xffffu), f32((a >> 16u) & 0xffffu), f32(b & 0xffffu));
  let world = u.aabbMin + (q * u.invLevels) * u.aabbSize;   // GPU-side dequant
  let uvp = quv[vi];
  let np = qnrm[vi];
  var o : VSOut;
  o.clip = u.viewProj * vec4<f32>(world, 1.0);
  o.uv = vec2<f32>(f32(uvp & 0xffffu), f32((uvp >> 16u) & 0xffffu)) / 65535.0;
  o.world = world;
  ${NRM_DECODE[normalEncoding] ?? NRM_DECODE[0]!}
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  // Face normal computed unconditionally (dpdx/dpdy require uniform control flow);
  // prefer the smooth interpolated normal, fall back to the face normal if it degenerates.
  let faceN = cross(dpdx(in.world), dpdy(in.world));
  let vertN = in.normal;
  var n = select(faceN, vertN, dot(vertN, vertN) > 0.01);
  n = normalize(n);
  let L = normalize(vec3<f32>(0.35, 0.75, 0.55));
  let diff = abs(dot(n, L));                 // two-sided (winding-agnostic)
  // texMix 0 = untextured "clay" (neutral paper grey, same lighting) so form reads without the
  // atlas; sampled unconditionally to keep derivatives/control flow uniform.
  let albedo = mix(vec3<f32>(0.72, 0.71, 0.68), textureSample(tex, samp, in.uv).rgb, u.texMix);
  // Crop preview (mesh editor): discard LAST so derivatives/samples above stay uniform.
  if (u.cropOn > 0.5 && (any(in.world < u.cropMin) || any(in.world > u.cropMax))) { discard; }
  return vec4<f32>(albedo * (0.4 + 0.6 * diff), 1.0);
}
`;

const SLOTS = 3;
/** Storage/index buffer binding sizes must be a multiple of 4 (WebGPU); round up with headroom. */
const roundUp16 = (n: number) => (n + 15) & ~15;

export class WebGPURenderer {
  private ctx: GPUCanvasContext;
  private format: GPUTextureFormat;
  private pipeline!: GPURenderPipeline;
  private uniform: GPUBuffer;
  private sampler: GPUSampler;
  private posBufs: (GPUBuffer | null)[] = [null, null, null];
  private posCap = 0;
  private uvBuf: GPUBuffer | null = null;
  private uvCap = 0;
  private nrmBuf: GPUBuffer | null = null;
  private nrmCap = 0;
  private idxBuf: GPUBuffer | null = null;
  private idxCap = 0;
  private texture: GPUTexture | null = null;
  private texView: GPUTextureView | null = null;
  private texW = 0;
  private texH = 0;
  private depth: GPUTexture | null = null;
  private slot = 0;
  private uni = new Float32Array(32);
  private normalEncoding = 0;
  private crop: { min: [number, number, number]; max: [number, number, number] } | null = null;
  // Wireframe (mesh editor): triangles expanded to a line-list (3 edges/tri) + a line-topology pipeline.
  private wireframe = false;
  private texMix = 1;          // viewport shading: 1 = textured, 0 = untextured clay
  private linePipeline: GPURenderPipeline | null = null;
  private lineIdxBuf: GPUBuffer | null = null;
  private lineIdxCap = 0;
  private lineIndexCount = 0;
  private lastIndices: Uint32Array | null = null;
  // Ground grid (analytic, infinite) + origin tripod (line geometry). Default ON — the floor is a
  // reference, not a tool: without it there is no way to tell a capture that sits on the ground from
  // one that floats, and nothing to read scale against.
  private gridOn = true;
  private gridPipeline: GPURenderPipeline | null = null;
  private gridUni: GPUBuffer | null = null;
  private gridBind: GPUBindGroup | null = null;
  private gridData = new Float32Array(40);   // invViewProj(16) + viewProj(16) + params(4) + fade(4)
  private gridParams: GridParams = { step0: 100, step1: 1000, fineFade: 1, floorY: 0, centerX: 0, centerZ: 0, fadeRadius: 1e4, axisLen: 300 };
  // Model transform (import-time orientation fix). Folded into the MESH's viewProj only — the grid
  // and tripod are world reference and must not move with the model.
  private model: Float32Array | null = null;
  private tripodPipeline: GPURenderPipeline | null = null;
  private tripodVbuf: GPUBuffer | null = null;
  private tripodCount = 0;
  private tripodBind: GPUBindGroup | null = null;

  private constructor(public readonly device: GPUDevice, canvas: HTMLCanvasElement | OffscreenCanvas) {
    this.ctx = canvas.getContext("webgpu") as unknown as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: "opaque" });
    this.uniform = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" });
    this.buildPipeline();
    // 1x1 white placeholder so the first render works even before the atlas loads.
    this.setTextureFromPixels(new Uint8Array([255, 255, 255, 255]), 1, 1);
    // Default (zero) normals so clips without per-vertex normals still render (shader falls
    // back to a face normal). Real normals overwrite this via uploadNormals().
    [this.nrmBuf, this.nrmCap] = this.growStorage(null, 0, 256);
  }

  static async create(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<WebGPURenderer> {
    if (!navigator.gpu) throw new Error("WebGPU unavailable (no navigator.gpu) — the player selects the WebGL2 fallback (§10.4) before reaching this");
    // powerPreference is silently ignored by Chrome's WebGPU on Windows and logs a warning on every
    // adapter request (crbug.com/369219127); omit it there (no functional effect) but keep the hint
    // on platforms that honor it.
    const onWindows = typeof navigator !== "undefined" && /Windows/i.test(navigator.userAgent ?? "");
    const adapter = await navigator.gpu.requestAdapter(onWindows ? {} : { powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    return new WebGPURenderer(device, canvas);
  }

  private buildPipeline(): void {
    const module = this.device.createShaderModule({ code: WGSL(this.normalEncoding) });
    this.pipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    this.linePipeline = null; // rebuilt lazily against the new module/encoding
  }

  private buildLinePipeline(): GPURenderPipeline {
    const module = this.device.createShaderModule({ code: WGSL(this.normalEncoding) });
    return this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "line-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
  }

  /** Wireframe mode (mesh editor): draw the mesh as its triangle edges. */
  setWireframe(on: boolean): void {
    this.wireframe = on;
    if (on && this.lastIndices) this.uploadLineIndices(this.lastIndices);
  }

  /** Viewport shading: textured (default) vs untextured clay. Judge FORM without the atlas. */
  setTextured(on: boolean): void { this.texMix = on ? 1 : 0; }

  /** Infinite ground grid + origin tripod overlay; builds the tripod's vertex buffer lazily. */
  setGrid(on: boolean): void {
    this.gridOn = on;
    if (on && !this.tripodVbuf) {
      const { data, count } = buildOriginTripod({ floorY: this.gridParams.floorY, axisLen: this.gridParams.axisLen });
      this.tripodVbuf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.tripodVbuf, 0, data);
      this.tripodCount = count;
    }
  }

  /** Live model transform (column-major mat4), or null for identity. Preview only — the bake applies
   *  the same transform to the geometry itself (encoder --up-axis/--center/...). */
  setModelMatrix(m: Float32Array | null): void { this.model = m; }

  /** Cell sizes, ground height and fade radius for the infinite grid — the player recomputes these
   *  from the live camera every frame (that is what keeps the grid readable at any zoom). */
  setGridParams(p: GridParams): void { this.gridParams = p; }

  private uploadLineIndices(indices: Uint32Array): void {
    // tri (a,b,c) → edges a-b, b-c, c-a. Shared edges draw twice — visually fine, no dedup cost.
    const lines = new Uint32Array(indices.length * 2);
    for (let t = 0, w = 0; t < indices.length; t += 3) {
      const a = indices[t]!, b = indices[t + 1]!, c = indices[t + 2]!;
      lines[w++] = a; lines[w++] = b;
      lines[w++] = b; lines[w++] = c;
      lines[w++] = c; lines[w++] = a;
    }
    const need = lines.byteLength;
    if (!this.lineIdxBuf || this.lineIdxCap < need) {
      this.lineIdxBuf?.destroy();
      this.lineIdxCap = roundUp16(Math.max(need, Math.ceil(this.lineIdxCap * 1.5), 256));
      this.lineIdxBuf = this.device.createBuffer({ size: this.lineIdxCap, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.lineIdxBuf, 0, lines);
    this.lineIndexCount = lines.length;
  }

  /** World-space crop box for the mesh-editor preview (fragment discard); null disables. */
  setCrop(crop: { min: [number, number, number]; max: [number, number, number] } | null): void {
    this.crop = crop;
  }

  /** Select the per-vertex normal unpack from the file's superblock normal_encoding (0 = i8×4, 1 = oct16). */
  setNormalEncoding(encoding: number): void {
    if (encoding === this.normalEncoding) return;
    this.normalEncoding = encoding;
    this.buildPipeline();
  }

  resize(width: number, height: number): void {
    const w = Math.max(1, width | 0), h = Math.max(1, height | 0);
    this.depth?.destroy();
    this.depth = this.device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  private growStorage(cur: GPUBuffer | null, cap: number, need: number): [GPUBuffer, number] {
    if (cur && cap >= need) return [cur, cap];
    cur?.destroy();
    const size = roundUp16(Math.max(need, Math.ceil(cap * 1.5), 256));
    return [this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), size];
  }

  /** Upload the index buffer (per I-frame: every frame in intra mode, once per GOP with temporal coding). */
  uploadTopology(indices: Uint32Array): void {
    const need = indices.byteLength;
    if (!this.idxBuf || this.idxCap < need) {
      this.idxBuf?.destroy();
      this.idxCap = roundUp16(Math.max(need, Math.ceil(this.idxCap * 1.5), 256));
      this.idxBuf = this.device.createBuffer({ size: this.idxCap, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.idxBuf, 0, indices);
    this.lastIndices = indices;                                // kept so wireframe can (re)build lazily
    if (this.wireframe) this.uploadLineIndices(indices);
  }

  uploadUVs(uvsQ: Uint16Array): void {
    const need = uvsQ.byteLength;
    [this.uvBuf, this.uvCap] = this.growStorage(this.uvBuf, this.uvCap, need);
    this.device.queue.writeBuffer(this.uvBuf, 0, uvsQ.buffer as ArrayBuffer, uvsQ.byteOffset, uvsQ.byteLength);
  }

  uploadNormals(normalsQ: Int8Array): void {
    const need = normalsQ.byteLength;
    [this.nrmBuf, this.nrmCap] = this.growStorage(this.nrmBuf, this.nrmCap, need);
    this.device.queue.writeBuffer(this.nrmBuf, 0, normalsQ.buffer as ArrayBuffer, normalsQ.byteOffset, normalsQ.byteLength);
  }

  /** Write the next frame's quantized positions into the next triple-buffer slot. */
  writePositions(positionsQ: Uint16Array): void {
    this.slot = (this.slot + 1) % SLOTS;
    const need = positionsQ.byteLength;
    let buf = this.posBufs[this.slot];
    if (!buf || this.posCap < need) {
      // grow all slots together so the cap stays uniform
      for (let i = 0; i < SLOTS; i++) this.posBufs[i]?.destroy();
      this.posCap = roundUp16(Math.max(need, Math.ceil(this.posCap * 1.5), 256));
      for (let i = 0; i < SLOTS; i++) this.posBufs[i] = this.device.createBuffer({ size: this.posCap, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      buf = this.posBufs[this.slot]!;
    }
    this.device.queue.writeBuffer(buf, 0, positionsQ.buffer as ArrayBuffer, positionsQ.byteOffset, positionsQ.byteLength);
  }

  setTextureFromBitmap(bitmap: ImageBitmap): void {
    this.allocTexture(bitmap.width, bitmap.height);
    this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: this.texture! }, [bitmap.width, bitmap.height]);
  }

  /** Import a decoded video frame (WebCodecs VideoFrame) into the sampled texture — GPU-side copy, no CPU pixels (spec §10.3). */
  setTextureFromVideoFrame(frame: VideoFrame): void {
    const w = frame.displayWidth, h = frame.displayHeight;
    if (!this.texture || this.texW !== w || this.texH !== h) this.allocTexture(w, h);
    this.device.queue.copyExternalImageToTexture({ source: frame }, { texture: this.texture! }, [w, h]);
  }

  private setTextureFromPixels(rgba: Uint8Array, w: number, h: number): void {
    this.allocTexture(w, h);
    this.device.queue.writeTexture({ texture: this.texture! }, rgba, { bytesPerRow: w * 4 }, [w, h]);
  }

  private allocTexture(w: number, h: number): void {
    this.texture?.destroy();
    this.texture = this.device.createTexture({
      size: [w, h], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.texView = this.texture.createView();
    this.texW = w; this.texH = h;
  }

  render(viewProj: Float32Array, aabb: Aabb, invLevels: number, indexCount: number): void {
    if (!this.idxBuf || !this.uvBuf || !this.nrmBuf || !this.posBufs[this.slot] || !this.depth || !this.texView) return;
    // The mesh draws through viewProj*model; everything below that reads `viewProj` (grid, tripod)
    // deliberately keeps the world one.
    this.uni.set(this.model ? multiply(viewProj, this.model) : viewProj, 0);
    this.uni[16] = aabb.min[0]; this.uni[17] = aabb.min[1]; this.uni[18] = aabb.min[2];
    this.uni[19] = invLevels;
    this.uni[20] = aabb.max[0] - aabb.min[0]; this.uni[21] = aabb.max[1] - aabb.min[1]; this.uni[22] = aabb.max[2] - aabb.min[2];
    const c = this.crop;
    this.uni[24] = c ? c.min[0] : 0; this.uni[25] = c ? c.min[1] : 0; this.uni[26] = c ? c.min[2] : 0;
    this.uni[27] = c ? 1 : 0;
    this.uni[28] = c ? c.max[0] : 0; this.uni[29] = c ? c.max[1] : 0; this.uni[30] = c ? c.max[2] : 0;
    this.uni[31] = this.texMix;
    this.device.queue.writeBuffer(this.uniform, 0, this.uni);

    // Pick the pipeline FIRST — with layout:"auto" the bind group must come from the active pipeline.
    const wire = this.wireframe && this.lineIdxBuf && this.lineIndexCount > 0;
    const activePipeline = wire ? (this.linePipeline ??= this.buildLinePipeline()) : this.pipeline;
    const bindGroup = this.device.createBindGroup({
      layout: activePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: this.posBufs[this.slot]! } },
        { binding: 2, resource: { buffer: this.uvBuf } },
        { binding: 3, resource: this.sampler },
        { binding: 4, resource: this.texView },
        { binding: 5, resource: { buffer: this.nrmBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.055, b: 0.07, a: 1 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depth.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });
    pass.setPipeline(activePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setIndexBuffer(wire ? this.lineIdxBuf! : this.idxBuf, "uint32");
    pass.drawIndexed(wire ? this.lineIndexCount : indexCount);
    // Overlay (after the mesh; depth write off so it never occludes, but the mesh occludes it).
    if (this.gridOn) {
      // Infinite grid: a fullscreen triangle, alpha-blended, writing its own frag_depth.
      const inv = invert(viewProj);
      if (inv) {
        if (!this.gridPipeline) {
          const gm = this.device.createShaderModule({ code: GRID_WGSL });
          this.gridPipeline = this.device.createRenderPipeline({
            layout: "auto",
            vertex: { module: gm, entryPoint: "vs" },
            fragment: {
              module: gm, entryPoint: "fs",
              targets: [{
                format: this.format,
                blend: {
                  color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                  alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
                },
              }],
            },
            primitive: { topology: "triangle-list" },
            depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
          });
          this.gridBind = null;
        }
        this.gridUni ??= this.device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const gp = this.gridParams;
        this.gridData.set(inv, 0);
        this.gridData.set(viewProj, 16);
        this.gridData[32] = gp.step0; this.gridData[33] = gp.step1; this.gridData[34] = gp.fineFade; this.gridData[35] = gp.floorY;
        this.gridData[36] = gp.centerX; this.gridData[37] = gp.centerZ; this.gridData[38] = gp.fadeRadius; this.gridData[39] = 0;
        this.device.queue.writeBuffer(this.gridUni, 0, this.gridData);
        this.gridBind ??= this.device.createBindGroup({ layout: this.gridPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.gridUni } }] });
        pass.setPipeline(this.gridPipeline);
        pass.setBindGroup(0, this.gridBind);
        pass.draw(3);
      }
      // Origin tripod (real lines — a 3D marker the flat grid can't express).
      if (this.tripodVbuf && this.tripodCount > 0) {
        if (!this.tripodPipeline) {
          const tm = this.device.createShaderModule({ code: LINES_WGSL });
          this.tripodPipeline = this.device.createRenderPipeline({
            layout: "auto",
            vertex: { module: tm, entryPoint: "vs", buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }] },
            fragment: { module: tm, entryPoint: "fs", targets: [{ format: this.format }] },
            primitive: { topology: "line-list" },
            depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
          });
          this.tripodBind = null;
        }
        this.tripodBind ??= this.device.createBindGroup({ layout: this.tripodPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }] });
        pass.setPipeline(this.tripodPipeline);
        pass.setBindGroup(0, this.tripodBind);
        pass.setVertexBuffer(0, this.tripodVbuf);
        pass.draw(this.tripodCount);
      }
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.posBufs.forEach((b) => b?.destroy());
    this.uvBuf?.destroy();
    this.nrmBuf?.destroy();
    this.idxBuf?.destroy();
    this.lineIdxBuf?.destroy();
    this.tripodVbuf?.destroy();
    this.gridUni?.destroy();
    this.texture?.destroy();
    this.depth?.destroy();
    this.uniform.destroy();
    this.device.destroy();
  }
}
