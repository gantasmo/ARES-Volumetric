/**
 * 2D video -> full volumetric: the per-frame assembly behind `ares depth --volumetric`.
 *
 * The relief path (depth-mesh.ts) meshes the capture's visible side only, in pseudo-metres from a
 * guessed near/far and a guessed FOV. A volumetric run (depth-io.ts, the volumetric keys) adds what
 * turns that relief into a whole subject: MoGe-2 metric depth, camera-space normals and intrinsics
 * from the unmasked frame, and a SAM 3D Body (MHR) mesh per frame computed with those intrinsics.
 * This module carries the encoder's passes over them:
 *
 *   pass A  SubjectPass     per frame: the subject mask restricted to the dilated footprint of the
 *                           body (a second person in the mask has no body and is dropped), and the
 *                           body's per-vertex colour accumulated from every frame in which a vertex
 *                           faced the camera unoccluded (BodyColorAccumulator)
 *   pass F  fitPass         per frame: the stabilized disparity fitted onto the metric target by a
 *                           tiled robust fit, then the fields smoothed over time (depth-metric.ts)
 *           anchorPass      per frame: the scale about the camera that lays the body's front surface
 *                           on the fitted shell, smoothed over time (depth-body.ts)
 *   pass C  DetailTrack     the fitted depth, times exp of the normal-map detail residual (screened
 *                           Poisson on log depth, median of three frames)
 *           assembleVolumetricFrame  shell mesh + the carved body pushed behind it + the hair and
 *                           cloth backing, in ARES world space, one merged mesh per frame
 *
 * Coordinates: OpenCV camera space (x right, y down, z forward, metres) everywhere here, pixel
 * (u, v) = (fx*x/z + cx, fy*y/z + cy) in map pixels with pixel centres at integer + 0.5. The one
 * conversion to ARES world space (x, -y, -z) is the last step of assembleVolumetricFrame.
 *
 * Atlas: the frame region (the picture at texSize square, as the relief path) on top, the body's
 * patch atlas (bakeBackAtlas, one 3x3 texel patch per MHR face) under it. The back region is the
 * same pixels every frame, so after each GOP's I-frame it costs the texture codec almost nothing.
 *
 * Streaming: every function works on one frame, or on a window of at most five body frames and three
 * detail frames; the per-frame tile fields (a few hundred floats each) are the only clip-long state.
 *
 * Measured 2026-09-19 (Node 22.19, 1080p skateboarding clip, map 518x294, texture 1024): 90 frames,
 * pass A 1.13 s, fit 0.34 s, anchor 0.78 s, pass C 8.16 s (mesh, detail and VP9 together, 91
 * ms/frame); 20,267 triangles and 395.0 KB per frame. 150 frames of the same clip (before the
 * per-frame focal normalization, focalScales): pass C 15.81 s, 28,612 triangles and 462.6 KB per
 * frame.
 */
import type { EncodeMeshFrame } from "./geometry-encode.js";
import {
  fitTileField, smoothTileFields, evalTileField, normalGradients, gradientVote, poissonDetail, median3,
  type TileField, type Intrinsics,
} from "./depth-metric.js";
import {
  shellMesh, rasterizeDepth, anchorScale, smoothSeries, smoothVertexWindow, assembleHybrid, BodyColorAccumulator,
  toAresSpace, type HybridOptions, type ShellOptions,
} from "./depth-body.js";
import { mergeMeshes } from "./depth-mesh.js";
import { resampleMask } from "./depth-layers.js";

/** The value of the clip's `volumetric.method` key. */
export const VOLUMETRIC_METHOD = "shell+sam-3d-body";
/**
 * Temporal smoothing radius, frames, of the tile fields. Measured 2026-09-19 on a 90-frame 1080p
 * skateboarding clip (map 518x294, subject at 5.6 m), before the per-frame focal normalization
 * (focalScales; after it the target moves 68.6 mm and the shell 9.4 mm at radius 8): MoGe-2's median
 * subject depth moves 92.5 mm from frame to frame (p50); the fitted shell's moves 64.1 mm at radius
 * 0, 27.5 at 4, 12.7 at 8 and 11.3 at 16, while its median distance to the per-frame MoGe-2 target
 * grows from 64 to 111 mm (the flicker it no longer follows).
 */
export const VOLUMETRIC_SMOOTH_RADIUS = 8;
/**
 * Gaussian radius, frames, of the anchor scale after smoothSeries' +-2 median. 0: the fitted shell is
 * already smooth in time, and the per-frame scale is what keeps the body's front on it; smoothing the
 * scale puts back the body's own depth jitter. Same clip, before the focal normalization: body front
 * against the shell p50 9.2 mm / max 69 mm and body centroid step p50 18.7 mm at radius 0; 17.6 / 103
 * / 22.1 at 4; 29.6 / 145 / 25.4 at 8.
 */
export const ANCHOR_SMOOTH_RADIUS = 0;
/** The body track is averaged over +-BODY_WINDOW frames with a Gaussian of sigma BODY_SIGMA. */
export const BODY_WINDOW = 2;
export const BODY_SIGMA = 1;
/** The subject mask keeps pixels within this fraction of the map width of the body's footprint. */
export const SUBJECT_DILATE = 0.02;
/** Frames the normal-map axis vote is taken over. */
export const VOTE_FRAMES = 5;

/* ------------------------------------------------------------------------------------------------ */
/* small helpers                                                                                     */

/** Median of the finite entries (NaN when none). */
export function median(values: ArrayLike<number>): number {
  const a: number[] = [];
  for (let i = 0; i < values.length; i++) { const v = values[i]!; if (Number.isFinite(v)) a.push(v); }
  if (!a.length) return NaN;
  a.sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m]! : (a[m - 1]! + a[m]!) / 2;
}

const round = (v: number, d: number): number => (Number.isFinite(v) ? +v.toFixed(d) : v);

/** Normalized pinhole intrinsics (fx = focal_px / W, fy = focal_px / H, cx, cy as fractions). */
export interface NormalizedIntrinsics { fx: number; fy: number; cx: number; cy: number }

/**
 * Map-pixel intrinsics of a crop of the run's maps. The focal is the run's (fx * runW, fy * runH);
 * the principal point moves by the crop's origin. Projecting a camera-space point with the result
 * lands in the cropped map's pixels.
 */
export function cropIntrinsics(k: NormalizedIntrinsics, runW: number, runH: number, box: { x: number; y: number }): Intrinsics {
  for (const [n, v] of [["fx", k.fx], ["fy", k.fy]] as const) if (!(v > 0 && v < Infinity)) throw new Error(`cropIntrinsics: ${n} must be a finite number > 0, got ${v}`);
  if (!Number.isFinite(k.cx) || !Number.isFinite(k.cy)) throw new Error(`cropIntrinsics: cx and cy must be finite, got ${k.cx}, ${k.cy}`);
  return { fx: k.fx * runW, fy: k.fy * runH, cx: k.cx * runW - box.x, cy: k.cy * runH - box.y };
}

/** Intrinsics of a gw x gh grid laid over a W x H map (the same camera, other pixels). */
export function scaleIntrinsics(K: Intrinsics, W: number, H: number, gw: number, gh: number): Intrinsics {
  const sx = gw / W, sy = gh / H;
  return { fx: K.fx * sx, fy: K.fy * sy, cx: K.cx * sx, cy: K.cy * sy };
}

/** Full vertical and horizontal field of view, degrees, of a W x H map with intrinsics K. */
export function fovOf(K: Intrinsics, W: number, H: number): { fovY: number; fovX: number } {
  return { fovY: (2 * Math.atan(H / 2 / K.fy) * 180) / Math.PI, fovX: (2 * Math.atan(W / 2 / K.fx) * 180) / Math.PI };
}

/**
 * Per-frame factors that put MoGe-2's metric z on the clip focal. MoGe-2 infers each frame's focal
 * and depth together, and a frame whose focal came out long came out far: on the 90-frame
 * skateboarding run (2026-09-19) the normalized fy per frame spans 1.4561 to 1.9156 around the clip
 * median 1.6493, and log fy_t against log median z in the subject mask correlates at 0.926 (slope
 * 1.075). The shell, the body fit and the normal gradients all use the clip focal, so frame t's z is
 * multiplied by clipFy / fy_t, which keeps every pixel's lateral position x = (u - cx) * z / f. A
 * frame without a finite positive estimate (NaN row) keeps 1, and so does a ratio within 1e-6 of 1:
 * the clip value is the manifest's 6-decimal rounding of a median of float32 rows. `perFrame` is
 * openDepthRun's perFrameIntrinsics() (frames x 4 normalized fx, fy, cx, cy), or null for all ones.
 */
export function focalScales(clipFy: number, perFrame: Float32Array | null, frames: number): Float64Array {
  const fn = "focalScales";
  if (!(clipFy > 0 && clipFy < Infinity)) throw new Error(`${fn}: clipFy must be a finite number > 0, got ${clipFy}`);
  if (!Number.isInteger(frames) || frames < 0) throw new Error(`${fn}: frames must be an integer >= 0, got ${frames}`);
  if (perFrame && perFrame.length < frames * 4) throw new Error(`${fn}: perFrame holds ${perFrame.length} values, ${frames * 4} needed`);
  const out = new Float64Array(frames).fill(1);
  if (!perFrame) return out;
  for (let t = 0; t < frames; t++) {
    const fy = perFrame[t * 4 + 1]!;
    if (!(fy > 0 && fy < Infinity)) continue;
    const r = clipFy / fy;
    if (Math.abs(r - 1) > 1e-6) out[t] = r;
  }
  return out;
}

/**
 * A metric depth map (0 = invalid) onto a gw x gh grid. Shrinking averages the valid samples of each
 * cell's footprint (a cell under half covered is invalid, so a silhouette never averages toward 0);
 * growing takes the nearest sample at the cell centre.
 */
export function resampleDepth(z: Float32Array, W: number, H: number, gw: number, gh: number): Float32Array {
  const out = new Float32Array(gw * gh);
  if (gw === W && gh === H) { out.set(z.subarray(0, W * H)); return out; }
  const rx = W / gw, ry = H / gh;
  for (let y = 0; y < gh; y++) {
    const y0 = Math.min(H - 1, Math.floor(y * ry)), y1 = Math.max(y0 + 1, Math.min(H, Math.round((y + 1) * ry)));
    const yc = Math.min(H - 1, Math.floor((y + 0.5) * ry));
    for (let x = 0; x < gw; x++) {
      if (rx <= 1 && ry <= 1) {
        const v = z[yc * W + Math.min(W - 1, Math.floor((x + 0.5) * rx))]!;
        out[y * gw + x] = v > 0 && v < Infinity ? v : 0;
        continue;
      }
      const x0 = Math.min(W - 1, Math.floor(x * rx)), x1 = Math.max(x0 + 1, Math.min(W, Math.round((x + 1) * rx)));
      let s = 0, n = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) { const v = z[yy * W + xx]!; if (v > 0 && v < Infinity) { s += v; n++; } }
      out[y * gw + x] = n * 2 >= (y1 - y0) * (x1 - x0) ? s / n : 0;
    }
  }
  return out;
}

/** A mesh without its unreferenced vertices (positions, uvs and indices remapped). */
export function compactMesh(m: EncodeMeshFrame): EncodeMeshFrame {
  const V = m.positions.length / 3;
  const remap = new Int32Array(V).fill(-1);
  let n = 0;
  for (let i = 0; i < m.indices.length; i++) { const v = m.indices[i]!; if (remap[v]! < 0) remap[v] = n++; }
  if (n === V) return m;
  const positions = new Float32Array(n * 3), uvs = m.uvs ? new Float32Array(n * 2) : undefined;
  for (let v = 0; v < V; v++) {
    const r = remap[v]!;
    if (r < 0) continue;
    positions[r * 3] = m.positions[v * 3]!; positions[r * 3 + 1] = m.positions[v * 3 + 1]!; positions[r * 3 + 2] = m.positions[v * 3 + 2]!;
    if (uvs && m.uvs) { uvs[r * 2] = m.uvs[v * 2]!; uvs[r * 2 + 1] = m.uvs[v * 2 + 1]!; }
  }
  const indices = new Uint32Array(m.indices.length);
  for (let i = 0; i < indices.length; i++) indices[i] = remap[m.indices[i]!]!;
  return { positions, uvs, indices };
}

/**
 * bakeBackAtlas's per-corner uvs (region pixel coordinates of texel centres) mapped into the whole
 * atlas: the back region sits under the texW x texH frame region, atlas texW x atlasH.
 */
export function backAtlasUvs(regionUvs: Float32Array, texW: number, texH: number, atlasH: number): Float32Array {
  const out = new Float32Array(regionUvs.length);
  for (let i = 0; i < regionUvs.length; i += 2) {
    out[i] = regionUvs[i]! / texW;
    out[i + 1] = (texH + regionUvs[i + 1]!) / atlasH;
  }
  return out;
}

/** One atlas frame: the texture frame (texW x texH x 3) with the back region's pixels under it. */
export function composeBackAtlas(frame: Uint8Array, texW: number, texH: number, back: Uint8Array, backH: number): Uint8Array {
  if (frame.length < texW * texH * 3) throw new Error(`composeBackAtlas: frame holds ${frame.length} bytes, ${texW * texH * 3} needed`);
  if (back.length < texW * backH * 3) throw new Error(`composeBackAtlas: back region holds ${back.length} bytes, ${texW * backH * 3} needed`);
  const out = new Uint8Array(texW * (texH + backH) * 3);
  out.set(frame.subarray(0, texW * texH * 3));
  out.set(back.subarray(0, texW * backH * 3), texW * texH * 3);
  return out;
}

/**
 * The shell's triangles split into surface pieces (vertices joined through kept triangles, union-find),
 * pieces under `minVerts` vertices dropped except the largest. shellMesh groups CELLS 4-connected
 * before its z-span cut, so cells joined only across cut edges stay one component there; a strip of
 * the mask's edge that took the background's depth then survives as a sheet floating behind the
 * subject. Returns the kept triangles and, per vertex, 1 when a kept triangle uses it.
 */
export function shellPieces(indices: Uint32Array, vertexCount: number, minVerts: number): { indices: Uint32Array; used: Uint8Array; dropped: number; pieces: number } {
  const parent = new Int32Array(vertexCount);
  for (let v = 0; v < vertexCount; v++) parent[v] = v;
  const find = (v: number): number => { while (parent[v]! !== v) { parent[v] = parent[parent[v]!]!; v = parent[v]!; } return v; };
  const join = (x: number, y: number): void => { const a = find(x), b = find(y); if (a !== b) parent[a] = b; };
  for (let i = 0; i < indices.length; i += 3) { join(indices[i]!, indices[i + 1]!); join(indices[i]!, indices[i + 2]!); }
  const touched = new Uint8Array(vertexCount);
  for (let i = 0; i < indices.length; i++) touched[indices[i]!] = 1;
  const size = new Int32Array(vertexCount);
  for (let v = 0; v < vertexCount; v++) if (touched[v]) size[find(v)]!++;
  let largest = -1, largestN = 0, pieces = 0;
  for (let v = 0; v < vertexCount; v++) if (size[v]! > 0) { pieces++; if (size[v]! > largestN) { largestN = size[v]!; largest = v; } }
  const keepRoot = (r: number): boolean => size[r]! >= minVerts || r === largest;
  const out = new Uint32Array(indices.length);
  let n = 0;
  for (let i = 0; i < indices.length; i += 3) {
    if (!keepRoot(find(indices[i]!))) continue;
    out[n++] = indices[i]!; out[n++] = indices[i + 1]!; out[n++] = indices[i + 2]!;
  }
  const used = new Uint8Array(vertexCount);
  for (let i = 0; i < n; i++) used[out[i]!] = 1;
  let dropped = 0;
  for (let v = 0; v < vertexCount; v++) if (touched[v] && !used[v]) dropped++;
  return { indices: out.slice(0, n), used, dropped, pieces };
}

/** Square dilation of a 0/1 map by `r` pixels, separable running counts: O(W*H) whatever r. */
function dilateSquare(src: Uint8Array, W: number, H: number, r: number, tmp: Uint8Array, out: Uint8Array): void {
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let c = 0;
    for (let x = 0; x < Math.min(W, r); x++) c += src[row + x]!;
    for (let x = 0; x < W; x++) {
      if (x + r < W) c += src[row + x + r]!;
      if (x - r - 1 >= 0) c -= src[row + x - r - 1]!;
      tmp[row + x] = c > 0 ? 1 : 0;
    }
  }
  for (let x = 0; x < W; x++) {
    let c = 0;
    for (let y = 0; y < Math.min(H, r); y++) c += tmp[y * W + x]!;
    for (let y = 0; y < H; y++) {
      if (y + r < H) c += tmp[(y + r) * W + x]!;
      if (y - r - 1 >= 0) c -= tmp[(y - r - 1) * W + x]!;
      out[y * W + x] = c > 0 ? 1 : 0;
    }
  }
}

/* ------------------------------------------------------------------------------------------------ */
/* body track                                                                                        */

/**
 * The run's body frames, read positionally and averaged over a short window. MHR topology is the same
 * every frame, so the window average is a per-vertex mean; it takes the frame-to-frame jitter of a
 * per-frame fit out of the pose (the lab measured 31 mm per frame of joint jitter on single-view
 * fits, mv_render_infer.py). `zScale` multiplies every z: an --fov override keeps each vertex on its
 * projected pixel under the new focal that way.
 */
export class BodyTrack {
  private readonly cache = new Map<number, Float32Array>();
  constructor(private readonly read: (t: number, out: Float32Array) => void, readonly frames: number, readonly vertexCount: number, private readonly zScale = 1) {
    if (!Number.isInteger(frames) || frames < 1) throw new Error(`BodyTrack: frames must be a positive integer, got ${frames}`);
    if (!Number.isInteger(vertexCount) || vertexCount < 1) throw new Error(`BodyTrack: vertexCount must be a positive integer, got ${vertexCount}`);
    if (!(zScale > 0 && zScale < Infinity)) throw new Error(`BodyTrack: zScale must be a finite number > 0, got ${zScale}`);
  }

  /** Frame t as the run holds it (times zScale on z). The returned array is shared: do not modify. */
  raw(t: number): Float32Array {
    if (!Number.isInteger(t) || t < 0 || t >= this.frames) throw new Error(`BodyTrack: frame ${t} is outside 0..${this.frames - 1}`);
    let f = this.cache.get(t);
    if (!f) {
      f = new Float32Array(this.vertexCount * 3);
      this.read(t, f);
      if (this.zScale !== 1) for (let i = 2; i < f.length; i += 3) f[i] = f[i]! * this.zScale;
      if (this.cache.size >= 4 * BODY_WINDOW + 4) {
        for (const k of [...this.cache.keys()]) if (Math.abs(k - t) > 2 * BODY_WINDOW + 1) this.cache.delete(k);
      }
      this.cache.set(t, f);
    }
    return f;
  }

  /** Frame t averaged over +-BODY_WINDOW frames (clamped to the clip), Gaussian sigma BODY_SIGMA. */
  smoothed(t: number): Float32Array {
    const lo = Math.max(0, t - BODY_WINDOW), hi = Math.min(this.frames - 1, t + BODY_WINDOW);
    const w: Float32Array[] = [];
    for (let s = lo; s <= hi; s++) w.push(this.raw(s));
    return smoothVertexWindow(w, t - lo, BODY_SIGMA);
  }
}

/** A copy of `body` scaled by `s` about the camera origin (every vertex stays on its own ray). */
export function scaleBody(body: Float32Array, s: number): Float32Array {
  if (!(s > 0 && s < Infinity)) throw new Error(`scaleBody: scale must be a finite number > 0, got ${s}`);
  const out = new Float32Array(body.length);
  for (let i = 0; i < body.length; i++) out[i] = body[i]! * s;
  return out;
}

/* ------------------------------------------------------------------------------------------------ */
/* pass A: subject restriction and the body's colours                                                */

export interface SubjectFrameResult { before: number; after: number; visible: number }

/**
 * Per frame of pass A. The subject mask of a text prompt ("person") holds every instance, and only one
 * of them has a body mesh: the mask is restricted to the body's rasterized footprint dilated by
 * `radius` pixels (hair, loose cloth and a held object stay in; a second person does not). When the
 * frame's RGB is at hand and the body was fitted on this frame, the body's visible vertices take
 * their colour from it (BodyColorAccumulator.add: facing the camera, unoccluded, on the mask).
 */
export class SubjectPass {
  readonly acc: BodyColorAccumulator;
  readonly radius: number;
  maskPixels = 0;
  keptPixels = 0;
  frames = 0;
  colourFrames = 0;
  visibleSum = 0;
  private readonly zbuf: Float32Array;
  private readonly foot: Uint8Array;
  private readonly tmp: Uint8Array;
  private readonly dil: Uint8Array;

  constructor(readonly W: number, readonly H: number, readonly K: Intrinsics, readonly faces: Uint32Array, readonly vertexCount: number,
    readonly sign: 1 | -1, radius: number) {
    if (!Number.isInteger(W) || !Number.isInteger(H) || W < 1 || H < 1) throw new Error(`SubjectPass: map size must be positive integers, got ${W}x${H}`);
    if (!Number.isInteger(radius) || radius < 0) throw new Error(`SubjectPass: radius must be an integer >= 0, got ${radius}`);
    this.radius = radius;
    this.acc = new BodyColorAccumulator(vertexCount, faces);
    const P = W * H;
    this.zbuf = new Float32Array(P);
    this.foot = new Uint8Array(P);
    this.tmp = new Uint8Array(P);
    this.dil = new Uint8Array(P);
  }

  /** `mask` (0/255, W x H) restricted into `out`; `rgb` (W x H x 3) or null. */
  frame(body: Float32Array, bodyValid: boolean, mask: Uint8Array, rgb: Uint8Array | null, out: Uint8Array): SubjectFrameResult {
    const { W, H } = this, P = W * H;
    rasterizeDepth(body, this.faces, W, H, this.K, this.zbuf);
    for (let i = 0; i < P; i++) this.foot[i] = this.zbuf[i]! < Infinity ? 1 : 0;
    if (this.radius > 0) dilateSquare(this.foot, W, H, this.radius, this.tmp, this.dil);
    else this.dil.set(this.foot);
    let before = 0, after = 0;
    for (let i = 0; i < P; i++) {
      const m = mask[i] ? 1 : 0;
      before += m;
      const k = m && this.dil[i] ? 255 : 0;
      out[i] = k;
      if (k) after++;
    }
    let visible = 0;
    if (rgb && bodyValid) {
      visible = this.acc.add(body, rgb, out, W, H, this.K, this.zbuf, this.sign);
      this.colourFrames++;
      this.visibleSum += visible;
    }
    this.maskPixels += before;
    this.keptPixels += after;
    this.frames++;
    return { before, after, visible };
  }
}

/* ------------------------------------------------------------------------------------------------ */
/* pass F: metric fit and body anchor                                                                */

export interface FitPassStats {
  frames: number;
  /** Frames with a field of their own (the rest are filled from neighbours by smoothTileFields). */
  fitted: number;
  noFit: number;
  linear: number;
  disparity: number;
  /** The clip's model after smoothing (the majority). */
  kind: string;
  /** p50 over fitted frames of the per-frame median |z residual| over every fit sample, metres: one
   *  affine per frame / the tiled field (the same samples, so the two compare). */
  medresGlobalP50: number;
  medresTiledP50: number;
  /** p50 over fitted frames of the global fit's median |z residual| over its own inliers, metres. */
  medresInliersP50: number;
  samplesP50: number;
  tile: number;
  tiles: [number, number];
  /** p50 of |frame-to-frame change| of the target's median depth inside the subject, metres. */
  metricStepP50: number;
}

/** Median |s[t] - s[t-1]| over consecutive finite entries (NaN when none). */
export function stepP50(s: ArrayLike<number>): number {
  const d: number[] = [];
  for (let t = 1; t < s.length; t++) { const v = Math.abs(s[t]! - s[t - 1]!); if (Number.isFinite(v)) d.push(v); }
  return median(d);
}

/**
 * Tiled fit of every frame's stabilized disparity onto its metric target inside the subject, then
 * smoothTileFields over +-radius frames. `read(t, est, metric, mask)` fills one frame of each (W x H).
 */
export function fitPass(frames: number, W: number, H: number, read: (t: number, est: Float32Array, metric: Float32Array, mask: Uint8Array) => void,
  radius = VOLUMETRIC_SMOOTH_RADIUS, onProgress?: (i: number, n: number) => void): { fields: (TileField | null)[]; stats: FitPassStats } {
  const P = W * H;
  const est = new Float32Array(P), metric = new Float32Array(P), mask = new Uint8Array(P);
  const raw: (TileField | null)[] = new Array(frames).fill(null);
  const g: number[] = [], tl: number[] = [], inl: number[] = [], smp: number[] = [];
  const metricMedian = new Float64Array(frames);
  const inMask: number[] = [];
  let linear = 0, disparity = 0;
  for (let t = 0; t < frames; t++) {
    read(t, est, metric, mask);
    inMask.length = 0;
    for (let i = 0; i < P; i++) { const z = metric[i]!; if (mask[i] && z > 0 && z < Infinity) inMask.push(z); }
    metricMedian[t] = median(inMask);
    const f = fitTileField(est, metric, mask, W, H);
    raw[t] = f;
    if (f) {
      if (f.kind === "linear") linear++; else disparity++;
      g.push(f.medresGlobal); tl.push(f.medresTiled); inl.push(f.global.medres); smp.push(f.samples);
    }
    if (onProgress && (t === frames - 1 || t % 10 === 0)) onProgress(t + 1, frames);
  }
  const fitted = linear + disparity;
  if (!fitted) throw new Error(`volumetric: no frame of ${frames} has a metric fit (under 200 samples where the subject mask, the metric depth and the estimate are all valid)`);
  const fields = smoothTileFields(raw, radius);
  const ref = fields.find((f) => f) as TileField;
  return {
    fields,
    stats: {
      frames, fitted, noFit: frames - fitted, linear, disparity, kind: ref.kind,
      medresGlobalP50: median(g), medresTiledP50: median(tl), medresInliersP50: median(inl), samplesP50: median(smp),
      tile: ref.tile, tiles: [ref.tilesX, ref.tilesY], metricStepP50: stepP50(metricMedian),
    },
  };
}

export interface AnchorPassStats {
  frames: number;
  /** Frames with a fitted body and at least 200 shared pixels. */
  anchored: number;
  scaleP50: number;
  scaleMin: number;
  scaleMax: number;
  /** p50 over anchored frames of median |shellZ - s*bodyZ|, metres, at the frame's own scale. */
  madP50: number;
  sharedP50: number;
  /** |median body front z (smoothed scale) - median shell z| over the shared pixels, p50 and max over frames, metres. */
  frontOffsetP50: number;
  frontOffsetMax: number;
  /** p50 of |frame-to-frame change| of the fitted shell's median depth, metres. */
  shellStepP50: number;
  /** p50 over frames of the per-frame median |shell z - metric target| inside the subject, metres (NaN without readMetric). */
  shellVsMetricP50: number;
}

/**
 * The per-frame scale about the camera that lays the (window-averaged) body's front surface on the
 * fitted shell, then smoothSeries (a +-2 median, then a Gaussian over +-radius frames; radius
 * ANCHOR_SMOOTH_RADIUS by default). Frames whose body was copied from a neighbour (body-valid 0) or
 * that share under 200 pixels are not anchors; smoothSeries fills them. `read(t, est, mask)` fills one
 * frame of the stabilized estimate and the restricted mask; `readMetric`, when given, adds the
 * shell's distance to the per-frame metric target to the statistics.
 */
export function anchorPass(o: {
  frames: number; W: number; H: number; K: Intrinsics; fields: (TileField | null)[]; faces: Uint32Array;
  body: BodyTrack; bodyValid: Uint8Array; read: (t: number, est: Float32Array, mask: Uint8Array) => void;
  readMetric?: (t: number, out: Float32Array) => void;
  radius?: number; onProgress?: (i: number, n: number) => void;
}): { scales: Float64Array; raw: Float64Array; ok: Uint8Array; stats: AnchorPassStats } {
  const { frames, W, H, K } = o;
  const P = W * H;
  const est = new Float32Array(P), mask = new Uint8Array(P), zb = new Float32Array(P), bz = new Float32Array(P);
  const metric = o.readMetric ? new Float32Array(P) : null;
  const raw = new Float64Array(frames), ok = new Uint8Array(frames);
  const shellMed = new Float64Array(frames).fill(NaN), frontMed = new Float64Array(frames).fill(NaN), vsMetric: number[] = [];
  const mads: number[] = [], shared: number[] = [];
  const s: number[] = [], b: number[] = [], r: number[] = [];
  for (let t = 0; t < frames; t++) {
    const f = o.fields[t];
    if (f) {
      o.read(t, est, mask);
      evalTileField(f, est, mask, zb);
      if (metric && o.readMetric) {
        o.readMetric(t, metric);
        r.length = 0;
        for (let i = 0; i < P; i++) { const m = metric[i]!; if (mask[i] && zb[i]! > 0 && m > 0 && m < Infinity) r.push(Math.abs(zb[i]! - m)); }
        if (r.length) vsMetric.push(median(r));
      }
      rasterizeDepth(o.body.smoothed(t), o.faces, W, H, K, bz);
      s.length = 0; b.length = 0;
      for (let i = 0; i < P; i++) if (mask[i] && zb[i]! > 0 && bz[i]! < Infinity) { s.push(zb[i]!); b.push(bz[i]!); }
      shellMed[t] = median(s); frontMed[t] = median(b);
      if (o.bodyValid[t]) {
        const a = anchorScale(bz, zb, mask, W, H);
        if (a && a.scale > 0 && a.scale < Infinity) { raw[t] = a.scale; ok[t] = 1; mads.push(a.mad); shared.push(a.shared); }
      }
    }
    if (o.onProgress && (t === frames - 1 || t % 10 === 0)) o.onProgress(t + 1, frames);
  }
  let anchored = 0;
  for (let t = 0; t < frames; t++) anchored += ok[t]!;
  if (!anchored) throw new Error(`volumetric: no frame of ${frames} anchors the body (a fitted body and 200 pixels shared with the metric shell)`);
  const scales = smoothSeries(raw, ok, o.radius ?? ANCHOR_SMOOTH_RADIUS);
  let lo = Infinity, hi = -Infinity, offMax = NaN;
  const rs: number[] = [], off: number[] = [];
  for (let t = 0; t < frames; t++) {
    if (ok[t]) { rs.push(raw[t]!); lo = Math.min(lo, raw[t]!); hi = Math.max(hi, raw[t]!); }
    const d = Math.abs(frontMed[t]! * scales[t]! - shellMed[t]!);
    // The maximum is kept here: Math.max(...off) overflows the call stack past about 125,000
    // frames on Node 22.19.
    if (Number.isFinite(d)) { off.push(d); if (!(d <= offMax)) offMax = d; }
  }
  return {
    scales, raw, ok,
    stats: {
      frames, anchored, scaleP50: median(rs), scaleMin: lo, scaleMax: hi, madP50: median(mads), sharedP50: median(shared),
      frontOffsetP50: median(off), frontOffsetMax: offMax, shellStepP50: stepP50(shellMed),
      shellVsMetricP50: median(vsMetric),
    },
  };
}

/**
 * The normal map's y convention, chosen once per clip: gradientVote of the normals read as stored
 * against the same normals with y negated, summed over `frames`. `get(t)` gives the frame's fitted
 * depth (0 = none) and its normals.
 */
export function chooseNormalAxis(frameList: number[], W: number, H: number, K: Intrinsics,
  get: (t: number) => { zb: Float32Array; normals: Int8Array }): { flipY: boolean; asStored: number; flipped: number; frames: number[] } {
  let asStored = 0, flipped = 0;
  for (const t of frameList) {
    const { zb, normals } = get(t);
    asStored += gradientVote(zb, normalGradients(normals, W, H, K, false), W, H);
    flipped += gradientVote(zb, normalGradients(normals, W, H, K, true), W, H);
  }
  return { flipY: flipped > asStored, asStored, flipped, frames: frameList };
}

/** Up to `n` frame indices spread evenly over 0..frames-1. */
export function spreadFrames(frames: number, n: number): number[] {
  const k = Math.min(frames, n), out: number[] = [];
  for (let i = 0; i < k; i++) out.push(Math.min(frames - 1, Math.floor(((i + 0.5) * frames) / k)));
  return [...new Set(out)];
}

/* ------------------------------------------------------------------------------------------------ */
/* pass C: detail, per-frame assembly                                                                */

interface DetailFrame { zb: Float32Array; r: Float32Array | null; rms: number; iters: number }

/**
 * Pass C's depth per frame: z = zb * exp(median3(r[t-1], r[t], r[t+1])), zb the fitted depth and r the
 * normal-map detail residual (poissonDetail). Frames are asked for in increasing order; the residual
 * of frame t+1 is computed one frame ahead and three frames stay cached. Without normals, z = zb.
 */
export class DetailTrack {
  private readonly cache = new Map<number, DetailFrame>();
  private readonly est: Float32Array;
  private readonly mask: Uint8Array;
  private readonly normals: Int8Array | null;
  constructor(private readonly o: {
    frames: number; W: number; H: number; K: Intrinsics; flipY: boolean; fields: (TileField | null)[];
    readEst: (t: number, out: Float32Array) => void; readMask: (t: number, out: Uint8Array) => void;
    readNormals: ((t: number, out: Int8Array) => void) | null;
  }) {
    const P = o.W * o.H;
    this.est = new Float32Array(P);
    this.mask = new Uint8Array(P);
    this.normals = o.readNormals ? new Int8Array(P * 3) : null;
  }

  private compute(t: number): DetailFrame {
    const { W, H } = this.o, P = W * H;
    const f = this.o.fields[t];
    const zb = new Float32Array(P);
    if (!f) return { zb, r: null, rms: 0, iters: 0 };
    this.o.readEst(t, this.est);
    this.o.readMask(t, this.mask);
    evalTileField(f, this.est, this.mask, zb);
    if (!this.normals || !this.o.readNormals) return { zb, r: null, rms: 0, iters: 0 };
    this.o.readNormals(t, this.normals);
    const d = poissonDetail(zb, W, H, normalGradients(this.normals, W, H, this.o.K, this.o.flipY));
    return { zb, r: d.r, rms: d.rms, iters: d.iters };
  }

  private at(t: number): DetailFrame {
    let d = this.cache.get(t);
    if (!d) { d = this.compute(t); this.cache.set(t, d); }
    return d;
  }

  /** z for frame t (metres, 0 = no shell) and frame t's own detail figures. */
  get(t: number): { z: Float32Array; rms: number; iters: number; detail: boolean } {
    if (!Number.isInteger(t) || t < 0 || t >= this.o.frames) throw new Error(`DetailTrack: frame ${t} is outside 0..${this.o.frames - 1}`);
    for (const k of [...this.cache.keys()]) if (k < t - 1) this.cache.delete(k);
    const cur = this.at(t);
    if (!cur.r) return { z: cur.zb, rms: 0, iters: 0, detail: false };
    const prev = t > 0 ? this.at(t - 1).r : null;
    const next = t + 1 < this.o.frames ? this.at(t + 1).r : null;
    const r = median3(prev, cur.r, next);
    const z = new Float32Array(cur.zb.length);
    for (let i = 0; i < z.length; i++) { const b = cur.zb[i]!; z[i] = b > 0 ? b * Math.exp(r[i]!) : 0; }
    return { z, rms: cur.rms, iters: cur.iters, detail: true };
  }
}

export interface VolumetricFrameStats {
  shellTris: number;
  shellVerts: number;
  /** Shell vertices in pieces under minCells (shellPieces). */
  shellPiecesDropped: number;
  bodyTris: number;
  backingTris: number;
  covered: number;
  flaps: number;
  pushed: number;
  backingCells: number;
}

/**
 * One frame's mesh: the metric shell (shellMesh), the anchored body carved where the shell shows it
 * and pushed behind it plus the hair and cloth backing (assembleHybrid), merged and converted to ARES
 * world space. `z` and `mask` are W x H map arrays; the mesh is built on a gridW x gridH grid over the
 * map (the map itself unless --grid asked otherwise). `body` is the scaled body, `backUv` the
 * atlas-normalized per-corner uvs of every MHR face, `frameUv` the frame region of the atlas.
 */
export function assembleVolumetricFrame(o: {
  z: Float32Array; mask: Uint8Array; W: number; H: number; K: Intrinsics; gridW: number; gridH: number;
  body: Float32Array; faces: Uint32Array; sign: 1 | -1; backUv: Float32Array; frameUv: { vScale: number; vOffset: number };
  shell?: ShellOptions; hybrid?: HybridOptions;
}): { mesh: EncodeMeshFrame; stats: VolumetricFrameStats } {
  const { W, H, gridW: gw, gridH: gh } = o;
  const same = gw === W && gh === H;
  const zg = same ? o.z : resampleDepth(o.z, W, H, gw, gh);
  const mg = same ? o.mask : resampleMask(o.mask, W, H, gw, gh);
  const Kg = same ? o.K : scaleIntrinsics(o.K, W, H, gw, gh);
  const shell = shellMesh(zg, gw, gh, Kg, o.frameUv, o.shell);
  const minVerts = o.shell?.minCells ?? Math.round((500 * gw * gh) / (848 * 480));
  const pieces = shellPieces(shell.indices, shell.cells, minVerts);
  // The hybrid carves against the shell that is emitted: eroded cells, dropped components, cells no
  // kept triangle uses and dropped pieces count as no shell, so the backing never extends past it.
  const zs = new Float32Array(gw * gh);
  for (let i = 0; i < zs.length; i++) { const k = shell.cellIndex[i]!; if (k >= 0 && pieces.used[k]) zs[i] = zg[i]!; }
  const hyb = assembleHybrid(zs, mg, gw, gh, Kg, o.body, o.faces, o.sign, o.backUv, o.frameUv, o.hybrid);
  const shellMeshC = compactMesh({ positions: shell.positions, uvs: shell.uvs, indices: pieces.indices });
  // The backing keeps a vertex for every region cell, including cells no backing triangle uses: the
  // merged mesh is compacted once more, so no unreferenced vertex reaches the encoder.
  let m = mergeMeshes(shellMeshC, hyb.body);
  m = compactMesh(mergeMeshes(m, hyb.backing));
  m.positions = toAresSpace(m.positions);
  return {
    mesh: m,
    stats: {
      shellTris: pieces.indices.length / 3, shellVerts: shellMeshC.positions.length / 3, shellPiecesDropped: pieces.dropped,
      bodyTris: hyb.body.indices.length / 3, backingTris: hyb.backing.indices.length / 3,
      covered: hyb.stats.covered, flaps: hyb.stats.flaps, pushed: hyb.stats.pushed, backingCells: hyb.stats.backingCells,
    },
  };
}

/** Per-frame VolumetricFrameStats summed, reported as per-frame means. */
export class FrameStatsSum {
  n = 0;
  private readonly s: Record<keyof VolumetricFrameStats, number> = { shellTris: 0, shellVerts: 0, shellPiecesDropped: 0, bodyTris: 0, backingTris: 0, covered: 0, flaps: 0, pushed: 0, backingCells: 0 };
  add(st: VolumetricFrameStats): void { this.n++; for (const k of Object.keys(this.s) as (keyof VolumetricFrameStats)[]) this.s[k] += st[k]; }
  mean(): Record<keyof VolumetricFrameStats, number> {
    const out = { ...this.s };
    for (const k of Object.keys(out) as (keyof VolumetricFrameStats)[]) out[k] = round(out[k] / Math.max(1, this.n), 1);
    return out;
  }
}
