/**
 * Relief-mesh builder: one stabilized depth map becomes one pinhole-unprojected grid, textured by
 * the video frame it came from.
 *
 * The ray table is built once per clip and reused every frame — only z changes — which is what
 * makes a 256x144 grid cost a multiply per vertex instead of a trig call. Ports the ray-table idea
 * from VJ-9000's depthcloud (a 55 degree vertical FOV, image y down mapped to world y up, the scene
 * in front of the camera along -Z) and replaces its linear 8-bit 0.6..4.0 m depth ramp with the
 * hyperbolic mapping a pinhole camera actually implies: inverse depth is linear in disparity, so
 * normalized disparity d maps to z = 1 / (d/near + (1-d)/far). Linear ramps flatten everything past
 * the middle of the range and stretch the foreground; this puts the samples where the camera put
 * them.
 *
 * Triangle culling across depth discontinuities is what stops the relief from being one rubber
 * sheet. Every cell that straddles a silhouette would otherwise be stretched from the subject's
 * edge back to the wall behind it, painting the subject's edge texels across metres of empty space.
 * A relative depth jump over `edge` drops the triangle and the vertices no triangle references are
 * compacted away.
 */
import type { EncodeMeshFrame } from "./geometry-encode.js";
import type { DepthKind } from "./depth-io.js";
import type { FillLayer } from "./depth-layers.js";

export interface DepthGrid {
  gridW: number;
  gridH: number;
  /** 2 per vertex. u = (x+0.5)/gridW, v = (y+0.5)/gridH with v = 0 at the TOP image row, which is
   *  the atlas convention this repo already uses (importers/obj.ts flips OBJ's bottom-left V, and
   *  both renderers upload with no Y flip). */
  uvs: Float32Array;
  /** Full grid, 2 triangles per cell, wound so the face normal points back at the camera. */
  indices: Uint32Array;
  /** Per-vertex camera-space ray direction at z = 1. */
  rayX: Float32Array;
  rayY: Float32Array;
}

export interface DepthMeshOptions {
  kind: DepthKind;
  near: number;
  far: number;
  /** Relative depth jump across a cell edge that marks a silhouette. Default 0.08. */
  edge?: number;
  /** Keep the full grid: every frame then has byte-identical topology and the muxer takes the
   *  persistent-topology I+P path. The stretched triangles this leaves at every silhouette are
   *  discarded at draw time from the clip's `relief.*` metadata (reliefMeta below). */
  sheets?: boolean;
  /** Subject mask on the grid, 0 = not part of the relief. Culled meshes drop those vertices; a
   *  sheet cannot (its topology is fixed), so it parks them on a backdrop plane at
   *  `far * BACKDROP`, past the `relief.depthMax` the renderer discards beyond. */
  mask?: Uint8Array | null;
  /**
   * Temporal hysteresis for the silhouette cut of a culled mesh: an edge whose relative depth jump
   * lies between `lo * edge` and `hi * edge` keeps the decision it had on the previous frame.
   * `state` holds one byte per grid edge (cutEdgeCount) and is read and rewritten; `primed` is false
   * for the first frame, which takes the plain rule. Defaults lo 0.67, hi 1.5: on 300 frames of a
   * film, with the stabilizer's median, the edges cut for exactly one frame were 160 per frame at
   * 0.8..1.25 and 142 at 0.67..1.5. A silhouette's jump is 25 % and more, well outside the band.
   */
  hysteresis?: { state: Uint8Array; primed: boolean; lo?: number; hi?: number };
}

/** Grid edges a hysteresis state tracks: horizontal, vertical, then one diagonal per cell. */
export const cutEdgeCount = (gridW: number, gridH: number): number => (gridW - 1) * gridH + gridW * (gridH - 1) + (gridW - 1) * (gridH - 1);

/** Where a sheet parks masked-out vertices, as a multiple of `far`. */
export const BACKDROP = 2;

export interface DepthGridOptions {
  /** Atlas placement of the frame: v' = vOffset + v * vScale. Defaults 0 and 1 (frame = atlas). */
  vScale?: number;
  vOffset?: number;
}

/** Ray table + UVs + full-grid index buffer for a gridW x gridH relief. `fovDeg` is VERTICAL. */
export function buildDepthGrid(gridW: number, gridH: number, aspect: number, fovDeg: number, o: DepthGridOptions = {}): DepthGrid {
  if (!Number.isInteger(gridW) || !Number.isInteger(gridH) || gridW < 2 || gridH < 2)
    throw new Error(`buildDepthGrid: grid must be at least 2x2 integers, got ${gridW}x${gridH}`);
  if (!(aspect > 0)) throw new Error(`buildDepthGrid: aspect must be > 0, got ${aspect}`);
  if (!(fovDeg > 0 && fovDeg < 180)) throw new Error(`buildDepthGrid: fov must be in (0,180), got ${fovDeg}`);
  const n = gridW * gridH;
  const uvs = new Float32Array(n * 2);
  const rayX = new Float32Array(n);
  const rayY = new Float32Array(n);
  const tan = Math.tan((fovDeg * Math.PI) / 180 / 2);
  for (let y = 0; y < gridH; y++) {
    const v = (y + 0.5) / gridH;
    const ry = (v - 0.5) * 2 * tan;
    for (let x = 0; x < gridW; x++) {
      const i = y * gridW + x;
      const u = (x + 0.5) / gridW;
      uvs[i * 2] = u; uvs[i * 2 + 1] = (o.vOffset ?? 0) + v * (o.vScale ?? 1);
      rayX[i] = (u - 0.5) * 2 * tan * aspect;
      rayY[i] = ry;
    }
  }
  const indices = new Uint32Array((gridW - 1) * (gridH - 1) * 6);
  let k = 0;
  for (let y = 0; y < gridH - 1; y++) {
    for (let x = 0; x < gridW - 1; x++) {
      const a = y * gridW + x, b = a + 1, c = a + gridW, d = c + 1;
      indices[k++] = a; indices[k++] = c; indices[k++] = b;
      indices[k++] = b; indices[k++] = c; indices[k++] = d;
    }
  }
  return { gridW, gridH, uvs, indices, rayX, rayY };
}

/** One separable axis pass: area-average when shrinking, bilinear when growing or unchanged. */
function resampleAxis(src: Float32Array, srcN: number, dstN: number, lines: number, srcStride: number, dstStride: number, step: number, out: Float32Array): void {
  if (dstN < srcN) {
    const ratio = srcN / dstN;
    for (let i = 0; i < dstN; i++) {
      const s0 = i * ratio, s1 = (i + 1) * ratio;
      const i0 = Math.floor(s0), i1 = Math.min(srcN - 1, Math.ceil(s1) - 1);
      for (let l = 0; l < lines; l++) {
        let sum = 0, wsum = 0;
        for (let s = i0; s <= i1; s++) {
          const w = Math.min(s + 1, s1) - Math.max(s, s0);
          if (w <= 0) continue;
          sum += w * src[l * srcStride + s * step]!;
          wsum += w;
        }
        out[l * dstStride + i * step] = wsum > 0 ? sum / wsum : src[l * srcStride + Math.min(srcN - 1, i0) * step]!;
      }
    }
    return;
  }
  const ratio = srcN / dstN;
  for (let i = 0; i < dstN; i++) {
    let s = (i + 0.5) * ratio - 0.5;
    if (s < 0) s = 0; else if (s > srcN - 1) s = srcN - 1;
    const i0 = Math.floor(s), i1 = Math.min(srcN - 1, i0 + 1), f = s - i0;
    for (let l = 0; l < lines; l++) {
      const a = src[l * srcStride + i0 * step]!, b = src[l * srcStride + i1 * step]!;
      out[l * dstStride + i * step] = a + (b - a) * f;
    }
  }
}

/** Resample one W x H map onto the gridW x gridH vertex lattice. */
export function resampleMap(map: Float32Array, W: number, H: number, gridW: number, gridH: number): Float32Array {
  if (map.length < W * H) throw new Error(`resampleMap: map holds ${map.length} floats, expected ${W * H}`);
  if (W === gridW && H === gridH) return map.slice(0, W * H);
  const tmp = new Float32Array(gridW * H);
  resampleAxis(map, W, gridW, H, W, gridW, 1, tmp);          // rows: W -> gridW
  const out = new Float32Array(gridW * gridH);
  resampleAxis(tmp, H, gridH, gridW, 1, 1, gridW, out);      // columns: H -> gridH
  return out;
}

/** Normalized disparity (or metres) on the grid -> camera-space depth per vertex. */
export function gridDepths(gridMap: Float32Array, n: number, opts: { kind: DepthKind; near: number; far: number }): Float32Array {
  const { near, far } = opts, metric = opts.kind === "metric-depth";
  const z = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let d = gridMap[i]!;
    if (!Number.isFinite(d)) d = metric ? far : 0;
    // Hyperbolic: inverse depth is what is linear in disparity. d = 1 -> near, d = 0 -> far.
    z[i] = metric ? (d < near ? near : d > far ? far : d) : 1 / (d / near + (1 - d) / far);
  }
  return z;
}

/**
 * One grid-resolution depth map → one relief mesh.
 *
 * `gridMap` is what stabilizeDepth produced, resampled to the grid: normalized disparity in [0,1]
 * with 1 = nearest for a relative run, metres for a metric one.
 */
export function depthFrameToMesh(gridMap: Float32Array, grid: DepthGrid, opts: DepthMeshOptions): EncodeMeshFrame {
  const { gridW, gridH, rayX, rayY, uvs, indices } = grid;
  const n = gridW * gridH;
  if (gridMap.length < n) throw new Error(`depthFrameToMesh: gridMap holds ${gridMap.length} floats, expected ${n}`);
  const near = opts.near, far = opts.far;
  if (!(near > 0) || !(far > near)) throw new Error(`depthFrameToMesh: need 0 < near < far, got near=${near} far=${far}`);
  const edge = opts.edge ?? 0.08;
  const metric = opts.kind === "metric-depth";

  const mask = opts.mask ?? null;
  const z = gridDepths(gridMap, n, opts);
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const zi = opts.sheets && mask && mask[i]! < 128 ? far * BACKDROP : z[i]!;
    positions[i * 3] = rayX[i]! * zi;
    positions[i * 3 + 1] = -rayY[i]! * zi;   // image y runs down, world y runs up
    positions[i * 3 + 2] = -zi;              // the scene sits in front of the camera along -Z
  }

  if (opts.sheets) {
    // The grid itself is the topology, identical every frame — the muxer's sameTopology check
    // passes and the whole clip codes as I + position-delta P frames. uvs/indices are SHARED with
    // the grid (and so between frames): nothing downstream mutates either.
    return { positions, uvs, indices };
  }

  // Silhouette cull: drop any triangle with an edge whose depth jump exceeds `edge` relative to
  // the nearer of its two ends. Decided once per grid edge, so hysteresis can carry the decision
  // from frame to frame: buildDepthGrid lays the cells out row-major, two triangles each, (a,c,b)
  // and (b,c,d), which is what locates a triangle's three edges.
  const hy = opts.hysteresis ?? null;
  const E = cutEdgeCount(gridW, gridH);
  if (hy && hy.state.length < E) throw new Error(`depthFrameToMesh: hysteresis state holds ${hy.state.length} edges, the grid has ${E}`);
  const cutAt = hy ? hy.state : new Uint8Array(E);
  const lo = edge * (hy?.lo ?? 0.67), hi = edge * (hy?.hi ?? 1.5), held = !!(hy && hy.primed);
  const H0 = (gridW - 1) * gridH, D0 = H0 + gridW * (gridH - 1);
  const decide = (e: number, a: number, b: number) => {
    const za = z[a]!, zb = z[b]!;
    const r = Math.abs(za - zb) / (za < zb ? za : zb);
    cutAt[e] = held ? (r > hi ? 1 : r < lo ? 0 : cutAt[e]!) : r > edge ? 1 : 0;
  };
  for (let y = 0; y < gridH; y++) for (let x = 0; x + 1 < gridW; x++) decide(y * (gridW - 1) + x, y * gridW + x, y * gridW + x + 1);
  for (let y = 0; y + 1 < gridH; y++) for (let x = 0; x < gridW; x++) decide(H0 + y * gridW + x, y * gridW + x, (y + 1) * gridW + x);
  for (let y = 0; y + 1 < gridH; y++) for (let x = 0; x + 1 < gridW; x++) decide(D0 + y * (gridW - 1) + x, y * gridW + x + 1, (y + 1) * gridW + x);
  const keep = new Uint8Array(indices.length / 3);
  let kept = 0;
  for (let t = 0, i = 0; i < indices.length; t++, i += 3) {
    const a = indices[i]!, b = indices[i + 1]!, c = indices[i + 2]!;
    if (mask && (mask[a]! < 128 || mask[b]! < 128 || mask[c]! < 128)) continue;
    const cell = t >> 1, x = cell % (gridW - 1), y = (cell - x) / (gridW - 1);
    const diag = cutAt[D0 + cell]!;
    const cutTri = (t & 1) === 0
      ? diag || cutAt[H0 + y * gridW + x] || cutAt[y * (gridW - 1) + x]                  // (a,c,b): a-c, c-b, b-a
      : diag || cutAt[(y + 1) * (gridW - 1) + x] || cutAt[H0 + y * gridW + x + 1];      // (b,c,d): b-c, c-d, d-b
    if (cutTri) continue;
    keep[t] = 1; kept++;
  }
  if (kept === indices.length / 3) return { positions, uvs, indices };

  // Compact: keep only the vertices some surviving triangle references.
  const remap = new Int32Array(n).fill(-1);
  const outIdx = new Uint32Array(kept * 3);
  let vn = 0, k = 0;
  for (let t = 0, i = 0; i < indices.length; t++, i += 3) {
    if (!keep[t]) continue;
    for (let j = 0; j < 3; j++) {
      const v = indices[i + j]!;
      let r = remap[v]!;
      if (r < 0) { r = vn++; remap[v] = r; }
      outIdx[k++] = r;
    }
  }
  const outPos = new Float32Array(vn * 3);
  const outUv = new Float32Array(vn * 2);
  for (let v = 0; v < n; v++) {
    const r = remap[v]!;
    if (r < 0) continue;
    outPos[r * 3] = positions[v * 3]!; outPos[r * 3 + 1] = positions[v * 3 + 1]!; outPos[r * 3 + 2] = positions[v * 3 + 2]!;
    outUv[r * 2] = uvs[v * 2]!; outUv[r * 2 + 1] = uvs[v * 2 + 1]!;
  }
  return { positions: outPos, uvs: outUv, indices: outIdx };
}

/**
 * The fill layer (depth-layers.ts) as geometry. `plate` is a second grid over the same lattice
 * whose UVs address the fill plate's region of the atlas.
 *
 * Culled: only the band is meshed. A cell joins when all four corners are band vertices or anchors
 * (the far ends of the cuts, at their own depth, which is what stitches the fill to the background
 * it continues) and at least one is a band vertex. An anchor stays at its own depth even when it
 * also lies inside another silhouette's band (a background vertex beside a subject is often the
 * near side of a cut further back); measured on a film frame, three anchors in four were. Edges
 * between band vertices are never cut: the fill is hidden except through a gap, and a slanted fill
 * there reads as background where a cut one reads as a hole. An edge that reaches an anchor keeps
 * the relief's edge rule, so the fill stitches only to the background it was seeded from.
 *
 * Sheets: the whole lattice again, fixed topology, the band at the fill depth and every other
 * vertex just behind the relief's own surface, where it can never be seen except through a cut.
 */
export function fillLayerToMesh(fill: FillLayer, z: Float32Array, plate: DepthGrid, opts: { edge: number; sheets?: boolean }): EncodeMeshFrame {
  const { gridW, gridH, rayX, rayY, uvs, indices } = plate;
  const n = gridW * gridH;
  const edge = opts.edge;
  const lz = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const band = fill.dist[i]! >= 0 && !fill.anchor[i];
    lz[i] = band ? fill.z[i]! : opts.sheets ? z[i]! * 1.01 : z[i]!;
  }
  const place = (src: number, out: Float32Array, dst: number) => {
    out[dst * 3] = rayX[src]! * lz[src]!; out[dst * 3 + 1] = -rayY[src]! * lz[src]!; out[dst * 3 + 2] = -lz[src]!;
  };
  if (opts.sheets) {
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) place(i, positions, i);
    return { positions, uvs, indices };
  }
  const member = (i: number) => fill.dist[i]! >= 0 || fill.anchor[i]! === 1;
  const inBand = (i: number) => fill.dist[i]! >= 0 && !fill.anchor[i];
  const broken = (a: number, b: number) => {
    if (inBand(a) && inBand(b)) return false;
    const za = lz[a]!, zb = lz[b]!;
    return Math.abs(za - zb) > edge * (za < zb ? za : zb);
  };
  const remap = new Int32Array(n).fill(-1);
  const idx: number[] = [];
  let vn = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i]!, b = indices[i + 1]!, c = indices[i + 2]!;
    if (!member(a) || !member(b) || !member(c)) continue;
    if (!inBand(a) && !inBand(b) && !inBand(c)) continue;
    if (broken(a, b) || broken(b, c) || broken(c, a)) continue;
    for (const v of [a, b, c]) { if (remap[v]! < 0) remap[v] = vn++; idx.push(remap[v]!); }
  }
  const positions = new Float32Array(vn * 3), outUv = new Float32Array(vn * 2);
  for (let v = 0; v < n; v++) {
    const r = remap[v]!;
    if (r < 0) continue;
    place(v, positions, r);
    outUv[r * 2] = uvs[v * 2]!; outUv[r * 2 + 1] = uvs[v * 2 + 1]!;
  }
  return { positions, uvs: outUv, indices: Uint32Array.from(idx) };
}

/** Concatenate two meshes into one frame (the relief and its fill layer). */
export function mergeMeshes(a: EncodeMeshFrame, b: EncodeMeshFrame): EncodeMeshFrame {
  const va = a.positions.length / 3;
  const positions = new Float32Array(a.positions.length + b.positions.length);
  positions.set(a.positions); positions.set(b.positions, a.positions.length);
  const uvs = new Float32Array((va + b.positions.length / 3) * 2);
  if (a.uvs) uvs.set(a.uvs);
  if (b.uvs) uvs.set(b.uvs, va * 2);
  const indices = new Uint32Array(a.indices.length + b.indices.length);
  indices.set(a.indices);
  for (let i = 0; i < b.indices.length; i++) indices[a.indices.length + i] = b.indices[i]! + va;
  return { positions, uvs, indices };
}

/**
 * The depth distribution of a clip's relief surface, taken a frame at a time so the clip is never
 * held: a histogram over inverse depth between `far` and `near`, the space the depth maps are linear
 * in. The player frames and sways a relief from it (`relief.pivot`, `relief.near`, `relief.far`).
 */
export class DepthHistogram {
  private readonly bins: Float64Array;
  private readonly lo: number;
  private readonly perUnit: number;
  private total = 0;

  constructor(near: number, private readonly far: number, bins = 4096) {
    this.bins = new Float64Array(bins);
    this.lo = 1 / far;
    this.perUnit = bins / (1 / near - 1 / far);
  }

  /** Vertices counted so far. */
  get count(): number { return this.total; }

  /** Count every vertex of a capture-space mesh; vertices parked on the sheet backdrop are skipped. */
  add(positions: Float32Array): void {
    const b = this.bins, n = b.length, parked = this.far * BACKDROP;
    for (let i = 2; i < positions.length; i += 3) {
      const z = -positions[i]!;
      if (!(z > 0) || z >= parked) continue;
      const k = Math.floor((1 / z - this.lo) * this.perUnit);
      b[k < 0 ? 0 : k >= n ? n - 1 : k]!++;
      this.total++;
    }
  }

  /** The depth a fraction `q` of the counted vertices lies nearer than (0.5 is the median); 0 when empty. */
  nearest(q: number): number {
    if (!this.total) return 0;
    const want = Math.min(1, Math.max(0, q)) * this.total;
    let acc = 0;
    for (let k = this.bins.length - 1; k >= 0; k--) {
      acc += this.bins[k]!;
      if (acc >= want && acc > 0) return 1 / (this.lo + (k + 0.5) / this.perUnit);
    }
    return this.far;
  }

  /**
   * Where to orbit a relief: the midpoint in DISPARITY of the nearest and farthest 5 %. Parallax is
   * linear in inverse depth, so about this depth the near and far ends move by the same angle in
   * opposite directions, and the largest parallax any orbit angle produces is the smallest it can be.
   * The median depth is not that point: in a frame that is mostly distant background it sits near
   * the back, and a near subject then swings across the frame.
   */
  framing(): { pivot: number; near: number; far: number } | null {
    if (!this.total) return null;
    const near = this.nearest(0.05), far = this.nearest(0.95);
    return { pivot: 2 / (1 / near + 1 / far), near, far };
  }
}

/**
 * Draw-time discard parameters for a relief, in CAPTURE space (camera at the origin looking down
 * -Z); the CLI carries them through the model transform into the clip's `relief.*` metadata.
 *
 * `slope` is the sine of the smallest angle between a surface and the capture ray through it that
 * still counts as surface. A triangle spanning a silhouette lies almost along the ray: across one
 * cell of angular size `cell`, a relative depth jump of `edge` tilts it to atan(cell / edge), so
 * that is the threshold, the draw-time equivalent of the `edge` cull.
 * `depthMax` is the depth beyond which nothing is drawn: sheets park masked-out vertices past it.
 */
export function reliefDiscard(gridH: number, fovDeg: number, edge: number, far: number): { slope: number; depthMax: number } {
  const cell = (2 * Math.tan((fovDeg * Math.PI) / 180 / 2)) / gridH;
  return { slope: edge > 0 ? cell / Math.hypot(cell, edge) : 0, depthMax: far * (1 + (BACKDROP - 1) / 2) };
}
