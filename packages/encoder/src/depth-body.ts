/**
 * Full-body completion for a 2D clip: the capture's metric relief shell is the visible side, a SAM 3D
 * Body (MHR) mesh anchored behind it is the unseen side, and the unseen side's texture is accumulated
 * from every frame in which that part of the body faced the camera.
 *
 * Ported from the Depthkit "3Dify" lab scripts: the shell mesher from depthkit-extract7.mjs:389-448,
 * the anchoring idea from dk-body4d-build.mjs, the carve and backing from dk-hybrid-build.mjs, the
 * patch atlas from asset-bake-atlas.mjs. Measured there (dk01-v10, 300 frames, build-v10.log): per frame
 * about 23,583 covered and 928 flap triangles dropped, 7,843 vertices pushed behind the shell, and
 * 5,681..18,456 of MHR's 36,874 triangles kept. Changes against the lab, each for a recorded defect:
 *
 *  - Coordinates are OpenCV camera space throughout (x right, y down, z forward, metres), pixel
 *    (u, v) = (fx*x/z + cx, fy*y/z + cy) with pixel centres at integer + 0.5. SAM 3D Body's
 *    pred_vertices + pred_cam_t are already in this space (probe 2026-09-19: projecting with the
 *    supplied K lands on the subject's box). ARES world space (x, -y, -z) is applied only at
 *    emission, by toAresSpace; that map is a rotation, so winding survives it.
 *  - The flap filter uses the data-derived outward sign (outwardSign). The lab tested the z of
 *    (b-a)x(c-a) on a z-negated mesh whose faces then pointed inward, so it dropped the back-facing
 *    out-of-mask triangles and kept the camera-facing flaps it was written to remove.
 *  - A pushed vertex is scaled along its camera ray to z_shell + pushGap. The lab shifted z alone,
 *    which moves the vertex off its pixel and opens a gap at the silhouette.
 *  - The backing's rim distance is uncapped. The lab stopped the BFS at 15 cells and skipped every
 *    deeper cell, which left a hole in any backing region wider than about 30 cells.
 *  - "Body behind the shell" is read from a far z-buffer of the KEPT body triangles as well as from
 *    their vertices. In the lab a limb between `back` and `cover` thick lost both sides to the carve
 *    while its back vertices still suppressed the backing, which left the limb open from behind.
 *  - The unseen side's texture comes from BodyColorAccumulator and bakeBackAtlas. The lab textured
 *    the back with the front pixel on the same camera ray, which is a smear.
 *
 * Measured 2026-09-19 (Node 22.19, one SAM 3D Body probe mesh, 18,439 vertices and 36,874 faces,
 * against a shell made from its own front z-buffer at 518x292; warm medians): rasterizeDepth 1.8 ms,
 * assembleHybrid 4.8 ms, BodyColorAccumulator.add 3.8 ms, finalize 1.9 ms, shellMesh 3.0 ms. The
 * carve on that frame: 32,203 triangles covered, 8,867 vertices pushed, 4,671 kept (22,277 of the
 * 22,491 camera-facing triangles carved; 4,457 of the 14,383 facing away kept, the rest lie within
 * `cover` of the front), 441 backing cells; 107 flaps once the mask is eroded by 4 px.
 *
 * All functions are per frame and allocation-light; nothing here holds a clip.
 */
import type { EncodeMeshFrame } from "./geometry-encode.js";
// Pinhole intrinsics in map pixels. One declaration (depth-metric.ts) so index.ts can re-export both
// modules without an ambiguous name.
import type { Intrinsics } from "./depth-metric.js";

/** Nearest z a vertex may have and still be projected (metres). Anything nearer is behind the camera
 *  for every purpose here: a triangle touching it is not rasterized and a vertex is "outside". */
const Z_MIN = 1e-3;

/* ------------------------------------------------------------------------------------------------ */
/* validation                                                                                        */
/* ------------------------------------------------------------------------------------------------ */

function checkMap(fn: string, W: number, H: number): void {
  if (!Number.isInteger(W) || !Number.isInteger(H) || W < 1 || H < 1)
    throw new Error(`${fn}: map size must be positive integers, got ${W}x${H}`);
}

function checkLen(fn: string, name: string, arr: ArrayLike<number>, n: number): void {
  if (arr.length < n) throw new Error(`${fn}: ${name} holds ${arr.length} values, expected at least ${n}`);
}

function checkK(fn: string, K: Intrinsics): void {
  if (!K || !(K.fx > 0) || !(K.fy > 0) || !Number.isFinite(K.fx) || !Number.isFinite(K.fy) || !Number.isFinite(K.cx) || !Number.isFinite(K.cy))
    throw new Error(`${fn}: intrinsics need finite fx > 0, fy > 0, cx, cy; got fx ${K?.fx} fy ${K?.fy} cx ${K?.cx} cy ${K?.cy}`);
}

function checkFrameUv(fn: string, uv: { vScale: number; vOffset: number }): void {
  if (!uv || !(uv.vScale > 0) || !Number.isFinite(uv.vScale) || !Number.isFinite(uv.vOffset))
    throw new Error(`${fn}: frame uv needs finite vScale > 0 and vOffset, got vScale ${uv?.vScale} vOffset ${uv?.vOffset}`);
}

/** An optional numeric option: default when undefined, else finite and within [lo, hi]. */
function optNum(fn: string, name: string, v: number | undefined, def: number, lo: number, hi = Infinity, integer = false): number {
  if (v === undefined) return def;
  if (typeof v !== "number" || !Number.isFinite(v) || v < lo || v > hi || (integer && !Number.isInteger(v)))
    throw new Error(`${fn}: ${name} must be ${integer ? "an integer" : "a number"} in [${lo}, ${hi}], got ${v}`);
  return v;
}

function checkFaces(fn: string, faces: Uint32Array, vertexCount: number): void {
  if (faces.length % 3 !== 0) throw new Error(`${fn}: faces holds ${faces.length} indices, not a multiple of 3`);
  for (let i = 0; i < faces.length; i++) {
    if (faces[i]! >= vertexCount) throw new Error(`${fn}: face index ${faces[i]} at ${i} is out of range for ${vertexCount} vertices`);
  }
}

function checkPositions(fn: string, positions: Float32Array): number {
  if (positions.length % 3 !== 0) throw new Error(`${fn}: positions holds ${positions.length} floats, not a multiple of 3`);
  return positions.length / 3;
}

function checkSign(fn: string, sign: number): void {
  if (sign !== 1 && sign !== -1) throw new Error(`${fn}: sign must be 1 or -1, got ${sign}`);
}

const isDepth = (z: number): boolean => z > 0 && z < Infinity;

/* ------------------------------------------------------------------------------------------------ */
/* shell mesher (depthkit-extract7.mjs:389-448)                                                      */
/* ------------------------------------------------------------------------------------------------ */

export interface ShellOptions {
  /** Largest z span of one triangle, metres. Default 0.05 (x7 EDGE_MM 50). */
  edge?: number;
  /** Smallest |cos| between a face normal and the camera ray through its centroid. Default 0.2
   *  (x7 SHEER_MIN, which tested n_z; the ray form stays correct off the optical axis). */
  sheer?: number;
  /** Components smaller than this are dropped, except the largest. Default
   *  round(500 * W*H / (848*480)): x7's 500 cells scaled to the map's area. */
  minCells?: number;
  /** Erosion passes: a cell on the map border or missing a 4-neighbour is removed. Default 1. */
  erode?: number;
}

export interface Shell {
  /** OpenCV camera metres, one vertex per surviving cell, at the cell centre. */
  positions: Float32Array;
  /** ((x+0.5)/W, vOffset + (y+0.5)/H * vScale): the frame's region of the atlas, v = 0 at the top. */
  uvs: Float32Array;
  /** Quad (a=(x,y), b=(x+1,y), c=(x,y+1), d=(x+1,y+1)) -> (a,c,b), (b,c,d): (p1-p0)x(p2-p0) points
   *  back at the camera, the same convention as depth-mesh.ts's buildDepthGrid. */
  indices: Uint32Array;
  /** cellIndex[y*W+x] = vertex index, or -1. */
  cellIndex: Int32Array;
  /** Surviving cells = vertex count. */
  cells: number;
}

/**
 * One metric depth map (metres, 0 or non-finite = no data) -> the capture's relief shell.
 *
 * Cells are eroded, split into 4-connected components (every component of minCells or more is kept,
 * plus the largest), and unprojected at their centres. A triangle is kept when all three cells
 * survive, its z span is within `edge`, and it is not seen edge-on (|cos| to the camera ray at its
 * centroid of `sheer` or more). A cell whose triangles are all dropped keeps its vertex so that
 * cellIndex stays one vertex per surviving cell.
 */
export function shellMesh(z: Float32Array, W: number, H: number, K: Intrinsics, uv: { vScale: number; vOffset: number }, o: ShellOptions = {}): Shell {
  const fn = "shellMesh";
  checkMap(fn, W, H);
  const n = W * H;
  checkLen(fn, "z", z, n);
  checkK(fn, K);
  checkFrameUv(fn, uv);
  const edge = optNum(fn, "edge", o.edge, 0.05, 0);
  const sheer = optNum(fn, "sheer", o.sheer, 0.2, 0, 1);
  const minCells = optNum(fn, "minCells", o.minCells, Math.round((500 * n) / (848 * 480)), 0, Infinity, true);
  const erode = optNum(fn, "erode", o.erode, 1, 0, Infinity, true);

  let cur = new Uint8Array(n);
  for (let i = 0; i < n; i++) cur[i] = isDepth(z[i]!) ? 1 : 0;
  for (let e = 0; e < erode; e++) {
    const next = new Uint8Array(n);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        if (cur[i] && cur[i - 1] && cur[i + 1] && cur[i - W] && cur[i + W]) next[i] = 1;
      }
    }
    cur = next;
  }

  // 4-connected components over the surviving cells.
  const comp = new Int32Array(n).fill(-1);
  const sizes: number[] = [];
  const stack = new Int32Array(n);
  for (let seed = 0; seed < n; seed++) {
    if (!cur[seed] || comp[seed]! >= 0) continue;
    const id = sizes.length;
    let sp = 0, count = 0;
    stack[sp++] = seed; comp[seed] = id;
    while (sp > 0) {
      const i = stack[--sp]!;
      count++;
      const x = i % W;
      if (x > 0 && cur[i - 1] && comp[i - 1]! < 0) { comp[i - 1] = id; stack[sp++] = i - 1; }
      if (x < W - 1 && cur[i + 1] && comp[i + 1]! < 0) { comp[i + 1] = id; stack[sp++] = i + 1; }
      if (i >= W && cur[i - W] && comp[i - W]! < 0) { comp[i - W] = id; stack[sp++] = i - W; }
      if (i + W < n && cur[i + W] && comp[i + W]! < 0) { comp[i + W] = id; stack[sp++] = i + W; }
    }
    sizes.push(count);
  }
  let largest = -1, largestN = 0;
  for (let c = 0; c < sizes.length; c++) if (sizes[c]! > largestN) { largestN = sizes[c]!; largest = c; }
  const keep = new Uint8Array(sizes.length);
  for (let c = 0; c < sizes.length; c++) keep[c] = sizes[c]! >= minCells || c === largest ? 1 : 0;

  const cellIndex = new Int32Array(n).fill(-1);
  let nv = 0;
  for (let i = 0; i < n; i++) if (cur[i] && keep[comp[i]!]) cellIndex[i] = nv++;
  const positions = new Float32Array(nv * 3);
  const uvs = new Float32Array(nv * 2);
  const { fx, fy, cx, cy } = K;
  for (let y = 0; y < H; y++) {
    const py = y + 0.5;
    for (let x = 0; x < W; x++) {
      const i = y * W + x, k = cellIndex[i]!;
      if (k < 0) continue;
      const px = x + 0.5, zz = z[i]!;
      positions[k * 3] = ((px - cx) * zz) / fx;
      positions[k * 3 + 1] = ((py - cy) * zz) / fy;
      positions[k * 3 + 2] = zz;
      uvs[k * 2] = px / W;
      uvs[k * 2 + 1] = uv.vOffset + (py / H) * uv.vScale;
    }
  }

  // At most 2 triangles per cell (each quad is owned by its top-left cell).
  const idx = new Uint32Array(nv * 6);
  let t = 0;
  const tri = (i: number, j: number, k: number): void => {
    const a = cellIndex[i]!, b = cellIndex[j]!, c = cellIndex[k]!;
    if (a < 0 || b < 0 || c < 0) return;
    const za = positions[a * 3 + 2]!, zb = positions[b * 3 + 2]!, zc = positions[c * 3 + 2]!;
    if (Math.max(za, zb, zc) - Math.min(za, zb, zc) > edge) return;
    const ax = positions[a * 3]!, ay = positions[a * 3 + 1]!;
    const ux = positions[b * 3]! - ax, uy = positions[b * 3 + 1]! - ay, uz = zb - za;
    const vx = positions[c * 3]! - ax, vy = positions[c * 3 + 1]! - ay, vz = zc - za;
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const gx = ax + (ux + vx) / 3, gy = ay + (uy + vy) / 3, gz = za + (uz + vz) / 3;
    const nl = Math.hypot(nx, ny, nz), gl = Math.hypot(gx, gy, gz);
    if (!(nl > 0) || Math.abs(nx * gx + ny * gy + nz * gz) < sheer * nl * gl) return;
    idx[t++] = a; idx[t++] = b; idx[t++] = c;
  };
  for (let y = 0; y < H - 1; y++) {
    for (let x = 0; x < W - 1; x++) {
      const a = y * W + x;
      if (cellIndex[a]! < 0 && cellIndex[a + 1]! < 0) continue;
      const b = a + 1, c = a + W, d = c + 1;
      tri(a, c, b); tri(b, c, d);
    }
  }
  return { positions, uvs, indices: idx.slice(0, t), cellIndex, cells: nv };
}

/* ------------------------------------------------------------------------------------------------ */
/* z-buffer                                                                                          */
/* ------------------------------------------------------------------------------------------------ */

/**
 * Scanline-free pixel-centre rasterizer shared by rasterizeDepth and the hybrid's far buffer.
 * Both windings are drawn. `far` keeps the largest z per pixel instead of the smallest; `keep`
 * restricts the pass to the flagged triangles. Depth is perspective-correct: 1/z is interpolated in
 * screen space. A triangle with any vertex at z <= Z_MIN is skipped (no near-plane clipping).
 */
function rasterPass(positions: Float32Array, indices: Uint32Array, W: number, H: number, K: Intrinsics,
  zb: Float32Array, face: Int32Array | null, far: boolean, keep: Uint8Array | null): void {
  const { fx, fy, cx, cy } = K;
  const T = (indices.length / 3) | 0;
  for (let t = 0; t < T; t++) {
    if (keep && !keep[t]) continue;
    const i0 = indices[t * 3]! * 3, i1 = indices[t * 3 + 1]! * 3, i2 = indices[t * 3 + 2]! * 3;
    const z0 = positions[i0 + 2]!, z1 = positions[i1 + 2]!, z2 = positions[i2 + 2]!;
    if (!(z0 > Z_MIN && z1 > Z_MIN && z2 > Z_MIN)) continue;
    const u0 = (fx * positions[i0]!) / z0 + cx, v0 = (fy * positions[i0 + 1]!) / z0 + cy;
    const u1 = (fx * positions[i1]!) / z1 + cx, v1 = (fy * positions[i1 + 1]!) / z1 + cy;
    const u2 = (fx * positions[i2]!) / z2 + cx, v2 = (fy * positions[i2 + 1]!) / z2 + cy;
    const area = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0);
    if (!(Math.abs(area) > 1e-12)) continue;
    const xa = Math.max(0, Math.ceil(Math.min(u0, u1, u2) - 0.5));
    const xb = Math.min(W - 1, Math.floor(Math.max(u0, u1, u2) - 0.5));
    const ya = Math.max(0, Math.ceil(Math.min(v0, v1, v2) - 0.5));
    const yb = Math.min(H - 1, Math.floor(Math.max(v0, v1, v2) - 0.5));
    if (xa > xb || ya > yb) continue;
    const inv = 1 / area, iz0 = 1 / z0, iz1 = 1 / z1, iz2 = 1 / z2;
    // Barycentric weights are affine in (sx, sy): w = A*sx + B*sy + C.
    const A0 = -(v2 - v1) * inv, B0 = (u2 - u1) * inv, C0 = ((v2 - v1) * u1 - (u2 - u1) * v1) * inv;
    const A1 = -(v0 - v2) * inv, B1 = (u0 - u2) * inv, C1 = ((v0 - v2) * u2 - (u0 - u2) * v2) * inv;
    const A2 = -(v1 - v0) * inv, B2 = (u1 - u0) * inv, C2 = ((v1 - v0) * u0 - (u1 - u0) * v0) * inv;
    for (let py = ya; py <= yb; py++) {
      const sy = py + 0.5, row = py * W;
      for (let px = xa; px <= xb; px++) {
        const sx = px + 0.5;
        const w0 = A0 * sx + B0 * sy + C0;
        if (w0 < 0) continue;
        const w1 = A1 * sx + B1 * sy + C1;
        if (w1 < 0) continue;
        const w2 = A2 * sx + B2 * sy + C2;
        if (w2 < 0) continue;
        const zz = 1 / (w0 * iz0 + w1 * iz1 + w2 * iz2);
        const p = row + px;
        if (far ? zz > zb[p]! : zz < zb[p]!) { zb[p] = zz; if (face) face[p] = t; }
      }
    }
  }
}

/**
 * Z-buffer of the nearest surface per map pixel: Infinity where nothing covers the pixel centre.
 * `faceOut` receives the triangle id per pixel, or -1. Both windings are drawn.
 */
export function rasterizeDepth(positions: Float32Array, indices: Uint32Array, W: number, H: number, K: Intrinsics,
  out?: Float32Array, faceOut?: Int32Array): Float32Array {
  const fn = "rasterizeDepth";
  checkMap(fn, W, H);
  checkK(fn, K);
  const V = checkPositions(fn, positions);
  checkFaces(fn, indices, V);
  const n = W * H;
  if (out) checkLen(fn, "out", out, n);
  if (faceOut) checkLen(fn, "faceOut", faceOut, n);
  const zb = out ?? new Float32Array(n);
  zb.fill(Infinity, 0, n);
  if (faceOut) faceOut.fill(-1, 0, n);
  rasterPass(positions, indices, W, H, K, zb, faceOut ?? null, false, null);
  return zb;
}

/**
 * Sign of the mesh's signed volume, taken about the vertex centroid: +1 when (b-a)x(c-a) points
 * outward. The lab measured +0.052 m^3 on an MHR export (outward) and the sign reversing under a
 * single-axis flip; reading it from the data keeps every facing test right whichever export arrives.
 */
export function outwardSign(positions: Float32Array, indices: Uint32Array): 1 | -1 {
  const fn = "outwardSign";
  const V = checkPositions(fn, positions);
  checkFaces(fn, indices, V);
  if (V === 0 || indices.length === 0) throw new Error(`${fn}: empty mesh (${V} vertices, ${indices.length / 3} faces)`);
  let ox = 0, oy = 0, oz = 0;
  for (let v = 0; v < V; v++) { ox += positions[v * 3]!; oy += positions[v * 3 + 1]!; oz += positions[v * 3 + 2]!; }
  ox /= V; oy /= V; oz /= V;
  let vol = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]! * 3, b = indices[t + 1]! * 3, c = indices[t + 2]! * 3;
    const ax = positions[a]! - ox, ay = positions[a + 1]! - oy, az = positions[a + 2]! - oz;
    const bx = positions[b]! - ox, by = positions[b + 1]! - oy, bz = positions[b + 2]! - oz;
    const cx = positions[c]! - ox, cy = positions[c + 1]! - oy, cz = positions[c + 2]! - oz;
    vol += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
  }
  if (!Number.isFinite(vol) || vol === 0) throw new Error(`${fn}: signed volume is ${vol / 6}; the mesh is flat, open to no side, or holds non-finite positions`);
  return vol > 0 ? 1 : -1;
}

/* ------------------------------------------------------------------------------------------------ */
/* anchoring                                                                                         */
/* ------------------------------------------------------------------------------------------------ */

export interface AnchorResult {
  /** Multiply the body's positions by this (about the camera origin). */
  scale: number;
  /** Pixels where body, shell and mask all exist. */
  shared: number;
  /** median |shellZ - scale*bodyZ| over those pixels, metres. */
  mad: number;
}

function medianOf(a: Float64Array): number {
  a.sort();
  const m = a.length >> 1;
  return a.length % 2 ? a[m]! : (a[m - 1]! + a[m]!) / 2;
}

/**
 * The scale about the camera origin that lays the body's front surface on the shell: the median of
 * shellZ / bodyZ over the pixels where both exist and the mask is on. Scaling about the origin keeps
 * every vertex on its own camera ray, so the body's silhouette stays where the image put it and only
 * its distance changes; the lab's translate-to-medians anchor moved the silhouette instead.
 * `bodyZ` is rasterizeDepth of the body (Infinity = empty); `shellZ` holds 0 or non-finite where
 * there is no shell. Null below `minShared` pixels (default 200).
 */
export function anchorScale(bodyZ: Float32Array, shellZ: Float32Array, mask: Uint8Array, W: number, H: number, minShared = 200): AnchorResult | null {
  const fn = "anchorScale";
  checkMap(fn, W, H);
  const n = W * H;
  checkLen(fn, "bodyZ", bodyZ, n);
  checkLen(fn, "shellZ", shellZ, n);
  checkLen(fn, "mask", mask, n);
  optNum(fn, "minShared", minShared, 200, 1, Infinity, true);
  let k = 0;
  for (let i = 0; i < n; i++) if (mask[i] && isDepth(bodyZ[i]!) && isDepth(shellZ[i]!)) k++;
  if (k < minShared) return null;
  const ratio = new Float64Array(k), bz = new Float64Array(k), sz = new Float64Array(k);
  k = 0;
  for (let i = 0; i < n; i++) {
    const b = bodyZ[i]!, s = shellZ[i]!;
    if (!mask[i] || !isDepth(b) || !isDepth(s)) continue;
    bz[k] = b; sz[k] = s; ratio[k] = s / b; k++;
  }
  const scale = medianOf(ratio);
  for (let j = 0; j < k; j++) ratio[j] = Math.abs(sz[j]! - scale * bz[j]!);
  return { scale, shared: k, mad: medianOf(ratio) };
}

/**
 * Temporal smoothing of a per-frame scalar (the anchor scale): a median over +-2 frames of the valid
 * entries, invalid entries then filled by linear interpolation between the nearest valid neighbours
 * (held flat past the ends), then a Gaussian over +-radius frames with sigma = radius/2, weights
 * renormalized at the clip ends. radius 0 skips the Gaussian. Throws when no entry is valid.
 */
export function smoothSeries(values: Float64Array, valid: Uint8Array, radius: number): Float64Array {
  const fn = "smoothSeries";
  const n = values.length;
  checkLen(fn, "valid", valid, n);
  optNum(fn, "radius", radius, 0, 0);
  const ok = (i: number): boolean => !!valid[i] && Number.isFinite(values[i]!);
  const med = new Float64Array(n);
  const have = new Uint8Array(n);
  const buf = new Float64Array(5);
  let any = false;
  for (let i = 0; i < n; i++) {
    if (!ok(i)) continue;
    let m = 0;
    for (let j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) if (ok(j)) buf[m++] = values[j]!;
    const w = buf.subarray(0, m);
    med[i] = medianOf(w);
    have[i] = 1;
    any = true;
  }
  if (n === 0) return med;
  if (!any) throw new Error(`${fn}: no valid entry among ${n}`);
  let prev = -1;
  for (let i = 0; i < n; i++) {
    if (!have[i]) continue;
    if (prev < 0) for (let j = 0; j < i; j++) med[j] = med[i]!;
    else for (let j = prev + 1; j < i; j++) med[j] = med[prev]! + ((med[i]! - med[prev]!) * (j - prev)) / (i - prev);
    prev = i;
  }
  for (let j = prev + 1; j < n; j++) med[j] = med[prev]!;
  const r = Math.floor(radius);
  if (r < 1) return med;
  const sigma = radius / 2, g = new Float64Array(2 * r + 1);
  for (let d = -r; d <= r; d++) g[d + r] = Math.exp(-(d * d) / (2 * sigma * sigma));
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, ws = 0;
    for (let d = -r; d <= r; d++) {
      const j = i + d;
      if (j < 0 || j >= n) continue;
      s += g[d + r]! * med[j]!; ws += g[d + r]!;
    }
    out[i] = s / ws;
  }
  return out;
}

/**
 * Gaussian-weighted per-vertex average of a window of same-topology frames, centred on
 * window[center]: weight exp(-(k-center)^2 / (2 sigma^2)). sigma 0 returns a copy of the centre.
 * Fixed MHR topology is what makes this a plain array average.
 */
export function smoothVertexWindow(window: Float32Array[], center: number, sigma: number): Float32Array {
  const fn = "smoothVertexWindow";
  if (!Array.isArray(window) || window.length === 0) throw new Error(`${fn}: window must hold at least one frame, got ${window?.length}`);
  optNum(fn, "center", center, 0, 0, window.length - 1, true);
  optNum(fn, "sigma", sigma, 0, 0);
  const L = window[center]!.length;
  for (let k = 0; k < window.length; k++) {
    if (window[k]!.length !== L) throw new Error(`${fn}: frame ${k} holds ${window[k]!.length} floats, the centre holds ${L}`);
  }
  if (sigma === 0) return window[center]!.slice();
  const acc = new Float64Array(L);
  let ws = 0;
  for (let k = 0; k < window.length; k++) {
    const w = Math.exp(-((k - center) * (k - center)) / (2 * sigma * sigma));
    if (w < 1e-6) continue;
    const f = window[k]!;
    for (let i = 0; i < L; i++) acc[i] = acc[i]! + w * f[i]!;
    ws += w;
  }
  const out = new Float32Array(L);
  for (let i = 0; i < L; i++) out[i] = acc[i]! / ws;
  return out;
}

/* ------------------------------------------------------------------------------------------------ */
/* hybrid assembly (dk-hybrid-build.mjs)                                                             */
/* ------------------------------------------------------------------------------------------------ */

export interface HybridOptions {
  /** A body vertex within this of the shell (or in front of it) is "covered", metres. Default 0.10. */
  cover?: number;
  /** A body vertex this far behind the shell marks "body behind" at its cell, metres. Default 0.06. */
  back?: number;
  /** Vertices nearer than shell + pushGap are scaled along their ray to shell + pushGap. Default 0.005. */
  pushGap?: number;
  /** Backing depth behind the shell at rimCells from the region rim and beyond, metres. Default 0.12. */
  bulge?: number;
  /** Backing depth at the rim, metres. Default 0.008. */
  bulgeMin?: number;
  /** Rim distance, in cells, over which the backing bulge ramps up. Default round(15 * W / 848). */
  rimCells?: number;
  /** How far past the mask, in pixels, a vertex may project before it is "outside". Default 2. */
  overshootPx?: number;
  /** Largest shell z span under one backing triangle, metres; a backing never bridges a silhouette
   *  the shell itself cut. Default 0.05 (ShellOptions.edge). */
  edge?: number;
}

export interface HybridResult {
  /** Kept MHR triangles, unwelded (3 vertices per face), wound outward ((p1-p0)x(p2-p0) points out
   *  of the body), positions pushed behind the shell, uvs from backUv. OpenCV camera metres. */
  body: EncodeMeshFrame;
  /** Hair and cloth backing behind shell cells with no body behind them, wound to face away from the
   *  camera, uvs = the cell's frame-region uv. OpenCV camera metres. */
  backing: EncodeMeshFrame;
  /** covered / flaps: triangles dropped by each rule; pushed: vertices moved behind the shell;
   *  keptTris: body triangles emitted; backingCells: backing vertices emitted. */
  stats: { covered: number; flaps: number; pushed: number; keptTris: number; backingCells: number };
}

const OUT = 1, COVER = 2;

/**
 * One frame of the hybrid: carve the anchored body where the shell already shows it, push what is
 * left behind the shell, and close hair and cloth (shell with no body behind) with a backing sheet.
 *
 * Each body vertex is classified at its projected cell against the nearest non-empty shell cell
 * within radius 2: outside (projects off the map, or lands on mask-off pixels farther than
 * overshootPx from the mask), covered (z < z_shell + cover), push (z < z_shell + pushGap: the vertex
 * is scaled by (z_shell + pushGap)/z, which keeps it on its ray), body-behind (z > z_shell + back).
 * A triangle with all three vertices covered is dropped; a triangle with an outside vertex is dropped
 * when its outward normal faces the camera (dot(n_out, centroid) < 0).
 *
 * Backing region: shell cells (shellZ > 0) where no kept body surface lies more than `back` behind the
 * shell, from a far z-buffer of the kept triangles and their vertices, grown by one 4-neighbour cell
 * (dk-hybrid-build.mjs:154-162). Each region cell gets a vertex at
 * z_shell + max(bulgeMin, bulge * min(1, dist/rimCells)), dist = 4-neighbour BFS distance to the
 * region rim, unprojected at the cell centre.
 *
 * `shellZ` is the depth the shell mesh was built from (metres, 0 or non-finite = no shell); `body`
 * and `faces` are the anchored MHR (OpenCV metres); `sign` = outwardSign of that topology; `backUv`
 * holds F x 3 x 2 atlas-normalized corner uvs (bakeBackAtlas's layout mapped into the atlas).
 */
export function assembleHybrid(shellZ: Float32Array, mask: Uint8Array, W: number, H: number, K: Intrinsics,
  body: Float32Array, faces: Uint32Array, sign: 1 | -1, backUv: Float32Array, frameUv: { vScale: number; vOffset: number },
  o: HybridOptions = {}): HybridResult {
  const fn = "assembleHybrid";
  checkMap(fn, W, H);
  const n = W * H;
  checkLen(fn, "shellZ", shellZ, n);
  checkLen(fn, "mask", mask, n);
  checkK(fn, K);
  const V = checkPositions(fn, body);
  checkFaces(fn, faces, V);
  checkSign(fn, sign);
  const F = faces.length / 3;
  checkLen(fn, "backUv", backUv, F * 6);
  checkFrameUv(fn, frameUv);
  const cover = optNum(fn, "cover", o.cover, 0.10, 0);
  const back = optNum(fn, "back", o.back, 0.06, 0);
  const pushGap = optNum(fn, "pushGap", o.pushGap, 0.005, 0);
  const bulge = optNum(fn, "bulge", o.bulge, 0.12, 0);
  const bulgeMin = optNum(fn, "bulgeMin", o.bulgeMin, 0.008, 0);
  const rimCells = optNum(fn, "rimCells", o.rimCells, Math.max(1, Math.round((15 * W) / 848)), 1, Infinity, true);
  const overshootPx = optNum(fn, "overshootPx", o.overshootPx, 2, 0);
  const edge = optNum(fn, "edge", o.edge, 0.05, 0);
  const { fx, fy, cx, cy } = K;

  const shellAt = (i: number): number => { const s = shellZ[i]!; return isDepth(s) ? s : 0; };
  // Nearest non-empty shell cell within a 5x5 window (Euclidean nearest; ties by scan order).
  const shellNear = (px: number, py: number): number => {
    const own = shellAt(py * W + px);
    if (own > 0) return own;
    let best = 0, bestD = Infinity;
    for (let dy = -2; dy <= 2; dy++) {
      const y = py + dy;
      if (y < 0 || y >= H) continue;
      for (let dx = -2; dx <= 2; dx++) {
        const x = px + dx;
        if (x < 0 || x >= W) continue;
        const s = shellAt(y * W + x), d = dx * dx + dy * dy;
        if (s > 0 && d < bestD) { bestD = d; best = s; }
      }
    }
    return best;
  };
  const rOver = Math.floor(overshootPx), r2Over = overshootPx * overshootPx;
  const maskNear = (px: number, py: number): boolean => {
    if (mask[py * W + px]) return true;
    for (let dy = -rOver; dy <= rOver; dy++) {
      const y = py + dy;
      if (y < 0 || y >= H) continue;
      for (let dx = -rOver; dx <= rOver; dx++) {
        const x = px + dx;
        if (x < 0 || x >= W || dx * dx + dy * dy > r2Over) continue;
        if (mask[y * W + x]) return true;
      }
    }
    return false;
  };

  // ---- 1. vertex classification and push ----
  const P = new Float32Array(body);
  const cls = new Uint8Array(V);
  const vCell = new Int32Array(V).fill(-1);
  const vBack = new Uint8Array(V);
  let pushed = 0;
  for (let v = 0; v < V; v++) {
    const x = body[v * 3]!, y = body[v * 3 + 1]!, z = body[v * 3 + 2]!;
    if (!(z > Z_MIN)) { cls[v] = OUT; continue; }
    const px = Math.floor((fx * x) / z + cx), py = Math.floor((fy * y) / z + cy);
    if (!(px >= 0 && px < W && py >= 0 && py < H)) { cls[v] = OUT; continue; }
    vCell[v] = py * W + px;
    let c = maskNear(px, py) ? 0 : OUT;
    const zs = shellNear(px, py);
    if (zs > 0) {
      if (z < zs + cover) c |= COVER;
      if (z < zs + pushGap) {
        const s = (zs + pushGap) / z;
        P[v * 3] = x * s; P[v * 3 + 1] = y * s; P[v * 3 + 2] = zs + pushGap;
        pushed++;
      }
      if (z > zs + back) vBack[v] = 1;
    }
    cls[v] = c;
  }

  // ---- 2. carve ----
  const keep = new Uint8Array(F);
  let covered = 0, flaps = 0, kept = 0;
  for (let t = 0; t < F; t++) {
    const a = faces[t * 3]!, b = faces[t * 3 + 1]!, c = faces[t * 3 + 2]!;
    const ca = cls[a]!, cb = cls[b]!, cc = cls[c]!;
    if (ca & cb & cc & COVER) { covered++; continue; }
    if ((ca | cb | cc) & OUT) {
      const ax = body[a * 3]!, ay = body[a * 3 + 1]!, az = body[a * 3 + 2]!;
      const ux = body[b * 3]! - ax, uy = body[b * 3 + 1]! - ay, uz = body[b * 3 + 2]! - az;
      const vx = body[c * 3]! - ax, vy = body[c * 3 + 1]! - ay, vz = body[c * 3 + 2]! - az;
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const gx = ax + (ux + vx) / 3, gy = ay + (uy + vy) / 3, gz = az + (uz + vz) / 3;
      if (sign * (nx * gx + ny * gy + nz * gz) < 0) { flaps++; continue; }
    }
    keep[t] = 1; kept++;
  }

  // ---- 3. where the kept body lies behind the shell ----
  const farZ = new Float32Array(n).fill(-Infinity);
  rasterPass(P, faces, W, H, K, farZ, null, true, keep);
  const behind = new Uint8Array(n);
  for (let i = 0; i < n; i++) { const s = shellAt(i); if (s > 0 && farZ[i]! > s + back) behind[i] = 1; }
  for (let t = 0; t < F; t++) {
    if (!keep[t]) continue;
    for (let k = 0; k < 3; k++) { const v = faces[t * 3 + k]!; if (vBack[v] && vCell[v]! >= 0) behind[vCell[v]!] = 1; }
  }
  const grown = new Uint8Array(behind);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (behind[i]) continue;
      if ((x > 0 && behind[i - 1]) || (x < W - 1 && behind[i + 1]) || (y > 0 && behind[i - W]) || (y < H - 1 && behind[i + W])) grown[i] = 1;
    }
  }

  // ---- 4. backing region, rim distance (uncapped BFS) ----
  const region = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (shellAt(i) > 0 && !grown[i]) region[i] = 1;
  const dist = new Int32Array(n).fill(-1);
  const queue = new Int32Array(n);
  let qe = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!region[i]) continue;
      if (x === 0 || y === 0 || x === W - 1 || y === H - 1 || !region[i - 1] || !region[i + 1] || !region[i - W] || !region[i + W]) {
        dist[i] = 0; queue[qe++] = i;
      }
    }
  }
  for (let qh = 0; qh < qe; qh++) {
    const i = queue[qh]!, x = i % W, d = dist[i]! + 1;
    if (x > 0 && region[i - 1] && dist[i - 1]! < 0) { dist[i - 1] = d; queue[qe++] = i - 1; }
    if (x < W - 1 && region[i + 1] && dist[i + 1]! < 0) { dist[i + 1] = d; queue[qe++] = i + 1; }
    if (i >= W && region[i - W] && dist[i - W]! < 0) { dist[i - W] = d; queue[qe++] = i - W; }
    if (i + W < n && region[i + W] && dist[i + W]! < 0) { dist[i + W] = d; queue[qe++] = i + W; }
  }

  // ---- 5. backing mesh ----
  const bIndex = new Int32Array(n).fill(-1);
  let nb = 0;
  for (let i = 0; i < n; i++) if (region[i]) bIndex[i] = nb++;
  const bPos = new Float32Array(nb * 3), bUv = new Float32Array(nb * 2);
  for (let y = 0; y < H; y++) {
    const py = y + 0.5;
    for (let x = 0; x < W; x++) {
      const i = y * W + x, k = bIndex[i]!;
      if (k < 0) continue;
      const px = x + 0.5;
      const Z = shellAt(i) + Math.max(bulgeMin, bulge * Math.min(1, dist[i]! / rimCells));
      bPos[k * 3] = ((px - cx) * Z) / fx; bPos[k * 3 + 1] = ((py - cy) * Z) / fy; bPos[k * 3 + 2] = Z;
      bUv[k * 2] = px / W; bUv[k * 2 + 1] = frameUv.vOffset + (py / H) * frameUv.vScale;
    }
  }
  const bIdx = new Uint32Array(nb * 6);
  let bt = 0;
  const btri = (i: number, j: number, k: number): void => {
    const a = bIndex[i]!, b = bIndex[j]!, c = bIndex[k]!;
    if (a < 0 || b < 0 || c < 0) return;
    const za = shellAt(i), zb = shellAt(j), zc = shellAt(k);
    if (Math.max(za, zb, zc) - Math.min(za, zb, zc) > edge) return;
    bIdx[bt++] = a; bIdx[bt++] = b; bIdx[bt++] = c;
  };
  for (let y = 0; y < H - 1; y++) {
    for (let x = 0; x < W - 1; x++) {
      const a = y * W + x;
      if (bIndex[a]! < 0 && bIndex[a + 1]! < 0) continue;
      const b = a + 1, c = a + W, d = c + 1;
      btri(a, b, c); btri(b, d, c); // reversed against the shell's (a,c,b), (b,c,d)
    }
  }

  // ---- 6. body output: unwelded, wound outward ----
  const oPos = new Float32Array(kept * 9), oUv = new Float32Array(kept * 6), oIdx = new Uint32Array(kept * 3);
  const order = sign === 1 ? [0, 1, 2] : [0, 2, 1];
  let w = 0;
  for (let t = 0; t < F; t++) {
    if (!keep[t]) continue;
    for (let k = 0; k < 3; k++) {
      const corner = order[k]!, v = faces[t * 3 + corner]!;
      oPos[w * 3] = P[v * 3]!; oPos[w * 3 + 1] = P[v * 3 + 1]!; oPos[w * 3 + 2] = P[v * 3 + 2]!;
      oUv[w * 2] = backUv[t * 6 + corner * 2]!; oUv[w * 2 + 1] = backUv[t * 6 + corner * 2 + 1]!;
      oIdx[w] = w;
      w++;
    }
  }

  return {
    body: { positions: oPos, uvs: oUv, indices: oIdx },
    backing: { positions: bPos, uvs: bUv, indices: bIdx.slice(0, bt) },
    stats: { covered, flaps, pushed, keptTris: kept, backingCells: nb },
  };
}

/* ------------------------------------------------------------------------------------------------ */
/* back texture                                                                                      */
/* ------------------------------------------------------------------------------------------------ */

/**
 * Per-vertex colour of the body, accumulated over every frame in which the vertex faced the camera
 * unoccluded. Holds 4 doubles per vertex (18,439 MHR vertices: 590 KB) and nothing per frame.
 */
export class BodyColorAccumulator {
  private readonly vertexCount: number;
  private readonly faces: Uint32Array;
  private readonly sum: Float64Array;
  private readonly weight: Float64Array;
  private readonly adjStart: Int32Array;
  private readonly adj: Int32Array;
  private readonly normals: Float32Array;

  constructor(vertexCount: number, faces: Uint32Array) {
    const fn = "BodyColorAccumulator";
    if (!Number.isInteger(vertexCount) || vertexCount < 1) throw new Error(`${fn}: vertexCount must be a positive integer, got ${vertexCount}`);
    checkFaces(fn, faces, vertexCount);
    this.vertexCount = vertexCount;
    this.faces = faces;
    this.sum = new Float64Array(vertexCount * 3);
    this.weight = new Float64Array(vertexCount);
    this.normals = new Float32Array(vertexCount * 3);
    // 1-ring adjacency (CSR, deduplicated).
    const deg = new Int32Array(vertexCount + 1);
    for (let i = 0; i < faces.length; i++) deg[faces[i]!] = deg[faces[i]!]! + 2;
    const start = new Int32Array(vertexCount + 1);
    for (let v = 0; v < vertexCount; v++) start[v + 1] = start[v]! + deg[v]!;
    const raw = new Int32Array(start[vertexCount]!);
    const fill = start.slice(0, vertexCount);
    const put = (v: number, q: number): void => { const at = fill[v]!; raw[at] = q; fill[v] = at + 1; };
    for (let t = 0; t < faces.length; t += 3) {
      const a = faces[t]!, b = faces[t + 1]!, c = faces[t + 2]!;
      put(a, b); put(a, c); put(b, a); put(b, c); put(c, a); put(c, b);
    }
    const adjStart = new Int32Array(vertexCount + 1);
    const adj = new Int32Array(raw.length);
    let m = 0;
    for (let v = 0; v < vertexCount; v++) {
      const seg = raw.subarray(start[v]!, start[v + 1]!).sort();
      adjStart[v] = m;
      for (let j = 0; j < seg.length; j++) if ((j === 0 || seg[j] !== seg[j - 1]) && seg[j] !== v) adj[m++] = seg[j]!;
    }
    adjStart[vertexCount] = m;
    this.adjStart = adjStart;
    this.adj = adj.slice(0, m);
  }

  /**
   * Adds one frame. A vertex counts when it projects inside the map onto a mask-on pixel, faces the
   * camera (outward vertex normal . direction to the camera > 0.2), and lies within max(0.02 m, 1% of
   * z) of `zbuf` (rasterizeDepth of this same body) at its pixel. Its colour is a mask-weighted
   * bilinear sample of `rgb` (W x H x 3), weighted by the facing cosine squared. Returns the number of
   * vertices that counted.
   */
  add(body: Float32Array, rgb: Uint8Array, mask: Uint8Array, W: number, H: number, K: Intrinsics, zbuf: Float32Array, sign: 1 | -1): number {
    const fn = "BodyColorAccumulator.add";
    checkMap(fn, W, H);
    const n = W * H, V = this.vertexCount;
    if (body.length !== V * 3) throw new Error(`${fn}: body holds ${body.length} floats, expected ${V * 3}`);
    checkLen(fn, "rgb", rgb, n * 3);
    checkLen(fn, "mask", mask, n);
    checkLen(fn, "zbuf", zbuf, n);
    checkK(fn, K);
    checkSign(fn, sign);
    const nrm = this.normals, faces = this.faces;
    nrm.fill(0);
    for (let t = 0; t < faces.length; t += 3) {
      const a = faces[t]! * 3, b = faces[t + 1]! * 3, c = faces[t + 2]! * 3;
      const ux = body[b]! - body[a]!, uy = body[b + 1]! - body[a + 1]!, uz = body[b + 2]! - body[a + 2]!;
      const vx = body[c]! - body[a]!, vy = body[c + 1]! - body[a + 1]!, vz = body[c + 2]! - body[a + 2]!;
      const nx = (uy * vz - uz * vy) * sign, ny = (uz * vx - ux * vz) * sign, nz = (ux * vy - uy * vx) * sign;
      nrm[a] = nrm[a]! + nx; nrm[a + 1] = nrm[a + 1]! + ny; nrm[a + 2] = nrm[a + 2]! + nz;
      nrm[b] = nrm[b]! + nx; nrm[b + 1] = nrm[b + 1]! + ny; nrm[b + 2] = nrm[b + 2]! + nz;
      nrm[c] = nrm[c]! + nx; nrm[c + 1] = nrm[c + 1]! + ny; nrm[c + 2] = nrm[c + 2]! + nz;
    }
    const { fx, fy, cx, cy } = K;
    let seen = 0;
    for (let v = 0; v < V; v++) {
      const x = body[v * 3]!, y = body[v * 3 + 1]!, z = body[v * 3 + 2]!;
      if (!(z > Z_MIN)) continue;
      const u = (fx * x) / z + cx, w = (fy * y) / z + cy;
      const px = Math.floor(u), py = Math.floor(w);
      if (!(px >= 0 && px < W && py >= 0 && py < H)) continue;
      const p = py * W + px;
      if (!mask[p]) continue;
      const nx = nrm[v * 3]!, ny = nrm[v * 3 + 1]!, nz = nrm[v * 3 + 2]!;
      const nl = Math.hypot(nx, ny, nz), pl = Math.hypot(x, y, z);
      if (!(nl > 0)) continue;
      const cos = -(nx * x + ny * y + nz * z) / (nl * pl);
      if (!(cos > 0.2)) continue;
      const zb = zbuf[p]!;
      if (!(zb < Infinity) || z - zb > Math.max(0.02, 0.01 * z)) continue;
      // Mask-weighted bilinear sample at the vertex's sub-pixel position.
      const sx = Math.min(W - 1, Math.max(0, u - 0.5)), sy = Math.min(H - 1, Math.max(0, w - 0.5));
      const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
      const ax = sx - x0, ay = sy - y0;
      let r = 0, g = 0, b = 0, ws = 0;
      const tap = (tx: number, ty: number, tw: number): void => {
        const q = ty * W + tx;
        if (tw <= 0 || !mask[q]) return;
        r += tw * rgb[q * 3]!; g += tw * rgb[q * 3 + 1]!; b += tw * rgb[q * 3 + 2]!; ws += tw;
      };
      tap(x0, y0, (1 - ax) * (1 - ay)); tap(x1, y0, ax * (1 - ay)); tap(x0, y1, (1 - ax) * ay); tap(x1, y1, ax * ay);
      if (!(ws > 0)) continue;
      const wt = cos * cos;
      this.sum[v * 3] = this.sum[v * 3]! + (wt * r) / ws;
      this.sum[v * 3 + 1] = this.sum[v * 3 + 1]! + (wt * g) / ws;
      this.sum[v * 3 + 2] = this.sum[v * 3 + 2]! + (wt * b) / ws;
      this.weight[v] = this.weight[v]! + wt;
      seen++;
    }
    return seen;
  }

  /**
   * Per-vertex colours (V x 3). Observed vertices take their weighted mean; each unobserved vertex
   * then takes the mean of its already-coloured 1-ring neighbours, one ring per pass for up to
   * `rings` passes (default 64); anything still uncoloured takes the mean observed colour (mid grey
   * when nothing was observed). Leaves the accumulator unchanged.
   */
  finalize(rings = 64): { colors: Uint8Array; observed: number; filled: number } {
    optNum("BodyColorAccumulator.finalize", "rings", rings, 64, 0, Infinity, true);
    const V = this.vertexCount;
    const col = new Float64Array(V * 3);
    const gen = new Int32Array(V).fill(-1);
    let observed = 0;
    const mean = [0, 0, 0];
    for (let v = 0; v < V; v++) {
      const w = this.weight[v]!;
      if (!(w > 0)) continue;
      for (let k = 0; k < 3; k++) { col[v * 3 + k] = this.sum[v * 3 + k]! / w; mean[k] = mean[k]! + col[v * 3 + k]!; }
      gen[v] = 0;
      observed++;
    }
    for (let k = 0; k < 3; k++) mean[k] = observed ? mean[k]! / observed : 128;
    let filled = 0;
    for (let r = 1; r <= rings; r++) {
      let added = 0;
      for (let v = 0; v < V; v++) {
        if (gen[v]! >= 0) continue;
        let s0 = 0, s1 = 0, s2 = 0, m = 0;
        for (let j = this.adjStart[v]!; j < this.adjStart[v + 1]!; j++) {
          const q = this.adj[j]!, gq = gen[q]!;
          if (gq < 0 || gq >= r) continue;
          s0 += col[q * 3]!; s1 += col[q * 3 + 1]!; s2 += col[q * 3 + 2]!; m++;
        }
        if (!m) continue;
        col[v * 3] = s0 / m; col[v * 3 + 1] = s1 / m; col[v * 3 + 2] = s2 / m;
        gen[v] = r;
        added++;
      }
      filled += added;
      if (!added) break;
    }
    const colors = new Uint8Array(V * 3);
    for (let v = 0; v < V; v++) {
      for (let k = 0; k < 3; k++) {
        const c = gen[v]! >= 0 ? col[v * 3 + k]! : mean[k]!;
        colors[v * 3 + k] = Math.max(0, Math.min(255, Math.round(c)));
      }
    }
    return { colors, observed, filled };
  }
}

/**
 * The body's back texture as a patch atlas (asset-bake-atlas.mjs): one 3x3 texel patch per face,
 * floor(width/3) patches per row. In a patch, corner texels (0,0), (2,0), (0,2) hold the colours of
 * the face's vertices a, b, c; (1,0), (0,1), (1,1) the means of ab, ac, bc, the texels whose centres
 * are the edges' midpoints; (2,1), (1,2), (2,2), outside the triangle, the face mean.
 * asset-bake-atlas.mjs put the face mean on (1,1) and the bc mean on (2,1): the midpoint of bc then
 * read the face mean, and the neighbour across bc read its own, a colour step along one edge of every
 * face. `pixels` is RGB, width x height x 3, v = 0 at the top row; unused texels hold the mean
 * vertex colour. `height` is rounded up to a multiple of 4 (at least 4). `uvs` (F x 3 x 2) are the
 * corner texel centres in REGION pixel coordinates, (px0 + 0.5, py0 + 0.5) for a and so on; the
 * integrator maps them into the atlas. 36,874 MHR faces at width 2048: 682 patches per row,
 * 55 rows, height 168.
 */
export function bakeBackAtlas(faces: Uint32Array, colors: Uint8Array, width: number): { height: number; pixels: Uint8Array; uvs: Float32Array } {
  const fn = "bakeBackAtlas";
  if (!Number.isInteger(width) || width < 3) throw new Error(`${fn}: width must be an integer >= 3, got ${width}`);
  if (colors.length % 3 !== 0) throw new Error(`${fn}: colors holds ${colors.length} bytes, not a multiple of 3`);
  const V = colors.length / 3;
  checkFaces(fn, faces, V);
  const F = faces.length / 3;
  const cols = Math.floor(width / 3), rows = Math.ceil(F / cols);
  const height = Math.max(4, Math.ceil((rows * 3) / 4) * 4);
  const pixels = new Uint8Array(width * height * 3);
  const mean = [0, 0, 0];
  for (let v = 0; v < V; v++) for (let k = 0; k < 3; k++) mean[k] = mean[k]! + colors[v * 3 + k]!;
  for (let k = 0; k < 3; k++) mean[k] = V ? Math.round(mean[k]! / V) : 128;
  for (let i = 0; i < width * height; i++) { pixels[i * 3] = mean[0]!; pixels[i * 3 + 1] = mean[1]!; pixels[i * 3 + 2] = mean[2]!; }
  const uvs = new Float32Array(F * 6);
  for (let t = 0; t < F; t++) {
    const px0 = (t % cols) * 3, py0 = Math.floor(t / cols) * 3;
    const a = faces[t * 3]! * 3, b = faces[t * 3 + 1]! * 3, c = faces[t * 3 + 2]! * 3;
    for (let k = 0; k < 3; k++) {
      const ca = colors[a + k]!, cb = colors[b + k]!, cc = colors[c + k]!;
      const avg = Math.round((ca + cb + cc) / 3);
      const put = (dx: number, dy: number, val: number): void => { pixels[((py0 + dy) * width + px0 + dx) * 3 + k] = val; };
      put(0, 0, ca); put(1, 0, Math.round((ca + cb) / 2)); put(2, 0, cb);
      put(0, 1, Math.round((ca + cc) / 2)); put(1, 1, Math.round((cb + cc) / 2)); put(2, 1, avg);
      put(0, 2, cc); put(1, 2, avg); put(2, 2, avg);
    }
    uvs[t * 6] = px0 + 0.5; uvs[t * 6 + 1] = py0 + 0.5;
    uvs[t * 6 + 2] = px0 + 2.5; uvs[t * 6 + 3] = py0 + 0.5;
    uvs[t * 6 + 4] = px0 + 0.5; uvs[t * 6 + 5] = py0 + 2.5;
  }
  return { height, pixels, uvs };
}

/** OpenCV camera space -> ARES world space: (x, -y, -z), a copy. A rotation, so winding is kept. */
export function toAresSpace(positions: Float32Array): Float32Array {
  checkPositions("toAresSpace", positions);
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    out[i] = positions[i]!; out[i + 1] = -positions[i + 1]!; out[i + 2] = -positions[i + 2]!;
  }
  return out;
}
