/**
 * Metric fusion and normal detail: the stabilized disparity estimate carried onto a metric target by
 * a tiled robust fit, then the relief below the fit's resolution added from a normal map by a
 * screened Poisson solve on the log-depth residual.
 *
 * The two depth sources of a 2D video disagree in complementary ways. The stabilized disparity
 * (depth-stabilize.ts) is coherent from frame to frame but affine-ambiguous: correct only up to an
 * unknown scale and shift. MoGe-2 depth is metric (measured 2026-09-19 on one 1920x1080 test frame:
 * fovY 38.0 deg, subject at 4.56 m, 526 ms per frame in fp32 on an RTX 2080 Ti) but inferred one
 * frame at a time, so its scale is free to change between frames. The fit takes the shape and the
 * timing from the estimate and the metres from the target. MoGe-2's normal map, from the same inference call, carries the folds and
 * features neither depth resolves.
 *
 * Ported from the Depthkit 3Dify lab scripts (depthkit-extract5/6/7.mjs; "x7" below), where the
 * metric target was the capture's sensor depth:
 *  - robustFit (x7:163-179): OLS, one trim at 3 * MAD + floor (no 1.4826 factor), OLS refit on the
 *    inliers. The residual is in metres of z for both models, so the two compete on one scale.
 *  - fitTileField (x7:190-256): one affine per frame leaves the target's local shape unexplained. On
 *    the sensor capture the lab measured the median fit residual at 47.0 mm global against 23.7 mm
 *    tiled (x5). x7 logged only the global figure; medresTiled here is the tiled one.
 *  - smoothTileFields: new. x7's sensor target did not flicker; a per-frame MoGe target does, and a
 *    per-frame field passes that flicker straight into the shell.
 *  - normalGradients, gradientVote, poissonDetail (x7:271-339), median3 (x7:345-355).
 *
 * Deliberate differences from x7:
 *  - The second model is 1/z = a*d + b (disparity affine in the estimate). x7 fitted
 *    z = a * (10000/d) + b on raw VDA units; the estimate here is normalized disparity in [0, 1],
 *    where d = 0 is a valid far sample and 1/d is not.
 *  - Tile centres sit at pixel-centre coordinates: u = (x + 0.5)/tile - 0.5 (x7: x/tile - 0.5).
 *  - Lengths are metres (x7: millimetres): the trim floor 1 mm is 0.001, the edge drop 25 mm 0.025.
 *  - median3 leaves a missing centre missing (x7 took the median of the two neighbours and a zero).
 *
 * Conventions: OpenCV camera space (x right, y down, z forward, metres). A pixel (u, v) =
 * (fx*x/z + cx, fy*y/z + cy) is in map pixels, with pixel centres at integer + 0.5. The estimate, the
 * target, the mask, the normal map and the intrinsics all share one W x H map grid.
 *
 * Every function is synchronous and allocates its buffers once per call, outside the pixel loops.
 * Timing: see poissonDetail.
 */

export type FitKind = "linear" | "disparity";

/** One robust affine fit. linear: z = a*d + b. disparity: 1/z = a*d + b. `medres` is the median
 *  |z residual| of the inliers in metres, `n` the inlier count. */
export interface RobustFit { kind: FitKind; a: number; b: number; medres: number; n: number }

/** Per-tile affine field over a W x H map. `a`, `b`, `w` hold tilesY rows of tilesX tiles; `w` is
 *  the local-fit weight each tile got before the spatial smoothing pass (0 = global parameters). */
export interface TileField {
  kind: FitKind;
  tile: number;
  tilesX: number;
  tilesY: number;
  W: number;
  H: number;
  a: Float32Array;
  b: Float32Array;
  w: Float32Array;
  /** The global fit; its `medres` is over its own inliers (after the trim). */
  global: RobustFit;
  /** Median |z_fit - target| in metres over every fit sample with the global fit applied: the same
   *  sample set as medresTiled, so the two compare. */
  medresGlobal: number;
  /** Median |z_fit - target| in metres over every fit sample with the tiled field applied. */
  medresTiled: number;
  /** Fit samples drawn from the frame (mask on, target and estimate valid, on the stride lattice). */
  samples: number;
}

export interface TileFitOptions {
  /** Tile edge in map pixels. Default max(16, round(72 * W / 848)): x7's 72 px at 848 wide. */
  tile?: number;
  /** Pooled samples below which a tile keeps the global parameters. Default 40. */
  minN?: number;
  /** Pooled samples at which a tile's own fit takes full weight. Default 200. */
  fullN?: number;
  /** Sampling lattice step in map pixels, both axes. Default 2. */
  stride?: number;
  /** Trim floor in metres (see robustFit). Default 0.001. */
  floor?: number;
  /** Fewer samples than this in the frame returns null. Default 200. */
  minSamples?: number;
}

/** Pinhole intrinsics in map pixels. */
export interface Intrinsics { fx: number; fy: number; cx: number; cy: number }

/** Log-depth gradients implied by a normal map, per map pixel. `ok` = 0 where no gradient. */
export interface NormalGradients { gu: Float32Array; gv: Float32Array; ok: Uint8Array }

export interface DetailOptions {
  /** Screening weight: pulls the residual toward 0 beyond about 1/sqrt(lambda) cells. Default 0.01. */
  lambda?: number;
  /** Residual clamp in log-depth units (0.035 = +-8 cm at 2.3 m). Default 0.035. */
  clamp?: number;
  /** CG iteration cap. Default 120. */
  iters?: number;
  /** CG stop at |res|^2 / |res0|^2 <= tol. Default 1e-6. */
  tol?: number;
  /** Metres. A neighbour pair whose base depths differ by more is no edge of the solve. Default 0.025. */
  edgeDrop?: number;
}

const DEF_FLOOR = 0.001;       // metres; x7's 1 mm trim floor
const DEF_MIN_N = 40;          // x7 MIN_N
const DEF_FULL_N = 200;        // x7 FULL_N
const DEF_STRIDE = 2;          // x7 samples every 2nd sensor pixel on both axes
const DEF_MIN_SAMPLES = 200;   // x7 degenerate-frame threshold
const DEF_LAMBDA = 0.01;       // x7 ARES_LAMBDA
const DEF_CLAMP = 0.035;       // x7 ARES_RCLAMP
const DEF_ITERS = 120;         // x7 CG_ITERS
const DEF_TOL = 1e-6;          // x7 CG stop on rr/rr0
const DEF_EDGE_DROP = 0.025;   // x7 EDGE_DROP_MM = 25
const NORMAL_MIN = 0.2;        // x7: |n| below this before normalizing = no normal
const D_MIN = 0.15;            // x7: |D| below this = grazing, gradient unbounded
const SPIKE_K = 4 * 1.4826;    // spike cut, in MADs of the per-tile deviation from the 5-frame median
const SPIKE_REL = 0.01;        // and 1 % of the tile's typical magnitude, so a noise-free series still has a cut

/* ------------------------------------------------------------------------------------------------ */
/* validation                                                                                        */

function checkDims(fn: string, W: number, H: number): void {
  if (!Number.isInteger(W) || !Number.isInteger(H) || W < 1 || H < 1) throw new Error(`${fn}: W and H must be positive integers, got ${W}x${H}`);
}

function checkLen(fn: string, name: string, a: ArrayLike<unknown> | null | undefined, need: number): void {
  if (!a || typeof a.length !== "number") throw new Error(`${fn}: ${name} is required, got ${a === null ? "null" : typeof a}`);
  if (a.length < need) throw new Error(`${fn}: ${name} holds ${a.length} values, expected ${need}`);
}

function checkK(fn: string, K: Intrinsics): void {
  if (!K || typeof K !== "object") throw new Error(`${fn}: K is required, got ${K === null ? "null" : typeof K}`);
  if (!(K.fx > 0 && K.fx < Infinity)) throw new Error(`${fn}: K.fx must be a finite number > 0, got ${K.fx}`);
  if (!(K.fy > 0 && K.fy < Infinity)) throw new Error(`${fn}: K.fy must be a finite number > 0, got ${K.fy}`);
  if (!Number.isFinite(K.cx)) throw new Error(`${fn}: K.cx must be finite, got ${K.cx}`);
  if (!Number.isFinite(K.cy)) throw new Error(`${fn}: K.cy must be finite, got ${K.cy}`);
}

function checkKind(fn: string, kind: unknown): asserts kind is FitKind {
  if (kind !== "linear" && kind !== "disparity") throw new Error(`${fn}: kind must be "linear" or "disparity", got ${String(kind)}`);
}

function optInt(fn: string, name: string, v: number | undefined, def: number, min: number): number {
  if (v === undefined) return def;
  if (!Number.isInteger(v) || v < min) throw new Error(`${fn}: ${name} must be an integer >= ${min}, got ${v}`);
  return v;
}

/** A finite number, >= min (or > min when `open`). */
function optNum(fn: string, name: string, v: number | undefined, def: number, min: number, open: boolean): number {
  if (v === undefined) return def;
  if (!Number.isFinite(v) || (open ? !(v > min) : !(v >= min))) throw new Error(`${fn}: ${name} must be a finite number ${open ? ">" : ">="} ${min}, got ${v}`);
  return v;
}

function checkField(fn: string, f: TileField): void {
  if (!f || typeof f !== "object") throw new Error(`${fn}: field is required, got ${f === null ? "null" : typeof f}`);
  checkKind(fn, f.kind);
  checkDims(fn, f.W, f.H);
  if (!Number.isInteger(f.tile) || f.tile < 1) throw new Error(`${fn}: field.tile must be an integer >= 1, got ${f.tile}`);
  if (f.tilesX !== Math.ceil(f.W / f.tile) || f.tilesY !== Math.ceil(f.H / f.tile))
    throw new Error(`${fn}: field tile grid ${f.tilesX}x${f.tilesY} does not match W=${f.W} H=${f.H} tile=${f.tile}`);
  const nT = f.tilesX * f.tilesY;
  checkLen(fn, "field.a", f.a, nT);
  checkLen(fn, "field.b", f.b, nT);
  checkLen(fn, "field.w", f.w, nT);
}

/* ------------------------------------------------------------------------------------------------ */
/* robust fit                                                                                        */

/** k-th smallest of a[0..n), in place (Hoare partition, median-of-three pivot). a holds no NaN. */
function selectK(a: Float64Array, n: number, k: number): number {
  let l = 0, m = n - 1;
  while (l < m) {
    const p0 = a[l]!, p1 = a[(l + m) >> 1]!, p2 = a[m]!;
    const x = p0 < p1 ? (p1 < p2 ? p1 : p0 < p2 ? p2 : p0) : (p0 < p2 ? p0 : p1 < p2 ? p2 : p1);
    let i = l, j = m;
    do {
      while (a[i]! < x) i++;
      while (x < a[j]!) j--;
      if (i <= j) { const s = a[i]!; a[i] = a[j]!; a[j] = s; i++; j--; }
    } while (i <= j);
    if (j < k) l = i;
    if (k < i) m = j;
  }
  return a[k]!;
}

/** |z - model(x)| in metres. A disparity model predicting 1/z <= 0 has no z: Infinity. */
function residual(a: number, b: number, x: number, z: number, disparity: boolean): number {
  if (disparity) {
    const q = a * x + b;
    return q > 0 ? Math.abs(z - 1 / q) : Infinity;
  }
  return Math.abs(z - (a * x + b));
}

interface FitOut { a: number; b: number; medres: number; n: number }

/** Centered two-pass OLS of y (z, or 1/z) on x over the kept samples. False when degenerate. */
function ols(x: Float64Array, z: Float64Array, n: number, disparity: boolean, keep: Uint8Array, out: FitOut): boolean {
  let sw = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) {
    if (!keep[i]) continue;
    sw++; sx += x[i]!; sy += disparity ? 1 / z[i]! : z[i]!;
  }
  if (sw < 3) return false;
  const mx = sx / sw, my = sy / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    if (!keep[i]) continue;
    const dx = x[i]! - mx;
    sxx += dx * dx; sxy += dx * ((disparity ? 1 / z[i]! : z[i]!) - my);
  }
  if (!(sxx > 1e-12 * sw * Math.max(1, mx * mx))) return false;
  const a = sxy / sxx, b = my - a * mx;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
  out.a = a; out.b = b;
  return true;
}

/**
 * x7:163-179 on prepared samples (x finite, z finite, z > 0 for the disparity model). `res` and
 * `keep` are scratch of at least n. One OLS, the MAD of its z residuals, inliers at
 * |res| <= 3 * mad + floor, one OLS refit on them, medres = median inlier |res| against the refit.
 */
function fitCore(x: Float64Array, z: Float64Array, n: number, disparity: boolean, floor: number, res: Float64Array, keep: Uint8Array, out: FitOut): boolean {
  if (n < 3) return false;
  keep.fill(1, 0, n);
  if (!ols(x, z, n, disparity, keep, out)) return false;
  let a = out.a, b = out.b;
  for (let i = 0; i < n; i++) res[i] = residual(a, b, x[i]!, z[i]!, disparity);
  const thr = 3 * selectK(res, n, n >> 1) + floor;
  let m = 0;
  for (let i = 0; i < n; i++) {
    const k = residual(a, b, x[i]!, z[i]!, disparity) <= thr ? 1 : 0;
    keep[i] = k; m += k;
  }
  if (m < 3 || !ols(x, z, n, disparity, keep, out)) return false;
  a = out.a; b = out.b;
  let k = 0;
  for (let i = 0; i < n; i++) if (keep[i]) res[k++] = residual(a, b, x[i]!, z[i]!, disparity);
  const medres = selectK(res, k, k >> 1);
  if (!Number.isFinite(medres)) return false;
  out.medres = medres; out.n = m;
  return true;
}

/**
 * Robust affine fit of the metric target `z` on the estimate `d` over the first n samples (x7:163-179).
 * Samples whose d or z is not finite (or whose z <= 0, for the disparity model) are skipped. Null when
 * fewer than 3 samples remain, when d has no spread, or when fewer than 3 inliers survive the trim.
 */
export function robustFit(d: ArrayLike<number>, z: ArrayLike<number>, n: number, kind: FitKind, floor: number = DEF_FLOOR): RobustFit | null {
  const fn = "robustFit";
  checkKind(fn, kind);
  if (!d || !z) throw new Error(`${fn}: d and z are required`);
  const cap = Math.min(d.length, z.length);
  if (!Number.isInteger(n) || n < 0 || n > cap) throw new Error(`${fn}: n must be an integer in [0, ${cap}], got ${n}`);
  if (!(floor >= 0 && floor < Infinity)) throw new Error(`${fn}: floor must be a finite number >= 0, got ${floor}`);
  const disparity = kind === "disparity";
  const x = new Float64Array(n), t = new Float64Array(n);
  let m = 0;
  for (let i = 0; i < n; i++) {
    const di = +d[i]!, zi = +z[i]!;
    if (!Number.isFinite(di) || !Number.isFinite(zi) || (disparity && !(zi > 0))) continue;
    x[m] = di; t[m] = zi; m++;
  }
  const out: FitOut = { a: 0, b: 0, medres: 0, n: 0 };
  if (!fitCore(x, t, m, disparity, floor, new Float64Array(m), new Uint8Array(m), out)) return null;
  return { kind, a: out.a, b: out.b, medres: out.medres, n: out.n };
}

/* ------------------------------------------------------------------------------------------------ */
/* tiled field                                                                                       */

/** Bilinear lookup along one axis: pixel p sits between tile centres i0 and i1 at fraction f. */
interface AxisTable { i0: Int32Array; i1: Int32Array; f: Float64Array }

function axisTable(n: number, tile: number, tiles: number): AxisTable {
  const i0 = new Int32Array(n), i1 = new Int32Array(n), f = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    let u = (p + 0.5) / tile - 0.5;
    if (u < 0) u = 0; else if (u > tiles - 1) u = tiles - 1;
    const a = Math.min(Math.max(0, tiles - 2), Math.floor(u));
    i0[p] = a;
    i1[p] = Math.min(tiles - 1, a + 1);
    f[p] = tiles > 1 ? Math.min(1, u - a) : 0;
  }
  return { i0, i1, f };
}

/** z from one (a, b) and one estimate value; 0 when the model has no positive finite z. */
function zOf(disparity: boolean, a: number, b: number, e: number): number {
  let z: number;
  if (disparity) {
    const q = a * e + b;
    z = q > 0 ? 1 / q : 0;
  } else z = a * e + b;
  return z > 0 && z < Infinity ? z : 0;
}

/**
 * Tiled locally-affine fit of the metric target on the estimate (x7:190-256).
 *
 * `est` is the stabilized normalized disparity (larger = nearer), `target` the metric z (0 or
 * non-finite = no target), `valid` the subject mask (nonzero = use). Samples are the stride lattice
 * points where all three hold. Below `minSamples` the frame has no field: null.
 *
 * Global fit: both models, the lower medres wins for the whole frame (x7:205-209); a tie goes to
 * linear. Per tile: the samples of its 3x3 tile neighbourhood pooled; below minN the tile keeps the
 * global parameters, else it is fitted with the frame's model and blended toward the global fit with
 * w = min(1, n/fullN), halved when its medres exceeds twice the global one. One 3x3 smoothing pass
 * over the parameter grid weights each neighbour 0.25 + w (+1 for the centre) (x7:221-246).
 * Measured 2026-09-19 on the synthetic 424x240 scenes of depth-metric.test.mjs (a and b ramps across
 * the frame): median residual 240 mm global against 29 mm tiled for the linear model, 95 mm against
 * 7.6 mm for the disparity model.
 */
export function fitTileField(est: Float32Array, target: Float32Array, valid: Uint8Array, W: number, H: number, o: TileFitOptions = {}): TileField | null {
  const fn = "fitTileField";
  checkDims(fn, W, H);
  const P = W * H;
  checkLen(fn, "est", est, P);
  checkLen(fn, "target", target, P);
  checkLen(fn, "valid", valid, P);
  const tile = optInt(fn, "tile", o.tile, Math.max(16, Math.round(72 * W / 848)), 1);
  const minN = optInt(fn, "minN", o.minN, DEF_MIN_N, 3);
  const fullN = optNum(fn, "fullN", o.fullN, DEF_FULL_N, 0, true);
  const stride = optInt(fn, "stride", o.stride, DEF_STRIDE, 1);
  const floor = optNum(fn, "floor", o.floor, DEF_FLOOR, 0, false);
  const minSamples = optInt(fn, "minSamples", o.minSamples, DEF_MIN_SAMPLES, 3);
  const tilesX = Math.ceil(W / tile), tilesY = Math.ceil(H / tile), nT = tilesX * tilesY;

  // ---- samples, with their tile and pixel ----
  const cap = Math.ceil(W / stride) * Math.ceil(H / stride);
  const xs = new Float64Array(cap), zs = new Float64Array(cap);
  const sTile = new Int32Array(cap), sPix = new Int32Array(cap);
  let S = 0;
  for (let y = 0; y < H; y += stride) {
    const trow = Math.min(tilesY - 1, Math.floor(y / tile)) * tilesX, row = y * W;
    for (let x = 0; x < W; x += stride) {
      const i = row + x;
      if (!valid[i]) continue;
      const t = target[i]!, e = est[i]!;
      if (!(t > 0 && t < Infinity) || !Number.isFinite(e)) continue;
      xs[S] = e; zs[S] = t;
      sTile[S] = trow + Math.min(tilesX - 1, Math.floor(x / tile));
      sPix[S] = i;
      S++;
    }
  }
  if (S < minSamples) return null;

  // ---- global fit: both models compete ----
  const res = new Float64Array(S), keep = new Uint8Array(S);
  const lin: FitOut = { a: 0, b: 0, medres: 0, n: 0 }, dis: FitOut = { a: 0, b: 0, medres: 0, n: 0 };
  const okL = fitCore(xs, zs, S, false, floor, res, keep, lin);
  const okD = fitCore(xs, zs, S, true, floor, res, keep, dis);
  if (!okL && !okD) return null;
  const disparity = okD && (!okL || dis.medres < lin.medres);
  const kind: FitKind = disparity ? "disparity" : "linear";
  const g = disparity ? dis : lin;
  const global: RobustFit = { kind, a: g.a, b: g.b, medres: g.medres, n: g.n };
  // The global fit over every sample, the set medresTiled is taken over. Its own medres is over the
  // inliers the trim kept: p50 over every third frame of the 90-frame skateboarding run (2026-09-19)
  // 41.9 mm, against 58.7 mm over every sample, where the tiled field reads 55.9 mm.
  for (let k = 0; k < S; k++) { const zf = zOf(disparity, g.a, g.b, xs[k]!); res[k] = zf > 0 ? Math.abs(zf - zs[k]!) : Infinity; }
  const medresGlobal = selectK(res, S, S >> 1);

  // ---- bin samples by tile (counting sort) ----
  const start = new Int32Array(nT + 1);
  for (let k = 0; k < S; k++) start[sTile[k]! + 1]!++;
  for (let t = 0; t < nT; t++) start[t + 1]! += start[t]!;
  const fill = start.slice(0, nT), order = new Int32Array(S);
  for (let k = 0; k < S; k++) order[fill[sTile[k]!]!++] = k;

  // ---- per-tile fits on the 3x3 tile neighbourhood, blended toward the global fit ----
  const ta = new Float64Array(nT).fill(g.a), tb = new Float64Array(nT).fill(g.b), tw = new Float64Array(nT);
  const poolX = new Float64Array(S), poolZ = new Float64Array(S);
  const tf: FitOut = { a: 0, b: 0, medres: 0, n: 0 };
  for (let ty = 0; ty < tilesY; ty++) for (let tx = 0; tx < tilesX; tx++) {
    let n = 0;
    for (let ny = Math.max(0, ty - 1); ny <= Math.min(tilesY - 1, ty + 1); ny++)
      for (let nx = Math.max(0, tx - 1); nx <= Math.min(tilesX - 1, tx + 1); nx++) {
        const j = ny * tilesX + nx;
        for (let p = start[j]!, e = start[j + 1]!; p < e; p++) { const k = order[p]!; poolX[n] = xs[k]!; poolZ[n] = zs[k]!; n++; }
      }
    if (n < minN) continue;
    if (!fitCore(poolX, poolZ, n, disparity, floor, res, keep, tf)) continue;
    let w = Math.min(1, n / fullN);
    if (tf.medres > 2 * g.medres) w *= 0.5;
    const i = ty * tilesX + tx;
    ta[i] = w * tf.a + (1 - w) * g.a;
    tb[i] = w * tf.b + (1 - w) * g.b;
    tw[i] = w;
  }

  // ---- one 3x3 weighted smoothing pass over the parameter grid ----
  const a = new Float32Array(nT), b = new Float32Array(nT), w = new Float32Array(nT);
  for (let ty = 0; ty < tilesY; ty++) for (let tx = 0; tx < tilesX; tx++) {
    let wa = 0, wb = 0, ww = 0;
    for (let ny = Math.max(0, ty - 1); ny <= Math.min(tilesY - 1, ty + 1); ny++)
      for (let nx = Math.max(0, tx - 1); nx <= Math.min(tilesX - 1, tx + 1); nx++) {
        const j = ny * tilesX + nx, wj = 0.25 + tw[j]! + (nx === tx && ny === ty ? 1 : 0);
        wa += ta[j]! * wj; wb += tb[j]! * wj; ww += wj;
      }
    const i = ty * tilesX + tx;
    a[i] = wa / ww; b[i] = wb / ww; w[i] = tw[i]!;
  }

  // ---- tiled residual over the fit samples, with the stored (float32) field ----
  const cx = axisTable(W, tile, tilesX), cy = axisTable(H, tile, tilesY);
  for (let k = 0; k < S; k++) {
    const i = sPix[k]!, y = Math.floor(i / W), x = i - y * W;
    const r0 = cy.i0[y]! * tilesX, r1 = cy.i1[y]! * tilesX, fy = cy.f[y]!;
    const c0 = cx.i0[x]!, c1 = cx.i1[x]!, fx = cx.f[x]!;
    const pa = (a[r0 + c0]! * (1 - fx) + a[r0 + c1]! * fx) * (1 - fy) + (a[r1 + c0]! * (1 - fx) + a[r1 + c1]! * fx) * fy;
    const pb = (b[r0 + c0]! * (1 - fx) + b[r0 + c1]! * fx) * (1 - fy) + (b[r1 + c0]! * (1 - fx) + b[r1 + c1]! * fx) * fy;
    const zf = zOf(disparity, pa, pb, xs[k]!);
    res[k] = zf > 0 ? Math.abs(zf - zs[k]!) : Infinity;
  }
  const medresTiled = selectK(res, S, S >> 1);

  return { kind, tile, tilesX, tilesY, W, H, a, b, w, global, medresGlobal, medresTiled, samples: S };
}

/** Median of five (six comparisons; the network of depth-stabilize.ts temporalMedian). */
function med5(v0: number, v1: number, v2: number, v3: number, v4: number): number {
  let s: number;
  if (v0 > v1) { s = v0; v0 = v1; v1 = s; }
  if (v2 > v3) { s = v2; v2 = v3; v3 = s; }
  if (v0 > v2) { s = v1; v1 = v3; v3 = s; v2 = v0; }
  if (v1 > v4) { s = v1; v1 = v4; v4 = s; }
  if (v1 > v2) { s = v4; v4 = v3; v3 = s; v2 = v1; }
  return v2 < v4 ? v2 : v4;
}

/** prev[t] / next[t]: nearest index at or before / at or after t with ok set, -1 when none. */
function nearestOk(ok: Uint8Array, T: number, prev: Int32Array, next: Int32Array): void {
  let p = -1;
  for (let t = 0; t < T; t++) { if (ok[t]) p = t; prev[t] = p; }
  let q = -1;
  for (let t = T - 1; t >= 0; t--) { if (ok[t]) q = t; next[t] = q; }
}

/** Fill column j of v (T rows of stride nT) where ok is 0, linearly between the nearest ok rows. */
function fillColumn(v: Float64Array, T: number, nT: number, j: number, ok: Uint8Array, prev: Int32Array, next: Int32Array): void {
  for (let t = 0; t < T; t++) {
    if (ok[t]) continue;
    const p = prev[t]!, q = next[t]!;
    if (p < 0 && q < 0) continue;
    if (p < 0) v[t * nT + j] = v[q * nT + j]!;
    else if (q < 0) v[t * nT + j] = v[p * nT + j]!;
    else { const f = (t - p) / (q - p); v[t * nT + j] = v[p * nT + j]! * (1 - f) + v[q * nT + j]! * f; }
  }
}

/** Row t (clamped to the clip) of column j of a T x nT series. */
function colAt(v: Float64Array, T: number, nT: number, t: number, j: number): number {
  return v[(t < 0 ? 0 : t >= T ? T - 1 : t) * nT + j]!;
}

/** Spike cut of column j: SPIKE_K MADs of its departures plus SPIKE_REL of its median magnitude, over
 *  the frames that hold their own fit. `sc` is scratch of at least T. */
function spikeCut(dev: Float64Array, v: Float64Array, use: Uint8Array, T: number, nT: number, j: number, sc: Float64Array): number {
  let k = 0;
  for (let t = 0; t < T; t++) if (use[t]) sc[k++] = dev[t]!;
  const s = selectK(sc, k, k >> 1);
  k = 0;
  for (let t = 0; t < T; t++) if (use[t]) sc[k++] = Math.abs(v[t * nT + j]!);
  return SPIKE_K * s + SPIKE_REL * selectK(sc, k, k >> 1);
}

/** Step 2 of smoothTileFields, in place on the T x nT series A and B. */
function rejectSpikes(A: Float64Array, B: Float64Array, use: Uint8Array, T: number, nT: number): void {
  const devA = new Float64Array(T), devB = new Float64Array(T), sc = new Float64Array(T);
  const good = new Uint8Array(T), gp = new Int32Array(T), gn = new Int32Array(T);
  let m = 0;
  for (let t = 0; t < T; t++) if (use[t]) m++;
  if (m < 3) return;
  for (let j = 0; j < nT; j++) {
    for (let t = 0; t < T; t++) {
      const a = colAt(A, T, nT, t, j), b = colAt(B, T, nT, t, j);
      devA[t] = Math.abs(a - med5(colAt(A, T, nT, t - 2, j), colAt(A, T, nT, t - 1, j), a, colAt(A, T, nT, t + 1, j), colAt(A, T, nT, t + 2, j)));
      devB[t] = Math.abs(b - med5(colAt(B, T, nT, t - 2, j), colAt(B, T, nT, t - 1, j), b, colAt(B, T, nT, t + 1, j), colAt(B, T, nT, t + 2, j)));
    }
    const cutA = spikeCut(devA, A, use, T, nT, j, sc), cutB = spikeCut(devB, B, use, T, nT, j, sc);
    let flagged = 0;
    for (let t = 0; t < T; t++) {
      const bad = use[t] === 1 && (devA[t]! > cutA || devB[t]! > cutB);
      good[t] = use[t] === 1 && !bad ? 1 : 0;
      if (bad) flagged++;
    }
    if (!flagged) continue;
    nearestOk(good, T, gp, gn);
    for (let t = 0; t < T; t++) {
      if (good[t] || !use[t]) continue;
      const p = gp[t]!, q = gn[t]!;
      if (p < 0 && q < 0) continue;
      const f = p < 0 ? 1 : q < 0 ? 0 : (t - p) / (q - p), p0 = p < 0 ? q : p, q0 = q < 0 ? p : q;
      A[t * nT + j] = A[p0 * nT + j]! * (1 - f) + A[q0 * nT + j]! * f;
      B[t * nT + j] = B[p0 * nT + j]! * (1 - f) + B[q0 * nT + j]! * f;
    }
  }
}

/**
 * Temporal smoothing of per-frame tile fields, per tile, over a clip's frames.
 *
 * A MoGe-2 target is inferred one frame at a time and its scale flickers; a field fitted to it frame by
 * frame passes the flicker into the shell as a surge of the whole surface. Three steps:
 *  1. Frames whose model differs from the clip's majority (a tie goes to the lower mean global medres)
 *     and null frames are filled per tile by linear interpolation between the nearest majority frames:
 *     parameters of the two models are not comparable, so they are never averaged together.
 *  2. Spikes: per tile, a frame whose a or b departs from the 5-frame median by more than
 *     4 * 1.4826 MADs of that tile's departures plus 1 % of its typical magnitude is replaced, a and b
 *     together, by interpolation between its nearest unflagged neighbours. A step lasting 3 frames or
 *     more is its own 5-frame median and passes unchanged; a 1- or 2-frame event does not.
 *  3. Gaussian over t, sigma = radius/2, window +-radius, weights renormalized at the clip's ends.
 * radius 0 runs step 1 only. Every non-null field must share one tile geometry. Returns new objects:
 * every frame is non-null when any input frame was (filled frames carry samples 0 and global.n 0,
 * with interpolated global parameters and medres), all null otherwise.
 */
export function smoothTileFields(fields: (TileField | null)[], radius: number): (TileField | null)[] {
  const fn = "smoothTileFields";
  if (!Array.isArray(fields)) throw new Error(`${fn}: fields must be an array, got ${typeof fields}`);
  if (!Number.isInteger(radius) || radius < 0) throw new Error(`${fn}: radius must be an integer >= 0, got ${radius}`);
  const T = fields.length;
  let ref: TileField | null = null;
  let nLin = 0, nDis = 0, mLin = 0, mDis = 0;
  for (let t = 0; t < T; t++) {
    const f = fields[t];
    if (f === null || f === undefined) continue;
    checkField(fn, f);
    if (!ref) ref = f;
    else if (f.tile !== ref.tile || f.W !== ref.W || f.H !== ref.H)
      throw new Error(`${fn}: frame ${t} has tile ${f.tile} on ${f.W}x${f.H}, frame geometry is tile ${ref.tile} on ${ref.W}x${ref.H}`);
    if (f.kind === "disparity") { nDis++; mDis += f.global.medres; } else { nLin++; mLin += f.global.medres; }
  }
  if (!ref) return fields.map(() => null);
  const kind: FitKind = nDis > nLin || (nDis === nLin && mDis < mLin) ? "disparity" : "linear";
  const nT = ref.tilesX * ref.tilesY;

  // Per-frame scalars ride along as extra columns: global a, b, medres, medresGlobal and medresTiled.
  const nS = 5;
  const use = new Uint8Array(T);
  const A = new Float64Array(T * nT), B = new Float64Array(T * nT), Wt = new Float64Array(T * nT), Sc = new Float64Array(T * nS);
  for (let t = 0; t < T; t++) {
    const f = fields[t];
    if (!f || f.kind !== kind) continue;
    use[t] = 1;
    for (let j = 0; j < nT; j++) { A[t * nT + j] = f.a[j]!; B[t * nT + j] = f.b[j]!; Wt[t * nT + j] = f.w[j]!; }
    Sc[t * nS] = f.global.a; Sc[t * nS + 1] = f.global.b; Sc[t * nS + 2] = f.global.medres; Sc[t * nS + 3] = f.medresGlobal; Sc[t * nS + 4] = f.medresTiled;
  }
  const prev = new Int32Array(T), next = new Int32Array(T);
  nearestOk(use, T, prev, next);
  for (let j = 0; j < nT; j++) { fillColumn(A, T, nT, j, use, prev, next); fillColumn(B, T, nT, j, use, prev, next); fillColumn(Wt, T, nT, j, use, prev, next); }
  for (let j = 0; j < nS; j++) fillColumn(Sc, T, nS, j, use, prev, next);

  if (radius > 0 && T >= 3) rejectSpikes(A, B, use, T, nT);

  let outA = A, outB = B;
  if (radius > 0) {
    const sigma = radius / 2, gw = new Float64Array(2 * radius + 1);
    for (let k = -radius; k <= radius; k++) gw[k + radius] = Math.exp(-(k * k) / (2 * sigma * sigma));
    outA = new Float64Array(T * nT); outB = new Float64Array(T * nT);
    for (let t = 0; t < T; t++) {
      const s0 = Math.max(0, t - radius), s1 = Math.min(T - 1, t + radius);
      let ws = 0;
      for (let s = s0; s <= s1; s++) ws += gw[s - t + radius]!;
      const inv = 1 / ws;
      for (let s = s0; s <= s1; s++) {
        const wgt = gw[s - t + radius]! * inv;
        for (let j = 0; j < nT; j++) { outA[t * nT + j]! += wgt * A[s * nT + j]!; outB[t * nT + j]! += wgt * B[s * nT + j]!; }
      }
    }
  }

  const out: (TileField | null)[] = new Array(T);
  for (let t = 0; t < T; t++) {
    const f = fields[t];
    const own = use[t] === 1 && !!f;
    out[t] = {
      kind, tile: ref.tile, tilesX: ref.tilesX, tilesY: ref.tilesY, W: ref.W, H: ref.H,
      a: Float32Array.from(outA.subarray(t * nT, (t + 1) * nT)),
      b: Float32Array.from(outB.subarray(t * nT, (t + 1) * nT)),
      w: Float32Array.from(Wt.subarray(t * nT, (t + 1) * nT)),
      global: own ? { ...f.global } : { kind, a: Sc[t * nS]!, b: Sc[t * nS + 1]!, medres: Sc[t * nS + 2]!, n: 0 },
      medresGlobal: own ? f.medresGlobal : Sc[t * nS + 3]!,
      medresTiled: own ? f.medresTiled : Sc[t * nS + 4]!,
      samples: own ? f.samples : 0,
    };
  }
  return out;
}

/**
 * The field applied: bilinear (a, b) between tile centres (x7:247-256), z per pixel in metres. 0 where
 * the mask is off, the estimate is not finite, or the model gives no positive finite z.
 */
export function evalTileField(f: TileField, est: Float32Array, mask: Uint8Array | null, out?: Float32Array): Float32Array {
  const fn = "evalTileField";
  checkField(fn, f);
  const { W, H, tilesX, tilesY, tile } = f;
  const P = W * H;
  checkLen(fn, "est", est, P);
  if (mask) checkLen(fn, "mask", mask, P);
  const o = out ?? new Float32Array(P);
  checkLen(fn, "out", o, P);
  const disparity = f.kind === "disparity";
  const cx = axisTable(W, tile, tilesX), cy = axisTable(H, tile, tilesY);
  const rowA = new Float64Array(tilesX), rowB = new Float64Array(tilesX);
  const fa = f.a, fb = f.b;
  for (let y = 0; y < H; y++) {
    const r0 = cy.i0[y]! * tilesX, r1 = cy.i1[y]! * tilesX, fy = cy.f[y]!;
    for (let tx = 0; tx < tilesX; tx++) {
      rowA[tx] = fa[r0 + tx]! * (1 - fy) + fa[r1 + tx]! * fy;
      rowB[tx] = fb[r0 + tx]! * (1 - fy) + fb[r1 + tx]! * fy;
    }
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const i = row + x;
      const e = est[i]!;
      if ((mask && !mask[i]) || !Number.isFinite(e)) { o[i] = 0; continue; }
      const c0 = cx.i0[x]!, c1 = cx.i1[x]!, fx = cx.f[x]!;
      o[i] = zOf(disparity, rowA[c0]! * (1 - fx) + rowA[c1]! * fx, rowB[c0]! * (1 - fx) + rowB[c1]! * fx, e);
    }
  }
  return o;
}

/* ------------------------------------------------------------------------------------------------ */
/* normal detail                                                                                     */

/**
 * Log-depth gradient per map pixel from the normal map (x7:275-285).
 *
 * `normals` is W x H x 3 signed bytes, n = value/127, in OpenCV camera axes (x right, y down, z
 * forward; MoGe-2's convention). A surface through the ray of pixel (u, v) with normal n has
 * z = c / D, D = nx(u - cx)/fx + ny(v - cy)/fy + nz, so d ln z/du = -(nx/fx)/D and
 * d ln z/dv = -(ny/fy)/D, per map pixel. The global sign of n cancels in the ratio; the axis
 * convention does not, and flipY negates ny for a map whose y axis points up. Rejected: |n| < 0.2
 * before normalizing (no normal) and |D| < 0.15 after (a grazing surface, gradient unbounded).
 */
export function normalGradients(normals: Int8Array, W: number, H: number, K: Intrinsics, flipY = false): NormalGradients {
  const fn = "normalGradients";
  checkDims(fn, W, H);
  const P = W * H;
  checkLen(fn, "normals", normals, P * 3);
  checkK(fn, K);
  const gu = new Float32Array(P), gv = new Float32Array(P), ok = new Uint8Array(P);
  const ifx = 1 / K.fx, ify = 1 / K.fy, sy = flipY ? -1 : 1;
  for (let y = 0; y < H; y++) {
    const vy = (y + 0.5 - K.cy) * ify, row = y * W;
    for (let x = 0; x < W; x++) {
      const i = row + x, o = i * 3;
      let nx = normals[o]! / 127, ny = sy * normals[o + 1]! / 127, nz = normals[o + 2]! / 127;
      const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (l < NORMAL_MIN) continue;
      nx /= l; ny /= l; nz /= l;
      const D = nx * (x + 0.5 - K.cx) * ifx + ny * vy + nz;
      if (Math.abs(D) < D_MIN) continue;
      gu[i] = -(nx * ifx) / D;
      gv[i] = -(ny * ify) / D;
      ok[i] = 1;
    }
  }
  return { gu, gv, ok };
}

function checkGradients(fn: string, g: NormalGradients, P: number): void {
  if (!g || typeof g !== "object") throw new Error(`${fn}: g is required, got ${g === null ? "null" : typeof g}`);
  checkLen(fn, "g.gu", g.gu, P);
  checkLen(fn, "g.gv", g.gv, P);
  checkLen(fn, "g.ok", g.ok, P);
}

const isDepth = (z: number): boolean => z > 0 && z < Infinity;

/**
 * Axis-convention vote (x7:305-308): the sum over neighbour pairs (right and down) of the normal
 * gradient times the base's own log-depth difference, over pairs with both depths, both gradients
 * and a depth step within the 0.025 m edge drop. Positive = the normal map and the base agree on
 * which way depth runs; negative = flip the map's y (or the base is flat noise).
 */
export function gradientVote(zb: Float32Array, g: NormalGradients, W: number, H: number): number {
  const fn = "gradientVote";
  checkDims(fn, W, H);
  const P = W * H;
  checkLen(fn, "zb", zb, P);
  checkGradients(fn, g, P);
  const L = new Float64Array(P);
  for (let i = 0; i < P; i++) { const z = zb[i]!; L[i] = isDepth(z) ? Math.log(z) : NaN; }
  const { gu, gv, ok } = g;
  let s = 0;
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const i = row + x;
      if (!ok[i] || !isDepth(zb[i]!)) continue;
      const z0 = zb[i]!;
      if (x + 1 < W) {
        const j = i + 1;
        if (ok[j] && isDepth(zb[j]!) && Math.abs(zb[j]! - z0) <= DEF_EDGE_DROP) s += 0.5 * (gu[i]! + gu[j]!) * (L[j]! - L[i]!);
      }
      if (y + 1 < H) {
        const j = i + W;
        if (ok[j] && isDepth(zb[j]!) && Math.abs(zb[j]! - z0) <= DEF_EDGE_DROP) s += 0.5 * (gv[i]! + gv[j]!) * (L[j]! - L[i]!);
      }
    }
  }
  return s;
}

/**
 * Screened Poisson on the log-depth residual (x7:286-339).
 *
 * Unknowns: every cell with a base depth zb > 0. Edges: right and down neighbours, dropped when the
 * base depths differ by more than edgeDrop (a silhouette or a fold the base already resolves: across
 * it the normals say nothing about the step) or either end has no gradient. Each edge asks
 * r_j - r_i = t_e, t_e = gN - (L_j - L_i) with gN the mean of the two normal gradients along the edge
 * axis and L = ln zb; the solve is min sum_e (r_j - r_i - t_e)^2 + lambda sum r^2, i.e.
 * (lambda I + G^T G) r = G^T t, by conjugate gradients from 0. r is clamped to +-clamp; the detailed
 * depth is z = zb * exp(r). r = 0 where not solved; rms is over the solved cells, after the clamp.
 *
 * Layout: the solve runs on the map grid itself, over each row's span of cells that have an edge. A
 * cell's edges are 4 flag bits, so the operator is (lambda + degree) r_i minus the flagged neighbours;
 * the all-four case (the interior) takes one unbranched sum. Cells inside a span without an edge have
 * no right-hand side and stay 0. Measured 2026-09-19 in node 22.19, 512x288 map, median of 10 warm
 * runs, synthetic target with detail at every pixel: subject ellipse of 31.6 % of the frame, 102 CG
 * iterations, fitTileField 7.6 ms + poissonDetail 31 ms (evalTileField 0.9 ms); every pixel an
 * unknown, 109 iterations, 21 ms + 96 ms. The compacted-index form of the same solve (4 neighbour
 * indices per unknown, one unbranched gather) took 124 ms on the full frame.
 */
export function poissonDetail(zb: Float32Array, W: number, H: number, g: NormalGradients, o: DetailOptions = {}): { r: Float32Array; iters: number; rms: number } {
  const fn = "poissonDetail";
  checkDims(fn, W, H);
  const P = W * H;
  checkLen(fn, "zb", zb, P);
  checkGradients(fn, g, P);
  const lambda = optNum(fn, "lambda", o.lambda, DEF_LAMBDA, 0, true);
  const clamp = optNum(fn, "clamp", o.clamp, DEF_CLAMP, 0, true);
  const iters = optInt(fn, "iters", o.iters, DEF_ITERS, 0);
  const tol = optNum(fn, "tol", o.tol, DEF_TOL, 0, false);
  const edgeDrop = optNum(fn, "edgeDrop", o.edgeDrop, DEF_EDGE_DROP, 0, true);
  const { gu, gv, ok } = g;

  // ---- unknowns, log base depth ----
  const L = new Float64Array(P);
  let N = 0;
  for (let i = 0; i < P; i++) { const z = zb[i]!; if (isDepth(z)) { L[i] = Math.log(z); N++; } }
  const rOut = new Float32Array(P);
  if (N === 0) return { r: rOut, iters: 0, rms: 0 };

  // ---- edges: flag bits 1 right, 2 left, 4 down, 8 up; right-hand side G^T t into res ----
  const fl = new Uint8Array(P);
  const res = new Float64Array(P);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const i = row + x, z0 = zb[i]!;
      if (!ok[i] || !isDepth(z0)) continue;
      if (x + 1 < W) {
        const j = i + 1, z1 = zb[j]!;
        if (ok[j] && isDepth(z1) && Math.abs(z1 - z0) <= edgeDrop) {
          const t = 0.5 * (gu[i]! + gu[j]!) - (L[j]! - L[i]!);
          res[i]! -= t; res[j]! += t;
          fl[i]! |= 1; fl[j]! |= 2;
        }
      }
      if (y + 1 < H) {
        const j = i + W, z1 = zb[j]!;
        if (ok[j] && isDepth(z1) && Math.abs(z1 - z0) <= edgeDrop) {
          const t = 0.5 * (gv[i]! + gv[j]!) - (L[j]! - L[i]!);
          res[i]! -= t; res[j]! += t;
          fl[i]! |= 4; fl[j]! |= 8;
        }
      }
    }
  }
  // Per-row span [s0, s1) of cells with an edge; an edgeless row has s0 = s1.
  const s0 = new Int32Array(H), s1 = new Int32Array(H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let a = W, b = 0;
    for (let x = 0; x < W; x++) if (fl[row + x]) { if (x < a) a = x; b = x + 1; }
    s0[y] = row + (a < b ? a : 0); s1[y] = row + (a < b ? b : 0);
  }
  const diag = new Float64Array(16);
  for (let f = 0; f < 16; f++) diag[f] = lambda + (f & 1) + ((f >> 1) & 1) + ((f >> 2) & 1) + ((f >> 3) & 1);

  // ---- CG from r = 0: res = rhs, p = res ----
  const r = new Float64Array(P), p = new Float64Array(P), Ap = new Float64Array(P);
  let rr = 0;
  for (let y = 0; y < H; y++) for (let i = s0[y]!, e = s1[y]!; i < e; i++) { const v = res[i]!; p[i] = v; rr += v * v; }
  const rr0 = rr || 1;
  let it = 0;
  for (; it < iters && rr / rr0 > tol; it++) {
    let pAp = 0;
    for (let y = 0; y < H; y++) {
      for (let i = s0[y]!, e = s1[y]!; i < e; i++) {
        const f = fl[i]!, pk = p[i]!;
        let v = diag[f]! * pk;
        if (f === 15) v -= p[i + 1]! + p[i - 1]! + p[i + W]! + p[i - W]!;
        else if (f !== 0) {
          if (f & 1) v -= p[i + 1]!;
          if (f & 2) v -= p[i - 1]!;
          if (f & 4) v -= p[i + W]!;
          if (f & 8) v -= p[i - W]!;
        }
        Ap[i] = v; pAp += pk * v;
      }
    }
    const alpha = rr / (pAp || 1);
    let rr2 = 0;
    for (let y = 0; y < H; y++) {
      for (let i = s0[y]!, e = s1[y]!; i < e; i++) {
        r[i]! += alpha * p[i]!;
        const v = res[i]! - alpha * Ap[i]!;
        res[i] = v; rr2 += v * v;
      }
    }
    const beta = rr2 / (rr || 1);
    for (let y = 0; y < H; y++) for (let i = s0[y]!, e = s1[y]!; i < e; i++) p[i] = res[i]! + beta * p[i]!;
    rr = rr2;
  }

  // rms over the solved cells (those with an edge). Over every cell with a base depth it would be
  // diluted by the edgeless ones the solve never touches.
  let ss = 0, solved = 0;
  for (let y = 0; y < H; y++) {
    for (let i = s0[y]!, e = s1[y]!; i < e; i++) {
      if (fl[i] !== 0) solved++;
      let v = r[i]!;
      if (v === 0) continue;
      v = v > clamp ? clamp : v < -clamp ? -clamp : v;
      rOut[i] = v; ss += v * v;
    }
  }
  return { r: rOut, iters: it, rms: solved ? Math.sqrt(ss / solved) : 0 };
}

/**
 * Temporal median of three frames of a residual (x7:345-355), 0 = missing. Where the frame and both
 * neighbours hold a value, their median; elsewhere the frame's own value (a missing centre stays
 * missing, so the median never gives a cell a value its own frame lacked). A null neighbour is the
 * clip's end, replicated as in x7: the frame passes through. `out` may alias `cur`.
 */
export function median3(prev: Float32Array | null, cur: Float32Array, next: Float32Array | null, out?: Float32Array): Float32Array {
  const fn = "median3";
  if (!cur) throw new Error(`${fn}: cur is required, got ${cur === null ? "null" : typeof cur}`);
  const n = cur.length;
  if (prev && prev.length !== n) throw new Error(`${fn}: prev holds ${prev.length} values, cur ${n}`);
  if (next && next.length !== n) throw new Error(`${fn}: next holds ${next.length} values, cur ${n}`);
  const o = out ?? new Float32Array(n);
  checkLen(fn, "out", o, n);
  if (!prev || !next) {
    if (o !== cur) o.set(cur);
    return o;
  }
  for (let i = 0; i < n; i++) {
    const a = prev[i]!, b = cur[i]!, c = next[i]!;
    o[i] = a !== 0 && b !== 0 && c !== 0
      ? (a < b ? (b < c ? b : a < c ? c : a) : (a < c ? a : b < c ? c : b))
      : b;
  }
  return o;
}
