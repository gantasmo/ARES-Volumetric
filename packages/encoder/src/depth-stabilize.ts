/**
 * Depth-run stabilizer: turns a stack of independently-inferred depth maps into ONE coherent
 * volume that can be unprojected frame by frame without the relief boiling.
 *
 * A monocular depth model is affine-invariant: every frame's disparity is correct only up to an
 * unknown scale and shift, chosen afresh for each frame. Play those frames back raw and the whole
 * scene surges toward and away from the camera at 30 Hz — the single worst artifact of naive
 * depth-to-mesh conversion, and the reason VJ-9000's depthcloud EMA-smoothed everything with a flat
 * alpha of 0.5 (which cures the surge by smearing every moving edge into a comet tail).
 *
 * Three steps, in order:
 *
 *  (a) Per-frame affine alignment. Solve the scale `a` and shift `b` that carry frame t onto the
 *      ALREADY-ALIGNED frame t-1, over the pixels the RGB says did not move. Two passes: the first
 *      from the identity estimate, the second after rejecting residual outliers by MAD (a moving
 *      subject is a large coherent outlier population, and an unrejected one drags the fit toward
 *      the subject's own depth change). `a` is clamped to [0.5, 2] so one bad frame cannot rescale
 *      the clip. See robustAffine below for why the solve matches moments instead of regressing.
 *
 *  (b) Motion-gated bidirectional temporal smoothing. Per pixel, alpha rises from `alphaStatic` on
 *      still pixels to 1 on moving ones, so a static wall is averaged hard while a waving hand
 *      takes its new value outright — no ghosting. Forward and backward EMAs are averaged, which
 *      is zero-lag: this runs offline over the whole clip, so there is no reason to accept the
 *      half-window delay a causal filter would impose.
 *
 *  (c) Clip-wide robust normalisation to [0,1] with 1 = nearest, from the 0.5/99.5 percentiles of
 *      a pixel subsample of every frame. Clip-wide, not per-frame, because a per-frame range is
 *      exactly the scale inconsistency step (a) just removed.
 *
 * Metric runs take (a) and (b) in the disparity domain (1/z, where the model's error actually
 * lives and where a linear fit is the right model) and come back out in metres; (c) is skipped —
 * metres are already absolute and normalising them would throw the scale away.
 */
import type { DepthKind } from "./depth-io.js";
import { memoryStore, type FrameStore } from "./depth-store.js";

export interface StabilizeOptions {
  kind: DepthKind;
  /** 0..1; 0 disables temporal smoothing entirely, 1 uses `alphaStatic` as given. Default 0.7. */
  strength?: number;
  /** EMA alpha on a perfectly still pixel at strength 1 (lower = smoother). The alpha in use is
   *  `alphaStatic ^ strength`. Default 0.05. */
  alphaStatic?: number;
  /** Motion gate, in 0..255 mean-abs-channel-difference units. Defaults 2 and 12. */
  motionLo?: number;
  motionHi?: number;
  /** Per-frame scale/shift alignment. Default true; false for a run whose engine is temporally
   *  consistent by construction (depth.json `temporal: "model"`), where a fit can only add noise. */
  align?: boolean;
  /** Grow the RGB gate across the interior of a moving surface (growGate below). Default true. */
  grow?: boolean;
  /** Temporal median window in frames (1 = off, 3 or 5) run after the EMA; see temporalMedian.
   *  Default 5, and off when strength is 0. */
  median?: number;
  /** Progress over the three whole-clip passes: `done` of `total` frame visits. */
  onProgress?: (done: number, total: number) => void;
}

export interface StabilizeStats {
  /** Frames that received an affine fit (every frame but the first). */
  aligned: number;
  scaleMin: number;
  scaleMax: number;
  scaleMean: number;
  shiftMean: number;
  /** Fits whose scale hit the [0.5, 2] clamp. */
  clamped: number;
  /** Fits that fell back to identity (too few usable pixels, or a singular normal matrix). */
  degenerate: number;
  /** Mean gate value over the clip, 0..255. */
  motionMean: number;
  /** True when RGB drove the gate; false when it was derived from the depth change itself. */
  rgbGated: boolean;
  /** Frames whose gate was grown across a moving surface's interior (growGate). */
  grownFrames: number;
  /** Effective EMA alpha on a still pixel at this strength. */
  alphaStill: number;
}

export interface StabilizeResult {
  /** relative-disparity → [0,1] with 1 = nearest. metric-depth → metres. */
  maps: Float32Array;
  kind: DepthKind;
  /** Normalisation percentiles. Disparity units for a relative run, metres for a metric one. */
  lo: number;
  hi: number;
  stats: StabilizeStats;
}

const EPS_Z = 1e-4;        // metres; guards 1/z on a zero or negative metric sample
const EPS_DISP = 1e-6;     // guards z = 1/disparity coming back out
const MAD_K = 3 * 1.4826;  // 3 sigma, via the normal-consistent MAD estimator

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

function smoothstep(e0: number, e1: number, x: number): number {
  if (e1 <= e0) return x >= e1 ? 1 : 0;
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** Separable 3x3 box blur of a 0..255 gate field, in place. Keeps a single pixel of sensor noise
 *  from opening the gate; a real motion edge is always several pixels wide. */
function boxBlur3(field: Uint8Array, off: number, w: number, h: number, scratch: Uint8Array): void {
  for (let y = 0; y < h; y++) {
    const r = y * w;
    for (let x = 0; x < w; x++) {
      const a = field[off + r + (x > 0 ? x - 1 : 0)]!;
      const b = field[off + r + x]!;
      const c = field[off + r + (x < w - 1 ? x + 1 : w - 1)]!;
      scratch[r + x] = ((a + b + c) / 3) | 0;
    }
  }
  for (let y = 0; y < h; y++) {
    const rU = (y > 0 ? y - 1 : 0) * w, rM = y * w, rD = (y < h - 1 ? y + 1 : h - 1) * w;
    for (let x = 0; x < w; x++) {
      field[off + rM + x] = ((scratch[rU + x]! + scratch[rM + x]! + scratch[rD + x]!) / 3) | 0;
    }
  }
}

/**
 * The RGB motion gate, one frame at a time. Feed it the sampled frames in order; it writes the gate
 * for frame t (the blurred mean-abs RGB change between frames t-1 and t, all zero for frame 0).
 */
export class RgbMotionGate {
  private prevRgb: Uint8Array | null = null;
  private readonly scratch: Uint8Array;

  constructor(private readonly width: number, private readonly height: number) {
    this.scratch = new Uint8Array(width * height);
  }

  next(rgb: Uint8Array, rgbOff: number, out: Uint8Array, outOff = 0): void {
    const P = this.width * this.height;
    const prev = this.prevRgb;
    if (!prev) {
      out.fill(0, outOff, outOff + P);
      this.prevRgb = rgb.slice(rgbOff, rgbOff + P * 3);
      return;
    }
    for (let i = 0; i < P; i++) {
      const j = i * 3, c = rgbOff + j;
      const d = Math.abs(rgb[c]! - prev[j]!) + Math.abs(rgb[c + 1]! - prev[j + 1]!) + Math.abs(rgb[c + 2]! - prev[j + 2]!);
      out[outOff + i] = (d / 3) | 0;
    }
    boxBlur3(out, outOff, this.width, this.height, this.scratch);
    prev.set(rgb.subarray(rgbOff, rgbOff + P * 3));
  }
}

/** Gate units lost per pixel of travel: evidence reaches 255 / GROW_DECAY pixels from its seed. */
const GROW_DECAY = 2;

/**
 * Grow the RGB gate across the interior of a moving surface.
 *
 * The RGB difference sees motion only where pixels change. A large untextured object moving as a
 * whole changes pixels along its leading and trailing edges and nowhere inside, so its interior
 * reads as static, and that interior (one big coherent depth population) is what drags the
 * scale/shift fit once it covers enough of the frame to defeat the MAD rejection.
 *
 * The depth itself says where that interior is. Seeds are the pixels the RGB gate already calls
 * moving AND that came NEARER under the provisional fit: the leading edge of an occluder, which
 * wears the occluder's depth in the current frame. (The trailing edge wears the revealed
 * background's, and growing from it would flood the static background instead.) From the seeds the
 * evidence travels over the current frame's depth surface: free across smooth depth, stopped by a
 * silhouette, and fading by GROW_DECAY per pixel so that a contact point (feet on a floor) leaks a
 * patch, not the room. Two raster sweeps each way is the standard geodesic approximation.
 *
 * `d` is the current frame already carried onto `ref` by the provisional fit. Returns the number
 * of pixels whose gate rose.
 */
function growGate(gate: Uint8Array, d: Float32Array, ref: Float32Array, w: number, h: number, spread: number, motionHi: number, field: Float32Array): number {
  const P = w * h;
  if (!(spread > 0)) return 0;
  const near = 0.05 * spread;
  let seeds = 0;
  for (let i = 0; i < P; i++) {
    const s = gate[i]! >= motionHi && d[i]! - ref[i]! > near;
    field[i] = s ? 255 : 0;
    if (s) seeds++;
  }
  if (!seeds) return 0;
  // A step costs nothing under 1% of the spread (shading, model noise) and everything at 6%.
  const k = 255 / (0.05 * spread), free = 0.01 * spread;
  const step = (a: number, b: number): number => {
    const j = Math.abs(d[a]! - d[b]!) - free;
    return GROW_DECAY + (j > 0 ? j * k : 0);
  };
  for (let sweep = 0; sweep < 2; sweep++) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let v = field[i]!;
      if (x > 0) { const c = field[i - 1]! - step(i, i - 1); if (c > v) v = c; }
      if (y > 0) { const c = field[i - w]! - step(i, i - w); if (c > v) v = c; }
      field[i] = v;
    }
    for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      let v = field[i]!;
      if (x < w - 1) { const c = field[i + 1]! - step(i, i + 1); if (c > v) v = c; }
      if (y < h - 1) { const c = field[i + w]! - step(i, i + w); if (c > v) v = c; }
      field[i] = v;
    }
  }
  let grown = 0;
  for (let i = 0; i < P; i++) {
    const v = field[i]! | 0;
    if (v > gate[i]!) { gate[i] = v; grown++; }
  }
  return grown;
}

/**
 * Per-pixel temporal median over `window` frames (3 or 5), in place, the clip's ends replicated.
 *
 * The EMA passes smooth only where the RGB gate calls a pixel still; where it moves they follow the
 * new value outright so a moving edge never ghosts, and there the model's frame-to-frame noise
 * reaches the mesh untouched. At the silhouette cut that noise is one-frame blinks: measured on 300
 * frames of a film, 295 grid edges per frame were cut for exactly one frame in moving regions. A
 * median removes a sample that disagrees with its neighbours in time and leaves a monotonic run
 * exactly as it was (the median of a monotonic window is its centre), so steady motion keeps its
 * timing and a scene cut stays a cut. What it does remove is an event shorter than half the window:
 * 3 frames drops one-frame events only, 5 frames also two-frame ones, so an object that crosses a
 * pixel within two frames (83 ms at 24 fps) loses its depth there. With the cut hysteresis in
 * depth-mesh.ts, the blinks in moving regions were 263 per frame without a median, 194 with 3
 * frames and 142 with 5. Resident: `window` frames of originals, whatever the clip length.
 */
export function temporalMedian(store: FrameStore<Float32Array>, frames: number, P: number, window: number): void {
  const half = window >> 1;
  const read = (t: number) => { const a = new Float32Array(P); store.read(Math.min(frames - 1, Math.max(0, t)), a); return a; };
  // buf[k] holds the ORIGINAL of frame t - half + k (clamped): the store is overwritten behind t.
  const buf: Float32Array[] = [];
  for (let k = -half; k <= half; k++) buf.push(read(k));
  const out = new Float32Array(P);
  for (let t = 0; t < frames; t++) {
    if (window === 3) {
      const a = buf[0]!, b = buf[1]!, c = buf[2]!;
      for (let i = 0; i < P; i++) {
        const x = a[i]!, y = b[i]!, z = c[i]!;
        out[i] = x < y ? (y < z ? y : x < z ? z : x) : (x < z ? x : y < z ? z : y);
      }
    } else {
      const a = buf[0]!, b = buf[1]!, c = buf[2]!, d = buf[3]!, e = buf[4]!;
      for (let i = 0; i < P; i++) {
        // Six-comparison median of five. Order the pairs (v0,v1) and (v2,v3) and put the pair with
        // the smaller minimum first: v0 is then below three others and cannot be the median. Pair
        // v1 with v4 in its place and repeat: the new minimum is below three others as well. The
        // median is the smaller of the two maxima left.
        let v0 = a[i]!, v1 = b[i]!, v2 = c[i]!, v3 = d[i]!, v4 = e[i]!, s: number;
        if (v0 > v1) { s = v0; v0 = v1; v1 = s; }
        if (v2 > v3) { s = v2; v2 = v3; v3 = s; }
        if (v0 > v2) { s = v1; v1 = v3; v3 = s; v2 = v0; }
        if (v1 > v4) { s = v1; v1 = v4; v4 = s; }
        if (v1 > v2) { s = v4; v4 = v3; v3 = s; v2 = v1; }
        out[i] = v2 < v4 ? v2 : v4;
      }
    }
    store.write(t, out);
    buf.shift();
    const next = t + half + 1;
    buf.push(next <= frames - 1 ? read(next) : buf[buf.length - 1]!);
  }
}

export interface StabilizeStreamInput {
  width: number;
  height: number;
  frames: number;
  /** Raw engine values of frame t (disparity, or metres for a metric run). */
  readRaw(t: number, out: Float32Array): void;
  /** Precomputed RGB gate (frame t = the change t-1 -> t), or null to derive it from the depth. */
  motion: FrameStore<Uint8Array> | null;
  /** Scratch store factories: memory for a short clip, files for a long one. */
  makeF32(name: string): FrameStore<Float32Array>;
  makeU8(name: string): FrameStore<Uint8Array>;
}

export interface StabilizeStreamResult {
  /** relative-disparity -> [0,1] with 1 = nearest. metric-depth -> metres. The caller closes it. */
  out: FrameStore<Float32Array>;
  kind: DepthKind;
  lo: number;
  hi: number;
  stats: StabilizeStats;
}

/**
 * The stabilizer over frame stores. Sequential passes, each touching one or two frames at a time,
 * so resident memory is a handful of frames whatever the clip length:
 *   1. forward: domain conversion, affine alignment onto the previous aligned frame, percentile
 *      sampling; aligned frames go to a store
 *   2. (no RGB only) forward: the gate derived from the aligned depth change
 *   3. backward: the backward EMA, to a second store
 *   4. forward: the forward EMA, averaged with the backward one, normalised, written in place
 */
export function stabilizeDepthStream(input: StabilizeStreamInput, opts: StabilizeOptions): StabilizeStreamResult {
  const { width, height, frames } = input;
  const P = width * height;
  const metric = opts.kind === "metric-depth";
  const strength = clamp(opts.strength ?? 0.7, 0, 1);
  const alphaStatic = clamp(opts.alphaStatic ?? 0.05, 0.01, 1);
  const motionLo = opts.motionLo ?? 2;
  const motionHi = opts.motionHi ?? 12;
  const align = opts.align ?? true;
  const grow = opts.grow ?? true;
  const progress = opts.onProgress;
  const rgbGated = !!input.motion;

  const work = input.makeF32("aligned");
  const motion = input.motion ?? input.makeU8("motion");
  const scratch = new Uint8Array(P);

  // ---- (a) affine alignment ----------------------------------------------------------------
  let scaleMin = Infinity, scaleMax = -Infinity, scaleSum = 0, shiftSum = 0, clampedN = 0, degenerate = 0, fitN = 0;
  const sampleN = Math.min(P, 4096);
  const resid = new Float32Array(sampleN);
  const sampX = new Float32Array(sampleN);
  const residStride = Math.max(1, Math.floor(P / sampleN));

  /**
   * Robust weighted fit of the `a`,`b` that carries frame `X` onto frame `Y`.
   *
   * Seeded at IDENTITY, not at a plain regression. Consecutive frames of a run are already close,
   * so the residual at identity is exactly what separates the background from whatever moved —
   * whereas an unrejected first fit is dragged bodily by the moving subject (a foreground cluster
   * far out along x has enormous leverage) and hands the MAD step a fit so wrong that nothing
   * looks like an outlier any more. Two passes: each finds the inliers of the current estimate and
   * re-solves over them.
   *
   * The solve itself is MOMENT MATCHING (a = sd(y)/sd(x), b = mean(y) - a.mean(x)) over the
   * inliers, not the ordinary least-squares slope. Least squares regressing the previous frame on
   * the current one is biased by construction whenever the current frame carries noise of its own:
   * its slope is attenuated by var(signal)/(var(signal) + var(noise)), which is ALWAYS below 1.
   * A single frame hides it; a chain of them multiplies it. At the ~2% per-frame disparity noise a
   * monocular model really has, that is a 0.4% shrink per frame and a 70% collapse of the clip's
   * depth range by frame 300 — the scene flattening into a wall as it plays. Moment matching has
   * no such bias when both frames carry the same noise, which two frames from one model do.
   *
   * When the gate leaves less than a tenth of the frame open (a handheld camera moves every
   * pixel), the fit runs ungated: moment matching compares two depth distributions, and those stay
   * comparable under camera motion even though no single pixel stays put.
   */
  function robustAffine(X: Float32Array, Y: Float32Array, gateIn: Uint8Array | null): { a: number; b: number; clamped: boolean; ok: boolean; spread: number } {
    let g = gateIn;
    if (g) {
      let open = 0;
      for (let i = 0; i < P; i++) open += 1 - smoothstep(motionLo, motionHi, g[i]!);
      if (open < 0.1 * P) g = null;
    }
    const gg = g;
    const wAt = (i: number) => (gg ? 1 - smoothstep(motionLo, motionHi, gg[i]!) : 1);
    // Reference population: every gated-in pixel, no rejection yet. Its count and its x-spread are
    // what each pass below is checked against, so a rejection that guts the fit is never taken.
    let SwAll = 0, SxAll = 0, SxxAll = 0;
    for (let i = 0; i < P; i++) {
      const w = wAt(i);
      if (w <= 0) continue;
      const x = X[i]!;
      SwAll += w; SxAll += w * x; SxxAll += w * x * x;
    }
    if (SwAll < 16) return { a: 1, b: 0, clamped: false, ok: false, spread: 0 };
    const varAll = SxxAll / SwAll - (SxAll / SwAll) * (SxAll / SwAll);

    let a = 1, b = 0, Sw = SwAll, Sx = SxAll, Sy = 0, ok = false, spreadOut = 0;
    for (let pass = 0; pass < 2; pass++) {
      let k = 0;
      for (let i = 0; i < P && k < sampleN; i += residStride) {
        if (wAt(i) <= 0) continue;
        sampX[k] = X[i]!;
        resid[k] = Math.abs(a * X[i]! + b - Y[i]!);
        k++;
      }
      if (k < 4) break;
      const rv = resid.subarray(0, k); rv.sort();
      const xs = sampX.subarray(0, k); xs.sort();
      const spread = xs[Math.floor(0.9 * (k - 1))]! - xs[Math.floor(0.1 * (k - 1))]!;
      spreadOut = spread;
      // The cut is floored at 1% of the frame's own spread. Without a floor, a frame that already
      // matches its reference exactly has a median residual of zero, and the rejection then keeps
      // only a razor-thin band of x — a perfectly conditioned fit replaced by a meaningless one.
      const cut = Math.max(MAD_K * rv[k >> 1]!, 0.01 * spread, 1e-9);
      let Sw2 = 0, Sx2 = 0, Sy2 = 0, Sxx2 = 0, Syy2 = 0;
      for (let i = 0; i < P; i++) {
        const w = wAt(i);
        if (w <= 0) continue;
        const x = X[i]!, y = Y[i]!;
        if (Math.abs(a * x + b - y) > cut) continue;
        Sw2 += w; Sx2 += w * x; Sy2 += w * y; Sxx2 += w * x * x; Syy2 += w * y * y;
      }
      if (Sw2 < 16 || Sw2 < 0.1 * SwAll) break;
      const mx = Sx2 / Sw2, my = Sy2 / Sw2;
      const varX = Sxx2 / Sw2 - mx * mx, varY = Syy2 / Sw2 - my * my;
      if (varX < 0.1 * varAll || varX <= 0 || varY <= 0) break;
      a = Math.sqrt(varY / varX);
      b = my - a * mx;
      Sw = Sw2; Sx = Sx2; Sy = Sy2;
      ok = true;
    }
    if (!Number.isFinite(a) || !Number.isFinite(b)) return { a: 1, b: 0, clamped: false, ok: false, spread: 0 };
    let clampedHere = false;
    if (a < 0.5 || a > 2) {
      a = clamp(a, 0.5, 2);
      // Re-fit the shift to the clamped scale so the pair still answers the same fit, instead of
      // a scale from one estimate paired with a shift from another.
      b = Sw > 0 ? (Sy - a * Sx) / Sw : 0;
      clampedHere = true;
    }
    return { a, b, clamped: clampedHere, ok, spread: spreadOut };
  }

  // Percentile subsample, taken on the fly at the same strided positions a whole-stack pass would
  // visit, so a streamed clip and an in-memory one normalise to the same numbers.
  const total = frames * P;
  const stride = Math.max(1, Math.floor(total / (1 << 21)));
  const sample = new Float32Array(Math.ceil(total / stride));
  let sampleK = 0;

  // Pass 1, chained: every frame onto its ALREADY-ALIGNED predecessor.
  let cur = new Float32Array(P), prev = new Float32Array(P);
  const gate = new Uint8Array(P);
  const fitted = new Float32Array(P), field = new Float32Array(P);
  let grownFrames = 0;
  progress?.(0, frames * 3);
  for (let t = 0; t < frames; t++) {
    input.readRaw(t, cur);
    // Domain: everything below works on disparity (1/z), where the model's own error lives.
    for (let i = 0; i < P; i++) {
      const v = cur[i]!;
      cur[i] = !Number.isFinite(v) ? 0 : metric ? 1 / Math.max(v, EPS_Z) : v;
    }
    if (t >= 1 && align) {
      if (rgbGated) motion.read(t, gate);
      let f = robustAffine(cur, prev, rgbGated ? gate : null);
      if (rgbGated && grow) {
        // Provisional fit -> which moving pixels came nearer -> grow the gate over that surface ->
        // fit again without it. The grown gate is what the smoothing passes read too: a surface
        // whose depth is changing as a whole must take its new value, not be averaged toward the old.
        for (let i = 0; i < P; i++) fitted[i] = f.a * cur[i]! + f.b;
        if (growGate(gate, fitted, prev, width, height, f.spread * f.a, motionHi, field) > 0) {
          motion.write(t, gate);
          grownFrames++;
          f = robustAffine(cur, prev, gate);
        }
      }
      if (!f.ok) degenerate++;
      if (f.clamped) clampedN++;
      if (!(f.a === 1 && f.b === 0)) for (let i = 0; i < P; i++) cur[i] = f.a * cur[i]! + f.b;
      if (f.a < scaleMin) scaleMin = f.a;
      if (f.a > scaleMax) scaleMax = f.a;
      scaleSum += f.a; shiftSum += f.b; fitN++;
    }
    const base = t * P;
    for (let g = Math.ceil(base / stride) * stride; g < base + P; g += stride) sample[sampleK++] = cur[g - base]!;
    work.write(t, cur);
    const sw = prev; prev = cur; cur = sw;
    progress?.(t + 1, frames * 3);
  }

  // ---- robust range (used both for the depth-derived gate and for step (c)) -----------------
  const sorted = sample.subarray(0, sampleK).sort();
  const at = (q: number) => sorted[clamp(Math.round(q * (sampleK - 1)), 0, sampleK - 1)]!;
  let lo = at(0.005), hi = at(0.995);
  if (!(hi > lo)) hi = lo + 1;

  // No RGB (ffmpeg unavailable, or a caller that only has depth): derive the gate from the depth
  // change itself, scaled by the clip's own disparity spread so motionLo/motionHi keep meaning.
  let motionSum = 0;
  if (!rgbGated) {
    const k = 255 / (hi - lo);
    gate.fill(0);
    motion.write(0, gate);
    if (frames > 0) work.read(0, prev);
    for (let t = 1; t < frames; t++) {
      work.read(t, cur);
      for (let i = 0; i < P; i++) gate[i] = clamp(Math.abs(cur[i]! - prev[i]!) * k, 0, 255) | 0;
      boxBlur3(gate, 0, width, height, scratch);
      motion.write(t, gate);
      const sw = prev; prev = cur; cur = sw;
    }
  }
  for (let t = 0; t < frames; t++) {
    motion.read(t, gate);
    for (let i = 0; i < P; i++) motionSum += gate[i]!;
  }

  // ---- (b) motion-gated bidirectional EMA, then (c) normalise or return to metres -----------
  // The still-pixel alpha is alphaStatic ^ strength: strength 0 leaves it at 1 (the filter is off),
  // strength 1 uses alphaStatic verbatim, and the default 0.7 gives 0.12 for the default 0.05, an
  // effective window of about 16 frames each way. Measured on a film run, the old linear map (0.475
  // at 0.7) left still pixels changing by 0.44 % of the depth range per frame at the 90th
  // percentile, which the silhouette cut turns into holes flickering along every edge.
  const alphaStill = Math.pow(alphaStatic, strength);
  const alphaLut = new Float64Array(256);
  for (let m = 0; m < 256; m++) alphaLut[m] = alphaStill + (1 - alphaStill) * smoothstep(motionLo, motionHi, m);
  const kNorm = 1 / (hi - lo);
  const finish = (buf: Float32Array) => {
    if (metric) for (let i = 0; i < P; i++) buf[i] = clamp(1 / Math.max(buf[i]!, EPS_DISP), 1e-3, 1e4);
    else for (let i = 0; i < P; i++) buf[i] = clamp((buf[i]! - lo) * kNorm, 0, 1);
  };
  if (strength > 0 && frames > 1) {
    const bw = input.makeF32("backward");
    const next = new Float32Array(P);
    work.read(frames - 1, next);
    bw.write(frames - 1, next);
    for (let t = frames - 2; t >= 0; t--) {
      work.read(t, cur);
      motion.read(t + 1, gate);
      for (let i = 0; i < P; i++) {
        const al = alphaLut[gate[i]!]!;
        next[i] = al * cur[i]! + (1 - al) * next[i]!;
      }
      bw.write(t, next);
      progress?.(frames + (frames - 1 - t), frames * 3);
    }
    const fprev = new Float32Array(P), b = new Float32Array(P);
    for (let t = 0; t < frames; t++) {
      work.read(t, cur);
      bw.read(t, b);
      if (t === 0) { fprev.set(cur); for (let i = 0; i < P; i++) cur[i] = 0.5 * (fprev[i]! + b[i]!); }
      else {
        motion.read(t, gate);
        for (let i = 0; i < P; i++) {
          const al = alphaLut[gate[i]!]!;
          const f = al * cur[i]! + (1 - al) * fprev[i]!;
          fprev[i] = f;
          cur[i] = 0.5 * (f + b[i]!);
        }
      }
      finish(cur);
      work.write(t, cur);
      progress?.(frames * 2 + t + 1, frames * 3);
    }
    bw.close();
  } else {
    for (let t = 0; t < frames; t++) {
      work.read(t, cur);
      finish(cur);
      work.write(t, cur);
      progress?.(frames * 2 + t + 1, frames * 3);
    }
  }
  if (!rgbGated) motion.close();
  // ---- temporal median over the smoothed, normalised frames: short outliers out --------------
  // The normalisation is monotonic per pixel, so the median commutes with it.
  const medianWindow = strength > 0 ? Math.max(1, Math.min(5, (opts.median ?? 5) | 1)) : 1;
  if (medianWindow > 1 && frames > 2) temporalMedian(work, frames, P, medianWindow);
  progress?.(frames * 3, frames * 3);

  let outLo = lo, outHi = hi;
  if (metric) {
    outLo = clamp(1 / Math.max(hi, EPS_DISP), 1e-3, 1e4);   // nearest percentile, in metres
    outHi = clamp(1 / Math.max(lo, EPS_DISP), 1e-3, 1e4);   // farthest percentile, in metres
  }
  return {
    out: work,
    kind: opts.kind,
    lo: outLo,
    hi: outHi,
    stats: {
      aligned: align ? Math.max(0, frames - 1) : 0,
      scaleMin: Number.isFinite(scaleMin) ? scaleMin : 1,
      scaleMax: Number.isFinite(scaleMax) ? scaleMax : 1,
      scaleMean: fitN ? scaleSum / fitN : 1,
      shiftMean: fitN ? shiftSum / fitN : 0,
      clamped: clampedN,
      degenerate,
      motionMean: total ? motionSum / total : 0,
      rgbGated,
      grownFrames,
      alphaStill,
    },
  };
}

/** The in-memory form: a whole stack in, a whole stack out. Short clips and tests. */
export function stabilizeDepth(
  maps: Float32Array,
  width: number,
  height: number,
  frames: number,
  rgb: Uint8Array | null,
  opts: StabilizeOptions,
): StabilizeResult {
  const P = width * height;
  if (maps.length < frames * P) throw new Error(`stabilizeDepth: maps holds ${maps.length} floats, expected ${frames * P} (${frames}x${height}x${width})`);
  if (rgb && rgb.length < frames * P * 3) throw new Error(`stabilizeDepth: rgb holds ${rgb.length} bytes, expected ${frames * P * 3}`);
  let motion: (FrameStore<Uint8Array> & { data: Uint8Array }) | null = null;
  if (rgb) {
    motion = memoryStore((n) => new Uint8Array(n), frames, P);
    const gate = new RgbMotionGate(width, height);
    for (let t = 0; t < frames; t++) gate.next(rgb, t * P * 3, motion.data, t * P);
  }
  const stores: (FrameStore<Float32Array> & { data: Float32Array })[] = [];
  const r = stabilizeDepthStream({
    width, height, frames, motion,
    readRaw: (t, out) => out.set(maps.subarray(t * P, (t + 1) * P)),
    makeF32: () => { const s = memoryStore((n) => new Float32Array(n), frames, P); stores.push(s); return s; },
    makeU8: () => memoryStore((n) => new Uint8Array(n), frames, P),
  }, opts);
  return { maps: stores[0]!.data, kind: r.kind, lo: r.lo, hi: r.hi, stats: r.stats };
}
