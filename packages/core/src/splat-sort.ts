/**
 * Back-to-front splat ordering for alpha compositing (spec §6.8 "per-frame depth sort").
 *
 * CPU counting sort on a 16-bit view-depth key: O(n + 65536) per sort, no comparisons, and the
 * output is an index list the renderers use as an indirection (`order[instance] → splat id`) so
 * the splat data itself is uploaded once per frame and never reshuffled. Depth is computed
 * straight from the quantized positions (the same dequant the shader does), so the sort can never
 * disagree with what is drawn.
 *
 * Re-sorting is the cost driver on big clouds (spec notes: "the per-frame depth sort … is the
 * number to bench first"), so `SplatSorter` only re-sorts when the view direction moved more than
 * a threshold or the frame changed; camera pans and zooms along a fixed direction reuse the order.
 * A GPU radix sort is the planned follow-up for multi-million-splat scenes.
 */
import type { Aabb } from "./quant.js";
import type { Mat4 } from "./camera.js";

const KEYS = 1 << 16;

/**
 * Fill `out` (length ≥ count) with splat ids ordered far → near for the given view matrix
 * (column-major; view-space −z is forward). `modelView` should already include any model transform.
 */
export function sortSplatsBackToFront(
  positionsQ: Uint16Array, count: number, aabb: Aabb, invLevels: number, modelView: Mat4, out: Uint32Array,
  scratch?: { depth: Float32Array; hist: Uint32Array },
): void {
  if (count === 0) return;
  const sx = (aabb.max[0] - aabb.min[0]) * invLevels;
  const sy = (aabb.max[1] - aabb.min[1]) * invLevels;
  const sz = (aabb.max[2] - aabb.min[2]) * invLevels;
  // Row 2 of the view matrix (column-major m[2], m[6], m[10], m[14]) gives view-space z.
  const vx = modelView[2]!, vy = modelView[6]!, vz = modelView[10]!, vw = modelView[14]!;
  // Fold the dequant into the dot product: z = vw + vx*(min.x + qx*sx) + …
  const c0 = vw + vx * aabb.min[0] + vy * aabb.min[1] + vz * aabb.min[2];
  const kx = vx * sx, ky = vy * sy, kz = vz * sz;

  const depth = scratch?.depth && scratch.depth.length >= count ? scratch.depth : new Float32Array(count);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < count; i++) {
    const b = i * 4;
    const z = c0 + positionsQ[b]! * kx + positionsQ[b + 1]! * ky + positionsQ[b + 2]! * kz;
    depth[i] = z;
    if (z < lo) lo = z;
    if (z > hi) hi = z;
  }
  const hist = scratch?.hist && scratch.hist.length >= KEYS ? scratch.hist : new Uint32Array(KEYS);
  hist.fill(0);
  const span = hi - lo;
  const scale = span > 0 ? (KEYS - 1) / span : 0;
  // Farther = more negative z → smaller key. Counting sort ascending by key = far first.
  for (let i = 0; i < count; i++) hist[((depth[i]! - lo) * scale) | 0]!++;
  let acc = 0;
  for (let k = 0; k < KEYS; k++) { const c = hist[k]!; hist[k] = acc; acc += c; }
  for (let i = 0; i < count; i++) {
    const k = ((depth[i]! - lo) * scale) | 0;
    out[hist[k]!++] = i;
  }
}

export class SplatSorter {
  private order: Uint32Array | null = null;
  private scratch = { depth: new Float32Array(0), hist: new Uint32Array(KEYS) };
  private lastDir: [number, number, number] = [0, 0, 0];
  private lastFrame = -1;
  private lastCount = 0;
  /** cos(angle) threshold for view-direction change before a re-sort (≈ 1.5°). */
  private threshold = Math.cos((1.5 * Math.PI) / 180);

  /** Returns a fresh back-to-front order when one is needed, else null (keep the last upload). */
  update(positionsQ: Uint16Array, count: number, aabb: Aabb, invLevels: number, modelView: Mat4, frameId: number): Uint32Array | null {
    const dx = modelView[2]!, dy = modelView[6]!, dz = modelView[10]!;
    const l = Math.hypot(dx, dy, dz) || 1;
    const dir: [number, number, number] = [dx / l, dy / l, dz / l];
    const same = frameId === this.lastFrame && count === this.lastCount &&
      dir[0] * this.lastDir[0] + dir[1] * this.lastDir[1] + dir[2] * this.lastDir[2] > this.threshold;
    if (same && this.order) return null;
    if (!this.order || this.order.length < count) this.order = new Uint32Array(count);
    if (this.scratch.depth.length < count) this.scratch.depth = new Float32Array(count);
    sortSplatsBackToFront(positionsQ, count, aabb, invLevels, modelView, this.order, this.scratch);
    this.lastDir = dir; this.lastFrame = frameId; this.lastCount = count;
    return this.order.length === count ? this.order : this.order.subarray(0, count);
  }

  /** Force the next update() to re-sort (e.g. after the clip or model transform changed). */
  invalidate(): void { this.lastFrame = -1; }
}
