/**
 * Sculpt op (action:"sculpt", sculpt+paint plan §A) — world-anchored vertex displacement inside a
 * range's interpolated region, per frame against that frame's own mesh. Same weighting law as
 * paint (SDF-feathered falloff), same "bake-side, never a texel coordinate" law, so a sculpt
 * authored on one frame of a per-frame-reconstructed capture applies wherever the region lands.
 *
 * Every brush is weld-aware: atlased meshes duplicate vertices along chart seams, and moving one
 * copy without its twins would tear the seam open. Displacements are computed on position-welded
 * groups (hole-patch.ts weldByPosition) and copied to every duplicate.
 */
import { prepareRangeSdfAt, type EditRange } from "@ares/core";
import type { EncodeMeshFrame } from "./geometry-encode.js";
import { computeSmoothNormals } from "./geometry-encode.js";
import { weldByPosition } from "./hole-patch.js";

export type SculptBrush = "move" | "inflate" | "smooth" | "flatten" | "pinch";

export interface SculptOp {
  range: EditRange;
  startFrame: number;
  endFrame: number;
  brush: SculptBrush;
  amount: number;
  offset: [number, number, number];
  iterations: number;
  /** falloff half-width in world units; 0 = derive per frame from the region */
  feather: number;
}

/** Validate and collect every enabled sculpt range (the encoder refuses malformed payloads loudly). */
export function collectSculptOps(ranges: EditRange[], frameCount: number): SculptOp[] {
  const out: SculptOp[] = [];
  for (const r of ranges) {
    if (r.action !== "sculpt" || r.enabled === false) continue;
    const id = r.id ?? "?";
    const p = r.sculpt ?? {};
    const brush = (p.brush ?? "move") as SculptBrush;
    if (!["move", "inflate", "smooth", "flatten", "pinch"].includes(brush)) throw new Error(`sculpt ${id}: unknown brush ${JSON.stringify(p.brush)}`);
    const amount = p.amount ?? (brush === "smooth" ? 1 : brush === "inflate" ? 0 : 0.5);
    if (!Number.isFinite(amount)) throw new Error(`sculpt ${id}: amount must be a number`);
    if ((brush === "flatten" || brush === "pinch") && (amount < 0 || amount > 1)) throw new Error(`sculpt ${id}: ${brush} amount is a 0..1 fraction, got ${amount}`);
    const offset = p.offset ?? [0, 0, 0];
    if (!Array.isArray(offset) || offset.length !== 3 || offset.some((v) => !Number.isFinite(v))) throw new Error(`sculpt ${id}: offset must be [x, y, z]`);
    const iterations = Math.max(1, Math.min(50, Math.round(p.iterations ?? 3)));
    const feather = p.feather !== undefined ? Number(p.feather) : 0;
    if (!(feather >= 0)) throw new Error(`sculpt ${id}: feather must be ≥ 0`);
    out.push({
      range: r, startFrame: Math.max(0, r.startFrame), endFrame: Math.min(frameCount - 1, r.endFrame),
      brush, amount, offset: [offset[0]!, offset[1]!, offset[2]!], iterations, feather,
    });
  }
  return out;
}

/** Mean brush radius of the range's strokes (0 when it has none) — the default feather source. */
function meanBrushRadius(r: EditRange): number {
  let sum = 0, n = 0;
  for (const kf of r.keyframes) for (const v of kf.volumes) if (v.type === "brushStrokes") for (const s of v.strokes) { sum += s.radius; n++; }
  return n ? sum / n : 0;
}

/** smoothstep falloff: 1 inside (d ≤ 0) → 0 at d ≥ feather. */
function weightOf(d: number, feather: number): number {
  if (d <= 0) return 1;
  if (feather <= 0 || d >= feather) return 0;
  const t = 1 - d / feather;
  return t * t * (3 - 2 * t);
}

export interface SculptStats { vertices: number; groups: number; feather: number; }

/**
 * Apply one sculpt op to one frame IN PLACE (positions only; normals are recomputed at mux).
 * Returns null when the region matches nothing at this frame.
 */
export function applySculptToFrame(frame: EncodeMeshFrame, op: SculptOp, frameIndex: number): SculptStats | null {
  const sdf = prepareRangeSdfAt(op.range, frameIndex);
  if (!sdf) return null;
  const pos = frame.positions;
  const n = pos.length / 3;
  const canon = weldByPosition(pos);
  // Per welded group: weight from the group's representative position.
  const weight = new Float32Array(n);
  let touched = 0;
  const groupIdx: number[] = [];
  for (let v = 0; v < n; v++) {
    if (canon[v] !== v) continue;
    const d = sdf(pos[v * 3]!, pos[v * 3 + 1]!, pos[v * 3 + 2]!);
    if (d < 0 || d < op.feather || op.feather === 0) groupIdx.push(v);
  }
  if (!groupIdx.length) return null;
  // Feather default: half the mean brush radius, else 5 % of the selected region's extent.
  let feather = op.feather;
  if (feather === 0) {
    const mr = meanBrushRadius(op.range);
    if (mr > 0) feather = mr * 0.5;
    else {
      const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
      for (const v of groupIdx) { const d = sdf(pos[v * 3]!, pos[v * 3 + 1]!, pos[v * 3 + 2]!); if (d >= 0) continue; for (let a = 0; a < 3; a++) { const x = pos[v * 3 + a]!; if (x < min[a]!) min[a] = x; if (x > max[a]!) max[a] = x; } }
      const ext = Number.isFinite(min[0]!) ? Math.hypot(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!) : 0;
      feather = ext * 0.05;
    }
  }
  const active: number[] = [];
  for (const v of groupIdx) {
    const w = weightOf(sdf(pos[v * 3]!, pos[v * 3 + 1]!, pos[v * 3 + 2]!), feather);
    if (w > 0) { weight[v] = w; active.push(v); }
  }
  if (!active.length) return null;

  const disp = new Float32Array(n * 3);   // per canonical vertex
  if (op.brush === "move") {
    for (const v of active) { const w = weight[v]!; disp[v * 3] = op.offset[0] * w; disp[v * 3 + 1] = op.offset[1] * w; disp[v * 3 + 2] = op.offset[2] * w; }
  } else if (op.brush === "inflate") {
    const nrm = computeSmoothNormals(pos, frame.indices);
    for (const v of active) { const w = weight[v]! * op.amount; disp[v * 3] = nrm[v * 3]! * w; disp[v * 3 + 1] = nrm[v * 3 + 1]! * w; disp[v * 3 + 2] = nrm[v * 3 + 2]! * w; }
  } else if (op.brush === "pinch" || op.brush === "flatten") {
    // Region centroid + best-fit plane normal (area-weighted mean of vertex normals) over the active set.
    let cx = 0, cy = 0, cz = 0, wsum = 0;
    for (const v of active) { const w = weight[v]!; cx += pos[v * 3]! * w; cy += pos[v * 3 + 1]! * w; cz += pos[v * 3 + 2]! * w; wsum += w; }
    cx /= wsum; cy /= wsum; cz /= wsum;
    if (op.brush === "pinch") {
      for (const v of active) { const w = weight[v]! * op.amount; disp[v * 3] = (cx - pos[v * 3]!) * w; disp[v * 3 + 1] = (cy - pos[v * 3 + 1]!) * w; disp[v * 3 + 2] = (cz - pos[v * 3 + 2]!) * w; }
    } else {
      const nrm = computeSmoothNormals(pos, frame.indices);
      let nx = 0, ny = 0, nz = 0;
      for (const v of active) { const w = weight[v]!; nx += nrm[v * 3]! * w; ny += nrm[v * 3 + 1]! * w; nz += nrm[v * 3 + 2]! * w; }
      const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
      for (const v of active) {
        const w = weight[v]! * op.amount;
        const h = (pos[v * 3]! - cx) * nx + (pos[v * 3 + 1]! - cy) * ny + (pos[v * 3 + 2]! - cz) * nz;   // height above the plane
        disp[v * 3] = -h * nx * w; disp[v * 3 + 1] = -h * ny * w; disp[v * 3 + 2] = -h * nz * w;
      }
    }
  } else { // smooth: umbrella Laplacian on the welded graph, `iterations` passes, blended by weight×amount
    const adj = new Map<number, Set<number>>();
    const idx = frame.indices;
    const link = (a: number, b: number) => { let s = adj.get(a); if (!s) { s = new Set(); adj.set(a, s); } s.add(b); };
    const activeSet = new Set(active);
    for (let t = 0; t < idx.length; t += 3) {
      const a = canon[idx[t]!]!, b = canon[idx[t + 1]!]!, c = canon[idx[t + 2]!]!;
      if (activeSet.has(a) || activeSet.has(b) || activeSet.has(c)) { link(a, b); link(a, c); link(b, a); link(b, c); link(c, a); link(c, b); }
    }
    const cur = new Float32Array(pos);          // working copy of canonical positions
    for (let it = 0; it < op.iterations; it++) {
      const next = new Float32Array(cur);
      for (const v of active) {
        const nb = adj.get(v);
        if (!nb || !nb.size) continue;
        let ax = 0, ay = 0, az = 0;
        for (const u of nb) { ax += cur[u * 3]!; ay += cur[u * 3 + 1]!; az += cur[u * 3 + 2]!; }
        const inv = 1 / nb.size, lambda = 0.5;   // umbrella Laplacian, λ = 0.5 per pass
        next[v * 3] = cur[v * 3]! + (ax * inv - cur[v * 3]!) * lambda;
        next[v * 3 + 1] = cur[v * 3 + 1]! + (ay * inv - cur[v * 3 + 1]!) * lambda;
        next[v * 3 + 2] = cur[v * 3 + 2]! + (az * inv - cur[v * 3 + 2]!) * lambda;
      }
      cur.set(next);
    }
    for (const v of active) { const w = weight[v]! * Math.max(0, Math.min(1, op.amount)); disp[v * 3] = (cur[v * 3]! - pos[v * 3]!) * w; disp[v * 3 + 1] = (cur[v * 3 + 1]! - pos[v * 3 + 1]!) * w; disp[v * 3 + 2] = (cur[v * 3 + 2]! - pos[v * 3 + 2]!) * w; }
  }
  // Apply to every vertex through its welded canonical copy — seams stay closed.
  let moved = 0;
  for (let v = 0; v < n; v++) {
    const c = canon[v]!;
    const dx = disp[c * 3]!, dy = disp[c * 3 + 1]!, dz = disp[c * 3 + 2]!;
    if (dx === 0 && dy === 0 && dz === 0) continue;
    pos[v * 3] = pos[v * 3]! + dx; pos[v * 3 + 1] = pos[v * 3 + 1]! + dy; pos[v * 3 + 2] = pos[v * 3 + 2]! + dz;
    moved++;
  }
  touched = moved;
  return { vertices: touched, groups: active.length, feather };
}
