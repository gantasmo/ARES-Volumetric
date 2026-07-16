/**
 * Temporal geometry preprocessing (spec §6.5, §8.3) — the P2 core bet.
 *
 * Turns a frame sequence into GOPs that share ONE topology, so the container can store
 * indices/UVs once per GOP (I-frame) and only position deltas per frame (P-frames), and so
 * per-vertex trajectories can be temporally SMOOTHED (the fix for per-frame-reconstruction
 * jitter). Two ways a GOP becomes persistent-topology:
 *
 *   1. Stable source (synthetic clips, or captures that keep vertex order): detected directly.
 *   2. Tracked (`--track`): each frame's surface is resampled onto the GOP keyframe's vertex
 *      set by incremental nearest-point following (Microsoft-style volcap re-topologizes every
 *      frame, so this estimates the correspondence the capture doesn't provide).
 *
 * If neither holds and tracking is off, the GOP falls back to independent intra frames (P1).
 */
import { computeAabb, unionAabb, type Aabb } from "@ares/core";
import type { EncodeMeshFrame } from "./geometry-encode.js";

export interface TemporalOptions {
  gopLength: number;
  track: boolean;
  /** trajectory low-pass half-window in frames (0 = off); needs persistent topology */
  smoothTemporal: number;
  /** Taubin spatial-smoothing iterations per frame (0 = off) */
  smoothSpatial: number;
  /** Force every GOP to the independent-intra path even when topology is stable. The multi-run
   *  coherent bake produces MANY stable-topology GOPs with DIFFERENT topologies between them; the
   *  temporal (I+P) container path shredded geometry there (single-GOP coherent-A was fine, the
   *  10-GOP full clip was not). All-intra is the proven-safe path (daniel-s0 encodes this way). */
  forceIntra?: boolean;
}

export interface TemporalGop {
  gopBox: Aabb;
  frameStart: number;
  temporal: boolean;
  /** temporal: reference topology + per-frame aligned positions (idx 0 = I-frame) */
  uvs?: Float32Array;
  indices?: Uint32Array;
  framePositions?: Float32Array[];
  /** per-frame UVs (tracked GOPs only — the re-atlased texture needs frame-local UVs, spec §7.4) */
  frameUvs?: Float32Array[];
  /** intra fallback: independent per-frame meshes */
  frames?: EncodeMeshFrame[];
  /** mean per-vertex tracking residual over the GOP, relative to bbox diagonal (diagnostics) */
  trackError?: number;
}

/* ------------------- spatial hash grid over triangles ------------------- */

/** Closest point on triangle abc to p (Ericson, Real-Time Collision Detection). */
function closestOnTriangle(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number,
  out: Float32Array, o: number,
): number {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let rx: number, ry: number, rz: number;
  if (d1 <= 0 && d2 <= 0) { rx = ax; ry = ay; rz = az; }
  else {
    const bpx = px - bx, bpy = py - by, bpz = pz - bz;
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) { rx = bx; ry = by; rz = bz; }
    else {
      const vc = d1 * d4 - d3 * d2;
      if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); rx = ax + v * abx; ry = ay + v * aby; rz = az + v * abz; }
      else {
        const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
        const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
        if (d6 >= 0 && d5 <= d6) { rx = cx; ry = cy; rz = cz; }
        else {
          const vb = d5 * d2 - d1 * d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); rx = ax + w * acx; ry = ay + w * acy; rz = az + w * acz; }
          else {
            const va = d3 * d6 - d5 * d4;
            if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / (d4 - d3 + (d5 - d6)); rx = bx + w * (cx - bx); ry = by + w * (cy - by); rz = bz + w * (cz - bz); }
            else { const denom = 1 / (va + vb + vc); const v = vb * denom, w = vc * denom; rx = ax + abx * v + acx * w; ry = ay + aby * v + acy * w; rz = az + abz * v + acz * w; }
          }
        }
      }
    }
  }
  out[o] = rx; out[o + 1] = ry; out[o + 2] = rz;
  const dx = rx - px, dy = ry - py, dz = rz - pz;
  return dx * dx + dy * dy + dz * dz;
}

/** Barycentric coords of P w.r.t. triangle ABC (P ≈ u·A + v·B + w·C). */
function barycentric(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number,
): [number, number, number] {
  const v0x = bx - ax, v0y = by - ay, v0z = bz - az;
  const v1x = cx - ax, v1y = cy - ay, v1z = cz - az;
  const v2x = px - ax, v2y = py - ay, v2z = pz - az;
  const d00 = v0x * v0x + v0y * v0y + v0z * v0z;
  const d01 = v0x * v1x + v0y * v1y + v0z * v1z;
  const d11 = v1x * v1x + v1y * v1y + v1z * v1z;
  const d20 = v2x * v0x + v2y * v0y + v2z * v0z;
  const d21 = v2x * v1x + v2y * v1y + v2z * v1z;
  const denom = d00 * d11 - d01 * d01 || 1;
  const v = (d11 * d20 - d01 * d21) / denom;
  const w = (d00 * d21 - d01 * d20) / denom;
  return [1 - v - w, v, w];
}

class TriangleGrid {
  private cells = new Map<number, number[]>();
  private inv: number;
  private minx = Infinity; private miny = Infinity; private minz = Infinity;
  constructor(private pos: Float32Array, private idx: Uint32Array, private uvs: Float32Array | undefined, cell: number) {
    this.inv = 1 / cell;
    for (let i = 0; i < pos.length; i += 3) {
      if (pos[i]! < this.minx) this.minx = pos[i]!;
      if (pos[i + 1]! < this.miny) this.miny = pos[i + 1]!;
      if (pos[i + 2]! < this.minz) this.minz = pos[i + 2]!;
    }
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t]!, b = idx[t + 1]!, c = idx[t + 2]!;
      const cxv = (pos[a * 3]! + pos[b * 3]! + pos[c * 3]!) / 3;
      const cyv = (pos[a * 3 + 1]! + pos[b * 3 + 1]! + pos[c * 3 + 1]!) / 3;
      const czv = (pos[a * 3 + 2]! + pos[b * 3 + 2]! + pos[c * 3 + 2]!) / 3;
      const key = this.key(this.ci(cxv, this.minx), this.ci(cyv, this.miny), this.ci(czv, this.minz));
      let bucket = this.cells.get(key);
      if (!bucket) { bucket = []; this.cells.set(key, bucket); }
      bucket.push(t);
    }
  }
  private ci(v: number, min: number) { return Math.floor((v - min) * this.inv); }
  private key(cx: number, cy: number, cz: number) {
    return (Math.imul(cx, 73856093) ^ Math.imul(cy, 19349663) ^ Math.imul(cz, 83492791)) >>> 0;
  }
  /**
   * Closest point on the mesh surface to (x,y,z); writes it to out[o..o+2] and (if this frame
   * has UVs and uvOut is given) the barycentric-interpolated UV to uvOut[uo..uo+1]. This is what
   * keeps the texture correct under tracking: each tracked vertex gets THIS frame's UV, matching
   * this frame's atlas (Microsoft-style volcap re-atlases every frame, spec §7.4). Returns sq dist.
   */
  nearestInto(x: number, y: number, z: number, out: Float32Array, o: number, uvOut?: Float32Array, uo = 0): number {
    const bx = this.ci(x, this.minx), by = this.ci(y, this.miny), bz = this.ci(z, this.minz);
    let bestD = Infinity, bestT = -1;
    const tmp = new Float32Array(3);
    let found = false;
    for (let r = 0; r <= 8; r++) {
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
        if (r > 0 && Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
        const bucket = this.cells.get(this.key(bx + dx, by + dy, bz + dz));
        if (!bucket) continue;
        for (const t of bucket) {
          const a = this.idx[t]!, b = this.idx[t + 1]!, c = this.idx[t + 2]!;
          const d = closestOnTriangle(x, y, z,
            this.pos[a * 3]!, this.pos[a * 3 + 1]!, this.pos[a * 3 + 2]!,
            this.pos[b * 3]!, this.pos[b * 3 + 1]!, this.pos[b * 3 + 2]!,
            this.pos[c * 3]!, this.pos[c * 3 + 1]!, this.pos[c * 3 + 2]!, tmp, 0);
          if (d < bestD) { bestD = d; bestT = t; out[o] = tmp[0]!; out[o + 1] = tmp[1]!; out[o + 2] = tmp[2]!; found = true; }
        }
      }
      if (found && r >= 2) break; // one extra safety ring past the first hit
    }
    if (!found) { out[o] = x; out[o + 1] = y; out[o + 2] = z; if (uvOut) { uvOut[uo] = 0; uvOut[uo + 1] = 0; } return bestD; }
    if (uvOut && this.uvs) {
      const a = this.idx[bestT]!, b = this.idx[bestT + 1]!, c = this.idx[bestT + 2]!;
      const [u, v, w] = barycentric(out[o]!, out[o + 1]!, out[o + 2]!,
        this.pos[a * 3]!, this.pos[a * 3 + 1]!, this.pos[a * 3 + 2]!,
        this.pos[b * 3]!, this.pos[b * 3 + 1]!, this.pos[b * 3 + 2]!,
        this.pos[c * 3]!, this.pos[c * 3 + 1]!, this.pos[c * 3 + 2]!);
      uvOut[uo] = u * this.uvs[a * 2]! + v * this.uvs[b * 2]! + w * this.uvs[c * 2]!;
      uvOut[uo + 1] = u * this.uvs[a * 2 + 1]! + v * this.uvs[b * 2 + 1]! + w * this.uvs[c * 2 + 1]!;
    }
    return bestD;
  }
}

/* ------------------------------ smoothing ------------------------------- */

interface Adjacency { offsets: Int32Array; neighbors: Int32Array; }

function buildAdjacency(vertexCount: number, indices: Uint32Array): Adjacency {
  const sets: Set<number>[] = Array.from({ length: vertexCount }, () => new Set<number>());
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i]!, b = indices[i + 1]!, c = indices[i + 2]!;
    sets[a]!.add(b); sets[a]!.add(c); sets[b]!.add(a); sets[b]!.add(c); sets[c]!.add(a); sets[c]!.add(b);
  }
  const offsets = new Int32Array(vertexCount + 1);
  for (let v = 0; v < vertexCount; v++) offsets[v + 1] = offsets[v]! + sets[v]!.size;
  const neighbors = new Int32Array(offsets[vertexCount]!);
  let p = 0;
  for (let v = 0; v < vertexCount; v++) for (const n of sets[v]!) neighbors[p++] = n;
  return { offsets, neighbors };
}

/** Taubin λ|μ smoothing: spatial denoise without the shrinkage of plain Laplacian. */
function taubin(pos: Float32Array, adj: Adjacency, iterations: number, lambda = 0.5, mu = -0.53): void {
  const n = pos.length / 3;
  const tmp = new Float32Array(pos.length);
  const step = (factor: number) => {
    for (let v = 0; v < n; v++) {
      const s = adj.offsets[v]!, e = adj.offsets[v + 1]!;
      const deg = e - s;
      if (deg === 0) { tmp[v * 3] = pos[v * 3]!; tmp[v * 3 + 1] = pos[v * 3 + 1]!; tmp[v * 3 + 2] = pos[v * 3 + 2]!; continue; }
      let ax = 0, ay = 0, az = 0;
      for (let j = s; j < e; j++) { const nb = adj.neighbors[j]!; ax += pos[nb * 3]!; ay += pos[nb * 3 + 1]!; az += pos[nb * 3 + 2]!; }
      ax /= deg; ay /= deg; az /= deg;
      tmp[v * 3] = pos[v * 3]! + factor * (ax - pos[v * 3]!);
      tmp[v * 3 + 1] = pos[v * 3 + 1]! + factor * (ay - pos[v * 3 + 1]!);
      tmp[v * 3 + 2] = pos[v * 3 + 2]! + factor * (az - pos[v * 3 + 2]!);
    }
    pos.set(tmp);
  };
  for (let i = 0; i < iterations; i++) { step(lambda); step(mu); }
}

/**
 * Weld-aware Taubin. Atlased meshes duplicate vertices along UV-chart seams (same position,
 * different UV). Index-based adjacency gives each copy only ITS side's neighbors, so plain Taubin
 * moves the copies apart and the mesh visibly cracks along chart boundaries (per-frame "cracked
 * egg" lines). Welding by exact position bits makes all copies smooth as ONE vertex — cracks are
 * impossible by construction — and stitches adjacency across the seam.
 */
function taubinWelded(pos: Float32Array, indices: Uint32Array, iterations: number): void {
  const n = pos.length / 3;
  const bits = new Uint32Array(pos.buffer, pos.byteOffset, pos.length);
  const canon = new Int32Array(n);          // vertex -> first vertex with the same position
  const compact = new Int32Array(n);        // canonical vertex -> compact id
  const map = new Map<string, number>();
  let nc = 0;
  for (let v = 0; v < n; v++) {
    const key = bits[v * 3]! + "," + bits[v * 3 + 1]! + "," + bits[v * 3 + 2]!;
    const c = map.get(key);
    if (c === undefined) { map.set(key, v); canon[v] = v; compact[v] = nc++; }
    else canon[v] = c;
  }
  const cpos = new Float32Array(nc * 3);
  for (let v = 0; v < n; v++) {
    if (canon[v] !== v) continue;
    const ci = compact[v]!;
    cpos[ci * 3] = pos[v * 3]!; cpos[ci * 3 + 1] = pos[v * 3 + 1]!; cpos[ci * 3 + 2] = pos[v * 3 + 2]!;
  }
  const cidx = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) cidx[i] = compact[canon[indices[i]!]!]!;
  taubin(cpos, buildAdjacency(nc, cidx), iterations);
  for (let v = 0; v < n; v++) {
    const ci = compact[canon[v]!]!;
    pos[v * 3] = cpos[ci * 3]!; pos[v * 3 + 1] = cpos[ci * 3 + 1]!; pos[v * 3 + 2] = cpos[ci * 3 + 2]!;
  }
}

/** Temporal low-pass on each vertex's trajectory (needs persistent topology). Fixes jitter. */
function smoothTemporal(framePositions: Float32Array[], halfWindow: number): void {
  const F = framePositions.length;
  if (F < 3 || halfWindow < 1) return;
  const len = framePositions[0]!.length;
  const src = framePositions.map((f) => f.slice());
  for (let f = 0; f < F; f++) {
    const out = framePositions[f]!;
    const lo = Math.max(0, f - halfWindow), hi = Math.min(F - 1, f + halfWindow);
    const inv = 1 / (hi - lo + 1);
    for (let i = 0; i < len; i++) {
      let s = 0;
      for (let g = lo; g <= hi; g++) s += src[g]![i]!;
      out[i] = s * inv;
    }
  }
}

/* --------------------------- GOP construction --------------------------- */

function sameTopology(a: EncodeMeshFrame, b: EncodeMeshFrame): boolean {
  if (a.positions.length !== b.positions.length || a.indices.length !== b.indices.length) return false;
  // spot-check indices (identical-source clips share the exact index buffer)
  const stride = Math.max(1, (a.indices.length / 64) | 0);
  for (let i = 0; i < a.indices.length; i += stride) if (a.indices[i] !== b.indices[i]) return false;
  return true;
}

export function buildTemporalGops(frames: EncodeMeshFrame[], opts: TemporalOptions): TemporalGop[] {
  const out: TemporalGop[] = [];
  const boxes = frames.map((f) => computeAabb(f.positions));
  // GOP segmentation follows TOPOLOGY RUNS, not fixed strides (2026-07-13). With fixed 30-frame
  // chunks, a topology reset landing mid-chunk sent the WHOLE chunk intra even when 25+ of its
  // frames shared topology — measured on the coherent-GOP full-clip bake: only 32/272 frames
  // survived as P despite ~200 frames sitting in stable runs (chunk boundaries never lined up
  // with the runner's adaptive cuts). Rules: a maximal same-topology run (capped at gopLength)
  // is one GOP; consecutive single-frame runs (per-frame-repacking captures — raw Daniel,
  // 4DViews) merge into intra batches capped at gopLength, so those sources keep the exact
  // chunk shape they had under fixed strides (e.g. 272 frames → 10 chunks, all-intra).
  const runs: { start: number; end: number }[] = [];
  for (let s = 0; s < frames.length; ) {
    let e = s + 1;
    while (e < frames.length && e - s < opts.gopLength && sameTopology(frames[s]!, frames[e]!)) e++;
    runs.push({ start: s, end: e });
    s = e;
  }
  const segs: { start: number; end: number; intraBatch: boolean }[] = [];
  for (const r of runs) {
    const single = r.end - r.start === 1;
    const prev = segs[segs.length - 1];
    if (single && prev?.intraBatch && prev.end - prev.start < opts.gopLength) { prev.end = r.end; continue; }
    segs.push({ start: r.start, end: r.end, intraBatch: single });
  }
  for (const seg of segs) {
    const start = seg.start;
    const end = seg.end;
    let gopBox = boxes[start]!;
    for (let f = start + 1; f < end; f++) gopBox = unionAabb(gopBox, boxes[f]!);
    const ref = frames[start]!;
    const member = frames.slice(start, end);

    const stable = member.every((f) => sameTopology(ref, f));
    const usePersistent = (stable || opts.track) && !opts.forceIntra;

    if (!usePersistent) {
      // Intra path (per-frame-independent topology). Correspondence-free jitter reduction: apply
      // per-frame WELD-AWARE Taubin smoothing (each frame has its own topology → its own weld+adjacency).
      // Non-shrinking (λ=0.5, μ=-0.53); also smooths the derived normals, softening faint facets.
      let m = member;
      if (opts.smoothSpatial > 0) {
        m = member.map((f) => {
          const p = f.positions.slice();
          taubinWelded(p, f.indices, opts.smoothSpatial);
          return { ...f, positions: p };
        });
      }
      out.push({ gopBox, frameStart: start, temporal: false, frames: m });
      continue;
    }

    let framePositions: Float32Array[];
    let frameUvs: Float32Array[] | undefined;
    let trackError = 0;
    if (stable) {
      framePositions = member.map((f) => f.positions.slice());
      // topology stable → UVs constant across the GOP; stored once on the I-frame.
    } else {
      // incremental nearest-point tracking onto the reference vertex set, transferring THIS
      // frame's UVs onto the tracked vertices so the per-frame atlas still maps correctly.
      const diag = Math.hypot(gopBox.max[0] - gopBox.min[0], gopBox.max[1] - gopBox.min[1], gopBox.max[2] - gopBox.min[2]) || 1;
      const cell = diag / 48;
      const nUv = (ref.uvs?.length ?? 0) / (ref.positions.length / 3) === 2;
      framePositions = [ref.positions.slice()];
      frameUvs = ref.uvs ? [ref.uvs.slice()] : undefined;
      let prev = ref.positions;
      let errAcc = 0, errN = 0;
      for (let f = 1; f < member.length; f++) {
        const grid = new TriangleGrid(member[f]!.positions, member[f]!.indices, nUv ? member[f]!.uvs : undefined, cell);
        const tracked = new Float32Array(prev.length);
        const uv = frameUvs ? new Float32Array((prev.length / 3) * 2) : undefined;
        for (let k = 0, ku = 0; k < prev.length; k += 3, ku += 2) {
          const d2 = grid.nearestInto(prev[k]!, prev[k + 1]!, prev[k + 2]!, tracked, k, uv, ku);
          errAcc += Math.sqrt(d2); errN++;
        }
        framePositions.push(tracked);
        if (frameUvs && uv) frameUvs.push(uv);
        prev = tracked;
      }
      trackError = errN ? (errAcc / errN) / diag : 0;
    }

    if (opts.smoothSpatial > 0) {
      const adj = buildAdjacency(ref.positions.length / 3, ref.indices);
      for (const fp of framePositions) taubin(fp, adj, opts.smoothSpatial);
    }
    if (opts.smoothTemporal > 0) smoothTemporal(framePositions, opts.smoothTemporal);

    out.push({ gopBox, frameStart: start, temporal: true, uvs: ref.uvs, indices: ref.indices, framePositions, frameUvs, trackError });
  }
  return out;
}
