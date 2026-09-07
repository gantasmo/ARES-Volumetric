/**
 * Task K POC — shared library. Temporally-coherent GOP: one topology + one atlas
 * layout (the "template", = frame 0 verbatim) deformed across a 30-frame GOP, with
 * per-frame texture re-baked into the FIXED template UVs (fixes the per-frame-UV
 * seam-artifact failure mode of the tracked path in packages/encoder/src/temporal.ts).
 *
 * Adapted copies (not exports) of packages/encoder/src/temporal.ts's TriangleGrid /
 * weld-aware Taubin machinery, per task instructions — this is a standalone POC
 * script, not a change to the shipped encoder.
 */
import { execFile } from "node:child_process";
import { readFile, writeFile, rm, mkdtemp } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);

export function ffmpegPath() {
  // Same discovery order as the encoder (packages/encoder/src/texture-video.ts): env, the usual
  // install locations, then PATH — instead of one developer's Windows path as the default.
  const env = process.env.FFMPEG || process.env.FFMPEG_PATH;
  if (env && existsSync(env)) return env;
  for (const c of ["C:\FFmpeg\bin\ffmpeg.exe", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"]) if (existsSync(c)) return c;
  return "ffmpeg";
}

/* --------------------------- closest-point-on-mesh --------------------------- */

/** Closest point on triangle abc to p (Ericson, Real-Time Collision Detection). */
export function closestOnTriangle(
  px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, out, o,
) {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let rx, ry, rz;
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

export function barycentric3(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz) {
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

/** Spatial-hash closest-point-on-surface (adapted from packages/encoder/src/temporal.ts). */
export class TriangleGrid {
  constructor(pos, idx, uvs, cell) {
    this.pos = pos; this.idx = idx; this.uvs = uvs;
    this.cells = new Map();
    this.inv = 1 / cell;
    this.minx = Infinity; this.miny = Infinity; this.minz = Infinity;
    for (let i = 0; i < pos.length; i += 3) {
      if (pos[i] < this.minx) this.minx = pos[i];
      if (pos[i + 1] < this.miny) this.miny = pos[i + 1];
      if (pos[i + 2] < this.minz) this.minz = pos[i + 2];
    }
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t], b = idx[t + 1], c = idx[t + 2];
      const cxv = (pos[a * 3] + pos[b * 3] + pos[c * 3]) / 3;
      const cyv = (pos[a * 3 + 1] + pos[b * 3 + 1] + pos[c * 3 + 1]) / 3;
      const czv = (pos[a * 3 + 2] + pos[b * 3 + 2] + pos[c * 3 + 2]) / 3;
      const key = this.key(this.ci(cxv, this.minx), this.ci(cyv, this.miny), this.ci(czv, this.minz));
      let bucket = this.cells.get(key);
      if (!bucket) { bucket = []; this.cells.set(key, bucket); }
      bucket.push(t);
    }
  }
  ci(v, min) { return Math.floor((v - min) * this.inv); }
  key(cx, cy, cz) {
    return (Math.imul(cx, 73856093) ^ Math.imul(cy, 19349663) ^ Math.imul(cz, 83492791)) >>> 0;
  }
  /** Closest point on the mesh surface to (x,y,z). Writes it to out[o..o+2] and (if uvOut given
   *  and this grid has UVs) the barycentric-interpolated UV to uvOut[uo..uo+1]. Returns sq dist.
   *  If triOut is given, writes the winning triangle's flat index (multiple of 3, -1 if none
   *  found) to triOut[to] — Phase 2 perf work uses this to seed local per-texel search (see
   *  buildLocalCandidates/nearestLocalScan below) from registration's per-vertex results. */
  nearestInto(x, y, z, out, o, uvOut, uo = 0, triOut, to = 0) {
    const bx = this.ci(x, this.minx), by = this.ci(y, this.miny), bz = this.ci(z, this.minz);
    let bestD = Infinity, bestT = -1;
    const tmp = TriangleGrid._tmp;
    let found = false;
    for (let r = 0; r <= 8; r++) {
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
        if (r > 0 && Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
        const bucket = this.cells.get(this.key(bx + dx, by + dy, bz + dz));
        if (!bucket) continue;
        for (const t of bucket) {
          const a = this.idx[t], b = this.idx[t + 1], c = this.idx[t + 2];
          const d = closestOnTriangle(x, y, z,
            this.pos[a * 3], this.pos[a * 3 + 1], this.pos[a * 3 + 2],
            this.pos[b * 3], this.pos[b * 3 + 1], this.pos[b * 3 + 2],
            this.pos[c * 3], this.pos[c * 3 + 1], this.pos[c * 3 + 2], tmp, 0);
          if (d < bestD) { bestD = d; bestT = t; out[o] = tmp[0]; out[o + 1] = tmp[1]; out[o + 2] = tmp[2]; found = true; }
        }
      }
      if (found && r >= 2) break;
    }
    if (triOut) triOut[to] = bestT;
    if (!found) { out[o] = x; out[o + 1] = y; out[o + 2] = z; if (uvOut) { uvOut[uo] = 0; uvOut[uo + 1] = 0; } return bestD; }
    if (uvOut && this.uvs) {
      const a = this.idx[bestT], b = this.idx[bestT + 1], c = this.idx[bestT + 2];
      const [u, v, w] = barycentric3(out[o], out[o + 1], out[o + 2],
        this.pos[a * 3], this.pos[a * 3 + 1], this.pos[a * 3 + 2],
        this.pos[b * 3], this.pos[b * 3 + 1], this.pos[b * 3 + 2],
        this.pos[c * 3], this.pos[c * 3 + 1], this.pos[c * 3 + 2]);
      uvOut[uo] = u * this.uvs[a * 2] + v * this.uvs[b * 2] + w * this.uvs[c * 2];
      uvOut[uo + 1] = u * this.uvs[a * 2 + 1] + v * this.uvs[b * 2 + 1] + w * this.uvs[c * 2 + 1];
    }
    return bestD;
  }
}
TriangleGrid._tmp = new Float32Array(3);

/* --------------------- Phase 2 perf: seeded local texel search --------------------- */
//
// Root cause (measured in the POC report): TriangleGrid's cell size is tuned for
// registration-scale queries (~11k/frame); at texel-bake scale (~2.5M/frame, ~220x more) its
// ~0.18 tri/cell density means every query expands to r>=2 (125 cells, mostly empty Map
// lookups) before the "found && r>=2" safety ring is satisfied. The fix is NOT a better global
// structure — it's to skip the global search almost entirely: registration already computed,
// for every TEMPLATE vertex, its nearest SOURCE triangle this frame (nearestInto's new triOut).
// A template triangle's texels are known (POC measurement) to land within ~0.15mm of the
// source surface, so the true nearest triangle is virtually always within 1-2 adjacency rings
// of the seed triangles of that template triangle's 3 corners. We build that small candidate
// set ONCE PER TEMPLATE TRIANGLE PER FRAME (20,000x, not 2.5Mx) and linear-scan it per texel;
// texels whose local best distance is suspiciously large fall back to the exact global
// TriangleGrid search (safety net, expected rare — see fallbackCount in bakeFrameFast's return).

/** CSR triangle-adjacency (edge-shared neighbors) of a mesh's index buffer. Triangle ids are
 *  flat indices (multiples of 3), matching TriangleGrid.nearestInto's bestT / triId convention. */
export function buildTriAdjacency(indices) {
  const numTris = indices.length / 3;
  const M = 1 << 20; // supports up to ~1M vertices in a safe-integer edge key
  const edgeOwner = new Map(); // edgeKey -> { t, slot }
  const nbrA = new Int32Array(numTris).fill(-1);
  const nbrB = new Int32Array(numTris).fill(-1);
  const nbrC = new Int32Array(numTris).fill(-1);
  const nbrSlots = [nbrA, nbrB, nbrC];
  for (let t = 0; t < indices.length; t += 3) {
    const ti = t / 3;
    const verts = [indices[t], indices[t + 1], indices[t + 2]];
    for (let slot = 0; slot < 3; slot++) {
      const v0 = verts[slot], v1 = verts[(slot + 1) % 3];
      const key = v0 < v1 ? v0 * M + v1 : v1 * M + v0;
      const prev = edgeOwner.get(key);
      if (prev === undefined) edgeOwner.set(key, { t, slot });
      else { nbrSlots[slot][ti] = prev.t; nbrSlots[prev.slot][prev.t / 3] = t; }
    }
  }
  const offsets = new Int32Array(numTris + 1);
  for (let i = 0; i < numTris; i++) {
    let cnt = 0;
    if (nbrA[i] >= 0) cnt++; if (nbrB[i] >= 0) cnt++; if (nbrC[i] >= 0) cnt++;
    offsets[i + 1] = offsets[i] + cnt;
  }
  const neighbors = new Int32Array(offsets[numTris]);
  let p = 0;
  for (let i = 0; i < numTris; i++) {
    if (nbrA[i] >= 0) neighbors[p++] = nbrA[i];
    if (nbrB[i] >= 0) neighbors[p++] = nbrB[i];
    if (nbrC[i] >= 0) neighbors[p++] = nbrC[i];
  }
  return { numTris, offsets, neighbors };
}

/** For each TEMPLATE triangle, the small set of SOURCE triangles to try first: the 3 corner
 *  vertices' registration-time seed triangles (vertexSeedTri, template-vertex-indexed, values
 *  are SOURCE flat triangle indices this frame), expanded `rings` hops via the source mesh's
 *  own adjacency. CSR-encoded, keyed by template triangle index (t/3). */
export function buildLocalCandidates(templateIndices, vertexSeedTri, srcAdj, rings = 2) {
  const numTemplateTris = templateIndices.length / 3;
  const offsets = new Uint32Array(numTemplateTris + 1);
  const perTri = new Array(numTemplateTris);
  const stampArr = new Int32Array(srcAdj.numTris).fill(-1);
  const queueArr = new Int32Array(srcAdj.numTris);
  let stamp = 0;
  for (let ti = 0; ti < numTemplateTris; ti++) {
    stamp++;
    let qTail = 0;
    const a = templateIndices[ti * 3], b = templateIndices[ti * 3 + 1], c = templateIndices[ti * 3 + 2];
    for (const v of [a, b, c]) {
      const seedFlat = vertexSeedTri ? vertexSeedTri[v] : -1;
      if (seedFlat === undefined || seedFlat < 0) continue;
      const seedIdx = (seedFlat / 3) | 0;
      if (stampArr[seedIdx] !== stamp) { stampArr[seedIdx] = stamp; queueArr[qTail++] = seedIdx; }
    }
    let ringStart = 0, ringEnd = qTail;
    for (let r = 0; r < rings; r++) {
      for (let qi = ringStart; qi < ringEnd; qi++) {
        const triIdx = queueArr[qi];
        const s = srcAdj.offsets[triIdx], e = srcAdj.offsets[triIdx + 1];
        for (let k = s; k < e; k++) {
          const nbIdx = (srcAdj.neighbors[k] / 3) | 0;
          if (stampArr[nbIdx] !== stamp) { stampArr[nbIdx] = stamp; queueArr[qTail++] = nbIdx; }
        }
      }
      ringStart = ringEnd; ringEnd = qTail;
    }
    const list = new Int32Array(qTail);
    for (let i = 0; i < qTail; i++) list[i] = queueArr[i] * 3;
    perTri[ti] = list;
    offsets[ti + 1] = offsets[ti] + qTail;
  }
  const candidates = new Int32Array(offsets[numTemplateTris]);
  let p = 0;
  for (let ti = 0; ti < numTemplateTris; ti++) { candidates.set(perTri[ti], p); p += perTri[ti].length; }
  return { offsets, candidates };
}

/** Linear-scan closest point over a bounded candidate list (from buildLocalCandidates) for one
 *  template triangle index `ti`, instead of TriangleGrid's cold expanding-ring cell search.
 *  Same semantics as TriangleGrid.nearestInto (writes point + UV), scoped to `candOffsets`/
 *  `candArray`. Returns Infinity (not found) if the candidate list is empty — caller decides
 *  whether to fall back to the exact global search. */
export function nearestLocalScan(candOffsets, candArray, ti, srcPositions, srcIndices, srcUvs, x, y, z, out, o, uvOut, uo = 0) {
  const s = candOffsets[ti], e = candOffsets[ti + 1];
  let bestD = Infinity, bestT = -1;
  const tmp = TriangleGrid._tmp;
  for (let k = s; k < e; k++) {
    const t = candArray[k];
    const a = srcIndices[t], b = srcIndices[t + 1], c = srcIndices[t + 2];
    const d = closestOnTriangle(x, y, z,
      srcPositions[a * 3], srcPositions[a * 3 + 1], srcPositions[a * 3 + 2],
      srcPositions[b * 3], srcPositions[b * 3 + 1], srcPositions[b * 3 + 2],
      srcPositions[c * 3], srcPositions[c * 3 + 1], srcPositions[c * 3 + 2], tmp, 0);
    if (d < bestD) { bestD = d; bestT = t; out[o] = tmp[0]; out[o + 1] = tmp[1]; out[o + 2] = tmp[2]; }
  }
  if (bestT < 0) return Infinity;
  if (uvOut && srcUvs) {
    const a = srcIndices[bestT], b = srcIndices[bestT + 1], c = srcIndices[bestT + 2];
    const [u, v, w] = barycentric3(out[o], out[o + 1], out[o + 2],
      srcPositions[a * 3], srcPositions[a * 3 + 1], srcPositions[a * 3 + 2],
      srcPositions[b * 3], srcPositions[b * 3 + 1], srcPositions[b * 3 + 2],
      srcPositions[c * 3], srcPositions[c * 3 + 1], srcPositions[c * 3 + 2]);
    uvOut[uo] = u * srcUvs[a * 2] + v * srcUvs[b * 2] + w * srcUvs[c * 2];
    uvOut[uo + 1] = u * srcUvs[a * 2 + 1] + v * srcUvs[b * 2 + 1] + w * srcUvs[c * 2 + 1];
  }
  return bestD;
}

/* ------------------------ weld-aware displacement smoothing ------------------------ */

/** Build (once, from the TEMPLATE's rest positions+indices) a weld map + adjacency on the
 *  compacted (weld-collapsed) graph. Reused every frame/round to smooth whatever per-vertex
 *  vector field is passed in (raw target-pull displacements), so UV-seam duplicate pairs
 *  always receive identical smoothed displacement (never crack). */
export function buildWeldAdjacency(positions, indices) {
  const n = positions.length / 3;
  const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
  const canon = new Int32Array(n);
  const compact = new Int32Array(n);
  const map = new Map();
  let nc = 0;
  for (let v = 0; v < n; v++) {
    const key = bits[v * 3] + "," + bits[v * 3 + 1] + "," + bits[v * 3 + 2];
    const c = map.get(key);
    if (c === undefined) { map.set(key, v); canon[v] = v; compact[v] = nc++; }
    else canon[v] = c;
  }
  const cidx = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) cidx[i] = compact[canon[indices[i]]];
  const sets = Array.from({ length: nc }, () => new Set());
  for (let i = 0; i < cidx.length; i += 3) {
    const a = cidx[i], b = cidx[i + 1], c = cidx[i + 2];
    sets[a].add(b); sets[a].add(c); sets[b].add(a); sets[b].add(c); sets[c].add(a); sets[c].add(b);
  }
  const offsets = new Int32Array(nc + 1);
  for (let v = 0; v < nc; v++) offsets[v + 1] = offsets[v] + sets[v].size;
  const neighbors = new Int32Array(offsets[nc]);
  let p = 0;
  for (let v = 0; v < nc; v++) for (const nb of sets[v]) neighbors[p++] = nb;
  return { n, nc, canon, compact, offsets, neighbors };
}

function taubinStep(vals, offsets, neighbors, nc, factor, tmp) {
  for (let v = 0; v < nc; v++) {
    const s = offsets[v], e = offsets[v + 1];
    const deg = e - s;
    if (deg === 0) { tmp[v * 3] = vals[v * 3]; tmp[v * 3 + 1] = vals[v * 3 + 1]; tmp[v * 3 + 2] = vals[v * 3 + 2]; continue; }
    let ax = 0, ay = 0, az = 0;
    for (let j = s; j < e; j++) { const nb = neighbors[j]; ax += vals[nb * 3]; ay += vals[nb * 3 + 1]; az += vals[nb * 3 + 2]; }
    ax /= deg; ay /= deg; az /= deg;
    tmp[v * 3] = vals[v * 3] + factor * (ax - vals[v * 3]);
    tmp[v * 3 + 1] = vals[v * 3 + 1] + factor * (ay - vals[v * 3 + 1]);
    tmp[v * 3 + 2] = vals[v * 3 + 2] + factor * (az - vals[v * 3 + 2]);
  }
  vals.set(tmp);
}

/** Weld-aware Taubin (lambda|mu, non-shrinking) smoothing of a per-vertex vector field
 *  (displacements), using a weld/adjacency built from the TEMPLATE's rest topology. */
export function smoothFieldWelded(field, weld, iterations, lambda = 0.5, mu = -0.53) {
  const { n, nc, canon, compact, offsets, neighbors } = weld;
  const cval = new Float32Array(nc * 3);
  for (let v = 0; v < n; v++) {
    if (canon[v] !== v) continue;
    const ci = compact[v];
    cval[ci * 3] = field[v * 3]; cval[ci * 3 + 1] = field[v * 3 + 1]; cval[ci * 3 + 2] = field[v * 3 + 2];
  }
  const tmp = new Float32Array(nc * 3);
  for (let i = 0; i < iterations; i++) {
    taubinStep(cval, offsets, neighbors, nc, lambda, tmp);
    taubinStep(cval, offsets, neighbors, nc, mu, tmp);
  }
  const out = new Float32Array(n * 3);
  for (let v = 0; v < n; v++) {
    const ci = compact[canon[v]];
    out[v * 3] = cval[ci * 3]; out[v * 3 + 1] = cval[ci * 3 + 1]; out[v * 3 + 2] = cval[ci * 3 + 2];
  }
  return out;
}

/* --------------------- chain registration (Phase 2: full-clip) --------------------- */

/** Percentile of a pre-sorted array. */
function percentile(sorted, p) {
  const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
  return sorted[idx];
}

/** Registration error of `positions` against `grid`'s source surface: mean/p95/max nearest-dist
 *  (mm, same units as source data) + pct of vertices >5mm. Matches Task K POC's register.mjs. */
export function measureRegError(positions, grid) {
  const n = positions.length / 3;
  const dists = new Float64Array(n);
  const tmp = new Float32Array(3);
  for (let i = 0; i < n; i++) {
    const d2 = grid.nearestInto(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2], tmp, 0);
    dists[i] = Math.sqrt(d2);
  }
  const sorted = Float64Array.from(dists).sort();
  let sum = 0, over5 = 0;
  for (const d of dists) { sum += d; if (d > 5) over5++; }
  return { mean: sum / n, p95: percentile(sorted, 0.95), max: sorted[sorted.length - 1], pctOver5mm: (over5 / n) * 100 };
}

/** One frame of CHAIN-based registration (the POC's winning variant — start from the PREVIOUS
 *  frame's deformed result, not the GOP template — measured 0.148mm flat error vs. growing error
 *  anchored to the template): rounds of {nearestInto each vertex onto the frame's source mesh
 *  -> target; weld-aware Taubin-smooth the DISPLACEMENT field; move}. Each round captures each
 *  vertex's winning SOURCE triangle (vertexSeedTri) for Phase 2's local-search bake seeding (the
 *  final executed round's values persist).
 *
 *  CONVERGENCE (opts): `rounds` is now the MINIMUM round count; pass opts.maxRounds (cap) and
 *  opts.epsMm to keep iterating past `rounds` until the per-round max vertex move falls below
 *  epsMm. This is the fix for the fast-motion spike (measured 2026-07-13: at frame 146, a leg
 *  region moving 50-80 mm/frame left a 73.6 mm off-surface shard after the old fixed 3 rounds —
 *  it simply had not converged. r12 -> 5.8 mm, r16 -> 2.9 mm; adaptive convergence reaches the
 *  same while letting slow frames stop at `rounds`). Displacement GATING was tried and REJECTED:
 *  it treats real articulation as an outlier and made the spike WORSE (73.6 -> 135 mm).
 *
 *  Backward-compatible: with no opts, maxRounds=rounds and epsMm=0 -> exactly `rounds` iterations,
 *  identical to the previous behavior. Returns { positions, vertexSeedTri, error, rounds: used }. */
export function registerChainFrame(basePositions, grid, weld, rounds, smoothIters, opts = {}) {
  const maxRounds = Math.max(rounds, opts.maxRounds ?? rounds);
  const epsMm = opts.epsMm ?? 0;
  let cur = basePositions.slice();
  const n = cur.length / 3;
  const target = new Float32Array(cur.length);
  const vertexSeedTri = new Int32Array(n).fill(-1);
  let used = 0;
  for (let r = 0; r < maxRounds; r++) {
    for (let i = 0; i < n; i++) {
      grid.nearestInto(cur[i * 3], cur[i * 3 + 1], cur[i * 3 + 2], target, i * 3, undefined, 0, vertexSeedTri, i);
    }
    const rawDisp = new Float32Array(cur.length);
    for (let i = 0; i < cur.length; i++) rawDisp[i] = target[i] - cur[i];
    const smoothed = smoothFieldWelded(rawDisp, weld, smoothIters);
    let maxMove = 0;
    for (let i = 0; i < n; i++) {
      const mv = Math.hypot(smoothed[i * 3], smoothed[i * 3 + 1], smoothed[i * 3 + 2]);
      if (mv > maxMove) maxMove = mv;
      cur[i * 3] += smoothed[i * 3]; cur[i * 3 + 1] += smoothed[i * 3 + 1]; cur[i * 3 + 2] += smoothed[i * 3 + 2];
    }
    used++;
    // `rounds` is the floor; only allow early-out once we've done at least that many.
    if (epsMm > 0 && r + 1 >= rounds && maxMove < epsMm) break;
  }
  const error = measureRegError(cur, grid);
  return { positions: cur, vertexSeedTri, error, rounds: used };
}

/* ------------------------------- OBJ I/O -------------------------------- */

export function writeObjText(positions, uvs, indices) {
  const lines = [];
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    lines.push(`v ${positions[i * 3]} ${positions[i * 3 + 1]} ${positions[i * 3 + 2]}`);
  }
  for (let i = 0; i < n; i++) {
    // OBJ importer flips V on read (top-left origin); un-flip on write so re-import round-trips.
    lines.push(`vt ${uvs[i * 2]} ${1 - uvs[i * 2 + 1]}`);
  }
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] + 1, b = indices[t + 1] + 1, c = indices[t + 2] + 1;
    lines.push(`f ${a}/${a} ${b}/${b} ${c}/${c}`);
  }
  return lines.join("\n") + "\n";
}

/* ------------------------------- PNG I/O --------------------------------- */

export function pngSize(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  return { width: dv.getUint32(16, false), height: dv.getUint32(20, false) };
}

export async function decodePng(pngPath, scratchDir) {
  const { width, height } = pngSize(await readFile(pngPath));
  const rawPath = join(scratchDir, `dec-${Math.random().toString(36).slice(2)}.raw`);
  await run(ffmpegPath(), ["-y", "-hide_banner", "-loglevel", "error", "-i", pngPath, "-f", "rawvideo", "-pix_fmt", "rgba", rawPath], { maxBuffer: 1 << 28 });
  const data = new Uint8Array(await readFile(rawPath));
  await rm(rawPath, { force: true });
  return { data, width, height };
}

import { encodePNG as encodePngBytes } from "../../packages/encoder/dist/png.js";
export function encodePng(rgba, width, height) {
  return encodePngBytes(rgba, width, height);
}

export async function mkScratch(prefix) {
  return mkdtemp(join(tmpdir(), prefix));
}

/* ------------------------- template atlas rasterization ------------------------- */

/** For every atlas pixel covered by a template UV triangle, record which triangle and its
 *  barycentric weights (bA,bB,bC for the triangle's 3 corners in index order). Perimeter
 *  (DDA) pixels are included too (thin/sliver triangle coverage), matching texel-copy.ts's
 *  rasterizeTriangle reasoning. -1 in triId = uncovered. */
export function rasterizeTemplateBary(uvs, indices, width, height) {
  const triId = new Int32Array(width * height).fill(-1);
  const baryA = new Float32Array(width * height);
  const baryB = new Float32Array(width * height);
  const baryC = new Float32Array(width * height);

  const setPixel = (x, y, t, ax, ay, bx, by, cx, cy) => {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const px = x + 0.5, py = y + 0.5;
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const sign = area >= 0 ? 1 : -1;
    const w0 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) * sign; // -> baryC
    const w1 = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) * sign; // -> baryA
    const w2 = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) * sign; // -> baryB
    const sum = w0 + w1 + w2 || 1;
    const p = y * width + x;
    triId[p] = t; baryA[p] = w1 / sum; baryB[p] = w2 / sum; baryC[p] = w0 / sum;
  };

  const ddaLine = (t, ax, ay, bx, by, cx, cy, x0, y0, x1, y1) => {
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let i = 0; i <= steps; i++) {
      const x = Math.round(x0 + ((x1 - x0) * i) / steps);
      const y = Math.round(y0 + ((y1 - y0) * i) / steps);
      setPixel(x, y, t, ax, ay, bx, by, cx, cy);
    }
  };

  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 2, b = indices[t + 1] * 2, c = indices[t + 2] * 2;
    const ax = uvs[a] * width, ay = uvs[a + 1] * height;
    const bx = uvs[b] * width, by = uvs[b + 1] * height;
    const cx = uvs[c] * width, cy = uvs[c + 1] * height;
    // perimeter first (covers slivers), interior overwrites with cleaner center-sample bary
    ddaLine(t, ax, ay, bx, by, cx, cy, ax, ay, bx, by);
    ddaLine(t, ax, ay, bx, by, cx, cy, bx, by, cx, cy);
    ddaLine(t, ax, ay, bx, by, cx, cy, cx, cy, ax, ay);
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (area === 0) continue;
    const sign = area > 0 ? 1 : -1;
    const minX = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(ax, bx, cx)));
    const minY = Math.max(0, Math.floor(Math.min(ay, by, cy)));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(ay, by, cy)));
    for (let y = minY; y <= maxY; y++) {
      const py = y + 0.5;
      for (let x = minX; x <= maxX; x++) {
        const px = x + 0.5;
        const w0 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) * sign;
        const w1 = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) * sign;
        const w2 = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) * sign;
        if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
          const sum = w0 + w1 + w2 || 1;
          const p = y * width + x;
          triId[p] = t; baryA[p] = w1 / sum; baryB[p] = w2 / sum; baryC[p] = w0 / sum;
        }
      }
    }
  }
  return { triId, baryA, baryB, baryC, width, height };
}

/** Dilation (adapted from texel-copy.ts dilateMask) restricted to a boolean occupancy Uint8Array. */
export function dilate(mask, w, h, radius) {
  const offsets = [];
  const r2 = radius * radius;
  for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) if (dx * dx + dy * dy <= r2) offsets.push([dx, dy]);
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (const [dx, dy] of offsets) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) out[ny * w + nx] = 1;
      }
    }
  }
  return out;
}

/** For every RING pixel (dilated but not occupied), find the nearest OCCUPIED pixel (BFS,
 *  small radius) so the bake can copy its already-baked color (gutter bleed). -1 = none found. */
export function buildGutterMap(occMask, w, h, radius) {
  const nearest = new Int32Array(w * h).fill(-1);
  const dist = new Float32Array(w * h).fill(Infinity);
  const queue = [];
  for (let p = 0; p < occMask.length; p++) if (occMask[p]) { nearest[p] = p; dist[p] = 0; queue.push(p); }
  let head = 0;
  const r2 = radius * radius;
  while (head < queue.length) {
    const p = queue[head++];
    const x = p % w, y = (p - x) / w;
    const src = nearest[p];
    const sx = src % w, sy = (src - sx) / w;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const ddx = nx - sx, ddy = ny - sy;
      const d2 = ddx * ddx + ddy * ddy;
      if (d2 > r2) continue;
      const np = ny * w + nx;
      if (d2 < dist[np]) { dist[np] = d2; nearest[np] = src; queue.push(np); }
    }
  }
  return nearest;
}

export function bilinearSample(data, w, h, u, v) {
  let fx = u * w - 0.5, fy = v * h - 0.5;
  if (fx < 0) fx = 0; if (fy < 0) fy = 0;
  const x0 = Math.min(w - 1, Math.floor(fx)), y0 = Math.min(h - 1, Math.floor(fy));
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const tx = fx - x0, ty = fy - y0;
  const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4, i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
  const out = new Uint8Array(4);
  for (let c = 0; c < 4; c++) {
    const top = data[i00 + c] * (1 - tx) + data[i10 + c] * tx;
    const bot = data[i01 + c] * (1 - tx) + data[i11 + c] * tx;
    out[c] = Math.round(top * (1 - ty) + bot * ty);
  }
  return out;
}

/** Bake ONE frame's coherent atlas + OBJ given a precomputed template raster (triId/bary/occ/
 *  dilated/gutter) and that frame's deformed template positions. Shared by the single-process
 *  bake.mjs (debug) and bake-worker.mjs (parallel across frames). Returns timing + texel count. */
export function bakeFrame({
  srcPositions, srcIndices, srcUvs, deformed,
  templateIndices, triId, baryA, baryB, baryC, occMask, dilated, gutter, width, height,
  srcAtlasData, srcAtlasWidth, srcAtlasHeight,
}) {
  const diag = bboxDiag(srcPositions) || 1;
  const grid = new TriangleGrid(srcPositions, srcIndices, srcUvs, diag / 48);
  const outAtlas = new Uint8Array(width * height * 4);
  const tmpPt = new Float32Array(3);
  const tmpUv = new Float32Array(2);
  let sampled = 0;
  for (let p = 0; p < triId.length; p++) {
    const t = triId[p];
    if (t < 0) continue;
    const a = templateIndices[t] * 3, b = templateIndices[t + 1] * 3, c = templateIndices[t + 2] * 3;
    const wA = baryA[p], wB = baryB[p], wC = baryC[p];
    const px = wA * deformed[a] + wB * deformed[b] + wC * deformed[c];
    const py = wA * deformed[a + 1] + wB * deformed[b + 1] + wC * deformed[c + 1];
    const pz = wA * deformed[a + 2] + wB * deformed[b + 2] + wC * deformed[c + 2];
    grid.nearestInto(px, py, pz, tmpPt, 0, tmpUv, 0);
    const rgba = bilinearSample(srcAtlasData, srcAtlasWidth, srcAtlasHeight, tmpUv[0], tmpUv[1]);
    const o = p * 4;
    outAtlas[o] = rgba[0]; outAtlas[o + 1] = rgba[1]; outAtlas[o + 2] = rgba[2]; outAtlas[o + 3] = 255;
    sampled++;
  }
  for (let p = 0; p < dilated.length; p++) {
    if (!dilated[p] || occMask[p]) continue;
    const src2 = gutter[p];
    if (src2 < 0) continue;
    const o = p * 4, so = src2 * 4;
    outAtlas[o] = outAtlas[so]; outAtlas[o + 1] = outAtlas[so + 1]; outAtlas[o + 2] = outAtlas[so + 2]; outAtlas[o + 3] = 255;
  }
  return { outAtlas, sampled };
}

/** Phase 2 fast bake: same contract/output as bakeFrame, but the per-texel nearest-source-surface
 *  query is SEEDED from registration's per-vertex results instead of a cold TriangleGrid search
 *  (see "Phase 2 perf" section above) — this is the ~220x-fewer-effective-candidates lever the
 *  POC report identified as the dominant bake cost. `vertexSeedTri` (template-vertex-indexed,
 *  values = this frame's SOURCE flat triangle ids, from registerChainFrame) seeds a small
 *  per-TEMPLATE-triangle candidate set (buildLocalCandidates, built once per frame here — NOT
 *  per texel). Any texel whose local-scan best distance exceeds `fallbackMm` (or has no local
 *  candidates) falls back to the exact global TriangleGrid search — a correctness safety net,
 *  expected to fire rarely (reported in fallbackCount/fallbackPct). */
export function bakeFrameFast({
  srcPositions, srcIndices, srcUvs, deformed,
  templateIndices, triId, baryA, baryB, baryC, occMask, dilated, gutter, width, height,
  srcAtlasData, srcAtlasWidth, srcAtlasHeight,
  vertexSeedTri, rings = 2, fallbackMm = 3,
}) {
  const srcAdj = buildTriAdjacency(srcIndices);
  const { offsets: candOffsets, candidates: candArray } = buildLocalCandidates(templateIndices, vertexSeedTri, srcAdj, rings);
  const diag = bboxDiag(srcPositions) || 1;
  const grid = new TriangleGrid(srcPositions, srcIndices, srcUvs, diag / 48); // fallback safety net only
  const outAtlas = new Uint8Array(width * height * 4);
  const tmpPt = new Float32Array(3);
  const tmpUv = new Float32Array(2);
  const fallbackD2 = fallbackMm * fallbackMm;
  let sampled = 0, fallbackCount = 0;
  for (let p = 0; p < triId.length; p++) {
    const t = triId[p];
    if (t < 0) continue;
    const a = templateIndices[t] * 3, b = templateIndices[t + 1] * 3, c = templateIndices[t + 2] * 3;
    const wA = baryA[p], wB = baryB[p], wC = baryC[p];
    const px = wA * deformed[a] + wB * deformed[b] + wC * deformed[c];
    const py = wA * deformed[a + 1] + wB * deformed[b + 1] + wC * deformed[c + 1];
    const pz = wA * deformed[a + 2] + wB * deformed[b + 2] + wC * deformed[c + 2];
    const ti = (t / 3) | 0;
    const d2 = nearestLocalScan(candOffsets, candArray, ti, srcPositions, srcIndices, srcUvs, px, py, pz, tmpPt, 0, tmpUv, 0);
    if (!(d2 <= fallbackD2)) {
      grid.nearestInto(px, py, pz, tmpPt, 0, tmpUv, 0);
      fallbackCount++;
    }
    const rgba = bilinearSample(srcAtlasData, srcAtlasWidth, srcAtlasHeight, tmpUv[0], tmpUv[1]);
    const o = p * 4;
    outAtlas[o] = rgba[0]; outAtlas[o + 1] = rgba[1]; outAtlas[o + 2] = rgba[2]; outAtlas[o + 3] = 255;
    sampled++;
  }
  for (let p = 0; p < dilated.length; p++) {
    if (!dilated[p] || occMask[p]) continue;
    const src2 = gutter[p];
    if (src2 < 0) continue;
    const o = p * 4, so = src2 * 4;
    outAtlas[o] = outAtlas[so]; outAtlas[o + 1] = outAtlas[so + 1]; outAtlas[o + 2] = outAtlas[so + 2]; outAtlas[o + 3] = 255;
  }
  return { outAtlas, sampled, fallbackCount, fallbackPct: sampled ? (100 * fallbackCount) / sampled : 0 };
}

export function bboxDiag(positions) {
  let minx = Infinity, miny = Infinity, minz = Infinity, maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (x < minx) minx = x; if (y < miny) miny = y; if (z < minz) minz = z;
    if (x > maxx) maxx = x; if (y > maxy) maxy = y; if (z > maxz) maxz = z;
  }
  return Math.hypot(maxx - minx, maxy - miny, maxz - minz);
}
