/**
 * ARAP (As-Rigid-As-Possible) registration POC.
 *
 * Fixes the wrong-surface-snap that nearest-point + Taubin registration can't: instead of pulling
 * each vertex to its nearest source point (which snaps to the wrong surface at ~0mm error while
 * tearing the fixed-topology edge to 30x — see 2026-07-14 report), we minimize
 *
 *      E(p') = Σ_i Σ_{j∈N(i)} || (p'_i - p'_j) - R_i (p_i - p_j) ||²   +   λ Σ_i c_i || p'_i - t_i ||²
 *              \___________ ARAP rigidity vs TEMPLATE rest ___________/     \___ fit to source ___/
 *
 * solved by local/global alternation (Sorkine-Alexa 2007): local = per-vertex optimal rotation R_i
 * (Müller 2016 quaternion iteration — always a proper rotation, no SVD/degeneracy), global = sparse
 * SPD solve (conjugate gradient) for p'. Rigidity is measured against the TEMPLATE (run frame 0), so
 * articulation (rotation) is free but tearing (stretch) costs energy — a lone vertex can't run to the
 * wrong surface without dragging its neighbors, so the global solve refuses it. Weld-aware (solves on
 * the compacted graph so UV-seam duplicates never crack). Run standalone to validate before wiring in.
 *
 *   node arap-poc.mjs <src-frames-dir> --template 121 --to 150 [--lambda 2] [--gate-mm 40] [--outer 8]
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";
import { TriangleGrid, buildWeldAdjacency, bboxDiag } from "../coherent-poc/lib.mjs";

/* ------------------------------- small linear algebra ------------------------------- */
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Signed volume (divergence theorem) of a CLOSED triangle mesh. Correspondence-free shred/implosion
 *  detector: unlike fit-to-nearest-surface it cannot be fooled by a vertex sitting on the wrong body
 *  part, and unlike max-edge it actually discriminates. Sum of per-tri tetrahedron volumes to origin. */
function meshVolume(pos, idx) {
  let v = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    v += (pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1])
        - pos[a + 1] * (pos[b] * pos[c + 2] - pos[b + 2] * pos[c])
        + pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c])) / 6;
  }
  return Math.abs(v);
}

// Rotation matrix (column-major cols r0,r1,r2) from unit quaternion [x,y,z,w].
function quatCols(q) {
  const [x, y, z, w] = q;
  const xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
  return {
    r0: [1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)],
    r1: [2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)],
    r2: [2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)],
  };
}
function quatMul(a, b) { // a*b, quats [x,y,z,w]
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}
// Extract the rotation part of A (columns a0,a1,a2) as a quaternion, warm-started from q (Müller 2016).
// We pass A = Σ w e'_ij e_ij^T so the returned R maximizes tr(R S), S = Σ w e_ij e'_ij^T — the ARAP rotation.
function extractRotation(a0, a1, a2, q) {
  for (let it = 0; it < 24; it++) {
    const R = quatCols(q);
    const c0 = cross(R.r0, a0), c1 = cross(R.r1, a1), c2 = cross(R.r2, a2);
    const denom = Math.abs(dot3(R.r0, a0) + dot3(R.r1, a1) + dot3(R.r2, a2)) + 1e-9;
    const ox = (c0[0] + c1[0] + c2[0]) / denom, oy = (c0[1] + c1[1] + c2[1]) / denom, oz = (c0[2] + c1[2] + c2[2]) / denom;
    const w = Math.hypot(ox, oy, oz);
    if (w < 1e-9) break;
    const half = w * 0.5, s = Math.sin(half) / w;
    const dq = [ox * s, oy * s, oz * s, Math.cos(half)];
    q = quatMul(dq, q);
    const ql = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    q = [q[0] / ql, q[1] / ql, q[2] / ql, q[3] / ql];
  }
  return q;
}

/* --------------------------------- compacted weld graph --------------------------------- */
// From the weld map, build the compacted (seam-collapsed) uniform-weight adjacency + rest positions.
function buildCompact(templatePos, weld) {
  const { n, nc, canon, compact, offsets, neighbors } = weld;
  const restC = new Float64Array(nc * 3);
  for (let v = 0; v < n; v++) {
    const ci = compact[canon[v]];
    restC[ci * 3] = templatePos[canon[v] * 3]; restC[ci * 3 + 1] = templatePos[canon[v] * 3 + 1]; restC[ci * 3 + 2] = templatePos[canon[v] * 3 + 2];
  }
  const deg = new Float64Array(nc);
  for (let i = 0; i < nc; i++) deg[i] = offsets[i + 1] - offsets[i];
  return { nc, offsets, neighbors, restC, deg };
}

/* ------------------------------ conjugate gradient (SPD) ------------------------------ */
// Solve (L + λ diag(c)) x = b for one coordinate. matvec: (M x)_i = (deg_i + λ c_i) x_i - Σ_j x_j.
function cgSolve(nc, offsets, neighbors, deg, lamC, b, x, iters, tol) {
  const r = new Float64Array(nc), p = new Float64Array(nc), Ap = new Float64Array(nc);
  const matvec = (v, out) => {
    for (let i = 0; i < nc; i++) {
      let s = (deg[i] + lamC[i]) * v[i];
      for (let k = offsets[i]; k < offsets[i + 1]; k++) s -= v[neighbors[k]];
      out[i] = s;
    }
  };
  matvec(x, Ap);
  let rs = 0;
  for (let i = 0; i < nc; i++) { r[i] = b[i] - Ap[i]; p[i] = r[i]; rs += r[i] * r[i]; }
  if (Math.sqrt(rs) < tol) return;
  for (let k = 0; k < iters; k++) {
    matvec(p, Ap);
    let pAp = 0; for (let i = 0; i < nc; i++) pAp += p[i] * Ap[i];
    if (pAp <= 1e-30) break;
    const alpha = rs / pAp;
    let rs2 = 0;
    for (let i = 0; i < nc; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; rs2 += r[i] * r[i]; }
    if (Math.sqrt(rs2) < tol) break;
    const beta = rs2 / rs;
    for (let i = 0; i < nc; i++) p[i] = r[i] + beta * p[i];
    rs = rs2;
  }
}

/* ----------------------------------- ARAP registration ----------------------------------- */
/** Register one frame: deform the template (compacted) to fit `grid` (source), ARAP-rigid vs template.
 *  `initC` = warm-start compacted positions (previous frame's result). Returns compacted positions. */
function arapRegisterFrame(cmp, grid, initC, opts) {
  const { nc, offsets, neighbors, restC, deg } = cmp;
  const { lambda = 2, gateMm = 40, outer = 8, cgIters = 80, flowTarget = null, flowConf = null, flowSnap = 30 } = opts;
  const snap2 = flowSnap * flowSnap;
  const pC = Float64Array.from(initC);
  const quats = opts.quats || Array.from({ length: nc }, () => [0, 0, 0, 1]);
  const tgt = new Float64Array(nc * 3), conf = new Float64Array(nc);
  const bx = new Float64Array(nc), by = new Float64Array(nc), bz = new Float64Array(nc), lamC = new Float64Array(nc);
  const tmp = new Float32Array(3);
  const gate2 = gateMm * gateMm;
  for (let o = 0; o < outer; o++) {
    // 1. targets: nearest source-surface point to each current compacted vertex + confidence gate.
    for (let i = 0; i < nc; i++) {
      const d2 = grid.nearestInto(pC[i * 3], pC[i * 3 + 1], pC[i * 3 + 2], tmp, 0);
      // HYBRID: nearest-point is accurate for the bulk; flow (a few-mm noisier) is used ONLY where it
      // disagrees with nearest-point by > flowSnap — those are the wrong-surface SNAPS flow catches.
      if (flowTarget && flowConf[i] > 0) {
        const gx = flowTarget[i * 3] - tmp[0], gy = flowTarget[i * 3 + 1] - tmp[1], gz = flowTarget[i * 3 + 2] - tmp[2];
        if (gx * gx + gy * gy + gz * gz > snap2) {
          tgt[i * 3] = flowTarget[i * 3]; tgt[i * 3 + 1] = flowTarget[i * 3 + 1]; tgt[i * 3 + 2] = flowTarget[i * 3 + 2];
          conf[i] = 1.0;
          continue;
        }
      }
      tgt[i * 3] = tmp[0]; tgt[i * 3 + 1] = tmp[1]; tgt[i * 3 + 2] = tmp[2];
      // gate: a target much farther than gateMm is likely a wrong/occluded match → downweight it and
      // let ARAP rigidly carry the vertex from its neighbors instead.
      conf[i] = d2 > gate2 ? 0.02 : 1.0;
    }
    // 2. local step: per-vertex optimal rotation R_i (warm-started quaternion).
    const R = new Array(nc);
    for (let i = 0; i < nc; i++) {
      // A_i = Σ_j (e'_ij) (e_ij)^T  (columns keyed by rest-edge component)
      let a00 = 0, a01 = 0, a02 = 0, a10 = 0, a11 = 0, a12 = 0, a20 = 0, a21 = 0, a22 = 0;
      const ix = pC[i * 3], iy = pC[i * 3 + 1], iz = pC[i * 3 + 2];
      const rix = restC[i * 3], riy = restC[i * 3 + 1], riz = restC[i * 3 + 2];
      for (let k = offsets[i]; k < offsets[i + 1]; k++) {
        const j = neighbors[k];
        const ex = ix - pC[j * 3], ey = iy - pC[j * 3 + 1], ez = iz - pC[j * 3 + 2];        // e' deformed
        const rx = rix - restC[j * 3], ry = riy - restC[j * 3 + 1], rz = riz - restC[j * 3 + 2]; // e rest
        a00 += ex * rx; a01 += ex * ry; a02 += ex * rz;
        a10 += ey * rx; a11 += ey * ry; a12 += ey * rz;
        a20 += ez * rx; a21 += ez * ry; a22 += ez * rz;
      }
      const q = extractRotation([a00, a10, a20], [a01, a11, a21], [a02, a12, a22], quats[i]);
      quats[i] = q; R[i] = quatCols(q);
    }
    // 3. global step: build RHS b_i = Σ_j 0.5 (R_i+R_j) e_ij  +  λ c_i t_i ;  M = L + λ diag(c).
    for (let i = 0; i < nc; i++) {
      const Ri = R[i];
      let sx = 0, sy = 0, sz = 0;
      const rix = restC[i * 3], riy = restC[i * 3 + 1], riz = restC[i * 3 + 2];
      for (let k = offsets[i]; k < offsets[i + 1]; k++) {
        const j = neighbors[k]; const Rj = R[j];
        const ex = rix - restC[j * 3], ey = riy - restC[j * 3 + 1], ez = riz - restC[j * 3 + 2];
        // 0.5 (R_i + R_j) e
        sx += 0.5 * ((Ri.r0[0] + Rj.r0[0]) * ex + (Ri.r1[0] + Rj.r1[0]) * ey + (Ri.r2[0] + Rj.r2[0]) * ez);
        sy += 0.5 * ((Ri.r0[1] + Rj.r0[1]) * ex + (Ri.r1[1] + Rj.r1[1]) * ey + (Ri.r2[1] + Rj.r2[1]) * ez);
        sz += 0.5 * ((Ri.r0[2] + Rj.r0[2]) * ex + (Ri.r1[2] + Rj.r1[2]) * ey + (Ri.r2[2] + Rj.r2[2]) * ez);
      }
      const lc = lambda * conf[i];
      lamC[i] = lc;
      bx[i] = sx + lc * tgt[i * 3]; by[i] = sy + lc * tgt[i * 3 + 1]; bz[i] = sz + lc * tgt[i * 3 + 2];
    }
    // 4. solve each coordinate (M shared), warm-started from current positions.
    const x = new Float64Array(nc), y = new Float64Array(nc), z = new Float64Array(nc);
    for (let i = 0; i < nc; i++) { x[i] = pC[i * 3]; y[i] = pC[i * 3 + 1]; z[i] = pC[i * 3 + 2]; }
    cgSolve(nc, offsets, neighbors, deg, lamC, bx, x, cgIters, 1e-4);
    cgSolve(nc, offsets, neighbors, deg, lamC, by, y, cgIters, 1e-4);
    cgSolve(nc, offsets, neighbors, deg, lamC, bz, z, cgIters, 1e-4);
    for (let i = 0; i < nc; i++) { pC[i * 3] = x[i]; pC[i * 3 + 1] = y[i]; pC[i * 3 + 2] = z[i]; }
  }
  opts.quats = quats;
  return pC;
}

/* ------------------------------------- metrics + main ------------------------------------- */
function scatterToFull(pC, templatePos, weld) {
  const { n, canon, compact } = weld;
  const out = new Float32Array(n * 3);
  for (let v = 0; v < n; v++) { const ci = compact[canon[v]]; out[v * 3] = pC[ci * 3]; out[v * 3 + 1] = pC[ci * 3 + 1]; out[v * 3 + 2] = pC[ci * 3 + 2]; }
  return out;
}
function edgeMetrics(pos, idx, restMaxEdge) {
  let maxE = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    maxE = Math.max(maxE,
      Math.hypot(pos[a] - pos[b], pos[a + 1] - pos[b + 1], pos[a + 2] - pos[b + 2]),
      Math.hypot(pos[b] - pos[c], pos[b + 1] - pos[c + 1], pos[b + 2] - pos[c + 2]),
      Math.hypot(pos[c] - pos[a], pos[c + 1] - pos[a + 1], pos[c + 2] - pos[a + 2]));
  }
  return { maxE, ratio: maxE / restMaxEdge };
}
function fitError(pos, grid) {
  const n = pos.length / 3, tmp = new Float32Array(3); let sum = 0, mx = 0; const ds = [];
  for (let i = 0; i < n; i++) { const d = Math.sqrt(grid.nearestInto(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], tmp, 0)); sum += d; if (d > mx) mx = d; ds.push(d); }
  ds.sort((a, b) => a - b);
  return { mean: sum / n, p95: ds[Math.floor(0.95 * n)], max: mx };
}
function templateMaxEdge(pos, idx) { return edgeMetrics(pos, idx, 1).maxE; }

// ---- optical-flow scene-flow: uniform-grid point lookup for the (src->dst) samples of a frame pair ----
function buildPairFlow(samples, cell) {
  const n = samples.length;
  const src = new Float32Array(n * 3), dst = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const s = samples[i]; src[i * 3] = s[0]; src[i * 3 + 1] = s[1]; src[i * 3 + 2] = s[2]; dst[i * 3] = s[3]; dst[i * 3 + 1] = s[4]; dst[i * 3 + 2] = s[5]; }
  const map = new Map();
  const key = (ix, iy, iz) => ix + "," + iy + "," + iz;
  for (let i = 0; i < n; i++) { const k = key(Math.floor(src[i * 3] / cell), Math.floor(src[i * 3 + 1] / cell), Math.floor(src[i * 3 + 2] / cell)); let a = map.get(k); if (!a) { a = []; map.set(k, a); } a.push(i); }
  return { src, dst, map, cell, key };
}
const median = (a) => { a.sort((p, q) => p - q); const n = a.length; return n & 1 ? a[n >> 1] : 0.5 * (a[n >> 1] + a[(n >> 1) - 1]); };
// Robust flow displacement at (x,y,z): MEDIAN of nearby samples' displacements (rejects isolated Farneback
// outliers) + a disagreement gate — if nearby flow disagrees (MAD > spread), it's unreliable → return null
// (fall back to nearest-point). This is what rescues the approach: raw inverse-distance averaging let one
// bad sample dominate and injected 195mm spikes on low-motion frames.
function flowDisp(pf, x, y, z, radius, minSamples, spread) {
  const c = pf.cell, ix = Math.floor(x / c), iy = Math.floor(y / c), iz = Math.floor(z / c), r2 = radius * radius;
  const dxs = [], dys = [], dzs = [];
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let e = -1; e <= 1; e++) {
    const arr = pf.map.get(pf.key(ix + a, iy + b, iz + e)); if (!arr) continue;
    for (const i of arr) { const sx = pf.src[i * 3], sy = pf.src[i * 3 + 1], sz = pf.src[i * 3 + 2]; const d2 = (sx - x) ** 2 + (sy - y) ** 2 + (sz - z) ** 2; if (d2 > r2) continue; dxs.push(pf.dst[i * 3] - sx); dys.push(pf.dst[i * 3 + 1] - sy); dzs.push(pf.dst[i * 3 + 2] - sz); }
  }
  if (dxs.length < minSamples) return null;
  const mx = median(dxs), my = median(dys), mz = median(dzs);
  // median-absolute-deviation spread: high = nearby flow disagrees (outliers / motion boundary) → distrust
  let sad = 0; for (let i = 0; i < dxs.length; i++) sad += Math.abs(dxs[i] - mx) + Math.abs(dys[i] - my) + Math.abs(dzs[i] - mz);
  if (sad / dxs.length > spread) return null;
  return [mx, my, mz];
}

async function main() {
  const args = process.argv.slice(2);
  const srcDir = args[0];
  // VOLUME RATIO (registered / source) — added 2026-07-15 because this tool's other two metrics are
  // BLIND to the failure actually seen in playback. Fit error is distance-to-nearest-surface, and a
  // vertex dragged onto the WRONG body part sits ~0mm from a real surface, so fit stays clean while
  // the mesh deflates. Max-edge doesn't discriminate either (measured: daniel-s0, the GOOD clip, has
  // a LARGER max edge than the coherent bakes). Measured on the shipped v3 bake, the template
  // collapses to 0.63x true volume around f143-149 — 33 LITRES of the subject gone — while both
  // metrics above report "CLEAN + FITS". Volume is correspondence-free and needs no nearest-surface,
  // and both meshes are closed manifolds (verified: 0 boundary, 0 non-manifold edges), so the
  // divergence-theorem sum is exact, not an approximation.
  const flag = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
  const tmplF = Number(flag("--template", 121)), toF = Number(flag("--to", 150));
  const lambda = Number(flag("--lambda", 2)), gateMm = Number(flag("--gate-mm", 40)), outer = Number(flag("--outer", 8));
  const pad = 5, fn = (i) => join(srcDir, `mesh-f${String(i).padStart(pad, "0")}.obj`);

  const tmpl = parseObj(await readFile(fn(tmplF), "utf8"));
  const weld = buildWeldAdjacency(tmpl.positions, tmpl.indices);
  const cmp = buildCompact(tmpl.positions, weld);
  const restMaxE = templateMaxEdge(tmpl.positions, tmpl.indices);
  console.log(`[arap] template f${tmplF}: ${tmpl.positions.length / 3} verts (${cmp.nc} welded) maxEdge=${restMaxE.toFixed(1)}mm | chain to f${toF} | λ=${lambda} gate=${gateMm}mm outer=${outer}`);

  // chain: warm-start each frame from the previous ARAP result; rest stays = template.
  // --predict: motion-compensated init = linear extrapolation 2·prev − prevprev (damped by `predictK`),
  // so the template STARTS near frame N+1's true pose → nearest-point stays on the right surface instead
  // of snapping to the wrong one when a limb moved >gate. Non-circular (uses only past frames).
  const PREDICT = args.includes("--predict");
  const predictK = Number(flag("--predict-k", 1)); // 1 = full velocity, 0.5 = damped
  // --flow <dir>: optical-flow scene-flow samples (pair_NNNNN.json) override nearest-point targets.
  const FLOW = flag("--flow", null);
  const flowR = Number(flag("--flow-r", 35)), flowMin = Number(flag("--flow-min", 3)), flowSnap = Number(flag("--flow-snap", 30)), flowSpread = Number(flag("--flow-spread", 20));
  const pairFlow = {};
  if (FLOW) {
    for (let f = tmplF; f < toF; f++) {
      try { const d = JSON.parse(await readFile(join(FLOW, `pair_${String(f).padStart(pad, "0")}.json`), "utf8")); pairFlow[f] = buildPairFlow(d.samples, flowR); } catch { /* pair missing */ }
    }
    console.log(`[arap] flow: loaded ${Object.keys(pairFlow).length} pair(s) from ${FLOW} (r=${flowR}mm min=${flowMin})`);
  }
  let prevC = Float64Array.from(cmp.restC), prevprevC = null;
  const opts = { lambda, gateMm, outer, quats: null, flowSnap };
  let lastPos = tmpl.positions, lastGrid = null, t0 = Date.now(), spikes = [];
  for (let f = tmplF + 1; f <= toF; f++) {
    const src = parseObj(await readFile(fn(f), "utf8"));
    const diag = bboxDiag(src.positions) || 1;
    const grid = new TriangleGrid(src.positions, src.indices, src.uvs, diag / 48);
    let initC = prevC;
    if (PREDICT && prevprevC) { initC = new Float64Array(prevC.length); for (let i = 0; i < initC.length; i++) initC[i] = prevC[i] + predictK * (prevC[i] - prevprevC[i]); }
    // flow targets for pair (f-1 -> f): appearance-tracked, override nearest-point where covered.
    opts.flowTarget = null; opts.flowConf = null; let cov = 0;
    if (FLOW && pairFlow[f - 1]) {
      const pf = pairFlow[f - 1], ft = new Float64Array(cmp.nc * 3), fc = new Float64Array(cmp.nc);
      for (let i = 0; i < cmp.nc; i++) { const x = prevC[i * 3], y = prevC[i * 3 + 1], z = prevC[i * 3 + 2]; const disp = flowDisp(pf, x, y, z, flowR, flowMin, flowSpread); if (disp) { ft[i * 3] = x + disp[0]; ft[i * 3 + 1] = y + disp[1]; ft[i * 3 + 2] = z + disp[2]; fc[i] = 1; cov++; } }
      opts.flowTarget = ft; opts.flowConf = fc;
    }
    const pC = arapRegisterFrame(cmp, grid, initC, opts);
    prevprevC = prevC; prevC = pC; lastGrid = grid;
    lastPos = scatterToFull(pC, tmpl.positions, weld);
    const vr = meshVolume(lastPos, tmpl.indices) / meshVolume(src.positions, src.indices);
    { const fe = fitError(lastPos, grid); spikes.push({ f, max: fe.max, vol: vr, cov: FLOW ? Math.round(100 * cov / cmp.nc) : null }); }
    if (f === toF || f % 10 === 0) {
      const em = edgeMetrics(lastPos, tmpl.indices, restMaxE);
      const fe = fitError(lastPos, grid);
      console.log(`  f${f}: maxEdge=${em.maxE.toFixed(1)}mm (${em.ratio.toFixed(2)}x tmpl) | fit mean=${fe.mean.toFixed(2)} p95=${fe.p95.toFixed(2)} max=${fe.max.toFixed(2)} mm | VOL ${vr.toFixed(3)}x`);
    }
  }
  const em = edgeMetrics(lastPos, tmpl.indices, restMaxE);
  const fe = fitError(lastPos, lastGrid);
  console.log(`\n[arap] DONE f${toF}: maxEdge ${em.maxE.toFixed(1)}mm = ${em.ratio.toFixed(2)}x template (OLD nearest+Taubin tore to 4.5x@f150)`);
  console.log(`       surface fit: mean ${fe.mean.toFixed(2)}mm, p95 ${fe.p95.toFixed(2)}mm, max ${fe.max.toFixed(2)}mm | ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  const worst = spikes.reduce((a, b) => (b.max > a.max ? b : a), { f: 0, max: 0 });
  const worstVol = spikes.reduce((a, b) => (b.vol < a.vol ? b : a), { f: 0, vol: 9 });
  const mode = FLOW ? "FLOW targets" : (PREDICT ? "PREDICT init k=" + predictK : "chain init (baseline)");
  const avgCov = FLOW ? (spikes.reduce((s, x) => s + (x.cov || 0), 0) / spikes.length).toFixed(0) + "% flow-covered" : "";
  console.log(`       RUN max-fit spike (${mode}): ${worst.max.toFixed(1)}mm @ f${worst.f} over ${spikes.length} frames ${avgCov}`);
  console.log(`       RUN worst VOLUME: ${worstVol.vol.toFixed(3)}x true @ f${worstVol.f}  (1.000 = perfect; v3 ships 0.63x here) <-- the metric fit/edge are blind to`);
  console.log(`       VERDICT: ${em.ratio < 1.6 && fe.mean < 3 ? "CLEAN + FITS — ARAP works" : em.ratio < 1.6 ? "clean but loose fit (raise λ)" : fe.mean < 3 ? "fits but still stretched (raise outer/lower gate)" : "tune λ/outer"}`);
}
main().catch((e) => { console.error(e); process.exit(1); });
