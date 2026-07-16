/**
 * ARAP (As-Rigid-As-Possible) registration — reusable core (validated in arap-poc.mjs).
 *
 * Deforms the run TEMPLATE to fit each source frame while keeping every local neighborhood as-rigid-
 * as-possible vs the template, so articulation (rotation) is free but tearing (edge stretch) costs
 * energy — the fix for nearest-point's wrong-surface snap (which read ~0mm error while tearing edges
 * to 30x). Local step = per-vertex optimal rotation (Müller 2016 quaternion iteration, always a proper
 * rotation, no SVD). Global step = sparse SPD solve (conjugate gradient) for positions. Weld-aware:
 * solves on the seam-collapsed compacted graph so UV duplicates never crack.
 *
 * Measured (Daniel f121→150, the chain that tore to 4.5x): λ=10 → maxEdge 1.0x template, fit mean <1mm.
 */

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function quatCols(q) { // rotation columns r0,r1,r2 from unit quaternion [x,y,z,w]
  const [x, y, z, w] = q;
  const xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
  return {
    r0: [1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)],
    r1: [2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)],
    r2: [2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)],
  };
}
function quatMul(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}
// Rotation part of A (cols a0,a1,a2), warm-started from q. Pass A = Σ w e'_ij e_ij^T (deformed⊗rest).
function extractRotation(a0, a1, a2, q) {
  for (let it = 0; it < 24; it++) {
    const R = quatCols(q);
    const c0 = cross(R.r0, a0), c1 = cross(R.r1, a1), c2 = cross(R.r2, a2);
    const denom = Math.abs(dot3(R.r0, a0) + dot3(R.r1, a1) + dot3(R.r2, a2)) + 1e-9;
    const ox = (c0[0] + c1[0] + c2[0]) / denom, oy = (c0[1] + c1[1] + c2[1]) / denom, oz = (c0[2] + c1[2] + c2[2]) / denom;
    const w = Math.hypot(ox, oy, oz);
    if (w < 1e-9) break;
    const s = Math.sin(w * 0.5) / w;
    q = quatMul([ox * s, oy * s, oz * s, Math.cos(w * 0.5)], q);
    const ql = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    q = [q[0] / ql, q[1] / ql, q[2] / ql, q[3] / ql];
  }
  return q;
}

/** Compacted (seam-collapsed) uniform-weight graph + rest positions, from a weld map (buildWeldAdjacency). */
export function buildCompact(templatePos, weld) {
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

// (L + λ diag(c)) x = b, SPD. matvec: (M x)_i = (deg_i + λ c_i) x_i - Σ_{j∈N(i)} x_j.
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

/** Register one frame: deform the template (compacted) to fit `grid` (source TriangleGrid), ARAP-rigid
 *  vs template. `initC` = warm-start compacted positions (previous frame's result). `opts.quats` is the
 *  per-vertex rotation warm-start (persisted back into opts). Returns compacted Float64Array positions. */
export function arapRegisterFrame(cmp, grid, initC, opts) {
  const { nc, offsets, neighbors, restC, deg } = cmp;
  // flowTarget/flowConf (optional): a per-vertex, appearance-tracked target from optical-flow scene flow.
  // HYBRID, not flow-everywhere: nearest-point is accurate for the BULK of the surface and flow carries a
  // few mm of its own noise, so flow only overrides where it DISAGREES with nearest-point by > flowSnap —
  // those disagreements are exactly the wrong-surface snaps flow exists to catch. Flow-everywhere was
  // measured WORSE than baseline (189mm vs 85.5mm, 2026-07-14): the noise corrupted the accurate 95%.
  // Vertices with no confident flow (occluded) fall back to gated nearest-point.
  const { lambda = 10, gateMm = 40, outer = 10, cgIters = 80, flowTarget = null, flowConf = null, flowSnap = 30 } = opts;
  const snap2 = flowSnap * flowSnap;
  const pC = Float64Array.from(initC);
  const quats = opts.quats || Array.from({ length: nc }, () => [0, 0, 0, 1]);
  const tgt = new Float64Array(nc * 3), conf = new Float64Array(nc);
  const bx = new Float64Array(nc), by = new Float64Array(nc), bz = new Float64Array(nc), lamC = new Float64Array(nc);
  const tmp = new Float32Array(3), gate2 = gateMm * gateMm;
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < nc; i++) {
      const d2 = grid.nearestInto(pC[i * 3], pC[i * 3 + 1], pC[i * 3 + 2], tmp, 0);
      if (flowTarget && flowConf[i] > 0) {
        const gx = flowTarget[i * 3] - tmp[0], gy = flowTarget[i * 3 + 1] - tmp[1], gz = flowTarget[i * 3 + 2] - tmp[2];
        if (gx * gx + gy * gy + gz * gz > snap2) {   // flow disagrees with nearest-point => a wrong-surface snap
          tgt[i * 3] = flowTarget[i * 3]; tgt[i * 3 + 1] = flowTarget[i * 3 + 1]; tgt[i * 3 + 2] = flowTarget[i * 3 + 2];
          conf[i] = flowConf[i];
          continue;
        }
      }
      tgt[i * 3] = tmp[0]; tgt[i * 3 + 1] = tmp[1]; tgt[i * 3 + 2] = tmp[2];
      conf[i] = d2 > gate2 ? 0.02 : 1.0;   // gate wrong/occluded matches → ARAP carries them rigidly
    }
    const R = new Array(nc);
    for (let i = 0; i < nc; i++) {
      let a00 = 0, a01 = 0, a02 = 0, a10 = 0, a11 = 0, a12 = 0, a20 = 0, a21 = 0, a22 = 0;
      const ix = pC[i * 3], iy = pC[i * 3 + 1], iz = pC[i * 3 + 2];
      const rix = restC[i * 3], riy = restC[i * 3 + 1], riz = restC[i * 3 + 2];
      for (let k = offsets[i]; k < offsets[i + 1]; k++) {
        const j = neighbors[k];
        const ex = ix - pC[j * 3], ey = iy - pC[j * 3 + 1], ez = iz - pC[j * 3 + 2];
        const rx = rix - restC[j * 3], ry = riy - restC[j * 3 + 1], rz = riz - restC[j * 3 + 2];
        a00 += ex * rx; a01 += ex * ry; a02 += ex * rz;
        a10 += ey * rx; a11 += ey * ry; a12 += ey * rz;
        a20 += ez * rx; a21 += ez * ry; a22 += ez * rz;
      }
      const q = extractRotation([a00, a10, a20], [a01, a11, a21], [a02, a12, a22], quats[i]);
      quats[i] = q; R[i] = quatCols(q);
    }
    for (let i = 0; i < nc; i++) {
      const Ri = R[i];
      let sx = 0, sy = 0, sz = 0;
      const rix = restC[i * 3], riy = restC[i * 3 + 1], riz = restC[i * 3 + 2];
      for (let k = offsets[i]; k < offsets[i + 1]; k++) {
        const j = neighbors[k]; const Rj = R[j];
        const ex = rix - restC[j * 3], ey = riy - restC[j * 3 + 1], ez = riz - restC[j * 3 + 2];
        sx += 0.5 * ((Ri.r0[0] + Rj.r0[0]) * ex + (Ri.r1[0] + Rj.r1[0]) * ey + (Ri.r2[0] + Rj.r2[0]) * ez);
        sy += 0.5 * ((Ri.r0[1] + Rj.r0[1]) * ex + (Ri.r1[1] + Rj.r1[1]) * ey + (Ri.r2[1] + Rj.r2[1]) * ez);
        sz += 0.5 * ((Ri.r0[2] + Rj.r0[2]) * ex + (Ri.r1[2] + Rj.r1[2]) * ey + (Ri.r2[2] + Rj.r2[2]) * ez);
      }
      const lc = lambda * conf[i]; lamC[i] = lc;
      bx[i] = sx + lc * tgt[i * 3]; by[i] = sy + lc * tgt[i * 3 + 1]; bz[i] = sz + lc * tgt[i * 3 + 2];
    }
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

/** Scatter compacted positions back to the full (un-welded) vertex array. */
export function scatterToFull(pC, weld) {
  const { n, canon, compact } = weld;
  const out = new Float32Array(n * 3);
  for (let v = 0; v < n; v++) { const ci = compact[canon[v]]; out[v * 3] = pC[ci * 3]; out[v * 3 + 1] = pC[ci * 3 + 1]; out[v * 3 + 2] = pC[ci * 3 + 2]; }
  return out;
}

/* ---------------- optical-flow scene-flow lookup (shared with arap-poc.mjs) ---------------- */
// Uniform-grid index over a frame pair's (src->dst) RAFT scene-flow samples.
export function buildPairFlow(samples, cell) {
  const n = samples.length;
  const src = new Float32Array(n * 3), dst = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const s = samples[i]; src[i*3]=s[0]; src[i*3+1]=s[1]; src[i*3+2]=s[2]; dst[i*3]=s[3]; dst[i*3+1]=s[4]; dst[i*3+2]=s[5]; }
  const map = new Map();
  const key = (ix, iy, iz) => ix + "," + iy + "," + iz;
  for (let i = 0; i < n; i++) { const k = key(Math.floor(src[i*3]/cell), Math.floor(src[i*3+1]/cell), Math.floor(src[i*3+2]/cell)); let a = map.get(k); if (!a) { a = []; map.set(k, a); } a.push(i); }
  return { src, dst, map, cell, key };
}
const _median = (a) => { const b = a.slice().sort((x, y) => x - y); const m = b.length >> 1; return b.length % 2 ? b[m] : (b[m-1] + b[m]) / 2; };
/** Robust flow displacement at a point: MEDIAN of nearby samples (rejects isolated outliers) + a
 *  disagreement gate — if nearby flow disagrees (MAD > spread) it is unreliable (motion boundary) => null. */
export function flowDisp(pf, x, y, z, radius, minSamples, spread) {
  const c = pf.cell, ix = Math.floor(x/c), iy = Math.floor(y/c), iz = Math.floor(z/c), r2 = radius * radius;
  const dxs = [], dys = [], dzs = [];
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let e = -1; e <= 1; e++) {
    const arr = pf.map.get(pf.key(ix+a, iy+b, iz+e)); if (!arr) continue;
    for (const i of arr) { const sx = pf.src[i*3], sy = pf.src[i*3+1], sz = pf.src[i*3+2];
      const d2 = (sx-x)**2 + (sy-y)**2 + (sz-z)**2; if (d2 > r2) continue;
      dxs.push(pf.dst[i*3]-sx); dys.push(pf.dst[i*3+1]-sy); dzs.push(pf.dst[i*3+2]-sz); }
  }
  if (dxs.length < minSamples) return null;
  const mx = _median(dxs), my = _median(dys), mz = _median(dzs);
  let sad = 0; for (let i = 0; i < dxs.length; i++) sad += Math.abs(dxs[i]-mx) + Math.abs(dys[i]-my) + Math.abs(dzs[i]-mz);
  if (sad / dxs.length > spread) return null;
  return [mx, my, mz];
}

/** Signed volume (divergence theorem) of a CLOSED triangle mesh. THE fidelity metric for this pipeline:
 *  correspondence-free, so unlike fit-to-nearest-surface it cannot be fooled by a vertex sitting on the
 *  WRONG body part (which reads ~0mm from a real surface), and unlike max-edge it actually discriminates
 *  (daniel-s0, the keeper clip, has a LARGER max edge than the coherent bakes). Measured on the
 *  shipped v3 bake: 0.63x true volume at f143 = 33 LITRES of the subject gone, while fit + edge both
 *  reported "CLEAN + FITS". Every negative verdict reached on fit/edge alone is suspect. */
export function meshVolume(pos, idx) {
  let v = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    v += (pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1])
        - pos[a + 1] * (pos[b] * pos[c + 2] - pos[b + 2] * pos[c])
        + pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c])) / 6;
  }
  return Math.abs(v);
}
