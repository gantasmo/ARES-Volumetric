/**
 * Multiview triangulation of SAM-3D-Body keypoints → clean metric skeleton in ARES space.
 *
 * Reads the pod output (fNNNNN_mv.json, from mv_render_infer.py): per frame, N views each with a known
 * 3x4 world→pixel projection P (defined in ARES mm) + that view's `pred_keypoints_2d`. For each of the 70
 * keypoints we robustly DLT-triangulate across the views → a 3D point directly in ARES millimetres (no
 * SAM-3D→ARES calibration, no scale guess — the cameras were in ARES space).
 *
 * Robust: full DLT, then drop the worst-reprojection view and re-solve until all residuals < τ or only
 * MIN_VIEWS remain (handles a view where a limb is self-occluded and the model hallucinated the keypoint).
 *
 * The reprojection-residual report is the FIRST correctness check: if residuals are ~1-3px the camera
 * convention (P vs keypoints_2d image frame) is right; if they're hundreds of px, keypoints_2d are likely
 * bbox-relative → re-run with --bbox-relative (offsets each view's kp by its bbox top-left).
 *
 *   node mv_triangulate.mjs <mvOutDir> [--tau-px 4] [--min-views 3] [--bbox-relative] [--out skel.json]
 */
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const A = process.argv.slice(2);
const dir = A[0];
const flag = (n, d) => { const i = A.indexOf(n); return i >= 0 ? A[i + 1] : d; };
const TAU = Number(flag("--tau-px", 4));
const MIN_VIEWS = Number(flag("--min-views", 3));
const BBOX_REL = A.includes("--bbox-relative");
const OUT = flag("--out", join(dir, "skeletons.json"));

/* ---- tiny linear algebra: symmetric-4x4 eigenvector for the smallest eigenvalue (Jacobi) ---- */
function smallestEigenvector4(M) {
  // M is 4x4 symmetric (row-major flat[16]); returns unit eigenvector of the smallest eigenvalue.
  const a = M.slice();
  const V = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const at = (m, i, j) => m[i * 4 + j];
  for (let sweep = 0; sweep < 50; sweep++) {
    // find largest off-diagonal
    let p = 0, q = 1, mx = 0;
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) { const v = Math.abs(at(a, i, j)); if (v > mx) { mx = v; p = i; q = j; } }
    if (mx < 1e-12) break;
    const app = at(a, p, p), aqq = at(a, q, q), apq = at(a, p, q);
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi), s = Math.sin(phi);
    for (let k = 0; k < 4; k++) {
      const akp = at(a, k, p), akq = at(a, k, q);
      a[k * 4 + p] = c * akp - s * akq; a[k * 4 + q] = s * akp + c * akq;
    }
    for (let k = 0; k < 4; k++) {
      const apk = at(a, p, k), aqk = at(a, q, k);
      a[p * 4 + k] = c * apk - s * aqk; a[q * 4 + k] = s * apk + c * aqk;
    }
    for (let k = 0; k < 4; k++) { const vkp = V[k * 4 + p], vkq = V[k * 4 + q]; V[k * 4 + p] = c * vkp - s * vkq; V[k * 4 + q] = s * vkp + c * vkq; }
  }
  let best = 0, bestVal = Infinity;
  for (let i = 0; i < 4; i++) { const d = at(a, i, i); if (d < bestVal) { bestVal = d; best = i; } }
  const v = [V[best], V[4 + best], V[8 + best], V[12 + best]];
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}

const mul34 = (P, X) => [P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3] * X[3],
                         P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7] * X[3],
                         P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11] * X[3]];
const flatP = (P) => [P[0][0], P[0][1], P[0][2], P[0][3], P[1][0], P[1][1], P[1][2], P[1][3], P[2][0], P[2][1], P[2][2], P[2][3]];

function reproj(P, X) { const x = mul34(P, [X[0], X[1], X[2], 1]); return [x[0] / x[2], x[1] / x[2]]; }
function residual(P, X, kp) { const [u, v] = reproj(P, X); return Math.hypot(u - kp[0], v - kp[1]); }

// DLT from a set of {P(flat12), kp[2]} → 3D point (homogeneous smallest-singular-vector via A^T A eigen).
function triangulate(obs) {
  const ATA = new Float64Array(16);
  const addRow = (row) => { for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) ATA[i * 4 + j] += row[i] * row[j]; };
  for (const { P, kp } of obs) {
    const [x, y] = kp;
    // x*(P2·X) - (P0·X) = 0 ;  y*(P2·X) - (P1·X) = 0
    addRow([x * P[8] - P[0], x * P[9] - P[1], x * P[10] - P[2], x * P[11] - P[3]]);
    addRow([y * P[8] - P[4], y * P[9] - P[5], y * P[10] - P[6], y * P[11] - P[7]]);
  }
  const Xh = smallestEigenvector4(ATA);
  const w = Xh[3];
  if (Math.abs(w) < 1e-12) return null;
  return [Xh[0] / w, Xh[1] / w, Xh[2] / w];
}

// robust: drop worst-residual view until all < TAU or MIN_VIEWS remain
function robustTriangulate(obs) {
  let cur = obs.slice();
  let X = triangulate(cur);
  while (cur.length > MIN_VIEWS) {
    const res = cur.map((o) => residual(o.P, X, o.kp));
    const wi = res.indexOf(Math.max(...res));
    if (res[wi] < TAU) break;
    cur = cur.filter((_, i) => i !== wi);
    X = triangulate(cur);
  }
  const res = cur.map((o) => residual(o.P, X, o.kp));
  return { X, nViews: cur.length, meanRes: res.reduce((a, b) => a + b, 0) / res.length, maxRes: Math.max(...res) };
}

/* ---------------------------------- main ---------------------------------- */
const files = readdirSync(dir).filter((f) => /^f\d{5}_mv\.json$/.test(f)).sort();
if (!files.length) { console.error(`no fNNNNN_mv.json in ${dir}`); process.exit(1); }
console.log(`[triangulate] ${files.length} frame(s) | τ=${TAU}px minViews=${MIN_VIEWS} bboxRel=${BBOX_REL}`);

const skeletons = {};
let allRes = [];
const perAz = {};   // diagnostic: mean reprojection residual grouped by camera azimuth (bad az → flips/outliers)
for (const file of files) {
  const rec = JSON.parse(readFileSync(join(dir, file), "utf8"));
  const views = rec.views.filter((v) => v.detected && v.keypoints_2d);
  const nkp = views.length ? views[0].keypoints_2d.length : 0;
  const joints = [], resPerKp = [], viewsPerKp = [];
  for (let k = 0; k < nkp; k++) {
    const obs = views.map((v) => {
      let kp = v.keypoints_2d[k];
      if (BBOX_REL && v.bbox) kp = [kp[0] + v.bbox[0], kp[1] + v.bbox[1]];
      return { P: flatP(v.P), kp, az: v.az };
    });
    if (obs.length < 2) { joints.push(null); resPerKp.push(NaN); viewsPerKp.push(obs.length); continue; }
    const { X, nViews, meanRes } = robustTriangulate(obs);
    joints.push(X); resPerKp.push(meanRes); viewsPerKp.push(nViews); allRes.push(meanRes);
    for (const o of obs) { const r = residual(o.P, X, o.kp); (perAz[o.az] ??= { s: 0, n: 0 }); perAz[o.az].s += r; perAz[o.az].n++; }
  }
  const valid = resPerKp.filter((r) => !isNaN(r));
  const mRes = valid.reduce((a, b) => a + b, 0) / (valid.length || 1);
  skeletons[rec.frame] = { joints, meanReprojPx: mRes, nViewsDetected: views.length, viewsPerKp };
  console.log(`  f${rec.frame}: ${views.length} views, ${joints.filter(Boolean).length}/${nkp} kp, mean reproj ${mRes.toFixed(2)}px`);
}

writeFileSync(OUT, JSON.stringify(skeletons));
const gRes = allRes.filter((r) => !isNaN(r));
console.log(`\n[triangulate] wrote ${OUT}`);
console.log(`   global mean reproj residual: ${(gRes.reduce((a, b) => a + b, 0) / gRes.length).toFixed(2)}px`);
console.log(`   ${gRes.filter((r) => r < TAU).length}/${gRes.length} keypoints under ${TAU}px`);
console.log(gRes.reduce((a, b) => a + b, 0) / gRes.length < 5
  ? "   ✅ residuals small — camera convention correct, skeletons are metric ARES-space."
  : "   ⚠ residuals large — try --bbox-relative, or check P/keypoint image-frame convention.");

// DIAGNOSTIC: mean reprojection residual by camera azimuth. If a few azimuths (e.g. back ~180°) are far
// worse, those views are the culprits (likely L/R-swapped) → reject/flip them. If it's flat, it's diffuse
// per-view pose disagreement (a harder problem than flips).
console.log("   per-azimuth mean reproj (px):", Object.keys(perAz).map(Number).sort((a, b) => a - b)
  .map((az) => `${az}°=${(perAz[az].s / perAz[az].n).toFixed(1)}`).join("  "));

// temporal jitter of the fused skeleton (success = well below the single-view 31mm/frame)
const frames = Object.keys(skeletons).map(Number).sort((a, b) => a - b);
let jit = [];
for (let i = 1; i < frames.length; i++) {
  const a = skeletons[frames[i - 1]].joints, b = skeletons[frames[i]].joints;
  if (frames[i] - frames[i - 1] !== 1) continue;
  let s = 0, n = 0;
  for (let k = 0; k < a.length; k++) if (a[k] && b[k]) { s += Math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1], a[k][2] - b[k][2]); n++; }
  if (n) jit.push(s / n);
}
if (jit.length) console.log(`   fused temporal jitter: mean ${(jit.reduce((a, b) => a + b, 0) / jit.length).toFixed(1)}mm/frame (single-view was ~31mm) over ${jit.length} adjacent pairs`);
