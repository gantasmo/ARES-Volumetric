/**
 * Skeleton-prior registration POC — the decisive test for the motion-robust fix.
 *
 * Compares, on the WORST high-motion span (run4/5, template f140 → f169, 30 frames):
 *   BASELINE  = current method: ARAP with previous-frame chain init (the 40mm gate abandons fast limbs).
 *   SKEL      = ARAP initialized by the SAM-3D-Body LBS pose-warp (limbs posed BEFORE the fit).
 *   SKEL-RAW  = the LBS warp alone, no ARAP (how close the skeleton gets on its own).
 *
 * Success metric (scoping report 2026-07-13): MAX per-vertex fit < 20mm across the 30-frame run, no AABB
 * blow-out. Baseline is expected to spike >100mm (matches the manifest: run4 f149 max-fit 172mm).
 *
 *   node skel-poc.mjs [--template 140] [--to 169] [--lambda 10] [--gate-mm 40] [--outer 10] [--K 4]
 *                     [--src .coherent-bake-work] [--poses sam3d-results/temporal-140-169/poses]
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";
import { TriangleGrid, buildWeldAdjacency, bboxDiag } from "../coherent-poc/lib.mjs";
import { buildCompact, arapRegisterFrame, scatterToFull, meshVolume } from "./arap.mjs";
import { loadPose, calibrate, buildSkinWeights, skelWarpInit } from "./skel-prior.mjs";

const A = process.argv.slice(2);
const flag = (n, d) => { const i = A.indexOf(n); return i >= 0 ? A[i + 1] : d; };
const tmplF = Number(flag("--template", 140)), toF = Number(flag("--to", 169));
const lambda = Number(flag("--lambda", 10)), gateMm = Number(flag("--gate-mm", 40)), outer = Number(flag("--outer", 10));
const K = Number(flag("--K", 4));
const smooth = Number(flag("--smooth", 0));       // centered temporal window half-width on joints+global_rot
const noGlobal = A.includes("--no-global");        // drop the (jittery) body global_rot; ARES owns global pose
const noRot = A.includes("--no-rot");              // pure translational LBS (ignore per-joint rotations)
const rotTranspose = A.includes("--rot-transpose");// transpose per-joint R (rotation-convention test)
const srcDir = flag("--src", ".coherent-bake-work");
const posesDir = flag("--poses", "sam3d-results/temporal-140-169/poses");
const fn = (i) => join(srcDir, `mesh-f${String(i).padStart(5, "0")}.obj`);

const transpose9 = (a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
// Preload + preprocess every pose in the run: optional per-joint transpose, temporal smoothing of joint
// positions + global_rot (kills the per-frame jitter that made the raw warp inject ~78mm at f141), no-global.
async function buildPoseCache() {
  const raw = new Map();
  for (let f = tmplF; f <= toF; f++) raw.set(f, await loadPose(posesDir, f));
  if (rotTranspose) for (const p of raw.values()) p.R = p.R.map(transpose9);
  const out = new Map();
  for (let f = tmplF; f <= toF; f++) {
    const p = raw.get(f);
    const jc = Float64Array.from(p.jc), gr = p.gr.slice();
    if (smooth > 0) {
      jc.fill(0); gr[0] = gr[1] = gr[2] = 0; let cnt = 0;
      for (let g = f - smooth; g <= f + smooth; g++) { const q = raw.get(g); if (!q) continue; cnt++; for (let i = 0; i < jc.length; i++) jc[i] += q.jc[i]; for (let k = 0; k < 3; k++) gr[k] += q.gr[k]; }
      for (let i = 0; i < jc.length; i++) jc[i] /= cnt; for (let k = 0; k < 3; k++) gr[k] /= cnt;
    }
    if (noGlobal) { gr[0] = gr[1] = gr[2] = 0; }
    out.set(f, { J: p.J, jc, R: p.R, gr });
  }
  return out;
}
const poses = await buildPoseCache();
console.log(`[skel-poc] opts: smooth=${smooth} noGlobal=${noGlobal} noRot=${noRot} rotTranspose=${rotTranspose}`);

function centroid(pos) { let x = 0, y = 0, z = 0; const n = pos.length / 3; for (let i = 0; i < pos.length; i += 3) { x += pos[i]; y += pos[i + 1]; z += pos[i + 2]; } return [x / n, y / n, z / n]; }
function templateMaxEdge(pos, idx) { let m = 0; for (let t = 0; t < idx.length; t += 3) { const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3; m = Math.max(m, Math.hypot(pos[a] - pos[b], pos[a + 1] - pos[b + 1], pos[a + 2] - pos[b + 2]), Math.hypot(pos[b] - pos[c], pos[b + 1] - pos[c + 1], pos[b + 2] - pos[c + 2]), Math.hypot(pos[c] - pos[a], pos[c + 1] - pos[a + 1], pos[c + 2] - pos[a + 2])); } return m; }
function maxEdge(pos, idx) { return templateMaxEdge(pos, idx); }
function fitStats(pos, grid) {
  const n = pos.length / 3, tmp = new Float32Array(3); let sum = 0, mx = 0, over5 = 0; const ds = new Float64Array(n);
  for (let i = 0; i < n; i++) { const d = Math.sqrt(grid.nearestInto(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], tmp, 0)); ds[i] = d; sum += d; if (d > mx) mx = d; if (d > 5) over5++; }
  const sorted = Array.from(ds).sort((a, b) => a - b);
  return { mean: sum / n, p95: sorted[Math.floor(0.95 * n)], max: mx, pctOver5: 100 * over5 / n };
}

async function loadFrame(i) { const o = parseObj(await readFile(fn(i), "utf8")); const diag = bboxDiag(o.positions) || 1; const grid = new TriangleGrid(o.positions, o.indices, o.uvs, diag / 48); return { o, grid, ctr: centroid(o.positions) }; }

const tmpl = parseObj(await readFile(fn(tmplF), "utf8"));
const weld = buildWeldAdjacency(tmpl.positions, tmpl.indices);
const cmp = buildCompact(tmpl.positions, weld);
const restMaxE = maxEdge(tmpl.positions, tmpl.indices);

const tmplPose = poses.get(tmplF);
const calib = calibrate(tmplPose, tmpl.positions);
const skin = buildSkinWeights(cmp.restC, cmp.nc, calib, tmplPose, K);
console.log(`[skel-poc] template f${tmplF}: ${tmpl.positions.length / 3} verts (${cmp.nc} welded), maxEdge ${restMaxE.toFixed(1)}mm`);
console.log(`[skel-poc] calib: scale ${calib.s.toFixed(1)} mm/m, joints-inside-mesh ${(calib.insideFrac * 100).toFixed(0)}%, Rmap det ok | K=${K} λ=${lambda} gate=${gateMm} outer=${outer}`);
console.log(`[skel-poc] run f${tmplF}→f${toF} (${toF - tmplF} frames). Success = MAX-fit < 20mm.\n`);
console.log("frame |     BASELINE (chain-ARAP)      |      SKEL-RAW (warp only)      |     SKEL (warp+ARAP)");
console.log("      | max   mean  o5%   edge         | max   mean  o5%                | max   mean  o5%   edge");

let baseInit = Float64Array.from(cmp.restC);   // baseline chains from previous ARAP result
const baseOpts = { lambda, gateMm, outer, quats: null };
const agg = { base: { max: 0, o5: 0, vol: 9 }, skel: { max: 0, o5: 0, vol: 9 }, raw: { max: 0, vol: 9 } };
let prevPose = tmplPose;

for (let f = tmplF + 1; f <= toF; f++) {
  const { o: srcMesh, grid, ctr } = await loadFrame(f);
  const framePose = poses.get(f);

  // BASELINE: chain init from previous baseline result.
  const basePC = arapRegisterFrame(cmp, grid, baseInit, baseOpts);
  baseInit = basePC;
  const basePos = scatterToFull(basePC, weld);
  const bf = fitStats(basePos, grid), be = maxEdge(basePos, tmpl.indices) / restMaxE;

  // SKEL: LBS-warp the template into this frame's pose (absolute, not chained), prealign to source, then ARAP.
  const warpC = skelWarpInit(cmp.restC, cmp.nc, calib, skin, tmplPose, framePose, { srcCentroid: ctr, noRot });
  const rawPos = scatterToFull(warpC, weld);
  const rf = fitStats(rawPos, grid);
  const skelPC = arapRegisterFrame(cmp, grid, warpC, { lambda, gateMm, outer, quats: null });
  const skelPos = scatterToFull(skelPC, weld);
  const sf = fitStats(skelPos, grid), se = maxEdge(skelPos, tmpl.indices) / restMaxE;
  // VOLUME vs the true source frame — the only metric here that can SEE implosion. fit/edge both
  // report 'clean' on a mesh missing a third of its volume, so a verdict from them alone is unsafe.
  const srcVol = meshVolume(srcMesh.positions, srcMesh.indices);
  const bv = meshVolume(basePos, tmpl.indices) / srcVol;
  const rv = meshVolume(rawPos, tmpl.indices) / srcVol;
  const sv = meshVolume(skelPos, tmpl.indices) / srcVol;

  agg.base.max = Math.max(agg.base.max, bf.max); agg.base.o5 = Math.max(agg.base.o5, bf.pctOver5);
  agg.skel.max = Math.max(agg.skel.max, sf.max); agg.skel.o5 = Math.max(agg.skel.o5, sf.pctOver5);
  agg.raw.max = Math.max(agg.raw.max, rf.max);
  agg.base.vol = Math.min(agg.base.vol, bv); agg.skel.vol = Math.min(agg.skel.vol, sv); agg.raw.vol = Math.min(agg.raw.vol, rv);
  const row = (x) => x.toFixed(x < 100 ? 1 : 0).padStart(5);
  if (f % 3 === 0 || f === toF) console.log(
    `f${f}  | ${row(bf.max)} ${row(bf.mean)} ${bf.pctOver5.toFixed(0).padStart(4)} ${be.toFixed(2)}x        | ` +
    `${row(rf.max)} ${row(rf.mean)} ${rf.pctOver5.toFixed(0).padStart(4)}          | ` +
    `${row(sf.max)} ${row(sf.mean)} ${sf.pctOver5.toFixed(0).padStart(4)} ${se.toFixed(2)}x` +
    `  || VOL base ${bv.toFixed(3)} raw ${rv.toFixed(3)} skel ${sv.toFixed(3)}`);
  prevPose = framePose;
}

console.log(`\n[skel-poc] RUN MAX over ${toF - tmplF} frames:`);
console.log(`   BASELINE  max-fit ${agg.base.max.toFixed(1)}mm  worst-frame pctOver5 ${agg.base.o5.toFixed(0)}%`);
console.log(`   SKEL-RAW  max-fit ${agg.raw.max.toFixed(1)}mm  (skeleton warp alone, no ARAP)`);
console.log(`   SKEL      max-fit ${agg.skel.max.toFixed(1)}mm  worst-frame pctOver5 ${agg.skel.o5.toFixed(0)}%`);
console.log(`   --- WORST VOLUME (1.000 = perfect; the metric max-fit is BLIND to) ---`);
console.log(`   BASELINE  worst vol ${agg.base.vol.toFixed(3)}x`);
console.log(`   SKEL-RAW  worst vol ${agg.raw.vol.toFixed(3)}x`);
console.log(`   SKEL      worst vol ${agg.skel.vol.toFixed(3)}x   => ${agg.skel.vol > agg.base.vol ? "SKEL BEATS BASELINE on volume" : "no volume win"}`);
const verdict = agg.skel.max < 20 ? "✅ PASS — MAX < 20mm, skeleton prior FIXES the divergence"
  : agg.skel.max < agg.base.max * 0.5 ? "⚠ IMPROVED but not <20mm — tune (K/lambda/prealign) or add flow residual"
  : "❌ NO WIN — check calibration (Rmap/scale) or rotation convention";
console.log(`   VERDICT: ${verdict}`);
console.log(`   (baseline→skel MAX ${agg.base.max.toFixed(0)}→${agg.skel.max.toFixed(0)}mm)`);
