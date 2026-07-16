/**
 * Task N (Phase 2): FULL-CLIP coherent-GOP runner with adaptive cuts.
 *
 * Usage: node coherent-clip.mjs <src-frames-dir> <out-frames-dir> [--gop 30] [--err-cut 1.8]
 *        [--min-run 1] [--register-only] [--rounds 3] [--smooth-iters 3] [--rings 2]
 *        [--fallback-mm 3] [--max-frames N] [--ckpt <dir>] [--workers N]
 *
 * --min-run N       hold a run open for >= N frames before an adaptive error-cut is allowed
 *                   (kills the fast-motion 1-frame-run cascade that fragments the atlas).
 * --register-only   run Pass 1 only, write the run distribution, skip the multi-hour bake.
 *
 * Two-pass design:
 *  Pass 1 (sequential, single-threaded — chain registration has an inherent frame-to-frame
 *  dependency so it cannot be parallelized across a run; different runs also can't be
 *  parallelized ahead of time because a run's END is only known once its error is measured):
 *  walks the whole clip frame-by-frame, chain-registering onto each frame's source mesh
 *  (the POC's winning "chain" variant). Tracks
 *  per-frame registration error (mean/p95/max). If p95 exceeds --err-cut, or the run reaches
 *  --gop frames, the run ends: the run's frames + per-vertex seed-triangles (for Phase 2's
 *  fast bake) are checkpointed to disk, and (for an error cut) the triggering frame becomes a
 *  FRESH template (intra) for the next run.
 *
 *  Pass 2 (parallel across worker_threads, run-by-run): for each run, rasterizes the template's
 *  UV bary map ONCE (not per frame, not per worker — shared to all workers for that run via
 *  SharedArrayBuffer), then bakes every non-template frame's atlas with the seeded local-search
 *  bakeFrameFast, writing directly into <out-frames-dir> under the clip's GLOBAL frame numbering
 *  (so consecutive runs sit side by side in one contiguous frames-dir, ready for
 *  `cli.js encode --gop <same N>`, which independently re-chunks by frame count; adaptive
 *  cuts that don't land on a multiple of --gop simply straddle chunk boundaries).
 *
 * Writes coherent-manifest.json (per-run template/cuts/error curves/bake timings + boundary
 * measurements) to <out-frames-dir>/../coherent-manifest.json.
 */
import { readFile, writeFile, mkdir, copyFile, readdir } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { cpus } from "node:os";
import { join, dirname } from "node:path";
import { performance } from "node:perf_hooks";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";
import {
  TriangleGrid, buildWeldAdjacency, registerChainFrame, measureRegError,
  rasterizeTemplateBary, dilate, buildGutterMap, decodePng, mkScratch, bboxDiag,
  writeObjText,
} from "../coherent-poc/lib.mjs";
import { buildCompact, arapRegisterFrame, scatterToFull, buildPairFlow, flowDisp } from "./arap.mjs";

function flag(args, name) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}

// HONEST progress: rate is measured from work actually completed so far (no fabricated %); ETA is
// remaining/measured-rate. Emitted as a parseable [PROGRESS] {json} line the app tails over SSE.
function emitProgress(stage, done, total, t0) {
  const elapsedS = (performance.now() - t0) / 1000;
  const rate = done > 0 ? done / elapsedS : 0;          // items (frames) per second, measured
  const etaS = rate > 0 ? (total - done) / rate : 0;
  console.log(`[PROGRESS] ${JSON.stringify({ stage, done, total, pct: total ? Math.round((100 * done) / total) : 0, elapsedS: Math.round(elapsedS), etaS: Math.round(etaS), fps: Number(rate.toFixed(3)) })}`);
}

async function discoverFrames(srcDir) {
  const files = await readdir(srcDir);
  const meshRe = /^mesh-f(\d+)\.obj$/i;
  const nums = [];
  let pad = 5;
  for (const f of files) {
    const m = meshRe.exec(f);
    if (m) { nums.push(Number(m[1])); pad = m[1].length; }
  }
  nums.sort((a, b) => a - b);
  const atlasOk = (i) => files.includes(`atlas-f${String(i).padStart(pad, "0")}.png`);
  for (const n of nums) if (!atlasOk(n)) throw new Error(`missing atlas for mesh frame ${n}`);
  return { indices: nums, pad };
}

function writePositionsBin(path, positionsList) {
  const vcount = positionsList[0].length / 3;
  const buf = Buffer.alloc(8 + positionsList.length * vcount * 3 * 4);
  buf.writeUInt32LE(positionsList.length, 0);
  buf.writeUInt32LE(vcount, 4);
  let off = 8;
  for (const p of positionsList) { Buffer.from(p.buffer, p.byteOffset, p.byteLength).copy(buf, off); off += p.byteLength; }
  return writeFile(path, buf);
}
function writeSeedTrisBin(path, seedList) {
  const vcount = seedList[0].length;
  const buf = Buffer.alloc(8 + seedList.length * vcount * 4);
  buf.writeUInt32LE(seedList.length, 0);
  buf.writeUInt32LE(vcount, 4);
  let off = 8;
  for (const s of seedList) { Buffer.from(s.buffer, s.byteOffset, s.byteLength).copy(buf, off); off += s.byteLength; }
  return writeFile(path, buf);
}

// Unique template edges + rest lengths, for TEAR detection (see --stretch-cut). A vertex that snaps
// onto the wrong source surface reads ~0mm nearest-surface reg-error (it IS on a surface) while its
// fixed-topology edge to a neighbor still on the RIGHT part stretches far past rest — the reg-error
// metric is structurally blind to this; edge stretch is not.
function buildTemplateEdges(positions, indices) {
  const seen = new Set();
  const eiArr = [], ejArr = [];
  for (let t = 0; t < indices.length; t += 3) {
    const v = [indices[t], indices[t + 1], indices[t + 2]];
    for (let s = 0; s < 3; s++) {
      let a = v[s], b = v[(s + 1) % 3];
      if (a > b) { const tmp = a; a = b; b = tmp; }
      const key = a * 2097152 + b; // < 2^21 verts, stays a safe integer
      if (seen.has(key)) continue;
      seen.add(key); eiArr.push(a); ejArr.push(b);
    }
  }
  const ei = Int32Array.from(eiArr), ej = Int32Array.from(ejArr);
  const rest = new Float32Array(ei.length);
  let maxRest = 0;
  for (let e = 0; e < ei.length; e++) {
    const a = ei[e] * 3, b = ej[e] * 3;
    const L = Math.hypot(positions[a] - positions[b], positions[a + 1] - positions[b + 1], positions[a + 2] - positions[b + 2]);
    rest[e] = L || 1e-6; if (L > maxRest) maxRest = L;
  }
  return { ei, ej, rest, maxRest: maxRest || 1 };
}
// Tear metric for a registered frame. The shred is a LONG ABSOLUTE edge (a 16mm edge snapping to
// 490mm across the body), so the detector is maxEdge / template-maxEdge — NOT a per-edge stretch
// ratio (that's dominated by short edges whose length naturally varies 2-3x frame to frame and would
// false-trigger on every frame). Normal frames sit ~1.0-1.5x; real tears are 2.5-4.5x. Also reports
// the worst per-edge ratio for diagnostics only.
function edgeStretch(positions, edges) {
  const { ei, ej, rest, maxRest } = edges;
  let maxAbs = 0, maxPerEdge = 0;
  for (let e = 0; e < ei.length; e++) {
    const a = ei[e] * 3, b = ej[e] * 3;
    const L = Math.hypot(positions[a] - positions[b], positions[a + 1] - positions[b + 1], positions[a + 2] - positions[b + 2]);
    if (L > maxAbs) maxAbs = L;
    const r = L / rest[e]; if (r > maxPerEdge) maxPerEdge = r;
  }
  return { maxStretch: maxAbs / maxRest, maxEdgeAbs: maxAbs, maxPerEdge };
}

async function main() {
  const args = process.argv.slice(2);
  const srcDir = args[0];
  const outDir = args[1];
  if (!srcDir || !outDir) {
    console.error("usage: node coherent-clip.mjs <src-frames-dir> <out-frames-dir> [--gop 30] [--err-cut 1.8] [--min-run 1] [--register-only] [--rounds 3] [--max-rounds 16] [--eps-mm 0.4] [--smooth-iters 3] [--rings 2] [--fallback-mm 3] [--gpu] [--python python] [--max-frames N] [--ckpt dir] [--workers N]");
    process.exit(1);
  }
  const GOP = Number(flag(args, "--gop") ?? 30);
  const ERR_CUT_MM = Number(flag(args, "--err-cut") ?? 1.8); // derived: 4x POC's chain p95 mean (0.442mm) ~1.77mm, rounded
  const ROUNDS = Number(flag(args, "--rounds") ?? 3);        // minimum registration rounds (floor)
  // Adaptive convergence past the floor: keep iterating (up to --max-rounds) until the per-round
  // max vertex move < --eps-mm. Fixes the fast-motion off-surface spike that fixed 3 rounds left
  // (frame 146: 73.6mm -> ~3mm) while slow frames still stop at ROUNDS. See registerChainFrame.
  const MAX_ROUNDS = Number(flag(args, "--max-rounds") ?? 16);
  const EPS_MM = Number(flag(args, "--eps-mm") ?? 0.4);
  const SMOOTH_ITERS = Number(flag(args, "--smooth-iters") ?? 3);
  const RINGS = Number(flag(args, "--rings") ?? 2);
  const FALLBACK_MM = Number(flag(args, "--fallback-mm") ?? 3);
  const MAX_FRAMES = flag(args, "--max-frames") ? Number(flag(args, "--max-frames")) : Infinity;
  const NWORKERS = Number(flag(args, "--workers") ?? cpus().length);
  // --gpu: run Pass 2 on the GPU (NVIDIA Warp) instead of CPU worker_threads — same nearest-source
  // + sample math, ~100x faster at 2048 (validated 42.4 dB PSNR vs CPU on the occupied atlas). One
  // python/gpu_bake.py process per run reads the run's positions.bin + template OBJ and writes atlases.
  const GPU_BAKE = args.includes("--gpu");
  const PYTHON = flag(args, "--python") ?? "python";
  // Visual verdict 2026-07-13: exact (POC bakeFrame, true global nearest — what coherent-A
  // used) is the DEFAULT; "fast" (seeded ≤fallbackMm-acceptance search) smeared every frame of
  // the full-clip bake and is kept only as an explicit preview knob (--bake fast).
  const BAKE_MODE = (flag(args, "--bake") ?? "exact") === "fast" ? "fast" : "exact";
  // --min-run N: suppress an adaptive (error) cut while the current run is shorter than N frames.
  // Fixes the fragmentation cascade: at a tight --err-cut, a barely-over-threshold frame in a
  // fast-motion stretch cuts to a fresh template, whose very next frame ALSO exceeds threshold →
  // a run of 1-frame runs (per-frame atlas repacks = the reported texture scramble).
  // Holding the run open until it reaches N frames keeps ONE stable atlas across the motion,
  // which is exactly what coherent-A does (a single 30-frame run).
  const MIN_RUN = Math.max(1, Number(flag(args, "--min-run") ?? 1));
  // --stretch-cut R: end the run when registration TEARS the fixed template — any edge stretched
  // > R x its template rest length. This is the tear detector the p95 reg-error cut is BLIND to
  // (measured 2026-07-13: end-of-GOP frames drifted far enough that leg/arm template verts snapped
  // onto the wrong source surface — reg-error 0.067mm "perfect" while a 16mm edge stretched to
  // 490mm, 30x). Cutting at the tear drops to a fresh CLEAN native template, so no emitted frame
  // carries a wrong-surface snap. This is what actually fixes the shred/melt, not more rounds.
  const STRETCH_CUT = Number(flag(args, "--stretch-cut") ?? 2.2);
  // --arap: register with As-Rigid-As-Possible instead of nearest-point + Taubin. The RIGHT fix for
  // the wrong-surface snap (validated 2026-07-14): rigidity is measured vs the template so articulation
  // (rotation) is free but tearing (edge stretch) costs energy — a vertex physically can't run to the
  // wrong surface. Measured on the chains that tore to 4.5x: maxEdge stays ~1.0x, fit mean <1.2mm.
  // λ balances fit vs rigidity (10 = clean + sub-mm); gate downweights wrong/occluded matches.
  const ARAP = args.includes("--arap");
  const ARAP_LAMBDA = Number(flag(args, "--arap-lambda") ?? 10);
  const ARAP_GATE = Number(flag(args, "--arap-gate") ?? 40);
  const ARAP_OUTER = Number(flag(args, "--arap-outer") ?? 10);
  // --flow <dir>: RAFT scene-flow samples (pair_NNNNN.json from tools/sam3d/mv_flow_capture.py).
  // Appearance-tracked correspondence: where flow disagrees with nearest-point by >flowSnap, that IS a
  // wrong-surface snap, and flow wins. Measured on f140-156 (arap-poc, same settings): worst VOLUME
  // 0.646x -> 0.860x true. NOTE the fit/edge metrics CANNOT see this (164.6->163.0mm, both "CLEAN").
  const FLOW_DIR = flag(args, "--flow") ?? null;
  const FLOW_R = Number(flag(args, "--flow-r") ?? 35), FLOW_MIN = Number(flag(args, "--flow-min") ?? 3);
  const FLOW_SNAP = Number(flag(args, "--flow-snap") ?? 30), FLOW_SPREAD = Number(flag(args, "--flow-spread") ?? 20);
  const pairFlow = {};   // filled after clipFrames is known (below)
  // --register-only: run Pass 1 (registration + cut decisions) and write the run distribution,
  // then stop BEFORE the multi-hour texture bake — a fast (~minutes) proof that the cut params
  // behave before committing the machine to the full bake.
  const REGISTER_ONLY = args.includes("--register-only");
  const DILATE_RADIUS = 2;
  const CKPT_DIR = flag(args, "--ckpt") ?? join(dirname(outDir), `.coherent-ckpt-${Date.now()}`);

  await mkdir(outDir, { recursive: true });
  await mkdir(CKPT_DIR, { recursive: true });

  const { indices, pad } = await discoverFrames(srcDir);
  const clipFrames = indices.slice(0, Math.min(indices.length, MAX_FRAMES));
  const fname = (i) => `mesh-f${String(i).padStart(pad, "0")}.obj`;
  const atlasName = (i) => `atlas-f${String(i).padStart(pad, "0")}.png`;

  // Load RAFT scene-flow pairs for whichever frames we actually have samples for. Frames with no pair
  // silently fall back to gated nearest-point, so a partial-coverage flow dir is valid and additive.
  if (FLOW_DIR) {
    for (const f of clipFrames) {
      try { const d = JSON.parse(readFileSync(join(FLOW_DIR, `pair_${String(f).padStart(pad, "0")}.json`), "utf8")); pairFlow[f] = buildPairFlow(d.samples, FLOW_R); } catch { /* pair absent -> nearest-point */ }
    }
    console.log(`[coherent-clip] flow: ${Object.keys(pairFlow).length} pair(s) from ${FLOW_DIR} (r=${FLOW_R} min=${FLOW_MIN} snap=${FLOW_SNAP} spread=${FLOW_SPREAD}); frames without a pair use nearest-point`);
  }

  console.log(`[coherent-clip] ${clipFrames.length} frame(s) from ${srcDir} (register=${ARAP ? `ARAP λ=${ARAP_LAMBDA} gate=${ARAP_GATE} outer=${ARAP_OUTER}` : "nearest+Taubin"} gop=${GOP} errCut=${ERR_CUT_MM}mm stretchCut=${STRETCH_CUT}x minRun=${MIN_RUN} bake=${BAKE_MODE} workers=${NWORKERS}${REGISTER_ONLY ? " REGISTER-ONLY" : ""}${BAKE_MODE === "fast" ? ` rings=${RINGS} fallbackMm=${FALLBACK_MM}` : ""})`);

  /* ------------------------------- PASS 1: registration + cuts ------------------------------- */
  const t0 = performance.now();
  const runs = [];       // { templateGlobalFrame, globalFrames[], positions[], vertexSeedTri[], errors[], cutReason, cutP95 }
  const cutsLog = [];
  let cur = null;

  function startRun(globalFrameIdx, mesh) {
    const weld = buildWeldAdjacency(mesh.positions, mesh.indices);
    cur = {
      templateGlobalFrame: globalFrameIdx,
      globalFrames: [globalFrameIdx],
      positions: [mesh.positions.slice()],
      vertexSeedTri: [new Int32Array(mesh.positions.length / 3).fill(-1)],
      errors: [{ frame: globalFrameIdx, mean: 0, p95: 0, max: 0, pctOver5mm: 0 }],
      bakeMs: [],
      templateIndices: mesh.indices,
      templateUvs: mesh.uvs,
      weld,
      edges: buildTemplateEdges(mesh.positions, mesh.indices),
      // ARAP state: compacted weld graph (built once per template) + warm-start position/rotation
      // fields threaded across the chain (each frame starts from the previous frame's result).
      cmp: ARAP ? buildCompact(mesh.positions, weld) : null,
      arapInitC: null,   // set lazily below (Float64 compacted positions, = template on frame 0)
      arapQuats: null,
      cutReason: null,
      cutP95: null,
      registerMs: 0,
    };
    if (ARAP) cur.arapInitC = Float64Array.from(cur.cmp.restC);
  }

  let firstMesh = parseObj(await readFile(join(srcDir, fname(clipFrames[0])), "utf8"));
  startRun(clipFrames[0], firstMesh);

  for (let k = 1; k < clipFrames.length; k++) {
    const gi = clipFrames[k];
    const tReg0 = performance.now();
    const mesh = parseObj(await readFile(join(srcDir, fname(gi)), "utf8"));
    const diag = bboxDiag(mesh.positions) || 1;
    const grid = new TriangleGrid(mesh.positions, mesh.indices, mesh.uvs, diag / 48);
    const chainPrev = cur.positions[cur.positions.length - 1];
    let positions, vertexSeedTri, error;
    if (ARAP) {
      const opts = { lambda: ARAP_LAMBDA, gateMm: ARAP_GATE, outer: ARAP_OUTER, quats: cur.arapQuats, flowSnap: FLOW_SNAP };
      // flow for the pair (previous clip frame -> this one), sampled at the PREVIOUS registered
      // positions — that is where the scene-flow samples live. Same as arap-poc's validated path.
      // Keyed by the pair's FROM frame (mv_flow_capture writes pair_<from>.json), NOT by loop index.
      const pf = pairFlow[clipFrames[k - 1]];
      if (pf) {
        const nc = cur.cmp.nc, ft = new Float64Array(nc * 3), fc = new Float64Array(nc);
        const init = cur.arapInitC;
        let cov = 0;
        for (let i = 0; i < nc; i++) {
          const x = init[i * 3], y = init[i * 3 + 1], z = init[i * 3 + 2];
          const disp = flowDisp(pf, x, y, z, FLOW_R, FLOW_MIN, FLOW_SPREAD);
          if (disp) { ft[i * 3] = x + disp[0]; ft[i * 3 + 1] = y + disp[1]; ft[i * 3 + 2] = z + disp[2]; fc[i] = 1; cov++; }
        }
        opts.flowTarget = ft; opts.flowConf = fc;
        if (gi % 10 === 0) console.log(`[coherent-clip]   f${gi}: flow-covered ${Math.round(100 * cov / nc)}%`);
      }
      const pC = arapRegisterFrame(cur.cmp, grid, cur.arapInitC, opts);
      cur.arapInitC = pC; cur.arapQuats = opts.quats;               // warm-start the next frame
      positions = scatterToFull(pC, cur.weld);
      error = measureRegError(positions, grid);
      vertexSeedTri = new Int32Array(positions.length / 3).fill(-1); // exact/GPU bake don't use seeds
    } else {
      ({ positions, vertexSeedTri, error } = registerChainFrame(chainPrev, grid, cur.weld, ROUNDS, SMOOTH_ITERS, { maxRounds: MAX_ROUNDS, epsMm: EPS_MM }));
    }
    cur.registerMs += performance.now() - tReg0;
    emitProgress("register", k, clipFrames.length - 1, t0);

    // TEAR detection: does this registered frame stretch the fixed template past breaking? This is
    // the criterion the blind p95 reg-error cut missed entirely (all 10 GOPs hit natural-cap while
    // end-of-GOP frames tore to 30x rest — see report). Cut here -> the torn registered frame is
    // DISCARDED and its NATIVE mesh becomes a fresh clean template (startRun uses `mesh`, not the
    // registered `positions`), so every emitted frame stays under STRETCH_CUT.
    const stretch = edgeStretch(positions, cur.edges);
    const tear = stretch.maxStretch > STRETCH_CUT;
    // Suppress the cut while the run is younger than MIN_RUN: a fresh template cannot rescue a
    // fast-motion frame (its own next frame re-exceeds threshold), so cutting there only
    // fragments the atlas. Keep chaining through the motion instead (accept the registration error).
    // The p95 reg-ERROR cut is for the old nearest-point path (drift = quality loss). ARAP deliberately
    // trades a little surface fit for rigidity (that's what kills tearing), so its p95 runs higher in
    // fast motion WITHOUT any tearing — applying the fit-cut there just fragments long runs into texture
    // keyframes for no quality gain. Under ARAP, rely on the stretch/tear safety net only.
    const isCut = (((!ARAP && error.p95 > ERR_CUT_MM) || tear)) && cur.globalFrames.length >= MIN_RUN;
    if (isCut) {
      const reason = tear ? "tear" : "adaptive";
      cutsLog.push({ atGlobalFrame: gi, p95: error.p95, mean: error.mean, max: error.max, maxStretch: stretch.maxStretch, reason, thresholdMm: ERR_CUT_MM });
      cur.cutReason = reason; cur.cutP95 = error.p95;
      runs.push(cur);
      console.log(`[coherent-clip] CUT at frame ${gi} (${reason}): p95=${error.p95.toFixed(3)}mm maxStretch=${stretch.maxStretch.toFixed(2)}x — new template (run ${runs.length} was ${cur.globalFrames.length} frames)`);
      startRun(gi, mesh);
      continue;
    }

    cur.globalFrames.push(gi);
    cur.positions.push(positions);
    cur.vertexSeedTri.push(vertexSeedTri);
    cur.errors.push({ frame: gi, ...error, maxStretch: stretch.maxStretch, maxEdgeAbs: stretch.maxEdgeAbs });

    if (cur.globalFrames.length >= GOP) {
      cur.cutReason = "natural-cap";
      runs.push(cur);
      console.log(`[coherent-clip] run ${runs.length} complete (natural cap): frames ${cur.templateGlobalFrame}..${gi} (${cur.globalFrames.length})`);
      if (k + 1 < clipFrames.length) {
        const nextMesh = parseObj(await readFile(join(srcDir, fname(clipFrames[k + 1])), "utf8"));
        // consume the next frame's template start right away, advance k
        k++;
        startRun(clipFrames[k], nextMesh);
      } else {
        cur = null;
      }
    }
  }
  if (cur && (runs.length === 0 || runs[runs.length - 1] !== cur)) {
    cur.cutReason = "end-of-clip";
    runs.push(cur);
    console.log(`[coherent-clip] final run ${runs.length}: frames ${cur.templateGlobalFrame}..${cur.globalFrames[cur.globalFrames.length - 1]} (${cur.globalFrames.length})`);
  }

  const tRegDone = performance.now();
  const registerWallMs = tRegDone - t0;
  console.log(`[coherent-clip] PASS 1 (registration) done: ${runs.length} run(s), ${cutsLog.length} adaptive cut(s), ${(registerWallMs / 1000).toFixed(1)}s`);

  if (REGISTER_ONLY) {
    const dist = runs.map((r, ri) => ({
      runIndex: ri,
      templateGlobalFrame: r.templateGlobalFrame,
      startGlobalFrame: r.globalFrames[0],
      endGlobalFrame: r.globalFrames[r.globalFrames.length - 1],
      length: r.globalFrames.length,
      cutReason: r.cutReason,
      cutP95: r.cutP95,
      p95Max: Math.max(...r.errors.map((e) => e.p95)),
      p95Mean: r.errors.reduce((s, e) => s + e.p95, 0) / r.errors.length,
      stretchMax: Math.max(0, ...r.errors.map((e) => e.maxStretch || 0)),
      stretchP999Max: Math.max(0, ...r.errors.map((e) => e.p999Stretch || 0)),
    }));
    const lengths = dist.map((d) => d.length);
    const single = lengths.filter((l) => l === 1).length;
    const summary = {
      registerOnly: true,
      clip: { srcDir, totalFrames: clipFrames.length, gop: GOP, errCutMm: ERR_CUT_MM, minRun: MIN_RUN },
      runCount: runs.length,
      adaptiveCuts: cutsLog.length,
      singleFrameRuns: single,
      longestRun: Math.max(...lengths),
      meanRunLength: lengths.reduce((s, l) => s + l, 0) / lengths.length,
      runLengthHistogram: lengths.reduce((h, l) => ((h[l] = (h[l] || 0) + 1), h), {}),
      runs: dist,
      registerMs: registerWallMs,
    };
    const outPath = join(dirname(outDir), "coherent-register-only.json");
    await writeFile(outPath, JSON.stringify(summary, null, 2));
    console.log(`[coherent-clip] REGISTER-ONLY: ${runs.length} runs, ${single} single-frame, ${cutsLog.length} adaptive cuts. Lengths: ${lengths.join(",")}`);
    console.log(`[coherent-clip] wrote ${outPath}`);
    return;
  }

  /* -------------------------------- PASS 2: parallel bake per run -------------------------------- */
  const tBake0 = performance.now();
  const totalBake = runs.reduce((s, r) => s + Math.max(0, r.globalFrames.length - 1), 0);
  let bakedDone = 0;
  const runManifests = [];
  for (let ri = 0; ri < runs.length; ri++) {
    const run = runs[ri];
    const runCkptDir = join(CKPT_DIR, `run-${ri}`);
    await mkdir(runCkptDir, { recursive: true });
    await writePositionsBin(join(runCkptDir, "positions.bin"), run.positions);
    await writeSeedTrisBin(join(runCkptDir, "vertex-seed-tris.bin"), run.vertexSeedTri);

    // Template frame: copy verbatim (no bake).
    await copyFile(join(srcDir, atlasName(run.templateGlobalFrame)), join(outDir, atlasName(run.templateGlobalFrame)));
    await writeFile(join(outDir, fname(run.templateGlobalFrame)), writeObjText(run.positions[0], run.templateUvs, run.templateIndices));

    const nonTemplateCount = run.globalFrames.length - 1;
    if (nonTemplateCount <= 0) {
      runManifests.push(runSummary(run, ri, 0, []));
      continue;
    }

    if (GPU_BAKE) {
      // GPU Pass 2: one python/Warp process bakes the whole run. It reads the positions.bin +
      // template OBJ written just above, writes atlas+OBJ per frame to outDir, and prints its own
      // honest [PROGRESS] (stage bake-gpu, offset by bakedDone) which flows straight to the log.
      const gpuScript = fileURLToPath(new URL("./gpu_bake.py", import.meta.url));
      const tRunGpu0 = performance.now();
      await new Promise((resolve, reject) => {
        const g = spawn(PYTHON, [gpuScript,
          "--template", join(outDir, fname(run.templateGlobalFrame)),
          "--positions", join(runCkptDir, "positions.bin"),
          "--src", srcDir, "--first", String(run.templateGlobalFrame),
          "--out", outDir, "--pad", String(pad),
          "--stage-total", String(totalBake), "--stage-done", String(bakedDone),
        ], { stdio: ["ignore", "inherit", "inherit"] });
        g.on("close", (c) => (c === 0 ? resolve() : reject(new Error(`gpu_bake (run ${ri}) exited ${c}`))));
        g.on("error", reject);
      });
      bakedDone += nonTemplateCount;
      const runGpuMs = performance.now() - tRunGpu0;
      console.log(`[coherent-clip] run ${ri} GPU bake done: ${nonTemplateCount} frame(s) in ${(runGpuMs / 1000).toFixed(1)}s`);
      runManifests.push(runSummary(run, ri, runGpuMs, []));
      continue;
    }

    // Rasterize template UV bary map ONCE for this run/template, SharedArrayBuffer-backed so
    // every worker reads the SAME memory (no per-worker recompute, no per-worker copy).
    const atlas0 = await readFile(join(srcDir, atlasName(run.templateGlobalFrame)));
    const dv = new DataView(atlas0.buffer, atlas0.byteOffset, atlas0.byteLength);
    const width = dv.getUint32(16, false), height = dv.getUint32(20, false);
    const tRaster0 = performance.now();
    const { triId, baryA, baryB, baryC } = rasterizeTemplateBary(run.templateUvs, run.templateIndices, width, height);
    const occMask = new Uint8Array(width * height);
    for (let p = 0; p < triId.length; p++) if (triId[p] >= 0) occMask[p] = 1;
    const dilated = dilate(occMask, width, height, DILATE_RADIUS);
    const gutter = buildGutterMap(occMask, width, height, DILATE_RADIUS + 1);
    const rasterMs = performance.now() - tRaster0;

    const sab = (TypedArrayCtor, srcArr) => {
      const sab = new SharedArrayBuffer(srcArr.byteLength);
      const view = new TypedArrayCtor(sab);
      view.set(srcArr);
      return view;
    };
    const raster = {
      triId: sab(Int32Array, triId), baryA: sab(Float32Array, baryA), baryB: sab(Float32Array, baryB), baryC: sab(Float32Array, baryC),
      occMask: sab(Uint8Array, occMask), dilated: sab(Uint8Array, dilated), gutter: sab(Int32Array, gutter),
      width, height,
    };

    const jobs = [];
    for (let off = 1; off < run.globalFrames.length; off++) jobs.push({ globalFrameIdx: run.globalFrames[off], offset: off });
    const nWorkers = Math.min(NWORKERS, jobs.length);
    const shards = Array.from({ length: nWorkers }, () => []);
    jobs.forEach((j, i) => shards[i % nWorkers].push(j));

    const templateIndicesShared = sab(Uint32Array, run.templateIndices);
    const templateUvsShared = sab(Float32Array, run.templateUvs);

    const perFrameResults = [];
    const tRunBake0 = performance.now();
    await Promise.all(shards.map((shardJobs, wi) => new Promise((resolve, reject) => {
      if (!shardJobs.length) return resolve();
      const w = new Worker(new URL("./bake-clip-worker.mjs", import.meta.url), {
        workerData: {
          srcDir, outDir, pad, templateIndices: templateIndicesShared, templateUvs: templateUvsShared,
          raster, runCkptDir, jobs: shardJobs, rings: RINGS, fallbackMm: FALLBACK_MM, bakeMode: BAKE_MODE,
        },
      });
      w.on("message", (msg) => {
        if (msg.type === "frame-done") {
          perFrameResults.push(msg);
          bakedDone++;
          emitProgress("bake", bakedDone, totalBake, tBake0);
        } else if (msg.type === "error") {
          console.error(`[coherent-clip] run ${ri} worker${wi} ERROR: ${msg.message}`);
        }
      });
      w.on("error", reject);
      w.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`worker${wi} exited ${code}`))));
    })));
    const runBakeMs = performance.now() - tRunBake0;
    console.log(`[coherent-clip] run ${ri} bake done: ${jobs.length} frame(s) in ${(runBakeMs / 1000).toFixed(1)}s wall clock (${nWorkers} workers), template raster ${(rasterMs / 1000).toFixed(2)}s`);

    runManifests.push(runSummary(run, ri, runBakeMs, perFrameResults, rasterMs));
  }
  const bakeWallMs = performance.now() - tBake0;

  function runSummary(run, ri, runBakeMs, perFrameResults, rasterMs = 0) {
    const meanFallbackPct = perFrameResults.length ? perFrameResults.reduce((s, r) => s + r.fallbackPct, 0) / perFrameResults.length : 0;
    return {
      runIndex: ri,
      templateGlobalFrame: run.templateGlobalFrame,
      startGlobalFrame: run.globalFrames[0],
      endGlobalFrame: run.globalFrames[run.globalFrames.length - 1],
      length: run.globalFrames.length,
      cutReason: run.cutReason,
      cutP95: run.cutP95,
      registerMs: run.registerMs,
      rasterMs,
      bakeWallMs: runBakeMs,
      bakeMeanMsPerFrame: perFrameResults.length ? perFrameResults.reduce((s, r) => s + r.ms, 0) / perFrameResults.length : 0,
      meanFallbackPct,
      errors: run.errors,
    };
  }

  /* -------------------------------- boundary measurements -------------------------------- */
  const boundaries = [];
  const scratch = await mkScratch("ares-clip-boundary-");
  function psnr(a, b) {
    let sq = 0, n = 0;
    for (let i = 0; i < a.length; i += 4) for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; sq += d * d; n++; }
    const mse = sq / n;
    return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
  }
  for (let ri = 0; ri < runs.length - 1; ri++) {
    const prevRun = runs[ri], nextRun = runs[ri + 1];
    const lastFrameOfPrev = prevRun.globalFrames[prevRun.globalFrames.length - 1];
    const lastErr = prevRun.errors[prevRun.errors.length - 1];
    const templateOfNext = nextRun.templateGlobalFrame;
    try {
      const a = await decodePng(join(outDir, atlasName(lastFrameOfPrev)), scratch);
      const b = await decodePng(join(outDir, atlasName(templateOfNext)), scratch);
      const boundaryPsnr = (a.width === b.width && a.height === b.height) ? psnr(a.data, b.data) : null;
      boundaries.push({
        fromRun: ri, toRun: ri + 1, lastFrameOfPrevRun: lastFrameOfPrev, templateOfNextRun: templateOfNext,
        lastFrameResidualMm: { mean: lastErr.mean, p95: lastErr.p95, max: lastErr.max },
        templateResidualMm: 0, // verbatim by construction
        atlasBoundaryPSNR: boundaryPsnr,
      });
    } catch (e) {
      boundaries.push({ fromRun: ri, toRun: ri + 1, lastFrameOfPrevRun: lastFrameOfPrev, templateOfNextRun: templateOfNext, error: String(e) });
    }
  }

  const totalMs = performance.now() - t0;
  const manifest = {
    clip: { srcDir, outDir, totalFrames: clipFrames.length, register: ARAP ? "arap" : "nearest-taubin", arapLambda: ARAP ? ARAP_LAMBDA : null, arapGate: ARAP ? ARAP_GATE : null, arapOuter: ARAP ? ARAP_OUTER : null, gop: GOP, errCutMm: ERR_CUT_MM, stretchCut: STRETCH_CUT, maxRounds: MAX_ROUNDS, epsMm: EPS_MM, minRun: MIN_RUN, rounds: ROUNDS, smoothIters: SMOOTH_ITERS, rings: RINGS, fallbackMm: FALLBACK_MM, bakeMode: BAKE_MODE, gpu: GPU_BAKE, workers: NWORKERS },
    runs: runManifests,
    cuts: cutsLog,
    boundaries,
    wallClock: { registerMs: registerWallMs, bakeMs: bakeWallMs, totalMs },
  };
  await writeFile(join(dirname(outDir), "coherent-manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`[coherent-clip] DONE: ${runs.length} run(s), ${cutsLog.length} cut(s), register=${(registerWallMs / 1000).toFixed(1)}s bake=${(bakeWallMs / 1000).toFixed(1)}s total=${(totalMs / 1000).toFixed(1)}s`);
  console.log(`[coherent-clip] manifest: ${join(dirname(outDir), "coherent-manifest.json")}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
