/**
 * Phase 2 validation: re-register + re-bake the POC's GOP (frames 0-29, chain variant only,
 * with per-vertex seed-triangle capture) using bakeFrameFast, and report:
 *  - per-frame single-threaded bake time (old bakeFrame vs new bakeFrameFast, first 3 frames)
 *  - full-GOP single-threaded bakeFrameFast time (extrapolate to parallel wall clock)
 *  - fallback rate (texels needing the exact global search)
 *  - texel-level diff vs the POC's kept frames-A atlases (mean/max abs diff per channel)
 *  - registration error stats (should match register.mjs's chain numbers closely)
 * Writes the re-baked frames-dir to OUT_DIR for downstream encode+PSNR checks.
 */
import { readFile, writeFile, mkdir, copyFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";
import {
  TriangleGrid, buildWeldAdjacency, registerChainFrame,
  rasterizeTemplateBary, dilate, buildGutterMap, decodePng, encodePng, mkScratch, bboxDiag,
  bakeFrame, bakeFrameFast, writeObjText,
} from "./lib.mjs";

// The source capture lives outside the repo, alongside it. Override with ARES_SRC_DIR.
const SRC_DIR = process.env.ARES_SRC_DIR
  || fileURLToPath(new URL("../../../Daniel_Microsoft_Volcap/Daniel_Volcap", import.meta.url));
const SCRATCH = process.env.ARES_SCRATCH || join(tmpdir(), "ares-coherent-poc-scratch");
const OUT_DIR = join(SCRATCH, "coherent-fast-validate");
const OLD_DIR = join(SCRATCH, "coherent-poc", "frames-A"); // POC's kept output
const GOP_START = 1;
const GOP_LEN = 30;
const ROUNDS = 3;
const SMOOTH_ITERS = 3;
const DILATE_RADIUS = 2;
const RINGS = Number(process.argv[2] || 2);
const FALLBACK_MM = Number(process.argv[3] || 3);

function fname(i) { return `mesh-f${String(i).padStart(5, "0")}.obj`; }
function atlasName(i) { return `atlas-f${String(i).padStart(5, "0")}.png`; }

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const t0 = performance.now();
  console.log(`[bench-fast] rings=${RINGS} fallbackMm=${FALLBACK_MM}`);

  const frames = [];
  for (let i = 0; i < GOP_LEN; i++) {
    frames.push(parseObj(await readFile(join(SRC_DIR, fname(GOP_START + i)), "utf8")));
  }
  const template = frames[0];
  const weld = buildWeldAdjacency(template.positions, template.indices);
  console.log(`[bench-fast] template: ${template.positions.length / 3} verts, ${template.indices.length / 3} tris`);

  // --- registration (chain only, with seed-tri capture) ---
  const tReg0 = performance.now();
  let chainPrev = template.positions;
  const deformedPositions = [template.positions.slice()];
  const vertexSeedTriPerFrame = [null]; // frame 0 = template, no bake needed
  const errors = [];
  for (let f = 1; f < GOP_LEN; f++) {
    const diag = bboxDiag(frames[f].positions) || 1;
    const grid = new TriangleGrid(frames[f].positions, frames[f].indices, frames[f].uvs, diag / 48);
    const { positions, vertexSeedTri, error } = registerChainFrame(chainPrev, grid, weld, ROUNDS, SMOOTH_ITERS);
    deformedPositions.push(positions);
    vertexSeedTriPerFrame.push(vertexSeedTri);
    errors.push({ frame: f, ...error });
    chainPrev = positions;
  }
  const tReg1 = performance.now();
  const meanErr = errors.reduce((s, e) => s + e.mean, 0) / errors.length;
  const meanP95 = errors.reduce((s, e) => s + e.p95, 0) / errors.length;
  console.log(`[bench-fast] registration: ${((tReg1 - tReg0) / 1000).toFixed(1)}s, mean err=${meanErr.toFixed(3)}mm meanP95=${meanP95.toFixed(3)}mm ` +
    `(POC reference: mean=0.148mm meanP95~0.442mm)`);

  // --- template raster (ONCE) ---
  const atlas0Path = join(SRC_DIR, atlasName(GOP_START));
  const hdr = await readFile(atlas0Path);
  const dv = new DataView(hdr.buffer, hdr.byteOffset, hdr.byteLength);
  const width = dv.getUint32(16, false), height = dv.getUint32(20, false);
  const tRaster0 = performance.now();
  const { triId, baryA, baryB, baryC } = rasterizeTemplateBary(template.uvs, template.indices, width, height);
  const occMask = new Uint8Array(width * height);
  for (let p = 0; p < triId.length; p++) if (triId[p] >= 0) occMask[p] = 1;
  const dilated = dilate(occMask, width, height, DILATE_RADIUS);
  const gutter = buildGutterMap(occMask, width, height, DILATE_RADIUS + 1);
  const tRaster1 = performance.now();
  console.log(`[bench-fast] template raster: ${((tRaster1 - tRaster0) / 1000).toFixed(2)}s (once per GOP)`);

  await copyFile(atlas0Path, join(OUT_DIR, atlasName(GOP_START)));
  await writeFile(join(OUT_DIR, fname(GOP_START)), writeObjText(template.positions, template.uvs, template.indices));

  const scratch = await mkScratch("ares-bench-fast-");
  const oldTimes = [], newTimes = [];
  let totalFallback = 0, totalSampled = 0;
  const diffStats = [];

  for (let f = 1; f < GOP_LEN; f++) {
    const srcFileIdx = GOP_START + f;
    const src = frames[f];
    const srcAtlas = await decodePng(join(SRC_DIR, atlasName(srcFileIdx)), scratch);
    const deformed = deformedPositions[f];
    const vertexSeedTri = vertexSeedTriPerFrame[f];

    // OLD path timing for the first 3 frames only (129.7s/frame reference — don't burn 60min here)
    if (f <= 3) {
      const tOld0 = performance.now();
      bakeFrame({
        srcPositions: src.positions, srcIndices: src.indices, srcUvs: src.uvs, deformed,
        templateIndices: template.indices, triId, baryA, baryB, baryC, occMask, dilated, gutter, width, height,
        srcAtlasData: srcAtlas.data, srcAtlasWidth: srcAtlas.width, srcAtlasHeight: srcAtlas.height,
      });
      const tOld1 = performance.now();
      oldTimes.push(tOld1 - tOld0);
      console.log(`[bench-fast] frame ${f} OLD bakeFrame: ${((tOld1 - tOld0) / 1000).toFixed(2)}s`);
    }

    const tNew0 = performance.now();
    const { outAtlas, sampled, fallbackCount, fallbackPct } = bakeFrameFast({
      srcPositions: src.positions, srcIndices: src.indices, srcUvs: src.uvs, deformed,
      templateIndices: template.indices, triId, baryA, baryB, baryC, occMask, dilated, gutter, width, height,
      srcAtlasData: srcAtlas.data, srcAtlasWidth: srcAtlas.width, srcAtlasHeight: srcAtlas.height,
      vertexSeedTri, rings: RINGS, fallbackMm: FALLBACK_MM,
    });
    const tNew1 = performance.now();
    newTimes.push(tNew1 - tNew0);
    totalFallback += fallbackCount; totalSampled += sampled;
    console.log(`[bench-fast] frame ${f} NEW bakeFrameFast: ${((tNew1 - tNew0) / 1000).toFixed(2)}s, ` +
      `fallback=${fallbackCount}/${sampled} (${fallbackPct.toFixed(2)}%)`);

    await writeFile(join(OUT_DIR, atlasName(srcFileIdx)), encodePng(outAtlas, width, height));
    await writeFile(join(OUT_DIR, fname(srcFileIdx)), writeObjText(deformed, template.uvs, template.indices));

    // texel diff vs POC's kept frames-A, if present
    try {
      const oldAtlas = await decodePng(join(OLD_DIR, atlasName(srcFileIdx)), scratch);
      let sumAbs = 0, maxAbs = 0, nBig = 0, n = 0;
      for (let p = 0; p < occMask.length; p++) {
        if (!occMask[p]) continue;
        const o = p * 4;
        for (let c = 0; c < 3; c++) {
          const d = Math.abs(outAtlas[o + c] - oldAtlas.data[o + c]);
          sumAbs += d; if (d > maxAbs) maxAbs = d; if (d > 20) nBig++;
          n++;
        }
      }
      diffStats.push({ frame: f, meanAbsDiff: sumAbs / n, maxAbsDiff: maxAbs, pctBig: (100 * nBig) / n });
    } catch (e) {
      // POC kept dir may not have this frame; skip
    }
  }

  const totalMs = performance.now() - t0;
  const meanOld = oldTimes.length ? oldTimes.reduce((s, x) => s + x, 0) / oldTimes.length : NaN;
  const meanNew = newTimes.reduce((s, x) => s + x, 0) / newTimes.length;
  console.log(`\n[bench-fast] === SUMMARY ===`);
  console.log(`  OLD bakeFrame mean (${oldTimes.length} frames): ${(meanOld / 1000).toFixed(2)}s/frame (POC single-thread ref: 129.7s)`);
  console.log(`  NEW bakeFrameFast mean (${newTimes.length} frames): ${(meanNew / 1000).toFixed(2)}s/frame`);
  console.log(`  speedup: ${(meanOld / meanNew).toFixed(1)}x (single-threaded)`);
  console.log(`  total fallback texels: ${totalFallback}/${totalSampled} (${(100 * totalFallback / totalSampled).toFixed(3)}%)`);
  console.log(`  total wall clock (this script): ${(totalMs / 1000).toFixed(1)}s`);
  if (diffStats.length) {
    const meanMeanAbs = diffStats.reduce((s, d) => s + d.meanAbsDiff, 0) / diffStats.length;
    const maxMaxAbs = Math.max(...diffStats.map((d) => d.maxAbsDiff));
    const meanPctBig = diffStats.reduce((s, d) => s + d.pctBig, 0) / diffStats.length;
    console.log(`  diff vs POC frames-A: mean|meanAbsDiff|=${meanMeanAbs.toFixed(3)} maxAbsDiff=${maxMaxAbs} mean%texels>20/255=${meanPctBig.toFixed(3)}%`);
    console.log(`  per-frame: ${JSON.stringify(diffStats.map((d) => ({ f: d.frame, mean: +d.meanAbsDiff.toFixed(2), max: d.maxAbsDiff, pctBig: +d.pctBig.toFixed(3) })))}`);
  } else {
    console.log(`  (no POC kept-frame comparison available)`);
  }
  await writeFile(join(OUT_DIR, "bench-summary.json"), JSON.stringify({ rings: RINGS, fallbackMm: FALLBACK_MM, meanOldMs: meanOld, meanNewMs: meanNew, totalFallback, totalSampled, diffStats, errors }, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
