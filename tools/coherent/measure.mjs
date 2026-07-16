/**
 * Task N Phase 2 measurements: registration error stats (overall + worst 5 frames) from the
 * manifest, and atlas-coherence PSNR (mean consecutive-frame) for the SOURCE atlas sequence vs
 * the COHERENT bake — reported both "within-run only" (comparable to the POC's 30-frame 31.33dB
 * number) and "including GOP-boundary pairs" (the honest full-clip average, boundaries expected
 * to pull it down since a boundary is a real atlas-layout change, not a defect).
 *
 * usage: node measure.mjs <coherent-manifest.json> <source-frames-dir> <coherent-frames-dir>
 */
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { decodePng, mkScratch } from "../coherent-poc/lib.mjs";

const [manifestPath, srcDir, coherentDir] = process.argv.slice(2);
if (!manifestPath || !srcDir || !coherentDir) {
  console.error("usage: node measure.mjs <coherent-manifest.json> <source-frames-dir> <coherent-frames-dir>");
  process.exit(1);
}

function psnr(a, b) {
  let sq = 0, n = 0;
  for (let i = 0; i < a.length; i += 4) for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; sq += d * d; n++; }
  const mse = sq / n;
  return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
}

async function meanConsecutivePsnr(dir, boundarySet) {
  const all = (await readdir(dir)).filter((f) => /^atlas-f\d+\.png$/i.test(f)).sort();
  const scratch = await mkScratch("ares-measure-psnr-");
  let prev = null, prevIdx = null;
  const within = [], boundary = [];
  for (const f of all) {
    const idx = Number(/(\d+)/.exec(f)[1]);
    const img = await decodePng(join(dir, f), scratch);
    if (prev) {
      const p = psnr(prev.data, img.data);
      if (boundarySet && boundarySet.has(idx)) boundary.push(p); else within.push(p);
    }
    prev = img; prevIdx = idx;
  }
  const mean = (arr) => arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : NaN;
  return { withinMean: mean(within), boundaryMean: mean(boundary), allMean: mean([...within, ...boundary]), nWithin: within.length, nBoundary: boundary.length };
}

async function main() {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const allErrors = manifest.runs.flatMap((r) => r.errors.filter((e) => e.p95 > 0 || e.mean > 0)); // exclude template's synthetic 0-entries
  const means = allErrors.map((e) => e.mean);
  const p95s = allErrors.map((e) => e.p95);
  const maxs = allErrors.map((e) => e.max);
  const meanOf = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  console.log(`[measure] registration error over ${allErrors.length} non-template frames:`);
  console.log(`  mean(mean)=${meanOf(means).toFixed(3)}mm mean(p95)=${meanOf(p95s).toFixed(3)}mm mean(max)=${meanOf(maxs).toFixed(3)}mm`);
  console.log(`  worst 5 by p95:`, allErrors.slice().sort((a, b) => b.p95 - a.p95).slice(0, 5).map((e) => ({ frame: e.frame, mean: +e.mean.toFixed(3), p95: +e.p95.toFixed(3), max: +e.max.toFixed(2) })));
  console.log(`  worst 5 by max:`, allErrors.slice().sort((a, b) => b.max - a.max).slice(0, 5).map((e) => ({ frame: e.frame, mean: +e.mean.toFixed(3), p95: +e.p95.toFixed(3), max: +e.max.toFixed(2) })));

  const boundarySet = new Set(manifest.boundaries.map((b) => b.templateOfNextRun));
  console.log(`[measure] boundary frames (template starts, excluded from "within-run" PSNR): ${[...boundarySet].join(", ")}`);

  console.log(`[measure] computing SOURCE atlas consecutive PSNR (${srcDir})...`);
  const srcPsnr = await meanConsecutivePsnr(srcDir, null);
  console.log(`  source: mean=${srcPsnr.allMean.toFixed(2)}dB over ${srcPsnr.nWithin + srcPsnr.nBoundary} pairs`);

  console.log(`[measure] computing COHERENT atlas consecutive PSNR (${coherentDir})...`);
  const cohPsnr = await meanConsecutivePsnr(coherentDir, boundarySet);
  console.log(`  coherent within-run: mean=${cohPsnr.withinMean.toFixed(2)}dB over ${cohPsnr.nWithin} pairs (POC reference: 31.33dB)`);
  console.log(`  coherent at boundaries: mean=${cohPsnr.boundaryMean.toFixed(2)}dB over ${cohPsnr.nBoundary} pairs (expected low — real atlas-layout change)`);
  console.log(`  coherent all pairs: mean=${cohPsnr.allMean.toFixed(2)}dB over ${cohPsnr.nWithin + cohPsnr.nBoundary} pairs`);
  console.log(`[measure] net gain (within-run coherent vs source): +${(cohPsnr.withinMean - srcPsnr.allMean).toFixed(2)}dB`);
}

main().catch((e) => { console.error(e); process.exit(1); });
