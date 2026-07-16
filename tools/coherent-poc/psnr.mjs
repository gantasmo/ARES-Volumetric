/**
 * Task K POC measurement: mean consecutive-frame PSNR of atlas-fNNNNN.png in a directory
 * (expected ~33dB for the source's per-frame-repacked atlas vs 42+dB for the coherent bake —
 * task hypothesis; this script measures the ACTUAL numbers, not the expectation).
 */
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { decodePng, mkScratch } from "./lib.mjs";

const dir = process.argv[2];
const limit = process.argv[3] ? Number(process.argv[3]) : Infinity;
if (!dir) { console.error("usage: node psnr.mjs <atlas-dir> [maxFrames]"); process.exit(1); }

function psnr(a, b) {
  let sq = 0, n = 0;
  for (let i = 0; i < a.length; i += 4) { // skip alpha
    for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; sq += d * d; n++; }
  }
  const mse = sq / n;
  if (mse === 0) return Infinity;
  return 10 * Math.log10((255 * 255) / mse);
}

async function main() {
  const all = (await readdir(dir)).filter((f) => /^atlas-f\d+\.png$/i.test(f)).sort();
  const files = all.slice(0, limit);
  console.log(`[psnr] ${dir}: ${files.length} atlas frame(s)`);
  const scratch = await mkScratch("ares-psnr-");
  let prev = null;
  const vals = [];
  for (const f of files) {
    const img = await decodePng(join(dir, f), scratch);
    if (prev) {
      const p = psnr(prev.data, img.data);
      vals.push(p);
      console.log(`  ${f}: PSNR vs prev = ${p.toFixed(2)} dB`);
    }
    prev = img;
  }
  const mean = vals.reduce((s, x) => s + x, 0) / vals.length;
  console.log(`[psnr] mean consecutive-frame PSNR over ${vals.length} pair(s): ${mean.toFixed(2)} dB`);
}

main().catch((e) => { console.error(e); process.exit(1); });
