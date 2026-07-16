/**
 * Task K POC measurement: pixel-diff RMS between two rendered screenshots (same camera,
 * consecutive frames) — the render-side "boiling" metric. Requires same dimensions.
 */
import { decodePng, mkScratch } from "./lib.mjs";

const [a, b] = process.argv.slice(2);
if (!a || !b) { console.error("usage: node rms-diff.mjs <a.png> <b.png>"); process.exit(1); }

const scratch = await mkScratch("ares-rmsdiff-");
const ia = await decodePng(a, scratch);
const ib = await decodePng(b, scratch);
if (ia.width !== ib.width || ia.height !== ib.height) {
  console.error(`size mismatch: ${a} ${ia.width}x${ia.height} vs ${b} ${ib.width}x${ib.height}`);
  process.exit(1);
}
let sq = 0, n = 0;
for (let i = 0; i < ia.data.length; i += 4) {
  for (let c = 0; c < 3; c++) { const d = ia.data[i + c] - ib.data[i + c]; sq += d * d; n++; }
}
const rms = Math.sqrt(sq / n);
console.log(rms.toFixed(4));
