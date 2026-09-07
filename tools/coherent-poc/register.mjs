/**
 * Task K POC, stage 1: REGISTRATION. Deforms the template (Daniel frame 0's mesh, topology
 * + UVs fixed for the whole GOP) onto each of frames 1..29's source surface.
 *
 * Recipe (task spec): 3 rounds of {nearestInto each template vertex onto the frame's source
 * mesh -> target point; move toward target; regularize by weld-aware Taubin smoothing of the
 * DISPLACEMENT field over the template's own adjacency}. Runs BOTH candidate bases per frame
 * — "template" (always start from frame 0's rest pose) and "chain" (start from the previous
 * frame's final deformed positions) — and reports both error curves so we can pick the winner
 * honestly instead of assuming.
 *
 * Output: registration.json (per-frame/per-variant mean/p95/max nearest-dist mm + %>5mm) and
 * a binary checkpoint (positions.bin) holding the SELECTED variant's deformed positions for
 * all 30 frames (frame 0 = template verbatim), consumed by bake.mjs.
 */
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";
import { TriangleGrid, buildWeldAdjacency, smoothFieldWelded, bboxDiag } from "./lib.mjs";

// The source capture lives outside the repo, alongside it. Override with ARES_SRC_DIR.
const SRC_DIR = process.env.ARES_SRC_DIR
  || fileURLToPath(new URL("../../../Daniel_Microsoft_Volcap/Daniel_Volcap", import.meta.url));
const OUT_DIR = process.argv[2] || join(tmpdir(), "ares-coherent-poc");
const GOP_START = Number(process.env.ARES_GOP_START || 1);   // mesh-f00001.obj = frame 0
const GOP_LEN = Number(process.env.ARES_GOP_LEN || 30);      // frames 0..LEN-1 from GOP_START
const ROUNDS = 3;
const SMOOTH_ITERS = 3;

function fname(i) { return `mesh-f${String(i).padStart(5, "0")}.obj`; }

function percentile(sorted, p) {
  const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
  return sorted[idx];
}

function registerOneFrame(base, grid, weld, rounds, smoothIters) {
  let cur = base.slice();
  const n = cur.length / 3;
  const target = new Float32Array(cur.length);
  for (let r = 0; r < rounds; r++) {
    for (let i = 0; i < n; i++) {
      grid.nearestInto(cur[i * 3], cur[i * 3 + 1], cur[i * 3 + 2], target, i * 3);
    }
    const rawDisp = new Float32Array(cur.length);
    for (let i = 0; i < cur.length; i++) rawDisp[i] = target[i] - cur[i];
    const smoothed = smoothFieldWelded(rawDisp, weld, smoothIters);
    for (let i = 0; i < cur.length; i++) cur[i] += smoothed[i];
  }
  return cur;
}

function measureError(positions, grid) {
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
  return {
    mean: sum / n,
    p95: percentile(sorted, 0.95),
    max: sorted[sorted.length - 1],
    pctOver5mm: (over5 / n) * 100,
  };
}

async function main() {
  const t0 = performance.now();
  console.log(`[register] loading frames ${GOP_START}..${GOP_START + GOP_LEN - 1}`);
  const frames = [];
  for (let i = 0; i < GOP_LEN; i++) {
    const text = await readFile(join(SRC_DIR, fname(GOP_START + i)), "utf8");
    frames.push(parseObj(text));
  }
  const tLoad = performance.now();
  console.log(`[register] loaded ${frames.length} frames in ${((tLoad - t0) / 1000).toFixed(1)}s ` +
    `(template: ${frames[0].positions.length / 3} verts, ${frames[0].indices.length / 3} tris)`);

  const template = frames[0];
  const weld = buildWeldAdjacency(template.positions, template.indices);
  console.log(`[register] template weld: ${template.positions.length / 3} verts -> ${weld.nc} welded (${weld.n - weld.nc} seam duplicates)`);

  const variants = { template: [template.positions.slice()], chain: [template.positions.slice()] };
  const errors = { template: [], chain: [] };
  let chainPrev = template.positions;

  for (let f = 1; f < GOP_LEN; f++) {
    const tf0 = performance.now();
    const diag = bboxDiag(frames[f].positions) || 1;
    const cell = diag / 48;
    const grid = new TriangleGrid(frames[f].positions, frames[f].indices, frames[f].uvs, cell);

    const posTemplateBase = registerOneFrame(template.positions, grid, weld, ROUNDS, SMOOTH_ITERS);
    const errTemplateBase = measureError(posTemplateBase, grid);
    variants.template.push(posTemplateBase);
    errors.template.push({ frame: f, ...errTemplateBase });

    const posChainBase = registerOneFrame(chainPrev, grid, weld, ROUNDS, SMOOTH_ITERS);
    const errChainBase = measureError(posChainBase, grid);
    variants.chain.push(posChainBase);
    errors.chain.push({ frame: f, ...errChainBase });
    chainPrev = posChainBase;

    const dt = performance.now() - tf0;
    console.log(`[register] frame ${f}: template-base mean=${errTemplateBase.mean.toFixed(2)}mm p95=${errTemplateBase.p95.toFixed(2)} max=${errTemplateBase.max.toFixed(2)} >5mm=${errTemplateBase.pctOver5mm.toFixed(1)}% | ` +
      `chain-base mean=${errChainBase.mean.toFixed(2)}mm p95=${errChainBase.p95.toFixed(2)} max=${errChainBase.max.toFixed(2)} >5mm=${errChainBase.pctOver5mm.toFixed(1)}% (${dt.toFixed(0)}ms)`);
  }

  const meanOf = (arr) => arr.reduce((s, e) => s + e.mean, 0) / arr.length;
  const lastOf = (arr) => arr[arr.length - 1].mean;
  const firstOf = (arr) => arr[0].mean;
  const growthTemplate = lastOf(errors.template) - firstOf(errors.template);
  const growthChain = lastOf(errors.chain) - firstOf(errors.chain);
  const selected = meanOf(errors.chain) < meanOf(errors.template) ? "chain" : "template";
  console.log(`[register] GOP-mean error: template-base=${meanOf(errors.template).toFixed(3)}mm (growth ${growthTemplate.toFixed(3)}mm) ` +
    `chain-base=${meanOf(errors.chain).toFixed(3)}mm (growth ${growthChain.toFixed(3)}mm) -> SELECTED: ${selected}`);

  // Persist checkpoint: JSON stats + binary positions (selected variant) for bake.mjs.
  const stats = {
    gopStart: GOP_START, gopLen: GOP_LEN, rounds: ROUNDS, smoothIters: SMOOTH_ITERS,
    templateVerts: template.positions.length / 3, templateTris: template.indices.length / 3,
    errors, selected,
    wallClockMs: performance.now() - t0,
  };
  await writeFile(join(OUT_DIR, "registration.json"), JSON.stringify(stats, null, 2));

  const selectedPositions = variants[selected];
  // positions.bin: u32 frameCount, u32 vertCount, then frameCount*vertCount*3 f32
  const vcount = template.positions.length / 3;
  const header = new Uint32Array([selectedPositions.length, vcount]);
  const bodyBytes = selectedPositions.length * vcount * 3 * 4;
  const buf = Buffer.alloc(8 + bodyBytes);
  buf.writeUInt32LE(header[0], 0);
  buf.writeUInt32LE(header[1], 4);
  let off = 8;
  for (const p of selectedPositions) {
    Buffer.from(p.buffer, p.byteOffset, p.byteLength).copy(buf, off);
    off += p.byteLength;
  }
  await writeFile(join(OUT_DIR, "positions.bin"), buf);
  // also stash BOTH variants (small: 30 * 11323 * 3 * 4 ~= 4MB each) for later inspection/plots
  for (const key of ["template", "chain"]) {
    const arr = variants[key];
    const b2 = Buffer.alloc(8 + arr.length * vcount * 3 * 4);
    b2.writeUInt32LE(arr.length, 0); b2.writeUInt32LE(vcount, 4);
    let o2 = 8;
    for (const p of arr) { Buffer.from(p.buffer, p.byteOffset, p.byteLength).copy(b2, o2); o2 += p.byteLength; }
    await writeFile(join(OUT_DIR, `positions-${key}.bin`), b2);
  }
  // template topology/UVs, needed by bake.mjs
  await writeFile(join(OUT_DIR, "template-uvs.bin"), Buffer.from(template.uvs.buffer, template.uvs.byteOffset, template.uvs.byteLength));
  await writeFile(join(OUT_DIR, "template-indices.bin"), Buffer.from(template.indices.buffer, template.indices.byteOffset, template.indices.byteLength));

  console.log(`[register] wrote checkpoint to ${OUT_DIR} — total wall clock ${((performance.now() - t0) / 1000).toFixed(1)}s`);
}

main().catch((e) => { console.error(e); process.exit(1); });
