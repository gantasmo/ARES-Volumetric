/**
 * Task K POC bake orchestrator — spreads the per-frame texel bake (measured ~130s/frame
 * single-threaded, too slow for 29 frames) across worker_threads, one
 * shard of frames per CPU core.
 */
import { Worker } from "node:worker_threads";
import { readFile, writeFile, mkdir, copyFile } from "node:fs/promises";
import { cpus } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";

// The source capture lives outside the repo, alongside it. Override with ARES_SRC_DIR.
const SRC_DIR = process.env.ARES_SRC_DIR
  || fileURLToPath(new URL("../../../Daniel_Microsoft_Volcap/Daniel_Volcap", import.meta.url));
const CKPT_DIR = process.argv[2];
const OUT_DIR = process.argv[3];
const GOP_START = 1;
const DILATE_RADIUS = 2;

function fname(i) { return `mesh-f${String(i).padStart(5, "0")}.obj`; }
function atlasName(i) { return `atlas-f${String(i).padStart(5, "0")}.png`; }
function writeObjText(positions, uvs, indices) {
  const lines = [];
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) lines.push(`v ${positions[i * 3]} ${positions[i * 3 + 1]} ${positions[i * 3 + 2]}`);
  for (let i = 0; i < n; i++) lines.push(`vt ${uvs[i * 2]} ${1 - uvs[i * 2 + 1]}`);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] + 1, b = indices[t + 1] + 1, c = indices[t + 2] + 1;
    lines.push(`f ${a}/${a} ${b}/${b} ${c}/${c}`);
  }
  return lines.join("\n") + "\n";
}

async function main() {
  const t0 = performance.now();
  await mkdir(OUT_DIR, { recursive: true });

  const posBuf = await readFile(join(CKPT_DIR, "positions.bin"));
  const frameCount = posBuf.readUInt32LE(0);
  const workFrames = Array.from({ length: frameCount - 1 }, (_, i) => i + 1); // 1..frameCount-1

  // Frame 0 = template verbatim.
  await copyFile(join(SRC_DIR, atlasName(GOP_START)), join(OUT_DIR, atlasName(GOP_START)));
  const frame0Text = await readFile(join(SRC_DIR, fname(GOP_START)), "utf8");
  const frame0 = parseObj(frame0Text);
  await writeFile(join(OUT_DIR, fname(GOP_START)), writeObjText(frame0.positions, frame0.uvs, frame0.indices));

  const nWorkers = Math.min(cpus().length, workFrames.length);
  const shards = Array.from({ length: nWorkers }, () => []);
  workFrames.forEach((f, i) => shards[i % nWorkers].push(f)); // round-robin so shards are balanced

  console.log(`[bake-parallel] ${workFrames.length} frame(s) across ${nWorkers} worker(s)`);

  let done = 0, totalSampled = 0;
  const perFrameMs = [];
  await Promise.all(shards.map((frameIndices, wi) => new Promise((resolve, reject) => {
    if (!frameIndices.length) return resolve();
    const w = new Worker(new URL("./bake-worker.mjs", import.meta.url), {
      workerData: { srcDir: SRC_DIR, ckptDir: CKPT_DIR, outDir: OUT_DIR, gopStart: GOP_START, frameIndices, dilateRadius: DILATE_RADIUS },
    });
    w.on("message", (msg) => {
      if (msg.type === "frame-done") {
        done++; totalSampled += msg.sampled; perFrameMs.push(msg.ms);
        console.log(`[bake-parallel] worker${wi} frame ${msg.frame}: ${msg.sampled} texels in ${(msg.ms / 1000).toFixed(2)}s (${done}/${workFrames.length} total)`);
      } else if (msg.type === "error") {
        console.error(`[bake-parallel] worker${wi} ERROR: ${msg.message}`);
      }
    });
    w.on("error", reject);
    w.on("exit", (code) => code === 0 ? resolve() : reject(new Error(`worker${wi} exited ${code}`)));
  })));

  const totalMs = performance.now() - t0;
  const meanFrameMs = perFrameMs.reduce((s, x) => s + x, 0) / perFrameMs.length;
  console.log(`[bake-parallel] done: ${workFrames.length} frames baked in ${(totalMs / 1000).toFixed(1)}s wall clock ` +
    `(${nWorkers} workers, mean ${(meanFrameMs / 1000).toFixed(2)}s/frame per-worker, ${totalSampled} texels total)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
