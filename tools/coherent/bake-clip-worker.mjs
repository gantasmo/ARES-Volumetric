/**
 * Phase 2 full-clip bake worker (worker_thread). Bakes an assigned subset of (run, frame) jobs.
 * Unlike the POC's bake-worker.mjs, the template UV raster (triId/baryA/baryB/baryC/occMask/
 * dilated/gutter) is rasterized ONCE by the coordinator (coherent-clip.mjs) per RUN and shared
 * to every worker via SharedArrayBuffer — no per-worker recompute, no per-worker copy. The
 * per-texel nearest-source-surface query uses bakeFrameFast (seeded local search), not the cold
 * TriangleGrid scan.
 */
import { parentPort, workerData } from "node:worker_threads";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";
import { decodePng, encodePng, mkScratch, bakeFrame, bakeFrameFast, writeObjText } from "../coherent-poc/lib.mjs";

const {
  srcDir, outDir, pad, templateIndices, templateUvs, raster, runCkptDir, jobs, rings, fallbackMm,
  bakeMode = "exact",
} = workerData;

function fname(i) { return `mesh-f${String(i).padStart(pad, "0")}.obj`; }
function atlasName(i) { return `atlas-f${String(i).padStart(pad, "0")}.png`; }

function readPositionsBin(buf) {
  const frameCount = buf.readUInt32LE(0);
  const vcount = buf.readUInt32LE(4);
  const frames = [];
  let off = 8;
  const bytesPerFrame = vcount * 3 * 4;
  for (let f = 0; f < frameCount; f++) {
    frames.push(new Float32Array(buf.buffer, buf.byteOffset + off, vcount * 3).slice());
    off += bytesPerFrame;
  }
  return { frameCount, vcount, frames };
}
function readSeedTrisBin(buf) {
  const frameCount = buf.readUInt32LE(0);
  const vcount = buf.readUInt32LE(4);
  const frames = [];
  let off = 8;
  const bytesPerFrame = vcount * 4;
  for (let f = 0; f < frameCount; f++) {
    frames.push(new Int32Array(buf.buffer, buf.byteOffset + off, vcount).slice());
    off += bytesPerFrame;
  }
  return frames;
}

async function main() {
  const templateIdx = new Uint32Array(templateIndices);
  const templateUv = new Float32Array(templateUvs);
  const { triId, baryA, baryB, baryC, occMask, dilated, gutter, width, height } = raster;

  const posBuf = await readFile(join(runCkptDir, "positions.bin"));
  const { frames: deformedPositions } = readPositionsBin(posBuf);
  const seedBuf = await readFile(join(runCkptDir, "vertex-seed-tris.bin"));
  const seedTris = readSeedTrisBin(seedBuf);

  const scratch = await mkScratch(`ares-clip-bake-w${process.pid}-`);

  for (const { globalFrameIdx, offset } of jobs) {
    const tf0 = performance.now();
    const srcText = await readFile(join(srcDir, fname(globalFrameIdx)), "utf8");
    const src = parseObj(srcText);
    const srcAtlas = await decodePng(join(srcDir, atlasName(globalFrameIdx)), scratch);

    // bake mode (visual verdict 2026-07-13: full-clip fast bake "looks really bad… every frame" vs
    // POC-A): "exact" = POC bakeFrame, true global nearest per texel — what coherent-A used;
    // "fast" = seeded local search that ACCEPTS any hit ≤ fallbackMm (3mm) off the true surface —
    // that acceptance slop (94% of texels), not the ~6% exact-fallback texels, was the smear.
    const common = {
      srcPositions: src.positions, srcIndices: src.indices, srcUvs: src.uvs,
      deformed: deformedPositions[offset],
      templateIndices: templateIdx, triId, baryA, baryB, baryC, occMask, dilated, gutter, width, height,
      srcAtlasData: srcAtlas.data, srcAtlasWidth: srcAtlas.width, srcAtlasHeight: srcAtlas.height,
    };
    const { outAtlas, sampled, fallbackCount = 0, fallbackPct = 0 } = bakeMode === "fast"
      ? bakeFrameFast({ ...common, vertexSeedTri: seedTris[offset], rings, fallbackMm })
      : bakeFrame(common);

    await writeFile(join(outDir, atlasName(globalFrameIdx)), encodePng(outAtlas, width, height));
    await writeFile(join(outDir, fname(globalFrameIdx)), writeObjText(deformedPositions[offset], templateUv, templateIdx));

    const dt = performance.now() - tf0;
    parentPort.postMessage({ type: "frame-done", globalFrameIdx, sampled, fallbackCount, fallbackPct, ms: dt });
  }
  parentPort.postMessage({ type: "worker-done", count: jobs.length });
}

main().catch((e) => { parentPort.postMessage({ type: "error", message: e.stack || String(e) }); process.exit(1); });
