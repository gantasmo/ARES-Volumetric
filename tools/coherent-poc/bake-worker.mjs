/**
 * Task K POC bake worker (worker_thread). Bakes an assigned subset of frames — recomputes the
 * (cheap, ~1s) template UV raster locally rather than transferring it, then loops per-frame.
 */
import { parentPort, workerData } from "node:worker_threads";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";
import {
  rasterizeTemplateBary, dilate, buildGutterMap, decodePng, encodePng, mkScratch, bakeFrame,
} from "./lib.mjs";

const { srcDir, ckptDir, outDir, gopStart, frameIndices, dilateRadius } = workerData;

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

async function main() {
  const posBuf = await readFile(join(ckptDir, "positions.bin"));
  const { frames: deformedPositions } = readPositionsBin(posBuf);
  const uvBuf = await readFile(join(ckptDir, "template-uvs.bin"));
  const idxBuf = await readFile(join(ckptDir, "template-indices.bin"));
  const templateUvs = new Float32Array(uvBuf.buffer, uvBuf.byteOffset, uvBuf.byteLength / 4);
  const templateIndices = new Uint32Array(idxBuf.buffer, idxBuf.byteOffset, idxBuf.byteLength / 4);

  const atlas0Path = join(srcDir, atlasName(gopStart));
  const header = await readFile(atlas0Path);
  const dv = new DataView(header.buffer, header.byteOffset, header.byteLength);
  const width = dv.getUint32(16, false), height = dv.getUint32(20, false);

  const { triId, baryA, baryB, baryC } = rasterizeTemplateBary(templateUvs, templateIndices, width, height);
  const occMask = new Uint8Array(width * height);
  for (let p = 0; p < triId.length; p++) if (triId[p] >= 0) occMask[p] = 1;
  const dilated = dilate(occMask, width, height, dilateRadius);
  const gutter = buildGutterMap(occMask, width, height, dilateRadius + 1);

  const scratch = await mkScratch(`ares-coherent-bake-w${process.pid}-`);

  for (const f of frameIndices) {
    const tf0 = performance.now();
    const srcFileIdx = gopStart + f;
    const srcText = await readFile(join(srcDir, fname(srcFileIdx)), "utf8");
    const src = parseObj(srcText);
    const srcAtlas = await decodePng(join(srcDir, atlasName(srcFileIdx)), scratch);

    const { outAtlas, sampled } = bakeFrame({
      srcPositions: src.positions, srcIndices: src.indices, srcUvs: src.uvs,
      deformed: deformedPositions[f],
      templateIndices, triId, baryA, baryB, baryC, occMask, dilated, gutter, width, height,
      srcAtlasData: srcAtlas.data, srcAtlasWidth: srcAtlas.width, srcAtlasHeight: srcAtlas.height,
    });

    await writeFile(join(outDir, atlasName(srcFileIdx)), encodePng(outAtlas, width, height));
    await writeFile(join(outDir, fname(srcFileIdx)), writeObjText(deformedPositions[f], templateUvs, templateIndices));

    const dt = performance.now() - tf0;
    parentPort.postMessage({ type: "frame-done", frame: f, sampled, ms: dt });
  }
  parentPort.postMessage({ type: "worker-done", count: frameIndices.length });
}

main().catch((e) => { parentPort.postMessage({ type: "error", message: e.stack || String(e) }); process.exit(1); });
