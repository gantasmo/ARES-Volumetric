/**
 * Task K POC, stage 2: TEXTURE BAKE. Per frame f, for every occupied texel of the FIXED
 * template atlas: texel -> (template triangle, barycentric) -> 3D point on the DEFORMED
 * template at frame f -> nearestInto frame f's SOURCE mesh -> source UV -> bilinear-sample
 * frame f's source atlas -> write into the coherent atlas for frame f. Dilation ring is
 * filled by copying the nearest already-baked occupied texel (gutter bleed), not re-sampled.
 *
 * Emits a frames-dir: mesh-f00001..030.obj (IDENTICAL topology+UVs = template's, positions
 * differ per frame) + atlas-f00001..030.png (2048x2048, baked), consumed by cli.js encode.
 */
import { readFile, writeFile, mkdir, copyFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { parseObj } from "../../packages/encoder/dist/importers/obj.js";
import {
  TriangleGrid, rasterizeTemplateBary, dilate, buildGutterMap, bilinearSample,
  decodePng, encodePng, mkScratch, bboxDiag,
} from "./lib.mjs";

// The source capture lives outside the repo, alongside it. Override with ARES_SRC_DIR.
const SRC_DIR = process.env.ARES_SRC_DIR
  || fileURLToPath(new URL("../../../Daniel_Microsoft_Volcap/Daniel_Volcap", import.meta.url));
const CKPT_DIR = process.argv[2] || join(tmpdir(), "ares-coherent-poc");
const OUT_DIR = process.argv[3] || join(CKPT_DIR, "frames-A");
const GOP_START = 1;
const DILATE_RADIUS = 2;
const ONLY_FRAME = process.argv[4] ? Number(process.argv[4]) : null; // debug: bake a single frame index

function fname(i) { return `mesh-f${String(i).padStart(5, "0")}.obj`; }
function atlasName(i) { return `atlas-f${String(i).padStart(5, "0")}.png`; }

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
  const { frameCount, vcount, frames: deformedPositions } = readPositionsBin(posBuf);
  const uvBuf = await readFile(join(CKPT_DIR, "template-uvs.bin"));
  const idxBuf = await readFile(join(CKPT_DIR, "template-indices.bin"));
  const templateUvs = new Float32Array(uvBuf.buffer, uvBuf.byteOffset, uvBuf.byteLength / 4);
  const templateIndices = new Uint32Array(idxBuf.buffer, idxBuf.byteOffset, idxBuf.byteLength / 4);
  console.log(`[bake] checkpoint: ${frameCount} frames, ${vcount} template verts, ${templateIndices.length / 3} tris`);

  // Frame 0 = template verbatim: OBJ from checkpoint positions[0] (== template rest pose) + its own atlas, copied.
  const scratch = await mkScratch("ares-coherent-bake-");

  // Rasterize the FIXED template atlas layout once. Atlas size probed from frame 0's PNG.
  const atlas0Path = join(SRC_DIR, atlasName(GOP_START));
  const { data: atlas0Header } = { data: await readFile(atlas0Path) };
  const dv = new DataView(atlas0Header.buffer, atlas0Header.byteOffset, atlas0Header.byteLength);
  const width = dv.getUint32(16, false), height = dv.getUint32(20, false);
  console.log(`[bake] template atlas ${width}x${height}`);

  const tRaster0 = performance.now();
  const { triId, baryA, baryB, baryC } = rasterizeTemplateBary(templateUvs, templateIndices, width, height);
  let occCount = 0;
  const occMask = new Uint8Array(width * height);
  for (let p = 0; p < triId.length; p++) if (triId[p] >= 0) { occMask[p] = 1; occCount++; }
  const dilated = dilate(occMask, width, height, DILATE_RADIUS);
  const gutter = buildGutterMap(occMask, width, height, DILATE_RADIUS + 1);
  let ringCount = 0;
  for (let p = 0; p < dilated.length; p++) if (dilated[p] && !occMask[p]) ringCount++;
  console.log(`[bake] template UV footprint: ${occCount} occupied texels (${(100 * occCount / (width * height)).toFixed(1)}%), ` +
    `${ringCount} dilation-ring texels, rasterized in ${((performance.now() - tRaster0) / 1000).toFixed(1)}s`);

  // Frame 0: copy verbatim.
  await copyFile(atlas0Path, join(OUT_DIR, atlasName(GOP_START)));
  const frame0Text = await readFile(join(SRC_DIR, fname(GOP_START)), "utf8");
  const frame0 = parseObj(frame0Text);
  await writeFile(join(OUT_DIR, fname(GOP_START)), writeObjText(frame0.positions, frame0.uvs, frame0.indices));

  const frameTimes = [];
  const range = ONLY_FRAME !== null ? [ONLY_FRAME] : Array.from({ length: frameCount - 1 }, (_, i) => i + 1);
  for (const f of range) {
    const tf0 = performance.now();
    const srcFileIdx = GOP_START + f;
    const srcText = await readFile(join(SRC_DIR, fname(srcFileIdx)), "utf8");
    const src = parseObj(srcText);
    const diag = bboxDiag(src.positions) || 1;
    const grid = new TriangleGrid(src.positions, src.indices, src.uvs, diag / 48);

    const atlasPath = join(SRC_DIR, atlasName(srcFileIdx));
    const srcAtlas = await decodePng(atlasPath, scratch);

    const outAtlas = new Uint8Array(width * height * 4);
    const deformed = deformedPositions[f];
    const tmpPt = new Float32Array(3);
    const tmpUv = new Float32Array(2);

    let sampled = 0;
    for (let p = 0; p < triId.length; p++) {
      const t = triId[p];
      if (t < 0) continue;
      const a = templateIndices[t] * 3, b = templateIndices[t + 1] * 3, c = templateIndices[t + 2] * 3;
      const wA = baryA[p], wB = baryB[p], wC = baryC[p];
      const px = wA * deformed[a] + wB * deformed[b] + wC * deformed[c];
      const py = wA * deformed[a + 1] + wB * deformed[b + 1] + wC * deformed[c + 1];
      const pz = wA * deformed[a + 2] + wB * deformed[b + 2] + wC * deformed[c + 2];
      grid.nearestInto(px, py, pz, tmpPt, 0, tmpUv, 0);
      const rgba = bilinearSample(srcAtlas.data, srcAtlas.width, srcAtlas.height, tmpUv[0], tmpUv[1]);
      const o = p * 4;
      outAtlas[o] = rgba[0]; outAtlas[o + 1] = rgba[1]; outAtlas[o + 2] = rgba[2]; outAtlas[o + 3] = 255;
      sampled++;
    }
    // gutter bleed: ring texels copy the nearest already-baked occupied texel.
    for (let p = 0; p < dilated.length; p++) {
      if (!dilated[p] || occMask[p]) continue;
      const src2 = gutter[p];
      if (src2 < 0) continue;
      const o = p * 4, so = src2 * 4;
      outAtlas[o] = outAtlas[so]; outAtlas[o + 1] = outAtlas[so + 1]; outAtlas[o + 2] = outAtlas[so + 2]; outAtlas[o + 3] = 255;
    }

    await writeFile(join(OUT_DIR, atlasName(srcFileIdx)), encodePng(outAtlas, width, height));
    await writeFile(join(OUT_DIR, fname(srcFileIdx)), writeObjText(deformed, templateUvs, templateIndices));

    const dt = performance.now() - tf0;
    frameTimes.push(dt);
    console.log(`[bake] frame ${f} (src ${fname(srcFileIdx)}): ${sampled} texels sampled in ${(dt / 1000).toFixed(2)}s`);
  }

  const totalMs = performance.now() - t0;
  console.log(`[bake] done: ${range.length} frame(s) baked in ${(totalMs / 1000).toFixed(1)}s ` +
    `(mean ${(frameTimes.reduce((s, x) => s + x, 0) / frameTimes.length / 1000).toFixed(2)}s/frame)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
