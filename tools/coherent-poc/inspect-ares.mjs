/**
 * Task K POC measurement: parse an .ares file with @ares/core's Demuxer and report, per chunk,
 * the geometry block-type sequence (GeometryI vs GeometryPB) + byte sizes, plus texture track
 * bytes. Confirms whether a bake is really I+P (temporal) vs all-intra, and gives bytes/frame
 * for the geometry track alone (texture track reported separately).
 */
import { readFile } from "node:fs/promises";
import { Demuxer, BlockType } from "../../packages/core/dist/index.js";

const path = process.argv[2];
if (!path) { console.error("usage: node inspect-ares.mjs <file.ares>"); process.exit(1); }

const buf = await readFile(path);
const file = Demuxer.parse(buf);
console.log(`[inspect] ${path}: ${file.header.frameCount} frames @ ${file.header.fps}fps, ${file.gopIndex.length} chunk(s)`);

let totalGeomBytes = 0, totalTexBytes = 0, iBlocks = 0, pBlocks = 0, iBytes = 0, pBytes = 0;
const perChunk = [];
for (const gop of file.gopIndex) {
  const chunk = Demuxer.chunkAt(file, gop);
  const geomBlocks = Demuxer.geometryBlocks(file, chunk);
  let chunkGeomBytes = 0, chunkI = 0, chunkP = 0;
  for (const b of geomBlocks) {
    chunkGeomBytes += b.data.byteLength;
    if (b.type === BlockType.GeometryI) { iBlocks++; iBytes += b.data.byteLength; chunkI++; }
    else if (b.type === BlockType.GeometryPB) { pBlocks++; pBytes += b.data.byteLength; chunkP++; }
  }
  const texBlockDir = chunk.blocks.find((bd) => bd.type === BlockType.TextureColor);
  const texBytes = texBlockDir ? texBlockDir.length : 0;
  totalGeomBytes += chunkGeomBytes;
  totalTexBytes += texBytes;
  perChunk.push({ frameStart: gop.frameStart, frameCount: gop.frameCount, chunkI, chunkP, chunkGeomBytes, texBytes });
}

for (const c of perChunk) {
  console.log(`  chunk frameStart=${c.frameStart} frames=${c.frameCount}: ${c.chunkI} I-block(s) + ${c.chunkP} P-block(s), geom=${c.chunkGeomBytes}B tex=${c.texBytes}B`);
}
console.log(`[inspect] TOTAL geometry: ${totalGeomBytes} bytes (${iBlocks} I-blocks=${iBytes}B, ${pBlocks} P-blocks=${pBytes}B) -> ${(totalGeomBytes / file.header.frameCount).toFixed(1)} B/frame`);
console.log(`[inspect] TOTAL texture:  ${totalTexBytes} bytes -> ${(totalTexBytes / file.header.frameCount).toFixed(1)} B/frame`);
console.log(`[inspect] TOTAL file: geometry+texture = ${totalGeomBytes + totalTexBytes} bytes (container total incl. headers is larger: ${buf.byteLength} bytes)`);
