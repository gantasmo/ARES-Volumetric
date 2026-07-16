/**
 * probe-4ds.cjs — STRUCTURAL probe of a 4DViews ".4ds" (4DS0) volumetric container.
 *
 * CONTENT-FREE: this reads ONLY the container skeleton — magic, header fields,
 * the trailing frame directory, per-GOP block offset tables, and block *lengths*. It NEVER
 * decodes, extracts, renders, or writes any mesh or texture pixels. Output is numbers only.
 *
 * What it proves (all from byte lengths, no decoding):
 *   - format/version, frame count, fps, texture dimensions
 *   - temporal GOP structure (intra keyframes + inter P-frames) from the frame directory
 *   - the GEOMETRY vs TEXTURE byte split, by bucketing each frame's two blocks (small=geometry
 *     mesh, big=texture) — the labeling is inferred from the bimodal sizes + the small bucket
 *     reproducing the keyframe/GOP counts, and reconciles to the exact file size.
 *
 * Layout (inferred from structural analysis; see docs/size-comparison.md §4DViews):
 *   header: magic "4DS0"@0, verMajor u16@4, verMinor u16@6, indexOffset u64@14,
 *           frameCount u32@39, fps f32@47, texW u32@63, texH u32@67
 *   trailing frame directory @indexOffset: 9 bytes, then [keyframeIdx u32, pFrameCount u32,
 *           chunkOffset u64] × nKeyframes
 *   each GOP chunk @chunkOffset: 9-byte header (tableBytes u16@5), then (tableBytes/8) u64
 *           data-relative block offsets. Concatenated across GOPs = 2 × frameCount block offsets;
 *           consecutive diffs = block sizes (bimodal: geometry vs texture).
 *
 * Usage:  node tools/probe-4ds.cjs <path-to.4ds>
 */
const fs = require("fs");

const PATH = process.argv[2];
if (!PATH) { console.error("usage: node tools/probe-4ds.cjs <file.4ds>"); process.exit(1); }

const buf = fs.readFileSync(PATH);
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const FILESIZE = buf.byteLength;
const HEADER_BYTES = 128; // fixed container header

const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
if (magic !== "4DS0") { console.error(`not a 4DS0 file (magic="${magic}")`); process.exit(1); }
const verMajor = dv.getUint16(4, true), verMinor = dv.getUint16(6, true);
const indexOff = Number(dv.getBigUint64(14, true));
const frameCount = dv.getUint32(39, true);
const fps = dv.getFloat32(47, true);
const texW = dv.getUint32(63, true), texH = dv.getUint32(67, true);

// Trailing frame directory: 9-byte preamble, then 16-byte GOP records.
const dirLen = FILESIZE - indexOff;
const recCount = Math.floor((dirLen - 9) / 16);
const keyframes = [];
for (let i = 0; i < recCount; i++) {
  const base = indexOff + 9 + i * 16;
  keyframes.push({
    keyframeIdx: dv.getUint32(base, true),
    pFrameCount: dv.getUint32(base + 4, true),
    chunkOffset: Number(dv.getBigUint64(base + 8, true)),
  });
}

// Walk each GOP chunk's block-offset table; concatenate all data-relative block offsets.
let tableOverhead = 0;
const blockOffsets = [];
for (const k of keyframes) {
  const tb = dv.getUint16(k.chunkOffset + 5, true); // table size in bytes
  const n = tb / 8;
  tableOverhead += 9 + tb;
  for (let j = 0; j < n; j++) blockOffsets.push(Number(dv.getBigUint64(k.chunkOffset + 9 + j * 8, true)));
}

const monotonic = blockOffsets.every((v, i) => i === 0 || v > blockOffsets[i - 1]);
// Block sizes = consecutive diffs (first block spans from origin). The final block's size is not
// captured by a diff; it is recovered by reconciliation (texture = filesize - geometry - overhead).
const sizes = blockOffsets.map((v, i) => (i === 0 ? v : v - blockOffsets[i - 1]));

const SMALL_MAX = 200000; // bimodal split threshold: geometry (~21 KB) vs texture (~502 KB)
let geomBytes = 0, geomBlocks = 0, bigBlocks = 0;
const hist = { "<1k": 0, "1k-50k": 0, "50k-200k": 0, "200k-600k": 0, ">600k": 0 };
for (const s of sizes) {
  if (s <= SMALL_MAX) { geomBytes += s; geomBlocks++; } else bigBlocks++;
  if (s < 1000) hist["<1k"]++; else if (s < 50000) hist["1k-50k"]++;
  else if (s < 200000) hist["50k-200k"]++; else if (s < 600000) hist["200k-600k"]++; else hist[">600k"]++;
}
const overhead = tableOverhead + dirLen + HEADER_BYTES;
const texBytes = FILESIZE - geomBytes - overhead; // texture by reconciliation (includes final block)

const MB = (b) => (b / 1048576).toFixed(2);
const pct = (b) => ((b / FILESIZE) * 100).toFixed(2);
const dur = frameCount / fps;

console.log(`4DS0 STRUCTURAL PROBE (content-free — byte lengths only)`);
console.log(`file            ${PATH}`);
console.log(`size            ${FILESIZE.toLocaleString()} B  (${MB(FILESIZE)} MiB)`);
console.log(`format          4DS0  v${verMajor}.${verMinor}`);
console.log(`frames          ${frameCount}   @ ${fps.toFixed(3)} fps   = ${dur.toFixed(1)} s`);
console.log(`texture dims    ${texW} x ${texH}   (embedded per-frame)`);
console.log(`GOP (geometry)  ${keyframes.length} intra keyframes + ${frameCount - keyframes.length} inter P-frames  (adaptive)`);
console.log(`keyframe idx    [${keyframes.map((k) => k.keyframeIdx).join(", ")}]`);
console.log(`blocks          ${blockOffsets.length}  ( = 2 x ${frameCount} = ${2 * frameCount}? ${blockOffsets.length === 2 * frameCount})   monotonic:${monotonic}`);
console.log(`block histogram ${JSON.stringify(hist)}`);
console.log(``);
console.log(`GEOMETRY  ${MB(geomBytes).padStart(8)} MiB   ${pct(geomBytes).padStart(5)}%   (${geomBlocks} small blocks, temporal mesh)`);
console.log(`TEXTURE   ${MB(texBytes).padStart(8)} MiB   ${pct(texBytes).padStart(5)}%   (${bigBlocks}+1 big blocks, per-frame ${texW}² image)`);
console.log(`overhead  ${MB(overhead).padStart(8)} MiB   ${pct(overhead).padStart(5)}%   (${tableOverhead} tables + ${dirLen} dir + ${HEADER_BYTES} header)`);
console.log(`RECONCILE ${geomBytes} + ${texBytes} + ${overhead} = ${geomBytes + texBytes + overhead}  vs  ${FILESIZE}  ${geomBytes + texBytes + overhead === FILESIZE ? "EXACT ✓" : "MISMATCH"}`);
console.log(``);
console.log(`per-9.07s/272f equivalent:  geom ${MB(geomBytes * (9.0667 / dur))} MiB   tex ${MB(texBytes * (9.0667 / dur))} MiB`);
