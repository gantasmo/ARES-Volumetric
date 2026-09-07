import { test } from "node:test";
import assert from "node:assert/strict";
import { synthSplatClip, quantizeSplatFrame, encodeSplatStateBlock, encodeSplatPBlock, matchSplatsByIndex, matchSplatsNearest, orderForPFrame, splatAabb, muxClip, filterSplatFrame, emptySplatFrame } from "../dist/index.js";
import { decodeSplatBlock, decodeSplatPBlock, Demuxer, BlockType, meshoptReady, ByteWriter, ByteReader } from "@ares/core";

function sameState(a, b) {
  assert.equal(a.count, b.count);
  assert.equal(a.shDegree, b.shDegree);
  assert.deepEqual(Array.from(a.positionsQ), Array.from(b.positionsQ));
  assert.deepEqual(Array.from(a.attrs), Array.from(b.attrs));
  if (a.sh || b.sh) assert.deepEqual(Array.from(a.sh), Array.from(b.sh));
}
/** Multiset comparison on (position, attrs) so ordering differences do not matter. */
function keyset(s) {
  const out = [];
  for (let i = 0; i < s.count; i++) out.push([s.positionsQ[i * 4], s.positionsQ[i * 4 + 1], s.positionsQ[i * 4 + 2], s.attrs[i * 3], s.attrs[i * 3 + 1], s.attrs[i * 3 + 2]].join(","));
  return out.sort();
}

test("varint round trip", () => {
  const w = new ByteWriter(32);
  for (const v of [0, 1, 127, 128, 300, 65535, 2 ** 21, 2 ** 31 + 5]) w.varint(v);
  const r = new ByteReader(w.finish());
  for (const v of [0, 1, 127, 128, 300, 65535, 2 ** 21, 2 ** 31 + 5]) assert.equal(r.varint(), v);
});

test("index correspondence P-frame reproduces the decoder state exactly and is far smaller than intra", async () => {
  await meshoptReady();
  const clip = synthSplatClip(3, 60, 1);          // 60 fps → small per-frame motion
  const box = clip.splatFrames.reduce((b, f) => { const s = splatAabb(f); return b ? { min: b.min.map((v, i) => Math.min(v, s.min[i])), max: b.max.map((v, i) => Math.max(v, s.max[i])) } : s; }, null);
  const q0 = quantizeSplatFrame(clip.splatFrames[0], box, 14), q1 = quantizeSplatFrame(clip.splatFrames[1], box, 14);
  const match = matchSplatsByIndex(q0, q1);
  assert.ok(match && match.survivors === q1.count && match.births === 0 && match.deaths === 0);
  const { state, survivorPrev, deaths } = orderForPFrame(q0, q1, match);
  const iBlock = encodeSplatStateBlock(q1);
  const pBlock = encodeSplatPBlock(q0, state, survivorPrev, deaths);
  const dec = decodeSplatPBlock(pBlock, decodeSplatBlock(encodeSplatStateBlock(q0)));
  sameState(dec, state);
  sameState(dec, q1);                                   // index order preserved → identical to the direct quantization
  assert.ok(pBlock.byteLength < iBlock.byteLength * 0.8, `P ${pBlock.byteLength} vs I ${iBlock.byteLength}`);
});

test("nearest-neighbour correspondence with births and deaths reconstructs the same splat set", async () => {
  await meshoptReady();
  const base = synthSplatClip(1, 30, 0).splatFrames[0];
  const box = splatAabb(base);
  // Frame B: drop every 7th splat (deaths), jitter the rest slightly, append 50 new splats (births), shuffle.
  const kept = filterSplatFrame(base, (i) => i % 7 !== 0);
  for (let i = 0; i < kept.count * 3; i++) kept.positions[i] += (Math.sin(i) * 0.5) * 1e-3;
  const born = emptySplatFrame(50, 0);
  for (let i = 0; i < 50; i++) { born.positions[i * 3] = box.min[0] + (i / 50) * (box.max[0] - box.min[0]); born.positions[i * 3 + 1] = box.min[1]; born.positions[i * 3 + 2] = box.min[2]; born.scales.set([0.01, 0.01, 0.01], i * 3); born.rotations.set([0, 0, 0, 1], i * 4); born.opacities[i] = 0.5; born.colors.set([0.2, 0.4, 0.6], i * 3); }
  const merged = emptySplatFrame(kept.count + 50, 0);
  merged.positions.set(kept.positions); merged.positions.set(born.positions, kept.count * 3);
  merged.scales.set(kept.scales); merged.scales.set(born.scales, kept.count * 3);
  merged.rotations.set(kept.rotations); merged.rotations.set(born.rotations, kept.count * 4);
  merged.opacities.set(kept.opacities); merged.opacities.set(born.opacities, kept.count);
  merged.colors.set(kept.colors); merged.colors.set(born.colors, kept.count * 3);
  const perm = Uint32Array.from({ length: merged.count }, (_, i) => i).sort((a, b) => Math.sin(a * 7.1) - Math.sin(b * 7.1));
  const shuffled = filterSplatFrame(merged, () => true);   // copy
  const { permuteSplatFrame } = await import("../dist/index.js");
  const B = permuteSplatFrame(shuffled, perm);

  const qA = quantizeSplatFrame(base, box, 14), qB = quantizeSplatFrame(B, box, 14);
  const diagQ = Math.hypot(16383, 16383, 16383);
  const match = matchSplatsNearest(qA, qB, 0.01 * diagQ);
  assert.equal(match.births, 50);
  assert.equal(match.deaths, base.count - kept.count);
  assert.equal(match.survivors, kept.count);
  const { state, survivorPrev, deaths } = orderForPFrame(qA, qB, match);
  const dec = decodeSplatPBlock(encodeSplatPBlock(qA, state, survivorPrev, deaths), decodeSplatBlock(encodeSplatStateBlock(qA)));
  assert.deepEqual(keyset(dec), keyset(qB));
  // Survivors keep the previous order: the first survivor's previous index is the smallest kept one.
  assert.equal(survivorPrev[0], 1);
});

test("mux: dynamic splat GOPs carry I + P blocks and decode frame by frame; off = all intra", async () => {
  await meshoptReady();
  const clip = synthSplatClip(6, 60, 1);
  const bytes = await muxClip({ fps: 60, splatFrames: clip.splatFrames, gopLength: 3, shDegree: 1, splatTemporal: { mode: "auto" } });
  const file = Demuxer.parse(bytes);
  let iCount = 0, pCount = 0, frames = 0;
  const box = (g) => Demuxer.chunkAt(file, g).gopAabb;
  for (const gop of file.gopIndex) {
    const chunk = Demuxer.chunkAt(file, gop);
    let cur = null;
    for (const b of Demuxer.geometryBlocks(file, chunk)) {
      if (b.type === BlockType.GeometryI) { cur = decodeSplatBlock(b.data); iCount++; }
      else { cur = decodeSplatPBlock(b.data, cur); pCount++; }
      assert.equal(cur.count, clip.splatFrames[0].count);
      // Reconstruction matches a direct quantization of the source frame over the same GOP box.
      const direct = quantizeSplatFrame(clip.splatFrames[frames], box(gop), 14, 1);
      assert.deepEqual(keyset(cur), keyset(direct));
      frames++;
    }
  }
  assert.equal(frames, 6);
  assert.equal(iCount, 2); assert.equal(pCount, 4);
  const intra = await muxClip({ fps: 60, splatFrames: clip.splatFrames, gopLength: 3, shDegree: 1, splatTemporal: { mode: "off" } });
  assert.ok(bytes.length < intra.length * 0.8, `temporal ${bytes.length} vs intra ${intra.length}`);
});
