import { test } from "node:test";
import assert from "node:assert/strict";
import { Demuxer, AresParseError, decodeGeometryBlock, decodePFrameBlock, decodeSplatBlock, meshoptReady, BlockType, GeometryProfile, H, crc32 } from "../dist/index.js";
import { synthClip, synthSplatClip, muxClip } from "@ares/encoder";

const rnd = (seed) => { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); };

async function decodeAll(bytes) {
  const file = Demuxer.parse(bytes);
  let frames = 0;
  for (const gop of file.gopIndex) {
    const chunk = Demuxer.chunkAt(file, gop);
    const blocks = Demuxer.geometryBlocks(file, chunk);
    let posQ = null;
    for (const b of blocks) {
      if (file.header.geometryProfile === GeometryProfile.SplatIPB) decodeSplatBlock(b.data);
      else if (b.type === BlockType.GeometryI) posQ = decodeGeometryBlock(b.data).positionsQ;
      else posQ = decodePFrameBlock(b.data, posQ).positionsQ;
      frames++;
    }
    Demuxer.textureFrames(file, chunk);
  }
  return { file, frames };
}

test("mesh clip: mux → demux → decode every frame (temporal I+P)", async () => {
  await meshoptReady();
  const clip = synthClip("talk", 6, 30);
  const bytes = await muxClip({ fps: 30, frames: clip.frames, gopLength: 3 });
  const { file, frames } = await decodeAll(bytes);
  assert.equal(frames, 6);
  assert.equal(file.gopIndex.length, 2);
  assert.equal(file.header.frameCount, 6);
  assert.equal(Number(file.header.durationUs), 200000);
  // Stable topology → I + P frames per GOP.
  const chunk = Demuxer.chunkAt(file, file.gopIndex[0]);
  const types = Demuxer.geometryBlocks(file, chunk).map((b) => b.type);
  assert.deepEqual(types, [BlockType.GeometryI, BlockType.GeometryPB, BlockType.GeometryPB]);
});

test("header CRC and version are enforced; truncation is a parse error, not a crash", async () => {
  const clip = synthClip("object", 2, 30);
  const bytes = await muxClip({ fps: 30, frames: clip.frames, gopLength: 30 });
  const bad = bytes.slice();
  bad[H.fps] ^= 0x40;                                   // header byte, CRC no longer matches
  assert.throws(() => Demuxer.parse(bad), AresParseError);
  const vmaj = bytes.slice();
  vmaj[H.versionMajor] = 9;
  new DataView(vmaj.buffer).setUint32(H.headerCrc32, crc32(vmaj, 0, H.headerCrc32), true);
  assert.throws(() => Demuxer.parse(vmaj), /version_major/);
  assert.throws(() => Demuxer.parse(bytes.subarray(0, 40)), AresParseError);
  const cut = bytes.subarray(0, bytes.length - 100);   // last chunk truncated
  assert.throws(() => { const f = Demuxer.parse(cut); for (const g of f.gopIndex) Demuxer.chunkAt(f, g); }, AresParseError);
});

test("fuzz: random corruption never hangs or escapes as a non-Error", { timeout: 120000 }, async () => {
  await meshoptReady();
  const mesh = await muxClip({ fps: 30, frames: synthClip("talk", 3, 30).frames, gopLength: 3 });
  const splat = await muxClip({ fps: 30, splatFrames: synthSplatClip(2, 30, 1).splatFrames, gopLength: 2 });
  const r = rnd(42);
  let ok = 0, rejected = 0;
  const t0 = Date.now();
  for (let iter = 0; iter < 240; iter++) {
    const src = iter % 2 ? splat : mesh;
    const c = src.slice();
    const flips = 1 + Math.floor(r() * 6);
    for (let k = 0; k < flips; k++) {
      // Skip the header (its CRC catches those) — the interesting surface is the body.
      const at = 64 + Math.floor(r() * (c.length - 64));
      c[at] = r() < 0.5 ? (c[at] ^ (1 << Math.floor(r() * 8))) : Math.floor(r() * 256);
    }
    const t = Date.now();
    try { await decodeAll(c); ok++; }
    catch (e) { assert.ok(e instanceof Error, `non-Error thrown: ${e}`); rejected++; }
    assert.ok(Date.now() - t < 5000, `iteration ${iter} took ${Date.now() - t} ms`);
  }
  assert.equal(ok + rejected, 240);
  assert.ok(Date.now() - t0 < 110000);
});
