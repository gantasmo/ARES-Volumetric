import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { parseOggPackets, oggOpusToPackets, opusPacketSamples, parseOpusHead, buildAudioBlock, packetsInWindow, transcodeToOpus, ffmpegAvailable, muxClip, synthClip, writeObj, decodedMeshToFrame } from "../dist/index.js";
import { Demuxer, HeaderFlags, TrackType, parseAudioBlock, decodeGeometryBlock, meshoptReady } from "@ares/core";

const CLI = fileURLToPath(new URL("../dist/cli.js", import.meta.url));

/** Build one Ogg page holding the given packets (no continuation; CRC left zero — the parser does not verify it). */
function oggPage(packets, serial, seq, granule) {
  const segs = [];
  for (const p of packets) { let n = p.length; while (n >= 255) { segs.push(255); n -= 255; } segs.push(n); }
  const total = packets.reduce((s, p) => s + p.length, 0);
  const page = new Uint8Array(27 + segs.length + total);
  const dv = new DataView(page.buffer);
  page.set([0x4f, 0x67, 0x67, 0x53], 0);
  page[4] = 0; page[5] = seq === 0 ? 2 : 0;
  dv.setBigInt64(6, BigInt(granule), true);
  dv.setUint32(14, serial, true); dv.setUint32(18, seq, true); dv.setUint32(22, 0, true);
  page[26] = segs.length;
  page.set(segs, 27);
  let o = 27 + segs.length;
  for (const p of packets) { page.set(p, o); o += p.length; }
  return page;
}
function opusHead(channels = 2, preSkip = 312) {
  const b = new Uint8Array(19);
  b.set(new TextEncoder().encode("OpusHead"), 0);
  b[8] = 1; b[9] = channels;
  const dv = new DataView(b.buffer);
  dv.setUint16(10, preSkip, true); dv.setUint32(12, 48000, true); dv.setInt16(16, 0, true); b[18] = 0;
  return b;
}
function opusTags() {
  const v = new TextEncoder().encode("test");
  const b = new Uint8Array(8 + 4 + v.length + 4);
  b.set(new TextEncoder().encode("OpusTags"), 0);
  new DataView(b.buffer).setUint32(8, v.length, true); b.set(v, 12);
  return b;
}
/** A fake 20 ms CELT packet (TOC config 19, code 0) of `len` bytes. */
const fakePacket = (len, fill = 0x55) => { const p = new Uint8Array(len); p.fill(fill); p[0] = 0x98; return p; };

test("opus TOC → samples; Ogg pages → packets (incl. 255-byte lacing) → timed packets", () => {
  assert.equal(opusPacketSamples(new Uint8Array([0x00])), 480);           // SILK NB 10 ms
  assert.equal(opusPacketSamples(new Uint8Array([0x98])), 960);           // CELT FB 20 ms
  assert.equal(opusPacketSamples(new Uint8Array([0x80])), 120);           // CELT NB 2.5 ms
  assert.equal(opusPacketSamples(new Uint8Array([0x99])), 1920);          // code 1 = 2 frames
  assert.equal(opusPacketSamples(new Uint8Array([0x9b, 0x03])), 2880);    // code 3, 3 frames
  const head = parseOpusHead(opusHead(1, 100));
  assert.equal(head.channels, 1); assert.equal(head.preSkip, 100); assert.equal(head.inputSampleRate, 48000);

  const big = fakePacket(700, 0x11);   // spans three lacing segments
  const pages = [oggPage([opusHead()], 7, 0, 0), oggPage([opusTags()], 7, 1, 0), oggPage([fakePacket(60), big, fakePacket(40)], 7, 2, 2880)];
  const stream = new Uint8Array(pages.reduce((s, p) => s + p.length, 0));
  let o = 0; for (const p of pages) { stream.set(p, o); o += p.length; }
  const pk = parseOggPackets(stream);
  assert.equal(pk.length, 5);
  assert.equal(pk[3].data.length, 700);
  assert.ok(pk[3].data.every((v, i) => (i === 0 ? v === 0x98 : v === 0x11)));
  const { head: h2, packets, durationUs } = oggOpusToPackets(stream);
  assert.equal(h2.preSkip, 312);
  assert.equal(packets.length, 3);
  assert.equal(packets[0].ptsUs, 0);                                   // starts inside the pre-skip → clamped to 0
  assert.equal(packets[1].ptsUs, Math.round(((960 - 312) / 48000) * 1e6));
  assert.equal(packets[0].durationUs, 20000);
  assert.equal(durationUs, Math.round(((3 * 960 - 312) / 48000) * 1e6));
});

test("audio block round trip and chunk windowing", () => {
  const packets = Array.from({ length: 10 }, (_, i) => ({ data: fakePacket(20 + i, i), ptsUs: i * 20000, durationUs: 20000 }));
  const win = packetsInWindow(packets, 40000, 120000);
  assert.deepEqual(win.map((p) => p.ptsUs), [40000, 60000, 80000, 100000]);
  const block = buildAudioBlock(win, 40000);
  const back = parseAudioBlock(block, 40000);
  assert.equal(back.length, 4);
  assert.deepEqual(back.map((p) => p.ptsUs), [40000, 60000, 80000, 100000]);
  assert.deepEqual(Array.from(back[2].data), Array.from(win[2].data));
  assert.equal(back[0].durationUs, 20000);
});

test("mux with an audio track: flag, track dir, packets land in their chunks, tail kept", async () => {
  await meshoptReady();
  const clip = synthClip("talk", 6, 30);      // 6 frames @ 30 fps = 200 ms; gop 3 → 2 chunks of 100 ms
  const packets = Array.from({ length: 15 }, (_, i) => ({ data: fakePacket(30, i), ptsUs: i * 20000, durationUs: 20000 })); // 300 ms of audio
  const bytes = await muxClip({ fps: 30, frames: clip.frames, gopLength: 3, audio: { fourcc: "OPUS", head: parseOpusHead(opusHead(2)), codecConfig: opusHead(2), packets, durationUs: 300000, sampleRate: 48000, channels: 2 } });
  const file = Demuxer.parse(bytes);
  assert.ok(file.header.headerFlags & HeaderFlags.HasAudio);
  const at = Demuxer.audioTrack(file);
  assert.ok(at && at.fourcc === "OPUS" && at.channels === 2 && at.codecConfig.length === 19);
  assert.equal(file.tracks.filter((t) => t.trackType === TrackType.Audio).length, 1);
  const perChunk = file.gopIndex.map((g) => Demuxer.audioPackets(file, Demuxer.chunkAt(file, g)));
  assert.equal(perChunk[0].length, 5);                       // 0..80 ms
  assert.equal(perChunk[1].length, 10);                      // 100 ms.. plus the tail beyond the video end
  const all = perChunk.flat();
  assert.deepEqual(all.map((p) => p.ptsUs), packets.map((p) => p.ptsUs));
  // Geometry still decodes alongside.
  const g = decodeGeometryBlock(Demuxer.geometryBlocks(file, Demuxer.chunkAt(file, file.gopIndex[0]))[0].data);
  assert.ok(g.vertexCount > 0);
});

test("ffmpeg: WAV sine → Opus packets → CLI encode with --audio → info reports the track", { timeout: 120000 }, async (t) => {
  if (!(await ffmpegAvailable())) { t.skip("ffmpeg not available"); return; }
  const dir = mkdtempSync(join(tmpdir(), "ares-audio-"));
  try {
    // 1.0 s, 48 kHz, mono, 16-bit sine at 440 Hz.
    const rate = 48000, n = rate;
    const wav = new Uint8Array(44 + n * 2);
    const dv = new DataView(wav.buffer);
    const str = (o, s) => wav.set(new TextEncoder().encode(s), o);
    str(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt "); dv.setUint32(16, 16, true);
    dv.setUint16(20, 1, true); dv.setUint16(22, 1, true); dv.setUint32(24, rate, true); dv.setUint32(28, rate * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    str(36, "data"); dv.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.round(Math.sin((2 * Math.PI * 440 * i) / rate) * 12000), true);
    const wavPath = join(dir, "sine.wav");
    writeFileSync(wavPath, wav);
    const track = await transcodeToOpus(wavPath, { bitrateKbps: 64 });
    assert.equal(track.channels, 1);
    assert.ok(track.packets.length >= 45 && track.packets.length <= 55, `${track.packets.length} packets`);
    assert.ok(Math.abs(track.durationUs - 1e6) < 80000, `duration ${track.durationUs}`);
    // Frames dir from a synth mesh clip, then encode with the audio.
    await meshoptReady();
    const mesh = await muxClip({ fps: 30, frames: synthClip("talk", 2, 30).frames, gopLength: 30 });
    const file = Demuxer.parse(mesh);
    const frames = join(dir, "frames"); mkdirSync(frames);
    const chunk = Demuxer.chunkAt(file, file.gopIndex[0]);
    const blocks = Demuxer.geometryBlocks(file, chunk);
    writeFileSync(join(frames, "mesh-f00001.obj"), writeObj(decodedMeshToFrame(decodeGeometryBlock(blocks[0].data), chunk.gopAabb, 14, 1)));
    writeFileSync(join(frames, "mesh-f00002.obj"), writeObj(decodedMeshToFrame(decodeGeometryBlock(blocks[0].data), chunk.gopAabb, 14, 1)));
    const out = join(dir, "a.ares");
    const log = execFileSync(process.execPath, [CLI, "encode", frames, "-o", out, "--no-texture", "--audio", wavPath, "--audio-bitrate", "64"], { encoding: "utf8" });
    assert.match(log, /audio: sine\.wav → Opus 64 kb\/s, mono/);
    assert.match(log, /audio: OPUS 48 kHz mono/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
