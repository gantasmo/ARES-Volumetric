import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { deflateRawSync } from "node:zlib";
import {
  synthSplatClip, encodeSplatBlock, decodedSplatToFrame, splatAabb, mortonOrder, permuteSplatFrame, filterSplatFrame, transformSplatFrame,
  parseSpz, writeSpz, parseSplatFile, writeSplatFile, parseGltfSplat, writeGlbSplat, parsePly, parsePlySplat, writeSplatPly, isSplatPly,
  readZipEntries, webpSize, muxClip, synthClip, decodedMeshToFrame, writeObj, writeMeshPly,
} from "../dist/index.js";
import { decodeSplatBlock, decodeSplatPBlock, decodeGeometryBlock, Demuxer, GeometryProfile, BlockType, meshoptReady, dequantScale, shRestCoeffs, SH_C0 } from "@ares/core";

const CLI = fileURLToPath(new URL("../dist/cli.js", import.meta.url));

function frameClose(a, b, opts = {}) {
  const posTol = opts.pos ?? 1e-5, tol = opts.tol ?? 1e-5;
  assert.equal(a.count, b.count);
  assert.equal(a.shDegree, b.shDegree);
  let worstQ = 1;
  for (let i = 0; i < a.count; i++) {
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(a.positions[i * 3 + k] - b.positions[i * 3 + k]) <= posTol, `pos ${i}/${k}: ${a.positions[i * 3 + k]} vs ${b.positions[i * 3 + k]}`);
      assert.ok(Math.abs(Math.log(a.scales[i * 3 + k] / b.scales[i * 3 + k])) <= (opts.scaleLog ?? tol), `scale ${i}/${k}`);
      assert.ok(Math.abs(a.colors[i * 3 + k] - b.colors[i * 3 + k]) <= (opts.color ?? tol), `color ${i}/${k}: ${a.colors[i * 3 + k]} vs ${b.colors[i * 3 + k]}`);
    }
    assert.ok(Math.abs(a.opacities[i] - b.opacities[i]) <= (opts.opacity ?? tol), `opacity ${i}`);
    const d = Math.abs(a.rotations[i * 4] * b.rotations[i * 4] + a.rotations[i * 4 + 1] * b.rotations[i * 4 + 1] + a.rotations[i * 4 + 2] * b.rotations[i * 4 + 2] + a.rotations[i * 4 + 3] * b.rotations[i * 4 + 3]);
    worstQ = Math.min(worstQ, d);
  }
  assert.ok(worstQ >= (opts.quat ?? 0.99999), `rotation |dot| ${worstQ}`);
  if (a.sh || b.sh) {
    assert.ok(a.sh && b.sh);
    for (let i = 0; i < a.sh.length; i++) assert.ok(Math.abs(a.sh[i] - b.sh[i]) <= (opts.sh ?? tol), `sh ${i}`);
  }
}

test("splat block encode → decode round trip stays within quantization", async () => {
  await meshoptReady();
  const f = synthSplatClip(1, 30, 1).splatFrames[0];
  const box = splatAabb(f);
  const block = encodeSplatBlock(f, box, 14);
  const dec = decodeSplatBlock(block);
  assert.equal(dec.count, f.count);
  assert.equal(dec.shDegree, 1);
  const back = decodedSplatToFrame(dec, box, dequantScale(14));
  const diag = Math.hypot(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]);
  frameClose(f, back, { pos: diag / (1 << 14) + 1e-6, scaleLog: 1 / 32 + 1e-6, color: 1 / 255 + 1e-6, opacity: 1 / 255 + 1e-6, sh: 1 / 128 + 1e-6, quat: 0.99999 });
  // Block is far smaller than the raw floats (positions 6 B + attrs 12 B + sh 12 B ≈ 30 B/splat before meshopt).
  assert.ok(block.byteLength < f.count * 30, `block ${block.byteLength} B for ${f.count} splats`);
  // SH cap truncates to degree 0.
  const d0 = decodeSplatBlock(encodeSplatBlock(f, box, 14, 0));
  assert.equal(d0.shDegree, 0); assert.equal(d0.sh, undefined);
});

test("morton order is a permutation and the frame permutes consistently", () => {
  const f = synthSplatClip(1, 30, 0).splatFrames[0];
  const order = mortonOrder(f, splatAabb(f));
  assert.equal(order.length, f.count);
  assert.equal(new Set(order).size, f.count);
  const g = permuteSplatFrame(f, order);
  assert.equal(g.positions[0], f.positions[order[0] * 3]);
  assert.equal(g.opacities[5], f.opacities[order[5]]);
  const h = filterSplatFrame(f, (i) => i % 2 === 0);
  assert.equal(h.count, Math.ceil(f.count / 2));
});

test("SPZ v2 and v3 write → parse round trip", () => {
  const f = synthSplatClip(1, 30, 1).splatFrames[0];
  for (const version of [2, 3]) {
    const bytes = writeSpz(f, { version, fractionalBits: 14 });
    const back = parseSpz(bytes);
    assert.equal(back.shDegree, 1);
    // v2 "first three" stores xyz at 8 bits and rebuilds w from the norm: when w is small, rounding can
    // push |xyz| past 1 and w collapses to 0 (a known v2 limitation; v3's smallest-three fixes it).
    frameClose(f, back, { pos: 1 / (1 << 14) + 1e-6, scaleLog: 1 / 32 + 1e-6, color: 0.15 * 2 * SH_C0 / 255 * 3 + 0.01, opacity: 1 / 255 + 1e-6, sh: 1 / 128 + 1e-6, quat: version === 2 ? 0.99 : 0.99999 });
    // Re-encoding the parsed frame is stable: a second round trip lands on the same values.
    const again = parseSpz(writeSpz(back, { version, fractionalBits: 14 }));
    frameClose(back, again, { pos: 1e-6, scaleLog: 1e-6, color: 1e-6, opacity: 1e-6, sh: 1e-6, quat: version === 2 ? 0.99 : 0.999999 });
  }
  assert.throws(() => parseSpz(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])), /magic/);
});

test(".splat write → parse round trip", () => {
  const f = synthSplatClip(1, 30, 0).splatFrames[0];
  const back = parseSplatFile(writeSplatFile(f));
  frameClose(f, back, { pos: 1e-6, scaleLog: 1e-6, color: 1 / 255 + 1e-6, opacity: 1 / 255 + 1e-6, quat: 0.9999 });
});

test("glTF KHR_gaussian_splatting write → parse round trip is exact", () => {
  const f = synthSplatClip(1, 30, 1).splatFrames[0];
  const glb = writeGlbSplat(f);
  const back = parseGltfSplat(glb);
  frameClose(f, back, { pos: 1e-6, scaleLog: 1e-6, color: 1e-5, opacity: 1e-6, sh: 1e-6, quat: 0.999999 });
  const json = JSON.parse(new TextDecoder().decode(glb.subarray(20, 20 + new DataView(glb.buffer).getUint32(12, true))));
  assert.deepEqual(json.extensionsUsed, ["KHR_gaussian_splatting"]);
  assert.equal(json.meshes[0].primitives[0].mode, 0);
  assert.equal(json.meshes[0].primitives[0].extensions.KHR_gaussian_splatting.kernel, "ellipse");
  assert.ok("KHR_gaussian_splatting:SH_DEGREE_1_COEF_2" in json.meshes[0].primitives[0].attributes);
});

test("3DGS PLY write → parse round trip; mesh PLY attributes; splat detection", () => {
  const f = synthSplatClip(1, 30, 1).splatFrames[0];
  const ply = writeSplatPly(f);
  assert.ok(isSplatPly(ply));
  const back = parsePlySplat(ply);
  frameClose(f, back, { pos: 1e-6, scaleLog: 1e-5, color: 1e-5, opacity: 1e-5, sh: 1e-6, quat: 0.999999 });
  const ascii = `ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 0 0 1 0 0 255 0 0\n1 0 0 0 0 1 1 0 0 255 0\n0 1 0 0 0 1 0 1 0 0 255\n3 0 1 2\n`;
  const m = parsePly(new TextEncoder().encode(ascii));
  assert.equal(m.vertexCount, 3);
  assert.deepEqual(Array.from(m.indices), [0, 1, 2]);
  assert.deepEqual(Array.from(m.uvs), [0, 0, 1, 0, 0, 1]);
  assert.deepEqual(Array.from(m.normals.slice(0, 3)), [0, 0, 1]);
  assert.ok(Math.abs(m.colors[0] - 1) < 1e-6 && Math.abs(m.colors[4] - 1) < 1e-6);
  assert.ok(!isSplatPly(new TextEncoder().encode(ascii)));
});

test("SOG helpers: zip entries (stored + deflate) and WebP header sizes", () => {
  const files = [["meta.json", new TextEncoder().encode('{"version":2}'), 0], ["a.webp", new Uint8Array([1, 2, 3, 4, 5]), 8]];
  const parts = [], cd = [];
  let off = 0;
  for (const [name, data, method] of files) {
    const nb = new TextEncoder().encode(name);
    const payload = method === 8 ? new Uint8Array(deflateRawSync(data)) : data;
    const lh = new Uint8Array(30 + nb.length);
    const dv = new DataView(lh.buffer);
    dv.setUint32(0, 0x04034b50, true); dv.setUint16(8, method, true); dv.setUint32(18, payload.length, true); dv.setUint32(22, data.length, true); dv.setUint16(26, nb.length, true);
    lh.set(nb, 30);
    const ce = new Uint8Array(46 + nb.length);
    const cv = new DataView(ce.buffer);
    cv.setUint32(0, 0x02014b50, true); cv.setUint16(10, method, true); cv.setUint32(20, payload.length, true); cv.setUint32(24, data.length, true); cv.setUint16(28, nb.length, true); cv.setUint32(42, off, true);
    ce.set(nb, 46);
    parts.push(lh, payload); cd.push(ce);
    off += lh.length + payload.length;
  }
  const cdLen = cd.reduce((s, c) => s + c.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true); ev.setUint16(10, files.length, true); ev.setUint32(12, cdLen, true); ev.setUint32(16, off, true);
  const zip = new Uint8Array(off + cdLen + 22);
  let p = 0;
  for (const b of [...parts, ...cd, eocd]) { zip.set(b, p); p += b.length; }
  const entries = readZipEntries(zip);
  assert.equal(new TextDecoder().decode(entries.get("meta.json")), '{"version":2}');
  assert.deepEqual(Array.from(entries.get("a.webp")), [1, 2, 3, 4, 5]);

  const vp8l = new Uint8Array(32);
  vp8l.set(new TextEncoder().encode("RIFF"), 0); vp8l.set(new TextEncoder().encode("WEBP"), 8); vp8l.set(new TextEncoder().encode("VP8L"), 12);
  vp8l[20] = 0x2f;
  const w = 2048 - 1, h = 128 - 1; const bits = (w & 0x3fff) | ((h & 0x3fff) << 14);
  vp8l[21] = bits & 0xff; vp8l[22] = (bits >> 8) & 0xff; vp8l[23] = (bits >> 16) & 0xff; vp8l[24] = (bits >> 24) & 0xff;
  assert.deepEqual(webpSize(vp8l), { width: 2048, height: 128 });
  const vp8x = new Uint8Array(32);
  vp8x.set(new TextEncoder().encode("RIFF"), 0); vp8x.set(new TextEncoder().encode("WEBP"), 8); vp8x.set(new TextEncoder().encode("VP8X"), 12);
  vp8x[24] = 0xff; vp8x[25] = 0x03; vp8x[27] = 0x3f;
  assert.deepEqual(webpSize(vp8x), { width: 1024, height: 64 });
});

test("splat clip muxes as the splat profile and demuxes frame by frame", async () => {
  const clip = synthSplatClip(5, 30, 1);
  const bytes = await muxClip({ fps: 30, splatFrames: clip.splatFrames, gopLength: 2, shDegree: 1 });
  const file = Demuxer.parse(bytes);
  assert.equal(file.header.geometryProfile, GeometryProfile.SplatIPB);
  assert.equal(file.header.frameCount, 5);
  assert.equal(file.superblock.shDegree, 1);
  assert.equal(file.tracks[0].codecFourcc, "SPLT");
  assert.equal(file.gopIndex.length, 3);
  await meshoptReady();
  let frames = 0;
  for (const gop of file.gopIndex) {
    const chunk = Demuxer.chunkAt(file, gop);
    let cur = null;
    for (const b of Demuxer.geometryBlocks(file, chunk)) {
      cur = b.type === BlockType.GeometryI ? decodeSplatBlock(b.data) : decodeSplatPBlock(b.data, cur);
      assert.equal(cur.count, clip.splatFrames[0].count);
      frames++;
    }
  }
  assert.equal(frames, 5);
});

test("mesh clip still muxes and round-trips a frame to OBJ/PLY", async () => {
  const clip = synthClip("talk", 3, 30);
  const bytes = await muxClip({ fps: 30, frames: clip.frames, gopLength: 30 });
  const file = Demuxer.parse(bytes);
  assert.equal(file.header.geometryProfile, GeometryProfile.MeshIPB);
  assert.equal(file.superblock.quantBitsUv, 16);
  await meshoptReady();
  const chunk = Demuxer.chunkAt(file, file.gopIndex[0]);
  const g = decodeGeometryBlock(Demuxer.geometryBlocks(file, chunk)[0].data);
  const fr = decodedMeshToFrame(g, chunk.gopAabb, file.superblock.quantBitsPos, file.superblock.normalEncoding);
  const obj = writeObj(fr);
  assert.ok(obj.includes("\nv ") && obj.includes("\nvt ") && obj.includes("\nvn ") && obj.includes("\nf "));
  const ply = parsePly(writeMeshPly(fr));
  assert.equal(ply.vertexCount, g.vertexCount);
  assert.equal(ply.indices.length, g.indexCount);
});

test("transformSplatFrame rotates orientations with the positions", () => {
  const f = synthSplatClip(1, 30, 0).splatFrames[0];
  const before = Array.from(f.rotations.subarray(0, 4));
  // 90° about Y, uniform scale 2 (column-major).
  const c = Math.cos(Math.PI / 2), s = Math.sin(Math.PI / 2);
  const m = new Float32Array([2 * c, 0, -2 * s, 0, 0, 2, 0, 0, 2 * s, 0, 2 * c, 0, 1, 2, 3, 1]);
  const p0 = Array.from(f.positions.subarray(0, 3));
  const s0 = f.scales[0];
  transformSplatFrame(f, m);
  assert.ok(Math.abs(f.positions[0] - (2 * c * p0[0] + 2 * s * p0[2] + 1)) < 1e-4);
  assert.ok(Math.abs(f.scales[0] - 2 * s0) < 1e-6);
  const after = Array.from(f.rotations.subarray(0, 4));
  const dot = before[0] * after[0] + before[1] * after[1] + before[2] * after[2] + before[3] * after[3];
  assert.ok(Math.abs(dot) < 0.9, "rotation changed by the 90° turn");
});

test("CLI: synth splat → export (.spz/.glb/.ply/.splat) → encode the exports back", { timeout: 120000 }, () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-splat-"));
  try {
    const clip = join(dir, "s.ares");
    execFileSync(process.execPath, [CLI, "synth", "--shape", "splat", "--frames", "4", "--sh-degree", "1", "-o", clip], { stdio: "pipe" });
    const info = execFileSync(process.execPath, [CLI, "info", clip], { encoding: "utf8" });
    assert.match(info, /geometry splat/);
    const spzDir = join(dir, "spz"); mkdirSync(spzDir);
    for (let i = 0; i < 4; i++) execFileSync(process.execPath, [CLI, "export", clip, "--frame", String(i), "-o", join(spzDir, `frame-${i}.spz`)], { stdio: "pipe" });
    for (const ext of ["glb", "ply", "splat"]) execFileSync(process.execPath, [CLI, "export", clip, "--frame", "1", "-o", join(dir, `f1.${ext}`)], { stdio: "pipe" });
    assert.ok(parseGltfSplat(new Uint8Array(readFileSync(join(dir, "f1.glb")))).count > 1000);
    assert.ok(isSplatPly(new Uint8Array(readFileSync(join(dir, "f1.ply")))));
    const out = join(dir, "re.ares");
    const log = execFileSync(process.execPath, [CLI, "encode", spzDir, "-o", out, "--gop", "2", "--rotate", "0,90,0"], { encoding: "utf8" });
    assert.match(log, /4 SPZ splat frame/);
    const file = Demuxer.parse(new Uint8Array(readFileSync(out)));
    assert.equal(file.header.geometryProfile, GeometryProfile.SplatIPB);
    assert.equal(file.header.frameCount, 4);
    assert.equal(file.superblock.shDegree, 1);
    // Mesh export still works.
    const mesh = join(dir, "m.ares");
    execFileSync(process.execPath, [CLI, "synth", "--shape", "talk", "--frames", "2", "-o", mesh], { stdio: "pipe" });
    execFileSync(process.execPath, [CLI, "export", mesh, "--frame", "1", "-o", join(dir, "m1.obj")], { stdio: "pipe" });
    assert.ok(readFileSync(join(dir, "m1.obj"), "utf8").includes("\nf "));
    // Usage errors are loud, not silent.
    assert.throws(() => execFileSync(process.execPath, [CLI, "encode", spzDir, "-o", out, "--gop", "--fps", "30"], { stdio: "pipe" }));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
