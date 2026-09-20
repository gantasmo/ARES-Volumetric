import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, readFileSync, writeFileSync, truncateSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  writeDepthRun, openDepthRun, ffmpegAvailable, ffmpegPath, rasterizeDepth, outwardSign,
  cropIntrinsics, scaleIntrinsics, fovOf, resampleDepth, compactMesh, shellPieces, backAtlasUvs, spreadFrames, focalScales,
} from "../dist/index.js";
import { Demuxer } from "@ares/core";

/*
 * `ares depth --volumetric` on a synthetic volumetric run: a 192x128 map, 4 frames, the SAM 3D Body
 * probe mesh (fixtures/mhr-probe.bin, 18,439 vertices) walking 2 cm a frame at 3 m, a second "person"
 * in the mask with no body, the metric target equal to the body's front surface on a 6 m backdrop,
 * the estimate its inverse, and camera-facing normals from the body's faces.
 */
const CLI = fileURLToPath(new URL("../dist/cli.js", import.meta.url));
const FIXTURE = fileURLToPath(new URL("./fixtures/mhr-probe.bin", import.meta.url));
const W = 192, H = 128, P = W * H, N = 4;
// Square pixels on a 192x128 source: fovY 40 degrees. The body covers about 2,500 map pixels, above
// fitTileField's 200 samples on its stride-2 lattice.
const FOCAL = H / 2 / Math.tan((20 * Math.PI) / 180);
const NORM = { fx: FOCAL / W, fy: FOCAL / H, cx: 0.5, cy: 0.5 };
const K = { fx: FOCAL, fy: FOCAL, cx: W / 2, cy: H / 2 };
const BLOB = { x0: 4, x1: 16, y0: 60, y1: 92 };     // the second person: mask only, no body

let mhrCache = null;
function mhr() {
  if (mhrCache) return mhrCache;
  const b = readFileSync(FIXTURE);
  assert.equal(b.toString("latin1", 0, 4), "MHR1");
  const V = b.readUInt32LE(4), F = b.readUInt32LE(8);
  const body = new Float32Array(V * 3);
  for (let i = 0; i < V * 3; i++) body[i] = b.readFloatLE(16 + i * 4);
  const faces = new Uint32Array(F * 3);
  for (let i = 0; i < F * 3; i++) faces[i] = b.readUInt16LE(16 + V * 12 + i * 2);
  // Recentred: the body's bounding-box centre on the optical axis at 3 m.
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < V; v++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], body[v * 3 + k]); hi[k] = Math.max(hi[k], body[v * 3 + k]); }
  const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  for (let v = 0; v < V; v++) { body[v * 3] -= c[0]; body[v * 3 + 1] -= c[1]; body[v * 3 + 2] += 3 - c[2]; }
  mhrCache = { V, F, body, faces, height: hi[1] - lo[1] };
  return mhrCache;
}

function meta(over = {}) {
  return {
    schema: "ares-depth/1", engine: "service", model: "depth-anything/Video-Depth-Anything-Small", modelKey: "video-small",
    kind: "relative-disparity", temporal: "model", width: W, height: H, frames: N, fps: 30,
    sampling: { fps: null, maxFrames: N }, video: "/tmp/clip.mp4",
    sourceFps: 30, sourceWidth: W, sourceHeight: H, sourceFrames: 6, sourceDurationS: 0.2,
    msPerFrame: 7.5, device: "cuda", dtype: "fp16", done: true,
    mask: { file: "mask.u8", prompt: "person", engine: "sam3-text-tracker", coverage: 0.2 },
    intrinsics: { ...NORM, source: "moge-2", fovY: 40, fovX: 2 * Math.atan(W / 2 / FOCAL) * 180 / Math.PI, frames: N },
    ...over,
  };
}

/** The arrays of a synthetic volumetric run (see the file header). */
function volumetricArrays() {
  const { V, body: b0, faces } = mhr();
  const sign = outwardSign(b0, faces);
  const maps = new Float32Array(N * P), mask = new Uint8Array(N * P), metric = new Float32Array(N * P);
  const normals = new Int8Array(N * P * 3), body = new Float32Array(N * V * 3), intr = new Float32Array(N * 4);
  const zb = new Float32Array(P), face = new Int32Array(P);
  for (let t = 0; t < N; t++) {
    const bt = body.subarray(t * V * 3, (t + 1) * V * 3);
    bt.set(b0);
    for (let v = 0; v < V; v++) bt[v * 3] += 0.02 * t;
    rasterizeDepth(bt, faces, W, H, K, zb, face);
    for (let i = 0; i < P; i++) {
      const x = i % W, y = (i - x) / W, o = t * P + i;
      const inBlob = x >= BLOB.x0 && x < BLOB.x1 && y >= BLOB.y0 && y < BLOB.y1;
      const onBody = zb[i] < Infinity;
      metric[o] = onBody ? zb[i] : inBlob ? 5 : 6;
      maps[o] = 1 / metric[o];
      mask[o] = onBody || inBlob ? 255 : 0;
      let n = [0, 0, -1];
      if (onBody) {
        const f = face[i], a = faces[f * 3] * 3, bb = faces[f * 3 + 1] * 3, c = faces[f * 3 + 2] * 3;
        const ux = bt[bb] - bt[a], uy = bt[bb + 1] - bt[a + 1], uz = bt[bb + 2] - bt[a + 2];
        const vx = bt[c] - bt[a], vy = bt[c + 1] - bt[a + 1], vz = bt[c + 2] - bt[a + 2];
        n = [(uy * vz - uz * vy) * sign, (uz * vx - ux * vz) * sign, (ux * vy - uy * vx) * sign];
        const l = Math.hypot(...n) || 1;
        n = n.map((q) => q / l);
      }
      for (let k = 0; k < 3; k++) normals[o * 3 + k] = Math.round(n[k] * 127);
    }
    intr.set([NORM.fx, NORM.fy, NORM.cx, NORM.cy], t * 4);
  }
  return { maps, extra: { mask, metric, normals, intrinsics: intr, body, bodyFaces: faces, bodyValid: new Uint8Array(N).fill(1) } };
}

async function makeVolumetricRun(dir, over = {}, drop = [], edit = null) {
  const { maps, extra } = volumetricArrays();
  for (const k of drop) delete extra[k];
  if (edit) edit(extra);
  await writeDepthRun(dir, meta(over), maps, extra);
}

/* ------------------------------------------------------------------------------------------------ */

test("depth-io: the volumetric keys round-trip through writeDepthRun and openDepthRun, and a short file is refused", async () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-volio-"));
  try {
    const { maps, extra } = volumetricArrays();
    await writeDepthRun(dir, meta(), maps, extra);
    const run = await openDepthRun(dir);
    try {
      assert.ok(run.hasMask && run.hasMetric && run.hasNormals && run.hasBody);
      const k = run.mapIntrinsics;
      assert.ok(Math.abs(k.fx - FOCAL) < 1e-4 && Math.abs(k.fy - FOCAL) < 1e-4 && k.cx === W / 2 && k.cy === H / 2);
      const { V } = mhr();
      assert.equal(run.meta.body.vertices, V);
      const two = new Float32Array(2 * V * 3);
      run.readBody(1, 2, two);
      assert.deepEqual([...two.subarray(0, 30)], [...extra.body.subarray(V * 3, V * 3 + 30)]);
      const m = new Float32Array(P);
      run.readMetric(3, 1, m);
      assert.deepEqual([...m], [...extra.metric.subarray(3 * P, 4 * P)]);
      const n = new Int8Array(P * 3);
      run.readNormals(2, 1, n);
      assert.deepEqual([...n.subarray(0, 300)], [...extra.normals.subarray(2 * P * 3, 2 * P * 3 + 300)]);
      assert.deepEqual([...run.bodyFaces().subarray(0, 12)], [...extra.bodyFaces.subarray(0, 12)]);
      assert.deepEqual([...run.bodyValid()], [1, 1, 1, 1]);
      const pf = run.perFrameIntrinsics();
      assert.equal(pf.length, N * 4);
      assert.ok(Math.abs(pf[5] - NORM.fy) < 1e-6);
      assert.throws(() => run.readBody(3, 2, new Float32Array(2 * V * 3)), /outside 0\.\.4/);
    } finally { run.close(); }
    truncateSync(join(dir, "body.f32"), 3 * mhr().V * 12);
    await assert.rejects(() => openDepthRun(dir), /body\.f32: holds \d+ bytes, \d+ needed/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("volumetric helpers: crop and grid intrinsics, depth resample, compaction, shell pieces, back uvs", () => {
  // A letterbox crop keeps the focal and moves the principal point by the crop origin.
  const c = cropIntrinsics(NORM, W, H, { x: 0, y: 8 });
  assert.ok(Math.abs(c.fx - FOCAL) < 1e-9 && c.cy === H / 2 - 8 && c.cx === W / 2);
  assert.ok(Math.abs(fovOf(K, W, H).fovY - 40) < 1e-9);
  const g = scaleIntrinsics(K, W, H, W / 2, H / 2);
  assert.ok(Math.abs(g.fx - FOCAL / 2) < 1e-9 && g.cx === W / 4);
  assert.throws(() => cropIntrinsics({ ...NORM, fx: 0 }, W, H, { x: 0, y: 0 }), /fx must be a finite number > 0, got 0/);

  // 4x2 -> 2x1: a cell with half its samples valid averages them, under half is invalid.
  const z = new Float32Array([2, 4, 0, 0, 2, 0, 0, 3]);
  assert.deepEqual([...resampleDepth(z, 4, 2, 2, 1)], [Math.fround(8 / 3), 0]);
  assert.deepEqual([...resampleDepth(new Float32Array([1, 2]), 2, 1, 4, 1)], [1, 1, 2, 2]);

  // Two quads apart: the 4-vertex piece is dropped below minVerts 5, the 6-vertex one is the largest.
  const idx = new Uint32Array([0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6, 6, 7, 8, 7, 9, 8]);
  const pc = shellPieces(idx, 11, 5);
  assert.equal(pc.pieces, 2);
  assert.equal(pc.dropped, 4);
  assert.deepEqual([...pc.indices], [...idx.subarray(6)]);
  const pos = new Float32Array(11 * 3).map((_, i) => i);
  const cm = compactMesh({ positions: pos, uvs: new Float32Array(22), indices: pc.indices });
  assert.equal(cm.positions.length / 3, 6);
  assert.deepEqual([...cm.positions.subarray(0, 3)], [12, 13, 14]);
  assert.equal(Math.max(...cm.indices), 5);

  // Region texel centres -> atlas: u over the width, v below the frame region.
  assert.deepEqual([...backAtlasUvs(new Float32Array([0.5, 0.5, 2.5, 2.5]), 64, 64, 128)], [0.5 / 64, 64.5 / 128, 2.5 / 64, 66.5 / 128]);
  assert.deepEqual(spreadFrames(90, 5), [9, 27, 45, 63, 81]);
  assert.deepEqual(spreadFrames(2, 5), [0, 1]);
});

test("focalScales: clip fy over each frame's fy, 1 on a NaN or non-positive row; bad arguments name the value", () => {
  const k = new Float32Array([0.9, 1.5, 0.5, 0.5, 0.9, 2, 0.5, 0.5, 0.9, NaN, 0.5, 0.5, 0.9, 0, 0.5, 0.5]);
  const s = focalScales(1.6, k, 4);
  assert.ok(Math.abs(s[0] - 1.6 / Math.fround(1.5)) < 1e-12 && Math.abs(s[1] - 0.8) < 1e-12, `${s[0]}, ${s[1]}`);
  assert.equal(s[2], 1); assert.equal(s[3], 1);
  assert.deepEqual([...focalScales(1.6, null, 3)], [1, 1, 1]);
  assert.throws(() => focalScales(0, k, 4), /clipFy must be a finite number > 0, got 0/);
  assert.throws(() => focalScales(1.6, k, 5), /perFrame holds 16 values, 20 needed/);
});

test("CLI: ares depth --volumetric puts each frame's metric z on the clip focal", { timeout: 180000 }, async () => {
  // MoGe-2 couples depth to its per-frame focal (z grows with fy). Frame t here reports fy * f[t] and
  // a target f[t] times farther; with the fields unsmoothed, the anchor stays at 1 only when every
  // frame is carried back onto the clip focal.
  const f = [1.08, 0.93, 1.05, 0.95];
  const dir = mkdtempSync(join(tmpdir(), "ares-vol-"));
  try {
    const runDir = join(dir, "run");
    await makeVolumetricRun(runDir, {}, [], (extra) => {
      for (let t = 0; t < N; t++) {
        extra.intrinsics[t * 4 + 1] = NORM.fy * f[t];
        for (let i = t * P; i < (t + 1) * P; i++) extra.metric[i] *= f[t];
      }
    });
    const video = join(dir, "clip.mp4");
    writeFileSync(video, Buffer.alloc(1024));
    const out = join(dir, "vol.ares");
    const log = execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", out, "--volumetric", "--no-texture", "--no-audio", "--fit-smooth", "0"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    assert.match(log, /\[ares\] volumetric: metric z on the clip focal .*per-frame factor 0\.9259\.\.1\.0753/);
    const v = JSON.parse(readFileSync(out + ".meta.json", "utf8")).volumetric;
    assert.ok(Math.abs(v.intrinsics.metricFocal.factorMin - 1 / 1.08) < 1e-4 && Math.abs(v.intrinsics.metricFocal.factorMax - 1 / 0.93) < 1e-4,
      JSON.stringify(v.intrinsics.metricFocal));
    assert.ok(Math.abs(v.anchor.scaleMin - 1) < 0.02 && Math.abs(v.anchor.scaleMax - 1) < 0.02, `anchor ${v.anchor.scaleMin}..${v.anchor.scaleMax}`);
    assert.ok(v.fit.medresTiledP50 < 0.01, `tiled residual ${v.fit.medresTiledP50} m`);
    assert.ok(v.fit.medresGlobalP50 >= v.fit.medresTiledP50 && v.fit.medresInliersP50 >= 0, JSON.stringify(v.fit));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("CLI: ares depth --volumetric --no-texture: shell plus body, no relief keys, every stat in the sidecar", { timeout: 180000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-vol-"));
  try {
    const runDir = join(dir, "run");
    await makeVolumetricRun(runDir);
    const video = join(dir, "clip.mp4");
    writeFileSync(video, Buffer.alloc(1024));      // ffmpeg cannot open it: no RGB, geometry only
    const out = join(dir, "vol.ares");
    const log = execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", out, "--volumetric", "--no-texture", "--no-audio"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    for (const stage of ["fit", "anchor", "mesh"]) assert.match(log, new RegExp(`progress ${stage} 4/4`));
    assert.match(log, /\[ares\] subject: mask coverage/);
    assert.match(log, /\[ares\] normals: axis vote .*: y as stored/);

    const file = Demuxer.parse(new Uint8Array(readFileSync(out)));
    assert.equal(file.header.frameCount, N);
    const sb = file.superblock.meta;
    assert.equal(sb["relief.camera"], undefined);
    assert.equal(sb["relief.forward"], undefined);
    assert.equal(sb["volumetric.method"], "shell+sam-3d-body");
    assert.equal(sb["volumetric.camera"].split(",").length, 3);
    assert.equal(sb["volumetric.forward"], "0,0,-1");
    assert.ok(Math.abs(Number(sb["volumetric.fov"]) - 40) < 1e-3, `fov ${sb["volumetric.fov"]}`);

    const side = JSON.parse(readFileSync(out + ".meta.json", "utf8"));
    const v = side.volumetric;
    assert.equal(v.method, "shell+sam-3d-body");
    assert.deepEqual(v.grid, [W, H]);                                   // the map size by default
    assert.equal(v.fit.fitted, N);
    assert.ok(v.fit.medresTiledP50 < 0.01, `tiled residual ${v.fit.medresTiledP50} m on an exact target`);
    assert.equal(v.anchor.anchored, N);
    // The metric target IS the body's front surface: the anchor leaves the body where it is.
    assert.ok(Math.abs(v.anchor.scaleP50 - 1) < 0.02, `anchor scale ${v.anchor.scaleP50}`);
    // The second person has no body: the restricted mask drops it.
    assert.ok(v.mask.coverageAfter < v.mask.coverageBefore - (BLOB.x1 - BLOB.x0) * (BLOB.y1 - BLOB.y0) / P * 0.9,
      `mask ${v.mask.coverageBefore} -> ${v.mask.coverageAfter}`);
    assert.equal(v.mask.framesWithoutRgb, N);
    assert.equal(v.detail.vote.flipY, false);
    assert.equal(v.body.validFrames, N);
    assert.equal(v.body.backfilled, 0);
    assert.deepEqual(v.intrinsics.metricFocal, { factorMin: 1, factorP50: 1, factorMax: 1, framesAtOne: N });
    assert.equal(v.body.vertices, mhr().V);
    assert.ok(v.perFrame.bodyTris > 0, "body triangles behind the shell");
    assert.equal(side.geometry.intraFrames, N);
    assert.ok(side.request === undefined && side.source.kind === "2d-video+depth-run");

    // The frame is more than its shell: decode frame 2 through the CLI's own exporter.
    const obj = join(dir, "f2.obj");
    execFileSync(process.execPath, [CLI, "export", out, "-o", obj, "--frame", "2"], { stdio: "pipe" });
    const text = readFileSync(obj, "utf8");
    const tris = (text.match(/^f /gm) || []).length;
    assert.ok(tris > v.perFrame.shellTris * 1.5, `frame 2: ${tris} triangles against a ${v.perFrame.shellTris}-triangle shell`);
    // The subject stands on the floor (center bottom) and keeps its metric height.
    const ys = [...text.matchAll(/^v (\S+) (\S+) (\S+)/gm)].map((m) => Number(m[2]));
    const tall = Math.max(...ys) - Math.min(...ys);
    assert.ok(Math.abs(tall - mhr().height) < 0.1, `height ${tall.toFixed(3)} m against the body's ${mhr().height.toFixed(3)} m`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("CLI: ares depth --volumetric with a texture: the back region sits under the frame and takes colour from the video", { timeout: 300000 }, async (t) => {
  if (!(await ffmpegAvailable())) { t.skip("ffmpeg not available"); return; }
  const dir = mkdtempSync(join(tmpdir(), "ares-vol-"));
  try {
    const video = join(dir, "clip.mp4");
    execFileSync(ffmpegPath(), ["-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
      "-i", `testsrc2=size=${W}x${H}:rate=30`, "-t", "0.2", "-pix_fmt", "yuv420p", video]);
    const runDir = join(dir, "run");
    await makeVolumetricRun(runDir);
    const out = join(dir, "vol.ares");
    const log = execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", out, "--volumetric", "--tex-size", "256", "--crf", "40"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    assert.match(log, /progress texture 4\/4/);
    const side = JSON.parse(readFileSync(out + ".meta.json", "utf8"));
    const v = side.volumetric;
    // 36,874 faces in 3x3 patches, 85 per row at 256 wide: 434 rows, 1302 texels, rounded up to 1304.
    assert.deepEqual(v.back.region, [256, 1304]);
    assert.deepEqual(v.back.atlas, [256, 1560]);
    assert.deepEqual(side.encode.atlas, [256, 1560]);
    assert.equal(v.back.colourFrames, N);
    assert.ok(v.back.observed > 1000, `observed ${v.back.observed} vertices`);
    assert.equal(v.back.observed + v.back.filled + v.back.mean, mhr().V);
    const inf = execFileSync(process.execPath, [CLI, "info", out], { encoding: "utf8" });
    assert.match(inf, /4 frames @ 30fps/);
    assert.match(inf, /texture: VP09 video 256x1560/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("CLI: ares depth --volumetric on a video shorter than the run: the body counts cover the frames written", { timeout: 300000 }, async (t) => {
  if (!(await ffmpegAvailable())) { t.skip("ffmpeg not available"); return; }
  const dir = mkdtempSync(join(tmpdir(), "ares-vol-"));
  try {
    const video = join(dir, "clip.mp4");
    execFileSync(ffmpegPath(), ["-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
      "-i", `testsrc2=size=${W}x${H}:rate=30`, "-frames:v", "3", "-pix_fmt", "yuv420p", video]);
    const runDir = join(dir, "run");
    await makeVolumetricRun(runDir, {}, [], (extra) => { extra.bodyValid[1] = 0; });
    const out = join(dir, "vol.ares");
    execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", out, "--volumetric", "--tex-size", "256", "--crf", "40", "--no-audio"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    const side = JSON.parse(readFileSync(out + ".meta.json", "utf8"));
    assert.equal(side.output.frames, 3);
    // Over the 3 frames written: frame 1 copied, frames 0 and 2 fitted (the run's frame 3 is cut).
    assert.equal(side.volumetric.body.validFrames, 2);
    assert.equal(side.volumetric.body.backfilled, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("CLI: ares depth --volumetric refuses relief-grid flags and a run without its body", { timeout: 120000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-vol-"));
  try {
    const runDir = join(dir, "run");
    await makeVolumetricRun(runDir);
    const video = join(dir, "clip.mp4");
    writeFileSync(video, Buffer.alloc(16));
    const run = (args) => {
      try { execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", join(dir, "x.ares"), "--volumetric", "--no-texture", ...args], { stdio: "pipe" }); return ""; }
      catch (e) { return String(e.stderr); }
    };
    assert.match(run(["--sheets"]), /--volumetric: --sheets shapes the relief grid.*got --sheets/);
    assert.match(run(["--inpaint"]), /--volumetric: --inpaint/);
    assert.match(run(["--decimate", "0.5"]), /got --decimate 0\.5/);
    assert.match(run(["--anchor-smooth", "-1"]), /--anchor-smooth: -1 is below the minimum 0/);

    const bare = join(dir, "bare");
    await makeVolumetricRun(bare, { intrinsics: undefined }, ["body", "bodyFaces", "bodyValid", "metric", "intrinsics"]);
    let err = "";
    try { execFileSync(process.execPath, [CLI, "depth", video, "--depth", bare, "-o", join(dir, "y.ares"), "--volumetric", "--no-texture"], { stdio: "pipe" }); }
    catch (e) { err = String(e.stderr); }
    assert.match(err, /--volumetric needs the run's mask, metric, body and intrinsics; .* lacks metric, body, intrinsics/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
