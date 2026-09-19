import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  readDepthRun, writeDepthRun, openDepthRun, stabilizeDepth, buildDepthGrid, resampleMap, depthFrameToMesh,
  muxClip, muxClipWithStats, MeshClipWriter, ffmpegAvailable, ffmpegPath,
  guidedResample, resizeRgbArea, buildFillLayer, DepthHistogram, BACKDROP, barsFromProfile, temporalMedian, memoryStore,
  cutEdgeCount,
} from "../dist/index.js";
import { Demuxer, reliefFromMeta } from "@ares/core";

const CLI = fileURLToPath(new URL("../dist/cli.js", import.meta.url));

function meta(over = {}) {
  return {
    schema: "ares-depth/1", engine: "service", model: "depth-anything/v2-small", modelKey: "small",
    kind: "relative-disparity", width: 8, height: 6, frames: 1, fps: 30,
    sampling: { fps: null, maxFrames: null }, video: "/tmp/clip.mp4",
    sourceFps: 30, sourceWidth: 96, sourceHeight: 64, sourceFrames: 12, sourceDurationS: 0.4,
    msPerFrame: 12.5, device: "cpu", dtype: "float32", done: true, ...over,
  };
}

/* ----------------------------- (a) unprojection ----------------------------- */

test("unprojection: d=1 lands every vertex at near, d=0 at far, and the centre vertex is on the axis", () => {
  const gridW = 9, gridH = 7, near = 0.5, far = 6;
  const grid = buildDepthGrid(gridW, gridH, 1, 55);
  const n = gridW * gridH;
  const opts = { kind: "relative-disparity", near, far, edge: 0.08 };

  const nearest = depthFrameToMesh(new Float32Array(n).fill(1), grid, opts);
  assert.equal(nearest.positions.length / 3, n, "constant depth culls nothing");
  for (let i = 0; i < n; i++) assert.ok(Math.abs(nearest.positions[i * 3 + 2] + near) < 1e-5, `vertex ${i} z`);

  const farthest = depthFrameToMesh(new Float32Array(n), grid, opts);
  for (let i = 0; i < n; i++) assert.ok(Math.abs(farthest.positions[i * 3 + 2] + far) < 1e-4, `vertex ${i} z`);

  // Odd grid => a vertex sits exactly at u=v=0.5, i.e. straight down the camera axis.
  const c = ((gridH - 1) / 2) * gridW + (gridW - 1) / 2;
  assert.ok(Math.abs(grid.uvs[c * 2] - 0.5) < 1e-6 && Math.abs(grid.uvs[c * 2 + 1] - 0.5) < 1e-6);
  assert.ok(Math.abs(nearest.positions[c * 3]) < 1e-6, "centre x");
  assert.ok(Math.abs(nearest.positions[c * 3 + 1]) < 1e-6, "centre y");

  // v = 0 is the TOP image row and world y runs up, so the first row sits above the last.
  assert.ok(nearest.positions[1] > nearest.positions[(n - 1) * 3 + 1], "row 0 is above the last row");
});

test("resampleMap: area-averages when shrinking, bilinear when growing, identity at the same size", () => {
  const W = 8, H = 2;
  const m = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) m[y * W + x] = x;
  const same = resampleMap(m, W, H, W, H);
  assert.deepEqual([...same], [...m]);
  const half = resampleMap(m, W, H, 4, H);          // pairs averaged: 0.5, 2.5, 4.5, 6.5
  assert.deepEqual([...half.subarray(0, 4)].map((v) => +v.toFixed(4)), [0.5, 2.5, 4.5, 6.5]);
  const up = resampleMap(m, W, H, 16, H);
  assert.ok(up[0] === 0 && Math.abs(up[15] - 7) < 1e-5 && up.length === 16 * H);
});

/* ------------------------------ (b) silhouette cull ------------------------------ */

/** gridW x gridH normalized disparity: `near` value in the given columns, `far` value elsewhere. */
function stepMap(gridW, gridH, nearCols) {
  const m = new Float32Array(gridW * gridH).fill(0.2);
  for (let y = 0; y < gridH; y++) for (const x of nearCols) m[y * gridW + x] = 0.9;
  return m;
}

test("cull: a vertical depth step drops exactly the straddling cells, and orphaned vertices are compacted", () => {
  const gridW = 8, gridH = 6;
  const grid = buildDepthGrid(gridW, gridH, 1, 55);
  const opts = { kind: "relative-disparity", near: 0.5, far: 6, edge: 0.08 };
  const fullTris = (gridW - 1) * (gridH - 1) * 2;

  // One step between column 3 and column 4: the five cells at x=3 go, nothing else does.
  const stepped = depthFrameToMesh(stepMap(gridW, gridH, [4, 5, 6, 7]), grid, opts);
  assert.equal(stepped.indices.length / 3, fullTris - (gridH - 1) * 2);
  assert.equal(stepped.positions.length / 3, gridW * gridH, "every column still carries a kept cell");

  // A one-column-wide ridge: cells x=0 and x=1 both straddle it, so columns 0 and 1 are left
  // unreferenced and must disappear from the vertex set.
  const ridge = depthFrameToMesh(stepMap(gridW, gridH, [1]), grid, opts);
  assert.equal(ridge.indices.length / 3, (gridW - 3) * (gridH - 1) * 2);
  assert.equal(ridge.positions.length / 3, (gridW - 2) * gridH);
  assert.equal(ridge.uvs.length / 2, ridge.positions.length / 3);
  let maxIdx = 0;
  for (const i of ridge.indices) maxIdx = Math.max(maxIdx, i);
  assert.equal(maxIdx, ridge.positions.length / 3 - 1, "indices are remapped onto the compacted set");

  // Flat map, same grid: nothing is culled anywhere.
  const flat = depthFrameToMesh(new Float32Array(gridW * gridH).fill(0.4), grid, opts);
  assert.equal(flat.indices.length / 3, fullTris);
});

test("cut hysteresis: an edge near the threshold keeps its last decision; a clear change still flips it", () => {
  const gridW = 8, gridH = 6, near = 2, far = 6, edge = 0.08;
  const grid = buildDepthGrid(gridW, gridH, 1, 55);
  const fullTris = (gridW - 1) * (gridH - 1) * 2;
  // Columns 4..7 at relative depth jump r against columns 0..3, set through disparity.
  const zOf = (d) => 1 / (d / near + (1 - d) / far), dOf = (z) => (1 / z - 1 / far) / (1 / near - 1 / far);
  const map = (r) => {
    const m = new Float32Array(gridW * gridH).fill(0.5);
    const zb = zOf(0.5), dn = dOf(zb / (1 + r));      // nearer side, a jump of r relative to it
    for (let y = 0; y < gridH; y++) for (let x = 4; x < gridW; x++) m[y * gridW + x] = dn;
    return m;
  };
  const state = new Uint8Array(cutEdgeCount(gridW, gridH));
  const tris = (r, primed) => depthFrameToMesh(map(r), grid, { kind: "relative-disparity", near, far, edge, hysteresis: { state, primed } }).indices.length / 3;
  const stepCut = fullTris - (gridH - 1) * 2;
  assert.equal(tris(0.10, false), stepCut, "first frame: the plain rule cuts a 10 % jump");
  assert.equal(tris(0.06, true), stepCut, "6 % is inside the band (5.4 to 12 %): the cut holds");
  assert.equal(tris(0.05, true), fullTris, "5 % is below it: the cut closes");
  assert.equal(tris(0.11, true), fullTris, "11 % is inside the band: closed stays closed");
  assert.equal(tris(0.13, true), stepCut, "13 % is above it: the cut opens");
  // Without a state the rule is the plain threshold.
  assert.equal(depthFrameToMesh(map(0.09), grid, { kind: "relative-disparity", near, far, edge }).indices.length / 3, stepCut);
});

test("sheets: identical topology every frame → the muxer takes the temporal I+P path; culled frames go all-intra", async () => {
  const gridW = 16, gridH = 12;
  const grid = buildDepthGrid(gridW, gridH, 1, 55);
  const base = { kind: "relative-disparity", near: 0.5, far: 6, edge: 0.08 };
  const fullTris = (gridW - 1) * (gridH - 1) * 2;
  // Four frames with DIFFERENT numbers of steps, so the culled topology genuinely differs.
  const maps = [[], [5], [5, 9], [5, 9, 13]].map((cols) => stepMap(gridW, gridH, cols));

  const sheets = maps.map((m) => depthFrameToMesh(m, grid, { ...base, sheets: true }));
  for (const f of sheets) {
    assert.equal(f.indices.length / 3, fullTris);
    assert.equal(f.positions.length / 3, gridW * gridH);
  }
  const ref = Buffer.from(sheets[0].indices.buffer, sheets[0].indices.byteOffset, sheets[0].indices.byteLength);
  for (const f of sheets.slice(1)) {
    assert.ok(ref.equals(Buffer.from(f.indices.buffer, f.indices.byteOffset, f.indices.byteLength)), "byte-identical index buffers");
  }
  const sheetStats = await muxClipWithStats({ fps: 30, frames: sheets, gopLength: 30, temporal: { forceIntra: false } });
  assert.ok(sheetStats.temporalFrames > 0, `expected temporal frames, got ${sheetStats.temporalFrames}`);
  assert.equal(sheetStats.intraFrames, 0);

  const culled = maps.map((m) => depthFrameToMesh(m, grid, base));
  assert.ok(new Set(culled.map((f) => f.indices.length)).size === culled.length, "each frame culls a different amount");
  const culledStats = await muxClipWithStats({ fps: 30, frames: culled, gopLength: 30, temporal: { forceIntra: true } });
  assert.equal(culledStats.intraFrames, 4);
  assert.equal(culledStats.temporalFrames, 0);
});

/* -------------------------------- (c) stabilizer -------------------------------- */

test("stabilize: random per-frame scale+shift of one field collapses back onto itself", () => {
  const W = 24, H = 16, N = 6, P = W * H;
  const field = new Float32Array(P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) field[y * W + x] = 0.15 + 0.7 * (x / (W - 1)) + 0.25 * Math.sin(y * 0.7);
  const maps = new Float32Array(N * P);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let t = 0; t < N; t++) {
    const a = 0.7 + 0.7 * rnd(), b = -0.3 + 0.6 * rnd();
    for (let i = 0; i < P; i++) maps[t * P + i] = a * field[i] + b;
  }
  // strength 0 isolates the affine alignment from the temporal filter.
  const r = stabilizeDepth(maps, W, H, N, null, { kind: "relative-disparity", strength: 0 });
  let worst = 0;
  for (let t = 1; t < N; t++) for (let i = 0; i < P; i++) worst = Math.max(worst, Math.abs(r.maps[t * P + i] - r.maps[i]));
  assert.ok(worst < 5e-3, `frames disagree by ${worst} after alignment`);
  for (let i = 0; i < r.maps.length; i++) assert.ok(r.maps[i] >= 0 && r.maps[i] <= 1, `sample ${i} = ${r.maps[i]} out of [0,1]`);
  assert.ok(r.stats.scaleMin > 0 && r.stats.scaleMax <= 2);
});

test("stabilize: per-frame noise does not shrink the clip's depth range over a long chain", () => {
  // Every frame is the same field plus its own noise, so the true alignment is the identity all
  // the way down. A regression-based fit would attenuate each step by var(F)/(var(F)+var(noise))
  // and the chain would multiply that into a clip-wide flattening; the last frame must still span
  // as much depth as the first.
  const W = 32, H = 24, N = 60, P = W * H;
  const field = new Float32Array(P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) field[y * W + x] = 0.15 + 0.6 * (x / (W - 1)) + 0.1 * (y / (H - 1));
  const maps = new Float32Array(N * P);
  let seed = 991;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5);
  for (let t = 0; t < N; t++) for (let i = 0; i < P; i++) maps[t * P + i] = field[i] + 0.04 * rnd();
  const r = stabilizeDepth(maps, W, H, N, null, { kind: "relative-disparity", strength: 0 });
  assert.ok(Math.abs(r.stats.scaleMean - 1) < 5e-3, `mean scale ${r.stats.scaleMean} should sit at 1`);
  const span = (t) => {
    const v = Float32Array.from(r.maps.subarray(t * P, (t + 1) * P)).sort();
    return v[Math.round(0.95 * (P - 1))] - v[Math.round(0.05 * (P - 1))];
  };
  const first = span(0), last = span(N - 1);
  assert.ok(last > 0.9 * first, `depth range collapsed over the clip: ${first} -> ${last}`);
});

test("stabilize: a lasting step where the RGB moves is followed at once; a one-frame spike where it is still is removed", () => {
  const W = 16, H = 8, N = 6, P = W * H;
  const ramp = new Float32Array(P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) ramp[y * W + x] = 0.1 + 0.6 * (x / (W - 1));
  const maps = new Float32Array(N * P);
  const rgb = new Uint8Array(N * P * 3);
  const inBlock = (x, y, x0) => x >= x0 && x < x0 + 5 && y >= 1 && y < 6;
  const LEFT = 2, RIGHT = 9;                       // the still block and the RGB-moving block
  for (let t = 0; t < N; t++) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const step = t >= 2 && inBlock(x, y, RIGHT);            // a new surface arrives and stays
      const spike = t === 2 && inBlock(x, y, LEFT);           // a one-frame depth glitch
      maps[t * P + i] = step || spike ? 0.95 : ramp[i];
      rgb.fill(step ? 255 : 40, (t * P + i) * 3, (t * P + i) * 3 + 3);   // only the step changes colour
    }
  }
  const r = stabilizeDepth(maps, W, H, N, rgb, { kind: "relative-disparity", strength: 0.7 });
  // The output is normalised to the clip's own range: compare against the same pixel's frame 0.
  const at = (t, x, y) => r.maps[t * P + y * W + x];
  assert.ok(at(2, RIGHT + 2, 3) > 0.9, `the step should be followed at once, got ${at(2, RIGHT + 2, 3)}`);
  assert.ok(at(3, RIGHT + 2, 3) > 0.9, "and held");
  assert.ok(Math.abs(at(2, LEFT + 2, 3) - at(0, LEFT + 2, 3)) < 0.02, `the spike should be gone, ${at(0, LEFT + 2, 3)} -> ${at(2, LEFT + 2, 3)}`);
  for (let i = 0; i < r.maps.length; i++) assert.ok(r.maps[i] >= 0 && r.maps[i] <= 1);
  assert.equal(r.stats.rgbGated, true);
});

test("temporalMedian: 3- and 5-frame windows match a sorted reference, ends replicated", () => {
  const P = 64, N = 23;
  let seed = 4242;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (const win of [3, 5]) {
    const data = new Float32Array(N * P);
    for (let i = 0; i < data.length; i++) data[i] = rnd() < 0.2 ? Math.round(rnd() * 4) / 4 : rnd();   // ties included
    const store = memoryStore((n) => new Float32Array(n), N, P, data.slice());
    temporalMedian(store, N, P, win);
    const h = win >> 1;
    for (let t = 0; t < N; t++) for (let i = 0; i < P; i++) {
      const v = [];
      for (let k = -h; k <= h; k++) v.push(data[Math.min(N - 1, Math.max(0, t + k)) * P + i]);
      v.sort((a, b) => a - b);
      assert.equal(store.data[t * P + i], Math.fround(v[h]), `window ${win}, frame ${t}, pixel ${i}`);
    }
  }
});

test("stabilize: a metric run comes back in metres", () => {
  const W = 12, H = 9, N = 4, P = W * H;
  const maps = new Float32Array(N * P);
  for (let t = 0; t < N; t++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    maps[t * P + y * W + x] = 1.2 + 2.5 * (x / (W - 1)) + 0.4 * (y / (H - 1));
  }
  const r = stabilizeDepth(maps, W, H, N, null, { kind: "metric-depth", strength: 0.7 });
  assert.equal(r.kind, "metric-depth");
  let worst = 0;
  for (let i = 0; i < maps.length; i++) worst = Math.max(worst, Math.abs(r.maps[i] - maps[i]));
  assert.ok(worst < 2e-3, `metres drifted by ${worst}`);
  assert.ok(r.lo > 1 && r.hi < 5, `metric lo/hi should be metres, got ${r.lo}..${r.hi}`);
});

/* ---------------------------------- (d) run I/O ---------------------------------- */

test("depth run: write → read round trip, and a partial run reads what is on disk", async () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-depthio-"));
  try {
    const W = 8, H = 6, N = 4;
    const maps = new Float32Array(N * W * H);
    for (let i = 0; i < maps.length; i++) maps[i] = i * 0.001;
    await writeDepthRun(dir, meta({ width: W, height: H, frames: N }), maps);
    const back = await readDepthRun(dir);
    assert.equal(back.frames, N);
    assert.equal(back.width, W);
    assert.equal(back.height, H);
    assert.equal(back.meta.modelKey, "small");
    assert.equal(back.meta.kind, "relative-disparity");
    assert.deepEqual([...back.maps], [...maps]);

    // A run killed partway: the sidecar promises 4 frames, 2 are on disk, done:false.
    writeFileSync(join(dir, "depth.json"), JSON.stringify(meta({ width: W, height: H, frames: N, done: false })));
    writeFileSync(join(dir, "depth.f32"), Buffer.from(maps.buffer, 0, 2 * W * H * 4));
    const partial = await readDepthRun(dir);
    assert.equal(partial.frames, 2);
    assert.equal(partial.maps.length, 2 * W * H);

    // Same shortfall WITHOUT done:false is a corrupt run, not a partial one.
    writeFileSync(join(dir, "depth.json"), JSON.stringify(meta({ width: W, height: H, frames: N })));
    await assert.rejects(() => readDepthRun(dir), /declares 4 frames but holds 2/);

    writeFileSync(join(dir, "depth.json"), JSON.stringify(meta({ schema: "ares-depth/9" })));
    await assert.rejects(() => readDepthRun(dir), /expected schema/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

/* ------------------------------ (e) CLI end to end ------------------------------ */

const RUN_W = 98, RUN_H = 70, RUN_N = 12;

/** A moving gaussian bump over a background ramp — enough structure for the affine fit to bite. */
function bumpRun() {
  const maps = new Float32Array(RUN_N * RUN_W * RUN_H);
  for (let t = 0; t < RUN_N; t++) {
    const cx = 20 + t * 5, cy = 35, s2 = 2 * 12 * 12;
    for (let y = 0; y < RUN_H; y++) for (let x = 0; x < RUN_W; x++) {
      const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      maps[t * RUN_W * RUN_H + y * RUN_W + x] = 0.2 + 0.4 * (x / (RUN_W - 1)) + 0.5 * Math.exp(-d2 / s2);
    }
  }
  return maps;
}

async function makeRun(dir, over = {}) {
  await writeDepthRun(dir, meta({ width: RUN_W, height: RUN_H, frames: RUN_N, sourceFrames: RUN_N, ...over }), bumpRun());
}

test("CLI: ares depth --no-texture --no-audio builds a clip from the depth run alone", { timeout: 120000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-depthcli-"));
  try {
    const runDir = join(dir, "run");
    await makeRun(runDir);
    // A file ffmpeg cannot open: the RGB motion gate is optional, so the encode must still land.
    const video = join(dir, "clip.mp4");
    writeFileSync(video, Buffer.alloc(1024));
    const out = join(dir, "out.ares");
    const log = execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", out,
      "--grid", "24", "--no-texture", "--no-audio"], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    assert.match(log, /progress stabilize \d+\/12/);
    assert.match(log, /progress mesh 12\/12/);
    assert.match(log, /12 frames @ 30fps/);
    const sidecar = JSON.parse(readFileSync(out + ".meta.json", "utf8"));
    assert.equal(sidecar.depth.kind, "relative-disparity");
    assert.equal(sidecar.depth.frames, 12);
    assert.deepEqual(sidecar.depth.grid, [24, 16]);          // 24 * 64/96 = 16
    assert.equal(sidecar.depth.sheets, false);
    assert.equal(sidecar.depth.stabilize, 0.7);
    assert.ok(sidecar.depth.normalization.hi > sidecar.depth.normalization.lo);
    assert.equal(sidecar.encode.textureCodec, null);

    // Every relief records its capture camera; a culled one carries no draw-time discard.
    const sb = Demuxer.parse(new Uint8Array(readFileSync(out))).superblock.meta;
    assert.equal(sb["relief.forward"], "0,0,-1");
    assert.equal(sb["relief.fov"], "55");
    assert.equal(sb["relief.camera"].split(",").length, 3);
    const [pv, nr, fr] = ["relief.pivot", "relief.near", "relief.far"].map((k) => Number(sb[k]));
    assert.ok(nr >= 0.5 && nr < pv && pv < fr && fr <= 6, `near ${nr} < pivot ${pv} < far ${fr} inside 0.5..6`);
    assert.ok(Math.abs(pv - 2 / (1 / nr + 1 / fr)) < 1e-4 * pv, "the pivot is the disparity midpoint of near and far");
    assert.equal(sb["relief.slope"], undefined);
    const r = reliefFromMeta(sb);
    assert.ok(r && r.slope === 0 && r.depthMax === 0 && r.pivot === Number(sb["relief.pivot"]));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("CLI: ares depth on a real video muxes geometry + texture; --sheets takes the temporal path", { timeout: 300000 }, async (t) => {
  if (!(await ffmpegAvailable())) { t.skip("ffmpeg not available"); return; }
  const dir = mkdtempSync(join(tmpdir(), "ares-depthcli-"));
  try {
    const video = join(dir, "clip.mp4");
    execFileSync(ffmpegPath(), ["-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
      "-i", "testsrc2=size=96x64:rate=30", "-t", "0.4", "-pix_fmt", "yuv420p", video]);
    const runDir = join(dir, "run");
    await makeRun(runDir);

    const out = join(dir, "out.ares");
    const log = execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", out,
      "--grid", "24", "--tex-size", "64", "--crf", "40"], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    assert.match(log, /progress rgb 12\/12/);
    assert.match(log, /progress texture 12\/12/);
    assert.match(log, /progress mux 12\/12/);

    const inf = execFileSync(process.execPath, [CLI, "info", out], { encoding: "utf8" });
    assert.match(inf, /12 frames @ 30fps/);
    assert.match(inf, /texture: VP09 video 64x64/);

    const sidecar = JSON.parse(readFileSync(out + ".meta.json", "utf8"));
    assert.equal(sidecar.depth.engine, "service");
    assert.equal(sidecar.depth.mapWidth, RUN_W);
    assert.equal(sidecar.depth.gate, "rgb");
    assert.equal(sidecar.encode.textureCodec, "vp9");
    assert.equal(sidecar.encode.texSize, 64);
    assert.equal(sidecar.geometry.intraFrames, 12);          // culled relief: topology changes per frame

    const outS = join(dir, "sheets.ares");
    execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", outS,
      "--grid", "24", "--sheets", "--no-texture", "--no-audio"], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    const sheetMeta = JSON.parse(readFileSync(outS + ".meta.json", "utf8"));
    assert.equal(sheetMeta.depth.sheets, true);
    assert.equal(sheetMeta.geometry.intraFrames, 0);
    assert.ok(sheetMeta.geometry.temporalFrames > 0);
    const r = reliefFromMeta(Demuxer.parse(new Uint8Array(readFileSync(outS))).superblock.meta);
    assert.ok(r && r.slope > 0 && r.depthMax > 6, "a sheet relief carries the draw-time discard");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

/* ----------------------- (f) coherent motion: the grown gate ----------------------- */

/**
 * An untextured block covering `cover` of the frame, approaching: it grows a pixel a side per frame
 * and its disparity rises, so the RGB changes only along its outline. The background is a static
 * textured ramp.
 */
function approachScene(W, H, N, cover) {
  const P = W * H;
  const maps = new Float32Array(N * P), rgb = new Uint8Array(N * P * 3);
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const tex = new Uint8Array(P);
  for (let i = 0; i < P; i++) tex[i] = 30 + ((rnd() * 180) | 0);
  const hw0 = Math.round((W * Math.sqrt(cover)) / 2), hh0 = Math.round((H * Math.sqrt(cover)) / 2);
  const inside = (t, x, y) => Math.abs(x - W / 2 + 0.5) < hw0 + t && Math.abs(y - H / 2 + 0.5) < hh0 + t;
  for (let t = 0; t < N; t++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, on = inside(t, x, y);
    maps[t * P + i] = on ? 0.6 + 0.03 * t : 0.15 + 0.25 * (x / (W - 1));
    rgb.fill(on ? 200 : tex[i], (t * P + i) * 3, (t * P + i) * 3 + 3);
  }
  return { maps, rgb, inside };
}

test("stabilize: an approaching untextured object covering most of the frame does not drag the background (grown gate)", () => {
  const W = 128, H = 96, N = 8, P = W * H;
  const s = approachScene(W, H, N, 0.55);
  const drift = (grow) => {
    const r = stabilizeDepth(s.maps, W, H, N, s.rgb, { kind: "relative-disparity", strength: 0, grow });
    let sum = 0, n = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (s.inside(N - 1, x, y)) continue;                   // background in every frame
      sum += Math.abs(r.maps[(N - 1) * P + y * W + x] - r.maps[y * W + x]); n++;
    }
    return { drift: sum / n, grown: r.stats.grownFrames };
  };
  const off = drift(false), on = drift(true);
  assert.ok(off.drift > 0.05, `the ungrown fit should show the drift under test, got ${off.drift}`);
  assert.ok(on.drift < 0.01, `background drifted by ${on.drift} of the range with the gate grown`);
  assert.ok(on.grown > 0);

  // Nothing moves: nothing is grown.
  const still = new Uint8Array(N * P * 3).fill(90);
  const flat = new Float32Array(N * P);
  for (let t = 0; t < N; t++) flat.set(s.maps.subarray(0, P), t * P);
  assert.equal(stabilizeDepth(flat, W, H, N, still, { kind: "relative-disparity", strength: 0 }).stats.grownFrames, 0);
});

/* ------------------------------ (g) guided resampling ------------------------------ */

test("guidedResample: across a colour edge a vertex takes depth from its own side only", () => {
  const W = 32, H = 4, gridW = 64, gridH = 8;
  const map = new Float32Array(W * H), guideMap = new Uint8Array(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    map[y * W + x] = x < 16 ? 0.2 : 0.8;
    guideMap.fill(x < 16 ? 40 : 200, (y * W + x) * 3, (y * W + x) * 3 + 3);
  }
  const guideGrid = resizeRgbArea(guideMap, W, H, gridW, gridH);
  const plain = resampleMap(map, W, H, gridW, gridH);
  const guided = guidedResample(map, W, H, guideMap, guideGrid, gridW, gridH);
  for (let y = 0; y < gridH; y++) {
    const at = (a, x) => a[y * gridW + x];
    assert.ok(at(plain, 31) > 0.25 && at(plain, 32) < 0.75, "bilinear spreads the step over the vertices beside it");
    assert.ok(Math.abs(at(guided, 31) - 0.2) < 1e-6 && Math.abs(at(guided, 32) - 0.8) < 1e-6, `row ${y}: ${at(guided, 31)} | ${at(guided, 32)}`);
  }
  for (const v of guided) assert.ok(Math.abs(v - 0.2) < 1e-6 || Math.abs(v - 0.8) < 1e-6, `vertex off both plateaus: ${v}`);

  // One colour everywhere: the range weight is flat and the result is a spatial average.
  const ramp = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) ramp[y * W + x] = x / (W - 1);
  const grey = new Uint8Array(W * H * 3).fill(120);
  const g2 = guidedResample(ramp, W, H, grey, resizeRgbArea(grey, W, H, gridW, gridH), gridW, gridH);
  const p2 = resampleMap(ramp, W, H, gridW, gridH);
  for (let x = 6; x < gridW - 6; x++) assert.ok(Math.abs(g2[x] - p2[x]) < 0.01, `x=${x}: guided ${g2[x]} vs bilinear ${p2[x]}`);
});

/* -------------------------------- (h) fill layer -------------------------------- */

test("buildFillLayer: the background is continued under a foreground block, ring by ring from its outline", () => {
  const gridW = 12, gridH = 8, n = gridW * gridH;
  const inBlock = (x, y) => x >= 4 && x <= 7 && y >= 2 && y <= 5;
  const z = new Float32Array(n), guide = new Uint8Array(n * 3);
  for (let y = 0; y < gridH; y++) for (let x = 0; x < gridW; x++) {
    const i = y * gridW + x, fg = inBlock(x, y);
    z[i] = fg ? 1 : 4;
    guide.set(fg ? [200, 100, 50] : [10, 20, 30], i * 3);
  }
  const fill = buildFillLayer(z, gridW, gridH, guide, { edge: 0.08, band: 2 });
  assert.equal(fill.count, 16, "the whole 4x4 block is within two rings of its outline");
  let anchors = 0;
  for (let y = 0; y < gridH; y++) for (let x = 0; x < gridW; x++) {
    const i = y * gridW + x;
    if (!inBlock(x, y)) { assert.equal(fill.dist[i], -1, `background vertex ${x},${y} is not in the band`); anchors += fill.anchor[i]; continue; }
    const outline = x === 4 || x === 7 || y === 2 || y === 5;
    assert.equal(fill.dist[i], outline ? 0 : 1, `ring of ${x},${y}`);
    assert.ok(Math.abs(fill.z[i] - 4) < 1e-6, `fill depth at ${x},${y} is ${fill.z[i]}, the background's is 4`);
    assert.deepEqual([...fill.colour.subarray(i * 3, i * 3 + 3)], [10, 20, 30]);
  }
  assert.equal(anchors, 16, "every background vertex beside the block anchors a cut");

  assert.equal(buildFillLayer(z, gridW, gridH, guide, { edge: 0.08, band: 0 }).count, 12, "band 0 keeps the outline only");
});

/* ---------------------------- (i) streamed clip writer ---------------------------- */

test("MeshClipWriter writes byte for byte the file muxClip builds in memory", async () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-writer-"));
  try {
    const gridW = 16, gridH = 12, N = 70, gop = 30;
    const grid = buildDepthGrid(gridW, gridH, 1.5, 55);
    const opts = { kind: "relative-disparity", near: 0.5, far: 6, edge: 0.08 };
    const maps = Array.from({ length: N }, (_, t) => stepMap(gridW, gridH, [2 + (t % 11), 3 + (t % 11)]));
    const tex = Array.from({ length: N }, (_, t) => ({ data: Uint8Array.from({ length: 40 + (t % 7) }, (_, k) => (t * 31 + k) & 255), isKey: t % gop === 0 }));
    const clipMeta = { title: "writer parity", "relief.fov": "55" };
    const textureVideo = { fourcc: "VP09", width: 64, height: 64 };

    for (const sheets of [false, true]) {
      const frames = maps.map((m) => depthFrameToMesh(m, grid, { ...opts, sheets }));
      const temporal = { forceIntra: !sheets };
      const ref = await muxClip({ fps: 30, frames, gopLength: gop, meta: clipMeta, temporal,
        textureVideo: { ...textureVideo, gops: [{ frames: tex }] } });
      const out = join(dir, `clip-${sheets}.ares`);
      const w = await MeshClipWriter.open({ out, fps: 30, gopLength: gop, textureVideo, temporal });
      for (let t0 = 0; t0 < N; t0 += gop) w.writeGop(frames.slice(t0, t0 + gop), tex.slice(t0, t0 + gop));
      const r = w.finish({ meta: clipMeta });
      const got = readFileSync(out);
      assert.equal(r.sizeBytes, ref.length);
      assert.ok(Buffer.from(ref).equals(got), `${sheets ? "sheets" : "culled"}: the streamed file differs from muxClip's`);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

/* ------------------------------ (j) subject mask I/O ------------------------------ */

test("openDepthRun reads a subject mask in step with the maps, and rejects a missing or short one", async () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-depthmask-"));
  try {
    const W = 8, H = 6, N = 4, P = W * H;
    const maps = new Float32Array(N * P).fill(0.5);
    const m = meta({ width: W, height: H, frames: N, mask: { file: "mask.u8", prompt: "person", engine: "sam3-text-tracker", coverage: 0.25 } });
    await writeDepthRun(dir, m, maps);
    const mask = new Uint8Array(N * P);
    for (let i = 0; i < mask.length; i++) mask[i] = (i % 5) === 0 ? 255 : 0;
    writeFileSync(join(dir, "mask.u8"), mask);

    const run = await openDepthRun(dir);
    try {
      assert.equal(run.hasMask, true);
      assert.equal(run.meta.mask.prompt, "person");
      const two = new Uint8Array(2 * P);
      run.readMask(1, 2, two);
      assert.deepEqual([...two], [...mask.subarray(P, 3 * P)]);
      assert.throws(() => run.readMask(3, 2, two), /outside/);
    } finally { run.close(); }

    writeFileSync(join(dir, "mask.u8"), mask.subarray(0, 3 * P));
    await assert.rejects(() => openDepthRun(dir), /holds 3 mask frame/);
    rmSync(join(dir, "mask.u8"));
    await assert.rejects(() => openDepthRun(dir), /names a subject mask/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("CLI: ares depth finds a letterbox, cuts it from the maps, the texture and the grid, and --crop none keeps it", { timeout: 300000 }, async (t) => {
  if (!(await ffmpegAvailable())) { t.skip("ffmpeg not available"); return; }
  const dir = mkdtempSync(join(tmpdir(), "ares-depthcrop-"));
  try {
    // A 96x64 picture padded to 96x96: 16 black rows above and below.
    const video = join(dir, "boxed.mp4");
    execFileSync(ffmpegPath(), ["-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
      "-i", "testsrc2=size=96x64:rate=30", "-t", "0.4", "-vf", "pad=96:96:0:16:black", "-pix_fmt", "yuv420p", video]);
    const W = 98, H = 98, N = 12;
    const maps = new Float32Array(N * W * H).fill(0.5);
    const runDir = join(dir, "run");
    await writeDepthRun(runDir, meta({ width: W, height: H, frames: N, sourceWidth: 96, sourceHeight: 96, sourceFrames: N }), maps);

    const out = join(dir, "out.ares");
    const log = execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", out,
      "--grid", "24", "--tex-size", "64", "--no-audio"], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    assert.match(log, /letterbox: picture 96x6\d\+0\+1\d of 96x96/);
    const crop = JSON.parse(readFileSync(out + ".meta.json", "utf8")).depth.crop;
    const [cw, ch, cx, cy] = crop.source;
    assert.equal(cw, 96); assert.equal(cx, 0);
    assert.ok(cy >= 16 && cy <= 18 && cy + ch <= 80 && cy + ch >= 78, `picture rows ${cy}..${cy + ch} inside 16..80`);
    const [mw, mh, , my] = crop.map;
    assert.equal(mw, 98);
    assert.ok(my >= 16 && my + mh <= 82, `map rows ${my}..${my + mh}`);
    assert.equal(JSON.parse(readFileSync(out + ".meta.json", "utf8")).depth.grid[1], Math.round(24 * ch / cw), "the grid follows the picture's aspect");

    const outN = join(dir, "full.ares");
    execFileSync(process.execPath, [CLI, "depth", video, "--depth", runDir, "-o", outN, "--grid", "24", "--no-texture", "--no-audio", "--crop", "none"], { stdio: "pipe" });
    const full = JSON.parse(readFileSync(outN + ".meta.json", "utf8")).depth;
    assert.equal(full.crop, null);
    assert.deepEqual(full.grid, [24, 24]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

/* ---------------------------- (k) relief framing (depth histogram) ---------------------------- */

test("DepthHistogram: surface depth percentiles over a clip, backdrop vertices excluded, pivot at the disparity midpoint", () => {
  const frame = (zs) => Float32Array.from(zs.flatMap((z) => [0.3, -0.2, -z]));
  const h = new DepthHistogram(0.5, 6);
  h.add(frame(Array(20).fill(1)));
  h.add(frame(Array(60).fill(2)));
  h.add(frame([...Array(20).fill(4), 6 * BACKDROP, 6 * BACKDROP, 6 * BACKDROP]));
  assert.equal(h.count, 100, "parked vertices are not surface");
  const close = (a, b) => Math.abs(a - b) < 2e-3 * b;
  assert.ok(close(h.nearest(0.5), 2), `median ${h.nearest(0.5)}`);
  const f = h.framing();
  assert.ok(close(f.near, 1) && close(f.far, 4), `near ${f.near} far ${f.far}`);
  assert.ok(close(f.pivot, 1.6), `pivot ${f.pivot}: 2 / (1/1 + 1/4) = 1.6`);
  assert.equal(new DepthHistogram(0.5, 6).framing(), null, "no vertices: no framing");
});

/* ------------------------------------ (l) letterbox ------------------------------------ */

test("barsFromProfile: edge rows and columns that never leave black are bars; a dark clip is not a letterbox", () => {
  const rows = new Uint8Array(100).fill(200), cols = new Uint8Array(160).fill(200);
  rows.fill(3, 0, 12); rows.fill(20, 12, 13); rows.fill(5, 90);
  assert.deepEqual(barsFromProfile(rows, cols), { x0: 0, y0: 13, x1: 160, y1: 90 });
  cols.fill(0, 0, 20); cols.fill(0, 140);
  assert.deepEqual(barsFromProfile(rows, cols), { x0: 20, y0: 13, x1: 140, y1: 90 });
  assert.equal(barsFromProfile(new Uint8Array(100).fill(200), new Uint8Array(160).fill(200)), null, "no bar");
  assert.equal(barsFromProfile(new Uint8Array(100).fill(10), new Uint8Array(160).fill(10)), null, "all black");
  const dim = new Uint8Array(100).fill(0); dim.fill(200, 45, 55);
  assert.equal(barsFromProfile(dim, new Uint8Array(160).fill(200)), null, "under a quarter of the rows is picture");
});
