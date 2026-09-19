import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { collectSculptOps, applySculptToFrame, synthClip, filterFrame, writeObj } from "../dist/index.js";
import { parseEditList, validateMasks, keepPredicateAt, prepareRangeSdfAt, morphBitmap, growKeyframe, mirrorKeyframe, rleEncodeMask, rleDecodeMask, isRangeEnabled } from "@ares/core";

const CLI = fileURLToPath(new URL("../dist/cli.js", import.meta.url));

/** Flat n×n grid on y = 0 (see mesh.test.mjs); noisy y when `noise` is set. */
function gridPlane(n, noise = 0) {
  const positions = new Float32Array(n * n * 3), uvs = new Float32Array(n * n * 2);
  let seed = 3;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k = j * n + i;
    positions[k * 3] = i; positions[k * 3 + 1] = noise ? (rnd() - 0.5) * noise : 0; positions[k * 3 + 2] = j;
    uvs[k * 2] = i / (n - 1); uvs[k * 2 + 1] = j / (n - 1);
  }
  const idx = [];
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  return { positions, uvs, indices: Uint32Array.from(idx) };
}
const boxRange = (extra = {}) => ({
  id: "s", mode: "delete", action: "sculpt", startFrame: 0, endFrame: 0,
  keyframes: [{ frame: 0, volumes: [{ type: "box", min: [3, -10, 3], max: [8, 10, 8] }] }], ...extra,
});

test("sculpt inflate/move/flatten/pinch/smooth displace only the region, weld-aware", () => {
  // inflate: the plane's normal is ±Y → vertices inside rise (or fall) by the amount, outside stay put.
  {
    const f = gridPlane(12);
    const [op] = collectSculptOps([boxRange({ sculpt: { brush: "inflate", amount: 2, feather: 0.01 } })], 1);
    const st = applySculptToFrame(f, op, 0);
    assert.ok(st && st.vertices > 0);
    const inside = f.positions[(5 * 12 + 5) * 3 + 1], outside = f.positions[(0 * 12 + 0) * 3 + 1];
    assert.ok(Math.abs(Math.abs(inside) - 2) < 1e-4, `inside moved by the amount (got ${inside})`);
    assert.equal(outside, 0);
  }
  // move: a literal offset.
  {
    const f = gridPlane(12);
    const [op] = collectSculptOps([boxRange({ sculpt: { brush: "move", offset: [0, 3, 0], feather: 0.01 } })], 1);
    applySculptToFrame(f, op, 0);
    assert.ok(Math.abs(f.positions[(5 * 12 + 5) * 3 + 1] - 3) < 1e-5);
    assert.equal(f.positions[1], 0);
  }
  // smooth + flatten: a noisy patch gets calmer; pinch pulls toward the centroid.
  {
    const f = gridPlane(12, 0.6);
    const before = variance(f, 4, 7);
    const [sm] = collectSculptOps([boxRange({ sculpt: { brush: "smooth", amount: 1, iterations: 5, feather: 0.01 } })], 1);
    applySculptToFrame(f, sm, 0);
    const afterSmooth = variance(f, 4, 7);
    assert.ok(afterSmooth < before * 0.5, `smooth: ${before} → ${afterSmooth}`);
    // flatten on a fresh noisy patch: what remains is the fitted plane's own tilt, far below the noise.
    const f2 = gridPlane(12, 0.6);
    const before2 = variance(f2, 4, 7);
    const [fl] = collectSculptOps([boxRange({ sculpt: { brush: "flatten", amount: 1, feather: 0.01 } })], 1);
    applySculptToFrame(f2, fl, 0);
    assert.ok(variance(f2, 4, 7) < before2 * 0.1, `flatten: ${before2} → ${variance(f2, 4, 7)}`);
    const g = gridPlane(12);
    const [pi] = collectSculptOps([boxRange({ sculpt: { brush: "pinch", amount: 0.5, feather: 0.01 } })], 1);
    applySculptToFrame(g, pi, 0);
    const x = g.positions[(4 * 12 + 4) * 3];
    assert.ok(x > 4 && x < 5.5, `pinch pulled x toward the centre (got ${x})`);
  }
  // weld-aware: duplicate a vertex position (seam) and check both copies move together.
  {
    const f = gridPlane(12);
    const dup = new Float32Array(f.positions.length + 3);
    dup.set(f.positions); dup.set(f.positions.subarray((5 * 12 + 5) * 3, (5 * 12 + 5) * 3 + 3), f.positions.length);
    const idx = Uint32Array.from(f.indices); idx[0] = 144;   // one triangle references the duplicate
    const g = { positions: dup, uvs: undefined, indices: idx };
    const [op] = collectSculptOps([boxRange({ sculpt: { brush: "move", offset: [0, 1, 0], feather: 0.01 } })], 1);
    applySculptToFrame(g, op, 0);
    assert.equal(dup[(5 * 12 + 5) * 3 + 1], dup[144 * 3 + 1]);
  }
  // validation
  assert.throws(() => collectSculptOps([boxRange({ sculpt: { brush: "bend" } })], 1), /unknown brush/);
  assert.throws(() => collectSculptOps([boxRange({ sculpt: { brush: "pinch", amount: 3 } })], 1), /0\.\.1/);
  assert.equal(collectSculptOps([boxRange({ enabled: false })], 1).length, 0, "muted ranges are skipped");
});

function variance(f, lo, hi) {
  const ys = [];
  for (let j = lo; j <= hi; j++) for (let i = lo; i <= hi; i++) ys.push(f.positions[(j * 12 + i) * 3 + 1]);
  const m = ys.reduce((s, v) => s + v, 0) / ys.length;
  return ys.reduce((s, v) => s + (v - m) * (v - m), 0) / ys.length;
}

test("keyframe interpolation modes: linear, hold, smooth", () => {
  const r = { id: "r", mode: "delete", startFrame: 0, endFrame: 10, keyframes: [
    { frame: 0, volumes: [{ type: "box", min: [0, 0, 0], max: [1, 1, 1] }] },
    { frame: 10, volumes: [{ type: "box", min: [10, 0, 0], max: [11, 1, 1] }] },
  ] };
  const at = (interp, f, x) => prepareRangeSdfAt({ ...r, interp }, f)(x, 0.5, 0.5);
  // Half-way in linear the two SDFs average; hold keeps keyframe 0; smooth at t=0.25 leans to keyframe 0.
  assert.ok(Math.abs(at("linear", 5, 0.5) - (0.5 * -0.5 + 0.5 * 9.5)) < 1e-6);
  assert.ok(Math.abs(at("hold", 5, 0.5) - -0.5) < 1e-6);
  const lin = at("linear", 2.5, 0.5), sm = at("smooth", 2.5, 0.5);
  assert.ok(sm < lin, "smooth eases: closer to keyframe 0 early on");
});

test("muted ranges are ignored by the keep predicate", () => {
  const list = parseEditList({ aresEdits: 1, ranges: [{ id: "a", mode: "delete", enabled: false, startFrame: 0, endFrame: 0, keyframes: [{ frame: 0, volumes: [{ type: "box", min: [-1, -1, -1], max: [1, 1, 1] }] }] }] });
  assert.equal(keepPredicateAt(list, 0), null);
  assert.equal(isRangeEnabled(list.ranges[0]), false);
  const f = gridPlane(4);
  const keep = keepPredicateAt({ aresEdits: 1, ranges: [{ ...list.ranges[0], enabled: true }] }, 0);
  assert.ok(keep && filterFrame(f, keep).indices.length < f.indices.length);
});

test("morphBitmap grows and shrinks; growKeyframe/mirrorKeyframe edit volumes in place", () => {
  const w = 8, h = 8, bits = new Uint8Array(w * h);
  bits[3 * w + 3] = 1;
  const g = morphBitmap(bits, w, h, 1);
  assert.equal(g.reduce((s, v) => s + v, 0), 9);
  const back = morphBitmap(g, w, h, -1);
  assert.deepEqual(Array.from(back), Array.from(bits));
  assert.equal(morphBitmap(g, w, h, -2).reduce((s, v) => s + v, 0), 0);

  const kf = { frame: 0, volumes: [
    { type: "box", min: [0, 0, 0], max: [2, 2, 2] },
    { type: "brushStrokes", strokes: [{ op: "add", radius: 5, points: [[1, 1, 1]] }] },
    { type: "mask2d", kind: "rect", rect: [-0.5, -0.5, 0.5, 0.5], camera: { azimuth: 0, elevation: 0, distance: 1, target: [0, 0, 0], aspect: 1 } },
    { type: "mask2d", kind: "bitmap", mask: { width: w, height: h, rle: rleEncodeMask(bits) }, camera: { azimuth: 0, elevation: 0, distance: 1, target: [0, 0, 0], aspect: 1 } },
  ] };
  assert.equal(growKeyframe(kf, 1, 0.1, 1), 4);
  assert.deepEqual(kf.volumes[0].min, [-1, -1, -1]);
  assert.equal(kf.volumes[1].strokes[0].radius, 6);
  assert.ok(Math.abs(kf.volumes[2].rect[0] + 0.6) < 1e-9);
  assert.equal(rleDecodeMask(kf.volumes[3].mask.rle, w * h).reduce((s, v) => s + v, 0), 9);
  growKeyframe(kf, -10, 0, 0);   // shrink past the centre: boxes/brushes clamp, never invert
  assert.ok(kf.volumes[0].min[0] <= kf.volumes[0].max[0]);
  assert.ok(kf.volumes[1].strokes[0].radius > 0);

  const m = { frame: 0, volumes: [{ type: "box", min: [1, 0, 0], max: [2, 1, 1] }, { type: "brushStrokes", strokes: [{ op: "add", radius: 1, points: [[3, 0, 0]] }] }, kf.volumes[2]] };
  const res = mirrorKeyframe(m, 0, 0);
  assert.deepEqual(res, { mirrored: 2, skipped: 1 });
  assert.equal(m.volumes.length, 5);
  assert.deepEqual(m.volumes[3].min, [-2, 0, 0]);
  assert.deepEqual(m.volumes[4].strokes[0].points[0], [-3, 0, 0]);
});

test("sculpt runs inside a real mesh frame without breaking topology", () => {
  const f = synthClip("talk", 1, 30).frames[0];
  const tris = f.indices.length;
  const r = { id: "s", mode: "delete", action: "sculpt", startFrame: 0, endFrame: 0, sculpt: { brush: "inflate", amount: 0.05 },
    keyframes: [{ frame: 0, volumes: [{ type: "box", min: [-0.3, 0.5, -0.3], max: [0.3, 1.2, 0.3] }] }] };
  const [op] = collectSculptOps([r], 1);
  const st = applySculptToFrame(f, op, 0);
  assert.ok(st && st.vertices > 0 && st.feather > 0);
  assert.equal(f.indices.length, tris);
});

const CAM = (aspect) => ({ azimuth: 0.3, elevation: 0.1, distance: 2, target: [0, 0, 0], aspect });
/** An all-zero bitmap of w×h: rleEncodeMask closes the final run, so the runs sum to w*h exactly. */
const BMP = (w, h) => ({ width: w, height: h, rle: rleEncodeMask(new Uint8Array(w * h)) });
const doc = (volumes, extra = {}) => ({ aresEdits: 1, ranges: [{ id: "r1", mode: "delete", startFrame: 0, endFrame: 10, keyframes: [{ frame: 3, volumes }], ...extra }] });

test("validateMasks reports the mask2d faults that bake as a silent no-op", () => {
  // Clean: a marquee rect volume (no mask, no aspect to check) beside a bitmap whose camera aspect
  // matches it. The rect branch is the regression that matters — every box-select authors one.
  assert.deepEqual(validateMasks(doc([
    { type: "mask2d", kind: "rect", rect: [-0.5, -0.25, 0.5, 0.25], camera: CAM(2) },
    { type: "mask2d", kind: "bitmap", mask: BMP(64, 32), camera: CAM(2) },
    { type: "box", min: [0, 0, 0], max: [1, 1, 1] },
  ])), []);

  // prepareVolume gates on `kind` and returns null without it: the range is in the timeline, the
  // bake reports success, and not one triangle moves.
  assert.match(validateMasks(doc([{ type: "mask2d", mask: BMP(64, 32), camera: CAM(2) }]))[0], /r1 kf 3 has no kind and will never match/);
  assert.match(validateMasks(doc([{ type: "mask2d", kind: "blob", mask: BMP(64, 32), camera: CAM(2) }]))[0], /unknown kind "blob"/);
  assert.match(validateMasks(doc([{ type: "mask2d", kind: "bitmap", mask: BMP(64, 32) }]))[0], /has no camera/);
  // A truncated write: the runs no longer cover the bitmap, so every pixel past them reads 0.
  assert.match(validateMasks(doc([{ type: "mask2d", kind: "bitmap", mask: { width: 64, height: 32, rle: [100, 5] } }]))[1], /rle runs sum to 105, not 64x32 = 2048/);
  assert.match(validateMasks(doc([{ type: "mask2d", kind: "bitmap", mask: BMP(64, 32), camera: CAM(1) }]))[0], /camera\.aspect 1\.0000 is not the bitmap's 64x32 = 2\.0000/);
  assert.equal(validateMasks(doc([{ type: "mask2d", kind: "bitmap", mask: BMP(64, 32), camera: CAM(1.99) }])).length, 0, "aspect within 1% is a rounding, not a fault");

  // rect faults: an unordered rect selects nothing, and so does one that has left the screen.
  assert.match(validateMasks(doc([{ type: "mask2d", kind: "rect", rect: [0.5, -0.5, -0.5, 0.5], camera: CAM(1) }]))[0], /is not ordered x0<x1, y0<y1/);
  assert.match(validateMasks(doc([{ type: "mask2d", kind: "rect", rect: [-3, -0.5, -1.5, 0.5], camera: CAM(1) }]))[0], /lies wholly outside NDC \[-1, 1\]/);
  assert.match(validateMasks(doc([{ type: "mask2d", kind: "rect", camera: CAM(1) }]))[0], /needs rect \[x0, y0, x1, y1\]/);
  // A rect that merely OVERHANGS the screen is what Grow writes at the viewport edge, and
  // prepareVolume's max-of-half-planes evaluates it correctly. Flagging it would fail verify-edits
  // on a document that bakes exactly as previewed.
  assert.deepEqual(validateMasks(doc([{ type: "mask2d", kind: "rect", rect: [-1.05, -1.05, 1.05, 1.05], camera: CAM(1) }])), []);

  // Keyframe-level faults. bracket() can only ever reach one of a duplicated pair; a keyframe with
  // no `volumes` array is the one shape that THROWS in the bake rather than no-opping.
  const box = { type: "box", min: [0, 0, 0], max: [1, 1, 1] };
  const kfDoc = (keyframes) => ({ aresEdits: 1, ranges: [{ id: "r1", mode: "delete", startFrame: 0, endFrame: 10, keyframes }] });
  const issues = validateMasks(kfDoc([{ frame: 3, volumes: [box] }, { frame: 3, volumes: [box] }, { frame: 5 }]));
  assert.equal(issues.length, 2);
  assert.match(issues[0], /range r1 has two keyframes at frame 3/);
  assert.match(issues[1], /range r1 kf 5 has no volumes array — prepareKeyframe throws on it/);

  // NOT a fault: a keyframe past the span's edge is what dragging a range handle leaves behind
  // (the drag moves startFrame/endFrame and never the keyframes), and bracket() still hands it to
  // every in-span frame below 40 as the anchor they lerp from.
  assert.deepEqual(validateMasks(kfDoc([{ frame: 0, volumes: [box] }, { frame: 40, volumes: [box] }])), []);
  // Unreachable IS a fault: with a second keyframe at 5, kf 0's window is [-∞, 5) and the span
  // starts at 50, so nothing ever brackets it.
  const far = validateMasks({ aresEdits: 1, ranges: [{ id: "r1", mode: "delete", startFrame: 50, endFrame: 60, keyframes: [
    { frame: 0, volumes: [box] }, { frame: 5, volumes: [box] }, { frame: 55, volumes: [box] },
  ] }] });
  assert.equal(far.length, 1);
  assert.match(far[0], /range r1 kf 0 is unreachable — no frame in \[50, 60\] brackets it/);
});

test("a trim-out-only bake still rebases: derived keyframes past the window are dropped, user keyframes are not", () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-trim-"));
  try {
    const frames = join(dir, "frames");
    mkdirSync(frames);
    synthClip("talk", 4, 30).frames.forEach((f, i) => writeFileSync(join(frames, `mesh-f${String(i + 1).padStart(5, "0")}.obj`), writeObj(f)));
    // A box far from the mesh: nothing is deleted, so what the assertions read is purely the rebase.
    const far = { type: "box", min: [100, 100, 100], max: [101, 101, 101] };
    writeFileSync(join(dir, "clip.edits.json"), JSON.stringify({ aresEdits: 1, fps: 30, ranges: [{ id: "r1", mode: "delete", startFrame: 0, endFrame: 20, keyframes: [
      { frame: 0, derived: true, volumes: [far] },
      { frame: 8, volumes: [far] },                     // user, outside the window: ALWAYS kept
      { frame: 9, derived: true, volumes: [far] },      // derived, outside the window: dropped
    ] }] }));
    // --trim-out with no --trim-in is the common trim and the one rebaseEditList used to skip
    // entirely, so before the early return was removed this log line did not exist at all. The
    // count is the assertion: 2 would mean the `derived` test was ignored, 0 that nothing ran.
    const log = execFileSync(process.execPath, [CLI, "encode", frames, "-o", join(dir, "t.ares"), "--no-texture", "--edits", join(dir, "clip.edits.json"), "--trim-out", "1"], { encoding: "utf8" });
    assert.match(log, /trim: keeping source frames 0\.\.1 of 4/);
    assert.match(log, /trim: dropped 1 derived keyframe\(s\) outside the 2-frame window/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the rebase leaves a copy range's frame refs alone: span is not its gate, and a typo still aborts", () => {
  const dir = mkdtempSync(join(tmpdir(), "ares-copytrim-"));
  try {
    const frames = join(dir, "frames");
    mkdirSync(frames);
    synthClip("talk", 4, 30).frames.forEach((f, i) => writeFileSync(join(frames, `mesh-f${String(i + 1).padStart(5, "0")}.obj`), writeObj(f)));
    // Span 3..3, region = the whole mesh. The demo authors exactly this shape: ensureRange stamps
    // startFrame from the playhead while srcFrame/dstFrames are typed independently.
    const sidecar = (copy) => JSON.stringify({ aresEdits: 1, ranges: [{ id: "c1", mode: "delete", action: "copy",
      startFrame: 3, endFrame: 3, keyframes: [{ frame: 3, volumes: [{ type: "box", min: [-99, -99, -99], max: [99, 99, 99] }] }], copy }] });
    const file = join(dir, "copy.edits.json");
    const run = (...args) => execFileSync(process.execPath, [CLI, "encode", frames, "-o", join(dir, "t.ares"), "--no-texture", "--edits", file, ...args], { encoding: "utf8" });

    // A copy op's bake frames are srcFrame/dstFrames — frame-copy.ts evaluates its region with
    // prepareRangeAt(range, srcFrame) and never reads the span — so the "wholly outside the
    // window" drop must not reach it, even when the trim excludes the span outright.
    writeFileSync(file, sidecar({ srcFrame: 0, dstFrames: [1], what: "geo" }));
    const log = run("--trim-out", "1");
    assert.match(log, /copy c1 \(srcFrame 0 → frame 1\)/);
    assert.doesNotMatch(log, /dropping range c1/);

    // Untrimmed, a typo'd frame ref must still ABORT through collectCopyOps rather than be quietly
    // clamped away by the rebase — the contract frame-copy.ts's header states and recolor.ts cites.
    writeFileSync(file, sidecar({ srcFrame: 300, dstFrames: [1], what: "geo" }));
    assert.throws(() => run(), /copy\.srcFrame 300 out of range \[0, 4\)/);
    // The partial case is the invisible one: a dstFrames list with one bad entry pasted to the
    // good frames and said nothing.
    writeFileSync(file, sidecar({ srcFrame: 0, dstFrames: [1, 300], what: "geo" }));
    assert.throws(() => run(), /copy\.dstFrames entry 300 out of range \[0, 4\)/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
