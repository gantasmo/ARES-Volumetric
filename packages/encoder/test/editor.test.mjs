import { test } from "node:test";
import assert from "node:assert/strict";
import { collectSculptOps, applySculptToFrame, synthClip, filterFrame } from "../dist/index.js";
import { parseEditList, keepPredicateAt, prepareRangeSdfAt, morphBitmap, growKeyframe, mirrorKeyframe, rleEncodeMask, rleDecodeMask, isRangeEnabled } from "@ares/core";

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
