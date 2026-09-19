import { test } from "node:test";
import assert from "node:assert/strict";
import { insideRangeAt, prepareRangeSdfAt, keepPredicateAt, filterIndicesByPredicate, rleEncodeMask, orbitViewProj } from "../dist/index.js";

// A propagated range carries ONE keyframe PER FRAME, which turns two latent bracket() defects into
// load-bearing ones (the sparse hand-authored ranges the evaluator was written against never hit an
// exact keyframe in the interior). These fixtures are the dense case.
const D = 4, FOV_Y = (50 * Math.PI) / 180;
const CAM = { azimuth: 0, elevation: 0, distance: D, target: [0, 0, 0], aspect: 1 };
/** Half the world height the camera sees at the target plane (camera.ts orbitViewHeight / 2). */
const HALF = D * Math.tan(FOV_Y / 2);
const ndcOf = (m, x, y, z) => {
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [(m[0] * x + m[4] * y + m[8] * z + m[12]) / w, (m[1] * x + m[5] * y + m[9] * z + m[13]) / w];
};
/** Flat-array position accessor in filterIndicesByPredicate's shape. */
const posOf = (p) => ({ x: (i) => p[i * 3], y: (i) => p[i * 3 + 1], z: (i) => p[i * 3 + 2] });

/** One 0/1 bitmap per frame: frame f sets exactly pixel column `cols[f]`, every row. */
function stripeRange(n, w, h, cols, extra = {}) {
  const keyframes = [];
  for (let f = 0; f < n; f++) {
    const bits = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) bits[y * w + cols[f]] = 1;
    keyframes.push({ frame: f, volumes: [{ type: "mask2d", kind: "bitmap", mask: { width: w, height: h, rle: rleEncodeMask(bits) }, camera: { ...CAM } }] });
  }
  return { id: "d", mode: "delete", startFrame: 0, endFrame: n - 1, keyframes, ...extra };
}

test("a keyframe on every frame resolves at frame f to frame f's own bitmap", () => {
  const N = 32, W = 32, H = 32;
  const m = orbitViewProj(CAM, 1);
  // Probe points on the target plane, one per pixel column (both projections agree there, so this
  // fixture is independent of the ortho case below).
  const X = [], cols = [];
  for (let f = 0; f < N; f++) {
    X.push((((f + 0.5) / W) * 2 - 1) * HALF);
    cols.push(Math.floor(((ndcOf(m, X[f], 0, 0)[0] + 1) / 2) * W));
  }
  assert.equal(new Set(cols).size, N, "fixture: each probe point must land in its own column");
  const r = stripeRange(N, W, H, cols);
  for (let f = 0; f < N; f++) {
    assert.ok(insideRangeAt(r, f, X[f], 0, 0), `frame ${f} must select bitmap ${f}`);
    if (f > 0) assert.ok(!insideRangeAt(r, f, X[f - 1], 0, 0), `frame ${f} must not still select bitmap ${f - 1}`);
    if (f < N - 1) assert.ok(!insideRangeAt(r, f, X[f + 1], 0, 0), `frame ${f} must not already select bitmap ${f + 1}`);
  }

  // An exact keyframe hit must compile ONE keyframe, not two. prepareVolume reads mask.height once
  // per keyframe it materialises, so this getter counts them: closed-on-b bracketing landed every
  // interior frame at t=1, which compiled and probed BOTH bracketing keyframes per centroid.
  let compiled = 0;
  const counted = { ...r, keyframes: r.keyframes.map((k) => ({ ...k, volumes: [{ ...k.volumes[0], mask: { width: W, rle: k.volumes[0].mask.rle, get height() { compiled++; return H; } } }] })) };
  prepareRangeSdfAt(counted, 17);
  assert.equal(compiled, 1, "frame 17 is an exact keyframe: only keyframe 17 is compiled");
});

test('interp:"hold" at an exact keyframe holds THAT keyframe, not the previous one', () => {
  const N = 16, W = 32, H = 32;
  const m = orbitViewProj(CAM, 1);
  const X = [], cols = [];
  for (let f = 0; f < N; f++) {
    X.push((((f + 0.5) / W) * 2 - 1) * HALF);
    cols.push(Math.floor(((ndcOf(m, X[f], 0, 0)[0] + 1) / 2) * W));
  }
  const r = stripeRange(N, W, H, cols, { interp: "hold" });
  // The regression this pins: "hold" forces t=0 ABOVE the t===0 fast path, so a bracket that
  // returned t=1 at an exact keyframe pinned every frame of a dense hold range one frame late.
  for (let f = 1; f < N; f++) {
    assert.ok(insideRangeAt(r, f, X[f], 0, 0), `hold at frame ${f} must select bitmap ${f}`);
    assert.ok(!insideRangeAt(r, f, X[f - 1], 0, 0), `hold at frame ${f} must not select bitmap ${f - 1}`);
  }
});

test("camera.ortho reaches the evaluator: an ortho-captured mask tests the ortho projection", () => {
  const W = 64, H = 64;
  // Both points sit 1.5 world units in FRONT of the target plane, where the two projections
  // disagree: ortho divides x by HALF (1.865), perspective by tan(25°)·(D−z) (1.166).
  const P1 = [1.2, 0, 1.5], P2 = [0.75, 0, 1.5];
  const orthoX = (p) => p[0] / HALF;
  const perspX = (p) => p[0] / (Math.tan(FOV_Y / 2) * (D - p[2]));
  // The mask is one NDC band, x in [0.5, 0.9]. P1 is inside it under ortho and off-screen under
  // perspective; P2 is outside it under ortho and inside it under perspective — so the two
  // projections select DIFFERENT triangles, and only one of them is what the user painted.
  assert.ok(orthoX(P1) > 0.5 && orthoX(P1) < 0.9 && perspX(P1) > 1, "fixture: P1 is ortho-inside, perspective-offscreen");
  assert.ok(orthoX(P2) < 0.5 && perspX(P2) > 0.5 && perspX(P2) < 0.9, "fixture: P2 is ortho-outside, perspective-inside");
  const bits = new Uint8Array(W * H);
  const lo = Math.ceil(((0.5 + 1) / 2) * W), hi = Math.floor(((0.9 + 1) / 2) * W);
  for (let y = 0; y < H; y++) for (let x = lo; x <= hi; x++) bits[y * W + x] = 1;
  const mask = { width: W, height: H, rle: rleEncodeMask(bits) };

  // Two triangles, one per probe point, each with its centroid exactly on the point.
  const positions = new Float32Array(18);
  [P1, P2].forEach((p, t) => {
    const off = [[0.01, 0, 0], [-0.005, 0.01, 0], [-0.005, -0.01, 0]];
    for (let v = 0; v < 3; v++) for (let c = 0; c < 3; c++) positions[(t * 3 + v) * 3 + c] = p[c] + off[v][c];
  });
  const indices = Uint32Array.from([0, 1, 2, 3, 4, 5]);
  const listWith = (camera) => ({ aresEdits: 1, ranges: [{ id: "o", mode: "delete", startFrame: 0, endFrame: 0, keyframes: [{ frame: 0, volumes: [{ type: "mask2d", kind: "bitmap", mask, camera }] }] }] });

  const kept = (camera) => Array.from(filterIndicesByPredicate(posOf(positions), indices, keepPredicateAt(listWith(camera), 0)));
  assert.deepEqual(kept({ ...CAM, ortho: true }), [3, 4, 5], "ortho capture deletes the triangle the ortho mask covers");
  assert.deepEqual(kept({ ...CAM }), [0, 1, 2], "the same mask under perspective deletes the other triangle");
});

test("camera.fov reaches the evaluator: a mask drawn through a narrow FOV tests that FOV", () => {
  const W = 64, H = 64;
  // Points on the target plane project to x / (tan(fov/2) D): at 50 degrees P1 lands at 0.64 NDC,
  // at 30 degrees (a relief zoomed in) at 1.12, off screen; P2 at 0.36 and 0.63.
  const P1 = [1.2, 0, 0], P2 = [0.67, 0, 0];
  const ndcX = (p, fovDeg) => p[0] / (Math.tan((fovDeg * Math.PI) / 360) * D);
  assert.ok(ndcX(P1, 50) > 0.5 && ndcX(P1, 50) < 0.9 && ndcX(P1, 30) > 1, "fixture: P1 is inside at 50, off screen at 30");
  assert.ok(ndcX(P2, 50) < 0.5 && ndcX(P2, 30) > 0.5 && ndcX(P2, 30) < 0.9, "fixture: P2 is outside at 50, inside at 30");
  const bits = new Uint8Array(W * H);
  const lo = Math.ceil(((0.5 + 1) / 2) * W), hi = Math.floor(((0.9 + 1) / 2) * W);
  for (let y = 0; y < H; y++) for (let x = lo; x <= hi; x++) bits[y * W + x] = 1;
  const mask = { width: W, height: H, rle: rleEncodeMask(bits) };
  const positions = new Float32Array(18);
  [P1, P2].forEach((p, t) => {
    const off = [[0.01, 0, 0], [-0.005, 0.01, 0], [-0.005, -0.01, 0]];
    for (let v = 0; v < 3; v++) for (let c = 0; c < 3; c++) positions[(t * 3 + v) * 3 + c] = p[c] + off[v][c];
  });
  const indices = Uint32Array.from([0, 1, 2, 3, 4, 5]);
  const listWith = (camera) => ({ aresEdits: 1, ranges: [{ id: "f", mode: "delete", startFrame: 0, endFrame: 0, keyframes: [{ frame: 0, volumes: [{ type: "mask2d", kind: "bitmap", mask, camera }] }] }] });
  const kept = (camera) => Array.from(filterIndicesByPredicate(posOf(positions), indices, keepPredicateAt(listWith(camera), 0)));
  assert.deepEqual(kept({ ...CAM, fov: 30 }), [0, 1, 2], "a 30 degree capture deletes the triangle its mask covers at 30 degrees");
  assert.deepEqual(kept({ ...CAM }), [3, 4, 5], "without fov the default 50 degrees applies");
});

test("sparse keyframes are untouched: linear still lerps, hold still holds the previous keyframe", () => {
  const r = { id: "s", mode: "delete", startFrame: 0, endFrame: 10, keyframes: [
    { frame: 0, volumes: [{ type: "box", min: [0, 0, 0], max: [1, 1, 1] }] },
    { frame: 10, volumes: [{ type: "box", min: [10, 0, 0], max: [11, 1, 1] }] },
  ] };
  const at = (interp, f) => prepareRangeSdfAt({ ...r, interp }, f)(0.5, 0.5, 0.5);
  assert.ok(Math.abs(at(undefined, 5) - (0.5 * -0.5 + 0.5 * 9.5)) < 1e-6, "t=0.5 mid-interval");
  assert.ok(Math.abs(at(undefined, 9) - (0.1 * -0.5 + 0.9 * 9.5)) < 1e-6, "t=0.9 one frame before the next keyframe");
  assert.ok(Math.abs(at(undefined, 0) - -0.5) < 1e-6, "the first keyframe alone at f=0");
  assert.ok(Math.abs(at(undefined, 10) - 9.5) < 1e-6, "the last keyframe alone at f=10");
  assert.ok(Math.abs(at("hold", 5) - -0.5) < 1e-6, "hold between sparse keyframes still holds keyframe 0");
  assert.ok(Math.abs(at("hold", 10) - 9.5) < 1e-6, "hold at the last keyframe is that keyframe");
});
