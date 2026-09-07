import { test } from "node:test";
import assert from "node:assert/strict";
import { synthClip, buildTemporalGops, decimateFrame, simplifierReady, filterFrame, detectHoleLoopsForRange, appendCaps, computeSmoothNormals, parseObj } from "../dist/index.js";
import { parseEditList, keepPredicateAt, quantizePositions, quantizeUVs, computeAabb, dequantScale } from "@ares/core";

test("temporal planner: stable topology → I+P GOPs; forceIntra → intra", () => {
  const frames = synthClip("talk", 8, 30).frames;
  const gops = buildTemporalGops(frames, { gopLength: 4, track: false, smoothTemporal: 0, smoothSpatial: 0 });
  assert.equal(gops.length, 2);
  for (const g of gops) {
    assert.ok(g.temporal, "stable topology should inter-code");
    assert.equal(g.framePositions.length, 4);
    assert.equal(g.indices.length, frames[0].indices.length);
  }
  const intra = buildTemporalGops(frames, { gopLength: 4, track: false, smoothTemporal: 0, smoothSpatial: 0, forceIntra: true });
  assert.ok(intra.every((g) => !g.temporal && g.frames.length === 4));
  // A topology change forces a GOP split.
  const mixed = frames.map((f, i) => (i === 5 ? { ...f, indices: f.indices.subarray(0, f.indices.length - 3) } : f));
  const split = buildTemporalGops(mixed, { gopLength: 8, track: false, smoothTemporal: 0, smoothSpatial: 0 });
  assert.ok(split.length >= 2, `topology reset should split the GOP (got ${split.length})`);
});

test("decimate keeps roughly the requested triangle ratio", async () => {
  await simplifierReady();
  const f = synthClip("object", 1, 30).frames[0];
  const before = f.indices.length / 3;
  const d = decimateFrame(f, 0.5);
  const after = d.indices.length / 3;
  assert.ok(after < before * 0.75 && after > before * 0.2, `${before} → ${after} tris`);
  assert.equal(d.uvs.length / 2, d.positions.length / 3);
});

/** Open square grid (n×n vertices), y = 0, UVs = grid coords. */
function gridPlane(n) {
  const positions = new Float32Array(n * n * 3), uvs = new Float32Array(n * n * 2);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k = j * n + i;
    positions[k * 3] = i; positions[k * 3 + 1] = 0; positions[k * 3 + 2] = j;
    uvs[k * 2] = i / (n - 1); uvs[k * 2 + 1] = j / (n - 1);
  }
  const idx = [];
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  return { positions, uvs, indices: Uint32Array.from(idx) };
}

test("edit list delete + hole patch: one loop found inside the region and capped", () => {
  const frame = gridPlane(12);
  const list = parseEditList({
    aresEdits: 1,
    ranges: [{ id: "cut", mode: "delete", startFrame: 0, endFrame: 0, patchHoles: {},
      // Off-centre on purpose: the acceptance rule is "centroid inside OR any rim vertex inside",
      // so a box over the plane's centre would also accept the plane's own outer rim (its centroid
      // is the centre). AUDIT.md lists tightening that rule under hole filling.
      // Box edges on whole cells (triangle centroids sit at i+1/3, i+2/3) so the hole rim is a clean
      // rectangle rather than a saw-tooth with corner-touching (non-manifold) vertices.
      keyframes: [{ frame: 0, volumes: [{ type: "box", min: [2, -1, 2], max: [5, 1, 5] }] }] }],
  });
  const keep = keepPredicateAt(list, 0);
  assert.ok(keep);
  const cut = filterFrame(frame, keep);
  assert.ok(cut.indices.length < frame.indices.length);
  const { accepted, rejectedRegion } = detectHoleLoopsForRange(cut, list.ranges[0], 0, "0");
  assert.equal(accepted.length, 1, "exactly one hole loop inside the delete box");
  assert.equal(rejectedRegion, 1, "the outer boundary of the open plane is rejected (outside the region)");
  const { frame: capped, capTriangles } = appendCaps(cut, accepted, [null]);
  assert.ok(capTriangles > 0);
  assert.equal(capped.indices.length, cut.indices.length + capTriangles * 3);
  const n = computeSmoothNormals(capped.positions, capped.indices);
  assert.equal(n.length, capped.positions.length);
});

test("quantization round trip is within one step; UVs are 16-bit", () => {
  const f = synthClip("object", 1, 30).frames[0];
  const box = computeAabb(f.positions);
  const q = quantizePositions(f.positions, box, 14);
  const inv = dequantScale(14);
  const n = f.positions.length / 3;
  let worst = 0;
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {
    const size = box.max[a] - box.min[a];
    const v = box.min[a] + q[i * 4 + a] * inv * size;
    worst = Math.max(worst, Math.abs(v - f.positions[i * 3 + a]) / (size || 1));
  }
  assert.ok(worst <= 0.5 / ((1 << 14) - 1) + 1e-9, `worst relative error ${worst}`);
  const uv = quantizeUVs(new Float32Array([0, 1, 0.5, -0.2, 1.7, 0.25]));
  assert.deepEqual(Array.from(uv), [0, 65535, 32768, 0, 65535, 16384]);
});

test("OBJ import fan-triangulates and keeps UVs", () => {
  const src = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n";
  const m = parseObj(src);
  assert.equal(m.positions.length / 3, 4);
  assert.equal(m.indices.length, 6);
  assert.ok(m.uvs && m.uvs.length === 8);
});
