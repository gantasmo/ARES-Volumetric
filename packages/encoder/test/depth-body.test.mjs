import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  shellMesh, rasterizeDepth, outwardSign, anchorScale, smoothSeries, smoothVertexWindow,
  assembleHybrid, BodyColorAccumulator, bakeBackAtlas, toAresSpace,
} from "../dist/depth-body.js";

/*
 * fixtures/mhr-probe.bin: one SAM 3D Body (MHR) mesh, pred_vertices + pred_cam_t from a 1920x1080
 * frame with K = [[1920,0,960],[0,1920,540],[0,0,1]], OpenCV camera metres (probe 2026-09-19).
 * Layout: "MHR1", u32 V, u32 F, u32 0, then V x 3 float32 LE, then F x 3 uint16 LE.
 */
const FIXTURE = fileURLToPath(new URL("./fixtures/mhr-probe.bin", import.meta.url));
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
  mhrCache = { V, F, body, faces };
  return mhrCache;
}
const probeK = (s) => ({ fx: 1920 * s, fy: 1920 * s, cx: 960 * s, cy: 540 * s });

function lcg(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}

/** Möller-Trumbore: every hit distance t > 1e-9 along (d) from (o). */
function rayHits(body, faces, o, d, nearestOnly) {
  let hits = 0, nearest = Infinity;
  for (let t = 0; t < faces.length / 3; t++) {
    const a = faces[t * 3] * 3, b = faces[t * 3 + 1] * 3, c = faces[t * 3 + 2] * 3;
    const e1x = body[b] - body[a], e1y = body[b + 1] - body[a + 1], e1z = body[b + 2] - body[a + 2];
    const e2x = body[c] - body[a], e2y = body[c + 1] - body[a + 1], e2z = body[c + 2] - body[a + 2];
    const px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (Math.abs(det) < 1e-18) continue;
    const inv = 1 / det;
    const tx = o[0] - body[a], ty = o[1] - body[a + 1], tz = o[2] - body[a + 2];
    const u = (tx * px + ty * py + tz * pz) * inv;
    if (u < 0 || u > 1) continue;
    const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
    const v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
    if (v < 0 || u + v > 1) continue;
    const dist = (e2x * qx + e2y * qy + e2z * qz) * inv;
    if (dist > 1e-9) { hits++; if (dist < nearest) nearest = dist; }
  }
  return nearestOnly ? nearest : hits;
}

const faceNormal = (P, a, b, c) => {
  const ux = P[b * 3] - P[a * 3], uy = P[b * 3 + 1] - P[a * 3 + 1], uz = P[b * 3 + 2] - P[a * 3 + 2];
  const vx = P[c * 3] - P[a * 3], vy = P[c * 3 + 1] - P[a * 3 + 1], vz = P[c * 3 + 2] - P[a * 3 + 2];
  return [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
};
const centroid = (P, a, b, c) => [0, 1, 2].map((k) => (P[a * 3 + k] + P[b * 3 + k] + P[c * 3 + k]) / 3);
const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
const reverseFaces = (faces) => { const r = new Uint32Array(faces); for (let t = 0; t < r.length; t += 3) { r[t + 1] = faces[t + 2]; r[t + 2] = faces[t + 1]; } return r; };

/* ------------------------------------------------------------------------------------------------ */

test("outwardSign: +1 on the MHR probe, confirmed by ray parity; -1 once the faces are reversed", () => {
  const { body, faces, F } = mhr();
  // Closed 2-manifold: every edge is shared by exactly two faces, so the signed volume is well defined.
  const edges = new Map();
  for (let t = 0; t < F; t++) {
    for (let k = 0; k < 3; k++) {
      const a = faces[t * 3 + k], b = faces[t * 3 + ((k + 1) % 3)];
      const key = a < b ? a * 65536 + b : b * 65536 + a;
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
  }
  assert.ok([...edges.values()].every((c) => c === 2), "MHR probe is a closed 2-manifold");

  const sign = outwardSign(body, faces);
  assert.equal(sign, 1);
  assert.equal(outwardSign(body, reverseFaces(faces)), -1);

  // Mesh-wide normal check: a point just off a face along sign*(b-a)x(c-a) lies outside the mesh
  // (even ray-crossing count) for nearly every sampled face.
  const rnd = lcg(7);
  let outside = 0;
  const N = 100;
  for (let s = 0; s < N; s++) {
    const t = Math.floor(rnd() * F);
    const a = faces[t * 3], b = faces[t * 3 + 1], c = faces[t * 3 + 2];
    const n = faceNormal(body, a, b, c), l = Math.hypot(...n), g = centroid(body, a, b, c);
    const o = g.map((x, k) => x + (sign * n[k] / l) * 1e-4);
    let d = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
    const dl = Math.hypot(...d); d = d.map((x) => x / dl);
    if (rayHits(body, faces, o, d, false) % 2 === 0) outside++;
  }
  assert.ok(outside >= 95, `${outside}/${N} offset points outside`);

  // A unit tetrahedron with known outward winding.
  const tet = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const tf = new Uint32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
  assert.equal(outwardSign(tet, tf), 1);
  assert.equal(outwardSign(tet, reverseFaces(tf)), -1);
});

test("rasterizeDepth: MHR at 480x270 covers the projected box, matches a ray cast, and sits on the vertices", () => {
  const { body, faces, V } = mhr();
  const W = 480, H = 270, K = probeK(0.25);
  const faceOut = new Int32Array(W * H);
  const zb = rasterizeDepth(body, faces, W, H, K, undefined, faceOut);
  assert.equal(zb.length, W * H);

  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
  for (let v = 0; v < V; v++) {
    const z = body[v * 3 + 2], u = (K.fx * body[v * 3]) / z + K.cx, w = (K.fy * body[v * 3 + 1]) / z + K.cy;
    u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, w); v1 = Math.max(v1, w);
  }
  let x0 = W, x1 = -1, y0 = H, y1 = -1, covered = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = y * W + x;
      assert.equal(zb[p] < Infinity, faceOut[p] >= 0, "faceOut agrees with the z-buffer");
      if (!(zb[p] < Infinity)) continue;
      covered++;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      // Every covered pixel centre is inside the projected box.
      assert.ok(x + 0.5 >= u0 && x + 0.5 <= u1 && y + 0.5 >= v0 && y + 0.5 <= v1, `pixel ${x},${y} outside the box`);
    }
  }
  assert.ok(covered > 3000, `${covered} pixels covered`);
  // The covered extent reaches the projected box to within one pixel on every side.
  assert.ok(x0 + 0.5 - u0 <= 1 && u1 - (x1 + 0.5) <= 1, `x ${x0}..${x1} vs ${u0.toFixed(2)}..${u1.toFixed(2)}`);
  assert.ok(y0 + 0.5 - v0 <= 1 && v1 - (y1 + 0.5) <= 1, `y ${y0}..${y1} vs ${v0.toFixed(2)}..${v1.toFixed(2)}`);

  // Exact: the nearest hit of a ray through the pixel centre, for sampled pixels in and around the box.
  const rnd = lcg(11);
  for (let s = 0; s < 250; s++) {
    const x = Math.floor(x0 - 3 + rnd() * (x1 - x0 + 7)), y = Math.floor(y0 - 3 + rnd() * (y1 - y0 + 7));
    const d = [(x + 0.5 - K.cx) / K.fx, (y + 0.5 - K.cy) / K.fy, 1];
    const zr = rayHits(body, faces, [0, 0, 0], d, true); // d.z = 1, so the distance is z
    const z = zb[y * W + x];
    if (zr === Infinity) assert.equal(z, Infinity, `pixel ${x},${y}: ray misses, z-buffer ${z}`);
    else assert.ok(Math.abs(z - zr) < 1e-4, `pixel ${x},${y}: z-buffer ${z} vs ray ${zr}`);
  }

  // At vertex pixels: no vertex is nearer than the z-buffer in its 3x3 pixel neighbourhood by more
  // than 1 cm, and camera-facing vertices sit on it (median under 3 mm, 85 % within 1 cm).
  const nrm = new Float64Array(V * 3);
  for (let t = 0; t < faces.length; t += 3) {
    const n = faceNormal(body, faces[t], faces[t + 1], faces[t + 2]);
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) nrm[faces[t + k] * 3 + j] += n[j];
  }
  const facing = [];
  for (let v = 0; v < V; v++) {
    const x = body[v * 3], y = body[v * 3 + 1], z = body[v * 3 + 2];
    const px = Math.floor((K.fx * x) / z + K.cx), py = Math.floor((K.fy * y) / z + K.cy);
    let m = Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) m = Math.min(m, zb[(py + dy) * W + px + dx]);
    assert.ok(m <= z + 0.01, `vertex ${v} at z ${z} is 1 cm nearer than the z-buffer ${m}`);
    const n = [nrm[v * 3], nrm[v * 3 + 1], nrm[v * 3 + 2]];
    const cos = -dot(n, [x, y, z]) / (Math.hypot(...n) * Math.hypot(x, y, z));
    if (cos > 0.5) facing.push(Math.abs(zb[py * W + px] - z));
  }
  facing.sort((a, b) => a - b);
  const within = facing.filter((e) => e < 0.01).length / facing.length;
  assert.ok(facing[facing.length >> 1] < 0.003, `facing median ${facing[facing.length >> 1]}`);
  assert.ok(within > 0.85, `facing within 1 cm: ${within}`);

  // Reuse of out/faceOut clears the previous frame.
  const empty = rasterizeDepth(new Float32Array(0), new Uint32Array(0), W, H, K, zb, faceOut);
  assert.equal(empty, zb);
  assert.ok(zb.every((z) => z === Infinity) && faceOut.every((f) => f === -1));
});

/* ------------------------------------------------------------------------------------------------ */

test("shellMesh: sphere on a plane loses the silhouette and sheer triangles, keeps components by size", () => {
  const W = 320, H = 240, K = { fx: 400, fy: 400, cx: 120, cy: 120 };
  const z = new Float32Array(W * H);
  const C = [0, 0, 2.3], R = 0.4, PLANE = 3.0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < 250; x++) {
      const d = [(x + 0.5 - K.cx) / K.fx, (y + 0.5 - K.cy) / K.fy, 1];
      // Ray-sphere: |t d - C|^2 = R^2, nearest root; z = t since d.z = 1.
      const a = dot(d, d), b = -2 * dot(d, C), c = dot(C, C) - R * R, disc = b * b - 4 * a * c;
      z[y * W + x] = disc >= 0 ? (-b - Math.sqrt(disc)) / (2 * a) : PLANE;
    }
  }
  // Islands right of a zero gap: A is 5x5 (3x3 after erosion), B is 30x30 (28x28).
  for (let y = 20; y < 25; y++) for (let x = 270; x < 275; x++) z[y * W + x] = 2.8;
  for (let y = 100; y < 130; y++) for (let x = 270; x < 300; x++) z[y * W + x] = 2.8;
  const uv = { vScale: 0.5, vOffset: 0.25 };

  const s = shellMesh(z, W, H, K, uv);
  const nv = s.positions.length / 3, nt = s.indices.length / 3;
  assert.equal(s.cells, nv);
  assert.equal(s.uvs.length, nv * 2);
  let mapped = 0;
  for (let i = 0; i < W * H; i++) if (s.cellIndex[i] >= 0) mapped++;
  assert.equal(mapped, nv);

  // Erosion: the map border and the islands' outer rings are gone.
  for (let x = 0; x < W; x++) { assert.equal(s.cellIndex[x], -1); assert.equal(s.cellIndex[(H - 1) * W + x], -1); }
  for (let y = 0; y < H; y++) assert.equal(s.cellIndex[y * W], -1);
  assert.equal(s.cellIndex[100 * W + 270], -1, "island B's rim is eroded");
  // Default minCells = round(500 * 320*240 / (848*480)) = 94: A (9 cells) dropped, B (784) kept.
  assert.equal(s.cellIndex[22 * W + 272], -1, "island A is below minCells");
  assert.ok(s.cellIndex[115 * W + 285] >= 0, "island B is kept");
  const only = shellMesh(z, W, H, K, uv, { minCells: 1e9 });
  assert.equal(only.cellIndex[115 * W + 285], -1, "island B dropped when only the largest survives");
  assert.ok(only.cellIndex[120 * W + 120] >= 0, "the largest component always survives");

  // Vertex at a cell centre, uv in the frame region.
  const k = s.cellIndex[120 * W + 120], zc = z[120 * W + 120];
  assert.ok(Math.abs(s.positions[k * 3] - ((120.5 - K.cx) * zc) / K.fx) < 1e-6);
  assert.ok(Math.abs(s.positions[k * 3 + 1] - ((120.5 - K.cy) * zc) / K.fy) < 1e-6);
  assert.ok(Math.abs(s.positions[k * 3 + 2] - zc) < 1e-6);
  assert.ok(Math.abs(s.uvs[k * 2] - 120.5 / W) < 1e-6 && Math.abs(s.uvs[k * 2 + 1] - (0.25 + (120.5 / H) * 0.5)) < 1e-6);
  for (let i = 0; i < nv; i++) {
    const u = s.uvs[i * 2], v = s.uvs[i * 2 + 1];
    assert.ok(u > 0 && u < 1 && v > 0.25 && v < 0.75, `uv ${u},${v}`);
  }

  // Kept triangles: z span within 5 cm, not edge-on, facing the camera; both surfaces present.
  const cosOf = (P, a, b, c) => { const n = faceNormal(P, a, b, c), g = centroid(P, a, b, c); return dot(n, g) / (Math.hypot(...n) * Math.hypot(...g)); };
  let sphereTris = 0, planeTris = 0;
  for (let t = 0; t < nt; t++) {
    const a = s.indices[t * 3], b = s.indices[t * 3 + 1], c = s.indices[t * 3 + 2];
    const zs = [a, b, c].map((i) => s.positions[i * 3 + 2]);
    assert.ok(Math.max(...zs) - Math.min(...zs) <= 0.05 + 1e-6, `triangle ${t} spans ${Math.max(...zs) - Math.min(...zs)}`);
    const cos = cosOf(s.positions, a, b, c);
    assert.ok(cos <= -0.2 + 1e-9, `triangle ${t}: cos ${cos} (edge-on or facing away)`);
    if (Math.max(...zs) < PLANE - 0.3) sphereTris++; else if (Math.min(...zs) > PLANE - 0.25) planeTris++;
  }
  assert.ok(sphereTris > 1000 && planeTris > 10000, `sphere ${sphereTris}, plane ${planeTris}`);
  // Nothing bridges sphere and plane (the 0.7 m silhouette jump).
  // Sheer: with the rule off, edge-on triangles within the 5 cm span appear at the sphere's rim.
  const loose = shellMesh(z, W, H, K, uv, { sheer: 0 });
  let edgeOn = 0;
  for (let t = 0; t < loose.indices.length / 3; t++) {
    const cos = cosOf(loose.positions, loose.indices[t * 3], loose.indices[t * 3 + 1], loose.indices[t * 3 + 2]);
    if (Math.abs(cos) < 0.2) edgeOn++;
  }
  assert.ok(edgeOn > 0, "the sheer rule has triangles to drop");
  assert.equal(loose.indices.length / 3 - nt, edgeOn, "sheer 0.2 drops exactly the edge-on triangles");
  // Without the span rule the silhouette is bridged.
  const bridged = shellMesh(z, W, H, K, uv, { edge: 10, sheer: 0 });
  let bridge = 0;
  for (let t = 0; t < bridged.indices.length / 3; t++) {
    const zs = [0, 1, 2].map((j) => bridged.positions[bridged.indices[t * 3 + j] * 3 + 2]);
    if (Math.max(...zs) - Math.min(...zs) > 0.3) bridge++;
  }
  assert.ok(bridge > 0, "edge 0.05 is what cuts the silhouette");

  assert.throws(() => shellMesh(z, W, H, { fx: 0, fy: 400, cx: 0, cy: 0 }, uv), /fx 0/);
  assert.throws(() => shellMesh(z, W, H, K, uv, { edge: -1 }), /edge .* got -1/);
  assert.throws(() => shellMesh(z.subarray(0, 10), W, H, K, uv), /z holds 10/);
});

/* ------------------------------------------------------------------------------------------------ */

test("anchorScale: recovers a 0.9 body scale against the body's own z-buffer", () => {
  const { body, faces } = mhr();
  const W = 518, H = 292, K = probeK(518 / 1920);
  const shellZ = rasterizeDepth(body, faces, W, H, K);
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) if (shellZ[i] < Infinity) mask[i] = 1; else shellZ[i] = 0;
  const small = body.map((x) => x * 0.9);
  const bodyZ = rasterizeDepth(small, faces, W, H, K);
  const r = anchorScale(bodyZ, shellZ, mask, W, H);
  assert.ok(r);
  assert.ok(Math.abs(r.scale - 1 / 0.9) < 1e-4, `scale ${r.scale}`);
  assert.ok(r.mad < 1e-4, `mad ${r.mad}`);
  assert.ok(r.shared > 3000, `shared ${r.shared}`);
  // Scaling about the camera origin keeps the silhouette: the scaled body covers the same pixels.
  let same = 0, diff = 0;
  for (let i = 0; i < W * H; i++) { if ((bodyZ[i] < Infinity) === (mask[i] === 1)) same++; else diff++; }
  assert.ok(diff <= 5, `${diff} pixels changed coverage`);
  // Mask restricts the pixels; too few shared pixels gives null.
  const left = new Uint8Array(mask);
  for (let y = 0; y < H; y++) for (let x = 290; x < W; x++) left[y * W + x] = 0;
  const rl = anchorScale(bodyZ, shellZ, left, W, H);
  assert.ok(rl && rl.shared < r.shared && Math.abs(rl.scale - 1 / 0.9) < 1e-4);
  assert.equal(anchorScale(bodyZ, shellZ, mask, W, H, r.shared + 1), null);
  assert.throws(() => anchorScale(bodyZ, shellZ, mask, W, H, 0), /minShared .* got 0/);
});

test("smoothSeries and smoothVertexWindow: outliers rejected, gaps filled, weights centred", () => {
  const vals = new Float64Array([1, 1, 1, 5, 1, 1, NaN, 1, 1, 1]);
  const valid = new Uint8Array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1]);
  const s0 = smoothSeries(vals, valid, 0);
  assert.deepEqual([...s0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "median-5 removes the spike, the gap is filled");
  const ramp = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const gap = new Uint8Array(10).fill(1); gap[0] = 0; gap[5] = 0;
  const sr = smoothSeries(ramp, gap, 0);
  assert.equal(sr[5], 5, "interior gap interpolated");
  assert.equal(sr[0], sr[1], "leading gap held flat");
  const long = Float64Array.from({ length: 30 }, (_, i) => i);
  const sg = smoothSeries(long, new Uint8Array(30).fill(1), 4);
  assert.ok(Math.abs(sg[15] - 15) < 1e-9, `Gaussian keeps a ramp's interior: ${sg[15]}`);
  assert.ok(sg[0] > 0 && sg[29] < 29, "and pulls its ends in");
  assert.throws(() => smoothSeries(vals, new Uint8Array(10), 2), /no valid entry among 10/);

  const w = [new Float32Array([0, 0]), new Float32Array([1, 2]), new Float32Array([2, 4])];
  assert.deepEqual([...smoothVertexWindow(w, 1, 0)], [1, 2]);
  const m = smoothVertexWindow(w, 1, 1);
  assert.ok(Math.abs(m[0] - 1) < 1e-6 && Math.abs(m[1] - 2) < 1e-6, "symmetric window around a linear motion");
  const e = smoothVertexWindow(w, 0, 1);
  assert.ok(e[0] > 0 && e[0] < 1, "edge frame pulled toward its neighbours");
  assert.throws(() => smoothVertexWindow([w[0], new Float32Array(3)], 0, 1), /frame 1 holds 3/);
});

/* ------------------------------------------------------------------------------------------------ */

/** Test-side far raster (largest z per pixel centre) of a triangle soup: independent of rasterPass. */
function farRaster(P, idx, W, H, K) {
  const out = new Float32Array(W * H).fill(-Infinity);
  for (let t = 0; t < idx.length / 3; t++) {
    const pts = [0, 1, 2].map((j) => { const i = idx[t * 3 + j] * 3; const z = P[i + 2]; return [(K.fx * P[i]) / z + K.cx, (K.fy * P[i + 1]) / z + K.cy, z]; });
    const [A, Bp, Cp] = pts;
    const area = (Bp[0] - A[0]) * (Cp[1] - A[1]) - (Bp[1] - A[1]) * (Cp[0] - A[0]);
    if (Math.abs(area) < 1e-12) continue;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    for (let y = Math.max(0, Math.floor(Math.min(...ys))); y <= Math.min(H - 1, Math.ceil(Math.max(...ys))); y++) {
      for (let x = Math.max(0, Math.floor(Math.min(...xs))); x <= Math.min(W - 1, Math.ceil(Math.max(...xs))); x++) {
        const sx = x + 0.5, sy = y + 0.5;
        const w0 = ((Cp[0] - Bp[0]) * (sy - Bp[1]) - (Cp[1] - Bp[1]) * (sx - Bp[0])) / area;
        const w1 = ((A[0] - Cp[0]) * (sy - Cp[1]) - (A[1] - Cp[1]) * (sx - Cp[0])) / area;
        const w2 = 1 - w0 - w1;
        if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9) continue;
        const z = 1 / (w0 / A[2] + w1 / Bp[2] + w2 / Cp[2]);
        if (z > out[y * W + x]) out[y * W + x] = z;
      }
    }
  }
  return out;
}

test("assembleHybrid: MHR against its own front carves the front, keeps the back, backs only uncovered shell", (t) => {
  const { body, faces, F } = mhr();
  const W = 518, H = 292, K = probeK(518 / 1920);
  const zb = rasterizeDepth(body, faces, W, H, K);
  const shellZ = new Float32Array(W * H), mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) if (zb[i] < Infinity) { shellZ[i] = zb[i]; mask[i] = 1; }
  const sign = outwardSign(body, faces);
  const backUv = new Float32Array(F * 6);
  for (let i = 0; i < backUv.length; i++) backUv[i] = (i % 97) / 97;
  const frameUv = { vScale: 0.8, vOffset: 0 };
  const cover = 0.10, back = 0.06, pushGap = 0.005;

  // Timing at 518x292: rasterize + assemble, warm, median of 7 (target under 60 ms).
  for (let r = 0; r < 3; r++) { rasterizeDepth(body, faces, W, H, K, zb); assembleHybrid(shellZ, mask, W, H, K, body, faces, sign, backUv, frameUv); }
  const times = [];
  let res;
  for (let r = 0; r < 7; r++) {
    const t0 = performance.now();
    rasterizeDepth(body, faces, W, H, K, zb);
    res = assembleHybrid(shellZ, mask, W, H, K, body, faces, sign, backUv, frameUv);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  t.diagnostic(`rasterizeDepth + assembleHybrid at 518x292: median ${times[3].toFixed(1)} ms, stats ${JSON.stringify(res.stats)}`);
  assert.ok(times[3] < 60, `median ${times[3]} ms`);

  const { body: out, backing, stats } = res;
  assert.equal(out.positions.length, stats.keptTris * 9);
  assert.equal(out.uvs.length, stats.keptTris * 6);
  assert.deepEqual([...out.indices.subarray(0, 6)], [0, 1, 2, 3, 4, 5]);
  assert.equal(stats.flaps, 0, "every vertex projects onto the mask");
  assert.ok(stats.covered > 0 && stats.pushed > 0);

  // Nothing in front of the shell: every kept vertex is at least pushGap behind it at its own pixel.
  const P = out.positions;
  for (let v = 0; v < P.length / 3; v++) {
    const z = P[v * 3 + 2];
    const px = Math.floor((K.fx * P[v * 3]) / z + K.cx), py = Math.floor((K.fy * P[v * 3 + 1]) / z + K.cy);
    if (px < 0 || py < 0 || px >= W || py >= H) continue;
    const s = shellZ[py * W + px];
    if (s > 0) assert.ok(z >= s + pushGap - 1e-6, `vertex ${v}: z ${z} vs shell ${s}`);
  }

  // Classify the input triangles independently: covered front carved, back kept.
  const vz = (v) => body[v * 3 + 2];
  // The shell a vertex is judged against: its own cell, else the nearest non-empty cell within 2.
  const shellAtV = (v) => {
    const z = vz(v), px = Math.floor((K.fx * body[v * 3]) / z + K.cx), py = Math.floor((K.fy * body[v * 3 + 1]) / z + K.cy);
    if (shellZ[py * W + px] > 0) return shellZ[py * W + px];
    let best = 0, bd = Infinity;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const x = px + dx, y = py + dy;
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      const sz = shellZ[y * W + x];
      if (sz > 0 && dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; best = sz; }
    }
    return best;
  };
  // The rule recomputed on the original mesh; output triangle k is the k-th kept face in order, each
  // corner on its own camera ray (pushed by scaling, never by a z shift).
  let frontCovered = 0, frontAll = 0, backAll = 0, backKept = 0, k = 0;
  for (let tt = 0; tt < F; tt++) {
    const a = faces[tt * 3], b = faces[tt * 3 + 1], c = faces[tt * 3 + 2];
    const cov = [a, b, c].every((v) => { const s = shellAtV(v); return s > 0 && vz(v) < s + cover; });
    const facesCam = sign * dot(faceNormal(body, a, b, c), centroid(body, a, b, c)) < 0;
    if (facesCam) frontAll++; else backAll++;
    if (cov) { if (facesCam) frontCovered++; continue; }
    for (let j = 0; j < 3; j++) {
      const v = faces[tt * 3 + j], o = k * 9 + j * 3, s = P[o + 2] / vz(v);
      if (!(s >= 1 - 1e-6) || Math.abs(P[o] - s * body[v * 3]) > 1e-5 || Math.abs(P[o + 1] - s * body[v * 3 + 1]) > 1e-5)
        assert.fail(`kept triangle ${k} corner ${j} is not face ${tt}'s vertex ${v} on its ray`);
    }
    k++;
    if (!facesCam) backKept++;
  }
  assert.equal(stats.keptTris, k, "kept = every triangle not fully covered (no flaps with this mask)");
  assert.equal(stats.covered, F - k);
  t.diagnostic(`camera-facing ${frontAll} (covered and carved ${frontCovered}), facing away ${backAll} (kept ${backKept})`);
  assert.ok(frontCovered > 0.9 * frontAll, `${frontCovered} of ${frontAll} camera-facing triangles carved`);
  // Every back triangle reaching past the cover band survives; the rest of the back lies within
  // `cover` of the front (limbs) and is carved with it.
  let deepAll = 0;
  for (let tt = 0; tt < F; tt++) {
    const a = faces[tt * 3], b = faces[tt * 3 + 1], c = faces[tt * 3 + 2];
    if (sign * dot(faceNormal(body, a, b, c), centroid(body, a, b, c)) < 0) continue;
    if ([a, b, c].some((v) => { const s = shellAtV(v); return !(s > 0) || vz(v) >= s + cover; })) deepAll++;
  }
  assert.equal(backKept, deepAll, "back triangles kept = those reaching past the cover band");
  assert.ok(backKept > 4000, `${backKept} back triangles kept`);

  // Output winding is outward: the same result from reversed faces with sign -1 and reversed uvs.
  const rf = reverseFaces(faces), ruv = new Float32Array(backUv);
  for (let tt = 0; tt < F; tt++) for (let j = 0; j < 2; j++) { ruv[tt * 6 + 2 + j] = backUv[tt * 6 + 4 + j]; ruv[tt * 6 + 4 + j] = backUv[tt * 6 + 2 + j]; }
  const rres = assembleHybrid(shellZ, mask, W, H, K, body, rf, -1, ruv, frameUv);
  assert.deepEqual(rres.body.positions, out.positions);
  assert.deepEqual(rres.body.uvs, out.uvs);

  // Backing only where no kept body lies more than `back` behind the shell.
  const far = farRaster(out.positions, out.indices, W, H, K);
  const B = backing.positions;
  assert.equal(B.length / 3, stats.backingCells);
  assert.ok(stats.backingCells > 50, `${stats.backingCells} backing cells`);
  for (let k = 0; k < B.length / 3; k++) {
    const z = B[k * 3 + 2];
    const u = (K.fx * B[k * 3]) / z + K.cx, v = (K.fy * B[k * 3 + 1]) / z + K.cy;
    const px = Math.floor(u), py = Math.floor(v), p = py * W + px;
    assert.ok(Math.abs(u - px - 0.5) < 1e-3 && Math.abs(v - py - 0.5) < 1e-3, "backing vertex at a cell centre");
    const s = shellZ[p];
    assert.ok(s > 0, "backing only behind the shell");
    assert.ok(z >= s + 0.008 - 1e-5 && z <= s + 0.12 + 1e-5, `backing depth ${z - s}`);
    assert.ok(!(far[p] > s + back), `backing at ${px},${py} where the body lies ${far[p] - s} m behind`);
    assert.ok(Math.abs(backing.uvs[k * 2] - (px + 0.5) / W) < 1e-6 && Math.abs(backing.uvs[k * 2 + 1] - ((py + 0.5) / H) * 0.8) < 1e-6);
  }
  // Backing faces away from the camera.
  for (let k = 0; k < backing.indices.length / 3; k++) {
    const a = backing.indices[k * 3], b = backing.indices[k * 3 + 1], c = backing.indices[k * 3 + 2];
    assert.ok(dot(faceNormal(B, a, b, c), centroid(B, a, b, c)) > 0, `backing triangle ${k} faces the camera`);
  }

  // Flaps: with the mask eroded by 4 px, camera-facing triangles with an outside vertex are dropped
  // and every kept triangle with an outside vertex faces away.
  let er = new Uint8Array(mask);
  for (let pass = 0; pass < 4; pass++) {
    const nx = new Uint8Array(W * H);
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (er[i] && er[i - 1] && er[i + 1] && er[i - W] && er[i + W]) nx[i] = 1;
    }
    er = nx;
  }
  const fr = assembleHybrid(shellZ, er, W, H, K, body, faces, sign, backUv, frameUv);
  assert.ok(fr.stats.flaps > 0, "eroded mask produces flaps");
  const outside = (v) => {
    const z = vz(v), px = Math.floor((K.fx * body[v * 3]) / z + K.cx), py = Math.floor((K.fy * body[v * 3 + 1]) / z + K.cy);
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      if (dx * dx + dy * dy > 4) continue;
      const x = px + dx, y = py + dy;
      if (x >= 0 && y >= 0 && x < W && y < H && er[y * W + x]) return false;
    }
    return true;
  };
  let flapExpect = 0;
  for (let tt = 0; tt < F; tt++) {
    const a = faces[tt * 3], b = faces[tt * 3 + 1], c = faces[tt * 3 + 2];
    const cov = [a, b, c].every((v) => { const s = shellAtV(v); return s > 0 && vz(v) < s + cover; });
    if (cov) continue;
    if ([a, b, c].some(outside) && sign * dot(faceNormal(body, a, b, c), centroid(body, a, b, c)) < 0) flapExpect++;
  }
  assert.equal(fr.stats.flaps, flapExpect);
  assert.equal(fr.stats.keptTris, stats.keptTris - flapExpect);

  assert.throws(() => assembleHybrid(shellZ, mask, W, H, K, body, faces, 0, backUv, frameUv), /sign must be 1 or -1, got 0/);
  assert.throws(() => assembleHybrid(shellZ, mask, W, H, K, body, faces, sign, backUv.subarray(0, 6), frameUv), /backUv holds 6/);
  assert.throws(() => assembleHybrid(shellZ, mask, W, H, K, body, faces, sign, backUv, frameUv, { rimCells: 0.5 }), /rimCells .* got 0.5/);
});

/* ------------------------------------------------------------------------------------------------ */

test("BodyColorAccumulator: front vertices take their side's colour, the back fills from neighbours", () => {
  const { body, faces, V } = mhr();
  const W = 480, H = 270, K = probeK(0.25);
  const zb = rasterizeDepth(body, faces, W, H, K);
  const us = [];
  for (let v = 0; v < V; v++) us.push((K.fx * body[v * 3]) / body[v * 3 + 2] + K.cx);
  us.sort((a, b) => a - b);
  const split = us[V >> 1];
  const rgb = new Uint8Array(W * H * 3), mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = y * W + x;
    if (x + 0.5 < split) rgb[p * 3] = 255; else rgb[p * 3 + 2] = 255;
    if (zb[p] < Infinity) mask[p] = 1;
  }
  const acc = new BodyColorAccumulator(V, faces);
  const sign = outwardSign(body, faces);
  const seen = acc.add(body, rgb, mask, W, H, K, zb, sign);
  const { colors, observed, filled } = acc.finalize();
  assert.equal(observed, seen);
  assert.ok(seen > 5000 && seen < V - 5000, `${seen} visible`);
  assert.equal(observed + filled, V, "one connected mesh: every unobserved vertex is reached by the rings");

  const nrm = new Float64Array(V * 3);
  for (let t = 0; t < faces.length; t += 3) {
    const n = faceNormal(body, faces[t], faces[t + 1], faces[t + 2]);
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) nrm[faces[t + k] * 3 + j] += sign * n[j];
  }
  let frontL = 0, frontR = 0, backL = 0, backR = 0;
  for (let v = 0; v < V; v++) {
    const x = body[v * 3], y = body[v * 3 + 1], z = body[v * 3 + 2];
    const u = (K.fx * x) / z + K.cx, p = Math.floor((K.fy * y) / z + K.cy) * W + Math.floor(u);
    const n = [nrm[v * 3], nrm[v * 3 + 1], nrm[v * 3 + 2]];
    const cos = -dot(n, [x, y, z]) / (Math.hypot(...n) * Math.hypot(x, y, z));
    const r = colors[v * 3], g = colors[v * 3 + 1], b = colors[v * 3 + 2];
    assert.equal(g, 0);
    const visible = cos > 0.2 && zb[p] < Infinity && z - zb[p] <= Math.max(0.02, 0.01 * z);
    if (visible && u < split - 2) { frontL++; assert.ok(r > 200 && b < 55, `front-left vertex ${v}: ${r},${b}`); }
    if (visible && u > split + 2) { frontR++; assert.ok(b > 200 && r < 55, `front-right vertex ${v}: ${r},${b}`); }
    if (cos < -0.2 && u < split - 10) { backL++; assert.ok(r > b, `back-left vertex ${v}: ${r},${b}`); }
    if (cos < -0.2 && u > split + 10) { backR++; assert.ok(b > r, `back-right vertex ${v}: ${r},${b}`); }
  }
  assert.ok(frontL > 1000 && frontR > 1000 && backL > 500 && backR > 500, `${frontL} ${frontR} ${backL} ${backR}`);

  // A second, black frame with the same pose averages in at equal weight: every observed colour
  // halves. finalize leaves the accumulator as it was.
  const one = acc.finalize(0);
  assert.equal(one.filled, 0, "rings 0 fills nothing from neighbours");
  acc.add(body, new Uint8Array(rgb.length), mask, W, H, K, zb, sign);
  const two = acc.finalize(0);
  assert.equal(two.observed, one.observed);
  let halved = 0;
  for (let v = 0; v < V; v++) {
    if (one.colors[v * 3 + 1] !== 0) continue;
    let ok = true;
    for (let c = 0; c < 3; c += 2) if (Math.abs(two.colors[v * 3 + c] - one.colors[v * 3 + c] / 2) > 1) ok = false;
    if (ok) halved++;
  }
  assert.equal(halved, V, "every colour halves (unobserved ones take the halved mean)");

  assert.throws(() => acc.add(body.subarray(0, 9), rgb, mask, W, H, K, zb, sign), /body holds 9 floats/);
  assert.throws(() => new BodyColorAccumulator(3, new Uint32Array([0, 1, 5])), /face index 5/);
});

test("bakeBackAtlas: corner uvs land on the vertex colours, each edge midpoint on its edge mean", () => {
  const colors = new Uint8Array([
    255, 0, 0, // 0
    0, 255, 0, // 1
    0, 0, 255, // 2
    100, 100, 100, // 3
  ]);
  const faces = new Uint32Array([0, 1, 2, 1, 3, 2, 2, 3, 0, 0, 3, 1, 3, 2, 1]);
  const width = 9; // 3 patches per row -> 2 rows -> 6 px -> height 8
  const { height, pixels, uvs } = bakeBackAtlas(faces, colors, width);
  assert.equal(height, 8);
  assert.equal(pixels.length, width * height * 3);
  const at = (x, y) => [0, 1, 2].map((k) => pixels[(y * width + x) * 3 + k]);
  for (let t = 0; t < faces.length / 3; t++) {
    const px0 = (t % 3) * 3, py0 = Math.floor(t / 3) * 3;
    for (let j = 0; j < 3; j++) {
      const u = uvs[t * 6 + j * 2], v = uvs[t * 6 + j * 2 + 1];
      assert.equal(u % 1, 0.5); assert.equal(v % 1, 0.5);
      const vi = faces[t * 3 + j];
      assert.deepEqual(at(Math.floor(u), Math.floor(v)), [...colors.subarray(vi * 3, vi * 3 + 3)], `face ${t} corner ${j}`);
      assert.ok(u > px0 && u < px0 + 3 && v > py0 && v < py0 + 3, "uv inside the face's own patch");
    }
    const c = [0, 1, 2].map((j) => colors.subarray(faces[t * 3 + j] * 3, faces[t * 3 + j] * 3 + 3));
    // The texel under each edge's uv midpoint holds that edge's mean: both faces sharing the edge
    // read the same colour there.
    const mid = (j0, j1) => at(Math.floor((uvs[t * 6 + j0 * 2] + uvs[t * 6 + j1 * 2]) / 2), Math.floor((uvs[t * 6 + j0 * 2 + 1] + uvs[t * 6 + j1 * 2 + 1]) / 2));
    const mean2 = (p, q) => [0, 1, 2].map((k) => Math.round((c[p][k] + c[q][k]) / 2));
    assert.deepEqual(mid(0, 1), mean2(0, 1), "ab edge");
    assert.deepEqual(mid(0, 2), mean2(0, 2), "ac edge");
    assert.deepEqual(mid(1, 2), mean2(1, 2), "bc edge");
    assert.deepEqual(at(px0 + 1, py0 + 1), mean2(1, 2), "bc midpoint texel");
    const avg = [0, 1, 2].map((k) => Math.round((c[0][k] + c[1][k] + c[2][k]) / 3));
    assert.deepEqual(at(px0 + 2, py0 + 1), avg, "outside the triangle: face mean");
    assert.deepEqual(at(px0 + 1, py0 + 2), avg);
    assert.deepEqual(at(px0 + 2, py0 + 2), avg);
  }
  // Unused texels: the mean vertex colour.
  assert.deepEqual(at(8, 7), [89, 89, 89]);

  // The MHR topology at 2048: 682 patches per row, 55 rows, height 168; every corner on its colour.
  const { faces: mf, V } = mhr();
  const mc = new Uint8Array(V * 3);
  for (let i = 0; i < mc.length; i++) mc[i] = (i * 37) % 251;
  const big = bakeBackAtlas(mf, mc, 2048);
  assert.equal(big.height, 168);
  for (let t = 0; t < mf.length / 3; t++) {
    for (let j = 0; j < 3; j++) {
      const x = Math.floor(big.uvs[t * 6 + j * 2]), y = Math.floor(big.uvs[t * 6 + j * 2 + 1]), vi = mf[t * 3 + j];
      for (let k = 0; k < 3; k++) if (big.pixels[(y * 2048 + x) * 3 + k] !== mc[vi * 3 + k]) assert.fail(`face ${t} corner ${j}`);
    }
  }
  assert.throws(() => bakeBackAtlas(faces, colors, 2), /width must be an integer >= 3, got 2/);
});

test("toAresSpace: (x, -y, -z), a copy", () => {
  const p = new Float32Array([1, 2, 3, -4, 5, 6]);
  const q = toAresSpace(p);
  assert.deepEqual([...q], [1, -2, -3, -4, -5, -6]);
  assert.deepEqual([...p], [1, 2, 3, -4, 5, 6]);
  assert.throws(() => toAresSpace(new Float32Array(4)), /4 floats/);
});
