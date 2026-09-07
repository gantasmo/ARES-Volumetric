import { test } from "node:test";
import assert from "node:assert/strict";
import {
  packQuaternion, unpackQuaternion, encodeScaleByte, decodeScaleByte, encodeShByte, decodeShByte,
  shStrideBytes, shRestCoeffs, sortSplatsBackToFront, SplatSorter, decodeSplatBlock, AresParseError,
  ByteWriter, ByteReader, quantizePositions, dequantScale, lookAt,
} from "../dist/index.js";

const rnd = (seed) => { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); };

test("smallest-three quaternion pack/unpack keeps orientation within 0.2°", () => {
  const r = rnd(7);
  let worst = 1;
  for (let i = 0; i < 2000; i++) {
    let q = [r() * 2 - 1, r() * 2 - 1, r() * 2 - 1, r() * 2 - 1];
    const l = Math.hypot(...q); q = q.map((v) => v / l);
    const u = unpackQuaternion(packQuaternion(q[0], q[1], q[2], q[3]));
    const dot = Math.abs(q[0] * u[0] + q[1] * u[1] + q[2] * u[2] + q[3] * u[3]);
    worst = Math.min(worst, dot);
    assert.ok(Math.abs(Math.hypot(...u) - 1) < 1e-6, "unit length");
  }
  // |dot| = cos(θ/2); 0.2° → cos(0.1°) ≈ 0.9999985
  assert.ok(worst > 0.99999, `worst |dot| ${worst}`);
});

test("quaternion pack is sign-invariant (q and -q map to the same code)", () => {
  const a = packQuaternion(0.1, -0.7, 0.2, 0.67);
  const b = packQuaternion(-0.1, 0.7, -0.2, -0.67);
  assert.equal(a, b);
});

test("scale byte is log-encoded with ~3 % steps and clamps", () => {
  for (const s of [0.001, 0.013, 0.1, 1, 10]) {
    const d = decodeScaleByte(encodeScaleByte(s));
    assert.ok(Math.abs(Math.log(d / s)) <= 1 / 32 + 1e-9, `scale ${s} → ${d}`);
  }
  assert.equal(encodeScaleByte(1e-9), 0);
  assert.equal(encodeScaleByte(1e9), 255);
});

test("SH byte round trip", () => {
  for (const v of [-1, -0.5, 0, 0.25, 0.9921875]) assert.ok(Math.abs(decodeShByte(encodeShByte(v)) - v) <= 1 / 256 + 1e-9);
  assert.equal(shRestCoeffs(0), 0); assert.equal(shRestCoeffs(1), 3); assert.equal(shRestCoeffs(2), 8); assert.equal(shRestCoeffs(3), 15);
  assert.equal(shStrideBytes(1), 12); assert.equal(shStrideBytes(2), 24); assert.equal(shStrideBytes(3), 48);
});

test("back-to-front sort orders by view depth, far first", () => {
  const box = { min: [-1, -1, -1], max: [1, 1, 1] };
  const bits = 14, inv = dequantScale(bits);
  // Three splats along +z at z = -0.5, 0, 0.5; camera at z = +3 looking at the origin → far first = z=-0.5.
  const pos = new Float32Array([0, 0, -0.5, 0, 0, 0.5, 0, 0, 0]);
  const q = quantizePositions(pos, box, bits);
  const view = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
  const out = new Uint32Array(3);
  sortSplatsBackToFront(q, 3, box, inv, view, out);
  assert.deepEqual(Array.from(out), [0, 2, 1]);
  const sorter = new SplatSorter();
  const o1 = sorter.update(q, 3, box, inv, view, 0);
  assert.ok(o1 && Array.from(o1).join() === "0,2,1");
  assert.equal(sorter.update(q, 3, box, inv, view, 0), null, "same view + frame → no re-sort");
  const view2 = lookAt([0, 0, -3], [0, 0, 0], [0, 1, 0]);
  const o2 = sorter.update(q, 3, box, inv, view2, 0);
  assert.ok(o2 && Array.from(o2).join() === "1,2,0", "opposite side reverses the order");
});

test("decodeSplatBlock rejects absurd counts before allocating", () => {
  const w = new ByteWriter(64);
  w.u32(0xffffffff).u8(0).u8(0).u16(0).u32(0).u32(0);
  assert.throws(() => decodeSplatBlock(w.finish()), AresParseError);
  const w2 = new ByteWriter(64);
  w2.u32(1).u8(4).u8(0).u16(0);
  assert.throws(() => decodeSplatBlock(w2.finish()), /sh_degree/);
});

test("ByteWriter grows from a zero initial capacity", () => {
  const w = new ByteWriter(0);
  w.u32(7).str("hello");
  const r = new ByteReader(w.finish());
  assert.equal(r.u32(), 7);
  assert.equal(r.str(), "hello");
});
