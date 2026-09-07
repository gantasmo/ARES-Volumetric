import { test } from "node:test";
import assert from "node:assert/strict";
import { FX_DEFAULTS, packFx, mergeFx, evalFxTrack, isFxIdentity, valueNoise3 } from "../dist/index.js";

test("packFx lays the block out as the shaders read it", () => {
  const p = mergeFx(FX_DEFAULTS, { clipOn: true, clipNormal: [0, 0, 1], clipOffset: -2, dissolve: 0.3, dissolveScale: 8, scanlines: 0.5, tint: [1, 0.5, 0.25], tintMix: 0.7, rim: 0.4, rimColor: [0.1, 0.2, 0.3], dissolveEdge: [0.9, 0.8, 0.7], wobbleAmp: 0.02, wobbleFreq: 3, scanFreq: 50, splatJitter: 0.01, splatScale: 1.5, splatOpacity: 0.6 });
  const d = packFx(p, 12.5, [1, 2, 3]);
  assert.deepEqual(Array.from(d.subarray(0, 4)), [0, 0, 1, -2]);
  assert.deepEqual(Array.from(d.subarray(4, 8)).map((v) => +v.toFixed(3)), [1, 0.3, 8, 0.5]);
  assert.deepEqual(Array.from(d.subarray(8, 12)).map((v) => +v.toFixed(3)), [1, 0.5, 0.25, 0.7]);
  assert.deepEqual(Array.from(d.subarray(12, 16)).map((v) => +v.toFixed(3)), [0.1, 0.2, 0.3, 0.4]);
  assert.deepEqual(Array.from(d.subarray(16, 20)).map((v) => +v.toFixed(3)), [0.9, 0.8, 0.7, 0.02]);
  assert.deepEqual(Array.from(d.subarray(20, 24)).map((v) => +v.toFixed(3)), [3, 12.5, 50, 0.01]);
  assert.deepEqual(Array.from(d.subarray(24, 28)).map((v) => +v.toFixed(3)), [1, 2, 3, 1.5]);
  assert.equal(+d[28].toFixed(3), 0.6);
  assert.equal(packFx(mergeFx(FX_DEFAULTS, { time: 3 }), 99, [0, 0, 0])[21], 3, "a pinned time wins over the clock");
  assert.ok(isFxIdentity(FX_DEFAULTS));
  assert.ok(!isFxIdentity(p));
});

test("evalFxTrack interpolates numbers and colours, holds outside, snaps booleans", () => {
  const track = { keyframes: [
    { frame: 20, params: { dissolve: 1, tint: [1, 0, 0], clipOn: true } },
    { frame: 10, params: { dissolve: 0, tint: [0, 0, 1], clipOn: false } },
  ] };
  assert.equal(evalFxTrack(track, 0).dissolve, 0);
  assert.equal(evalFxTrack(track, 30).dissolve, 1);
  const mid = evalFxTrack(track, 15);
  assert.ok(Math.abs(mid.dissolve - 0.5) < 1e-9);
  assert.deepEqual(mid.tint.map((v) => +v.toFixed(3)), [0.5, 0, 0.5]);
  assert.equal(mid.clipOn, false);
  assert.equal(evalFxTrack(track, 20).clipOn, true);
  assert.equal(evalFxTrack(null, 5).scanFreq, FX_DEFAULTS.scanFreq);
  const base = mergeFx(FX_DEFAULTS, { rim: 0.9 });
  assert.equal(evalFxTrack(track, 15, base).rim, 0.9, "untouched params come from the base");
});

test("value noise stays in [0,1] and is continuous", () => {
  let prev = valueNoise3(0.3, 0.2, 0.1);
  for (let i = 1; i <= 200; i++) {
    const v = valueNoise3(0.3 + i * 0.01, 0.2, 0.1);
    assert.ok(v >= 0 && v <= 1);
    assert.ok(Math.abs(v - prev) < 0.08, `jump ${Math.abs(v - prev)} at ${i}`);
    prev = v;
  }
});
