import { test } from "node:test";
import assert from "node:assert/strict";
import { AresPlayer, frameToClockUs, clockUsToFrame, reliefFromMeta } from "../dist/index.js";

// AresPlayer is NOT constructed here and must not be: create() needs a canvas, navigator.gpu and
// requestAnimationFrame (player.ts), none of which exist under `node --test`. Importing the class is
// safe — the module touches no DOM at load time — so the surface is asserted on the prototype, and
// the only arithmetic worth testing was lifted out into two pure functions that need no player.

test("the frame-indexed capture kit is on the prototype with the documented arity", () => {
  const p = AresPlayer.prototype;
  // [method, arity]. Arity counts required parameters only: JS excludes defaulted ones from .length,
  // so captureFrameBlob(maxDim = 1008, ...) is 0 and seekFrame(idx) is 1.
  for (const [name, arity] of [["seekFrame", 1], ["getClipFps", 0], ["textureDebug", 0], ["captureFrameBlob", 0], ["frameGeomQ", 0]]) {
    assert.equal(typeof p[name], "function", `${name} is missing from AresPlayer.prototype`);
    assert.equal(p[name].length, arity, `${name} arity`);
  }
  // The methods these are additive to are still here — this step changes no existing behavior.
  for (const name of ["seek", "captureFrame", "exportFrame", "pickRaster", "getCamera", "getViewAspect", "setEditPreview", "getStats"]) {
    assert.equal(typeof p[name], "function", `${name} disappeared`);
  }
});

test("frame ↔ clock round-trips exactly at any clip fps", () => {
  for (const fps of [12, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120]) {
    for (const f of [0, 1, 2, 39, 40, 41, 137, 271, 272, 999, 4321]) {
      assert.equal(clockUsToFrame(frameToClockUs(f, fps), fps), f, `fps ${fps} frame ${f}`);
    }
  }
  // The regression the epsilon in clockUsToFrame exists for: 40/30 s is 39.999999999999986 frames in
  // binary floating point, so a bare floor lands on 39 — an off-by-one on about half of all frames.
  assert.equal(clockUsToFrame((40 / 30) * 1e6, 30), 40);
  assert.equal(Math.floor(((40 / 30) * 1e6) / (1e6 / 30)), 39, "the unguarded arithmetic really is wrong");
  // seekFrame(n) is seek(n / fps) by construction: same clock, same index, no float round-trip.
  for (const fps of [24, 29.97, 30, 60]) {
    for (const f of [0, 7, 143, 271]) assert.equal(clockUsToFrame((f / fps) * 1e6, fps), clockUsToFrame(frameToClockUs(f, fps), fps));
  }
});

test("frameToClockUs clamps and rounds its index rather than emitting a negative clock", () => {
  assert.equal(frameToClockUs(-5, 30), 0);
  assert.equal(frameToClockUs(0, 30), 0);
  assert.equal(frameToClockUs(2.4, 30), frameToClockUs(2, 30));
  assert.equal(frameToClockUs(2.6, 30), frameToClockUs(3, 30));
});

test("reliefFromMeta: a relief needs its capture camera and axis; the discard and framing fields default to 0", () => {
  assert.equal(reliefFromMeta(null), null);
  assert.equal(reliefFromMeta({ title: "capture" }), null, "a capture is not a relief");
  assert.equal(reliefFromMeta({ "relief.camera": "0,1", "relief.forward": "0,0,-1" }), null, "a camera needs three numbers");
  const culled = reliefFromMeta({ "relief.camera": "-2.2,0.6,3.25", "relief.forward": "0,0,-1", "relief.fov": "55", "relief.pivot": "1.06" });
  assert.deepEqual(culled, { camera: [-2.2, 0.6, 3.25], forward: [0, 0, -1], slope: 0, depthMax: 0, fov: 55, pivot: 1.06, near: 0, far: 0, aspect: 0 });
  const sheet = reliefFromMeta(new Map([["relief.camera", "0,0,0"], ["relief.forward", "0,0,-1"], ["relief.slope", "0.024"], ["relief.depthMax", "9"], ["relief.pivot", "nope"], ["relief.near", "0.7"], ["relief.far", "5.5"], ["relief.aspect", "2.359"]]));
  assert.equal(sheet.slope, 0.024);
  assert.equal(sheet.depthMax, 9);
  assert.equal(sheet.pivot, 0, "an unparseable value reads as unknown");
  assert.equal(sheet.near, 0.7);
  assert.equal(sheet.far, 5.5);
  assert.equal(sheet.aspect, 2.359);
});
