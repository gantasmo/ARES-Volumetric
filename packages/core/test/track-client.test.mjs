import { test } from "node:test";
import assert from "node:assert/strict";
import { keepPredicateAt, validateMasks, rleEncodeMask } from "../dist/index.js";
import {
  createTrackWriter, createTrackClient, packFrameBatch, rescalePoints, coalesceGaps, preflight,
  maskIsSolid, TRACK_RES, MASK_RES_DEFAULT, COMMIT_EVERY,
} from "../../../apps/demo/track.js";

// apps/demo/track.js is the SAM propagation engine (docs/sam-propagation-plan.md steps 5-6). It is
// imported here rather than exercised in a browser because it was written with no DOM at module
// scope precisely so its gap policy — the part that silently destroys a user's frame when it is
// wrong — is assertable headlessly. Nothing in this file constructs a player, a canvas or a fetch.

const CAM = { azimuth: 0.4, elevation: 0.05, distance: 2, target: [0, 0, 0], aspect: 16 / 9 };
const W = 16, H = 9;

/** A distinct 16x9 bitmap per frame: one lit pixel at column (f % 16). Distinct so a keyframe can
 *  be traced back to the frame it was emitted for. */
function maskFor(f) {
  const bits = new Uint8Array(W * H);
  bits[(f % H) * W + (f % W)] = 1;
  return { width: W, height: H, rle: rleEncodeMask(bits) };
}
const maskEvent = (frame, objId = 1, score = 0.9) => ({ frame, objId, w: W, h: H, rle: maskFor(frame).rle, score });

function newWriter(mode, { seedFrame = 0, ...extra } = {}) {
  const doc = { aresEdits: 1, ranges: [] };
  let ids = 0, commits = 0;
  const writer = createTrackWriter({
    doc,
    newRangeId: () => "r" + ++ids,
    nextColor: () => "#6f8dc0",
    onCommit: () => { commits++; },
    ...extra,
  });
  writer.begin({
    seedFrame, from: 0, to: 39, mode, action: "delete", color: "#6f8dc0", camera: CAM,
    proxy: { width: TRACK_RES, height: 567, shade: "shaded", grid: false },
    seedMask: maskFor(seedFrame), objIds: [1], checkpoint: "C:/sam3", dtype: "float16",
    trackRes: TRACK_RES, maskRes: MASK_RES_DEFAULT, trackId: "trk-test",
  });
  return { doc, writer, commits: () => commits };
}

test("packFrameBatch emits [u32le frameIdx][u32le byteLen][jpeg] with no envelope", () => {
  const a = new Uint8Array([1, 2, 3]), b = new Uint8Array([9, 9]);
  const buf = packFrameBatch([{ frame: 7, bytes: a }, { frame: 8, bytes: b }]);
  assert.equal(buf.length, 8 + 3 + 8 + 2);
  const dv = new DataView(buf.buffer);
  assert.equal(dv.getUint32(0, true), 7);
  assert.equal(dv.getUint32(4, true), 3);
  assert.deepEqual([...buf.slice(8, 11)], [1, 2, 3]);
  assert.equal(dv.getUint32(11, true), 8);
  assert.equal(dv.getUint32(15, true), 2);
  assert.deepEqual([...buf.slice(19, 21)], [9, 9]);
});

test("rescalePoints scales x and y independently (captureFrame caps the LONG side)", () => {
  // PORTRAIT capture: captureFrame's maxDim caps the long side, which here is the HEIGHT, so the
  // plan's original "scale by trackRes / capture.width" puts every prompt point off the subject.
  const cap = { width: 576, height: 1024 }, proxy = { width: 567, height: TRACK_RES };
  const [[x, y]] = rescalePoints([[100, 200]], cap, proxy);
  assert.equal(x, 100 * (567 / 576));
  assert.equal(y, 200 * (TRACK_RES / 1024));
  assert.notEqual(y, 200 * (TRACK_RES / cap.width), "width-only scaling is 1.78x off in y");

  // And when the viewport itself changed between the click capture and the sweep, the two factors
  // genuinely differ — the case a single shared factor cannot express at all.
  const [[x2, y2]] = rescalePoints([[100, 200]], cap, { width: TRACK_RES, height: 567 });
  assert.equal(x2, 100 * (TRACK_RES / 576));
  assert.equal(y2, 200 * (567 / 1024));
});

test("coalesceGaps returns inclusive pairs", () => {
  assert.deepEqual(coalesceGaps([144, 143, 145, 147, 200]), [[143, 145], [147, 147], [200, 200]]);
});

test("preflight refuses in priority order with the shipped strings", () => {
  const ok = { ok: true, trackReady: true };
  assert.equal(preflight({ health: ok, splat: true }), "splat clip · tracking needs mesh triangles");
  assert.equal(preflight({ health: ok, xfDefault: false }), "model transform active · reset it before tracking");
  assert.equal(preflight({ health: ok, fxIdentity: false }), "fx active · reset fx before tracking");
  assert.equal(preflight({ health: null }), "SAM service not running");
  assert.equal(preflight({ health: { ok: true, trackLoading: true } }), "track model loading · retry shortly");
  assert.equal(preflight({ health: ok }), null);
});

test("delete mode: a gap splits the track into one range per contiguous run, and the gap frames survive", () => {
  const { doc, writer } = newWriter("delete");
  for (let f = 0; f <= 20; f++) {
    if (f >= 10 && f <= 12) writer.gap({ frame: f, objId: 1, reason: "occluded" });
    else writer.mask(maskEvent(f));
  }
  writer.end();

  assert.equal(doc.ranges.length, 2, "one range per contiguous run");
  const [a, b] = doc.ranges;
  assert.equal(a.track.trackId, b.track.trackId, "runs of one track share a trackId");
  assert.deepEqual([a.track.run, b.track.run], [0, 1]);
  assert.deepEqual([a.startFrame, a.endFrame], [0, 9]);
  assert.deepEqual([b.startFrame, b.endFrame], [13, 20]);
  assert.deepEqual(a.track.gaps, [[10, 12]]);
  for (const r of doc.ranges) assert.ok(r.keyframes.every((k, i) => i === 0 || k.frame > r.keyframes[i - 1].frame));

  // The regression this policy exists for: with no range spanning the gap, keepPredicateAt returns
  // null (edits.ts:581) and the frame survives whole rather than being filtered by an empty mask.
  assert.equal(keepPredicateAt(doc, 11), null);
  assert.notEqual(keepPredicateAt(doc, 9), null);
  assert.deepEqual(validateMasks(doc), []);
});

test("keep mode: ONE range spans the gap and every gap frame carries the last non-empty mask at conf 0", () => {
  const { doc, writer } = newWriter("keep");
  for (let f = 0; f <= 20; f++) {
    if (f >= 10 && f <= 12) writer.gap({ frame: f, objId: 1, reason: "empty" });
    else writer.mask(maskEvent(f));
  }
  writer.end();

  assert.equal(doc.ranges.length, 1, "isolate never splits: a gap frame with no keep range shows the whole scene");
  const r = doc.ranges[0];
  assert.deepEqual([r.startFrame, r.endFrame], [0, 20]);
  assert.equal(r.keyframes.length, 21, "one keyframe per frame, gaps included");
  const last = maskFor(9).rle;
  for (const f of [10, 11, 12]) {
    const k = r.keyframes.find((x) => x.frame === f);
    assert.equal(k.conf, 0, "a filled gap is flagged for the drift caret");
    assert.deepEqual(k.volumes[0].mask.rle, last, "gap frames carry the last non-empty mask");
  }
  // NEVER an empty keep mask: keepPredicateAt drops every point outside all active keep regions,
  // so one all-zero keep keyframe deletes the entire frame.
  for (const k of r.keyframes) {
    const rle = k.volumes[0].mask.rle;
    let ones = 0;
    for (let i = 1; i < rle.length; i += 2) ones += rle[i];
    assert.ok(ones > 0, `keep keyframe at frame ${k.frame} is empty`);
  }
  assert.deepEqual(validateMasks(doc), []);
});

test("the seed frame is the one user keyframe; every other frame is derived and carries the run's conf", () => {
  const { doc, writer } = newWriter("delete");
  for (let f = 0; f <= 39; f++) writer.mask(maskEvent(f, 1, 0.87));
  const stats = writer.end();

  assert.equal(doc.ranges.length, 1);
  const kfs = doc.ranges[0].keyframes;
  assert.equal(kfs.length, 40);
  assert.equal(stats.user, 1);
  assert.equal(kfs.filter((k) => !k.derived).length, 1);
  assert.equal(kfs.find((k) => !k.derived).frame, 0);
  assert.ok(kfs[0].conf === undefined, "a user keyframe carries no propagation confidence");
  assert.equal(kfs[5].conf, 0.87);
  assert.equal(new Set(kfs.map((k) => k.frame)).size, 40, "no duplicate frames");
  for (const k of kfs) {
    assert.equal(k.volumes.length, 1);
    assert.equal(k.volumes[0].type, "mask2d");
    assert.equal(k.volumes[0].kind, "bitmap");
    assert.equal(k.volumes[0].camera, CAM, "every derived volume shares the frozen camera object");
    assert.ok(k.volumes[0].depth === undefined, "no depth band is carried forward from the seed frame");
  }
  assert.deepEqual(validateMasks(doc), []);
});

test("a bidirectional run's reverse leg re-emits the seed frame and must not overwrite its user keyframe", () => {
  const { doc, writer } = newWriter("delete", { seedFrame: 20 });
  for (let f = 20; f <= 25; f++) writer.mask(maskEvent(f));      // forward leg, inclusive of the seed
  for (let f = 20; f >= 16; f--) writer.mask(maskEvent(f));      // reverse leg, also inclusive of it
  const seed = doc.ranges[0].keyframes.find((k) => k.frame === 20);
  assert.ok(!seed.derived, "the re-emitted seed frame stays a user keyframe");
  assert.equal(doc.ranges.length, 1, "the two legs meet at the seed and stay one range");
  assert.deepEqual([doc.ranges[0].startFrame, doc.ranges[0].endFrame], [16, 25]);
});

test("a frame that bridges two runs merges them instead of leaving a phantom occlusion", () => {
  const { doc, writer } = newWriter("delete");
  writer.mask(maskEvent(0));
  writer.mask(maskEvent(1));
  writer.mask(maskEvent(4));
  writer.mask(maskEvent(3));
  assert.equal(doc.ranges.length, 2);
  writer.mask(maskEvent(2));                     // bridges [0,1] and [3,4]
  assert.equal(doc.ranges.length, 1);
  const r = doc.ranges[0];
  assert.deepEqual([r.startFrame, r.endFrame], [0, 4]);
  assert.deepEqual(r.keyframes.map((k) => k.frame), [0, 1, 2, 3, 4]);
});

test("a correction replaces only derived keyframes inside [frame, nextUser - 1] and promotes the frame", () => {
  const { doc, writer } = newWriter("delete");
  for (let f = 0; f <= 39; f++) writer.mask(maskEvent(f));
  // A prior correction at 30 left a user keyframe there: it is the window's upper bound.
  const k30 = doc.ranges[0].keyframes.find((k) => k.frame === 30);
  delete k30.derived; delete k30.conf;

  assert.equal(writer.nextUserFrame(1, 20), 30);
  const win = writer.beginCorrection({ objId: 1, frame: 20, until: 30 });
  assert.deepEqual(win, { from: 20, to: 29 });

  const kfs = doc.ranges[0].keyframes;
  assert.deepEqual(kfs.filter((k) => k.frame > 20 && k.frame < 30), [], "the window is cleared");
  assert.ok(kfs.some((k) => k.frame === 30 && !k.derived), "the bounding user keyframe survives");
  assert.ok(kfs.some((k) => k.frame === 31 && k.derived), "nothing past the bound is touched");
  const at20 = kfs.find((k) => k.frame === 20);
  assert.ok(!at20.derived && at20.conf === undefined, "the corrected frame is promoted to a user keyframe");

  writer.seed({ frame: 20, objId: 1, kind: "points", mode: "nudge", points: [[10, 20]], labels: [1] });
  for (let f = 21; f <= 29; f++) writer.mask(maskEvent(f));
  writer.endCorrection();
  writer.end();
  assert.equal(doc.ranges[0].keyframes.length, 40, "the window refilled");
  assert.equal(doc.ranges[0].track.seeds.length, 1, "exactly one seed appended");
  assert.deepEqual(validateMasks(doc), []);
});

test("keyframes land as they stream: a commit tick fires every COMMIT_EVERY keyframes", () => {
  const { writer, commits } = newWriter("delete");
  for (let f = 0; f < COMMIT_EVERY * 2; f++) writer.mask(maskEvent(f));
  assert.equal(commits(), 2, "a tab reload loses at most one batch, not the whole track");
  writer.end();
  assert.equal(commits(), 3, "plus the final flush");
});

// ---------------------------------------------------------------- corrections ----
// The correction loop is where the writer can make a track WORSE than leaving it alone, so these
// pin the three ways it did: a re-seed that authored a second overlapping range the bake unions
// with the drift, a nudge that froze the rejected mask as a user keyframe, and an all-zero mask
// reaching a keep keyframe.

test("re-seed corrects the track in place: no second range, and the retired object's masks are dropped", () => {
  const { doc, writer } = newWriter("delete");
  for (let f = 0; f <= 11; f++) writer.mask(maskEvent(f, 1, 0.30));       // a drifted pass
  assert.equal(doc.ranges.length, 1);

  const win = writer.beginCorrection({
    objId: 2, fromObjId: 1, frame: 6, until: writer.nextUserFrame(1, 6),
    volumes: [{ type: "mask2d", kind: "bitmap", mask: maskFor(31), camera: CAM }],
  });
  assert.deepEqual(win, { from: 6, to: 39 });
  assert.deepEqual(doc.ranges[0].keyframes.map((k) => k.frame), [0, 1, 2, 3, 4, 5, 6],
    "the drifted window is cleared on the object that OWNS it, not on the fresh id");

  // The retired object cannot be removed from the inference session (track.py:778), so the re-run
  // emits one event per object per frame: obj 2's correction AND obj 1's drift.
  for (let f = 6; f <= 11; f++) {
    writer.mask(maskEvent(f, 2, 0.95));
    if (f > 6) writer.mask(maskEvent(f, 1, 0.30));
  }
  writer.endCorrection();
  writer.end();

  assert.equal(doc.ranges.length, 1, "no second range over the same frames for keepPredicateAt to union");
  const r = doc.ranges[0];
  assert.equal(r.track.objId, 2, "provenance names the object the tracker answers for from here on");
  assert.equal(r.keyframes.length, 12);
  assert.deepEqual(r.keyframes.filter((k) => !k.derived).map((k) => k.frame), [0, 6],
    "the seed and the corrected frame, and nothing else");
  for (let f = 7; f <= 11; f++)
    assert.equal(r.keyframes.find((k) => k.frame === f).conf, 0.95, `frame ${f} kept the drift`);
  assert.deepEqual(validateMasks(doc), []);
});

test("a nudge promotes the corrected frame to the correction's OWN mask, not the one it rejected", () => {
  const { doc, writer } = newWriter("delete");
  for (let f = 0; f <= 11; f++) writer.mask(maskEvent(f, 1, 0.30));
  const drifted = doc.ranges[0].keyframes.find((k) => k.frame === 6).volumes[0];
  const fix = { type: "mask2d", kind: "bitmap", mask: maskFor(31), camera: CAM };

  writer.beginCorrection({ objId: 1, frame: 6, until: Infinity, volumes: [fix] });
  // The prompt is non-conditioning on an already-tracked frame, so the re-run starts at frame + 1
  // and NEVER re-emits frame 6: whatever is there when the window is promoted is what ships.
  for (let f = 7; f <= 11; f++) writer.mask(maskEvent(f, 1, 0.95));
  writer.endCorrection();
  writer.end();

  const k6 = doc.ranges[0].keyframes.find((k) => k.frame === 6);
  assert.ok(!k6.derived && k6.conf === undefined, "the corrected frame is a user keyframe");
  assert.equal(k6.volumes[0], fix, "the frame the user clicked carries the mask they clicked");
  assert.notEqual(k6.volumes[0], drifted, "a user keyframe is unfixable, so it must not be the drift");
});

test("an all-zero mask never reaches a keep keyframe: presence is not solidity", () => {
  const doc = { aresEdits: 1, ranges: [] };
  let ids = 0;
  const writer = createTrackWriter({ doc, newRangeId: () => "r" + ++ids });
  const empty = { width: W, height: H, rle: rleEncodeMask(new Uint8Array(W * H)) };
  assert.equal(empty.rle.length, 1, "an all-zero bitmap is ONE run — and the object is still truthy");
  assert.equal(maskIsSolid(empty), false);
  assert.deepEqual(validateMasks({
    aresEdits: 1,
    ranges: [{ id: "r1", mode: "keep", startFrame: 0, endFrame: 0,
      keyframes: [{ frame: 0, volumes: [{ type: "mask2d", kind: "bitmap", mask: empty, camera: { ...CAM, aspect: W / H } }] }] }],
  }), [], "and it validates CLEAN, which is why the writer has to be the gate");

  writer.begin({
    seedFrame: 0, from: 0, to: 20, mode: "keep", action: "delete", camera: CAM,
    proxy: { width: TRACK_RES, height: 567 }, seedMask: empty, objIds: [1],
  });
  assert.equal(writer.gap({ frame: 0, objId: 1, reason: "occluded" }), null);
  writer.end();
  assert.equal(doc.ranges.length, 0, "no keep range at all");
  assert.equal(keepPredicateAt(doc, 0), null, "so frame 0 survives whole instead of baking empty");
});

test("a resume ADOPTS the ranges already in the sidecar instead of rebuilding beside them", () => {
  // The forward leg of a bidirectional track, already flushed. track_run clears the event log and
  // bumps run_id per leg (track.py:975-976), so the offered run is the reverse leg alone and these
  // frames exist nowhere but the sidecar.
  const { doc, writer } = newWriter("delete", { seedFrame: 6 });
  for (let f = 6; f <= 11; f++) writer.mask(maskEvent(f));
  writer.end();
  assert.equal(doc.ranges.length, 1);

  let ids = 100;
  const replay = createTrackWriter({ doc, newRangeId: () => "r" + ++ids });
  replay.begin({
    seedFrame: 6, from: 0, to: 11, mode: "delete", action: "delete", camera: CAM,
    proxy: { width: TRACK_RES, height: 567 }, seedMask: null, objIds: [1], trackId: "trk-test",
  });
  replay.adopt(doc.ranges.slice());
  for (let f = 6; f >= 2; f--) replay.mask(maskEvent(f));      // the reverse leg, inclusive of the seed
  replay.end();

  assert.equal(doc.ranges.length, 1, "one range, not a second one beside the forward leg");
  assert.deepEqual(doc.ranges[0].keyframes.map((k) => k.frame), [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "the forward leg's frames survive the replay of the reverse leg");
  assert.ok(!doc.ranges[0].keyframes.find((k) => k.frame === 6).derived, "the seed stays a user keyframe");
  assert.deepEqual(validateMasks(doc), []);
});

// --------------------------------------------------------------- the transport ----
// The client half runs headlessly too: the player, fetch and EventSource all arrive as arguments.
// EventSource.close() is modelled to the letter of the HTML spec — it aborts the connection and
// dispatches NOTHING — because that is precisely what wedged the cancel path.

class StubES {
  constructor(url) { this.url = url; this.ls = {}; this.closed = false; this.onerror = null; StubES.last = this; }
  addEventListener(name, fn) { (this.ls[name] = this.ls[name] || []).push(fn); }
  emit(name, data) { if (!this.closed) for (const fn of this.ls[name] || []) fn({ data: JSON.stringify(data) }); }
  close() { this.closed = true; }
}

function stubClient({ frames = 6 } = {}) {
  const doc = { aresEdits: 1, ranges: [] };
  let ids = 0;
  const writer = createTrackWriter({ doc, newRangeId: () => "r" + ++ids });
  const player = {
    autoOrbit: true, isPlaying: true, grid: true, preview: "live", frame: 0,
    seekFrame(f) { this.frame = f; },
    getStats() { return { frameIndex: this.frame }; },
    textureDebug: () => ({ settled: true }),
    // 4:3, deliberately NOT the selection's 16:9: the reviewer's case is a window resized between
    // the SAM click and K, which camKeyOf does not hash and therefore does not invalidate.
    captureFrameBlob: async () => ({ bytes: new Uint8Array([0xff, 0xd8, 0xff]), width: TRACK_RES, height: 756 }),
    isGrid() { return this.grid; },
    setGrid(v) { this.grid = v; },
    setEditPreview(v) { this.preview = v; },
    pause() { this.isPlaying = false; },
  };
  const ok = (o) => ({ ok: true, json: async () => o });
  const fetchImpl = async (url) => {
    if (url.startsWith("/sam/track/open"))
      return ok({ session: "trk_1", checkpoint: "sam3", dtype: "float16", trackRes: TRACK_RES, maskRes: MASK_RES_DEFAULT, reusedFrames: 0 });
    if (url.startsWith("/sam/track/prompt")) return ok({ objIds: [1], conditioning: true });
    return ok({ ok: true });
  };
  const client = createTrackClient({
    player, writer, doc, clipBase: "", fetch: fetchImpl, EventSource: StubES,
    raf: (fn) => setTimeout(fn, 0), camKey: () => "cam-0",
  });
  const run = () => client.run({
    clip: "c", camKey: "cam-0", frames, from: 0, to: frames - 1, seedFrame: 0,
    camera: CAM, capture: { width: 1024, height: 576 },
    objects: [{ objId: 1, points: [[10, 20]], labels: [1] }],
  });
  return { doc, writer, player, client, run };
}

/** Resolve once the run has opened its SSE stream, or fail rather than hang the suite. */
async function awaitStream(ms = 2000) {
  StubES.last = null;
  const t0 = Date.now();
  while (!StubES.last) {
    if (Date.now() - t0 > ms) throw new Error("the stream never opened");
    await new Promise((r) => setTimeout(r, 5));
  }
  return StubES.last;
}
const bounded = (p, ms = 2000) => Promise.race([
  p, new Promise((_, rej) => setTimeout(() => rej(new Error("the run never settled")), ms)),
]);

test("cancel settles the in-flight stream, so the freeze is always undone", async () => {
  const h = stubClient();
  const pending = h.run();
  const es = await awaitStream();
  es.emit("start", { run: 1, dir: "forward", start: 0, max: 6, objIds: [1] });
  es.emit("mask", { frame: 0, objId: 1, w: W, h: H, rle: maskFor(0).rle, score: 0.9 });
  es.emit("mask", { frame: 1, objId: 1, w: W, h: H, rle: maskFor(1).rle, score: 0.9 });

  // close() dispatches no event of any kind and the service's cooperative cancel only reaches its
  // next frame ~304 ms later, so without a settle handle this await never returns.
  await h.client.cancel();
  const out = await bounded(pending);

  assert.equal(out.cancelled, true);
  assert.equal(h.player.autoOrbit, true, "autoOrbit restored");
  assert.equal(h.player.grid, true, "grid restored");
  assert.equal(h.player.isPlaying, false, "playback stays paused: the sweep left the playhead elsewhere");
  assert.equal(h.doc.ranges.length, 1, "every mask already emitted is already a keyframe");
  assert.ok(h.doc.ranges[0].track.seeds.length === 1, "writer.end() ran: the seeds are per-range copies");
});

test("a stream that fails still finishes the track's provenance", async () => {
  const h = stubClient();
  const pending = h.run();
  const es = await awaitStream();
  es.emit("start", { run: 1, dir: "forward", start: 0, max: 6, objIds: [1] });
  for (const f of [0, 1]) es.emit("mask", { frame: f, objId: 1, w: W, h: H, rle: maskFor(f).rle, score: 0.9 });
  es.emit("gap", { frame: 2, objId: 1, reason: "occluded" });
  es.emit("mask", { frame: 3, objId: 1, w: W, h: H, rle: maskFor(3).rle, score: 0.9 });
  // The 30 s idle reaper strands the run: the browser sees a bare transport error.
  es.onerror();

  await assert.rejects(bounded(pending));
  assert.equal(h.player.autoOrbit, true, "the freeze is undone on the failure path too");
  const gaps = h.doc.ranges.map((r) => r.track.gaps).filter(Boolean);
  assert.deepEqual(gaps[0], [[2, 2]], "the occlusion is recorded, not left indistinguishable from a truncation");
  for (const r of h.doc.ranges) assert.ok(Array.isArray(r.track.seeds), "per-range seed copies, which undo needs");
});

test("the sweep stores the aspect of the PROXY it captured, not the selection's", async () => {
  const h = stubClient();
  const pending = h.run();          // capture 1024x576 = 1.7778, proxy 1008x756 = 1.3333
  const es = await awaitStream();
  es.emit("start", { run: 1, dir: "forward", start: 0, max: 6, objIds: [1] });
  es.emit("mask", { frame: 0, objId: 1, w: W, h: H, rle: maskFor(0).rle, score: 0.9 });
  es.emit("done", { frames: 1, of: 6, ms: 300, gaps: 0, cancelled: false });
  await bounded(pending);
  assert.equal(h.doc.ranges[0].keyframes[0].volumes[0].camera.aspect, TRACK_RES / 756);
  assert.notEqual(h.doc.ranges[0].keyframes[0].volumes[0].camera.aspect, CAM.aspect);
});
