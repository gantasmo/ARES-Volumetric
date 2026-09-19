/**
 * SAM video-tracker propagation — the browser half (docs/sam-propagation-plan.md steps 5 and 6).
 *
 * A selection made on one frame is swept through the clip under a FROZEN camera: every frame is
 * rendered to a JPEG proxy, streamed to the local service, and the tracker's per-frame mask comes
 * back as one `derived: true` mask2d keyframe. Nothing new reaches the bake — a propagated track is
 * the same {type:"mask2d", kind:"bitmap", mask, camera} volume the click, lasso and marquee already
 * author, so keepPredicateAt, filterFrame, recolor and paint inherit a time-varying region with
 * zero encoder changes.
 *
 * PURE MODULE, and deliberately so: no DOM at module scope, no import of main.js, no window
 * globals. Every dependency (the player handle, fetch, EventSource, rAF, the document, the undo
 * gesture) arrives as an argument, so the whole engine runs under `node --test` with three stubs.
 * main.js is 3,500 lines because everything ended up in it; this is the piece that does not.
 *
 * The service contracts below are MEASURED against the running service (plan "Step 4 outcomes"),
 * not read off the design:
 *  - trackRes is fixed at 1008. tools/sam-service/track.py:612 400s anything else, because
 *    prompt_encoder.image_embedding_size is image_size // patch_size and is fixed at construction.
 *  - A cold GET /sam/track/run cannot start the service: the dev-server proxy auto-starts on POST
 *    only, so a cold GET is a JSON 503 that an EventSource surfaces as a bare onerror with no
 *    message. Every run here is preceded by GET /sam/health and POST /sam/track/open.
 *  - /track/prompt returns no mask for the seed frame (the run re-emits it from
 *    cond_frame_outputs), so a run always starts AT the seed frame, never seed+1.
 *  - A prompt during a live run answers 409 (track.py:782). The correction loop cancels first.
 *  - /track/results is indexed by EVENT INDEX with next/more/total (track.py:1178), not by frame:
 *    a reverse run emits frames descending and a multi-object run emits one event per object per
 *    frame. `run` pins which run a replay belongs to.
 *  - 304 ms/frame measured, so 272 frames is ~83 s of streaming; a second object costs +44%.
 */

/** The tracker's native input edge. NOT a knob: track.py:612 rejects any other value. */
export const TRACK_RES = 1008;
/** Long side of the STORED bitmap. Free, unlike trackRes — plan Decisions #3. */
export const MASK_RES_DEFAULT = 768;
/** Frames per POST /sam/track/frames. 16 keeps a request ~1.5 MB and a retry cheap. */
export const UPLOAD_BATCH = 16;
/** Keyframes between save flushes. The run is one undo step but must survive a tab reload. */
export const COMMIT_EVERY = 16;
/** rAF retries waiting for the atlas. pumpTexture converges over up to 30 (player.ts:1044); 8 is
 *  ~130 ms, well past the measured settle, and a miss is logged rather than silently accepted. */
export const SETTLE_TRIES = 8;
/** Poll interval while replaying a run whose SSE stream this tab no longer holds. */
export const REPLAY_POLL_MS = 1000;

// ------------------------------------------------------------------ readouts ----
// Every string here is a shipped UI string: technical nomenclature, no sentence case, no prose.

export const captureLine = (done, of) => `capture ${done}/${of} · ${TRACK_RES} px`;
export const uploadLine = (bytes) => `upload ${(bytes / 1e6).toFixed(1)} MB`;
export const trackLine = (label, done, of, objId, msPerFrame, etaS) =>
  `${label} ${done}/${of} · obj ${objId} · ${(msPerFrame / 1000).toFixed(2)} s/f · eta ${Math.round(etaS)} s`;
export const doneLine = (done, of, ms, kf, gaps) =>
  `track ${done}/${of} · ${Math.round(ms / 1000)} s · ${TRACK_RES} · ${kf} kf` + (gaps ? ` · ${gaps} gaps` : "");
export const cancelLine = (kf) => `track cancelled · ${kf} kf`;
export const gapLine = (frame, reason) => `frame ${frame} · ${reason === "empty" ? "empty mask" : "occluded"}`;
export const lagLine = (frame) => `frame ${frame} · texture lag`;

// ----------------------------------------------------------------- preflight ----

/**
 * Refusals before any GPU work, because each of these silently produces WRONG masks on every frame
 * rather than failing loudly:
 *  - a non-identity model transform: pickRaster (player.ts:374) and prepareVolume (edits.ts:295)
 *    project through bare orbitViewProj while both renderers draw through viewProj × model, so the
 *    image SAM segments and the region test the bake runs disagree by the transform;
 *  - non-identity fx: it displaces vertices and discards fragments in the SHADER only, so the
 *    proxy render carries geometry the evaluator cannot see;
 *  - a splat clip: exportFrame and pickRaster both return null, so there is nothing to test a
 *    region against.
 * Returns a UI string, or null when the sweep may start.
 */
export function preflight({ health, splat, xfDefault = true, fxIdentity = true } = {}) {
  if (splat) return "splat clip · tracking needs mesh triangles";
  if (!xfDefault) return "model transform active · reset it before tracking";
  if (!fxIdentity) return "fx active · reset fx before tracking";
  if (!health || !health.ok) return "SAM service not running";
  if (health.trackError) return "track failed: " + health.trackError;
  if (health.trackLoading && !health.trackReady) return "track model loading · retry shortly";
  return null;
}

// ------------------------------------------------------------ frame batching ----

/**
 * The upload wire format, verbatim from track.py:648 — a bare stream of records with no envelope:
 *   [u32le frameIdx][u32le byteLen][byteLen bytes of JPEG] …
 * The service applies the prefix it already parsed and is keyed by frame index, so re-POSTing a
 * whole batch after a failure is idempotent: the frames that landed are simply overwritten.
 */
export function packFrameBatch(items) {
  let n = 0;
  for (const it of items) n += 8 + it.bytes.length;
  const out = new Uint8Array(n);
  const dv = new DataView(out.buffer);
  let off = 0;
  for (const it of items) {
    dv.setUint32(off, it.frame, true);
    dv.setUint32(off + 4, it.bytes.length, true);
    out.set(it.bytes, off + 8);
    off += 8 + it.bytes.length;
  }
  return out;
}

/**
 * Prompt coordinates, capture pixels → proxy pixels, x and y scaled INDEPENDENTLY.
 * captureFrame/captureFrameBlob cap the LONG side (player.ts:435), so a portrait viewport is scaled
 * by maxDim/height and a landscape one by maxDim/width: one shared factor is only ever right by
 * accident, and a wrong seed point is a whole track segmenting the wrong object.
 */
export function rescalePoints(points, capture, proxy) {
  const sx = proxy.width / Math.max(1, capture.width);
  const sy = proxy.height / Math.max(1, capture.height);
  return points.map(([x, y]) => [x * sx, y * sy]);
}

/**
 * A mask with at least one set pixel. PRESENCE is not the test: rleEncodeMask emits the 0-run first
 * (edits.ts:50-59), so an all-zero bitmap encodes to a SINGLE run and the object itself is still
 * truthy. That mask passes validateMasks clean — its runs do sum to width*height (edits.ts:305) —
 * and then deletes the entire frame through keepPredicateAt, which drops every point outside all
 * active keep regions (edits.ts:592-595). Reachable from a SAM click that returned nothing.
 */
export const maskIsSolid = (m) => !!m && Array.isArray(m.rle) && m.rle.length > 1;

/** Frames the tracker reported nothing for, as the INCLUSIVE pairs TrackProvenance.gaps stores. */
export function coalesceGaps(frames) {
  const s = [...new Set(frames)].sort((a, b) => a - b);
  const out = [];
  for (const f of s) {
    const last = out[out.length - 1];
    if (last && f === last[1] + 1) last[1] = f;
    else out.push([f, f]);
  }
  return out;
}

// -------------------------------------------------------------------- writer ----

/**
 * The keyframe writer: the only thing in this file that touches the edit list.
 *
 * Pure — it takes the document object and mutates it, and knows nothing about fetch, the player or
 * the DOM. That is what makes the gap policy testable, and the gap policy is the part that destroys
 * a user's frame when it is wrong.
 *
 * GAP POLICY IS PER MODE (plan Decisions #2):
 *  - mode:"delete" splits the track into ONE RANGE PER CONTIGUOUS RUN. With no range spanning the
 *    gap, keepPredicateAt returns null at that frame (edits.ts:581) and the frame survives whole.
 *  - mode:"keep" (isolate) keeps ONE range spanning the gap and writes a derived keyframe at each
 *    gap frame carrying the LAST NON-EMPTY mask with conf 0. An empty keep mask is never emitted:
 *    keepPredicateAt (edits.ts:578-586) drops every point outside all active keep regions, so one
 *    all-zero keep keyframe deletes the entire frame.
 *
 * Keyframes land AS THEY STREAM. The caller wraps the whole run in one gestureBegin/gestureCommit
 * pair (main.js:1113/1114) so it still collapses to a single undo step, and onCommit fires every
 * COMMIT_EVERY keyframes so a tab reload loses at most 16 frames — not the whole track.
 */
export function createTrackWriter({
  doc,
  newRangeId,
  nextColor = () => undefined,
  applyRangeAction = null,
  onRangeCreated = null,
  onCommit = null,
  commitEvery = COMMIT_EVERY,
} = {}) {
  if (!doc || !Array.isArray(doc.ranges)) throw new Error("track writer needs a document with a ranges array");

  let plan = null;                 // begin()'s argument, the whole track's shape
  const objs = new Map();          // objId -> { runs: [range], byFrame: Map(frame -> range), gaps: [] }
  // A re-seed's FRESH objId, aliased onto the bucket of the object it replaces. The library has no
  // per-object removal (track.py:778), so the drifted object stays in the inference session and the
  // run keeps emitting for it; the alias is what makes the new id refill the OLD object's ranges
  // instead of authoring a second set over the same frames.
  const aliases = new Map();
  let pending = 0;                 // keyframes since the last commit tick
  const stats = { kf: 0, user: 0, derived: 0, gaps: 0, filled: 0 };

  const obj = (objId) => {
    const key = aliases.get(objId) ?? objId;
    let o = objs.get(key);
    if (!o) { o = { runs: [], byFrame: new Map(), gaps: [], last: null, seeds: [] }; objs.set(key, o); }
    return o;
  };
  /** The one frame a correction promotes. Per-object, because a multi-object run emits for every
   *  object at that frame and only the corrected one may be promoted. */
  const correctingAt = (frame) => !!plan.correcting && plan.correcting.frame === frame;
  /** Events for any object but the one being corrected are dropped for the window's duration — see
   *  beginCorrection: they are the drift the correction exists to replace. */
  const notCorrecting = (objId) => !!plan.correcting && plan.correcting.objId !== objId;

  const provenanceFor = (o, runIdx) => ({
    tool: "sam3-tracker-video",
    checkpoint: plan.checkpoint ?? "",
    trackRes: plan.trackRes ?? TRACK_RES,
    maskRes: plan.maskRes ?? MASK_RES_DEFAULT,
    dtype: plan.dtype ?? "",
    proxy: plan.proxy,
    camera: plan.camera,
    trackId: plan.trackId,
    objId: o.objId,
    run: runIdx,
    seeds: o.seeds,
    span: { from: plan.from, to: plan.to },
    ranAt: plan.ranAt,
  });

  function newRun(o, frame) {
    const r = {
      id: newRangeId(),
      color: plan.color ?? nextColor(),
      mode: plan.mode,
      startFrame: frame,
      endFrame: frame,
      keyframes: [],
      track: provenanceFor(o, o.runs.length),
    };
    if (plan.action && plan.action !== "delete") r.action = plan.action;
    // setRangeAction's payload defaults (main.js) are main.js's business: the writer only ever
    // hands it a range it just built, never one the user has already shaped.
    applyRangeAction?.(r, plan.action ?? "delete");
    o.runs.push(r);
    doc.ranges.push(r);
    onRangeCreated?.(r);
    return r;
  }

  /** Contiguous-run index within the object, 0-based (TrackProvenance.run). Recomputed after a
   *  merge so the numbering still reads left to right in the timeline. */
  function renumber(o) {
    o.runs.sort((a, b) => a.startFrame - b.startFrame);
    o.runs.forEach((r, i) => { r.track.run = i; });
  }

  /** A frame that bridges two runs — the reverse leg of a bidi track meeting the forward leg, or a
   *  correction filling a hole — makes them ONE run. Leaving them split would leave the gap policy
   *  claiming an occlusion that is no longer there. */
  function merge(o, a, b) {
    for (const kf of b.keyframes) { a.keyframes.push(kf); o.byFrame.set(kf.frame, a); }
    a.keyframes.sort((x, y) => x.frame - y.frame);
    a.startFrame = Math.min(a.startFrame, b.startFrame);
    a.endFrame = Math.max(a.endFrame, b.endFrame);
    o.runs.splice(o.runs.indexOf(b), 1);
    const di = doc.ranges.indexOf(b);
    if (di !== -1) doc.ranges.splice(di, 1);
    renumber(o);
    return a;
  }

  function runFor(o, frame) {
    // A frame that already carries a keyframe keeps the range holding it. Without this a replay
    // refilling an ADOPTED range whose neighbours are empty would author a second range beside it
    // and orphan the keyframe the sidecar already holds.
    const own = o.byFrame.get(frame);
    if (own) return own;
    // keep/isolate never splits: one range spans the gap, because a frame with no active keep
    // range shows the FULL scene, which inside an isolate is a flash of everything the user asked
    // to remove.
    if (plan.mode === "keep") return o.runs[0] ?? newRun(o, frame);
    const left = o.byFrame.get(frame - 1), right = o.byFrame.get(frame + 1);
    if (left && right && left !== right) return merge(o, left, right);
    return left ?? right ?? newRun(o, frame);
  }

  function place(o, kf) {
    const r = runFor(o, kf.frame);
    const at = r.keyframes.findIndex((k) => k.frame === kf.frame);
    if (at === -1) {
      r.keyframes.push(kf);
      r.keyframes.sort((a, b) => a.frame - b.frame);
      stats.kf++;
    } else {
      r.keyframes[at] = kf;
    }
    if (kf.derived) stats.derived++; else stats.user++;
    o.byFrame.set(kf.frame, r);
    if (kf.frame < r.startFrame) r.startFrame = kf.frame;
    if (kf.frame > r.endFrame) r.endFrame = kf.frame;
    if (++pending >= commitEvery) { pending = 0; onCommit?.(); }
    return r;
  }

  const volumeFor = (mask, depth) => ({
    type: "mask2d",
    kind: "bitmap",
    mask,
    camera: plan.camera,
    ...(depth ? { depth } : {}),
  });

  return {
    /**
     * plan: { seedFrame, from, to, mode, action, color, camera, proxy, seedMask, seedDepth,
     *         trackId, checkpoint, dtype, trackRes, maskRes, ranAt }
     * Nothing is written here — the run re-emits the seed frame's own mask from cond_frame_outputs
     * (track.py, plan step-4 outcome #3), and THAT is the mask that must be stored: it is the
     * tracker's, at maskRes, in the tracker's own convention. seedMask is only the fallback for a
     * seed frame the run somehow never emits, and the first `last mask` a keep gap can carry.
     */
    begin(p) {
      plan = { trackRes: TRACK_RES, maskRes: MASK_RES_DEFAULT, ...p };
      plan.trackId = plan.trackId ?? `trk-${Date.now().toString(36)}`;
      plan.ranAt = plan.ranAt ?? new Date().toISOString();
      for (const objId of plan.objIds ?? []) {
        const o = obj(objId);
        o.objId = objId;
        o.last = plan.seedMask ?? null;
      }
    },

    /** One `mask` event → one keyframe. The rle is stored VERBATIM: the service already emits the
     *  edit list's own convention (alternating run lengths, 0-run first, edits.ts:49), so there is
     *  no decode and no re-encode anywhere on this path. */
    mask(ev, depth = null) {
      // A correction window belongs to ONE object. Every other object of the session is still being
      // propagated (the run loops out.object_ids, track.py:1055) and its masks at these frames are
      // exactly the drift the correction was pressed to replace, so they are dropped rather than
      // written back — including at the corrected frame, which only its own object may promote.
      if (notCorrecting(ev.objId)) return null;
      const o = obj(ev.objId);
      if (o.objId === undefined) { o.objId = ev.objId; o.last = plan.seedMask ?? null; }
      const mask = { width: ev.w, height: ev.h, rle: ev.rle };
      const existing = o.byFrame.get(ev.frame);
      const prior = existing?.keyframes.find((k) => k.frame === ev.frame);
      // A user keyframe outranks anything a run emits: the reverse leg of a bidi track re-emits the
      // seed frame (propagate is INCLUSIVE of its start), and a correction's window ends at the
      // next user keyframe. Neither may overwrite one.
      if (prior && !prior.derived && !correctingAt(ev.frame)) { o.last = mask; return prior; }
      const seed = ev.frame === plan.seedFrame || correctingAt(ev.frame);
      const kf = seed
        ? { frame: ev.frame, volumes: [volumeFor(mask, depth ?? plan.seedDepth ?? null)] }
        : { frame: ev.frame, derived: true, conf: ev.score, volumes: [volumeFor(mask, depth)] };
      o.last = mask;
      return place(o, kf);
    },

    /** One `gap` event. Delete splits here (no keyframe at all); keep fills with the last non-empty
     *  mask at conf 0, which is also what puts the drift caret and the N key on the frame. */
    gap(ev) {
      if (notCorrecting(ev.objId)) return null;
      const o = obj(ev.objId);
      if (o.objId === undefined) { o.objId = ev.objId; o.last = plan.seedMask ?? null; }
      o.gaps.push(ev.frame);
      stats.gaps++;
      if (plan.mode !== "keep") return null;
      const mask = maskIsSolid(o.last) ? o.last : (maskIsSolid(plan.seedMask) ? plan.seedMask : null);
      // With no mask to carry, writing nothing is the only safe move — and an all-zero one counts
      // as none: it is a truthy object whose runs still sum to width*height, so it validates clean
      // and then deletes the whole frame. Inventing a mask across a gap is the failure
      // docs/rgbd-rebuild-pipeline.md:93-98 records poisoning a 64-frame batch.
      if (!mask) return null;
      stats.filled++;
      return place(o, { frame: ev.frame, derived: true, conf: 0, volumes: [volumeFor(mask, null)] });
    },

    /** Record a prompt on the track, in application order (TrackSeed). Coordinates are PROXY-render
     *  pixels, not capture pixels, so replaying `seeds` against the same proxy renders reproduces
     *  the track. */
    seed(s) {
      const o = obj(s.objId);
      if (o.objId === undefined) o.objId = s.objId;
      o.seeds.push(s);
    },

    /** The first user keyframe strictly after `frame` for this object — the exclusive upper bound
     *  of a correction window (Mocha's AdjustTrack rule: corrections are a sparse layer over a
     *  dense automatic result, never an in-place overwrite of it). */
    nextUserFrame(objId, frame) {
      let best = Infinity;
      for (const r of obj(objId).runs)
        for (const k of r.keyframes)
          if (!k.derived && k.frame > frame && k.frame < best) best = k.frame;
      return best;
    },

    /**
     * Clear the derived keyframes of [frame, until - 1] so a re-propagation can refill them, and
     * promote `frame` itself from derived to user. Only DERIVED keyframes inside the window are
     * touched; every user keyframe survives, which is what makes a correction bounded.
     *
     * `fromObjId` is the object whose drifted keyframes the window holds, `objId` the one the re-run
     * will emit under. Re-seed makes them differ, and the new id is ALIASED onto the old object's
     * bucket rather than given one of its own: its masks refill the SAME ranges, where a second set
     * over the same frames would simply be UNIONED with the drift at bake time (keepPredicateAt,
     * edits.ts:583-595) — a correction that strictly enlarges the error.
     *
     * `volumes` is the correction's own mask, and passing it is not optional for a Nudge: the prompt
     * is non-conditioning on an already-tracked frame, so the re-run starts at frame + 1 and never
     * re-emits this frame. Without it the promoted keyframe keeps the exact mask the user pressed
     * Nudge to reject — as a USER keyframe, which is what every later run and every rebase refuse to
     * overwrite (edits.ts, cli.ts rebaseEditList).
     */
    beginCorrection({ objId, fromObjId = objId, frame, until, volumes = null }) {
      const key = aliases.get(fromObjId) ?? fromObjId;
      const o = obj(fromObjId);
      const hi = Number.isFinite(until) ? until - 1 : Infinity;
      for (const r of [...o.runs]) {
        const keep = [];
        for (const k of r.keyframes) {
          if (k.derived && k.frame > frame && k.frame <= hi) { o.byFrame.delete(k.frame); stats.kf--; stats.derived--; continue; }
          keep.push(k);
        }
        r.keyframes = keep;
        if (!keep.length) {
          o.runs.splice(o.runs.indexOf(r), 1);
          const di = doc.ranges.indexOf(r);
          if (di !== -1) doc.ranges.splice(di, 1);
          continue;
        }
        r.startFrame = Math.min(r.startFrame, keep[0].frame);
        r.endFrame = keep[keep.length - 1].frame;
      }
      renumber(o);
      if (objId !== fromObjId) {
        aliases.set(objId, key);
        o.objId = objId;
        // The provenance has to name the object the tracker will answer for from here on, or the
        // next Nudge would prompt the retired id and trackDerived would stop matching the range.
        for (const r of o.runs) r.track.objId = objId;
      }
      const r = o.byFrame.get(frame);
      const kf = r?.keyframes.find((k) => k.frame === frame);
      if (kf) {
        delete kf.derived;
        delete kf.conf;
        if (volumes) kf.volumes = volumes;
        stats.user++;
      }
      plan.correcting = { objId, frame };
      return { from: frame, to: Number.isFinite(hi) ? hi : plan.to };
    },

    endCorrection() { if (plan) delete plan.correcting; },

    /**
     * Adopt ranges an interrupted run already flushed to the sidecar, so a replay REFILLS them
     * instead of rebuilding a second set beside them.
     *
     * Mandatory for a bidirectional track: track_run clears the event log and bumps run_id on every
     * request (track.py:975-976), so the offered run is the REVERSE leg alone and the forward leg's
     * frames exist nowhere but the sidecar. Dropping the ranges before replaying would destroy every
     * frame it already computed and saved. place() replaces by frame, so a replay over an adopted
     * range is idempotent and the overlap costs nothing.
     */
    adopt(ranges) {
      for (const r of ranges) {
        if (!r.track) continue;
        const o = obj(r.track.objId);
        o.objId = r.track.objId;
        if (!o.seeds.length && r.track.seeds?.length) o.seeds = r.track.seeds.slice();
        for (const g of r.track.gaps ?? []) for (let f = g[0]; f <= g[1]; f++) o.gaps.push(f);
        o.runs.push(r);
        for (const k of r.keyframes) {
          o.byFrame.set(k.frame, r);
          stats.kf++;
          if (k.derived) stats.derived++; else stats.user++;
        }
        // The last mask a keep gap can carry has to survive the reload too, or the first gap after
        // a resume would write nothing and the isolate would flash the whole scene.
        const tail = r.keyframes[r.keyframes.length - 1]?.volumes?.[0]?.mask;
        if (maskIsSolid(tail)) o.last = tail;
      }
      for (const o of objs.values()) renumber(o);
      return this.stats();
    },

    /** Finish the track: seed fallback, gap provenance, span clamp, one last commit tick. */
    end() {
      if (!plan) return this.stats();
      for (const o of objs.values()) {
        // The run always re-emits the seed frame, so this only fires when it did not: a gap on the
        // seed frame, or a run that never started. Landing the pending selection there keeps the
        // range anchored on a user keyframe rather than on nothing.
        if (maskIsSolid(plan.seedMask) && !o.byFrame.has(plan.seedFrame) && o.runs.length === 0)
          place(o, { frame: plan.seedFrame, volumes: [volumeFor(plan.seedMask, plan.seedDepth ?? null)] });
        const gaps = coalesceGaps(o.gaps);
        for (const r of o.runs) {
          // Copied per range rather than shared: undo deep-clones the ranges, and a shared array
          // would come back as N independent copies anyway — with the identity silently gone.
          if (gaps.length) r.track.gaps = gaps.map((g) => [g[0], g[1]]);
          r.track.seeds = o.seeds.slice();
        }
        renumber(o);
      }
      pending = 0;
      onCommit?.();
      return this.stats();
    },

    ranges() { return [...objs.values()].flatMap((o) => o.runs); },
    stats() { return { ...stats, ranges: this.ranges().length }; },
    lastMask(objId) { return obj(objId).last; },
  };
}

// -------------------------------------------------------------------- client ----

const raf16 = (fn) => setTimeout(fn, 16);

/**
 * The transport half: capture sweep, upload, seeding, the SSE run, replay and cancellation.
 *
 * deps:
 *   player        AresPlayer handle — seekFrame, textureDebug, captureFrameBlob, getCamera,
 *                 getViewAspect, isSplat, isGrid/setGrid, setEditPreview, autoOrbit.
 *   writer        createTrackWriter() result.
 *   doc           the live edit-list document (only read, for the pre-track backup).
 *   clipBase      sidecar name, i.e. what POST /edits/<clipBase> writes.
 *   fetch         fetch implementation (default globalThis.fetch).
 *   EventSource   EventSource implementation (default globalThis.EventSource).
 *   raf           requestAnimationFrame implementation (default globalThis.requestAnimationFrame).
 *   onStatus      (line) => void — the one-line readout.
 *   onLog         (line) => void — the service's own `log` lines and per-frame notices.
 *   onProgress    ({ phase, done, of, ... }) => void.
 *   getShade      optional () => shade mode, recorded in TrackProvenance.proxy.shade. The proxy is
 *                 what SAM saw, and shade mode changes it.
 *   camKey        optional () => string, main.js's own camKeyOf formatter. Sampled at the freeze and
 *                 re-asserted per capture: the freeze stops autoOrbit but nothing locks camera
 *                 input, and a proxy rendered from a moved camera is segmented against a stored
 *                 `camera` that no longer describes it.
 *   onRestore     optional () => void, called once the freeze is undone — main.js re-applies its
 *                 own edit preview there rather than having this module guess the document back.
 *   depthAt      optional and SYNCHRONOUS ({ frame, w, h, rle }) => {zmin, zmax} | null. The
 *                 player is seeked to `frame` before the call. Absent ⇒ derived keyframes carry NO
 *                 depth band, i.e. the propagated region selects THROUGH the mesh even when the
 *                 seed was captured with X-ray off. The seed's own band cannot be carried forward:
 *                 it is an NDC z window measured at one frame, and a subject that walks toward the
 *                 camera leaves it within a few frames.
 */
export function createTrackClient(deps = {}) {
  const {
    player, writer, doc, clipBase = "",
    fetch: fetchImpl = globalThis.fetch?.bind(globalThis),
    EventSource: ES = globalThis.EventSource,
    raf = globalThis.requestAnimationFrame ? globalThis.requestAnimationFrame.bind(globalThis) : raf16,
    onStatus = () => {}, onLog = () => {}, onProgress = () => {},
    depthAt = null,
  } = deps;

  let session = null;          // the open service session id
  let runId = 0;               // the `run` id the last `start` event carried
  let primaryObj = 1;          // the object the readout names; a multi-object run costs +44%/object
  let es = null;               // the live EventSource, if any
  let settleStream = null;     // settles the live stream() promise, for a cancel that cannot wait
  let cancelled = false;
  let running = false;
  let frozen = null;           // saved player state, restored in a finally

  const jsonPost = async (url, body) => {
    const res = await fetchImpl(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await res.json().catch(() => null);
    if (!res.ok) throw new Error(j?.detail ?? j?.error ?? `${url} ${res.status}`);
    return j;
  };
  const jsonGet = async (url) => {
    const res = await fetchImpl(url);
    const j = await res.json().catch(() => null);
    if (!res.ok) throw new Error(j?.detail ?? j?.error ?? `${url} ${res.status}`);
    return j;
  };

  /**
   * ONE-SHOT PRE-TRACK BACKUP, on the route that already exists.
   *
   * serve.mjs:769 keeps a one-deep `.bak` per sidecar, and the writer flushes every 16 frames, so a
   * 272-frame run fires ~17 saves and the .bak holds mid-track state by the second one (plan
   * step-4 outcome #9). POST /edits/<name> sanitizes the name to [a-z0-9._-] and appends
   * ".edits.json" (serve.mjs:708), so posting the pre-track document to "<clip>.pretrack" writes
   * apps/demo/<clip>.pretrack.edits.json — a real second file, restorable by GET on the same path
   * or by one file copy. No server affordance is invented here and none is missing.
   */
  async function backup() {
    if (!clipBase) return false;
    try {
      await fetchImpl("/edits/" + encodeURIComponent(clipBase + ".pretrack"), {
        method: "POST", body: JSON.stringify(doc),
      });
      return true;
    } catch { return false; }   // a failed backup must never block the track
  }

  /**
   * FREEZE. The tracker sees exactly what the canvas shows, so anything that moves or draws under
   * the subject is a stable distractor a memory bank latches onto:
   *  - autoOrbit off: a moving camera invalidates every stored `camera` on the very first frame;
   *  - grid off: grid and tripod draw into the same pass as the mesh (renderer.ts), and a static
   *    world grid under a moving subject is the textbook latch;
   *  - edit preview off: pickRaster and the render both read the edit-filtered curIndices, so
   *    tracking with a live delete range active renders the subject's own hole and tracks THAT.
   */
  function freeze() {
    frozen = {
      autoOrbit: player.autoOrbit,
      grid: player.isGrid?.() ?? false,
      shade: deps.getShade?.() ?? "shaded",
      camKey: deps.camKey?.() ?? null,
    };
    player.autoOrbit = false;
    // Playback too: seekFrame does NOT pause (player.ts:920-935 leaves `playing` alone) and every
    // await in the sweep is a real task boundary the player's own rAF loop runs through, so a clip
    // left playing advances under the capture and captureToCanvas copies whatever is presented
    // (player.ts:433) — frame f's record carrying another frame's pixels. Not resumed in restore():
    // the sweep left the playhead at the far end of the span, which is not where it was paused.
    if (player.isPlaying) player.pause?.();
    player.setGrid?.(false);
    player.setEditPreview?.(null);
    return frozen;
  }
  function restore() {
    if (!frozen) return;
    player.autoOrbit = frozen.autoOrbit;
    player.setGrid?.(frozen.grid);
    frozen = null;
    deps.onRestore?.();          // main.js re-applies the edit preview through its own preview()
  }

  /** Wait for the atlas. seek/seekFrame present GEOMETRY synchronously, but pumpTexture converges
   *  the texture over up to 30 rAF retries (player.ts:1044): without this the tracker is handed
   *  frame N's mesh wearing frame N−k's atlas and every mask after a cut is segmented off the wrong
   *  pixels. Bounded at SETTLE_TRIES and logged, never blocking the sweep forever. */
  function settle() {
    const dbg = player.textureDebug?.();
    if (!dbg || dbg.settled) return Promise.resolve(true);
    return new Promise((resolve) => {
      let tries = 0, done = false;
      const finish = (v) => { if (!done) { done = true; clearTimeout(timer); resolve(v); } };
      // Bounded by the WALL CLOCK as well as by ticks. A hidden tab suspends rAF entirely and
      // `tries` counts EXECUTED ticks, so the bound above never trips: the sweep parks forever with
      // Esc unreachable (`cancelled` is only read at the top of the sweep loop) while the service's
      // 600 s idle reaper drops the session underneath it. A timeout keeps running in a hidden tab,
      // and a miss here is the same logged lag a slow atlas already produces.
      const timer = setTimeout(() => finish(false), SETTLE_TRIES * 16 + 100);
      const tick = () => {
        const d = player.textureDebug();
        if (d.settled) return finish(true);
        if (++tries >= SETTLE_TRIES) return finish(false);
        raf(tick);
      };
      raf(tick);
    });
  }

  async function postFrames(batch, tries = 3) {
    const body = packFrameBatch(batch);
    for (let attempt = 1; ; attempt++) {
      try {
        const res = await fetchImpl(`/sam/track/frames?session=${encodeURIComponent(session)}`, {
          method: "POST", headers: { "content-type": "application/octet-stream" }, body,
        });
        const j = await res.json().catch(() => null);
        if (!res.ok) throw new Error(j?.detail ?? `frames ${res.status}`);
        return j;
      } catch (e) {
        // Idempotent by frame index and the service keeps whatever prefix it already parsed
        // (track.py:648), so the whole batch simply goes again.
        if (attempt >= tries) throw e;
        onLog(`[track] frame batch retry ${attempt}: ${e.message}`);
      }
    }
  }

  /**
   * The sweep. One capture per frame under the frozen camera, 16 frames per POST.
   * `first` is the frame already captured to learn the proxy size — captured once, uploaded once.
   */
  async function sweep({ from, to, first = null }) {
    let batch = [], bytes = 0, done = 0;
    const of = to - from + 1;
    for (let f = from; f <= to; f++) {
      if (cancelled) break;
      // The freeze turns autoOrbit off but nothing locks camera INPUT: the proxy is rendered from
      // the live orbit (captureToCanvas → renderCurrent, player.ts:435) while every stored volume
      // carries the one frozen plan.camera, so an orbit drag or a P toggle mid-sweep segments images
      // the bake will never re-project through — and the only symptom is a wrong bake. Refusing the
      // run costs the sweep; not refusing it costs the whole tail of the clip, silently.
      if (frozen?.camKey && deps.camKey() !== frozen.camKey) throw new Error("camera moved · restore view");
      let cap;
      if (first && f === first.frame) {
        cap = first;
      } else {
        player.seekFrame(f);
        if (!(await settle())) onLog(lagLine(f));
        // Nothing suspends the timeline during a sweep — Space, the arrow keys and a timeline drag
        // all seek the same player through the awaits above — and captureToCanvas copies whatever
        // is PRESENTED. A moved playhead would upload another frame's pixels under this frame's
        // index, and the mask comes back under f and is stored verbatim as f's own keyframe.
        const at = player.getStats?.().frameIndex;
        if (at !== undefined && at !== f) { player.seekFrame(f); await settle(); }
        cap = await player.captureFrameBlob(TRACK_RES, "image/jpeg", 0.95);
      }
      if (!cap) throw new Error(`capture failed at frame ${f}`);
      const buf = new Uint8Array(cap.bytes ?? await cap.blob.arrayBuffer());
      batch.push({ frame: f, bytes: buf });
      bytes += buf.length;
      done++;
      if (batch.length >= UPLOAD_BATCH) { await postFrames(batch); batch = []; }
      onStatus(captureLine(done, of));
      onProgress({ phase: "capture", done, of, bytes });
    }
    if (batch.length) await postFrames(batch);
    onStatus(uploadLine(bytes));
    return { frames: done, bytes };
  }

  /** One `mask` event → one keyframe, with the optional per-frame depth band. SYNCHRONOUS on
   *  purpose: mask events must reach the writer in emission order, and an await here would let a
   *  later frame's keyframe land first on any depthAt that yields. pickRaster is synchronous
   *  (player.ts:374), so nothing on this path needs to. */
  function applyMask(ev) {
    let depth = null;
    // The band is measured off the LIVE render, so a camera that has moved since the freeze would
    // band the region against depths from another viewpoint. No band at all is the honest answer
    // there — a mask2d with no depth simply selects through the mesh.
    if (depthAt && (!frozen?.camKey || deps.camKey() === frozen.camKey)) {
      // pickRaster reads the CURRENTLY PRESENTED frame only (player.ts:374), and by the time mask f
      // arrives the sweep has long since finished, so the player has to be put back on frame f.
      // Geometry is synchronous on seek; the atlas does not matter here, only positions do.
      player.seekFrame(ev.frame);
      depth = depthAt({ frame: ev.frame, w: ev.w, h: ev.h, rle: ev.rle }) ?? null;
    }
    writer.mask(ev, depth);
  }

  /**
   * Consume GET /sam/track/run as SSE. Resolves with the `done` payload.
   *
   * Everything that can fail does so BEFORE the head as 400/503/409 JSON (track.py:929), which an
   * EventSource cannot read — it surfaces as a bare onerror. So a bare onerror with no event
   * received is reported as exactly that, and every precheck this client can run itself (health,
   * open, seeded objects) has already run by here.
   */
  function stream({ start, dir, max, label = "track" }) {
    return new Promise((resolve, reject) => {
      const q = `session=${encodeURIComponent(session)}&start=${start}&dir=${dir}&max=${max}`;
      es = new ES("/sam/track/run?" + q);
      let sawEvent = false, finished = false;
      const shut = () => { try { es?.close(); } catch { /* already closed */ } es = null; settleStream = null; running = false; };
      // cancel() cannot wait for the service's own `done`: EventSource.close() aborts the connection
      // and dispatches NOTHING (HTML spec), while the cooperative cancel only reaches the next loop
      // iteration ~304 ms later. Without a handle to settle this promise the `await` in run() never
      // returns, its finally never runs, and the freeze — autoOrbit, the grid, the edit preview —
      // stays in force for the life of the tab.
      settleStream = () => {
        if (finished) return;
        finished = true;
        shut();
        onStatus(cancelLine(writer.stats().kf));
        resolve({ frames: 0, of: max, ms: 0, cancelled: true });
      };
      const on = (name, fn) => es.addEventListener(name, (e) => {
        sawEvent = true;
        let d = null;
        try { d = JSON.parse(e.data); } catch { d = e.data; }
        fn(d);
      });
      running = true;
      on("start", (d) => {
        runId = d.run;
        primaryObj = d.objIds?.[0] ?? primaryObj;
        onLog(`[track] run ${d.run} · ${d.dir} ${d.start} · ${d.max} frames`);
      });
      on("log", (d) => onLog(typeof d === "string" ? d : JSON.stringify(d)));
      on("mask", (d) => applyMask(d));
      on("gap", (d) => { writer.gap(d); onLog(gapLine(d.frame, d.reason)); });
      on("progress", (d) => {
        onStatus(trackLine(label, d.done, d.of, primaryObj, d.msPerFrame, d.etaS));
        onProgress({ phase: "track", ...d });
      });
      on("done", (d) => {
        finished = true;
        shut();
        const s = writer.stats();
        onStatus(d.cancelled ? cancelLine(s.kf) : doneLine(d.frames, d.of, d.ms, s.kf, d.gaps));
        resolve(d);
      });
      // NOT on(): the service's own failure event is NAMED "error" (track.py:1089) and EventSource
      // dispatches its TRANSPORT failure under that same name, so both arrive on this one listener
      // and listeners run before onerror. Taking them both here swallowed the refusal diagnostic
      // below — a 400/503 before the head reached the caller as the generic "track failed", which
      // is the one message that cannot tell a dropped stream from a service that refused the run.
      // Only the service's event carries data; a transport failure has none, so it falls through.
      es.addEventListener("error", (e) => {
        if (typeof e.data !== "string") return;   // transport failure: onerror below owns it
        sawEvent = true;
        finished = true;
        shut();
        let d = null;
        try { d = JSON.parse(e.data); } catch { /* a non-JSON error frame is still an error */ }
        reject(new Error(d?.message ?? "track failed"));
      });
      es.onerror = () => {
        if (finished) return;
        shut();
        // A cold service answers the GET with a JSON 503 the EventSource cannot show. Every caller
        // here has already POSTed /track/open, so this is a dropped stream, not a cold start —
        // and the masks already written are already keyframes.
        reject(new Error(sawEvent ? "track stream closed" : "track stream refused: the service did not accept the run"));
      };
    });
  }

  return {
    session: () => session,
    runId: () => runId,
    isRunning: () => running,

    /** GET /sam/health, so the refusal strings above can be honest. A GET does NOT auto-start the
     *  service (the proxy guard starts it on POST only), which is exactly why this is a gate and
     *  not a start. */
    async health() { try { return await jsonGet("/sam/health"); } catch { return null; } },

    async listSessions() { return jsonGet("/sam/track/sessions"); },

    async open({ clip, camKey, frames, maskRes = MASK_RES_DEFAULT, capture }) {
      const j = await jsonPost("/sam/track/open", {
        clip, camKey, frames, trackRes: TRACK_RES, maskRes,
        captureW: capture?.width ?? 0, captureH: capture?.height ?? 0,
      });
      session = j.session;
      return j;
    },

    /**
     * Seed. EVERY object of the frame in ONE call: add_inputs_to_inference_session ends with
     * `inference_session.obj_with_new_inputs = obj_ids`, an ASSIGNMENT
     * (processing_sam3_tracker_video.py:736), so a second call for the same frame drops the first
     * object's pending prompt and tracks it as though nothing had been clicked.
     *
     * Returns the service's answer, including `conditioning`. When it is FALSE the frame was
     * already tracked, the prompt stored as non-conditioning, and re-running FROM that frame would
     * re-predict it from neighbouring memory and overwrite the correction — so the caller must
     * start at frame + 1. That is the difference between Nudge and Re-seed, and it comes from the
     * library, not from taste.
     */
    async prompt({ frame, objects, clearOldInputs = true, capture, proxy }) {
      const objs = objects.map((o) => ({
        objId: o.objId,
        ...(o.points ? { points: capture && proxy ? rescalePoints(o.points, capture, proxy) : o.points, labels: o.labels } : {}),
        ...(o.box ? { box: o.box } : {}),
        ...(o.mask ? { mask: o.mask } : {}),
      }));
      const j = await jsonPost("/sam/track/prompt", { session, frame, clearOldInputs, objects: objs });
      for (const o of objs)
        writer.seed({
          frame, objId: o.objId,
          kind: o.mask ? "mask" : o.box ? "box" : "points",
          mode: clearOldInputs ? "seed" : "nudge",
          ...(o.points ? { points: o.points, labels: o.labels } : {}),
          ...(o.box ? { box: o.box } : {}),
          ...(o.mask ? { mask: o.mask } : {}),
        });
      return j;
    },

    /**
     * The whole forward (or reverse, or bidirectional) track: gate, freeze, backup, open, sweep,
     * seed, stream. The caller owns gestureBegin/gestureCommit around this so the run is ONE undo
     * step; the writer's onCommit flushes a save every 16 keyframes inside it.
     */
    async run({
      clip, camKey, frames, from, to, seedFrame, dir = "forward",
      mode = "delete", action = "delete", color, camera, capture, objects,
      seedMask = null, seedDepth = null, maskRes = MASK_RES_DEFAULT, trackId,
    }) {
      cancelled = false;
      freeze();
      try {
        await backup();
        // The proxy size has to be known BEFORE /track/open: it is part of the retired frame
        // store's key (track.py:478), so opening with a zero size means a Retrack never matches the
        // store it just filled and sweeps the whole clip again. One capture answers it, and that
        // same capture is then the first frame of the sweep rather than being thrown away.
        player.seekFrame(from);
        if (!(await settle())) onLog(lagLine(from));
        const firstCap = await player.captureFrameBlob(TRACK_RES, "image/jpeg", 0.95);
        if (!firstCap) throw new Error("capture failed: the canvas cannot encode image/jpeg");
        const proxy = { width: firstCap.width, height: firstCap.height };

        primaryObj = objects[0]?.objId ?? 1;
        const open = await this.open({ clip, camKey, frames, maskRes, capture: proxy });
        const need = to - from + 1;
        if (open.reusedFrames >= frames) onLog(`[track] ${open.reusedFrames} frames reused · sweep skipped`);
        else await sweep({ from, to, first: { ...firstCap, frame: from } });
        if (cancelled) return { cancelled: true, ...writer.stats() };

        writer.begin({
          seedFrame, from, to, mode, action, color,
          // The aspect comes from the PROXY the sweep just captured, not from the selection: camKeyOf
          // (main.js:2475) hashes the orbit and ortho but NOT the viewport aspect, so a window resize
          // between the SAM click and K leaves samSel.cam.aspect stale while every bitmap is sized
          // from this render. prepareVolume builds orbitViewProj(cam, aspect) (edits.ts:360) and maps
          // NDC straight into the bitmap, so the pair has to agree or the whole track selects a
          // horizontally displaced region — which `ares verify-edits` reports (edits.ts:308) only
          // after the 83 s run, and the bake merely warns about.
          camera: { ...camera, aspect: proxy.width / proxy.height },
          proxy: { width: proxy.width, height: proxy.height, shade: frozen?.shade ?? "shaded", grid: false },
          seedMask, seedDepth, trackId,
          checkpoint: open.checkpoint, dtype: open.dtype, trackRes: open.trackRes, maskRes: open.maskRes,
          objIds: objects.map((o) => o.objId),
        });

        const seeded = await this.prompt({ frame: seedFrame, objects, clearOldInputs: true, capture, proxy });
        onLog(`[track] seeded obj ${seeded.objIds.join(", ")} at frame ${seedFrame}`);

        // Run FROM the seed frame, never seed + 1: /track/prompt returns no mask for it and the run
        // re-emits it out of cond_frame_outputs.
        const out = { forward: null, reverse: null };
        let stats;
        try {
          if (dir !== "reverse")
            out.forward = await stream({ start: seedFrame, dir: "forward", max: to - seedFrame + 1 });
          if ((dir === "reverse" || dir === "bidirectional") && !cancelled)
            out.reverse = await stream({ start: seedFrame, dir: "reverse", max: seedFrame - from + 1 });
        } finally {
          // end() is what writes the gap pairs, the per-range seed copies and the run numbering — on
          // every path, not just the clean one. A stream that REJECTS (the 30 s idle reaper, a
          // dropped proxy connection, a 409) has already left keyframes in the document, and a
          // tracked range with no `gaps` is indistinguishable from a truncated track.
          stats = writer.end();
        }
        return { ...out, ...stats, cancelled };
      } finally {
        restore();
        running = false;
      }
    },

    /**
     * The correction sequence a live run's 409 forces: cancel, re-seed, re-run, bounded to
     * [frame, nextUserKeyframe - 1].
     *
     * kind "nudge"   — clearOldInputs false, so the new point ACCUMULATES on the existing object
     *                  (the parameter defaults to true, processing_sam3_tracker_video.py:576). The
     *                  frame is already tracked, so the prompt is non-conditioning and the re-run
     *                  starts at frame + 1; the corrected frame keeps its own mask, promoted to a
     *                  user keyframe.
     * kind "reseed"  — a FRESH objId with clearOldInputs true. It has no frames_tracked entry, so
     *                  the prompt IS conditioning, the memory bank genuinely restarts, and the
     *                  re-run re-emits the corrected frame itself. `fromObjId` names the object it
     *                  replaces: the window is cleared on THAT object and the new id refills its
     *                  ranges, because the retired object cannot be removed from the session (the
     *                  library has no per-object removal, track.py:778) and would otherwise author a
     *                  second range the bake unions with the first.
     */
    async correct({ frame, objId, fromObjId = objId, kind = "nudge", objects, capture, proxy, dir = "forward", volumes = null }) {
      if (running) await this.cancel();
      cancelled = false;
      primaryObj = objId;
      const until = writer.nextUserFrame(fromObjId, frame);
      const win = writer.beginCorrection({ objId, fromObjId, frame, until, volumes });
      try {
        const seeded = await this.prompt({
          frame, objects, clearOldInputs: kind !== "nudge", capture, proxy,
        });
        const start = seeded.conditioning ? frame : frame + 1;
        if (!seeded.conditioning)
          onLog(`[track] frame ${frame} is non-conditioning · re-propagating from ${start}`);
        const max = (Number.isFinite(win.to) ? win.to : frame) - start + 1;
        if (max < 1) return { frames: 0, window: win };
        const done = await stream({ start, dir, max, label: "retrack" });
        return { ...done, window: win };
      } finally {
        writer.endCorrection();
      }
    },

    /**
     * Re-anchor: rebuild the session from every seed in frame order, one prompt per frame, then
     * re-propagate the window. Closing with keepFrames retires the frame store under its
     * clip+camera+res+size key, so the re-open reuses it and the sweep is skipped entirely.
     */
    async reanchor({ clip, camKey, frames, seeds, maskRes = MASK_RES_DEFAULT, proxy, from, to, dir = "forward" }) {
      if (running) await this.cancel();
      await this.close({ keepFrames: true });
      await this.open({ clip, camKey, frames, maskRes, capture: proxy });
      const byFrame = new Map();
      for (const s of seeds) {
        if (!byFrame.has(s.frame)) byFrame.set(s.frame, []);
        byFrame.get(s.frame).push(s);
      }
      for (const f of [...byFrame.keys()].sort((a, b) => a - b))
        await jsonPost("/sam/track/prompt", { session, frame: f, clearOldInputs: true, objects: byFrame.get(f) });
      return stream({ start: from, dir, max: to - from + 1, label: "retrack" });
    },

    /**
     * Replay a run this tab lost — a reload, or a stream that dropped mid-run.
     *
     * /track/results is indexed by EVENT INDEX, not by frame: a reverse run emits frames descending
     * and a multi-object run emits one event per object per frame, so a frame number cannot order a
     * replay. `run` pins which run the page belongs to; a changed id means the session started a
     * new run underneath and the replay is abandoned rather than mixed. While the run is still
     * live its SSE stream belongs to the dead reader and a second GET /run answers 409, so the only
     * way to follow it is to keep asking for pages.
     */
    async replay({ session: sid, run: wantRun = null, from = 0, pollMs = REPLAY_POLL_MS, maxPolls = 3600 }) {
      session = sid;
      let cursor = from, pinned = wantRun, polls = 0;
      for (;;) {
        const page = await jsonGet(`/sam/track/results?session=${encodeURIComponent(sid)}&from=${cursor}`);
        if (pinned === null) { pinned = page.run; runId = page.run; }
        if (page.run !== pinned) throw new Error(`track run changed under the replay: ${pinned} → ${page.run}`);
        for (const ev of page.events) {
          if (ev.event === "mask") applyMask(ev);
          else if (ev.event === "gap") writer.gap(ev);
        }
        cursor = page.next;
        onProgress({ phase: "replay", done: cursor, of: page.total });
        if (page.more) continue;
        if (!page.running || cancelled) return { events: cursor, run: pinned, running: page.running, ...writer.stats() };
        if (++polls > maxPolls) return { events: cursor, run: pinned, running: true, ...writer.stats() };
        await new Promise((r) => setTimeout(r, pollMs));
      }
    },

    /** Cancel: the POST goes FIRST and the EventSource is closed after it, because whether the dev
     *  server's /sam proxy propagates a client disconnect to FastAPI's request.is_disconnected() is
     *  not something this tree has ever exercised. Every mask already emitted is already a keyframe,
     *  so a cancel leaves a shorter but valid track. */
    async cancel() {
      cancelled = true;
      let out = null;
      if (session) { try { out = await jsonPost("/sam/track/cancel", { session }); } catch { /* gone */ } }
      // Settle the in-flight stream FIRST: close() dispatches no event of any kind, so nothing else
      // would ever resolve it and run()'s finally would never restore the player.
      settleStream?.();
      try { es?.close(); } catch { /* already closed */ }
      es = null;
      running = false;
      return out;
    },

    /** keepFrames retires the JPEG store under its clip+camera+res+size key for the reaper's 10
     *  minutes, so a Retrack skips the sweep. The memory bank is freed either way. */
    async close({ keepFrames = true } = {}) {
      if (!session) return null;
      const sid = session;
      session = null;
      try { return await jsonPost("/sam/track/close", { session: sid, keepFrames }); } catch { return null; }
    },
  };
}
