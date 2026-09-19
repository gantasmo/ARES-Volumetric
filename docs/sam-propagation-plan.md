# Masklet: SAM 3 video-tracker propagation over frozen-camera proxy renders

## Verdict

DESIGN 1 (Masklet) wins. I re-verified its load-bearing facts myself rather than trusting the judges. Reading model.safetensors' header directly: 1797 tensors, 538 under detector_model.vision_encoder, ZERO under tracker_model.vision_encoder. modeling_sam3_tracker_video.py:703 sets base_model_prefix="tracker_model" and line 1602 sets _keys_to_ignore_on_load_unexpected=[r"^detector_model."], so Sam3TrackerVideoModel.from_pretrained("facebook/sam3") really does silently random-init its whole vision tower. modeling_sam3_video.py:511 builds the tracker with remove_vision_encoder=True, and modeling_sam3_tracker_video.py:1909 checks inference_session.cache BEFORE ever touching that None encoder, with sam3_video/modeling_sam3_video.py:1630 populating exactly that cache. So the towerless-tracker seam is the library's own internal path, not a workaround, and Sam3VideoModel is the only 0-missing load. Confirmed installed: transformers 5.13.0 with sam3, sam3_tracker, sam3_tracker_video, sam3_video, sam2, sam2_video all present. The design doc's blocker at docs/editor-v2-design.md:280-284 is stale.

GRAFTED IN FROM DESIGN 3, all four verified: (1) The bracket() fix, which is strictly better than Design 1's. I read edits.ts:345-356: `if (r.interp === "hold") t = 0` runs at line 347, ABOVE the `if (t === 0) return evA` fast path at 353, so Design 1's `if (t === 1)` patch cannot reach hold. bracket() returns t=1 at every dense keyframe (loop condition `f >= a.frame && f <= b.frame`), so a dense hold range today returns the PREVIOUS frame's mask at every frame. Changing the loop to `f < b.frame` fixes the 2x double-compile AND the hold lag in one character. (2) The 3D nearest-surface audit as a drift signal, scoped to one frame of lookback as an auditor and never as a propagator. Design 1's only signal is the tracker grading itself, and its own failure-modes section concedes that misses the case that matters: a confident lock onto the wrong limb after a crossing. (3) validateMasks plus `ares verify-edits`, because parseEditList (edits.ts:202-225) validates nothing about volumes and edits.ts:261 silently no-ops a mask2d missing `kind`. (4) The rebaseEditList clamp on derived keyframes.

GRAFTED FROM DESIGN 2: only the multi-view anchor union, which costs nothing because prepareKeyframe already unions a keyframe's volumes (edits.ts:299-307). Its propagator is rejected: docs/targeted-temporal.md:44-48 records this repo's own measurement that nearest-point pull is scoped to STATIC spans, and :58-61 records contact-region melting. It is also hard-dependent on a source frames folder, so it delivers nothing for a clip brought in through /import-ares (serve.mjs:944), which this very branch just shipped.

DELETED AS PROVEN FALSE. Design 3's Shift+S binding is dead on arrival: apps/demo/main.js has `case "s": case "S": clickTool("sam")` and Shift+S yields e.key === "S". K and N are genuinely unbound; the plan uses those. Design 3's empty-mask policy is fatal: commitSamSelection sets r.mode = "keep" for isolate (main.js:2601), and keepPredicateAt (edits.ts:471-478) drops every point outside all active keep regions, so an all-zero keep mask deletes the entire frame. Design 1's per-run range split is correct, because with no range active keepPredicateAt returns null and the frame survives untouched. Design 3's rleB64 second mask encoding is dropped: it permanently forks the wire format and needs a second Python implementation, when compact serialization alone already fits.

TWO THINGS I SETTLED THAT NO JUDGE DID. First, I ran the probe both SAM designs listed as their number one risk: Sam3TrackerVideoProcessor.from_pretrained(SAM3_DIR) LOADS CLEANLY offline in the installed env, target_size 1008, with Sam3ImageProcessor plus Sam2VideoVideoProcessor and post_process_masks present. That risk is gone and step 2 shrinks. Second, the ortho bug is worse than reported: player.getCamera() at player.ts:164 RETURNS ortho, main.js:2440 spreads it into samSel.cam, and P toggles it live at main.js:3488, so every SAM mask captured in orthographic is already being region-tested through a perspective frustum today. Design 2 runs its one view-dependent step straight through that code path and never mentions it.

## Approach

Freeze the orbit camera, render the clip's frames in the browser as a synthetic video, drive SAM 3's video tracker over them in the resident local service, and write one `derived: true` mask2d keyframe per frame into the existing edit range. Nothing new reaches the bake: the propagated artifact is the same `{type:"mask2d", kind:"bitmap", mask, camera, depth}` volume the click, lasso and marquee already author, so keepPredicateAt, filterFrame, recolor, paint, sculpt and hole capping all inherit a time-varying region with zero encoder changes. The service loads one `Sam3VideoModel` (the only 0-missing-tensor load path), serves /segment_text off its own detector_model, grafts that tower into the image tracker for /segment exactly as main.py:180 already does, and gets tracker_model plus tracker_neck for +19.5 M params, roughly 38 MiB in fp16, for the entire memory bank. The towerless tracker is fed by priming its vision-feature cache, which is the sequence Sam3VideoModel runs internally at modeling_sam3_video.py:1613-1632. No new serve.mjs route is needed because /sam/track/* falls through the existing wildcard proxy, which pipes both directions and omits Content-Length when upstream does, so a FastAPI StreamingResponse streams uncut. The network runs once, at authoring time; its output is thresholded to a binary RLE bitmap on disk, so the bake is pure arithmetic on integers and doubles with no GPU, no torch and no network, forever. Occlusion gaps split the track into one range per contiguous run rather than emitting empty masks, drift is flagged by two independent estimators disagreeing (tracker score versus a 3D nearest-surface transfer of the previous frame's selection), and a correction re-propagates only to the next user keyframe.

## Prerequisites

- Installed SAM env is transformers 5.13.0 / torch 2.6.0+cu124, NOT the requirements.txt pins (5.16.1 / 2.14.0). Every signature in this plan was read from the installed tree. Check with: tools/sam-service/env/Scripts/python.exe -c "import transformers,torch;print(transformers.__version__,torch.__version__)". Do NOT rebuild the env to the pins before step 2 records a mask fixture, because that is an untested change to exactly this API surface.
- The facebook/sam3 checkpoint is present at C:\\Users\\dtruj\\.cache\\huggingface\\hub\\models--facebook--sam3\\snapshots\\3c879f39826c281e95690f02c7821c4de09afae7 with model.safetensors (3,439,938,512 bytes). Check with: python -c "import struct,json;f=open(PATH,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));print(len(h))" and expect 1797. main.py:45-80 already resolves this path via the HF hub cache branch.
- Sam3TrackerVideoProcessor.from_pretrained(SAM3_DIR) loads offline. ALREADY VERIFIED by me in this session: returns target_size 1008 with Sam3ImageProcessor + Sam2VideoVideoProcessor and post_process_masks present. Re-check only if the env is rebuilt.
- The vit_h fallback is DEAD on this machine and must not be assumed. main.py:84-87 defaults VITH_CHECKPOINT to models/sam_vit_h_4b8939_fp16.safetensors; only sam_vit_h_4b8939.pth exists (2,564,550,879 bytes). Check with: ls tools/sam-service/models/. With SAM_BACKEND=auto a sam3 failure leaves both backends failed. Step 4 changes the default to the .pth that exists.
- packages/encoder/dist/cli.js must be built before any bake or CLI verb: run `npm run build` (tsc -b). The test script is `tsc -b && node --test "packages/*/test/*.test.mjs"`, so tests import from ../dist/index.js and @ares/core.
- Confirm no shipped sidecar relies on the interp:"hold" one-frame lag before step 1 lands: grep -l '"interp"' apps/demo/*.edits.json and inspect any hit. Step 1 changes hold behavior at an exact keyframe from returning the previous keyframe to returning that keyframe.

## Steps

### Step 1: Evaluator fix pack: bracket() exactness, ortho passthrough, mask validation

ships_alone: true
Files: packages/core/src/edits.ts | packages/encoder/src/cli.ts | packages/core/test/edits-dense.test.mjs | packages/encoder/test/editor.test.mjs

Four changes, all of them live-bug fixes that dense keyframes turn from latent into load-bearing, plus the validator every later step leans on.

(a) edits.ts bracket(), line 316. Change the loop condition `if (f >= a.frame && f <= b.frame)` to `if (f >= a.frame && f < b.frame)`. I traced this: with keyframes on every frame, f === b.frame matches at index i with t = 1, so prepareRangeSdfAt compiles and probes BOTH bracketing keyframes per centroid (a clean 2x cost and 2x resident decoded-mask memory), and worse, `if (r.interp === "hold") t = 0` at line 347 runs ABOVE the `if (t === 0) return evA` fast path at 353, so a dense hold range returns the PREVIOUS frame's mask at every frame. With `f < b.frame`, f falls to the next interval where a is the exact keyframe and t = 0. Sparse behavior is unchanged: keyframes at 0 and 10 still give t = 0.5 at f = 5, and f = 10 still hits the `f >= last.frame` early return. This IS a behavior change for interp:"hold" at an exact keyframe and needs its own CHANGELOG line under a behavior-change heading, not folded into the feature entry.

(b) edits.ts prepareVolume, line 262-264. The OrbitState literal is built as `{azimuth, elevation, distance, target}` and drops `ortho`, which camera.ts:141-142 honors. This is live today, not latent: player.getCamera() at player.ts:164 returns `ortho?: boolean`, main.js:2440 spreads it into samSel.cam, and P toggles ortho at main.js:3488. Add `ortho: v.camera.ortho` to the literal, and add `ortho?: boolean` to Mask2dVolume['camera'] (which is a CLOSED object type: the `[k: string]: unknown` index signature at edits.ts:36 sits on the volume, not on camera, so the demo is already writing an undeclared field).

(c) edits.ts: export `validateMasks(list: EditList): string[]`. Per range, per keyframe, per mask2d volume it checks: `kind` present (edits.ts:261 gates on it and returns null silently otherwise), `mask.rle` run sum exactly equals width*height, `camera` present, `camera.aspect` within 1% of width/height, keyframe.frame within [startFrame, endFrame], no two keyframes sharing a frame. Returns human-readable strings, never throws.

(d) cli.ts: after the parseEditList call at line 319, run validateMasks and console.warn each issue prefixed `[ares] edits:`. Add an `ares verify-edits <file.edits.json>` verb to the dispatch and USAGE that prints per-range keyframe counts, the user/derived split, mask resolution, and every validateMasks line, exiting 1 when any issue is found. Also make rebaseEditList (cli.ts:196-203) DROP keyframes with `derived === true` whose rebased frame falls outside [0, n-1], while keeping every user keyframe exactly as today. The existing no-drop rule exists because dropping an interpolation anchor reshapes surviving frames (cli.ts:185-190); that reasoning inverts under dense keyframes, where every surviving frame already carries its own keyframe.

**Test:** packages/core/test/edits-dense.test.mjs (new): (1) a range with a distinct bitmap on every frame of [0..31] and no interp resolves at frame f to bitmap f, asserted by probing a world point that is inside only bitmap f; (2) the same range with interp:"hold" resolves to bitmap f, not bitmap f-1 (fails before the change); (3) a mask2d with camera.ortho true selects a different and correct triangle set than the same volume with ortho absent (fails before the change); (4) a sparse two-keyframe range still lerps at t=0.5. packages/encoder/test/editor.test.mjs: add validateMasks cases for a missing kind, a truncated rle whose runs do not sum to width*height, a duplicate keyframe frame, and an aspect mismatch. Run: npm test.

### Step 2: Service probe and track smoke CLI: measure what the design doc guesses

ships_alone: true
Files: tools/sam-service/track_smoke.py | docs/editor-v2-design.md

A read-only CLI, never run by CI, that settles the numbers before any UI is built on them. It replaces the stale blocker at docs/editor-v2-design.md:280-284 (the 24 GB 4090 OOM was the facebookresearch loader, which materializes the whole video on device; nothing about it applies to this path).

Part 1, the load probe. Construct Sam3VideoModel.from_pretrained(SAM3_DIR, dtype=SAM_DTYPE) and assert len(missing_keys) == 0, vm.tracker_model.vision_encoder is None, type(vm.detector_model).__name__ == 'Sam3Model', vm.tracker_model.num_maskmem == 7, vm.low_res_mask_size == 288. Then assert the negative control: Sam3TrackerVideoModel.from_pretrained(SAM3_DIR) reports 538 missing vision_encoder keys. That negative control is the whole reason this design loads the video model rather than the tracker, and it should be provable on demand.

Part 2, the seam. Wire the three-call sequence copied from modeling_sam3_video.py:1613-1632: vision_embeds = vm.detector_model.get_vision_features(px); feats, pos = vm.get_vision_features_for_tracker(vision_embeds); session.cache.cache_vision_features(f, {'vision_feats': feats, 'vision_pos_embeds': pos}). Then run vm.tracker_model(inference_session=session, frame_idx=f) and confirm _prepare_vision_features (modeling_sam3_tracker_video.py:1909) took the cache branch rather than dereferencing the None encoder.

Part 3, the measurements. Given a directory of numbered JPEGs plus a click coordinate: seed with Sam3TrackerVideoProcessor.add_inputs_to_inference_session, propagate forward and reverse, write per-frame PNG masks plus timings.json holding ms/frame split into encoder and head, torch.cuda.max_memory_allocated, and host RSS. Run the whole thing twice, once in fp16 and once in bf16, and write per-frame IoU between the two mask stacks. fp16 is off Meta's documented path for the video tracker (every model-card video example uses bfloat16) and no published fp16 video validation exists, so this diff is the gate on shipping fp16 on this sm_75 card. Note the memory bank is hardcoded bf16 regardless of session dtype (modeling_sam3_tracker_video.py:2701) and cast back at consumption (:2512), so only one elementwise store cast per frame is affected.

Also rewrite docs/editor-v2-design.md 8.4 with what is actually installed and what was actually measured, and correct 6.2 line 177, which promises a distance-transform SDF lerp for mask2d that the binary branch at edits.ts:288 does not implement.

**Test:** No node test: this is a measurement tool and CI must never load 3.4 GB of weights. Its own assertions (0 missing keys, tracker vision_encoder is None, 538 missing on the negative control, cache branch taken) are the test, and they run from the CLI. Record timings.json and the fp16-vs-bf16 IoU in the docs edit so the numbers are reviewable.

### Step 3: Player capture kit: frame-indexed seek, clip fps, texture settle, blob capture

ships_alone: true
Files: packages/core/src/player.ts | apps/demo/main.js

Four small additive methods, no behavior change to anything existing, each useful on its own.

(a) `seekFrame(idx: number): void`. The body of seek() at player.ts:850 keyed on a frame index instead of seconds. A propagation sweep must not round-trip through a float division.

(b) `getClipFps(): number` returning this.file.header.fps, which is read privately at player.ts:420, 802 and 888 and exposed nowhere. PlayerStats.fps (player.ts:992) is the rolling RENDER-rate EMA, not the clip's fps, which is why the demo hardcodes /30 at main.js:1663 (tlSeek) and main.js:3441 (the keyboard step). Fix both call sites in the same change: any non-30 fps clip currently seeks to the wrong frame.

(c) Extend textureDebug() (player.ts:964) from {hasVideo, applied, error} to {hasVideo, applied, appliedIdx, settled, error}, where appliedIdx exposes the private texAppliedIdx (player.ts:694) and settled is `!this.textureVideo || this.texAppliedIdx === <presented frame>`. seek() is synchronous for geometry but pumpTexture (player.ts:699-720) settles the atlas over up to 30 rAF retries, so without this a sweep hands SAM frame N's mesh wearing frame N-k's atlas and there is no way to know.

(d) `captureFrameBlob(maxDim = 1008, type = 'image/jpeg', quality = 0.95): Promise<Blob | null>`. renderCurrent() then canvas.toBlob. captureFrame (player.ts:388) returns a PNG data URL, which at 272 frames is roughly 400 MB of base64 plus 15-30 ms of main-thread deflate per frame.

**Test:** packages/core/test/player-api.test.mjs (new): assert the four symbols exist on the prototype with the right arity, and that seekFrame(n) and seek(n / getClipFps()) resolve to the same frameIndex on a synthClip-built file. Manual check the owner can see: load a clip, drag the timeline, confirm the playhead lands on the same frame it did before, then check a non-30 fps clip now seeks correctly where it previously did not.

### Step 4: Service video tracker and track routes

ships_alone: true
Files: tools/sam-service/track.py | tools/sam-service/main.py | tools/serve.mjs

tools/sam-service/track.py (new, keeps main.py from growing past its 724 lines) holds: the lazy _ensure_tracker_video() with a latched _track_error following the /upscale and /detail discipline (main.py:546, 633), so a tracker failure 503s in isolation and never touches /segment or /segment_text; a TrackSession store (id, clip key, frozen-camera hash, trackRes, maskRes, the uploaded JPEGs as dict[int, bytes], the Sam3TrackerVideoInferenceSession, the obj-id map, a cooperative cancel flag, last_touch) capped at 2 live sessions with a 10-minute idle reaper; preprocess(jpeg) on a one-frame-ahead worker thread so CPU decode hides under GPU time; seed() honoring the library's documented exclusions (masks are mutually exclusive with points and boxes, processing_sam3_tracker_video.py:611-613; a box cannot use clear_old_inputs=False, :702-707); propagate() as a generator yielding SSE event dicts; and mask_to_rle() emitting exactly the edits.ts convention (alternating run lengths, 0-run first) after threshold-at-0 and box downsample to maskRes.

CRITICAL, verified by reading processing_sam3_tracker_video.py:736: add_inputs_to_inference_session ends with `inference_session.obj_with_new_inputs = obj_ids`, an ASSIGNMENT not an append. Every object of a frame must be seeded in ONE call followed by one forward at that frame, or the first object's pending prompt is silently dropped.

main.py changes, all additive: add a `sam3_video` entry first in the _loader order map at line 235 ({"auto": ["sam3_video", "sam3", "vit_h"]}). _load_sam3_video() loads Sam3VideoModel (1797 tensors, 0 missing), sets _sam3_concept_model = vm.detector_model (a genuine Sam3Model, so _segment_text_sam3 at :406 is untouched), and loads Sam3TrackerModel for the click path grafting vm.detector_model.vision_encoder into it, which is the graft already at :180 still guarded by _assert_vision_encoder_shared (:193). On any failure it falls through to today's exact sam3 path with video disabled, preserving the fault isolation the loader comment at :253-259 exists to protect. Add `_track_lock = threading.Lock()` beside _lock (:123): a run holds _track_lock for its duration but takes _lock only around each single-frame forward, so an interactive /segment click interleaves at one frame's latency instead of blocking for minutes. Extend /health (:326) with trackReady, trackError, trackLoading, trackSessions, trackRes. Change the VITH_CHECKPOINT default (:84-87) from the nonexistent sam_vit_h_4b8939_fp16.safetensors to sam_vit_h_4b8939.pth, which is the file that actually exists, so the documented fallback stops being dead.

tools/serve.mjs, exactly two edits and no new route. (1) Line 502 GUARDED: `sam\/start` becomes `sam\/(start|track\/)`. Verified that /sam/track/* otherwise falls through the wildcard proxy at :659, which forwards any method and path, pipes both directions, and copies Content-Length only when upstream sent one, so a FastAPI StreamingResponse streams uncut. (2) Line 709, the /edits POST cap: 4_000_000 becomes 16_000_000 and overflow answers 413 with {error, bytes, cap} instead of req.destroy(), which today sends no response at all while main.js:1048's .catch(() => {}) swallows it. Add the file's mandatory why-comment with the measured numbers. Note the one-deep .bak at :716 now backs up a multi-megabyte file: keep it, but the range row in step 5 surfaces size so a user is never surprised.

**Test:** packages/encoder/test/mask-rle-parity.test.mjs (new): a golden-vector test pinning the Python encoder to rleEncodeMask semantics. The JS side holds a fixed 64x64 pattern and its expected run array; a checked-in fixture JSON produced by track.py's mask_to_rle over the same pattern must decode through rleDecodeMask to a byte-identical bitmap. CI needs no GPU. Route smoke, run by hand: curl -X POST http://127.0.0.1:8137/sam/track/open with a body, confirm a session id; POST a frame; POST a prompt; curl the SSE run and confirm one mask event per frame.

### Step 5: Forward track: proxy sweep, keyframe writer, timeline span

ships_alone: true
Files: apps/demo/track.js | apps/demo/main.js | apps/demo/index.html | apps/demo/settings.js

apps/demo/track.js (new, on the menu.js precedent so main.js does not grow) owns: the pre-track gates, the camera and viewport freeze, the sweep, the upload loop, the SSE consumer, and the readout strings.

GATES, all refusals before any GPU work, because each would silently produce wrong masks on every frame. A non-identity model transform: pickRaster (player.ts:359) and prepareVolume (edits.ts:262) project through bare orbitViewProj while both renderers draw through viewProj x model (renderer.ts:757-759, renderer-gl2.ts:820), so the image SAM segments and the region test disagree. Non-identity fx (isFxIdentity, fx.ts:59): displaces vertices and discards fragments in the shader only. A splat clip: exportFrame and pickRaster return null (player.ts:274, 357).

FREEZE: player.autoOrbit = false; save and clear grid via isGrid()/setGrid(false) (grid and tripod are drawn into the same pass, renderer.ts:930-968, and a static world grid under a moving subject is exactly the stable distractor a memory bank latches onto); save and clear the live edit preview via setEditPreview(null), which is mandatory because pickRaster and the render both read the edit-filtered curIndices, so tracking with a live delete range active would render the subject's own hole and track it. Record shade mode and restore everything in a finally.

SWEEP, per frame: seekFrame(f); await textureDebug().settled bounded to 8 rAF, then proceed and log `texture lag`; captureFrameBlob(trackRes, 'image/jpeg', 0.95); batch 16 frames per binary POST.

WRITER, in main.js as editorApi.commitTrack(): ONE doMutation (main.js:1095) for the whole span so the track is one undo step. It writes keyframes directly into range.keyframes and must NOT go through addVolumeAtCurrent (main.js:2102), which PUSHES into an existing keyframe (so a re-track would union with a correction) and calls preview() per volume (so 272 calls would fire 272 previews and 272 save timers). Seed frame gets a user keyframe with `derived` absent; every other frame gets {frame, derived: true, volumes: [vol]}. Leave `interp` absent: linear with a keyframe on every frame is exact after step 1, and hold buys nothing.

UNDO COST, needed here because this is where dense ranges first land. snapshotState (main.js:1073-1079) deep-clones the whole editable state through JSON and pushUndoIfChanged (main.js:1087) stringifies it TWICE per mutation, with UNDO_CAP 50 (main.js:1068). Replace with cloneStateForUndo/sameState that deep-clone everything EXCEPT mask objects, which are passed by reference and compared by identity. The invariant this needs is narrow and holds today: no code mutates a mask object in place. growKeyframe mutates v.min, v.rect and strokes[].radius in place but REPLACES v.mask (edits.ts:414) precisely so the decode WeakMap is not stale. Add a comment at edits.ts:414 recording that undo now depends on it, plus a test.

SIDECAR: change doSave (main.js:1048) to serialize compact when the document contains any derived keyframe and keep JSON.stringify(edits, null, 1) otherwise. Pretty-printing costs a measured 4.1x at RLE nesting depth; making it conditional keeps small hand-authored sidecars inspectable while letting a tracked one fit.

TIMELINE: renderRanges (main.js:1786) must stop emitting one absolutely-positioned .tlKf span per keyframe for a tracked range. At 272 diamonds in a 12px lane that is roughly one glyph per 1.4px and nothing is clickable. Render a hatched span bar plus a filled diamond at each user keyframe only; the k.derived branch at main.js:1805 already picks the right glyph for the ones that remain.

UI: one new row inside the existing data-sec="sam" section (index.html:924) after #samSelRow, which is already flex-wrap, so the rail gains one row and .railBody (index.html:201) stays the only scrolling container. Key K for Track, registered in all three required places: the KEYS table (main.js:3310), the keydown switch (main.js:3443) and the #hint tooltip (index.html:1162). K and N are verified unbound; Shift+S is NOT, because `case "s": case "S"` already claims it. settings.js:364 #depSam takes the same tracker flag as the rail so the two health readouts finally agree.

**Test:** packages/encoder/test/track-commit.test.mjs (new), pure and headless: build a 40-frame edit list the way commitTrack does, assert exactly one keyframe per frame, exactly one non-derived keyframe at the seed, keyframes sorted and unique, every volume carrying kind:"bitmap" and the same camera object, and that keepPredicateAt resolves frame f to bitmap f. Also assert the undo clone: cloneStateForUndo shares mask object identity while deep-cloning ranges, and sameState reports no-change for an untouched document. Manual: track a clip, scrub, confirm the tint follows the subject and the timeline shows one hatched span rather than 272 diamonds.

### Step 6: Bidirectional, occlusion gaps, and the correction loop

ships_alone: true
Files: apps/demo/track.js | apps/demo/main.js | tools/sam-service/track.py

DIRECTION: forward, reverse and bidi. Reverse is a second call, never a flag on the first. propagate_in_video_iterator with reverse=True yields range(start, end-1, -1) INCLUSIVE of the start frame (modeling_sam3_tracker_video.py:2808-2816) and short-circuits to [] when start is 0. ALWAYS pass start_frame_idx explicitly: leaving it None defaults to min() over every conditioning frame (:2795-2805), which silently re-runs the whole clip after a correction.

GAPS: when object_score_logits <= 0 or the thresholded area is 0, emit a `gap` event and write NO keyframe. Then split the track into ONE RANGE PER CONTIGUOUS RUN, sharing a trackId. This is not cosmetic, it is the correctness fix for the isolate case: commitSamSelection sets r.mode = "keep" for isolate (main.js:2601), and keepPredicateAt (edits.ts:471-478) drops every point outside all active keep regions, so a single empty keep keyframe would black out the entire frame. With the range simply not spanning the gap, keepPredicateAt returns null and the frame survives untouched (verified at edits.ts:470). The precedent for never inventing a mask across a gap is docs/rgbd-rebuild-pipeline.md:93-98, where backfilling an empty mask poisoned a 64-frame batch.

CORRECTION, two operations, both bounded to [correctedFrame, nextUserKeyframe - 1]. Nudge: click or shift-click, then re-run with clear_old_inputs=False so the new point accumulates rather than replacing (the parameter defaults to True, processing_sam3_tracker_video.py:576). Re-seed: allocate a fresh objId and seed it here with clear_old_inputs=True. The distinction is real and comes from the library: forward computes is_init_cond_frame = frame_idx not in frames_tracked_per_obj[obj_idx] (modeling_sam3_tracker_video.py:1801), so a click on an already-tracked frame is NOT a conditioning frame and never enters the pool _select_closest_cond_frames draws from (:2185). A fresh objId has no such entry, so Re-seed genuinely restarts the memory bank. That is the AE Roto Brush base-frame semantics expressed in this library's actual API. Escalation, when three Nudges have not held: Re-anchor rebuilds the session from the full seeds list in frame order, one add_inputs_to_inference_session per frame followed by one forward, which makes every user keyframe a true conditioning frame at the cost of re-encoding the window.

Either way the corrected frame is PROMOTED from derived to user (delete derived, delete conf, replace volumes) and appended to range.track.seeds, and only derived keyframes strictly inside the bounded window are replaced. This is Mocha's AdjustTrack rule: corrections are a sparse layer over a dense automatic result, never an in-place overwrite of it. All of it inside one doMutation, so one Ctrl+Z restores the pre-correction track.

HONEST CEILING, surfaced in the UI rather than hidden: max_cond_frame_num is 4 in the checkpoint's tracker_config, so a fifth anchor evicts an earlier one via _select_closest_cond_frames. Past four seeds the readout reads `4 of 5 seeds anchored` and the right move is splitting the range.

SESSION REUSE: the service still holds every uploaded JPEG, so a correction costs inference over the bounded window and zero upload. Keep the frame store keyed by clip + camera hash + trackRes for 10 minutes after close so a Retrack skips the sweep entirely. If the service restarted and returns 404, re-open silently, replay range.track.seeds in order, and re-upload only the frames the window needs.

CANCEL: POST /sam/track/cancel sets a cooperative flag checked between frames, sent BEFORE es.close(), because whether the Node proxy propagates a client disconnect to FastAPI's request.is_disconnected() is not something this tree has ever exercised. Every mask already emitted is already a keyframe, so a cancel leaves a shorter but valid track. Bind cancel to Esc with a precedence branch above the nav fallback (main.js:3489).

**Test:** packages/encoder/test/track-gaps.test.mjs (new): build a track with a gap at [143..147], assert it produces two ranges sharing a trackId with no keyframe in the gap, and assert keepPredicateAt(list, 145) returns null when no other range is active (the isolate blackout regression). Also assert a correction at frame 204 replaces only derived keyframes in [204, nextUser-1], promotes 204 to a user keyframe, leaves every other user keyframe untouched, and appends exactly one seed. Manual: track, scrub to a drifted frame, Nudge, confirm the repair stops at the next user keyframe.

### Step 7: Drift audit: two independent estimators, conf, and the jump-to-worst key

ships_alone: true
Files: packages/core/src/surface-transfer.ts | packages/core/src/index.ts | packages/encoder/src/temporal.ts | apps/demo/track-worker.js | apps/demo/track.js | apps/demo/main.js

The one thing Design 1 lacked and its own failure-modes section conceded: its only drift signal is the tracker grading itself, which is blind to a confident lock onto the wrong limb after an occlusion crossing.

packages/core/src/surface-transfer.ts (new) lifts closestOnTriangle (the Ericson kernel at temporal.ts:52) and TriangleGrid (temporal.ts:110) out of the encoder into core, and temporal.ts then imports them instead of keeping private copies. This deduplicates code that already exists twice and creates exactly the primitive the clothing and hole-reconstruction roadmap items need. Exports buildTriangleGrid(positions, indices) and transferLabels(gridA, labelsA, positionsB, indicesB) returning a Uint8Array: for each triangle of mesh B, the label of the nearest surface point on mesh A.

apps/demo/track-worker.js (new) holds exactly ONE frame of lookback. Per frame it takes exportFrame() output plus the previous frame's selected triangle set, transfers labels, rasterizes the prediction into the frozen camera with rasterizeIds (raster.ts:28) at mask resolution, and IoUs it against SAM's mask. conf = min(sigmoid(object_score_logits), iou), stored on EditKeyframe.conf.

SCOPE DISCIPLINE, and this is why this is an auditor and not a propagator. docs/targeted-temporal.md:44-48 records this repo's own measurement that nearest-point pull collapses protruding features under motion or accumulated chain drift and is scoped to STATIC spans only, and :58-61 records melting where surfaces come close. One frame of lookback with no chaining stays inside that envelope; 272 frames of chaining does not, which is why Design 2's propagator is rejected. Roughly 31 ms per frame in a worker against 130 to 365 ms of GPU on the next frame, so it is free.

UI: a caret in var(--warn) under any keyframe with conf < 0.60, and the N key jumping to the lowest-conf frame not yet corrected. That is XMem++'s annotation-candidate selector built from parts already in the tree, and it is the difference between a propagation and a workflow. Gate the caret on conf < 0.60 AND a non-trivial mask area, and treat the threshold as provisional: the auditor's own known failure is contact regions, so validate it against a hand-labelled span before promising it in docs. Also apply the repo's own instrumentation rule from docs/targeted-temporal.md:27 and gate the summary statistic on MAX, not p95.

Add a temporal denoise pass over the mask stack before the keyframes are written: a close-then-open per frame plus a 3-frame temporal median, using morphBitmap (edits.ts:361), which is already deterministic and already exported. SAM's memory bank gives inter-frame coherence but its boundary still flickers a pixel or two on a high-frequency silhouette, and on a delete range that reads as a shimmering cut edge. Erode the mask 2px before computing the depth band, so a single mask-edge pixel landing on background surface cannot blow zmax out to background depth and silently disable the visible-only gate (edits.ts:279-282).

**Test:** packages/core/test/surface-transfer.test.mjs (new): two synthetic meshes offset by a known translation, assert transferLabels moves a labelled patch to the geometrically corresponding triangles and that a patch further than the gate is not transferred. packages/encoder/test/editor.test.mjs: assert temporal.ts still produces identical GOP plans after the refactor (a byte-compare of buildTemporalGops output on a fixture, proving the lift changed nothing). packages/core/test/edits-dense.test.mjs: assert the 3-frame temporal median removes a single-frame dropout and leaves a genuine two-frame change intact.

### Step 8: Text-seeded tracks, multi-instance, and resolution tiers

ships_alone: true
Files: apps/demo/track.js | apps/demo/main.js | tools/sam-service/track.py | AUDIT.md | docs/editor-v2-design.md | CHANGELOG.md

TEXT SEEDING, routed through the tracker rather than the concept model. Sam3VideoModel (the PCS path) has NO click-refinement API at all: Sam3VideoProcessor exposes only __call__, add_text_prompt, init_video_session, three non-overlap helpers and postprocess_outputs. So a text-seeded track that could never be corrected is a dead end. Instead run the existing single-frame /segment_text on the seed frame, let the user pick instances with the chips already at main.js:2682, and seed each chosen instance into the tracker via add_inputs_to_inference_session(input_masks=...). One propagation path, one refinable model, and text-seeded tracks are correctable by click. This is the missing middle of the clothing-region op at AUDIT.md:129: text prompt (already shipped) then propagation (this) then recolor (already shipped bake-side).

MULTI-INSTANCE: N objects in ONE session, seeded in ONE add_inputs_to_inference_session call per frame (obj_with_new_inputs is assigned, not appended, processing_sam3_tracker_video.py:736). Cost scales sub-linearly because the vision encoder runs once per frame regardless of object count and only the head loops serially (modeling_sam3_tracker_video.py:1780-1783, deliberately serial because per-object prompts differ; only _batch_encode_memories at :2706 is batched): roughly +12% for a second object, not +100%. Each object becomes its own range group keyed by trackId and objId.

RESOLUTION TIERS: trackRes 1008 (the checkpoint's native size, per processor_config.json) or 560 for draft. Encoder compute scales as (560/1008)^2 = 0.309, so draft is roughly 2.5 to 3x faster at a documented accuracy cost. This reproduces the professional proxy-track workflow: track cheap, find drift with N, correct, re-run final. Store maskRes separately from trackRes (768 default) so mask fidelity and inference cost are independent knobs.

RANGE ID COLLISION, worth fixing here because every track keys to a range id: ensureRange assigns "r" + (edits.ranges.length + 1) (main.js:2096), which collides after a deletion, and restoreState re-finds activeRange by id (main.js:1123), so a collision makes undo re-target the wrong range. Switch to a monotonic counter stored on the document.

DOCS: close AUDIT.md:111 and the Not-done tails at :207 and :210. Correct docs/editor-v2-design.md 11 line 359, which documents the mask as {encoding, width, height, data} against the shipped {width, height, rle}. Rewrite the SAM section tooltip at index.html:925, which currently promises "the range tweens between them": a binary bitmap SDF lerp is a hard crossfade that pops at t=0.5, not a tween. Fix the two mis-pointing SAM strings while in main.js: line 2434 says "press Start SAM below" for a button that is ABOVE it (index.html:930) and labelled "Start", and line 2632 is two sentences of conversational voice with a contraction. Add a THIRD-PARTY-NOTICES.md line for the SAM License (Meta's custom license, commercial use permitted with acknowledgement, export-control and patent-retaliation terms), which is not Apache-2.0 like SAM 2.

**Test:** packages/encoder/test/track-multi.test.mjs (new): assert a two-object track produces two range groups with distinct objId and shared trackId, that ranges from different objects can overlap in frame span without interfering in keepPredicateAt, and that a mask-seeded range validates clean through validateMasks. Assert the new range-id allocator produces no duplicate after a delete-then-create sequence. Manual: text-prompt a garment, pick an instance chip, track, correct with a click, bake, and confirm the recolor lands on the garment across the clip.

## Sidecar types

// ---- packages/core/src/edits.ts ------------------------------------------------------------
// Three additions and one correction. All bake-inert except `camera.ortho`, which is a fix:
// player.getCamera() (player.ts:164) already RETURNS ortho, main.js:2440 already spreads it into
// samSel.cam, and P toggles it live (main.js:3488), so every ortho-captured mask is being
// region-tested through a perspective frustum today. Note `camera` is a CLOSED object type: the
// `[k: string]: unknown` index signature at edits.ts:36 sits on the VOLUME, not on camera, so the
// demo is already writing an undeclared field and this makes the type describe reality.

export interface Mask2dVolume {
  type: "mask2d";
  kind?: "rect" | "bitmap";
  rect?: [number, number, number, number];
  mask?: { width: number; height: number; rle: number[] };
  camera?: {
    azimuth: number; elevation: number; distance: number;
    target: [number, number, number]; aspect: number;
    /** Orthographic capture (camera.ts OrbitState.ortho). MUST reach orbitViewProj or the region
     *  is tested through a projection the mask was never drawn in. */
    ortho?: boolean;
  };
  depth?: { zmin: number; zmax: number };
  [k: string]: unknown;
}

export interface EditKeyframe {
  frame: number;
  /** Machine-generated by a propagation run; regenerable, a re-track may replace it. Rendered as
   *  a hollow diamond (apps/demo/main.js:1805). */
  derived?: boolean;
  /** Propagation confidence 0..1 = min(sigmoid(tracker object score), IoU against the 3D
   *  nearest-surface transfer of the previous frame's selection). Advisory: no evaluator reads
   *  it. Drives the timeline caret and the N key. Absent on user keyframes. */
  conf?: number;
  volumes: EditVolume[];
}

/** One prompt applied to a track, in application order. Replaying `seeds` in order against the
 *  same checkpoint and the same proxy renders reproduces the track. */
export type TrackSeed =
  | { frame: number; objId: number; kind: "points"; mode: "seed" | "nudge";
      /** Proxy-render pixels (TrackProvenance.proxy), NOT capture-image pixels. */
      points: [number, number][]; labels: (0 | 1)[] }
  | { frame: number; objId: number; kind: "box"; mode: "seed"; box: [number, number, number, number] }
  | { frame: number; objId: number; kind: "mask"; mode: "seed";
      mask: { width: number; height: number; rle: number[] }; text?: string; instance?: number };

/** How a range's derived keyframes were produced. Pure provenance: nothing in the evaluator, the
 *  preview or the bake reads it. It exists so a propagated span is auditable and re-runnable
 *  after the service has restarted and its session is gone. */
export interface TrackProvenance {
  tool: "sam3-tracker-video";
  /** Resolved checkpoint path or hub id, as GET /sam/health reports in `weights`. */
  checkpoint: string;
  /** Tracker input edge: 1008 is the checkpoint's native size, 560 is the draft tier. */
  trackRes: 560 | 768 | 1008;
  /** Long side of the STORED bitmaps; masks are downsampled to this before RLE. */
  maskRes: number;
  dtype: "bfloat16" | "float16" | "float32";
  /** The frozen proxy render every derived mask was segmented from. Recorded in full because
   *  captureFrame draws the LIVE canvas: shade mode and grid change what SAM sees. */
  proxy: { width: number; height: number; shade: string; grid: false };
  /** The frozen camera. Identical to every derived volume's `camera`, stored once for auditing. */
  camera: NonNullable<Mask2dVolume["camera"]>;
  /** Groups the ranges produced by one track run; a gap splits one object into several ranges. */
  trackId: string;
  /** Tracker object id, stable for the whole session. */
  objId: number;
  /** Contiguous-run index within this object, 0-based. */
  run?: number;
  seeds: TrackSeed[];
  /** Frames the tracker reported no object for, INCLUSIVE pairs. No keyframe exists in these and
   *  no range spans them, so keepPredicateAt returns null and the frame survives untouched. */
  gaps?: [number, number][];
  span: { from: number; to: number };
  ranAt?: string;
}

export interface EditRange {
  // ...every existing field unchanged (edits.ts:70-179)...
  /** Present iff this range's keyframes came from a SAM video-tracker run. Bake-inert. Its
   *  absence is what distinguishes a hand-authored range from a tracked one in the UI. */
  track?: TrackProvenance;
}

/** Validate every mask2d in the document. Returns human-readable issues, never throws.
 *  parseEditList (edits.ts:202-225) validates nothing about volumes and prepareVolume (edits.ts:261)
 *  silently no-ops a mask2d missing `kind`, so a propagated range with a mismatched resolution
 *  bakes as an invisible no-op. This is the loud check that costs one decode per keyframe. */
export function validateMasks(list: EditList): string[];

## Routes

Five routes on the FastAPI service, reached at /sam/track/*. NO new route block in tools/serve.mjs: /sam/track/* already falls through the wildcard proxy at serve.mjs:659, which forwards any method with the query string preserved, pipes the body up with req.pipe(up) at :686 with no size cap, and pipes the response down with ur.pipe(res) at :673 copying only Content-Type and, when present, Content-Length. A FastAPI StreamingResponse sends no Content-Length, so nothing buffers. I verified this by reading the proxy body.

THE ONLY SERVER EDITS. (1) serve.mjs:502 GUARDED: `sam\/start` becomes `sam\/(start|track\/)`. Verified the guard runs at :519 before the proxy and that sameOrigin (:504-510) accepts a same-origin fetch or EventSource. (2) serve.mjs:709: the /edits POST cap rises from 4_000_000 to 16_000_000 and overflow answers 413 with {error, bytes, cap} instead of req.destroy(), which today sends no response while main.js:1048 swallows the rejection.

POST /sam/track/open
  req {"clip":"daniel-v11","camKey":"-0.22400,-0.07800,2040.62,...","frames":272,"trackRes":1008,"maskRes":768}
  200 {"session":"trk_9f2c41a0","trackRes":1008,"maskRes":768,"dtype":"float16","device":"cuda",
       "checkpoint":"C:\\...\\snapshots\\3c879f39...","reusedFrames":0,"loadMs":8120}
      reusedFrames > 0 means a store from an earlier run under the same clip + camKey + trackRes
      survived the 10-minute reaper and the client SKIPS the capture sweep entirely.
  503 {"detail":"video tracker failed to load: <reason>"}   (latched, never retried, never touches
                                                             /segment or /segment_text)
  409 {"detail":"track session limit reached: close trk_1a2b first"}   (cap 2 live sessions)

POST /sam/track/frames?session=<id>
  Content-Type: application/octet-stream. Body is a stream of records with no envelope:
    [u32le frameIdx][u32le byteLen][byteLen bytes of JPEG] ...
  Client sends 16 frames per request. JPEG, not PNG: canvas.toDataURL costs 15 to 30 ms of
  main-thread deflate per frame and loopback bytes are free.
  200 {"stored":16,"total":112,"bytes":24641536}
  400 {"detail":"truncated frame record at offset 3512096"}

POST /sam/track/prompt
  req {"session":"trk_9f2c41a0","frame":0,"clearOldInputs":true,
       "objects":[{"objId":1,"points":[[512,301],[540,388]],"labels":[1,0]},
                  {"objId":2,"mask":{"width":768,"height":432,"rle":[...]}}]}
  EVERY object of a frame MUST arrive in ONE call. Verified at processing_sam3_tracker_video.py:736:
  add_inputs_to_inference_session ends with `inference_session.obj_with_new_inputs = obj_ids`, an
  assignment, so a second call silently drops the first object's pending prompt. The service runs
  one forward at that frame to consume it. Coordinates are proxy-render pixels; the client
  rescales samSel.points from capture-image pixels by trackRes/samSel.capture.width and the service
  passes original_size explicitly.
  200 {"ok":true,"objIds":[1,2],"conditioning":true,"consumedAt":0}
  400 {"detail":"mask prompts cannot be combined with points or boxes on one object"}
  400 {"detail":"box prompt requires clearOldInputs true"}

GET /sam/track/run?session=<id>&start=<int>&dir=forward|reverse&max=<int>
  SSE over frames the session already holds, so a correction needs no upload. Head and frame format
  are the house standard, byte-identical to serve.mjs:1371-1372:
    res.writeHead(200, {...HEADERS, "Content-Type":"text/event-stream", Connection:"keep-alive"})
    `event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`
  No id:, no retry:, no heartbeat, matching every other SSE route in the tree. `progress` fires
  every frame so the stream is never silent for more than about 0.5 s, which keeps it well inside
  Node 22's default 300,000 ms server.requestTimeout that serve.mjs never configures.
  Every precheck that can fail does so BEFORE the head as 400/503 JSON (unknown session, start out
  of range, frames not uploaded), the hard law for SSE routes here.

  event: start     {"session":"trk_9f2c41a0","start":0,"dir":"forward","max":272,"objIds":[1],
                    "trackRes":1008,"maskRes":768,"frames":272}
  event: log       "[track] sam3 video, float16, cuda, 1 object, forward 0 to 271"
  event: mask      {"frame":0,"objId":1,"w":768,"h":432,"rle":[41213,7,505,19,...],
                    "score":0.94,"area":0.0812}
                   `rle` is ALREADY in edits.ts convention (alternating run lengths, 0-run first,
                   row-major, edits.ts:40-50). The client stores the array verbatim: no PNG decode,
                   no re-encode, no translation between the service and the sidecar.
  event: gap       {"frame":143,"objId":1,"reason":"occluded"}   // object_score_logits <= 0
                   {"frame":144,"objId":1,"reason":"empty"}      // area 0 after threshold
  event: progress  {"done":144,"of":272,"msPerFrame":368,"etaS":47}
  event: done      {"frames":272,"masks":268,"gaps":4,"ms":99840}
  event: error     {"message":"cuda out of memory, retry at trackRes 560"}

POST /sam/track/cancel   req {"session":"trk_9f2c41a0"}   200 {"ok":true,"stoppedAt":143}
  Sets a cooperative flag the generator checks between frames; the in-flight frame finishes, then
  the stream sends `done` and ends. Sent BEFORE es.close(), because whether the Node proxy
  propagates a client disconnect to FastAPI's request.is_disconnected() is not something this tree
  has ever exercised.

POST /sam/track/close    req {"session":"trk_9f2c41a0","keepFrames":true}   200 {"ok":true,"freedMB":914}

GET /sam/health (existing, extended at main.py:326)
  ...every existing field..., "trackReady":true,"trackLoading":false,"trackError":null,
  "trackSessions":1,"trackRes":1008

LOCKING. main.py:123 has ONE global lock serializing /segment, /segment_text, /upscale and /detail.
A run under that lock would freeze every interactive click for minutes. Track routes take a new
_track_lock for the whole run, serializing runs against each other, but take _lock only around each
single-frame forward, so an interactive click interleaves at one frame of latency.

## UI strings

- Track
- Retrack
- Nudge
- Re-seed
- Re-anchor
- Cancel
- forward
- reverse
- bidirectional
- range
- clip
- draft 560
- native 1008
- propagation direction from the seed frame
- frame span to track
- tracker input edge; 560 is 2.5x faster and off the checkpoint's native resolution
- propagate the pending selection across the span with the SAM video tracker (K)
- stop the track; keyframes already written are kept (Esc)
- add a click at this frame and re-propagate to the next user keyframe
- new object id at this frame and re-propagate to the next user keyframe
- rebuild the session from every seed and re-propagate the window
- jump to the lowest-confidence frame of the active track (N)
- capture 96/272 · 1008 px
- upload 6.1 MB
- track 143/272 · obj 1 · 0.37 s/f · eta 48 s
- track 272/272 · 114 s · 1008 · 268 kf · 4 gaps
- track 272/272 · 114 s · 1008 · no drift
- track 272/272 · 114 s · 1008 · drift 14 frames < 0.60
- retrack 22/60 · obj 1
- track cancelled · 143 kf
- frame 204 · empty mask
- frame 118 · texture lag
- 4 of 5 seeds anchored · split the range
- camera moved · restore view
- model transform active · reset it before tracking
- fx active · reset fx before tracking
- splat clip · tracking needs mesh triangles
- SAM service not running
- track model loading · retry shortly
- track busy · one run at a time
- track failed: 
- sidecar write failed: 
- sidecar 3.4 MB of 16 MB
- sam3 · fp16 · cuda · video
- sam3 · fp16 · cuda · video off
- track · 2 seeds · 4 flagged · 1008 · obj 1
- derived @ 204 · conf 0.42 · click to jump
- keyframe @ 137 · click to jump
- drift gate tripped @ 174 · conf 0.31
- tracked ranges hold each frame's own mask; interpolation does not apply
- track the SAM selection across the span
- jump to the lowest-confidence frame of the active track
- [ares] track r1: 272 keyframes, 1 user, 4 gaps
- [ares] edits: mask2d at r1 kf 88 has no kind and will never match
- r1 · 272 kf (1 user, 271 derived) · mask2d bitmap 768x432 · conf p50 0.91 p05 0.44 · 3 below 0.60 · 0 empty · ok

## Tests

- packages/core/test/edits-dense.test.mjs (new, step 1): dense keyframes resolve at frame f to bitmap f; interp:"hold" resolves to bitmap f and not f-1 (fails before the bracket() change); a camera.ortho volume selects a different and correct set than the same volume with ortho dropped (fails before the prepareVolume change); a sparse two-keyframe range still lerps at t=0.5; the 3-frame temporal median removes a single-frame dropout and preserves a genuine two-frame change (step 7).
- packages/encoder/test/editor.test.mjs (extended, steps 1 and 7): validateMasks catches a missing kind, an rle whose runs do not sum to width*height, a duplicate keyframe frame, and an aspect mismatch; buildTemporalGops produces byte-identical output after closestOnTriangle and TriangleGrid move to core, proving the lift changed nothing.
- packages/core/test/player-api.test.mjs (new, step 3): seekFrame, getClipFps, textureDebug().appliedIdx and captureFrameBlob exist with the right arity; seekFrame(n) and seek(n / getClipFps()) land on the same frameIndex over a synthClip file.
- packages/encoder/test/mask-rle-parity.test.mjs (new, step 4): golden-vector test pinning track.py's Python RLE encoder to rleEncodeMask semantics. A checked-in fixture produced by the Python side over a fixed 64x64 pattern must decode through rleDecodeMask to a byte-identical bitmap. Needs no GPU and no service.
- packages/encoder/test/track-commit.test.mjs (new, step 5): a committed 40-frame track has exactly one keyframe per frame, exactly one non-derived keyframe at the seed, sorted unique frames, every volume carrying kind:"bitmap" and the same camera object; keepPredicateAt resolves frame f to bitmap f; cloneStateForUndo shares mask object identity while deep-cloning ranges and sameState reports no-change on an untouched document.
- packages/encoder/test/track-gaps.test.mjs (new, step 6): a gap at [143..147] produces two ranges sharing a trackId with no keyframe inside it, and keepPredicateAt(list, 145) returns null when no other range is active. This is the regression test for the isolate blackout: commitSamSelection sets mode:"keep" (main.js:2601) and keepPredicateAt drops everything outside all active keep regions (edits.ts:471-478). Also: a correction at 204 replaces only derived keyframes in [204, nextUser-1], promotes 204 to user, and appends exactly one seed.
- packages/encoder/test/track-bake.test.mjs (new, step 6): BAKE DETERMINISM. keepPredicateAt over a 272-keyframe fixture yields a byte-identical kept-triangle index set on two consecutive runs and after a JSON.parse(JSON.stringify(...)) round-trip; rebaseEditList under a trim keeps every user keyframe and drops only out-of-window derived ones.
- packages/core/test/surface-transfer.test.mjs (new, step 7): two synthetic meshes offset by a known translation; transferLabels moves a labelled patch to the geometrically corresponding triangles, and a patch beyond the gate is not transferred.
- packages/encoder/test/track-multi.test.mjs (new, step 8): a two-object track produces two range groups with distinct objId and a shared trackId; overlapping spans do not interfere in keepPredicateAt; a mask-seeded range validates clean; the new range-id allocator produces no duplicate after a delete-then-create sequence.
- Manual gates the numbers cannot replace, per docs/targeted-temporal.md:66-72 (every distance metric here has a 0.5 to 1 mm floor and registration error read ~0 for the flattened-nose failure): visual review of a full propagated span against the source before any docs entry claims quality, and a hand-labelled span to validate the conf < 0.60 caret threshold before step 7 promises it.

## Perf budget

Target: 272 frames, 1 object, sm_75 fp16 (_best_dtype at main.py:92-106 returns float16 below compute capability 8.0, on this repo's own measurement that the card does 43.0 TFLOP/s fp16 against 7.3 bf16). Every GPU number below is an ESTIMATE anchored on main.py:295-298, which records 360 to 550 ms per warm click on a 6 GB 3060 with the 1008x1008 vision encoder as "the bulk". Step 2 replaces them with measurements before any UI promises a duration.

CAPTURE SWEEP (browser), per frame: seekFrame plus synchronous geometry present 2 to 5 ms; texture settle 1 to 2 rAF 16 to 33 ms; renderCurrent plus drawImage 3 to 6 ms; toBlob JPEG at 1008 long side 6 to 12 ms. About 40 ms per frame, so 272 frames is roughly 11 s, about 14 s including batching. JPEG payload about 220 KB per frame, 60 MB total, under 0.2 s of loopback wire time.

PROPAGATION (service), per frame: vision encoder at 1008 about 320 ms; tracker head plus memory attention for 1 object about 45 ms (11.7 M params, at most 10 spatial memory entries because num_maskmem is 7 and max_cond_frame_num is 4, both bounded by config and NOT by clip length, plus 16 object pointers); JPEG decode and preprocess 12 to 20 ms on CPU, prefetched one frame ahead so it hides under the GPU step. About 365 ms per frame.
  Forward over 272 frames: about 99 s. Bidirectional from frame 136: 136 plus 136 evaluations, the same 99 s.
  FIRST TRACK END TO END: 14 s capture plus 99 s propagate plus about 1 s write, roughly 114 s.
  At trackRes 560: encoder compute scales by (560/1008)^2 = 0.309, so about 115 ms plus 45 ms head = 160 ms per frame, roughly 44 s for the clip.
  RETRACK over a 60-frame window: 0 s capture because the service still holds the frames, plus 60 x 0.365 = about 22 s. A Nudge over 30 frames is about 11 s final, 5 s draft. That bounded number is what the correction loop lives on.

SCALING IN OBJECT COUNT is sub-linear because the vision encoder runs once per frame regardless and only the head loops serially (modeling_sam3_tracker_video.py:1780-1783; only _batch_encode_memories at :2706 is batched): 1 object 99 s, 2 objects about 111 s (+12%), 4 objects about 136 s (+37%).

VRAM, fp16, peak. Sam3VideoModel weights 859.9 M params x 2 B = about 1,640 MiB. Non-shared Sam3TrackerModel tensors about 16 MB (the vision tower is shared by graft). Vision-feature cache at size 1: (288^2 + 144^2 + 72^2) x 256 = 27.87 M elements held as feats AND pos_embeds, about 106 MiB at 1008, about 33 MiB at 560. Memory bank on device about 7 MB, because inference_state_device is cpu and only object_pointer and object_score_logits stay resident. Activations and allocator about 2,000 MB, extrapolated from main.py:96-108, which records a 3,671 MB whole-GPU peak against about 1,653 MB of weights for today's pair. PEAK about 3.8 GiB, so 7.1 GB free on an 11 GB card and about 2.1 GB free on the 6 GB 3060 that main.py already ships in.
  DELTA AGAINST TODAY: today's resident pair is about 848 M params; the new set is about 868 M. +19.5 M params, roughly 38 MiB in fp16. That is the entire cost of video tracking, and it is the strongest argument for this design.

HOST RAM. JPEG frame store 272 x 220 KB = 60 MB. One preprocessed frame resident at a time, 6.1 MB, popped after use; WITHOUT the pop it would be 272 x 6.1 MB = 1.66 GB, which is why frames are stored as JPEG rather than as a preprocessed tensor stack. Memory-bank history offloaded to cpu, per object per frame: maskmem_features 5184x64 bf16 (hardcoded at modeling_sam3_tracker_video.py:2701) 648 KB, maskmem_pos_enc 648 KB, pred_masks 288^2 fp16 162 KB, high_res_masks 1008^2 fp16 1,984 KB, object_pointer 512 B, about 3.36 MiB. Times 272 frames is about 914 MiB for one object, about 3.6 GiB for four. The 2-session cap exists for this.

SIDECAR. 272 keyframes at 768x432 is roughly 1,700 runs per frame, about 1.8 MB compact and about 7.4 MB pretty-printed at RLE nesting depth (a measured 4.1x). The 4,000,000-byte cap at serve.mjs:709 destroys the socket with NO response while main.js:1048 swallows it, so pretty-printing loses a two-minute track in silence. Fixed twice over: conditional compact serialization when the document holds any derived keyframe, and a 16 MB cap that answers 413. Two tracked objects are about 3.6 MB compact, comfortable.

PREVIEW AND BAKE. Per presented frame after the bracket() fix: one mask decode (about 0.3 ms) plus about 20k centroid tests (perspective divide, depth compare, array index, about 0.4 ms) = about 0.7 ms inside a 33 ms frame. WITHOUT the fix it is 1.4 ms, because bracket returns t = 1 at every dense keyframe and both bracketing keyframes compile and decode. Resident decoded masks: 272 x 768 x 432 = 90 MB with the current unbounded WeakMap keyed on the document's live mask objects, which is worth capping to a small LRU when tracks become routine. Bake adds about 0.19 s to a multi-minute encode.

UNDO. With mask payloads passed by reference: about 54 KB and about 0.4 ms per snapshot. With today's JSON deep-clone plus double stringify (main.js:1073-1092) and UNDO_CAP 50: about 1.8 MB x 4 serialization passes, roughly 40 ms on EVERY discrete mutation and about 90 MB resident across the stack. That is the difference between a usable editor and a laggy one after a track lands.

3D AUDIT, fully hidden: buildTriangleGrid over 20k triangles about 6 ms, transferLabels about 14 ms, rasterizeIds at mask resolution about 11 ms, about 31 ms in a worker against 365 ms of GPU on the next frame.

## Open questions (owner calls)

- Stored mask resolution. maskRes 768 long side is chosen here from sidecar arithmetic, but nobody has evaluated it visually. The shipped click captures at 1024 (main.js:2440), the depth band rasters at 384, and docs/editor-v2-design.md:150 mentions 512. On a 2 m subject filling the frame, 768 is roughly 1.3 world-mm per pixel, which is finer than the 0.5 to 1 mm metric floor this toolchain already lives with, but edge accuracy against sidecar size is a product call that only a side-by-side review settles.
- Depth-band separability of a subject and a prop it contacts. Every mask2d carries ONE [zmin, zmax] band from a 384x384 raster (main.js:2493), gated at edits.ts:279-282. A rider and the board under their feet occupy overlapping NDC z from any single frozen camera, so one band cannot separate them: select the board and the shoe comes with it. This is a schema decision (per-mask multi-band, or a coarse per-frame depth map beside the bitmap) that outranks mask boil as a fidelity limit, and no design addressed it.
- Atlas tile budget for a propagated delete. Hole capping runs per frame (hole-patch.ts:180) and allocates a 16 px atlas tile per accepted boundary loop by free-space scan (cli.ts:700-709). A 272-frame moving delete could demand hundreds of tiles in an atlas that is already repacked every frame. The repo cannot answer whether the space exists, or what should happen when it runs out, without running a bake of a real tracked delete.
- Whether a freshly propagated range should land with enabled:false (muted) pending review. Muting keeps it out of both preview and bake (edits.ts:468, cli.ts:320) and matches this project's habit of gating every recipe change on visual evaluation, but it also means the user sees nothing until they find the checkbox. This is a workflow preference, not a technical constraint.
- EditRange.scrubTexels (edits.ts:178) is declared with zero implementations anywhere, and docs/editor-v2-design.md:301 justifies it as privacy deletion that must remove pixels rather than references. It is the one already-specified feature that CANNOT work without a per-frame region, so propagation is its enabling dependency. Whether it moves up the queue now that the region exists is an owner call.
- The SAM License. SAM 3 weights are under Meta's custom license (commercial use permitted, with acknowledgement, no-reverse-engineering, export-control and patent-retaliation terms that Meta may amend), while SAM 2 and 2.1 are Apache-2.0 and Cutie is MIT. Whether propagation should ever ship a permissive fallback tier, and how the acknowledgement obligation is discharged in THIRD-PARTY-NOTICES.md and in any distributed build, is a licensing decision the repo cannot make.

## Critic: must resolve before step 1

- Decide the tab-reload contract before step 5 is designed, because it changes step 5's writer: incremental keyframe writes inside gestureBegin/gestureCommit versus one terminal doMutation. The plan currently claims both.
- Correct step 1(d): rebaseEditList returns at packages/encoder/src/cli.ts:193 when trim-in is 0, so the derived-keyframe drop and the range clamp never run on a trim-out-only bake. Fix the early return in the same edit.
- Specify validateMasks' kind:"rect" branch (packages/core/src/edits.ts:261 accepts rect volumes with no mask), or step 1(d)'s new `ares verify-edits` will exit 1 on any sidecar containing a box-select volume.
- Add `frameGeomQ(): { positionsQ, indices, box, invLevels, frameIndex }` to step 3's player kit. Step 7's auditor is unimplementable without it — rasterizeIds (raster.ts:28-36) takes quantized positions plus the GOP AABB and invLevels, none of which exportFrame() returns.
- Rewrite step 3's test plan: AresPlayer cannot be constructed under `node --test` (player.ts:394/:440/:716 need document, navigator.gpu, requestAnimationFrame) and synthClip is an encoder export (packages/encoder/src/synth.ts:157) that packages/core/test cannot import.
- Decide the gap policy per mode before step 6: run-splitting is correct for mode:"delete" and wrong for mode:"keep" (isolate), where it makes the whole scene reappear for the gap (edits.ts:471-474 returns null with no active range).
- Fix step 4's vit_h item: changing VITH_CHECKPOINT's default to the .pth does not revive the fallback, because main.py:226 loads through safetensors' load_file. The loader itself has to branch on the extension.

## Critic: verified OK

- bracket() defect is real and the one-character fix is correct. packages/core/src/edits.ts:319 `if (f >= a.frame && f <= b.frame)` returns t=1 at every dense keyframe; edits.ts:350 forces t=0 for interp:"hold" ABOVE the edits.ts:353 `if (t === 0) return evA` fast path, so a dense hold range does return the previous frame's mask. Changing :319 to `f < b.frame` leaves the sparse cases untouched (f<=first and f>=last early-return at edits.ts:314-315; keyframes at 0/10 still give t=0.5 at f=5).
- The ortho bug is live, not latent. player.ts:164-168 getCamera() returns `ortho`; apps/demo/main.js:2440 spreads it into samSel.cam; main.js:2528-2530 samVolume passes samSel.cam straight through as `camera`; edits.ts:263 rebuilds the OrbitState literal WITHOUT ortho; camera.ts:141-143 orbitMatrices honors it. The shipped sidecar apps/demo/daniel-s0-v2.edits.json already serializes "ortho": false, and Mask2dVolume['camera'] (edits.ts:33) is a closed type — the index signature at edits.ts:36 is on the volume.
- The towerless-tracker seam is the library's own path, exactly as claimed. modeling_sam3_tracker_video.py:703 base_model_prefix = "tracker_model"; :1602 _keys_to_ignore_on_load_unexpected = [r"^detector_model."]; :1604/:1617 `self.vision_encoder = AutoModel.from_config(...) if not remove_vision_encoder else None`; :1900 _prepare_vision_features; modeling_sam3_video.py:511-512 builds the tracker with remove_vision_encoder=True, :541 tracker_neck, :545 get_vision_features_for_tracker, :1630 cache_vision_features.
- The single-call multi-object seeding rule is real: processing_sam3_tracker_video.py:736 and :800 both END with `inference_session.obj_with_new_inputs = obj_ids`, an assignment. Prompt exclusivity at :611-613 ('masks cannot be provided together with points or boxes') and the box rule at :702-707 ('cannot add box without clearing old points') are verbatim as cited. clear_old_inputs defaults True at :577/:638.
- Reverse propagation semantics confirmed at modeling_sam3_tracker_video.py:2810-2816 — reverse yields range(start, end-1, -1) INCLUSIVE of start and short-circuits to [] when start_frame_idx == 0. Passing start_frame_idx explicitly is necessary: :2792-2806 otherwise takes min() over all cond frames (and raises when there are none). The bf16 memory-bank store is hardcoded at :2701.
- Installed env is transformers 5.13.0 / torch 2.6.0+cu124 / torchvision 0.21.0+cu124, NOT the requirements.txt pins. All of sam3, sam3_tracker, sam3_tracker_video, sam3_video, sam2, sam2_video are present under tools/sam-service/env/Lib/site-packages/transformers/models/.
- The design doc's blocker at docs/editor-v2-design.md:280-284 is stale as claimed (it names the facebookresearch loader path and issue #511), and docs/editor-v2-design.md:177 does promise the distance-transform SDF lerp that edits.ts:288 does not implement. docs/editor-v2-design.md:359 does document `{encoding, width, height, data}` against the shipped `{width, height, rle}` (edits.ts:32).
- Keys K and N are genuinely unbound and Shift+S is genuinely taken. apps/demo/main.js:3470 `case "s": case "S": clickTool("sam")`; the whole switch at main.js:3442-3494 contains no k/K or n/N. The KEYS table (main.js:3311-3322), the keydown switch, and the #hint tooltip (index.html:1162) are indeed the three registration sites.
- The isolate-blackout hazard the plan avoids is real: apps/demo/main.js:2601-2603 sets `r.mode = "keep"` for isolate, and edits.ts:476-481 drops every point outside all active keep regions.
- serve.mjs mechanics are as described: GUARDED at serve.mjs:502 with `sam\/start` (so `sam\/(start|track\/)` is the correct edit and the guard runs at :519 before the proxy); the wildcard /sam/* proxy at :659-687 forwards any method, pipes both directions (`req.pipe(up)` :686, `ur.pipe(res)` :673) and copies Content-Length only when upstream sent one (:671); the /edits POST cap is `if (body.length > 4_000_000) req.destroy()` at :709 with no response, and the one-deep .bak is at :716.
- tools/sam-service/main.py structure is as cited: 724 lines, _lock at :123, the vision-tower graft at :180 with _assert_vision_encoder_shared at :193, the loader order map at :236, /health at :325, /segment at :461, /segment_text at :499, /upscale at :601, /detail at :705.
- Prerequisite 6 is satisfied and clean: apps/demo/daniel-s0-v2.edits.json is the only sidecar in the tree and contains no `interp` field, so the bracket() hold behavior change breaks nothing shipped.
- `derived` is already a schema field (edits.ts:69) with a renderer and no writer: apps/demo/main.js:1805 draws `k.derived ? "◇" : "◆"` in var(--text-faint), positioned absolutely per keyframe inside a 12 px lane — the 272-diamond problem is real.
- Test wiring is sound. `npm test` runs `tsc -b && node --test "packages/*/test/*.test.mjs"`, packages/core/test already exists (container/fx/splat), core tests import from ../dist/index.js and encoder tests from ../dist/index.js plus @ares/core (the @ares/* workspace symlinks are in place). validateMasks exported from edits.ts reaches both hosts via packages/core/src/index.ts:17 `export * from "./edits.js"`, and buildTemporalGops is already exported (packages/encoder/src/index.ts:2, temporal.ts:279). closestOnTriangle (temporal.ts:52) and TriangleGrid (temporal.ts:110) are private today, and moving them to core keeps the dependency direction legal (encoder→core).
- The `ares verify-edits` verb slots in cleanly: dispatch is a flat if/else chain at cli.ts:1076-1082 with USAGE at cli.ts:1053, and the delete/keep sweep at cli.ts:652-661 already tolerates a null predicate (`if (keep) frames[f] = filterFrame(...)`).

## Critic: gaps

### Gap 1 [critical]: Tab reload / mid-run disconnect destroys the entire track, and the plan contradicts itself about it. commitTrack is ONE doMutation at the END of the run, so no mask is persisted until the SSE stream completes. A reload loses every mask, leaves the FastAPI session holding _track_lock for the remaining minutes, and there is no route to find or reattach it (open/frames/prompt/run/cancel/close — no session list, no result replay, no Last-Event-ID).

**Evidence:** apps/demo/main.js:1095-1100 doMutation is one snapshot+push around one fn; step 5 says 'ONE doMutation for the whole span'. apps/demo/main.js:1048-1049 doSave is a 600 ms debounce with .catch(() => {}). tools/serve.mjs:659-687 the /sam proxy registers NO req.on("close") (unlike every native long route: serve.mjs:1375, 1549, 1669), so a dead client neither cancels nor is detected. Step 6 asserts 'Every mask already emitted is already a keyframe, so a cancel leaves a shorter but valid track' and ui_strings has 'track cancelled · 143 kf' — both false if commitTrack only runs at the end.

**Fix:** Pick one: (a) write keyframes incrementally inside a gestureBegin()/gestureCommit() pair (apps/demo/main.js:1104-1111) so the whole run still collapses to one undo step while landing in `edits` as it streams, and flushSave() every N frames; and (b) add GET /sam/track/sessions plus GET /sam/track/results?session=&from=<frame> so a fresh tab can enumerate live sessions and replay masks already emitted. Without (b), also add an idle-cancel: the reaper must cancel a RUNNING session whose stream has had no reader for >30 s, or _track_lock is held by a job nobody is watching.

### Gap 2 [critical]: Step 7's drift auditor cannot call rasterizeIds with exportFrame() output — the signature does not match and the two extra arguments it needs are private to the player.

**Evidence:** packages/core/src/raster.ts:28-36 — rasterizeIds(positionsQ: Uint16Array /* stride 4 */, indices, box: Aabb, invLevels: number, viewProj, w, h). packages/core/src/player.ts:273-306 exportFrame() returns dequantized `positions: Float32Array` (stride 3) and no box/invLevels. The values it needs live at player.ts:359-361 as `this.invLevels` and `ref.gopBox`, both private. Step 7 states the worker 'takes exportFrame() output ... and rasterizes the prediction into the frozen camera with rasterizeIds (raster.ts:28)'.

**Fix:** Add to step 3 a fourth player accessor `frameGeomQ(): { positionsQ: Uint16Array; indices: Uint32Array; box: Aabb; invLevels: number; frameIndex: number } | null` (four lines, mirrors pickRaster's preamble at player.ts:357-361) and feed the worker that — it is also the transferable form, so the postMessage is zero-copy. Alternatively add a `rasterizeIdsF(positions: Float32Array, ...)` overload in raster.ts, but that duplicates the dequant loop.

### Gap 3 [high]: Step 4's vit_h revival does not work: pointing VITH_CHECKPOINT at the .pth that exists makes the loader raise a DIFFERENT error, because the loader is safetensors-only. The fallback stays dead.

**Evidence:** tools/sam-service/main.py:219 `from safetensors.torch import load_file`; main.py:226 `model.load_state_dict(load_file(VITH_CHECKPOINT))`. tools/sam-service/models/ holds only sam_vit_h_4b8939.pth (a torch pickle), RealESRGAN_x4plus.pth, realesr-general-x4v3.pth. safetensors' load_file on a pickle raises SafetensorError (HeaderTooLarge), not FileNotFoundError, so the guard at main.py:222 does not catch it.

**Fix:** In _load_vith, branch on the extension: for `.pth` use `sam_model_registry["vit_h"](checkpoint=VITH_CHECKPOINT)` (segment_anything torch.loads it itself and skips the separate load_state_dict), keeping the safetensors path for `.safetensors`. Then drop the now-redundant `model.to(torch.float32)` comment about fp16 storage, which is only true of the safetensors file.

### Gap 4 [high]: Two of the plan's named test files cannot run under this project's `npm test`. packages/core/test/player-api.test.mjs would need a DOM to build a player and would need synthClip, which is an ENCODER export core cannot depend on. packages/encoder/test/track-commit.test.mjs claims to assert cloneStateForUndo and commitTrack, both of which live in apps/demo/main.js, which throws at module scope in Node.

**Evidence:** npm test is `tsc -b && node --test "packages/*/test/*.test.mjs"` (package.json:20). AresPlayer needs a canvas + navigator.gpu + rAF: player.ts:394 document.createElement, :440 navigator.gpu, :716/:923 requestAnimationFrame; no existing test in packages/*/test imports AresPlayer (grep returns nothing). synthClip is exported from packages/encoder/src/synth.ts:157, not core; packages/core/src/index.ts has no encoder import and cannot get one (encoder depends on core). apps/demo/main.js:13 `const canvas = $("view")` executes at import time and main.js:8 imports the bare specifier "@ares/core".

**Fix:** Step 3's test: keep only the prototype-arity assertions (a static `import { AresPlayer } from "../dist/index.js"` is safe — no module-scope DOM access) and move the fps→index arithmetic into a pure exported helper in core so it can be asserted without a player. Step 5's test: it can only assert the SHAPE of a track edit list, so say so; to actually test the writer, extract commitTrack's keyframe-building and cloneStateForUndo/sameState into a pure module under packages/core/src (or a new apps/demo/track-commit.js with no DOM at module scope) and import that.

### Gap 5 [high]: Step 1(d)'s rebaseEditList change never executes on the most common trim. rebaseEditList returns before doing any work when trim-in is 0, so a trim-out-only bake neither clamps ranges nor drops out-of-window derived keyframes.

**Evidence:** packages/encoder/src/cli.ts:193 `if (from === 0) return list;` is the first statement of rebaseEditList; the keyframe map at cli.ts:203 and the range clamp at cli.ts:200-201 are both below it. Step 1(d) says 'make rebaseEditList DROP keyframes with `derived === true` whose rebased frame falls outside [0, n-1]'.

**Fix:** Replace the early return with `if (from === 0 && n >= (max endFrame + 1)) return list;`, or simply delete it and let the from=0 path fall through (the shifts become no-ops, the clamp and the derived-drop still apply). Add the trim-out-only case to the track-bake test, which currently only exercises 'rebaseEditList under a trim'.

### Gap 6 [high]: The occlusion-gap policy is presented as the correctness fix for isolate, but for a keep/isolate range it produces the opposite failure instead of the right answer: the gap frames show the FULL scene, not the isolated subject.

**Evidence:** packages/core/src/edits.ts:471-473 — keepPredicateAt filters ranges by `f >= r.startFrame && f <= r.endFrame`, and edits.ts:474 `if (!del.length && !keep.length) return null`. With the track split into runs that skip [143..147], frame 145 has no active range from this track, so nothing is filtered and every triangle survives. apps/demo/main.js:2601-2603 sets `r.mode = "keep"` for isolate. A 5-frame flash of the whole scene inside an isolate is a visible defect, not a benign no-op.

**Fix:** Split into runs only for `mode:"delete"`. For `mode:"keep"` (isolate), keep one range spanning the gap and write a derived keyframe at each gap frame carrying the last non-empty mask, with `conf: 0` so the drift caret and the N key land on it. Alternatively, refuse to commit a keep-mode track that has gaps and make the user choose. Either way the docs/rgbd-rebuild-pipeline.md:93-98 'never invent a mask across a gap' precedent applies to delete, not to keep — say which.

### Gap 7 [medium]: apps/demo/track-worker.js cannot resolve `@ares/core`. Module workers do not inherit the document's import map.

**Evidence:** apps/demo/index.html:1164-1170 defines the importmap in a document-scoped <script type="importmap">; apps/demo/main.js:8 relies on it. The only worker in the tree, packages/core/src/worker-decode.ts:65 `new Worker(url, { type: "module", name: "ares-geom-decode" })`, lives inside packages/core and resolves relative paths — it never uses a bare specifier.

**Fix:** Import the built absolute path from the worker: `import { buildTriangleGrid, transferLabels } from "/packages/core/dist/surface-transfer.js"` and `rasterizeIds` from "/packages/core/dist/raster.js" — importing /packages/core/dist/index.js instead would drag player.js and both renderers into the worker for no reason.

### Gap 8 [medium]: Step 7 never says where frame f's geometry comes from at audit time. The architecture uploads every frame first and only then streams masks, so by the time mask f arrives the player has finished the sweep and is no longer presenting frame f.

**Evidence:** Routes: POST /sam/track/frames (16 frames per request, all frames uploaded) then GET /sam/track/run streams `mask` events. player.ts:560-566 keeps only the CURRENT frame's curPosQ/curIndices; player.ts:660 even nulls curPosQ when the worker decoder is in use. Retaining all 272 frames is ~24 MB of positionsQ plus ~65 MB of index buffers, which perf_budget does not account for.

**Fix:** State the mechanism: on each `mask` event, seekFrame(f) + frameGeomQ() (a few ms, geometry present is synchronous per player.ts:850-855) and run the transfer in the worker. Add that seek cost to the perf budget, and note that the audit therefore forces the viewport to scrub during the run — decide whether that is acceptable or whether the audit becomes a post-pass.

### Gap 9 [medium]: The ortho fix repairs the bake but leaves the live preview wrong, because camKeyNow() does not include ortho, so toggling P mid-selection neither invalidates the selection nor switches the painter out of direct-blit mode.

**Evidence:** apps/demo/main.js:2408-2411 camKeyNow builds the key from azimuth/elevation/distance/target only. main.js:3488 P toggles ortho live. main.js:2567-2568 samWatch chooses samPaintDirect (raw mask pixels blitted 1:1 over the canvas) whenever camKeyNow() matches the capture key. After step 1(b) the stored mask evaluates in the capture's projection (correct), while the on-screen tint is blitted over a render in the other projection.

**Fix:** Add `c.ortho ? 1 : 0` to the camKeyNow tuple (apps/demo/main.js:2409). One token, and it also makes 'camera moved · restore view' fire for a P toggle, which is the honest behavior.

### Gap 10 [medium]: validateMasks as specified will false-positive on every marquee (`kind:"rect"`) volume, which has no mask and therefore no width/height to check `camera.aspect` against.

**Evidence:** packages/core/src/edits.ts:261 accepts `v.kind === "rect" ? v.rect : v.kind === "bitmap" && v.mask` — rect volumes are legal and carry no `mask`. apps/demo/main.js:2340-2366 (box-select pointerup) authors exactly those. The plan's per-volume checks are 'kind present, mask.rle run sum exactly equals width*height, camera present, camera.aspect within 1% of width/height' with no branch.

**Fix:** Branch validateMasks on kind: for "rect" check `rect` present, length 4, x0<x1, y0<y1, all in [-1,1]; for "bitmap" check mask present, run sum == width*height, and the aspect ratio. Report a missing/unknown `kind` as its own issue since that is the silent no-op at edits.ts:295.

### Gap 11 [medium]: getClipFps only fixes two of the seven hardcoded /30 sites, and leaves the sidecar itself lying about fps.

**Evidence:** apps/demo/main.js:1045 `const edits = { aresEdits: 1, ..., fps: 30, ... }` — written into every sidecar. Other /30 sites the plan does not name: main.js:40 (share link `&t=`), :740 (history entry fps/durationS), :847-848 (the transport time readout), :1630 and :1634 (trim readout and its tooltip), :3447 (End key). The plan names only :1663 and :3441.

**Fix:** Set `edits.fps = player.getClipFps()` once the file is loaded and route every /30 through it. Otherwise the CHANGELOG claim that non-30 clips now seek correctly is only two-sevenths true, and the sidecar still records fps 30.

### Gap 12 [low]: The prompt-coordinate rescale is wrong whenever the capture's long side is its height. captureFrame caps the LONG side, not the width.

**Evidence:** packages/core/src/player.ts:391 `const scale = Math.min(1, maxDim / Math.max(1, Math.max(c.width, c.height)))`. apps/demo/main.js:2449 stores points as `fx * samSel.capture.width, fy * samSel.capture.height`. The routes section says 'the client rescales samSel.points from capture-image pixels by trackRes/samSel.capture.width'.

**Fix:** Rescale by `proxyWidth / capture.width` and `proxyHeight / capture.height` independently — both are known, since captureFrameBlob returns the proxy dimensions. Or normalize to 0..1 at the client and let the service multiply by original_size, which is what processing_sam3_tracker_video.py:665-676 does internally anyway.

### Gap 13 [low]: Step 8 proposes adding a THIRD-PARTY-NOTICES.md line for the SAM License that already exists.

**Evidence:** THIRD-PARTY-NOTICES.md:39 — '- **SAM 3 / SAM ViT-H** weights: Meta's SAM License.' (also :47-49 for SAM 3D Body).

**Fix:** Drop the item, or replace it with what is genuinely absent: the acknowledgement obligation's discharge text, and a note distinguishing the SAM 3 weights (Meta custom license) from segment-anything / safetensors / huggingface_hub (Apache-2.0, already listed at THIRD-PARTY-NOTICES.md:34).

### Gap 14 [low]: packages/encoder/test/track-bake.test.mjs is listed in `tests` as '(new, step 6)' but no step creates it — step 6's `files` are track.js, main.js, track.py and its `test` names only track-gaps.test.mjs.

**Evidence:** Plan step 6 files array and test field vs the tests array entry 'packages/encoder/test/track-bake.test.mjs (new, step 6): BAKE DETERMINISM.'

**Fix:** Add it to step 6's files and test, or fold its two assertions (byte-identical kept-triangle set across two runs and across a JSON round-trip; rebase under a trim) into track-gaps.test.mjs.

### Gap 15 [low]: Every edits.ts line citation in the plan is 1-4 lines low, which will send an implementer to the wrong line in a 502-line file where several similar guards sit close together.

**Evidence:** bracket()'s loop condition is packages/core/src/edits.ts:319, not 316. `if (r.interp === "hold") t = 0` is :350, not 347. The prepareVolume camera literal is :263, not 262-264 (the OrbitState object is on one line). rebaseEditList starts at packages/encoder/src/cli.ts:192-193, not 196-203. The _loader order map is tools/sam-service/main.py:236, not 235. (edits.ts:353 `if (t === 0) return evA`, edits.ts:36, edits.ts:414, edits.ts:457, main.py:123/180/193 are all exact.)

**Fix:** Re-grep the four cited files and correct the citations before the CHANGELOG entry quotes them.

### Gap 16 [low]: Steps 5 and 6 put nine new controls into 'one new row' of a rail clamped to 220-480 px.

**Evidence:** apps/demo/index.html:940 #samSelRow is `class="row" style="...flex-wrap:wrap"` and already carries #samSelInfo + #samActSel + #samApply + #samClear. The new row must hold Track, direction, span, resolution, Cancel, plus step 6's Nudge, Re-seed, Re-anchor and the N-jump readout. apps/demo/index.html:201 `.rail .railBody { overflow-y: auto }` is the one scrolling container.

**Fix:** Split by state, the way #samSelRow already is: one row visible only while a selection is pending (Track + direction + span + res), one visible only while a run is live (progress + Cancel), one visible only when the active range has `track` (Retrack, Nudge, Re-seed, Re-anchor, N). Never more than four controls on screen at once.

---

## Decisions, 2026-09-09

Resolved so implementers do not re-litigate them. These override anything above.

1. **Tab-reload contract.** Incremental: the track writer wraps the sweep in one
   `gestureBegin()` / `gestureCommit()` pair so the run collapses to a single undo step while
   keyframes land in `edits` as they stream, with a `flushSave()` every 16 frames. Plus
   `GET /sam/track/sessions` and `GET /sam/track/results?session=&from=<frame>` for replay, and an
   idle-cancel reaper that cancels a RUNNING session whose stream has had no reader for 30 s.
   Critic gap 1, option (a) + (b).
2. **Occlusion-gap policy is per mode.** `mode:"delete"` splits into one range per contiguous run.
   `mode:"keep"` (isolate) keeps ONE range spanning the gap and writes a derived keyframe at each
   gap frame carrying the last non-empty mask, `conf: 0`, so the drift caret and the N key land on
   it. Never emit an empty keep mask: `keepPredicateAt` would delete the whole frame. Critic gap 6.
3. **Stored mask resolution** defaults to 768 px on the long side, exposed as a resolution tier
   control. Reviewable against 512 and 1024 once a track exists; not a blocker.
4. **A freshly propagated range lands enabled**, not muted. The drift caret and the N key are the
   review mechanism; muting is one existing checkbox away if the user wants it.
5. **vit_h loader branches on the file extension**: `.pth` goes through
   `sam_model_registry["vit_h"](checkpoint=...)`, `.safetensors` keeps `load_file`. Pointing the
   default at the `.pth` alone does NOT revive the fallback. Critic gap 3.
6. **Test plan.** No test constructs an `AresPlayer` (needs document / navigator.gpu / rAF) and
   nothing in `packages/core/test` imports from the encoder. Player tests assert prototype symbols
   and arity only; anything with real logic moves into a pure exported helper first. Critic gap 4.
7. **Every file:line citation is re-grepped before use.** The plan's `edits.ts` citations run 1-4
   lines low. Critic gap 15.
8. **Deferred, not dropped** (owner calls, revisit after a track exists): per-mask multi-band depth
   or a per-frame depth map, so a rider and the board they stand on can be separated; the atlas
   tile budget for a 272-frame moving delete; whether `scrubTexels` moves up the queue now that a
   per-frame region exists; whether a permissively licensed fallback tier (SAM 2, Apache-2.0) ships
   beside the SAM 3 weights.

---

## Measured corrections, 2026-09-09

`tools/sam-service/track_smoke.py` ran on this machine. Where a measurement contradicts the design
above, the measurement wins. Environment: Python 3.13.7, transformers 5.13.0, torch 2.6.0+cu124,
two RTX 2080 Ti (11,264 MiB each, compute capability 7.5, so `_best_dtype` picks fp16).

1. **The negative control does not exist.** The Verdict, the Prerequisites and the critic's
   "verified OK" all assert that `Sam3TrackerVideoModel.from_pretrained("facebook/sam3")` reports
   538 missing `vision_encoder` keys and silently random-inits its tower. Measured, it loads
   CLEAN: 845 tensors, 0 missing, 0 unexpected, 0 mismatched. Transformers 5.13.0's flexible
   cross-architecture loader remaps `detector_model.vision_encoder.*` onto `vision_encoder.*`
   (516 trunk tensors, all bit-identical) and `tracker_neck.*` onto `vision_encoder.neck.*`
   (22 tensors, bit-identical). The critic quoted the source lines correctly but never ran the load.
2. **The conclusion survives, for a better reason.** 454.04 M of that model's 465.78 M parameters
   are a SECOND COPY of the tower the concept model already holds — about 908 MiB at fp16, resident
   and idle. `Sam3VideoModel` plus the towerless-tracker seam avoids that copy. The probe keeps the
   bit-identity assertions so a future transformers that stops remapping fails loudly.
3. **The +19.5 M / ~38 MiB delta is confirmed**, once the click path is counted: 859.92 M
   (detector 840.38, tracker 11.74, tracker_neck 7.80) + 4.22 M for `Sam3TrackerModel`'s non-vision
   remainder = 864.14 M, against today's 844.60 M. Delta +19.54 M = 37.3 MiB at fp16.
4. **The perf estimates were pessimistic.** Measured at fp16 on the 2080 Ti: encoder 189.3 ms
   median, head 75.6 ms median, 291.6 ms/frame, so 272 frames is about 79 s, not 99 s. Peak device
   allocation 2,263 MiB (1,835 weights + 428 activations), reserved peak 3,744 MiB, against an
   estimated 3.8 GiB. First frame 708.2 ms; model load 4.3 s; 0 vision-cache misses across both the
   seeded and the propagated forward, which is the structural proof that the seam works.
5. **fp16 is the right dtype on this card, by a wide margin.** bf16 measured 1,286.3 ms/frame and
   5,789 MiB peak — 4.4x slower and 2.6x the memory, on a card with no bf16 tensor-core path.
6. **The fp16-vs-bf16 IoU is NOT a usable gate yet.** The only numbered-frame directory on this
   machine is the capture's ATLAS PNGs, which are chart-scrambled: measured, the object score falls
   from 19.8 logits at the seed to a 0.3-2.6 band and mask area oscillates tenfold frame to frame,
   so the two dtype runs' memory banks diverge and stay diverged (median IoU 0.9245, min 0.0000;
   lock-gated 0.9867 on the one frame both runs held). Timings and VRAM are shape-determined and
   therefore valid. Re-run the IoU over frozen-camera proxy renders now that step 3 shipped
   `captureFrameBlob`.
7. **`post_process_masks` takes a LIST.** Element i must be the full 4-D tensor
   (`image_processing_sam3.py:667-673`); passing the `(1,1,288,288)` tensor directly makes
   `masks[0]` 3-D and `F.interpolate` rejects it. Step 4's `track.py` calls this per frame.
8. **`remove_vision_encoder=True` is `modeling_sam3_video.py:512`,** not 511.

---

## Step 4 outcomes, 2026-09-09 — these override steps 5-8 where they disagree

Step 4 shipped and was driven live against the real weights on an RTX 2080 Ti. What the running
service taught us that the plan did not know:

1. **`trackRes` is not a free parameter.** `prompt_encoder.image_embedding_size` is fixed at
   `image_size // patch_size` = 1008/14 = 72 when the model is CONSTRUCTED, and the detector's
   tokens are viewed through exactly that grid. `/track/open` 400s any other edge. Step 8's
   "draft 560" tier does not exist and cannot without position-embedding interpolation. Do not
   build a resolution-tier control for `trackRes`; `maskRes` (the stored bitmap) is still free.
2. **A cold `GET /sam/track/run` cannot start the service and cannot report why.** The proxy
   auto-starts only on POST, so a cold GET returns a JSON 503 that an EventSource surfaces as a
   bare `onerror`. The client MUST `POST /track/open` or read `/sam/health` first. This is a client
   contract, not a bug.
3. **`/track/prompt` returns no mask for the seed frame.** The run re-emits it from
   `cond_frame_outputs`, so the client must run FROM the seed frame, not seed+1.
4. **A prompt during a live run answers 409.** Cancel the run before re-seeding; the correction
   loop must sequence that way.
5. **`/track/results` is indexed by EVENT INDEX, not frame number** — a reverse run emits frames in
   decreasing order and a multi-object run emits one event per object per frame, so a frame number
   does not order a replay. The response carries `next`/`more`/`total`, and a per-session `run_id`
   (in the `start` event, `/track/results` and `/track/sessions`) pins which run a replay belongs to.
6. **The `done` event reports `frames` actually forwarded plus `of`** for the span, so progress
   computed from `frames` alone is correct on a cancelled or reaped run.
7. **Measured cost:** 304 ms/frame with the inference state on the host (the shipped default),
   266 ms/frame with it on the card; the host round trip costs ~13%, about 10 s over 272 frames,
   and saves ~914 MiB of device memory. A SECOND OBJECT costs +44%, not the plan's +12% — the
   encoder runs once per frame but the head, `post_process_masks` and the RLE encode are serial per
   object. First frame after seeding 450-700 ms; `/track/prompt` 1.56 s including session build.
8. **An interactive click interleaves at about one frame** (219/236/181 ms measured while a run
   streamed), because the model lock is taken per frame rather than per run.
9. **The one-deep `.bak` does not survive a track.** Decisions #1 has the writer flushing every 16
   frames, so a 272-frame run fires ~17 saves and the `.bak` holds mid-track state by the second
   one. Step 5 must take a one-shot `.pretrack` backup at `gestureBegin` instead.
10. **The vision-tower graft must share the WHOLE detector**, not just `vision_encoder`: measured on
    this card, sharing only the tower still adds 788.1 MiB because `vm.to(DEVICE)` moves the
    detector's 386 M non-vision parameters too. Sharing the whole detector adds 50.7 MiB. Saved:
    737.4 MiB.
