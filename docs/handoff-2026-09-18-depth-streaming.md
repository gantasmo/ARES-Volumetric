# Handoff 2026-09-18: `ares depth` streaming, relief framing, letterbox, fill layer

Branch `feat/installer-and-import`. Nothing is committed. Two sessions worked on this the same
day; the second resumed from the first one's cutoff. docs/depth-2d-to-25d.md is the current
description of the pipeline and CHANGELOG.md `Unreleased` lists the changes; this note records
state, measurements and what is open.

## Verification state

- `tsc -b` clean. `node --test "packages/*/test/*.test.mjs"`: 95 of 95.
- `packages/encoder/test/depth.test.mjs` covers: the grown gate, guided resampling, the fill layer,
  `MeshClipWriter` against `muxClip` byte for byte (culled and sheets), `openDepthRun` with a mask,
  `DepthHistogram`, `barsFromProfile`, the letterbox end to end, the `relief.*` keys on a culled
  and a sheet clip, `temporalMedian` against a sorted reference, the cut hysteresis, and the
  stabilizer following a lasting step while removing a one-frame spike.
  `packages/core/test/player-api.test.mjs` covers `reliefFromMeta`;
  `packages/core/test/edits-dense.test.mjs` covers a mask2d camera's `fov` reaching the evaluator.
- Visual checks were screenshots in headless Chrome (`tools/relief-shot.mjs`). The user judges
  the visuals; no visual default below has been approved.

## The original failure (first session)

`ares depth` on `MOTHRA VS. GODZILLA.mp4` (5,627 frames, 518x294 maps) exited with
`cannot read depth.f32: File size (3427788336) is greater than 2 GiB`: every stage held the whole
clip. It now streams (depth-io `openDepthRun`, depth-store, `stabilizeDepthStream`,
video-frames `openRawFrames` + `encodeRawTextureGop`, muxer `MeshClipWriter`).

## Measured

| run | result |
| --- | --- |
| full film, 5,627 f, no crop | 644 s wall, peak working set 688 MB, 7.2 GB scratch, 2,058,179,701 bytes, 357 KB/f |
| full film, letterbox crop (the current `apps/demo/MOTHRA_VS__GODZILLA.ares`) | 516.5 s wall, 1,581,994,669 bytes, 274.6 KB/f, 52,032 tris/f, 217 chunks, relief pivot 1.087 (near 0.598, far 5.992) |
| same, memory | peak working set 3,181 MB. A re-run sampling `process.memoryUsage()` every 250 ms held 520 to 550 MB RSS through pass C; the peak (3,189 MB RSS, 3,023 MB of array buffers) came from the closing `info`, which read the whole 1.58 GB output and copied it (the whole-file path ran under 1.5 GiB). `info` now reads the head and then one chunk at a time: 96 MB peak, 0.95 s on that file. |
| player | loads the 2,058,179,701-byte clip whole; frame 2000 on screen about 9 s after navigation (headless Chrome, loopback) |
| 300 f of the film, crop | 71.4 MB; `--inpaint` 96.6 MB (fill 15,154 tris/f); `--snap-ramps` 72.2 MB (272 vertices snapped/f) |
| fill coverage | share of relief triangles culled across a jump over 25 % that the fill covers, frames 60 and 200: 21 % and 22 % plain, 70 % and 61 % with `--snap-ramps` |

## Done in the second session

- **Relief metadata, encoder side** (`cli.ts` `depth()`, `depth-mesh.ts` `DepthHistogram`): every
  relief writes `relief.camera`, `relief.forward`, `relief.fov`, `relief.near`, `relief.far`
  (nearest and farthest 5 % of the surface) and `relief.pivot` (their midpoint in disparity);
  sheets add `relief.slope` and `relief.depthMax`. The first version used the median depth as the
  pivot; on a frame that is mostly distant sky it sat at 4.87 of a 0.5..6 range and a near subject
  swung across the frame, which is why it changed.
- **Player framing** (`player.ts` `frameRelief`, `autoOrbitStep`): called from the constructor
  (it was only reached through `focus()`), eye exactly on the capture camera (the stand-back by
  `tan(fov/2) / tan(25°)` changed the perspective and opened a gap around every near subject),
  sway half-angle `0.06 / (pivot / near - 1)` clamped to 0.02..0.2 rad.
- **Letterbox** (`video-frames.ts` `detectLetterbox`, `barsFromProfile`; `cli.ts` `mapCrop`,
  `--crop auto|none|W:H:X:Y`, default auto): the film's picture is 1920x814+0+133.
- **Fill layer** (`depth-layers.ts`, `depth-mesh.ts` `fillLayerToMesh`): seeded only by jumps over
  25 % (`FILL_MIN_JUMP`; sky noise near the far plane was flooding bands), 16 Gauss-Seidel sweeps
  over the band (`relaxBand`), anchors at their own depth, band-to-band edges never cut.
  `--snap-ramps` (`snapRamps`, opt-in) moves vertices partway down a silhouette ramp to the side
  they match in colour.
- **Dev server** (`tools/serve.mjs`): static files stream with Range support (`sendFileRange`,
  also used by `/depth/source`). `/depth-convert` now sends `subject` to the service (it was
  dropped), reports the mask pass as its own progress stage, and passes `decimate`, `inpaint`,
  `inpaintBand`, `noGuided`, `guideSigma`, `snapRamps`, `crop`.
- **Depth card** (`apps/demo/depth-card.js`): `video-*` models first in the service list, subject
  prompt (service only), guided, fill, band, snap, decimate (disabled with sheets), letterbox.
  Measured layout: no overflow at 1920x1080 with every group open; at 1366x768 the card region
  already overflowed by 11 px with every group closed, and each open group adds its one new row.
- **`info`** reads the head and one chunk at a time.
- **Messages** that told a person to act now state the fault: `cli.ts` (two ffmpeg messages),
  `apps/demo/track.js` preflight (test updated), `tools/msix/build.mjs` (three; the server still
  matches `Developer Mode is off`), `tools/launch.ps1`.
- **Review of the Video-Depth-Anything code the dead agent left** (`depth_vda.py`, `depth.py`,
  `depth_subject.py`): `WindowAligner` and `_frames_vda` reproduce upstream's
  `infer_video_depth` window count, keyframe slots, least-squares fit, cross-fade weights and
  padding for clip lengths 20, 30, 40, 44 and 54 (checked against the clone's
  `video_depth.py` and `utils/util.py`). `requirements.txt` gained `einops==0.8.2`. The 409
  `missing` paths were not exercised live.

## Open

0. **Reported by the user after the second session.** In the viewer the background looks huge and
   the foreground (characters) looks tiny. Playback jitters. The user asked for both to be resolved
   and the jitter smoothed. What was found and changed (a third pass, same day):
   - Off the capture camera, content at depth z is z times larger than it looks in the picture.
     The relative depth range was 0.5..6 m (12:1), and the viewer's wheel dollied the eye back
     from the capture camera, which shrinks near content faster than far content. The default is
     now 2..6 m (3:1, `cli.ts`), a relief opens through the FOV that fits its whole picture
     (`relief.aspect`), and the wheel zooms a relief by FOV (`player.ts` `zoomRelief`,
     `OrbitState.fov` in `camera.ts`, carried into mask2d cameras in `edits.ts`).
   - Playback timing is clean: `apps/demo/verify/playback-trace.html` on 8 s of the film showed
     60 Hz frames, 24 fps content in the 3:2 cadence, texture and geometry on the same frame on
     473 of 476 frames, one 100 ms hitch. The jitter is geometric.
   - Geometric jitter: still pixels changed by 5.7e-3 of the depth range per frame at the 90th
     percentile raw and 4.4e-3 after the old stabilizer; now 2.2e-3 (`alphaStatic ^ strength`).
     Grid edges cut for exactly one frame fell from 376 to 143 per frame (5-frame temporal median
     after the EMA in `depth-stabilize.ts`, per-edge cut hysteresis 0.67..1.5 in `depth-mesh.ts`).
     `apps/demo/verify/relief-flicker.html` on frames 20..80 with a fixed unlit checker: 0.848,
     0.885 and 1.381 luma levels per frame before, from the capture camera and 0.1 and 0.2 rad
     off-axis; 0.550, 0.550 and 0.781 after. Cost of the 5-frame median: an object that crosses a
     pixel within two frames loses its depth there (`--median 3` limits it to one frame).
   - Material to judge: `docs/img/relief-proportions-phone-f125.png`, `-f250.png`,
     `docs/img/relief-proportions-mothra-f480.png`, `-f1440.png`, `-f2160.png`, `-f4560.png`;
     top row the previous encode, bottom row the new one; columns capture view, orbit 0.15 rad,
     orbit 0.3 rad raised 0.08, zoom out (the old wheel's 2x dolly against the new 0.5x FOV zoom).
     Clips: `apps/demo/MOTHRA_VS__GODZILLA-v3.ares` (new defaults, 536.3 s encode, 636 MB peak
     working set, 1,664,490,917 bytes) next to `apps/demo/MOTHRA_VS__GODZILLA.ares` (previous), and
     `apps/demo/_tmp-phone-v3.ares` next to `apps/demo/_tmp-phone-cull.ares`.
1. **Visual judgement by the user.** Comparison sheets: `docs/img/relief-variants-mothra-f60.png`
   and `-f200.png`, rows top to bottom `crop`, `snap`, `fill`, `fill-snap`, columns sway −0.073,
   0, +0.073 rad. Clips: `apps/demo/_tmp-mothra300-{crop,snap,fill,fill-snap}.ares` and the full
   `apps/demo/MOTHRA_VS__GODZILLA.ares`. Decisions pending: eye on the capture camera, pivot and
   sway size, letterbox crop on by default, whether `--snap-ramps` and `--inpaint` become defaults.
2. **Dev server restart.** The server on 127.0.0.1:8137 (PID 10524, started 05:52) runs the
   `serve.mjs` from before this session: no streaming, no new `/depth-convert` parameters. The
   packages it serves (`packages/core/dist`) are rebuilt, so the player changes are live on reload.
3. **Range-request loader in the player** (docs/depth-2d-to-25d.md handoff item 1).
4. **Near, far and FOV from the picture** (item 2).
5. **Inference size**: 518 wide against the models' 518 short side (item 8).
6. **SAM service instances** left by the first session: 127.0.0.1:7263 (PID 4816, the port
   `SAM_URL` defaults to, so the dev server uses it) and 127.0.0.1:7273 (PID 54832).
7. **Scratch clips** to delete once judged: `apps/demo/_tmp-mothra300-{crop,snap,fill,fill-snap,v3}.ares`,
   `apps/demo/_tmp-phone-{cull,sheets,v3}.ares`, `apps/demo/_tmp-range-{0.5,1.5,2,3}.ares` (the same
   300 film frames at near 0.5, 1.5, 2 and 3 m with far 6 m and the earlier smoothing, for choosing
   the depth ratio), `G:\ares-out\mothra300.ares`, each with its `.meta.json`. Once the new
   defaults are accepted, `apps/demo/MOTHRA_VS__GODZILLA-v3.ares` replaces
   `apps/demo/MOTHRA_VS__GODZILLA.ares`.
8. **Depth run directories** in `%TEMP%\ares-depth-test\` (nine runs, several GB): keep until the
   visual decisions are made; `mothra-full-video-small` is the acceptance run.
