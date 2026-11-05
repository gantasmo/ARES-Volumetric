# Handoff: 2D video to one volumetric clip per person

Written 2026-09-19 at the end of the session that built this path. The feature works end to end;
this note says what runs, what it measured, what broke, and what to do next.
Design and contracts: [video-to-4d-people.md](video-to-4d-people.md). The model's own install and
settings: [../tools/4danyone/README.md](../tools/4danyone/README.md).

## What works

One run of `/avatar-convert` turned `C:\Users\dtruj\Videos\ARES4D\VID_1782289226553.mp4` into
`apps/demo/_tmp-route4d-p1.ares` (45 frames, 22.9 MB) in 1216 s, unattended:

| stage | measured |
| --- | --- |
| mask, 200 frames | 1 person found, window chosen at frames 112..156, whole body in view on 45 of 45 |
| generate, 6 views, 45-frame window | denoise 4 steps at 42 s, peak 10.0 GB over both cards |
| mesh, 45 frames | 3.7 s per deformed frame, 10 to 12 s per keyframe, 40,000 faces, texels seen 64 to 71 % |
| encode | 22.9 MB, 509 KB per frame, geometry temporal I+P on all 45 frames, zero intra |

The Convert card drives the same chain: `completion` → `4DAnyone views`, then the Views block
(views, pitch, window, voxel, faces, re-key, people, minimum height). Settings shows the three
`4danyone-*` rows as status only.

Pieces: [../tools/sam-service/avatar.py](../tools/sam-service/avatar.py) (ids, cutouts, per-person
clips), [../tools/sam-service/avatar_mesh.py](../tools/sam-service/avatar_mesh.py) (views to
textured frames), [../tools/avatar-run.mjs](../tools/avatar-run.mjs) (the chain and every default),
the `/avatar-convert` route in [../tools/serve.mjs](../tools/serve.mjs), and the card in
[../apps/demo/depth-card.js](../apps/demo/depth-card.js).

## The access violation: cause found, clip now converts

`C:\Users\dtruj\Videos\ARES4D\1_c.mp4` (named `VID-20260819-WA0002.mp4` at the time; 496x368, 125 frames) converts. Measured
2026-09-19 through `/avatar-convert` at a 45-frame window, 6 views, pitch 15:
`apps/demo/wa0002-p1.ares`, 45 frames, 1750.8 s.

The failure was device memory, not a bad pointer. Two things hid that:

1. **The server was running code older than the fix.** The process on 8137 had started at 15:13 and
   `serve.mjs` was last written at 16:01, so the `samStop` call added after the people stage was not
   in the running instance. A re-run from the card would have failed for the original reason.
2. **With the service stopped, the same clip at window 61 raises a clean `torch.OutOfMemoryError`**
   at `wan_video_dit.py:263`, the attention output projection: GPU 0 at 11.00 GiB with 0 bytes free,
   9.36 GiB held by PyTorch, a 578 MiB allocation refused. The silent `0xC0000005` was the same
   exhaustion reached with 2 to 3 GB already gone to a resident SAM 3, where the driver faulted
   instead of letting the allocator report.

A 61-frame window does not fit two 11 GB cards at the 704x1280 raster. The default is now 45, in
`AVATAR_DEFAULTS.frames` and on the card, which is the value [../tools/4danyone/README.md](../tools/4danyone/README.md)
already documented.

### What the placement leaves on card 0

`windows-turing.patch` puts blocks 0..14 **and every non-block child** (patch, text and time embeds,
head) on `cuda:0`, and only blocks 15..29 on `cuda:1`, so card 0 carries the heavier half.
`FDANYONE_SPLIT_AT` moves that boundary and is unset; [../tools/avatar-run.mjs](../tools/avatar-run.mjs)
never passes it. Lowering it is the untried lever for a window longer than 45, and it costs one run
to measure.

## The re-key threshold arrived as zero

The first successful conversion keyed all 45 frames and coded geometry intra-only. The deform was
never attempted: `avatar_mesh.py` keys unconditionally when `--rekey <= 0`, and the route handed it
0. `Number(null)` is 0, so the route's `num()` helper read an **absent** knob whose range contains
zero as zero rather than as its default. `rekey` (0..1) became "mesh every frame afresh" and `pitch`
(-15..45) a camera ring at eye level. Fixed in `serve.mjs`: a missing or empty value now takes the
default. The card always sent every knob, so this only reached a caller that omitted one.

Both meshes built from the same generated views, 45 frames:

| `--rekey` | frames | mesh stage | container | per frame | geometry |
| --- | --- | --- | --- | --- | --- |
| 0 (the bug) | 45 key | 562 s | 20.73 MB, crf 32 | 471.8 KB | intra-only, 45 intra |
| 0.02 | 10 key, 35 deformed | 326 s | 15.52 MB, crf 30 | 353.2 KB | temporal I+P, 0 intra |

The fit residual on the deformed frames reads p50 9.2 mm, p95 18.6 mm, max 19.0 mm against the 20 mm
threshold, and re-keys land at frames 7, 14, 20, 25, 29, 32, 35, 39, 44. That residual is vertex
distance to the hull only: the deformed frames of this run are scrambled (see the audit below), so
the smaller file is not a usable result.

## Two service faults found on the way

- `samStop` could not tell a kill from a no-op. It discards the PowerShell one-liner's output and
  logged "stopped" unconditionally, so a service that kept the port (no listener found, an owner
  that is not `python*`, a denied `Stop-Process`) left the route starting generation on a card still
  2 to 3 GB short, with the card log reading "stopped". It now returns whether the port went quiet,
  and `/avatar-convert` fails with a named error instead of generating.
- The `/sam` proxy restarts the service on any refused POST, so touching a SAM route from the open
  tab mid-run put SAM 3 back on `cuda:0` under a generation that had been given the whole card. A
  run now holds the service down: `samEnsure` refuses while `samHold` is set and the proxy does not
  auto-start, both naming the run.
- `/avatar-convert` was the one machine-acting route missing from the same-origin `GUARDED` list.

## Audit of this path (2026-09-19 and 20): what it measured and where the work goes next

The output of this path was rejected on sight, so every stage was measured on the 24-view, three-pitch
set of `1_a.mp4` (frames 0, 11, 22, 33, 44 of the generated window). Scripts and numbers were kept
outside the repo; the findings are recorded here.

**The generated views do not show one 3D object.** Two neighbouring cameras place the same textured
point at depths 45 to 55 mm apart (median), within 10 mm for 6 to 11 % of points. The same test on
views rendered from one known 3D surface through the same cameras gives 5 mm and 49 to 62 %. Cameras
denoised in the same group of six disagree as much as cameras never grouped together. Upstream's own
benchmark puts each view at 25.5 dB PSNR against real footage, which bounds how closely two views
can agree. 14 % of each view's silhouette is absent from the carved shape: a part that different
views draw in different places is carved away.

**The mesher cannot reproduce its own inputs.** Re-rendered from the input cameras the mesh scores
16.7 dB PSNR and 0.55 SSIM (median over 24 cameras). From the front camera the blended bake keeps
14 % of the view's fine-detail energy; best-single-view selection keeps 96 % and scores lower
(15.9 dB) because adjacent texels come from views that disagree. Leaving a camera out of the bake
costs about 1 dB, so view count is not the limit.

**The shape is a visual hull.** `1_a`: 138 to 146 L at 24 views, 116 to 121 L at `--tolerance 0`,
about 200 L at 6 views, against a fitted unclothed body of 82.4 L. `1_c`: 344 L at 6 views against
109.3 L. Upstream carves with strict intersection at 2 cm and uses the hull only to seed splat
training. No mask of the 24 views shows a gap between the crossed legs of `1_a`. MoGe-2 depth was
tested as a remedy and lands no nearer to the depth where the views agree than the hull does
(42.5 mm against 35.0 mm).

**The deform does not hold.** After `wrap()` was moved onto the welded mesh the UV seams stay closed
(0.0 mm), and flipped faces still reach 10.7 %, edge stretch p99 23x and mesh volume 415 L against a
200 L hull. The fit residual passes those frames: it measures vertex distance to the hull and cannot
see a scrambled surface. `tools/sam-service/test/mesh_health.py` reports the measures that can. The
Convert card sends `re-key 0.02`, so card runs of this path produce those frames.

**Other findings.** xatlas at default options cuts each frame into 945 to 1335 charts (median chart
2 faces). The route never passes `--fps`, so a 24 fps source plays 25 % fast. Every input so far is
under upstream's stated requirement (1080p or higher, 9:16, at least 121 frames). The card defaults
to 6 views on one pitch ring, which upstream labels an initial test. Decoding (1430 s of a 2407 s
24-view run) and meshing use `cuda:0` only. The container, the texture codec (40.1 dB) and the camera
matrices (equal to upstream's) were checked and are not at fault. A 45-frame window is 1.5 s; one
24-view window costs 42 minutes, and upstream's method for joining windows over time is not public.

**Direction.** This path rebuilds the whole person from generated views, the side the camera filmed
included. The recipe that produced the accepted RGBD results keeps the capture's own pixels for
everything the camera saw ([rgbd-rebuild-pipeline.md](rgbd-rebuild-pipeline.md)), and its 2D-video
port already runs whole clips on both cards ([depth-2d-to-volumetric.md](depth-2d-to-volumetric.md)):
`1_a.mp4`, 248 frames, 531 s, judged substantially better on sight. The work continues there:

1. Camera motion. That path builds each frame in its own camera coordinates, so a tilting camera
   moves the subject. On `1_a` the camera tilts up 43 degrees by frame 169; rotation recovered from
   background feature tracks (residual 0.58 px) cuts the body's vertical travel from 139 cm to 11 cm.
2. Subject span. The subject leaves `1_a` at frame 170; the body fit loses its depth from frame 148
   (48 % of the body in frame) and the tracker reseeds on a false detection at frame 193.
3. Body texture. The MHR asset carries its own UV layout (`character_torch.mesh.texcoords` in
   `mhr_model.pt`: 19,455 coordinates, 11 islands, 93 % of the atlas) which nothing uses yet; the
   body is textured today with a 3x3 texel patch per face. A character card baked into that layout
   from real frames across the whole clip, with generated views only where no frame saw the body, is
   where 4DAnyone's output belongs.

## Test material

Clips: `C:\Users\dtruj\Videos\ARES4D`, relabelled on 2026-09-19: a name starting with `1` has one
focal person, a name starting with `2` has two. `1_c.mp4` is the robe clip measured above (496x368,
person 221 px tall). `1_a.mp4` (480x852, 248 frames) is the audit clip: the camera tilts from the whole
body up to a sign and the person leaves the frame at frame 170. `2_b.mp4` is an outside view of the
`2_a.mp4` scene. The tuning clip under **What works** (`VID_1782289226553.mp4`) has been removed from
the set.

Results kept: `apps/demo/_tmp-route4d-p1.ares` (the route's own run),
`apps/demo/_tmp-4d-vid1782.ares` (the same clip, command line, 8 frames),
`apps/demo/wa0002-p1.ares` (VID-20260819-WA0002, the all-key mesh) and
`apps/demo/wa0002-p1-rekey.ares` (the same generated views meshed at `--rekey 0.02`). The run
directory behind both is `C:\Users\dtruj\AppData\Local\Temp\ares-avatar-94UiF5`.
Renders: `C:\Users\dtruj\AppData\Local\Temp\ares-4d\route-shots.png` and `views-sheet.png`.

## Correctness check that does not need a GPU run

`avatar_mesh.py` is verified against a synthetic scene with a known answer: an ellipsoid body and a
sphere head ray-traced through 4DAnyone's own camera ring
([../tools/sam-service/test/synth_rig.py](../tools/sam-service/test/synth_rig.py), then
[verify_synth.py](../tools/sam-service/test/verify_synth.py)):

```sh
cd tools/ext/4danyone
venv/Scripts/python.exe ../../sam-service/test/synth_rig.py <views-dir> 12 6
venv/Scripts/python.exe ../../sam-service/avatar_mesh.py <views-dir> <frames-dir> --mask luma
venv/Scripts/python.exe ../../sam-service/test/verify_synth.py <frames-dir>/mesh-f00001.obj <frames-dir>/atlas-f00001.png 0
```

Surface error came back p50
4.0 mm and p95 9.9 mm on a keyframe, p50 4.3 mm and p95 12.5 mm on a deformed frame, with the
texture on the correct stripe for 92 % of vertices. Re-run that after any change to the carve, the
deform or the bake.
