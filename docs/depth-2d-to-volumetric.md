# 2D video to full volumetric

Status 2026-09-19. A 2D video of a person becomes a `.ares` clip whose per-frame mesh covers the
subject from every side. The side the camera saw is the capture's own relief, in metres, with
normal-map detail. The side it never saw is a SAM 3D Body (MHR) mesh anchored behind that relief,
and its texture is gathered from every frame in which that part of the body faced the camera. The
clip plays as an ordinary turntable in the player.

This extends the 2.5D pipeline ([depth-2d-to-25d.md](depth-2d-to-25d.md)): the same run
directory, the same `ares depth` streaming passes, and the same dev server route, each with a
volumetric switch.

## Pipeline

```
video --> service (tools/sam-service)                        --> run dir --> ares depth --volumetric --> clip.ares
          mask     SAM 3 text prompt ("person")                  depth.json    A  RGB gate, subject restriction, body colours
          depth    Video-Depth-Anything (subject on black)       depth.f32     B  stabilize
          geometry MoGe-2, unmasked frame, one worker per GPU    mask.u8       F  metric fit, body anchor
          body     SAM 3D Body from the mask's box, per GPU      metric.f32    C  detail, shell + body + backing,
                                                                 normals.i8       atlas = frame + back region
                                                                 body.f32 ...  finish: volumetric.* metadata
```

1. `GET /depth-convert?...&volumetric=1` (tools/serve.mjs) installs the components, starts the
   service and submits `/depth/run` with `volumetric: true`.
2. The service runs the mask pass, the depth pass, then the geometry and body phases
   (`depth_volume.py`, `volume_worker.py`), each on every CUDA device at once.
3. `ares depth <video> --depth <run> --volumetric` (packages/encoder/src/cli.ts, volumetric.ts)
   builds the clip.

## Run contract extension (ares-depth/1)

Every key is optional and additive; `openDepthRun` (packages/encoder/src/depth-io.ts) validates
each named file against the frame count the way it validates the mask. All per-frame files share
depth.f32's frame order, map size and orientation (row 0 = top). The service preallocates every
file at full size and adds a key to depth.json (tmp file + `os.replace`) only once its phase has
filled it, so a killed job leaves a valid partial run.

| key | file | layout |
| --- | --- | --- |
| `intrinsics` | `intrinsics.f32` | `{fx, fy, cx, cy}` normalized (fx = focal_px / W, fy = focal_px / H), the clip median of MoGe-2's per-frame estimates; the file holds frames x 4 float32 LE, NaN = no estimate. `source`, `fovY`, `fovX`, `frames` |
| `metric` | `metric.f32` | frames x H x W float32 LE, depth z in metres from MoGe-2 on the unmasked frame, area-resized over valid pixels, 0 = invalid; each frame's z belongs to that frame's own focal in `intrinsics.f32`. `model`, `units`, `dtype`, `msPerFrame`, `validFraction`, `medianInMask`, `workers` |
| `normals` | `normals.i8` | frames x H x W x 3 int8, round(n * 127), unit normals in OpenCV camera space (camera-facing n.z < 0), 0,0,0 = invalid. `space`, `meanZInMask` |
| `body` | `body.f32`, `body-faces.u32`, `body-valid.u8`, `body-boxes.f32` | frames x V x 3 float32 LE, `pred_vertices + pred_cam_t` in OpenCV camera metres with the map intrinsics; F x 3 uint32 LE faces written once; one byte per frame (1 = fitted on this frame, 0 = copied from the nearest valid frame); frames x 4 float32 box edges in map pixels (NaN = no subject). `vertices` 18439, `faceCount` 36874, `engine`, `model`, `fp16`, `inference`, `validFrames`, `backfilled`, `conventionIoU`, `iouMean`, `iouMin`, `copiedMaskFrames`, `workers` |

`workers` lists one record per worker process of the phase: `gpu`, the frame range `start`..`end`,
`frames` emitted, `retry` (a re-run of a lost worker's frames), `lost` (the reason, or null),
`msPerFrame`, `loadMs`, `wallMs`, `peakMiB`, `reservedMiB`. The mask pass of a subject run adds
`mask.detected` (`mask-detected.u8`, one byte per frame: 1 where SAM 3 found the subject, 0 where
the frame repeats a neighbouring frame's mask) beside `mask.filled`, and `mask.ids`
(`mask-ids.u8`, frames x H x W uint8: the SAM 3 tracker's object id on each mask pixel, the object
with the highest mask logit where two overlap, 0 elsewhere; a copied frame copies its ids). `mask.objects`
lists each id's `frames` (detected), `first`, `last` and mean `coverage`. The tracker holds at most
`DEPTH_SUBJECT_MAX_OBJECTS` (4) objects, and after it loses every object and reseeds, a returning
person gets a new id. Checked 2026-09-19 on 60 frames of the skateboard clip: 4 ids, each on the
same person on every frame, and `mask-ids.u8 > 0` equal to `mask.u8 > 0` on every pixel.

Coordinates: OpenCV camera space (x right, y down, z forward, metres); pixel (u, v) =
(fx*x/z + cx, fy*y/z + cy) in map pixels with pixel centres at integer + 0.5. The encoder converts
to ARES world space (x, -y, -z) only when it emits a mesh.

`writeDepthRun(dir, meta, maps, extras)` writes any of these arrays for tests.

## Service phases

`DepthRunRequest.volumetric` needs `subject` and a CUDA device (400 otherwise; with
`DEPTH_VOLUME_GPUS` set to indices of no card, the 400 names the value and the cards there are) and
every volume component (409 with the ids otherwise). Components on disk that fail the cached import
probe (`volume_worker.py --probe`, a child process) also answer 409, with the import errors; a failed
probe runs again on the first `/depth/health` 120 s later. After the depth pass the resident depth
model is freed and two phases run, one `volume_worker.py` process per GPU with
`CUDA_VISIBLE_DEVICES=k` (SAM 3D Body moves its batches to the literal device `"cuda"`), frames
split into contiguous shards and written positionally:

- **geometry**: MoGe-2 (`Ruicheng/moge-2-vitl-normal`, fp16 autocast) on the unmasked frame decoded
  at up to 1280 on the long side, metric depth and normals resized to the map size, the normalized
  intrinsics per frame.
- **body**: SAM 3D Body (`jetjodh/sam-3d-body-dinov3`, FP16_TYPE float16, inference `full`) on the
  frame at up to 1920, from a box per frame: the 8-connected mask component that overlaps the
  previous frame's choice most, padded 5 % per side. A frame whose mask the mask pass copied
  (`mask-detected.u8` 0: before the subject's first detection, or while the tracker has lost it)
  gets no box: SAM 3D Body returns a mesh for any box, and the copied mask may hold no one. The
  model's camera is the clip-median intrinsics times the decode size. A frame with no box, no
  output, a non-finite value or a vertex nearer than 0.05 m takes the nearest valid frame's mesh
  (`body-valid` 0). The first valid frame's projected footprint must reach IoU 0.3 against the mask
  (`body-convention` otherwise).

A worker that errors or exits before finishing (an out-of-memory on one card, say) costs only the
frames it had not emitted: they are queued and re-run on another GPU once that GPU's own worker has
exited, while the other workers carry on. A shard lost twice, or lost with no other GPU left, fails
the job with `worker-failed`. Checked 2026-09-19 on both RTX 2080 Ti with a gpu1 worker started on a
card that does not exist: both phases of a 20-frame run finished on gpu0 (the lost shard's frames
after gpu0's own), with metric depth identical to the two-GPU run.

`/depth/status` reports `geometry` and `body` in `phases` with `done`, `total`, `msPerFrame`,
`gpus` and `framesPerSecond`. Error codes: `body-convention`, `body-not-found`, `geometry-failed`,
`worker-failed`, `no-cuda`, `missing-components`.

## The encoder: `ares depth --volumetric`

```
ares depth <video> --depth <run-dir> --volumetric [-o out.ares]
  [--no-detail] [--fit-smooth 8] [--anchor-smooth 0] [--grid N] [--fov deg]
  plus the texture, audio, stabilize, gop, crop and transform flags of the relief path
```

The run must carry `mask`, `metric`, `body` and `intrinsics` (the error names the missing ones);
`normals` are optional. `--sheets`, `--inpaint`, `--decimate` and `--snap-ramps` are refused;
`--near`, `--far`, `--edge` and the guided-resample flags do not apply. The mesh is built at the
map size unless `--grid` is given. The focal comes from the run: after a letterbox crop the focal
in map pixels is unchanged and the principal point moves by the crop origin. `--fov` replaces the
focal and scales the body's z by the focal ratio so every vertex keeps its projected pixel.

MoGe-2 infers each frame's focal together with its depth, and a frame whose focal came out long came
out far: on the measured clip the normalized fy per frame spans 1.4561 to 1.9156 around the clip
median 1.6493, and log fy against log subject depth correlates at 0.926. The shell, the body fit and
the normal gradients use the clip focal, so the encoder multiplies frame t's metric z by
fy_clip / fy_t (`focalScales`; 1 on a frame without an estimate), which keeps every pixel's lateral
position. On the measured clip that took MoGe-2's subject depth step from 92.5 to 68.6 mm per frame
(p50) and the body's largest front offset from the shell from 85.1 to 57.3 mm.

| pass | per frame | module |
| --- | --- | --- |
| A | RGB motion gate (as the relief path); the body rasterized at the map size; the subject mask restricted to that footprint dilated by 2 % of the map width (a second person in the mask has no body and is dropped); the body's visible vertices (facing the camera at cos > 0.2, within max(2 cm, 1 %) of the body's z-buffer, on the mask) accumulate colour weighted by cos squared | `SubjectPass`, `BodyColorAccumulator` |
| B | stabilizer, unchanged | depth-stabilize.ts |
| F fit | `fitTileField(stabilized disparity, metric, restricted mask)`: a robust global fit (linear or disparity model, lower median residual wins) and a per-tile field blended toward it; then `smoothTileFields` over +-8 frames | `fitPass`, depth-metric.ts |
| F anchor | the body averaged over +-2 frames (Gaussian sigma 1), rasterized, and scaled about the camera so its front surface lies on the fitted shell (median of shell z / body z over the shared pixels); `smoothSeries` with its +-2 median and no Gaussian | `anchorPass`, depth-body.ts |
| vote | once per clip, five frames: `gradientVote` of the normals as stored against y flipped | `chooseNormalAxis` |
| C | z = zb * exp(r), zb the fitted depth and r the normal-map residual (screened Poisson on log depth), median of three frames with r one frame ahead; shell mesh (5 cm z span per triangle, a face seen edge-on dropped, surface pieces under the component minimum dropped); the anchored body carved where the shell covers it, pushed along the camera ray behind it, flaps facing the camera dropped; a backing sheet behind shell cells with no body behind them; merged, converted to ARES space | `DetailTrack`, `assembleVolumetricFrame` |

The body's back texture is baked once after pass A: `BodyColorAccumulator.finalize` fills the
vertices no frame saw from their 1-ring neighbours, and `bakeBackAtlas` lays one 3x3 texel patch
per MHR face into a region `texSize` wide under the frame region. The atlas is `texSize` x
(`texSize` + region height); the back region is the same pixels in every frame. Body triangles are
unwelded (3 vertices per face) so each corner carries its own patch uv, and the topology changes
per frame, so geometry is coded intra.

Progress lines add the stages `fit` and `anchor` to the contract (`[ares] progress <stage> i/n`).

**Metadata.** A volumetric clip carries no `relief.camera` or `relief.forward`, so the player
frames it as a turntable. The superblock keeps `volumetric.method` (`shell+sam-3d-body`),
`volumetric.camera`, `volumetric.forward` and `volumetric.fov` (degrees, vertical). The sidecar
gets a `volumetric` block: intrinsics (map pixels and normalized, and `metricFocal`: the per-frame
z factors' minimum, median and maximum), grid, mask coverage before and after the restriction, fit
(model counts; `medresGlobalP50` and `medresTiledP50` over every fit sample, `medresInliersP50` for
the global fit over its own inliers; the temporal steps), anchor (scales, residual,
front offset), detail (vote, rms over the solved cells, CG iterations), body (validity counted over
the frames written, sign, IoU, the service's per-worker records), back texture
(observed, filled, region, atlas), per-frame mesh means and per-pass times.

## Dev server and Convert card

`/depth-convert` takes `volumetric=1` (0 or 1, validated): service engine only, `subject` defaults
to `person`, and `sheets`, `inpaint`, `snapRamps` and `decimate` are refused. On a machine where
nvidia-smi reports no NVIDIA GPU the stream ends with an error before any component installs. Before
submitting, the route reads `/depth/health`: a service without the `volume` key (started from code
that predates this mode, which would drop the field and run a plain subject job) is restarted once,
and the job waits for the import probe and needs `volume.available`. A job status without
`volumetric: true` cancels the job. The route adds the
installer's `VOLUMETRIC_COMPONENTS` (`moge-code`, `moge-2`, `sam3d-body-code`, `sam3d-body`,
`dinov3-hub`, `sam3d-body-runtime`) to the job's components, passes `volumetric` to `/depth/run`,
forwards the `geometry` and `body` phases as `progress` stages with the GPU count and frames per
second, labels the service's error codes, and passes `--volumetric` to the encoder with no near,
far or edge. The route's request block in the sidecar sits under `request` (which now also holds
`video`); the encoder's own `source` block is kept.

The Python packages of the two phases are listed in `tools/sam-service/requirements-volumetric.txt`,
apart from `requirements.txt`: pip resolves a whole `-r` file before installing anything, and the
MoGe git URL would stop the base environment's install on a machine without git. The installer's
`moge-code` row pip-installs its MoGe clone and the `sam3d-body-runtime` row (pip only) the eleven
pins, both with the private MinGit on PATH; neither reads the file.

The Convert card's grid row has a `completion` select (`none`, `SAM 3D Body`), enabled for the
service engine on a machine with a CUDA device (`/depth/engines` `cuda`, and the running service's
selected device count); without one the engine note reads `completion: CUDA device absent`. The
progress bar of a volumetric run stays at 0 until the mask pass reports. Switched on, it sets the subject to `person` when blank, the inference width to
924, the grid to `map` (the depth map's own size), and disables the relief-grid controls (FOV,
range, edge, sheets, decimate, fill, snap, guided). History recipes record `volumetric`, grid `map`
and a blank FOV, and reopen with completion on.

## Measured (2026-09-19, 2x RTX 2080 Ti 11 GB, Node 22.19)

Source: a 1920x1080 30 fps phone clip of a skateboarder, map 518x294, Video-Depth-Anything Small.

**Encoder on a 90-frame run** (the service run of the engine work, `ares depth --volumetric`
defaults, VP9 CRF 30, texture 1024; re-measured after the per-frame focal normalization):

| figure | value |
| --- | --- |
| intrinsics | fx 480.55, fy 484.88 map px; fovY 33.73 deg, fovX 56.65 deg |
| subject mask | 4.051 % of the map, 3.326 % inside the body footprint (10 px dilation) |
| metric focal | per-frame z factor 0.8609 to 1.1327 |
| fit | 90/90 frames (23 linear, 67 disparity; clip model disparity); median residual p50 over every fit sample 58.3 mm for the global fit, 54.7 mm for the tiled field (40.8 mm for the global fit over its own inliers) |
| temporal | MoGe-2's subject median depth moves 68.6 mm per frame (p50; 92.5 mm before the focal normalization); the fitted shell's 9.4 mm; shell to per-frame MoGe-2 p50 91.0 mm |
| anchor | 90/90 frames, scale p50 1.115 (1.056 to 1.182), anchor residual p50 47.4 mm over 4,200 pixels, body front to shell p50 9.3 mm, max 57.3 mm |
| normals | vote 0.029 as stored, 0.010 y flipped; detail rms p50 0.0041 (log depth, over the solved cells), 73 CG iterations |
| back texture | 16,062 of 18,439 vertices observed, 2,377 filled; region 1024x328, atlas 1024x1352 |
| mesh per frame | 20,267 triangles: shell 7,706, body 11,114, backing 1,448; 22,696 body triangles carved as covered, 3,064 flaps, 8,618 vertices pushed |
| size | 36,398,607 bytes, 395.0 KB/frame (texture 34.3 KB/frame), intra-only |
| times | pass A 1.13 s, stabilize 0.62 s, fit 0.34 s, anchor 0.78 s, pass C 8.16 s, finish 0.85 s |

Over the same samples the tiled field's median residual is 3.6 mm below one affine per frame; the
inlier figure is lower because the trim removed the samples the global fit explains worst.

**Smoothing radii** (the same 90 frames, `volumetric.ts` constants; measured before the per-frame
focal normalization):

| tile-field radius | shell median-depth step p50 | shell to per-frame MoGe-2 p50 |
| --- | --- | --- |
| 0 | 64.1 mm | 64.0 mm |
| 4 | 27.5 mm | 98.6 mm |
| 8 (default) | 12.7 mm | 111.1 mm |
| 16 | 11.3 mm | 116.5 mm |

| anchor Gaussian radius | body front to shell p50 / max | body centroid step p50 |
| --- | --- | --- |
| 0 (default) | 9.2 / 69 mm | 18.7 mm |
| 4 | 17.6 / 103 mm | 22.1 mm |
| 8 | 29.6 / 145 mm | 25.4 mm |

The fitted shell is already smooth in time, and the per-frame scale is what keeps the body's front
on it: a Gaussian on the scale puts SAM 3D Body's own depth jitter back.

**Whole route on 150 frames** (`/depth-convert?...&volumetric=1&subject=person&model=video-small&maxFrames=150`,
SAM service started by the route; measured before the per-frame focal normalization): 396.8 s from
request to `done`, of which the service job took 320.1 s.

| phase | figure |
| --- | --- |
| mask | 150 frames, 508.0 ms/frame, 77.3 s wall, 4 instances of "person", coverage 6.12 % |
| depth | 7.43 ms/frame |
| geometry | 2 GPUs, 190.6 and 185.1 ms/frame, 8.7 s load, 15.5 s wall after load, 9.69 frames/s, 2,978 MiB peak per worker; fovY 34.23 deg, fovX 57.40 deg, subject median depth 5.107 m |
| body | 2 GPUs, 2,087.1 and 2,159.8 ms/frame, 34.5 s load, 164.8 s wall after load, 0.91 frames/s, 3,470 MiB peak per worker; 150/150 fitted, footprint IoU 0.844 on frame 0, mean 0.806, min 0.621 |
| encoder | mask 6.122 % -> 5.463 % inside the body footprint; fit 150/150 (49 linear, 101 disparity), tiled residual p50 43.3 mm; subject depth step p50 83.1 mm in MoGe-2, 26.3 mm in the shell; anchor scale p50 1.111 (1.016 to 1.199), body front to shell p50 9.1 mm, max 86.0 mm; 17,554 vertices observed, 885 filled |
| mesh per frame | 28,612 triangles: shell 13,590, body 12,308, backing 2,714 |
| encoder times | pass A 1.77 s, stabilize 0.86 s, fit 0.82 s, anchor 1.84 s, pass C 15.81 s, finish 2.80 s |
| clip | 71,058,836 bytes, 462.6 KB/frame (texture 33.3 KB/frame), 150 frames, VP9 1024x1352, Opus |

A 30-frame run of the same route took 110.1 s, 35 s of it the two body workers' model load.

## Lineage

The machinery is a port of the Depthkit "3Dify" lab scripts, which rebuilt RGBD captures (a sensor
depth half) into volumetric clips; here a monocular model and MoGe-2 take the sensor's place:

| lab script | here |
| --- | --- |
| `depthkit-extract7.mjs` robust fit, tiled field, normal gradients, gradient vote, screened Poisson detail, temporal median, shell mesher | depth-metric.ts, `shellMesh` |
| `dk-body4d-build.mjs` body anchoring | `anchorScale` (a scale about the camera; the lab translated to medians, which moved the silhouette) |
| `dk-hybrid-build.mjs` carve, push, backing | `assembleHybrid` |
| `asset-bake-atlas.mjs` patch atlas | `bakeBackAtlas` |

Changes against the lab, each for a recorded defect or a missing input: the target of the fit is
MoGe-2 metric depth and its fields are smoothed over time (the sensor did not flicker); the flap
filter uses the mesh's measured outward sign (the lab's test selected the wrong set on a z-negated
mesh); a pushed vertex moves along its camera ray; the backing's rim distance is uncapped (the lab
left holes in regions wider than about 30 cells); the back texture is accumulated from the frames
that saw it (the lab textured the back with the front pixel on the same ray); the shell drops
surface pieces joined only across cut edges.

## Known limits

- The unseen side is the MHR template: bald, no loose cloth beyond the 12 cm backing bulge, no
  held objects. Texture on it is what the subject showed the camera during the clip; vertices no
  frame saw take their neighbours' colours.
- 3x3 texel patches do not align with the 2x2 chroma blocks of 4:2:0 video, so neighbouring
  faces' patches share chroma.
- The body is unwelded (3 vertices per kept face) and the topology changes every frame: geometry
  is intra-only at about 390 KB/frame on the measured clip.
- MoGe-2's scale differs from SAM 3D Body's: the anchor scale is 1.01 to 1.18 on the measured clip,
  so the body is resized to the shell each frame.
- People only: SAM 3D Body is a human body model. A mask frame with no body keeps the neighbour's
  mesh (`body-valid` 0).
- Before the subject's first detection and while the tracker has lost it, the mask pass repeats a
  neighbouring mask. The body phase fits nothing there, but the encoder still meshes the repeated
  mask's region and draws the nearest fitted body on those frames.
- The player's turntable orbits the clip's bounding-box centre, so a subject that travels across
  the frame moves around the orbit centre.

## Open items

1. A UV unwrap of MHR (xatlas), so the body welds (18,439 vertices) and its texture is a chart
   atlas instead of per-face patches.
2. Temporal coding: a fixed-topology body track with constant uvs could code as I+P separately from
   the shell if the container carried two geometry blocks per frame.
3. Hair and cloth: the backing is a bulge behind the shell; a learned completion for the back of
   the head and loose clothing is not evaluated.
4. Contact: the feet are not constrained to a ground plane.
5. The fit and anchor radii were measured on one clip; the defaults need a second clip with a
   turning subject.
6. SAM 3D Body's mask prompt (`masks=`) is unused and untested.
7. Two GPUs: the geometry and body phases run one worker per RTX 2080 Ti, but the SAM 3 mask pass
   and the Video-Depth-Anything pass run on `cuda:0` alone (77.3 s of the 320.1 s service time on
   150 frames was the mask pass, with the second card idle). A split needs the tracker state
   re-seeded per shard and overlapping video-depth windows.
