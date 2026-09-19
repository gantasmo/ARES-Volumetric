# 2D video → 2.5D volumetric: the depth pipeline

Status 2026-09-18. A plain 2D video becomes a `.ares` clip whose geometry is a per-frame relief
mesh: a monocular depth model estimates a depth map per frame, the encoder unprojects each map
through a pinhole ray table, cuts the mesh at silhouettes, and the video frame itself is the
texture. The result plays in the viewer like any other clip: orbit, edit ranges, audio. The
player opens a relief at the camera that shot it and sways it about the capture axis.

Every stage streams. A feature-length source is gigabytes of depth maps and of mesh; nothing in
the encoder holds a whole clip, and the dev server streams clips of any length with Range support.

## Provenance

The feature is a port of the "depthcloud" live source of VJ-9000
(github.com/gantasmo/VJ-9000, same author). What was copied and where it went:

| VJ-9000 source | Did | ARES |
| --- | --- | --- |
| `src/akvj/depthWorker.ts` | Web Worker: transformers.js Depth-Anything-V2-small, WebGPU fp16 → wasm q8 ladder | `apps/demo/depth-worker.js` (browser engine) |
| `src/useDepthCloud.ts` | frame sampling, pinhole ray table (55° vertical FOV), depth → pseudo-metres, EMA | `apps/demo/depth-browser.js` (sampling + upload), `packages/encoder/src/depth-mesh.ts` (ray table + unprojection), `packages/encoder/src/depth-stabilize.ts` (temporal filter) |
| `src/akvj/AkvjCloudRenderer.ts` | Three.js point-cloud renderer with VJ styles, bloom, audio reactivity | not ported: ARES renders through `AresPlayer`; the geometry is a mesh in a `.ares` file |

VJ-9000 is untouched; it stays the live-performance version of the idea.

## Pipeline

```
video ──► probe ──► depth engine ──────────► run dir ─────► ares depth ──────────────────► clip.ares
                    service | browser        depth.json     letterbox → A: RGB gate        + .meta.json
                    (+ SAM 3 subject mask)   depth.f32      → B: stabilize → C: per GOP:
                                             [mask.u8]      resample, mesh, fill, texture,
                                                            chunk → finish: header + index
```

1. **Probe.** `GET /probe-video?path=` (tools/serve.mjs) returns size, frame rate, frame count,
   duration and whether the file carries audio. The Convert card sizes its defaults from it.
2. **Depth engine.** One of two engines writes a run directory (contract below). The service
   engine can run a SAM 3 subject mask pass first.
3. **`ares depth`** (packages/encoder/src/cli.ts) consumes the run in passes, each touching a few
   frames at a time (see The encoder).
4. The dev server route `GET /depth-convert` runs steps 2 (service engine) and 3 inside one SSE
   stream for the Convert tab; the browser engine runs step 2 in the tab and hands the finished
   upload job to the same route.

## The depth run contract

`<dir>/depth.json`:

```json
{
  "schema": "ares-depth/1",
  "engine": "service | browser",
  "model": "depth-anything/Video-Depth-Anything-Small",
  "modelKey": "video-small",
  "kind": "relative-disparity | metric-depth",
  "temporal": "model | none",
  "width": 518, "height": 294,
  "frames": 300,
  "fps": 30,
  "sampling": { "fps": null, "maxFrames": null },
  "video": "C:/clips/take.mp4",
  "sourceFps": 29.97, "sourceWidth": 1920, "sourceHeight": 1080, "sourceFrames": 300, "sourceDurationS": 10.01,
  "msPerFrame": 7.6, "device": "cuda", "dtype": "fp16",
  "mask": { "file": "mask.u8", "prompt": "person", "engine": "sam3-text-tracker", "coverage": 0.21 },
  "done": true
}
```

`<dir>/depth.f32`: `frames × height × width` float32 little-endian, frame-major then row-major,
row 0 at the top of the image, no header. `relative-disparity` is the model's raw
affine-invariant inverse depth (larger = nearer, unnormalized). `metric-depth` is metres (larger =
farther). The consumer accepts any map size; the engines aim for a width of 518 and both
dimensions multiples of 14 (the DINOv2 patch size).

`temporal` says whether scale and shift are consistent across the clip as written: `"model"` for
Video-Depth-Anything, where the stabilizer skips its per-frame alignment; `"none"` (or absent) for
per-frame models, whose scale and shift change from frame to frame.

`mask` is present only when a subject mask pass ran. `<dir>/<mask.file>` is `frames × height ×
width` uint8, 0 outside the subject and 255 inside, in the same frame order and orientation as
`depth.f32`. The depth model saw each frame with the background painted black.

**Frame sampling.** Every consumer reproduces the engine's frame set with the same ffmpeg
filter chain, so texture frame *i* and depth frame *i* are the same source frame:

```
ffmpeg -i <video> -vf "<fps=F,><crop=W:H:X:Y:exact=1,>scale=W:H:flags=area" [-frames:v N] -f rawvideo -pix_fmt rgb24 -
```

`fps=F,` is present only when `sampling.fps` is set and `-frames:v N` only when
`sampling.maxFrames` is set. The engines never crop; the `crop` step is the encoder's letterbox
cut, placed after `fps` because it selects no frames. The browser engine steps a `<video>` element
to `(i + 0.5) / F` seconds instead, which matches ffmpeg's `fps` filter to within a frame at the
boundaries.

## Engines

### service: `tools/sam-service/depth.py`

An `APIRouter(prefix="/depth")` in the existing SAM service (same process, same conventions as
`track.py`: lazy load latched on failure, one job at a time, the GPU lock taken per batch so an
interactive `/segment` click interleaves, `health_fields()` merged into `/health`).

| Route | Purpose |
| --- | --- |
| `GET /depth/health` | ready/loading/error, loaded model, device, dtype, which keys run without a download, Video-Depth-Anything components, subject availability |
| `POST /depth/run` | `{ video, out, model, fps, maxFrames, inferWidth, batch, ffmpeg, subject }` → `{ job }`; 409 `{ error, missing: [component ids] }` when a component is absent |
| `GET /depth/status?job=` | state, `phase` (`mask` or `depth`), per-phase done/total/msPerFrame, map size, last log lines |
| `POST /depth/cancel` | stops the job and its ffmpeg child |

Three model families:

- `video-small`, `video-base`, `video-large`: Video-Depth-Anything (`depth_vda.py`). The upstream
  code is a clone under `tools/ext/video-depth-anything` (never edited, never on `sys.path`: its
  DINOv2 and temporal head are imported through a pinned package spec and composed under the
  upstream attribute names, and `load_state_dict(strict=True)` proves the composition), the
  checkpoints are `tools/sam-service/models/video_depth_anything_{vits,vitb,vitl}.pth`. Upstream's
  offline scheme, streamed: 32-frame windows stepping 22 frames, the previous window's inputs at
  the keyframe slots, a least-squares scale and shift fitted on the two reference slots, an
  8-frame cross-fade, the tail held back until the next window. Resident state is one window of
  inputs, the previous window's, 8 depth frames and 2 reference maps, whatever the clip length.
  fp32 weights under autocast, the upstream recipe. Output is relu'd affine-invariant inverse
  depth: `relative-disparity`, `temporal: "model"`. Small is Apache-2.0; Base and Large are
  CC-BY-NC-4.0. Component ids: `vda-code`, `vda-small`, `vda-base`, `vda-large`.
- `small`, `base`, `large` (relative) and `metric-indoor-{small,base,large}`,
  `metric-outdoor-{small,base,large}` (metres): Depth-Anything-V2 through transformers, resolved
  from the shared Hugging Face cache and downloaded on first use, batches of eight in the
  service's dtype. `temporal: "none"`.

ffmpeg decodes straight into a raw rgb24 pipe at the inference size; depth is written
unnormalized as float32, and `depth.json` is rewritten after every batch so a killed job leaves
a valid partial run.

**Subject mask** (`depth_subject.py`). With `subject` set (a text prompt such as `person`), a
mask pass runs before the depth pass over the same frame set, decoded at 1008×1008 (the SAM 3
processor's own size). SAM 3's text detector seeds its video tracker on the first frame where the
prompt scores, the tracker carries each instance forward, and the detector runs again every 24
frames (and on every frame while nothing is tracked) for instances that enter later. Tracker
memory older than 24 frames is dropped after each step, so the session stays a constant size.
Frames where the subject is absent repeat the last mask; the frames before the first detection
take the first. A prompt that matches under 0.02 % of the clip fails the job before any depth is
computed (`errorCode: "subject-not-found"`). The depth pass then sees every frame with the
background black, and the mask is written to `mask.u8`.

### browser: `apps/demo/depth-worker.js` + `apps/demo/depth-browser.js`

The VJ-9000 worker, with the changes that matter for an offline conversion:

- transformers.js loads from the jsDelivr CDN as an ES module (the demo app has no bundler);
  the page is cross-origin isolated (COOP/COEP from the dev server), which is what lets the wasm
  fallback use threads.
- The output is the model's float `predicted_depth` tensor, not the depth-estimation pipeline's
  8-bit `RawImage`.
- Frames are sampled by seeking the `<video>` to `(i + 0.5) / F` and waiting for `seeked`,
  with two frames in flight: every sampled frame is inferred, none are dropped.
- Maps upload in batches of eight through `POST /depth/upload/begin`,
  `POST /depth/upload?job=&index=&count=` (raw float32 body, sequential, the server appends and
  rewrites `depth.json` after every batch) and `POST /depth/upload/finish`; `/depth/upload/cancel`
  reclaims an abandoned job, and a finished-but-unconverted job is reaped after an hour.
- The `<video>` loads the picked file through `GET /depth/source?path=`, which answers Range
  requests (206); without them Chrome re-downloads from byte 0 on every seek.

Models: `onnx-community/depth-anything-v2-{small,base,large}`. Precision `auto` is WebGPU
fp16 with a wasm q8 fallback; `fp32` and `q8` can be forced. No Video-Depth-Anything, no subject
mask.

## The encoder: `ares depth`

```
ares depth <video> --depth <run-dir> [-o out.ares]
  unprojection: [--fov 55] [--near 2, or 0.5 for a metric run] [--far 6, or 20] [--grid 256]
                [--edge 0.08] [--sheets] [--decimate ratio] [--crop auto|none|W:H:X:Y]
  surface:      [--no-guided] [--guide-sigma 14] [--snap-ramps] [--inpaint] [--inpaint-band cells]
  depth:        [--stabilize 0.7] [--median 3] [--no-grow]
  texture:      [--tex-size 1024] [--texture-codec vp9|av1] [--crf 30] [--no-texture]
  audio:        [--no-audio] [--audio file] [--audio-offset s] [--audio-bitrate kbps]
  also:         [--gop 30] [--smooth-temporal N] [--up-axis x|y|z] [--center bottom|mass|none]
                [--scale N] [--rotate x,y,z] [--translate x,y,z] [--meta-extra-file f.json]
```

**Streaming.** The run is opened, never read whole (`openDepthRun`: positional reads of one frame
or one GOP). Every whole-clip pass reads and writes `FrameStore`s (`depth-store.ts`): memory for a
short clip, files in a temp directory once `frames × W × H × 4` passes 256 MB. The passes:

| pass | reads | writes |
| --- | --- | --- |
| letterbox | the source's keyframes, half size, gray | a crop rectangle |
| A | an ffmpeg decode at the map size, two frames resident | the RGB motion gate, one byte per pixel per frame |
| B | the run and the gate | the stabilized maps (align forward, gate growth, EMA backward, EMA forward + normalise in place) |
| C | the stabilized maps and a texture decode, one GOP at a time | per frame: resample, mesh, fill; per GOP: one closed VP9/AV1 GOP (`encodeRawTextureGop`) and one chunk appended by `MeshClipWriter`, three GOPs in flight |
| finish | the chunk scratch file | header, superblock, GOP index, track directory, then the chunks, with the centring offset applied as a translation of every chunk's quantisation box |

`MeshClipWriter` writes byte for byte the file `muxClip` builds in memory (a test checks both the
culled and the sheet path). `ares info` reads only the file head for clips over 1.5 GB.

**Letterbox** (`video-frames.ts`, `--crop`). Bars are rows and columns that stay black in every
keyframe (`-skip_frame nokey`, half size, gray; a source with fewer than 8 keyframes is sampled at
2 fps over its first minute). The picture's edges are rounded inward to whole map pixels, the
source crop is re-derived from them, and from there the maps, the RGB gate, the mask, the grid
aspect and the texture all use the picture only. A bar reads as infinitely far to a relative
model; uncropped, it becomes a slab of geometry at `far` and holds the normalisation's low
percentile at zero. `--crop none` keeps the full frame; `--crop W:H:X:Y` (source pixels) sets it.

**Stabilize** (`depth-stabilize.ts`). For a per-frame model, each frame *t* ≥ 1 gets a scale and
shift fitted onto the previous stabilized frame over static pixels (weight from the RGB change
against the previous frame after a 3×3 box blur; two passes with MAD outlier rejection; scale
clamped to [0.5, 2]). The fit is moment matching (`a = sd(prev) / sd(cur)`, `b = mean(prev) −
a · mean(cur)` over the inliers), not least squares: an OLS slope is attenuated by the noise in
the regressor and the chain multiplies it (measured 0.989 per frame at 2 % per-frame noise, a
70 % collapse of the depth range by frame 300). When the gate leaves under 10 % of the frame open
(a handheld camera), the fit runs ungated. For a `temporal: "model"` run the alignment is skipped.

The RGB change sees a large untextured object moving as a whole only along its outline, and its
interior drags the fit once it covers enough of the frame. `growGate` grows the gate over that
interior: seeds are RGB-moving pixels that came nearer under a provisional fit, the evidence
spreads over the current depth surface (free across smooth depth, stopped by silhouettes, fading
2 gate units per pixel), and the fit is repeated. On a synthetic approaching block the background
drifted by 0.19 of the normalised range at 70 % coverage without it and 0.012 with it (0.021 to
0.004 at 55 %). `--no-grow` turns it off.

Then a motion-gated temporal filter: per pixel `alpha = a + (1 − a) · smoothstep(lo, hi,
motion)` with `a = alphaStatic ^ strength` (alphaStatic 0.05, so 0.12 at the default
`--stabilize 0.7`), run backward and forward and averaged (offline, so zero lag). A still pixel is
averaged over about 16 frames each way; a moving one takes its new value outright. After it, a
per-pixel temporal median over `--median` frames (default 5; 1 is off) removes what still stands
alone in time and leaves any monotonic run as it was, so steady motion keeps its timing and a
scene cut stays a cut; an event of one or two frames (a 5-frame window) is removed, which is also
what happens to an object that crosses a pixel within 83 ms at 24 fps. `--stabilize` scales the
effect; 0 turns off the filter and the median. Finally a clip-wide robust normalization (percentiles
0.5 and 99.5 over a strided subsample of all frames) maps disparity to [0, 1] with 1 = nearest.
Metric depth goes through the same alignment and smoothing on its reciprocal and stays in metres.

**Resampling onto the grid** (`depth-layers.ts`). By default the map is resampled with a joint
bilateral filter guided by the video frame (Kopf et al. 2007): each map sample is weighted by how
closely its colour matches the vertex's, so a vertex on one side of an image edge takes its depth
from that side. `--no-guided` falls back to area averaging (shrinking) or bilinear (growing).
`--snap-ramps` then takes every vertex that is more than 25 % farther than its nearest neighbour
and more than 25 % nearer than its farthest (a vertex partway down the model's soft silhouette
ramp) and gives it the value of whichever of the two it matches in colour, two passes.

**Mesh** (`depth-mesh.ts`). A `gridW × gridH` grid (`gridH` follows the picture's aspect) with
UVs `((x + 0.5) / gridW, (y + 0.5) / gridH)`, the atlas being the video frame. Ray table:
`rayX = (u − 0.5) · 2 · tan(fov/2) · aspect`, `rayY = (v − 0.5) · 2 · tan(fov/2)`. Depth from
normalized disparity is hyperbolic, `z = 1 / (d / near + (1 − d) / far)` (d = 1 at `near`,
d = 0 at `far`); metric depth is clamped to `[near, far]`. Position `(rayX · z, −rayY · z, −z)`
(image y down, world y up, scene in front of the camera along −Z). A triangle is dropped when any
of its edges spans more than `edge` of its depth (`|z_a − z_b| / min(z_a, z_b) > edge`), and
unreferenced vertices are compacted. The decision is made per grid edge and carried from frame to
frame: an edge whose jump lies between 0.67 and 1.5 times `edge` keeps the decision it had on the
previous frame, so an edge that hovers at the threshold does not blink open and shut. On 300
frames of a film, the edges cut for exactly one frame were 376 per frame with the earlier
settings (0.5..6 m, still-pixel alpha 0.475, no median, no hysteresis) and are 143. With a subject mask, triangles touching a masked-out vertex
are dropped too. `--decimate r` simplifies each culled frame to the fraction `r` of its triangles
(meshoptimizer).

`--sheets` keeps the full grid every frame: every frame has the same topology and the muxer codes
geometry as I+P position deltas. The triangles across silhouettes are removed at draw time
instead: such a triangle lies almost along the ray of the camera that captured it, so both
renderers discard a fragment when `|dot(faceNormal, rayFromCaptureCamera)| < slope`, with
`slope = cell / hypot(cell, edge)` and `cell` the angular size of one grid cell (the draw-time
form of the `edge` rule). Masked-out vertices of a sheet are parked at `2 · far`, past the
`relief.depthMax` beyond which nothing draws. `--decimate` and `--sheets` exclude each other.

**Fill layer** (`--inpaint`, `depth-layers.ts`). The second layer of a layered depth image. A cut
whose depth jump exceeds 25 % seeds it: the near end is the occluder's outline, the far end (the
anchor) the background it hides. Depth and colour enter at the outline from the anchors and are
carried inward ring by ring for `--inpaint-band` cells (default 16 % of the grid width), then 16
Gauss-Seidel sweeps average the band's interior over its uncut neighbours with the outline held.
The fill always sits at least `edge` behind the surface it lies under. Culled, only the band is
meshed: cells whose corners are band vertices or anchors, anchors at their own depth, edges
between band vertices never cut, edges to an anchor under the `edge` rule. The atlas grows to
`texSize × 1.5 texSize`: the frame on top, a half-height plate under it holding the frame with
the band's colours painted in.

**Relief metadata.** Every relief clip carries the capture camera in its superblock metadata:
`relief.camera` and `relief.forward` (the capture origin and axis carried through the model
transform), `relief.fov`, and the surface depth distribution taken over every vertex of every
frame (an inverse-depth histogram, streamed): `relief.near` and `relief.far` (the depths of the
nearest and farthest 5 %) and `relief.pivot`, their midpoint in disparity, and `relief.aspect`,
the picture's width over height. Sheets add `relief.slope` and `relief.depthMax` for the
draw-time discard.

**Player framing** (`packages/core/src/player.ts`). `reliefFromMeta` reads those keys. The
player opens a relief with the eye on the capture camera, looking down the capture axis at the
pivot, through the capture FOV widened as far as the canvas needs to show the whole picture
(`relief.aspect` against the canvas aspect), and orbits about the pivot. The wheel zooms a relief
by field of view (`OrbitState.fov`, carried by `getCamera`/`setCamera` and by every mask2d
volume's camera): a dolly would move the eye off the capture camera, and there content at depth
*z* is *z* times larger than it looks in the picture. Orbiting by an angle *a* moves content at
depth *z* by `a · (p / z − 1)` against the pivot *p*; the auto-orbit is a sway whose half-angle
gives the nearest 5 % a parallax of 0.06 rad at its ends, at most 0.2 rad and at least 0.02.

**Texture and audio.** Texture frames come from the video through the same sampling (and crop),
decoded at `--tex-size` square with lanczos, one GOP at a time, and encoded to one closed VP9 or
AV1 GOP per geometry GOP. The video's own audio stream is transcoded to Opus unless `--no-audio`.

## Dev server routes (`tools/serve.mjs`)

| Route | Purpose |
| --- | --- |
| `GET /probe-video?path=` | ffprobe facts for the card |
| `GET /depth/engines` | passive status of both engines (never starts the service) |
| `GET /depth/source?path=` | Range-capable video bytes for the browser engine |
| `POST /depth/upload/begin`, `/depth/upload`, `/depth/upload/finish`, `/depth/upload/cancel` | browser-engine run assembly |
| `GET /depth-convert?video=&name=&engine=&model=&…` | SSE: depth (service) → `ares depth`; `depthJob=` for the browser engine; `keepRun=1` keeps the run dir |

`/depth-convert` parameters beyond the model: `fps`, `maxFrames`, `inferWidth`, `subject`
(service only), `grid`, `fov`, `near`, `far`, `edge`, `sheets=1`, `decimate`, `crop=none|W:H:X:Y`,
`noGuided=1`, `guideSigma`, `snapRamps=1`, `inpaint=1`, `inpaintBand`, `stabilize`,
`textureCodec`, `texSize`, `crf`, `gop`, `smoothTemporal`, `center`, `noAudio=1`. Every numeric
value is validated up front. Components the job needs (the encoder build, ffmpeg, the Python
environment, the chosen model's weights, SAM 3 for a subject) install inside the same SSE stream.
A subject run reports `progress` with `stage: "mask"` before the depth stage.

Static files, `.ares` clips included, stream from disk with single-range `Range` support (206,
416 past the end); nothing is read into one buffer.

`/depth-convert`, `/depth/upload*`, `/depth/source` and `/probe-video` are in the same-origin
guard list with the other routes that do work.

## What changed against VJ-9000

| | VJ-9000 depthcloud | ARES 2D → 2.5D |
| --- | --- | --- |
| purpose | live source at ~8 fps, 320 px | offline conversion, every frame |
| depth values | 8-bit `RawImage` from the pipeline | float `predicted_depth` (both engines) |
| resolution | 320 wide | 518 wide by default (multiples of 14), selectable |
| inference | one frame per `postMessage`, timer-sampled | service: batched CUDA, or 32-frame video windows; browser: seek-stepped, two in flight |
| depth mapping | linear, 0.6–4.0 m pseudo-metres | hyperbolic disparity → depth (correct for inverse depth), metric models pass through |
| temporal | EMA α = 0.5 on the fly (ghosts on motion) | a temporally consistent model, or scale/shift alignment; motion-gated bidirectional smoothing; clip-wide normalization |
| geometry | GPU points, no connectivity | relief mesh with silhouette cuts, or a fixed-topology grid coded as I+P deltas with a draw-time discard; optional fill layer |
| output | canvas stream | `.ares` clip with the video as texture, the video's audio, provenance sidecar |

## Measured (2026-09-18, RTX 2080 Ti, fp16, warm caches)

Service engine, inference only, 90 frames of 640×360 → 518×294, batch 8:

| model | ms/frame | 90-frame job wall | peak VRAM over resident | download |
| --- | --- | --- | --- | --- |
| small | 4.7–4.9 | 1.6–1.9 s | +467 MiB | 95 MB |
| base | 8.0–8.1 | 1.9–2.1 s | +782 MiB | 372 MB |
| large | 19.6–19.8 | 3.5 s | +1528 MiB | 1280 MB |
| metric-indoor-small | 4.0 | 1.2 s | +407 MiB | 95 MB |

Video-Depth-Anything Small over the full 5,627 frames of a 1080p film at 518×294: 7.48 ms per
frame (the run's own `msPerFrame`).

Whole conversions through `/depth-convert` (service engine, base, VP9 1024², CRF 30), wall time
end to end including the encoder, before the streaming rebuild:

| clip | depth | texture | mux | total | output |
| --- | --- | --- | --- | --- | --- |
| 192 f, 1280×720, 24 fps, grid 256, cull | 4.7 s (8.2 ms/f) | 11.9 s | 9.2 s | 30 s | 73.0 MB, 389 KB/f, 72k tris/f intra |
| same, `--sheets` | 2.2 s | 11.9 s | 6.8 s | 25 s | 42.0 MB, 224 KB/f, I+P |
| 496 f, 1920×1080, 30 fps, grid 192, cull | 5.3 s (7.8 ms/f) | 19.5 s | 11.8 s | 44 s | 103.7 MB, 214 KB/f, 40k tris/f intra |

`ares depth` alone after the streaming rebuild (grid 256, VP9 1024², CRF 30, video-small runs).
Pass C is meshing, texture encoding and chunk writing together, as the encoder logs it:

| run | options | time | output |
| --- | --- | --- | --- |
| 5,627 f, 1920×1080 film, no crop, 0.5..6 m | cull | 644 s total, pass C 601 s, peak working set 688 MB, 7.2 GB scratch | 2,058,179,701 bytes, 357 KB/f, 68,672 tris/f, texture 9.0 KB/f |
| same film, current defaults (letterbox, 2..6 m, 5-frame median, cut hysteresis) | cull | 536 s total, pass C 493 s, peak working set 636 MB | 1,664,490,917 bytes, 289 KB/f, 53,465 tris/f |
| 300 f of the same film | cull, letterbox 1920×814 detected | pass C 28 s | 71.4 MB, 244 KB/f, 52,308 tris/f |
| same | `--inpaint` | pass C 40 s | 96.6 MB, 330 KB/f, fill 15,154 tris/f |
| same | `--snap-ramps` | pass C 25 s | 72.2 MB, 247 KB/f, 272 vertices snapped/f |
| 300 f, 3840×2160 phone clip | cull | pass C 33 s | 104.8 MB, 358 KB/f |
| same | `--sheets` | | 58.9 MB, 201 KB/f |

On two frames of the film, the fill layer covers these shares of the relief triangles culled
across a jump over 25 %: 21 % and 22 % without `--snap-ramps`, 70 % and 61 % with it.

Geometry dominates the file: the intra-coded grid costs ~5.4 bytes per triangle per frame, so
`--grid` is the size lever (256 → 192 roughly halves it) and `--sheets` is the other.

The player loads the 2,058,179,701-byte clip whole and presents frame 2000 about 9 s after
navigation in headless Chrome over loopback.

Browser engine (headless Chrome, WebGPU fp16, small, 518×294): 482 ms for the first frame, then
33–35 ms per frame; the wasm q8 fallback with no adapter ran at 852 ms per frame.

## Tuning notes

- `--fov` is the one unknown a 2D video does not carry. 55° vertical suits phone and mirrorless
  footage; a wide-angle action camera is nearer 70–80°, a long lens 20–30°. After a letterbox
  crop it is the vertical FOV of the picture, not of the frame.
- `--near`/`--far` set the depth range for relative models, and `far / near` is what a viewer
  sees off the capture camera: content at depth *z* is *z* times larger than it looks in the
  picture, so at 0.5..6 m (12:1) the characters of a shot stand in front of a background twelve
  times their scale. The default 2..6 m is 3:1. A metric run keeps its metres (0.5..20 m).
- `--edge` 0.08 is a relative depth jump. Near the far plane small disparity differences are
  large relative jumps (at zero disparity 0.7 % of the normalised range is 8 %), so a distant sky
  carries small cuts that closer surfaces with the same noise do not.
- `--grid` 256 is about 37k vertices per frame at 16:9; 512 quadruples that for close-ups.
- Metric models skip the near/far guess and give real scale, but they are less robust on footage
  far from their training domain.
- The texture is the picture scaled to `--tex-size` square; with `--inpaint` the atlas is 1.5
  times taller.

## Known limitations

- 2.5D is a relief from one viewpoint: orbiting off-axis reveals what the cuts leave behind the
  subject. The fill layer continues the background into that region; it is an extrapolation.
- Relative-depth models are consistent within a frame and, with a video model or after
  stabilization, across frames, but the absolute scale is still a guess (`--near`/`--far`).
- Silhouette cuts change the vertex set per frame, so the culled path codes every frame intra
  (`--sheets` is the I+P path).
- The player holds a whole clip in memory; a clip over the browser's largest `ArrayBuffer` does
  not load. The format has 64-bit chunk offsets and a GOP index, so a Range-request loader is
  possible (see the handoff).
- The browser engine's frame set matches ffmpeg's `fps` filter to within a frame; for
  frame-exact texture/depth alignment on long clips prefer the service engine.
- Base and Large checkpoints (V2 and Video) are CC-BY-NC-4.0; Small is Apache-2.0. The model
  selector's tooltip names the licences; nothing gates on them.

## Handoff: what to do next

Done since the first version of this list: the video-consistent model (Video-Depth-Anything),
the subject mask pre-pass, fixed-topology silhouettes (as a draw-time discard, no per-vertex
data), guided resampling, a first fill layer, `--decimate` in `ares depth`, and the coherent-motion
drift (the grown gate). In priority order, what is open:

1. **Range-request loader in the player.** `AresPlayer.create` fetches the whole clip into one
   `ArrayBuffer`. Fetch the head (header, superblock, GOP index, track directory) and then chunks
   by byte range around the playhead; `FrameRef.block`, the texture frame table, the audio packet
   list and the editor's use of `file.buf` are what assume the whole buffer today.
2. **Scale and FOV from the picture.** Near, far and FOV are guesses (0.5 m, 6 m, 55°). A metric
   model that also estimates focal length (MoGe-2, Depth Pro) on a few keyframes could fit the
   relative disparity's scale and shift and give all three; not evaluated.
3. **Fill layer quality.** The fill is a ring-by-ring extension of depth and colour. Measure it on
   people and rooms, compare a learned inpainter for the plate's colour, and decide whether
   `--snap-ramps` becomes the default.
4. **Node-native inference.** `onnxruntime-node` with the DirectML provider would run the ONNX
   checkpoints from `ares depth` itself on any Windows GPU; weigh the native dependency first.
5. **Browser engine throughput.** Feed frames as `VideoFrame`s through the WebGPU path
   (`createImageBitmap` → texture) instead of `getImageData`, and batch frames per inference.
6. **Browser tests.** A headless Chrome run of `apps/demo/verify/depth-browser-test.html` in CI.
7. **Live preview.** A point-cloud preview of the first frames while the service runs would give
   early feedback on FOV and near/far.
8. **Inference size.** The service infers at 518 wide; Depth-Anything and Video-Depth-Anything
   are trained with 518 on the SHORT side (924×518 for 16:9). Measure the quality and time of
   `inferWidth` 924 against 518.
