# Changelog

## Unreleased

Groundwork for SAM 3 mask propagation: the region evaluator made exact under a keyframe on every
frame, the capture surface a propagation sweep needs, a sidecar pre-flight that fails loudly, and
the local service side of the propagation itself. Nothing new reaches the bake, a propagated span is
an ordinary `mask2d` keyframe list.

### Behaviour change
- `interp: "hold"` at an exact keyframe now resolves to THAT keyframe, not the previous one.
  `bracket()` bracketed half-open on `a` and closed on `b`, so a frame that IS a keyframe matched
  as `b` with `t = 1`, and `hold` forces `t = 0` above the fast path that would have caught it. On
  sparse hand-authored ranges the two keyframes are far apart and this was invisible; with a
  keyframe per frame it pinned every frame of a hold range one frame late. Linear ranges are
  unaffected in output and compile one keyframe per frame instead of two.
- A 2.5D relief clip (`ares depth`) opens at the camera that shot it, looking down the capture axis
  at the midpoint in disparity of its nearest and farthest 5 % of surface, through the capture FOV
  widened as far as the canvas needs to show the whole picture, and its auto-orbit is a sway about
  that point sized so the nearest content moves 0.06 rad at the ends (0.02 to 0.2 rad), where a
  capture turntables. The wheel zooms a relief by field of view, which keeps the eye on the
  capture camera; a capture still dollies. Read from the clip's `relief.*` metadata; other clips
  are unchanged.
- `ares depth` maps a relative run to 2..6 m by default (0.5..6 m before); a metric run keeps
  0.5..20 m. Off the capture camera, content at depth z is z times larger than it looks in the
  picture, and at 12:1 the characters of a shot stood in front of a background twelve times their
  scale.
- The depth stabilizer smooths harder and removes short outliers: the motion-gated EMA's
  still-pixel alpha is now `alphaStatic ^ strength` with alphaStatic 0.05 (0.12 at the default
  `--stabilize 0.7`, 0.475 before), and a 5-frame temporal median follows it (`--median`, 1 turns
  it off). An object that crosses a pixel within two frames (83 ms at 24 fps) loses its depth
  there; `--median 3` limits that to one frame. The silhouette cut keeps each grid edge's previous
  decision while its depth jump lies within 0.67 to 1.5 times the threshold. On 300 frames of a
  film the grid edges cut for exactly one frame fell from 376 to 143 per frame, and the
  frame-to-frame change of a fixed checker texture seen 0.2 rad off the capture axis from 1.38 to
  0.78 luma levels.
- `ares depth` crops a letterbox or pillarbox by default: rows and columns black in every keyframe
  leave the maps, the texture and the grid aspect. `--crop none` restores the full frame.
- The dev server streams static files from disk with Range support instead of reading each into
  one buffer, which stopped at 2 GiB.

### Added
- 2D video → 2.5D conversion, ported from VJ-9000's "depthcloud" source (github.com/gantasmo/VJ-9000)
  and rebuilt as an offline pipeline (docs/depth-2d-to-25d.md). The Convert tab's `Video…` button
  (or a dropped mp4/webm/mov/mkv) opens a card that runs one of two depth engines against the same
  run contract (`depth.json` + `depth.f32`, one float32 map per sampled frame): the local Python
  service (`tools/sam-service/depth.py`, Depth-Anything-V2 small/base/large and the metric
  indoor/outdoor variants on CUDA, batched, float output) or the VJ-9000 worker itself in the tab
  (`apps/demo/depth-worker.js`, transformers.js on WebGPU with a wasm fallback, maps uploaded
  through `/depth/upload/*`). `ares depth <video> --depth <run-dir>` then stabilizes the maps
  (per-frame scale/shift alignment, motion-gated bidirectional smoothing, clip-wide robust
  normalization), unprojects each through a pinhole ray table into a relief mesh with silhouette
  cuts at depth discontinuities (`--sheets` keeps the full grid so geometry codes as I+P deltas),
  textures it with the video frame, carries the video's audio, and writes the provenance sidecar.
  Against the original: float depth instead of 8-bit, 518 px instead of 320, hyperbolic
  disparity-to-depth instead of linear, motion-gated smoothing instead of a fixed EMA, and every
  frame converted instead of a timer-sampled 8 fps. Measured on an RTX 2080 Ti at fp16: Small
  4.7 ms, Base 8.0 ms, Large 19.7 ms per 518×294 frame on the service; a 192-frame 720p clip
  converts end to end in 30 s (4.7 s of it depth), a 496-frame 1080p clip in 44 s; the browser
  engine runs Small at 33–35 ms per frame on WebGPU in headless Chrome.
- Dev server routes for it: `/probe-video`, `/depth/engines`, a Range-capable `/depth/source`,
  `/depth/upload/{begin,finish,cancel}` + `/depth/upload`, and the SSE `/depth-convert`. Settings
  lists the five Depth-Anything-V2 checkpoints under a Depth group (Base joins the Best and
  Balanced profiles, Small the Smallest).
- SAM 3 video-tracker propagation in the local service, `tools/sam-service/track.py`: `/sam/track/`
  `open`, `frames`, `prompt`, `run`, `cancel`, `close`, plus `sessions` and `results` so a tab that
  reloaded mid-run finds the run it lost instead of opening a second one. A selection made once on
  one frame follows the subject through the clip, `run` streams one SSE `mask` per frame, and a
  frame the tracker scores at or below zero (or thresholds to nothing) emits a `gap` and no mask
  rather than an empty one. The masks arrive ALREADY in the edit list's RLE convention (alternating
  run lengths, 0-run first, `rleEncodeMask`), so the browser stores the array verbatim and the bake
  is unchanged: still no GPU, no torch and no network at bake time. Measured on an RTX 2080 Ti at
  fp16, 291.6 ms/frame (encoder 189.3 median, head 75.6 median), 2,263 MiB peak device allocation.
- The service loads one `Sam3VideoModel` and feeds its towerless tracker by priming the vision
  cache, which is the sequence the library itself runs per frame (`modeling_sam3_video.py`
  `:1613-1632`, measured at 0 cache misses). It shares the resident concept model as that video
  model's DETECTOR instead of moving a second 840.38 M-parameter copy onto the card: measured,
  737.4 MiB of device memory the graft does not spend, 50.7 MiB left for the tracker and its neck.
  The load is lazy on the first `/track/open`, waits for the interactive model loader rather than
  racing it on the same card, and latches a real failure the way `/upscale` and `/detail` do, so a
  tracker that cannot load 503s in isolation and `/segment` and `/segment_text` are untouched.
- Frames upload as a bare `[u32le frameIdx][u32le byteLen][JPEG]` stream, 16 per request, and the
  store outlives the session for ten minutes keyed by clip, camera, tracker input edge AND capture
  size, so a Retrack skips the browser's capture sweep and a resize does not silently inherit
  masks at the old aspect. A run holds its own lock for the whole propagation but takes the model
  lock only around each single-frame forward, so an interactive click interleaves at one frame's
  latency instead of blocking for the two minutes a 272-frame clip takes.
- `ares verify-edits <file.edits.json>`: per-range keyframe counts, the user/derived split, mask
  resolution and every `validateMasks` line, exit 1 on any issue. A sidecar pre-flight that needs
  no frames directory, for gating a bake in a script.
- `validateMasks(list)` in `@ares/core`: the loud check for the mask2d faults `prepareVolume`
  answers with "never matches" instead of an error, a missing `kind`, an RLE whose runs do not sum
  to `width*height`, a camera whose aspect is not the bitmap's, an unreachable or unordered
  keyframe. Returns strings, never throws. `ares encode --edits` warns each line.
- Player: `seekFrame(idx)`, `getClipFps()`, `captureFrameBlob(maxDim, type, quality)` (a Blob and
  its own dimensions, against `captureFrame`'s PNG data URL at 15 to 30 ms of main-thread deflate
  per frame), `frameGeomQ()` (the presented frame's quantized topology, the shape `rasterizeIds`
  and a worker both want), and `appliedIdx` + `settled` on `textureDebug()` so a batch consumer can
  tell whether the atlas has caught up with the mesh.
- `frameToClockUs` / `clockUsToFrame` exported from `@ares/core`, the frame/clock arithmetic the
  seek path was carrying privately.

- `ares depth` streams every stage: the run is read a frame at a time (`openDepthRun`), the
  stabilizer's passes go through frame stores that spill to temp files past 256 MB
  (`stabilizeDepthStream`), texture frames decode and encode one GOP at a time
  (`openRawFrames`, `encodeRawTextureGop`), and chunks append to a scratch file that
  `MeshClipWriter` lays out behind the header at the end, byte for byte the file `muxClip` builds.
  A 5,627-frame 1080p film encodes in 644 s at a 688 MB peak working set into a 2,058,179,701-byte
  clip, which the player loads.
- Video-Depth-Anything in the depth service: model keys `video-small`, `video-base`,
  `video-large`, upstream's 32-frame windows and stitching run as the windows arrive, manifest
  `temporal: "model"` so the encoder skips its per-frame alignment. A missing clone or checkpoint
  answers 409 with the component ids, which `/depth-convert` installs.
- Subject mask before depth: `subject` on `/depth/run` (and the depth card's `subject` field) runs a
  SAM 3 text-detector plus tracker pass first; the depth model sees the subject on black, the mask
  ships as `mask.u8` in the run, and `ares depth` drops (culled) or parks (sheets) what lies
  outside it.
- `relief.*` superblock metadata on every relief clip (capture camera, axis, FOV, surface depth
  percentiles, pivot), and for `--sheets` a draw-time discard in both renderers: a triangle lying
  along the capture ray is a silhouette and is not drawn, so sheets keep one topology for I+P
  coding and still show cut silhouettes.
- `ares depth` surface options: image-guided (joint bilateral) resampling onto the grid by default
  (`--no-guided`, `--guide-sigma`), `--snap-ramps` for vertices partway down a silhouette ramp,
  `--decimate`, and a fill layer (`--inpaint`, `--inpaint-band`) that continues the background
  behind silhouettes and textures it from a plate under the frame in the atlas.
- The stabilizer grows its RGB motion gate over the interior of a large untextured surface moving
  as a whole (`--no-grow` turns it off): on a synthetic scene covering 70 % of the frame the
  background drift fell from 0.19 to 0.012 of the normalised range.
- The depth card offers the Video-Depth-Anything keys (first in the list) and the subject prompt,
  guided, fill, band, snap, decimate and letterbox controls; `/depth-convert` passes them through.
- `apps/demo/verify/relief-shot.html` and `tools/relief-shot.mjs`: a clip rendered in the real
  player in headless Chrome to a PNG, at a given frame and camera (`zoom`, `fov` for reliefs).
  `apps/demo/verify/playback-trace.html` records the presented geometry and texture frame on every
  animation frame of real playback, and `apps/demo/verify/relief-flicker.html` measures the
  frame-to-frame change of the drawn image and of its holes over a frame range and a set of views;
  the driver writes their `window.__result` for an out path ending in `.json`.
- `OrbitState.fov` (vertical degrees, `ORBIT_FOV_DEG` 50 when absent) in `@ares/core`, carried by
  `getCamera`/`setCamera`, the crop guides and the camera of every mask2d volume, so a mask drawn
  through a relief's own FOV is tested through it. `relief.aspect` records the picture's aspect.

### Fixed
- `ares depth` on a long clip failed with `File size (3427788336) is greater than 2 GiB`: it read
  the depth run with one `readFile` and held every stage of the clip in memory. It now streams.
- `/depth-convert` installed SAM 3 for a subject run but never sent the prompt to the depth
  service, so no subject mask was made.
- The SAM service never auto-started: `powershell.exe` spawned with `detached: true` (dev server
  `samEnsure` and `ARES.mjs sam`) exits 0 within a second without running
  `run-sam-service.ps1` at all on Windows PowerShell 5.1.26100, whatever the window style or
  stdio, so every start timed out after 180 s with a stale transcript. Spawned attached (still
  hidden) the same command binds the port in ~3 s and is ready in ~13 s, and the child still
  outlives the server.
- Dev server: `/install`, `/shell/*`, `/hf-token`, `/import-ares`, `/delete-ares`, `/save-thumb`,
  `/open-info` and `/resolve-dir` join the same-origin guard list. Each one does work or touches
  the machine (winget/pip/git installs, HKCU shell keys, an MSIX install, a stored HF token,
  deleting or importing clips), and any page in the browser could reach them with an
  `EventSource` or a simple cross-origin POST (2026-09-18 audit).
- `mask2d` volumes captured in orthographic were region-tested through a perspective frustum:
  `camera.ortho` was serialized by the demo, dropped by the type and never passed to
  `orbitViewProj`. It now travels with the rest of the orbit state, and because an ortho matrix's
  bottom row is `[0,0,0,1]` the behind-the-camera guard cannot fire there, the evaluator applies the
  same `[0,1]` clip window the renderer does. Without it an ortho X-ray mask (which carries no depth
  band) selected geometry behind the capture camera.
- The edit-list rebase no longer skips a bake whose trim moves only the out point: `endFrame` is
  clamped to the shorter window and `derived` keyframes past it are dropped (a user keyframe is
  still kept wherever it lands, it is the anchor the surviving frames interpolate from). A copy
  range is exempt from the span drop, its bake frames are `copy.srcFrame`/`dstFrames` and
  `frame-copy.ts` never reads the span, and its frame refs are left untouched on an untrimmed bake
  so `collectCopyOps` still aborts on a typo'd index rather than clamping it away.
- A `POST /edits/<clip>` over the size cap answers 413 with the byte count and the cap instead of
  destroying the connection. `req.destroy()` sent no response at all and `doSave`'s `.catch(() => {})`
  swallowed it whole: the sidecar silently stopped being written and the work was gone on the next
  reload with nothing logged anywhere. The cap itself rises from 4 MB to 16 MB, because one
  propagated range writes a bitmap keyframe per frame: a 768x432 silhouette measures 1,185 runs,
  16,290 bytes pretty-printed at nine levels of nesting, 4.4 MB over 272 frames on one object with
  nothing else in the document.
- The `/sam` proxy tells the service when the browser walked away. `pipe()` only unpipes a dead
  client socket and never destroys the upstream request, so an abandoned `/sam/track/run` went on
  computing for a reader that will never come back: measured, a 60-frame run abandoned after four
  masks ran its whole span (16.7 s of GPU) before anything noticed. `/sam/track/` also joins
  `/sam/start` in the same-origin guard, because opening a track session pins 2,263 MiB of device
  memory and a run is 291.6 ms/frame for the length of a clip.
- The ViT-H fallback backend can load again. `VITH_CHECKPOINT` defaulted to a
  `sam_vit_h_4b8939_fp16.safetensors` that has never been in the tree, and the loader was
  safetensors-only, so `load_file` raised `SafetensorError` on the `.pth` that is actually there,
  from inside a try that only guards a missing file. With `SAM_BACKEND=auto` a sam3 failure left
  BOTH backends failed. The default names the file that exists and the loader branches on the
  extension.
- The demo divides by the clip's header fps everywhere instead of a hardcoded 30, in the transport
  readout, the timeline scrub, the keyboard step, the Media rail and the Compare tab. On a 24 fps
  240-frame clip the scrub landed on frame 192 where it read 240, and the duration read 8.0 s for
  10.0 s. `PlayerStats.fps` is the rolling render-rate EMA and was never the right divisor.

## 0.1.0: 2026-09-07

**ARES Volumetric plays volumetric video in a browser, from a single file.** A capture that would
normally arrive as thousands of meshes and gigabytes of PNGs becomes one `.ares` file: quantized,
meshopt-compressed geometry interleaved with a hardware-decodable AV1/VP9 video texture in one
GOP-aligned stream, with an optional Opus audio track. The player fetches that one file, dequantizes
vertex positions in the vertex shader, and uploads each decoded video frame straight to the GPU. A
real 272-frame capture is 49.7 MB in one request, against 1.58 GB across 544 files as raw OBJ+PNG.

This is the first versioned cut, the format, the browser runtime, the `ares` CLI, the Three.js and
React wrappers, the demo app and the specification, as they stand after the 2026-09-07 audit
([AUDIT.md](AUDIT.md)). **Added** below is what this release contains rather than a delta from an
earlier version; **Fixed** is relative to the unversioned initial public drop (commit `0907ee5`).

### Added
- Gaussian splat profile (spec §6.8, §11.6.3): `SPLT` geometry track, meshopt-coded splat streams
  quantized over the chunk AABB, SH degree 0–3, Morton-ordered frames. Both renderers draw it
  (EWA-projected instanced quads, premultiplied over, CPU counting sort gated on view change).
- Importers: Niantic SPZ v1–v4, 3DGS PLY, `.splat`, glTF/GLB with `KHR_gaussian_splatting`,
  PlayCanvas SOG (bundle or directory). Exporters: SPZ, 3DGS PLY, GLB, `.splat`; OBJ/PLY for meshes.
- `ares export <file.ares> -o <out> [--frame N]`, `ares synth --shape splat`, splat flags on
  `ares encode` (`--sh-degree`, `--splat-min-alpha`, `--splat-box-alpha`, `--splat-order`,
  `--quant-bits`).
- Convert tab recognises splat sequences; the viewer gates mesh-only tools for splat clips.
- `npm test` (node --test), GitHub Actions CI (Ubuntu/Windows × Node 22/24), esbuild bundles
  (`npm run bundle`), `AresPlayerOptions.workerUrl`, publish-ready package manifests.

- Editor: lasso and measure tools, grow/shrink/invert/mirror of the active range, camera bookmarks,
  per-range mute/name/interpolation (linear/hold/smooth), a bake-side sculpt action (move, inflate,
  smooth, flatten, pinch), analysis views (normals, UV checker, depth, points with size), an Export
  section (frame → OBJ, still → PNG, turntable → WebM), and a `?` shortcuts panel.
- Runtime: `AresPlayer.exportFrame()`, `setShadeMode()`, `setPointSize()`, `orbitSpeed`.
- Playback effects (both renderers, meshes and splats): clip plane, dissolve, tint, rim, scanlines,
  wobble, splat jitter/size/opacity; keyframed in the sidecar; audio-reactive binding; FX rail section.
- Dynamic splat profile: P-frames with position/attribute/SH deltas, births and deaths;
  `--splat-temporal auto|index|nn|off`.
- Audio: an Opus track per chunk (`ares encode --audio file`), WebCodecs decode into Web Audio,
  audio-led clock, mute/volume in the player and the demo transport, Convert-card audio row.

- Dev server: host pinning and a same-origin guard on every route that does work or spends,
  JSON-only `/runpod/launch`, separator-aware static guard, SSH host keys pinned on first use.
- Demo: Compare pauses off-tab and answers Space, batch queue survives re-analyse, Inspect reads
  only the container skeleton and shows profile + audio.

### Changed
- One launcher for the whole app, at the repo root: `ARES.mjs` (`npm start`, or `ARES.vbs` to
  double-click on Windows, `ARES-console.cmd` for a visible console) with modes
  `app | probe | bench | sam`. It replaces the four root `.vbs` launchers and their three
  PowerShell workers; `npm run serve` still starts the bare dev server.
- Package manifests point at the real repository (`gantasmo/ARES-Volumetric`) and carry
  `homepage`, `bugs` and a monorepo `repository.directory`. Each publishable package now ships its
  own README and a copy of the MIT licence, `@webgpu/types` moved to `dependencies` (its types
  appear in `@ares/core`'s public declarations), the browser packages no longer demand Node 22.15,
  and `@ares/three` publishes its bundle deterministically.
- `npm run release` builds the downloadable archive for a GitHub release: browser bundles, the four
  npm tarballs, the spec and the notices, as a directory, a zip and a release-notes file.

### Fixed
- WebGL2 renderer: crop preview and wireframe were silent no-ops.
- Geometry decoder caps vertex/index/splat counts before allocating (untrusted input).
- `ByteWriter(0)` looped forever; superblock `quant_bits_uv` misreported 14 (UVs are 16-bit).
- CLI: value flags followed by another flag are errors instead of NaN; numeric ranges validated;
  usage text complete; `ares info` reads its argument; mesh/atlas frame pairing uses natural sort.
- PLY mesh import keeps UVs, normals and colours.
