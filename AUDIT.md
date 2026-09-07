# ARES audit — 2026-09-07

Static audit of the whole repo (core runtime, encoder, demo + dev server, spec/docs/tools,
uncommitted working tree) plus a survey of the September 2026 landscape. Nothing was run on a
GPU. This file is also the work tracker: the **Progress** section at the bottom records what
has been addressed since the audit and where.

Owner's ordered queue (2026-09-07): standards alignment → splat profile → packaging and
process → editor tools → audio → hygiene → demo and dev server → dynamic splat video profile
→ VFX on volumes and splats. Remaining sections are queued after those.

---

## 1. Fixes and unfinished work found in the code

### Working tree, before the next commit
- `packages/encoder/src/paint.ts` is untracked but imported by `cli.ts` and `texel-copy.ts`, so a commit today ships a tree that does not compile.
- `tools/sam-service/models/RealESRGAN_x4plus.pth` is a 64 MB weight that is not gitignored. `apps/demo/.ares-activity.jsonl` is the same class of file as the already-ignored `history.json`.
- The SAM service diff adds `/upscale` and `/detail` endpoints that import spandrel and diffusers, but `tools/sam-service/requirements.txt` was not updated and nothing in the demo calls them. The code comment says they were added for a different project.
- README links two untracked docs, `docs/targeted-temporal.md` and `docs/rgbd-rebuild-pipeline.md`, so they need to land together.

### Runtime correctness (`packages/core/src`)
- Vertex and index counts are read as u32 and allocated immediately in `geometry.ts:37`, so a crafted file allocates tens of GB despite the untrusted-input claim in `bytes.ts`.
- A pending texture retry fires after dispose at `player.ts:596` and touches a destroyed device. There is no disposed flag, and the six pointer listeners are never removed.
- WebGPU adapter or device failure is fatal at `player.ts:352` even though a WebGL2 renderer exists. Neither renderer handles device or context loss.
- The video decoder never reconfigures after an error and never flushes, so tail frames can go missing (`texture-video.ts:69`).
- GOP start is assumed to be a keyframe without checking block type. An fps of zero yields Infinity PTS, and stored start PTS is ignored, so variable frame rate files play wrong.
- Prefetch ignores trim and ping-pong direction and transfers away P-frame state for a frame that may never present (`player.ts:527`).
- GL2 crop and wireframe are silent no-ops with no capability flag for the host. Neither renderer handles devicePixelRatio.
- Smaller items: a new bind group every frame, quadratic GOP lookup on load, `ByteWriter(0)` loops forever, a stored CRC of zero disables the check, chunk CRC is never verified, the 50° FOV is hardcoded in three places.

### Encoder correctness (`packages/encoder/src`)
- The flag reader has no parser (`cli.ts:40`). Typos are silently ignored, and a flag followed by a flag yields NaN timing. No validation exists for fps, gop, crf, tex-size, or codec name.
- Mesh and atlas files are paired by lexicographic sort rather than frame number, so unpadded names misalign every frame without error.
- PLY import drops UVs, normals, and colors, so the roadmap's "PLY plus PNG primary path" cannot texture. OBJ ignores `mtllib`, `usemtl`, groups, and lines with leading whitespace.
- The header advertises 14-bit UV quantization while the stream writes 16-bit (`quant.ts:54` vs `muxer.ts:201`).
- Decimation runs per frame independently and can silently break a stable topology run into intra frames. The achieved vs requested ratio is never reported.
- Track error is a mean, which the project's own notes say is blind to spikes, and there is no bail-out threshold. Topology stability is spot-checked on 64 indices.
- Hole fill accepts a loop if any single rim vertex lies inside the region, so pre-existing capture boundaries that graze a delete get capped.
- Paint weight is 0.5 at the region surface and triangles are culled by centroid, so strokes under-cover at their edges. Heal radius is a hardcoded 3.
- The usage string omits trim, transform, repack-detect, and paint flags. `ares info` takes the last argument even when it is a flag.

### Demo and dev server (`apps/demo`, `tools/serve.mjs`)
- Every long-running side-effect route is GET for EventSource with no Origin or token check. Any web page open in the browser can start encodes, run the PowerShell installer, open a file dialog, or hit RunPod routes that cost money.
- The static-file guard uses `startsWith` without a separator check (`serve.mjs:1438`), so a sibling folder whose name begins with the repo name passes.
- Compare's render loop is never cancelled on tab switch (`compare.js:106`), so two extra players decode forever behind the viewer.
- The batch queue lives in a module array and vanishes from the UI on re-analyse. Closing the tab kills the run because there is no server-side job store.
- `probe.js` buffers the whole file despite its header promising slice-only reads.
- Dead code: the coherent progress route has no client, the overwrite 409 path can never fire, a duplicate `r` key case, Compare's promised Space shortcut is disabled by the global handler.
- `main.js` is 2757 lines with a 1758-line editor init, global state on `window`, and the SAM depth capture loop copied three times. The one test copies functions instead of importing them.

### Docs and spec drift
- README claims the GOP index already supports tiers. The muxer writes a 28-byte record with none of the tier fields that `spec/11-file-format.md` specifies.
- The whitepaper predates the targeted-temporal work that Convert now enables by default.
- The RGBD pipeline scripts live in an untracked scratchpad, so the documented pipeline cannot be reproduced from a clone. The SVF runbook cites scratchpad paths that moved, and the SVF exporter has never been run end to end.
- The bench corpus directory is empty, so every published bench number is still synthetic.

### Project hygiene
- Zero tests, no test script, no CI, no lint. All four packages are private 0.0.0 with no publishable exports maps, no `sideEffects` flag, no `files` field, and `dist` gitignored.
- THIRD-PARTY-NOTICES omits ffmpeg, the Python stack, and the new dreamshaper-8 dependency (use-restricted OpenRAIL-M).
- `tools/coherent-poc` and `tools/coherent` duplicate about a dozen functions, three PS1 launchers repeat the same discovery blocks, three base64-to-PNG decoders exist.
- Hardcoded developer paths remain as defaults in `serve.mjs`, the SAM service, and coherent-poc. RunPod API and SSH private keys sit in plaintext with host key checking disabled.

---

## 2. Incomplete features and improvements

### Streaming and delivery (spec P3, not started)
- HTTP range-request loading with a prefetch ring; the GOP byte offsets already exist in the index.
- Async seek that does not re-roll P-frames on the main thread, reusing the worker path.
- Tier count and offsets in the GOP index, a one-command LOD ladder export, a real tier switch.
- A manifest for CDN caching and byte-range CORS, plus an offline cache for installed PWAs.

### Audio (declared, absent)
- Mux an Opus or AAC track, drive the clock from AudioContext time, expose gain and mute.

### Gaussian splat profile (spec P4, not started)
- SPZ, SOG, compressed PLY, and 3DGS PLY importers (the splat notes validated SPZ byte-exact).
- A depth-sorted splat renderer in both backends with per-GOP AABB quantization, an alpha/extent prefilter, shDegree 0 as the fast path.
- Temporal splat coding on birth/death lists with Morton ordering.
- Export through the glTF splat extension (KHR_gaussian_splatting).

### Point cloud view
- A points render mode for any mesh clip, plus a true point profile for faceless PLY input.

### Importers and exporters (spec P5, partial)
- glTF/GLB sequences, Alembic, USD/USDZ, FBX, Draco, Depthkit CPP, image sequences with camera JSON.
- Promote the 4DS and SVF tools to real importers.
- An `ares export` command for any frame or range to OBJ, GLB, Alembic, or USD.
- V-DMC transcode in and out once ISO/IEC 23090-29 tooling stabilizes.

### Texture and material channels
- HEVC and H.264 texture codecs for Apple devices (AV1 hardware decode reaches ~1/3 of iOS sessions).
- KTX2 stills, aux tracks for normal/roughness/occlusion, alpha mode, color management on both upload paths.

### Encoder throughput
- Worker-thread fan-out for import, decimate, rasterize, atlas patching; parallel ffmpeg per GOP.
- Stream frames instead of holding the whole sequence plus per-frame float masks in RAM (a 272-frame paint range ≈ 4.5 GB today).
- Resumable jobs, a real arg parser with help, named presets, dry run, JSON output, PSNR/SSIM/Hausdorff/quantization error reports.
- Verify the ffmpeg feature set at startup.

### Editor tools missing vs a modern mesh editor
- Selection: lasso/polygon, paint-select, grow, shrink, invert, select linked, by normal, by color, by SAM instance across frames.
- Sculpt brushes (move, smooth, inflate, flatten, pinch, crease, clay) as world-anchored ops (roadmap item 3).
- Interactive texture paint with clone stamp and inpaint preview.
- Interactive hole fill with preview, curvature-continuing filler.
- Keyframe curve editor with interpolation modes, onion skin, region naming, solo, lock, per-region opacity, copy-paste across clips.
- Symmetry, snapping, measurement, camera bookmarks, turntable export, HDRI/light rigs, ground shadow, background.
- In-editor smoothing and decimation, frame export to OBJ/GLB, screenshot and video recording.
- Op-level undo instead of 50 full-state snapshots.
- SAM mask propagation through time (largest editor quality lever, unresolved).

### Player UX
- Playback rate, J-K-L shuttle, numeric frame entry, loop range independent of trim, timecode, dropped-frame and GPU memory readouts, frame-time graph.
- Pinch zoom, responsive layout, fullscreen, permalinks with camera and time, a command palette for the keyboard map.
- Render modes: normals, UV checker, depth, texel density, vertex color, AO.
- Accessibility: aria labels, tab keyboard semantics, focus rings, reduced-motion.

### Web player widget
- Framework-free `<ares-player>` custom element (poster, lazy load, autoplay on visible, controls, src swapping); React/Vue/Svelte wrappers generated from it.
- Embed page, oEmbed endpoint, share dialog with iframe snippet.
- Single-file CDN build so the bare meshoptimizer import resolves without an import map.
- Media Session hooks, PiP via canvas capture, analytics events, WebXR view-in-AR, USDZ Quick Look fallback.
- The Three.js wrapper renders to its own canvas; a true Three.js material/mesh path is needed for lighting, shadows, raycasting.
- Bridges for Babylon.js, PlayCanvas, A-Frame, Needle, 8th Wall.

### Hole filling and clothing editing
- Replace the planar centroid fan with advancing-front/Liepa fill with curvature continuation and Poisson texture blending; track caps across frames.
- A clothing region op: SAM text prompt for garments, propagated through time, with recolor, retexture from reference, remove-and-reconstruct, swap-in fitted garment mesh.
- Use the SAM 3D Body template the RGBD pipeline already fits as the completion prior for holes on people.
- Video inpainting on the atlas sequence for occluded or damaged texels, conditioned on reference photos (roadmap item 5).

### Packaging and process
- Unit tests for demuxer, quantization, edits, temporal, hole patch; golden-file round-trip; demuxer fuzz harness (P6 entry ticket).
- CI for build, test, smoke run; semantic versioning; publishable exports maps.
- Split the demo into modules behind a bundler; move pipeline scripts out of scratchpad into tools with a manifest.

---

## 3. Cutting-edge directions as of September 2026

### Standards alignment
- MPEG V-DMC (ISO/IEC 23090-29) reached FDIS July 2025, IS expected March 2026. A V-DMC↔ARES transcoder and comparison positions ARES as the web delivery layer for the standard studios will adopt.
- Khronos announced KHR_gaussian_splatting early 2026, ratification targeted Q2. ARES splat import/export should speak it, alongside Niantic SPZ and the open-sourced PlayCanvas SOG (streamed LOD tree).

### Dynamic splat video profile
- Spark 2.0 streams LoD splat worlds on the web; PD-4DGS, 4DGC, StreamSTGS, ClipGStream demonstrate bandwidth-adaptive 4DGS streaming. A splat profile with GOP-aligned birth/death lists and a progressive LoD ladder makes ARES the first single-file container carrying both meshes and dynamic splats.
- Hybrid scenes: splat environment + mesh performer in one file with a shared clock and AABB.

### Humanoid to rigged avatar
- SAM 3D Body (CVPR 2026) outputs the open Momentum Human Rig, importable into Blender/Unity/Unreal. ARES already fits it per frame in the RGBD pipeline; promote that to a rig track in the container.
- UniRig / SkinTokens auto-rig arbitrary meshes; Puppeteer animates them. Bake-to-rigged-avatar export from a chosen frame is achievable now.
- LHM / PF-LHM produce animatable Gaussian avatars from one image in seconds; a splat-avatar export lets users animate a captured person with new motion.
- Retargeting editor: drive the capture rig with a new motion clip, blend live and synthetic poses, blendshape delivery for faces (spec 15 open item).

### Photo and video to volume
- Environments/objects: Depth Anything 3, MapAnything, VGGT give feed-forward metric geometry and cameras from 1–100s of photos. A Convert mode that runs one, meshes or splats the result, and encodes it is a complete photo-to-ARES path. World Labs Marble emits SPZ.
- People from monocular video: Mesh4D reconstructs and tracks a 4D mesh from one video; C4G, NoPo4D, UFO-4D do feed-forward 4D Gaussians. Tracked fixed-topology mesh from phone video is the "dream input" the RGBD doc names.
- Objects: SAM 3D Objects and Hunyuan-class generators for single-image object volumes.
- A local job runner abstracting RunPod, Colab, and local GPU behind one queue.

### VFX on volumes and splats
- Surface-emitted particles, dissolve and holo-glitch shaders, clipping planes/cross-sections, per-region material overrides, outline/rim passes, as timeline ops in the sidecar.
- Per-splat compute effects (position noise, color grading, size pulses).
- Audio-reactive parameters bound to the audio track or OSC/MIDI (theDAW link).
- Post-processing hook in both renderers (bloom, DOF, grading).

### Editor intelligence
- SAM 3 mask propagation across time, text-prompt garment/hair selection, instance tracking.
- Generative texture restoration conditioned on reference photos, video diffusion inpainting, relighting from normals/albedo.
- Scripting/MCP API for the editor; headless bake server.

### Engine and DCC conversion
- Alembic and USD import make Blender, Unity, Unreal, Houdini, Maya, Cinema 4D exports first-class sources; Unity/Unreal geometry caches are Alembic underneath.
- glTF with skinning/animation as an import for rigged characters, with bake-skinning-to-per-frame.

### Plugin and suite targets
- Blender add-on (mesh cache import, export; Blender has an official design task for native splats).
- Unity package (native decoder + WebGL path; UnityGaussianSplatting integration).
- Unreal 5.7 plugin with a Sequencer track (beside Luma's splat plugin).
- After Effects / Premiere plugin (camera + depth passes, timeline-bound frame time).
- theDAW: audio-reactive volumetric visualizer with transport sync.
- TouchDesigner, Notch, Resolume (live); Lens Studio, Effect House (social AR); Needle, 8th Wall (web AR); Webflow, Framer, Shopify, WordPress (embeds); Figma, Spline (previews); OBS source; visionOS RealityKit and Quest native players.

### Sources
- PD-4DGS https://arxiv.org/pdf/2605.11427 · 4DGC https://arxiv.org/pdf/2503.18421 · StreamSTGS https://arxiv.org/pdf/2511.06046 · ClipGStream https://arxiv.org/pdf/2604.13746 · Spark 2.0 https://www.worldlabs.ai/blog/spark-2.0
- V-DMC status https://ieeexplore.ieee.org/document/10890499/ · MPEG-I Part 29 https://mpeg.expert/v-dmc/
- KHR_gaussian_splatting https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release · https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md
- SOG https://blog.playcanvas.com/playcanvas-open-sources-sog-format-for-gaussian-splatting/ · splat-transform https://github.com/playcanvas/splat-transform
- SAM 3D Body https://mer.vin/2026/06/sam-3d-body-promptable-full-body-3d-mesh-from-one-image-cvpr-2026-mhr/ · Meta SAM 3D https://ai.meta.com/blog/sam-3d/
- UniRig https://github.com/VAST-AI-Research/UniRig · Puppeteer https://arxiv.org/pdf/2508.10898 · LHM https://arxiv.org/abs/2503.10625 · PF-LHM https://arxiv.org/html/2506.13766v1
- Mesh4D https://arxiv.org/pdf/2601.05251 · C4G https://arxiv.org/abs/2605.31595 · NoPo4D https://arxiv.org/html/2605.22190 · UFO-4D https://arxiv.org/pdf/2602.24290
- Depth Anything 3 https://arxiv.org/pdf/2511.10647 · MapAnything https://github.com/facebookresearch/map-anything
- Engine support https://radiancefields.com/3d-gaussian-splatting-engine-support · https://www.thefuture3d.com/blog/state-of-gaussian-splatting-2026/
- WebGPU in iOS 26 https://appdevelopermagazine.com/webgpu-in-ios-26/ · AV1/WebCodecs device data https://webcodecsfundamentals.org/datasets/codec-analysis-2026/

---

## Progress

Updated as items are addressed. Format: date · item · where.

- 2026-09-07 · audit written · this file.
- 2026-09-07 · **Standards alignment + splat profile (intra)** · `packages/core/src/{splat,splat-sort,geometry,renderer,renderer-gl2,player,camera}.ts`, `packages/encoder/src/{splat-frame,export}.ts`, `packages/encoder/src/importers/{spz,ply,splat-file,gltf-splat,sog}.ts`, muxer/synth/cli, demo Convert card + viewer gating, serve.mjs `/analyse` + `/encode`, spec §6.8 / §11.6.3, README. Importers: SPZ v1–v4 (gzip and zstd), 3DGS PLY, `.splat`, glTF/GLB `KHR_gaussian_splatting`, SOG (zip or directory, WebP via ffmpeg). Exporters: SPZ v2/v3, 3DGS PLY, GLB, `.splat`, plus OBJ/PLY for meshes (`ares export`). Renderer: EWA-projected instanced quads, SH degree 0–3, premultiplied over, CPU counting sort with view-change gating, both backends. Tests: `npm test` (node --test) covers block/SPZ/PLY/glTF/.splat round trips, zip/WebP helpers, mux/demux, CLI synth→export→encode.
- 2026-09-07 · **Fixes en route** · GL2 crop discard + wireframe implemented (were no-ops); untrusted-count allocation guards in the geometry decoder; `ByteWriter(0)` infinite loop; superblock `quant_bits_uv` now reports the real 16; CLI flags validated (`--fps --gop 30` is an error, not NaN), usage string complete, `ares info` reads its argument; mesh/atlas pairing by natural sort; PLY mesh import now keeps UVs, normals and colours; dev-environment: `node_modules/@ares/*` links pointed at a vanished D: drive and were rebuilt as junctions to this checkout.
- 2026-09-07 · **Packaging and process** · `.github/workflows/ci.yml` (Ubuntu + Windows × Node 22/24: build, test, bundle, encoder smoke), `npm test` / `test:quick` / `bundle` / `release:check`, `tools/bundle.mjs` (esbuild ESM + IIFE + worker bundles, sourcemaps, minified variants), `AresPlayerOptions.workerUrl`, worker client no longer throws from its constructor; package manifests versioned 0.1.0 with `exports` (types conditions), `sideEffects: false`, `files`, `engines`, `publishConfig`, `prepublishOnly`, pinned internal deps (`npm pack --dry-run` clean for all four); `CONTRIBUTING.md`, `CHANGELOG.md`, `.editorconfig`. Tests now also cover the container (mux→demux→decode, CRC/version/truncation, a 240-iteration corruption fuzz), temporal planning, decimation, edit-list delete + hole capping, quantization, OBJ import. Not done: ESLint config, splitting the demo behind a bundler, moving `scratchpad/` pipeline scripts into `tools/` (needs the owner's read on which are live).
- 2026-09-07 · **Editor tools** · lasso select (A) and measure (T) tools; grow / shrink (= / −, one brush radius per step, masks by matching pixels), invert (I, delete ⇄ keep), mirror across the centre X plane; camera bookmarks (◈, C cycles, per clip on this machine); per-range mute, name and keyframe interpolation (linear / hold / smooth — `interp` in `edits.ts`); sculpt action (move / inflate / smooth / flatten / pinch, weld-aware, SDF-feathered, bake-side — `packages/encoder/src/sculpt.ts`, spec-level payload in `edits.ts`); analysis views in the shade control (normals, UV checker, depth, points = point-cloud view, both renderers, point size slider); Export section (frame → OBJ via `AresPlayer.exportFrame()`, still → PNG, 8 s turntable → WebM via MediaRecorder); `?` shortcuts panel. Tests: `packages/encoder/test/editor.test.mjs`. Not done: interactive sculpt/paint preview (bake-only by the existing law), SAM propagation, onion skin, HDRI/lighting rigs, op-level undo, hole-fill algorithm upgrade (queued under hole filling).
- 2026-09-07 · **Audio** · Opus track in the container: `--audio <file>` (any format ffmpeg reads) transcodes to Ogg Opus 48 kHz (`packages/encoder/src/audio-mux.ts`), an Ogg demuxer + Opus TOC timing (`ogg.ts`) turns it into timed packets, the muxer lays them into each chunk as an Audio block (track 2, FourCC `OPUS`, codec_config = OpusHead, `HasAudio` flag); the runtime decodes with WebCodecs `AudioDecoder` into Web Audio (`packages/core/src/audio.ts`) and the AudioContext clock leads the video clock while playing (re-anchored on seek / loop wrap; silent on ping-pong's reverse leg). Player API: `setVolume`, `setMuted`, `hasAudio`, `isMuted`, options `audio`, `volume`, `muted`; stats `audioLabel`. Demo: transport mute (U) + volume, HUD audio line, Convert card audio file row (native picker) → `/encode?audio=…`; `ares info` reports the track. Tests: `packages/encoder/test/audio.test.mjs` (Ogg lacing, TOC table, block round trip, chunk windowing, mux/demux, ffmpeg WAV → Opus → CLI). Not done: AAC alternative, playback-rate with pitch correction, audio in the Compare tab, re-attaching audio on Edit-rail bakes (the Convert card carries it; a bake re-encodes from frames — add an `audio` field to the sidecar's provenance next).
- 2026-09-07 · **Hygiene** · `.gitignore` covers `apps/demo/.ares-activity.jsonl`, `tools/sam-service/models/`, `tools/*.log`, `.skel-*.log`; `tools/sam-service/requirements.txt` declares spandrel / diffusers / accelerate for the `/upscale` + `/detail` endpoints; `THIRD-PARTY-NOTICES.md` rewritten (meshoptimizer attribution in bundles, draco3d bench-only, ffmpeg shelled out with the LGPL/GPL build note, esbuild, the Python stack, SAM weights, Real-ESRGAN, dreamshaper-8 OpenRAIL-M, SPZ/SOG/.splat/3DGS-PLY/KHR specs implemented from public descriptions); developer-machine defaults replaced by env-driven neutral paths (Forge root, Forge checkpoints, SAM ViT-H checkpoint in serve.mjs and main.py, ffmpeg discovery in coherent-poc); CPU-only torch no longer crashes on `torch.cuda.OutOfMemoryError`. Not done: deduplicating `tools/coherent-poc` vs `tools/coherent` (research scripts the owner runs; no test harness to prove parity), RunPod keys at rest (a secrets manager is the owner's call).
- 2026-09-07 · **Demo and dev server** · serve.mjs: host pinning (421 on non-loopback Host — DNS rebinding), same-origin guard on every working/spending route via `Sec-Fetch-Site` / Origin / Referer (`/encode`, `/convert-4ds`, `/enhance`, `/sam/start`, `/forge/start`, `/setup/*`, `/runpod/*`, `/pick`, `/log`, `/edits/*`, `/showcase`, `/deps/*`), `/runpod/launch` requires `application/json` (415), separator-aware static-path guard, SSH `StrictHostKeyChecking=accept-new` with a real `known_hosts` under `.runpod/`, dead `/coherent/progress` route and the unreachable 4DS 409 removed. Demo: Compare's shared clock stops when its tab is not up and Space works there; the batch queue re-renders after a re-analyse; probe.js reads the container skeleton through `File.slice()` only and reports profile + audio; the duplicate `r` key case removed. Not done: splitting main.js behind a bundler, a server-side job store for the batch queue, the SAM per-frame-only limitation.
- 2026-09-07 · **Dynamic splat video profile** · splat P-frames (spec §11.6.3 P-frame): position deltas mod 2¹⁶, survivors' attrs + SH as byte deltas mod 256, delta-coded death list, appended births; encoder correspondence by index (`auto` verifies most splats sit within the match radius) or nearest neighbour on a hash grid over quantized positions, intra fallback below `--splat-min-survive`; flags `--splat-temporal auto|index|nn|off`, `--splat-match`, `--splat-min-survive`; `quantizeSplatFrame` keeps the encoder on the decoder's exact state; player chains P-frames and re-rolls from the keyframe on jumps; `ares export` and `info` chain too; varints in `ByteReader`/`ByteWriter`. Tests: `packages/encoder/test/splat-temporal.test.mjs` (index and NN correspondence with births/deaths reconstruct the exact set, temporal < 0.8 × intra). Not done: attribute-aware matching (colour/scale similarity), B-frames, a LoD ladder, GPU sort.
- 2026-09-07 · **VFX on volumes and splats** · `packages/core/src/fx.ts`: one 8×vec4 effects block shared by both renderers and both profiles — clip plane (cross-section / reveal), 3D value-noise dissolve with a coloured burn rim, tint, fresnel rim (meshes), hologram scanlines, sinusoidal wobble, per-splat jitter / size / opacity — plus a keyframed track (`FxTrack`, linear interpolation, hold outside) and `AresPlayer.setFx / getFx / resetFx / setFxTrack / setFxOverride / currentFx / getAudioLevel`. Effects are playback-time and non-destructive: the sidecar stores `edits.fx = { keyframes, react }`, the encoder ignores it. Demo: FX rail section (clip axis/offset/flip, dissolve + grain + edge colour, tint, rim, scanlines, wobble, splat row, ◆ keyframe / remove / clear / reset, keyframe chips that seek), audio-reactive binding (loudness × gain onto one parameter, through the override layer; the analyser taps the pre-gain node so mute still reacts), undo covers the track. Tests: `packages/core/test/fx.test.mjs`. Not done: surface-emitted particles, post-processing hooks (bloom / DOF), OSC / MIDI input, baking dissolve or clip into a delete range.
- 2026-09-07 · **Single launcher** · `ARES.mjs`, at the repo root where anyone can find it, is the only launcher now, on every OS: modes `app` (default) / `probe` / `bench` / `sam`, flags `--port --src --detach --no-open --no-build --no-install --help`, one log at `tools/launch.log`, dependency and build steps gated on staleness, a synth clip generated when the checkout has no `.ares`, and server reuse only when `/__ares` reports this checkout as its root (another tree's server on the port is skipped, not opened). `ARES.vbs` is the windowless Windows shim and passes its argument through, `ARES-console.cmd` is the visible-console variant, and `tools/launch.ps1` shrank to a Node bootstrap (find Node, winget-install the LTS build if absent, hand over, message box on failure). Deleted: `Launch ARES Probe.vbs`, `Play ARES Demo.vbs`, `Run ARES Bench.vbs`, `Launch SAM Service.vbs`, `tools/demo.ps1`, `tools/bench.ps1`, `tools/launch-debug.cmd` — about 300 lines of triplicated Node discovery, install, build, port-scan and open blocks. `npm start` runs the launcher, `npm run serve` the bare server, `npm run probe` the probe. All five package manifests now point at the real repository (`gantasmo/ARES-Volumetric`, not the never-existing `gantasmo/ares`) and carry `homepage`, `bugs` and a monorepo `repository.directory`. References updated in README, CONTRIBUTING, bench README and report, briefing, whitepaper, the probe page, the Settings card, serve.mjs and the SAM service.
- 2026-09-07 · **Release packaging** · `tools/release.mjs` (`npm run release`) assembles `dist/release/ares-volumetric-<version>/`, the same tree zipped (an inline store+deflate zip writer, since Node ships none), and a release-notes file for `gh release create --notes-file`; both generated documents open by saying what ARES is rather than what changed, since a download has no "before" to diff against. `spec/build.py` no longer truncates its own HTML output when the `markdown` module is missing (it opened the file before rendering, which is how a 0-byte `ARES-Runtime-Specification.html` reached the working tree), and the release builder refuses to package an empty document. Post-release verification sweep (five lenses, every finding adversarially re-checked) fixed: the shipped getting-started snippet called a non-existent `new AresPlayer()` / `load()` API, packages claimed a README and a licence they did not carry, `@webgpu/types` was a devDependency while `GPUDevice` appears in the published declarations, `engines` blocked Node 20 for browser-only packages, `@ares/three` shipped an unreachable bundle, README claimed Node 18 and a SharedArrayBuffer requirement the runtime does not have, and AUDIT/README contradicted themselves about the dynamic splat profile.
- Not done in this pass (queued): GPU sort, splat worker decode, hybrid mesh+splat files, KHR_gaussian_splatting compression sub-extensions, SuperSplat `compressed.ply` (use SOG), V-DMC transcode.
