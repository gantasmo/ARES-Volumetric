# ARES: a runtime-first volumetric video format for the browser

Technical whitepaper. Written 2026-07-10 against the code and measurements in this
repository; the normative reference is the ARES Runtime Specification, Draft 0.2
(`../ARES-Runtime-Specification.md`, built from `../spec/`). Where this document and
the spec disagree, the spec wins. Following the spec's own convention, claims here are
labeled measured, estimated, or projected; a number without a label is measured.

## 1. Overview

ARES is a container format (`.ares`) and a browser runtime for volumetric video: mesh
sequences captured or generated at 30 frames per second, textured, and played back inside a
web page. The format treats a volumetric frame as a compressed set of GPU state changes
rather than as a 3D model to be parsed (spec section 1.3). Three commitments follow from
that framing. Temporal coherence comes first, with GOP structure and
I/P frame machinery borrowed from video coding. The browser's hardware video decoder does
the heaviest work, decoding the texture track through WebCodecs. And the wire format maps
onto GPU buffers with near-zero CPU transformation: vertex positions stay quantized as
16-bit integers from encoder to vertex shader, where a WGSL function dequantizes them.

The reference implementation is an npm/TypeScript monorepo containing the runtime
(`@ares/core`), an encoder with a CLI (`@ares/encoder`), Three.js and React wrappers, a
four-tab demo application with a mesh editor, a benchmark harness, and local tooling for
encoding, measurement, and AI-assisted texture work.

Headline result, measured on a real 272-frame Microsoft-style volumetric capture (11.3k
vertices per frame, 2048x2048 atlas, 9.1 s): the raw OBJ+PNG source occupies 1.58 GB in 544
files; a Draco-GLB-per-frame baseline occupies 1.13 GB in 272 files; the current ARES
encode of the same capture is one 49.7 MB file, fetched in one request, decoding geometry
in well under a millisecond per frame on the main thread and rendering at 60 fps on the
machines measured. The first full ARES encode of this capture was 67.1 MB; the difference is three
measured optimizations described in sections 6 and 10.

## 2. The delivery problem

Volumetric capture pipelines commonly export one mesh file and one texture image per frame.
For the reference capture that means 272 OBJ files (485.7 MB) plus 272 PNG atlases
(1.11 GB). Playing that back in a browser means hundreds of HTTP requests and per-frame parse
and upload work on the main thread, with no way to seek or buffer ahead. Compressing each
frame with Draco shrinks the geometry to 20.3 MB but leaves the texture untouched and the
one-file-per-frame delivery model in place, while moving the cost into decode: Draco's
decoder is about 279 KB of WASM and measures around five times slower per frame than the
meshopt path in section 10.

Video solved the same shape of problem decades ago with temporal prediction backed by
hardware decoders and GOP-aligned seeking. No open volumetric format brings all of that to
the browser. That is the gap ARES targets (spec section 3.10): browser-native decode plus
temporal geometry compression over persistent topology, in one format.

## 3. Prior art and positioning

The spec surveys the field in section 3: Microsoft HoloVideo/MRC, 4DViews, Depthkit, UVOL,
V-PCC and G-PCC, Arcturus AVV, and the Khronos glTF volumetric subgroup. Two findings from
that survey shape the design. UVOL is the closest open prior art and the most honest
baseline. Arcturus AVV's per-vertex motion vectors are the same mechanism ARES formalizes
as P-frames. The differentiator against the Khronos effort is priority: glTF extensions
optimize for interchange between tools, ARES optimizes for the last hop into a running
page.

The strongest empirical anchor comes from a byte-level teardown of a shipping 4DViews
`.4ds` file, performed with a structural parser (`tools/probe-4ds.cjs`) that reads only
container skeleton: header fields, frame directory, and block lengths. It never decodes
mesh or texture content — ordinary interoperability practice — and it self-validates by
reconciling geometry + texture + overhead to the exact file size with zero bytes left over.
The file examined (455 frames, 15.2 s, 1440x1440 texture) splits as follows:

| | Bytes | Share | Structure |
|---|---|---|---|
| Geometry | 9,717,626 | 4.1% | 30 intra keyframes (~100 KB each) + 425 P-frames (~15-21 KB), adaptive GOPs averaging 15.2 frames |
| Texture | 228,496,451 | 95.9% | one ~502 KB GPU-block image per frame (~1.94 bpp), no inter-frame compression |
| Overhead | 8,167 | <0.1% | headers and directories |

ARES's first encode of the reference capture had the opposite proportions: 52.4 MB of
intra geometry against 14.7 MB of VP9 video texture. Normalized to the same 9 s, the
4DViews temporal mesh codec produces geometry about 9.5x smaller than ARES's intra
geometry, while ARES's video texture is about 9x smaller than 4DViews's per-frame texture
(cross-content normalization, so the ratios carry some slack). Each format wins by roughly
an order of magnitude on the axis the other ignores. No shipping format yet combines
temporal geometry with video texture; closing the geometry side of that union is the core
of the ARES roadmap.

## 4. Design principles

The spec states seven principles in section 1; the ones that most explain the code are
these. All expensive work belongs in the offline encoder, and the runtime stays
deliberately small and dumb (spec section 10): demux, schedule, hand buffers to the GPU.
The container is representation-agnostic through a profile byte, with persistent-topology
mesh I/P/B as profile 0, Gaussian splats as profile 1, and an experimental video-geometry
profile behind a Pareto gate. Unknown blocks must be skippable by length so the format can
grow without breaking old decoders.

The spec also imposes a process discipline. Every performance
claim in it carries a status tag (ASSERTED, PROJECTED, ASSUMPTION, or OPEN), an
assumptions register tracks what has not been validated (section 16.3), and section 13.6
states the rule that projections are commitments to measure, not to hit. The repository
follows the same discipline: prediction failures are recorded next to the shipped
results (section 11 lists two), and every published size number has a reproduction path
(section 13).

## 5. Container format

A `.ares` file is one buffer with four regions after a fixed 64-byte header: superblock,
GOP index, track directory, and a run of self-contained chunks. The header carries the
magic bytes `ARES`, version, profile and codec enums, frame rate as an f32, frame count,
duration in microseconds, four u64 section offsets, and a CRC-32 over the header bytes.
A stored CRC of zero means unchecked, an intentional escape hatch. A version_major above
the supported value is rejected; minor versions are additive.

The superblock stores clip-global facts: the axis-aligned bounding box, position and UV
quantization bit depths (14 by default), the normal encoding selector, GOP length, an
optional still-texture blob descriptor, and free-form key/value metadata. The GOP index is
a 28-byte record per GOP (start PTS, frame range, byte offset and length), which makes
seeking a binary search plus one HTTP range request in the planned streaming mode. The
track directory maps track ids to FourCC codecs and carries the exact codec configuration
string that `VideoDecoder.configure` needs.

Each chunk (magic `CNK0`) is one GOP and is independently decodable: it carries its own
AABB, used as the dequantization range for that GOP, a block directory, geometry I and P
blocks, and one closed video GOP for the texture track. The container is purpose-built
rather than an MP4 or Matroska mapping (spec section 11.9) so that geometry and texture
share one GOP-aligned chunk and one PTS domain, and so the parser stays small enough to
audit; the implementation's demuxer bounds-checks every offset and treats the file as
untrusted input throughout (spec N6). MP4 interop is an acknowledged open question (Q8),
deferred rather than dismissed.

## 6. Geometry coding

Positions quantize to unsigned 16-bit integers per axis over the GOP bounding box, 14
effective bits by default, stored x, y, z, pad so each vertex occupies exactly two u32
words in the GPU storage buffer. The CPU never converts positions back to floats. The
WGSL vertex shader fetches the two words, unpacks three u16 values, and computes
`world = aabbMin + q * invLevels * aabbSize`. The benchmark's quantization sweep (section
10) shows RMS position error halving with each added bit, from 1.4e-4 of the bounding-box
diagonal at 11 bits to 4.4e-6 at 16; 14 bits lands at 1.8e-5, which is subpixel for
human-scale captures at normal viewing distances.

Intra frames compress the quantized attribute streams and the index buffer with the
meshoptimizer vertex and index codecs. The choice of meshopt over Draco is a measured
tradeoff: Draco edgebreaker encodes the benchmark clip to 2.9% of raw against
meshopt+Brotli's 6.4%, but its decode measures 5.5 ms per frame against meshopt's 1.1 ms
(2.1 ms with Brotli), and it ships a decoder roughly 10x larger
(285,948 bytes of WASM against 29,059 bytes of ESM including embedded WASM). The runtime
budget rules, so meshopt is the default and Draco remains an optional profile for
size-critical uses (spec section 6.7).

Two encoder refinements shipped after the first full encode, both measured on the
reference capture. Meshopt's `reorderMesh` pass, lossless by construction since the runtime
re-uploads topology anyway, cut geometry from 52.4 to 34.0 MB (-35%) and the delivered
file from 67.1 to 49.0 MB; index data had been 38% of the file. Octahedral 16-bit normals
replaced the legacy signed-byte encoding at the same four bytes per vertex, improving
angular precision from roughly 0.5-1 degree to roughly 0.01 degree; this was predicted to
be size-neutral and measured at +2.9 MB (16-bit residuals carry more entropy than 8-bit),
a prediction failure recorded at the time and kept as a deliberate fidelity-up.

P-frames store meshopt-coded 16-bit position deltas against the previous frame, applied
with wrap-around arithmetic (`(prev + delta) & 0xffff`) so reconstruction is exact and
drift-free; a seek re-rolls deltas from the covering keyframe. The encoder chooses coding
per GOP automatically: if topology is bit-stable across the GOP (checked by index
spot-sampling), it emits I+P; under `--track` it forces persistent topology by resampling
every frame onto the keyframe's vertex set via a spatial-hash nearest-point search,
transferring each frame's own UVs by barycentric interpolation because re-atlased captures
lay out a fresh atlas every frame; otherwise the GOP falls back to intra. Intra frames get
the reorder optimization; temporal I-frames do not, because P deltas require stable vertex
order across the GOP.

The persistent-topology bet (spec section 6.5) is that for most human captures 95-99% of
connectivity is stable within a GOP, so topology, UVs, and materials can be stored once
per GOP while per-frame data reduces to sparse position residuals. The spec classifies
captures by topology behavior (section 6.4): stable, semi-static, dynamic, and unknown,
with each class degrading gracefully toward intra coding. The reference capture is the
hard case: independent per-frame reconstruction with vertex counts drifting between 11,211
and 11,397 and no correspondence at all. Tracking was implemented, evaluated against this
capture, and pulled from the pipeline for this capture class, because resampling degrades
re-atlased texture mapping more than the size win justifies. The 4DViews teardown keeps
temporal geometry validated as the right target; section 12 describes the research
direction (quad remeshing co-designed with a persistent atlas) that would make it
applicable here.

One failure lesson from this work is general enough to state as a rule. The first Taubin
smoothing implementation used index-based adjacency; atlased meshes duplicate vertices
along UV-chart seams, the duplicates smoothed apart, and every frame showed cracked seams.
The fix welds vertices by exact position bits, smooths once per welded group, and scatters
the result back; the same weld applied to normal computation removed a shading line that
had followed every chart boundary (1,193 seam groups and 2,524 duplicated vertices on this
capture; the post-fix crack test measures a maximum seam-group spread of exactly zero).
Any per-vertex geometry filter on an atlased mesh must weld positional duplicates first.

## 7. Texture coding

The texture track is ordinary video: the per-frame PNG atlas sequence is encoded to VP9 or
AV1 with one ffmpeg invocation per geometry GOP, forcing one closed video GOP per chunk so
every chunk stays independently decodable. On the reference capture this turned 1.11 GB of
PNGs into 14.7 MB of VP9, later 12.5 MB of AV1 at matched quality. The video-codec win is
the single largest number in the project, and it is also where the 4DViews comparison
inverts: the shipping `.4ds` file spends about 130 MB (9 s equivalent) on per-frame
GPU-block images with no inter-frame compression.

Decode goes through WebCodecs `VideoDecoder` rather than a `<video>` element. The spec
records the reasoning as a sanity check (section 7.1): a `<video>` element owns its own
clock and cannot provide frame-accurate pull access, which volumetric playback needs
because geometry frame N must present with texture frame N. The runtime feeds coded
chunks with a lookahead of four frames, keys them by frame index, and on present takes the
freshest decoded frame at or below the requested index, so a lagging decoder costs at most
one frame of texture staleness instead of a blank. Each decoded `VideoFrame` imports to
the GPU without a CPU pixel copy.

The spec also documents why the tempting shortcut of packing XYZ coordinates into RGB
video fails (section 8.5.1): 8-bit channels quantize positions to 256 levels, 4:2:0 chroma
subsampling halves the resolution of two of the three coordinates, DCT loss that is
invisible in pictures becomes visible wobble in geometry, and YUV conversion mangles
packed bytes. A corrected variant (displacement maps, 4:4:4 at 10-12 bits, geometry-image
parameterization) survives as an experimental profile gated on beating the binary delta
path in a measured Pareto comparison.

A requirement that pairs with persistent topology: the atlas must hold still within a GOP
(spec section 7.4). Video codecs earn their compression from motion estimation, and an
atlas that re-lays-out every frame gives the codec a slideshow of unrelated images. For
re-atlased captures the encoder therefore transfers per-frame UVs (section 6); for
temporally stable atlases the video codec does the temporal work on its own.

## 8. Runtime architecture

`AresPlayer.create({canvas, src})` fetches the container in one request, parses it,
flattens the geometry blocks into frame references, picks a renderer, wires the texture
path, and presents frame zero; time to first frame is measured at that point. The public
surface is small: play, pause, seek, tick for host-driven clocks, resize, camera get/set,
crop and wireframe and edit-preview controls for the editor, a raster-picking helper, and
a stats snapshot (TTFF, per-frame decode and CPU cost as exponential moving averages, fps,
frame index, vertex count, delivered bytes).

The WebGPU renderer keeps three position buffer slots and rotates through them, so decode
writes never race the frame in flight. Topology and UVs upload once per GOP, when an
I-frame is decoded; P-frames stream only position bytes. Buffers grow in place by 1.5x
when a larger GOP arrives. The fragment shader computes a derivative-based face normal
unconditionally (uniform control flow requires it), prefers the interpolated vertex
normal when one exists, and applies the editor's crop-box discard last.

The WebGL2 fallback implements the same renderer interface over vertex attributes: raw
u16 positions bound non-normalized and dequantized in GLSL with the same formula the WGSL
uses. It selects automatically when `navigator.gpu` is absent (or under `?gl2=1`) and was
verified at 61 fps with the AV1 texture on the reference capture. Its texture upload tries
`texImage2D(VideoFrame)` and permanently switches to a `createImageBitmap` path the first
time that throws, guarded by a generation counter so stale async uploads never overwrite
newer frames. Crop preview and wireframe are documented no-ops on this backend; baking is
unaffected.

Worker-thread decode is opt-in (`?worker=1`). The design constraint is buffer ownership:
geometry blocks are subarray views into the single fetched file, so they are copied once
outward (transferring would detach the whole file), while the previous frame's position
buffer transfers zero-copy with the player nulling its reference first so the synchronous
path can never observe a detached buffer. The worker prefetches exactly the next
sequential frame; seeks stay synchronous. One environmental limitation is documented and
handled: import maps do not reach module workers in all browsers, so the bare
`meshoptimizer` specifier can fail to resolve inside the worker, in which case the player
logs the condition and decodes on the main thread. On this capture decode costs well under
a millisecond per frame, so the offload matters mainly for heavier content or
bundler-hosted deployments.

Playback keeps a microsecond clock decoupled from requestAnimationFrame. The spec's sync
policy (section 8.6) forbids presenting mismatched PTS; the implementation approximates it
from the texture side, pairing geometry frame N with the freshest decoded texture frame at
or before N, as described in section 7. The Three.js wrapper (`AresObject`)
and React wrapper (`<Ares/>`) hand the clock to the host; both are thin, and the current
revision renders to its own canvas rather than into the host scene graph, a P1-era
limitation recorded in the source.

## 9. Authoring pipeline and tooling

The encoder CLI (`ares synth | encode | info`) covers synthetic test clips, real capture
encoding, and container inspection. `encode` discovers per-frame OBJ or PLY meshes and
`atlas-*.png` textures (descending into a single frames subfolder when handed a parent
directory), imports them (the OBJ importer deduplicates by position/UV index pairs,
fan-triangulates, resolves negative indices, and flips V for top-left atlases), optionally
bakes an edit list, builds GOPs with the automatic temporal plan from section 6, encodes
texture video per GOP, and muxes the container. `info` round-trips the result through the
runtime demuxer, which keeps encoder and decoder honest against each other.

The dev server (`tools/serve.mjs`, zero dependencies) exists because SharedArrayBuffer,
which the worker path uses, requires cross-origin isolation, and common static servers do
not send the COOP/COEP headers; the capability probe reports isolation as unavailable
under them. The server sends the headers on every response, and it grew endpoints from
there. The endpoints make the demo app self-sufficient on a local machine: `/encode` runs the real CLI and streams progress as server-sent
events; `/pick` opens a native Windows folder dialog (spawned as an STA PowerShell
process, because WinForms dialogs hang on MTA threads); `/analyse` returns mesh count,
vertex count, atlas dimensions, and a size estimate for a folder; `/edits/*` stores
non-destructive edit sidecars; `/sam/*` reverse-proxies the local segmentation service,
necessary because the server's own COEP header blocks direct cross-origin fetches;
`/enhance` drives the texture enhancement tiers.

Texture enhancement has two tiers. The fast tier runs a vendored `realesrgan-ncnn-vulkan`
binary (BSD-3-Clause, tiled at 256 px to fit 6 GB of VRAM) per frame with an optional
ffmpeg blend pass for partial strength, measured at roughly 26 s per frame producing
4096x4096 output, with no server or Python involved. The generative tier posts frames to a
local Stable Diffusion Forge install's img2img API, auto-launching Forge headless when it
is down (cold start 30-60 s) and capping denoising strength at 0.5. Both write an
`-enhanced` sibling folder that drops straight into the encode path. Section 11 covers the
caveat on skin.

The edit format stores no vertex or triangle
index anywhere. Per-frame-reconstructed captures have
no stable vertex numbering, and the encoder's reorder pass renumbers within a frame, so
selections are stored as world-anchored regions (boxes, brush-stroke capsule chains,
camera-plus-screen-rect masks with optional depth bands) keyframed on the timeline.
Between keyframes, regions interpolate by lerping their signed-distance fields, which
needs no correspondence. A triangle is dropped when its centroid falls inside an active
delete region, a rule that never splits a UV-mapped triangle and therefore keeps geometry
and texture paired by construction; the deleted triangles' texels go unsampled, so
no re-atlasing pass is needed and an entire class of texture-transfer failure is skipped.
The same predicate module drives the live preview and the bake, so what the preview shows
is what the encoder removes. Selection follows Blender's convention: in X-ray mode a
marquee selects through the mesh, in solid mode a CPU-rasterized ID pass restricts
selection to visible surface within a depth band.

A local SAM segmentation service (FastAPI, port 7263) backs click-to-select in the editor.
As of 2026-07-10 its primary backend is Meta SAM 3 (transformers `Sam3TrackerModel`, bf16,
weights local at the repo root), with the original SAM v1 ViT-H as automatic fallback; the
dev server starts and monitors the service from inside the app (`/sam/start`, plus a status
row in the Edit panel), so no separate launcher is needed. Measured on the RTX 3060: a
0.93-score mask per click at 360-550 ms warm, ~2.6 GB VRAM total. The editor-side click
tool shipped the same day: a click captures the held frame, the returned mask previews as
a tint and applies as an RLE-coded bitmap region (a `mask2d` variant in the shared edit
module) keyframed into the active range, with the same depth-band visible-only law as the
marquee. Per-image feature caching in the SAM 3 backend and text-prompt selection remain
queued work.

Finally, the Inspect tab and its CLI sibling (`tools/probe-4ds.cjs`) apply the structural
parse described in section 3 to any `.ares` or `.4ds` container, in the browser or the
terminal, uploading nothing.

## 10. Measured results

Intra geometry codecs, Phase 0 benchmark (bench/README.md, run 2026-07-08 on an AMD Ryzen
7 6800H, Node 24, single thread, 14-bit positions; synthetic talk clip, 40,962 vertices,
81,920 triangles). Decoder sizes come from the benchmark's regenerable output (bench/results/intra-latest.json), a 2026-07-09
re-run that reproduces the sizes below exactly while its decode timings drift by fractions
of a millisecond:

| Codec | KB/frame | Share of raw | Decode ms/frame | Decoder size |
|---|---|---|---|---|
| raw f32 | 1440 | 100% | ~0 | none |
| quantized u16 | 1200 | 83% | 0.3 | none |
| quantized + Brotli | 494 | 34% | 5.3 | none |
| meshopt | 210 | 14.6% | 1.1 | 29 KB |
| meshopt + Brotli | 92 | 6.4% | 2.1 | 29 KB |
| Draco edgebreaker cl7 | 41 | 2.9% | 5.5 | 279 KB |

meshopt+Brotli already lands inside the spec's projected 5-15% band before any temporal
coding, which met the P0 exit criterion and fixed meshopt as the intra default. The
harness caveats apply: synthetic corpus, Node timings rather than in-browser workers, and
Draco's error figure is an analytic quantization bound because edgebreaker reorders
vertices.

Delivered size, reference capture, all figures measured except where labeled
(docs/size-comparison.md; reproduction in section 13):

| Format | Geometry | Texture | Total | Files |
|---|---|---|---|---|
| Raw OBJ+PNG | 485.7 MB | 1135.8 MB | 1.58 GB | 544 |
| Draco-GLB sequence | 20.3 MB | 1135.8 MB (PNGs kept) | 1.13 GB | 272 |
| ARES, first encode | 52.4 MB | 14.7 MB VP9 | 67.1 MB | 1 |
| ARES, current keeper | 37.3 MB | 12.5 MB AV1 | 49.7 MB | 1 |
| 4DViews .4ds | ~5.5 MB (9s-equiv) | ~130 MB (9s-equiv) | 2.3 MB (720p, estimate) to ~135 MB (HR, measured) | 1 |

The 67.1 MB encode is 24.2x smaller than raw and 17.2x smaller than the Draco-GLB
sequence; the current keeper is roughly 33x smaller than raw. Section 6 covers the two
geometry steps between those encodes (the reorder, then oct16); the remaining delta is
texture, where AV1 (SVT-AV1, crf 30) took 14.7 to 12.5 MB. The keeper
`daniel-s0.ares` uses smoothing level zero, chosen by visual evaluation on 2026-07-10
over the smoothed variants ("much less warping").

Playback, measured: the P1 exit numbers on an AMD 680M iGPU with the 8.8k-vertex synth
clip were TTFF ~160 ms against a 500 ms budget, 0.35 ms main-thread CPU per frame against
a 3 ms budget, and 60 fps. On the real capture, the recorded main-thread geometry decode
figure is ~0.4 ms per frame, with live HUD readings
between 0.4 and 0.8 ms across machines and sessions; the WebGL2 fallback holds 61 fps, and
playback stays at 59-60 fps in the HUD on the machines tried. Playback metrics are deliberately not tabulated further
because they are machine-dependent; the HUD reports them live.

## 11. Known limitations and failure lessons

Documentation of what does not work, or did not, belongs next to the results.

Geometry size still loses to Draco intra (37.3 vs 20.3 MB on the reference capture) and
by roughly 7x to the temporal mesh codec measured in section 3 (the 9.5x figure there is
computed against the first encode's 52.4 MB geometry). Intra coding re-stores topology every frame; that
is the price of correspondence-free capture, and it is the roadmap's main target.

Per-frame reconstruction shimmers. Each frame is an independent surface, so the mesh
"boils" frame to frame, and because texture is UV-pinned to that surface, geometry jitter
reads as texture shimmer too. Spatial smoothing alone cannot fix a temporal artifact and
adds a slight UV-versus-surface swim of its own, which is why the smoothing-off encode is
the current keeper and temporal denoise sits in the queue with an explicit motion-blur
risk attached.

Two predictions failed and are recorded as such. Oct16 normals were predicted size-neutral
and measured +2.9 MB. An earlier HUD version derived its "Draco" column from ARES's own
byte count scaled by 0.38, a shadow of ARES rather than a baseline; it was replaced with a
real draco3d encode, and any screenshot predating that fix should not be trusted for the
Draco column.

The default texture enhancement model is unusable on skin. `realesrgan-x4plus` is a
general photo GAN; the input here is a UV atlas of disjoint charts with seams and gutters,
and the model invents lines, waves, and bands across chart boundaries, worst on faces. The
cause is representational rather than a settings issue, and the candidate fixes
(face-restoration models, per-chart enhancement with gutter awareness, screen-space
enhancement at render time, or frequency-separation smoothing) are queued research.

Smaller items: worker decode falls back to the main thread in browsers whose module
workers do not inherit import maps; the WebGL2 backend has no crop preview or wireframe;
the quality-tier API is a no-op pending the P3 ladder; the player currently fetches whole
files rather than streaming ranges; B-frames and the rANS entropy stage exist as format
enums without decode paths; the splat profile is specified but unimplemented; and SAM mask
propagation across frames is unresolved on 6 GB hardware (the local SAM 3.1 video-tracker
checkpoint requires a heavier stack and OOM'd a 24 GB card in independent reports; SAM 2.1
small is the documented fallback).

## 12. Roadmap

Near-term: the visual evaluation gate across encode
recipes (in progress; smoothing-off is the keeper so far), SAM toolset follow-ons (feature
cache, text prompts, temporal propagation — the click tool itself shipped 2026-07-10),
meshopt decimation behind a `--decimate` flag (locked atlas-seam borders, UVs untouched;
estimated 5-10 MB off the ~34 MB geometry, to be confirmed with the baseline harness
before any number is quoted), and temporal denoise via approximate nearest-point
correspondence, deliberately last.

The research direction with the largest combined payoff is quad remeshing co-designed with
a fresh atlas: solve one quad topology and one atlas layout per GOP and deform it, instead
of accepting independent per-frame meshes and atlases. That would give trajectory
smoothing real correspondences, so jitter drops; enable the temporal geometry coding the
4DViews teardown validated, cutting into the 37.3 MB; and hold the atlas still for the
video codec, removing seam smear. A per-frame remesher bolted onto the current pipeline was
evaluated and rejected: it targets the wrong axis, destroys
the existing per-frame UVs, and adds inter-frame variance.

The staged spec roadmap continues with P3 streaming (GOP-index seeking over HTTP ranges,
prefetch, an adaptive bitrate ladder with tier switches at GOP boundaries), P4 the
Gaussian splat profile, P5 the importer suite (glTF, Alembic, USD, Depthkit, 4DViews,
HoloVideo), and P6 hardening: fuzzing the demuxer, WebGL2 polish, and a spec freeze at
v1.0. Live capture and generated-avatar pipelines are explicitly out of scope for v1; the
header reserves a live flag so low-latency delivery arrives later as an extension rather
than a redesign.

## 13. Reproducibility

Every published size figure regenerates from the repository. `node
tools/measure-baselines.cjs <captureDir> <file.ares>` measures the raw folder, re-encodes
every OBJ frame with real draco3d (position 14, UV 12, normal 10 bits, edgebreaker), walks
the `.ares` chunk directory for the geometry/texture split, and prints the 4DViews
estimate lines with their assumptions visible. `ARES.vbs bench` (or `node
bench/dist/run.js`) reruns the intra codec benchmark and rebuilds the Pareto report at
`/bench/report/`. `node tools/probe-4ds.cjs <file.4ds>` reprints the structural teardown,
writing nothing. Playback numbers come from the demo HUD on the machine at hand.

## 14. References

Within this repository: the specification source (`../spec/`, chapters cited by section
number above), `size-comparison.md`, `editor-v2-design.md`, and `../bench/README.md`
(the benchmark harness regenerates `bench/results/intra-latest.json` locally).
External prior art is surveyed with citations in spec section 3 and Appendix E; the
figures quoted here for 4DViews tiers combine one measured file with published bitrates,
labeled accordingly where they appear.
