## 3. Survey of existing formats

This section surveys what exists as of mid-2026, what each format does well, where it is weak for
*browser delivery specifically*, and the concrete lesson ARES draws. The goal is explicitly to
learn from years of accumulated field experience — including these systems' known pain points — not
to reinvent their mistakes.

Each system is judged against seven criteria relevant to browser delivery:

| Criterion | Meaning |
|---|---|
| Storage efficiency | Encoded size relative to geometric complexity |
| Decode complexity | CPU/GPU resources required during playback |
| Streaming capability | Ability to progressively load and seek |
| Browser suitability | Compatibility with JS/WebGPU without proprietary runtimes |
| Temporal compression | Exploitation of frame-to-frame coherence |
| Extensibility | Ease of supporting new rendering techniques |
| Production maturity | Stability of the existing pipeline |

**The baseline — raw mesh sequences.** The simplest representation stores one complete mesh per
frame (PLY/OBJ/STL/FBX/Alembic/glTF). A 30 fps sequence contains thirty complete models per second
regardless of how little changes. It is simple, randomly accessible, and DCC-compatible, but it
duplicates geometry, has no temporal compression, and creates a large number of filesystem objects.
Every other format — and ARES's projected numbers ([§13](#13-benchmark-methodology-and-projected-performance))
— is measured against this reference.

### 3.1 Microsoft HoloVideo / MR volumetric codec

The lineage from Microsoft's Mixed Reality Capture Studios: textured mesh sequences delivered in an
`.mp4`-wrapped codec where **geometry is packed into the video bitstream alongside texture**, so a
single hardware video decode recovers both.

- **Strengths.** Proves the thesis that a hardware video decoder can carry geometry; single-decode
  A/V-style sync; battle-tested at scale.
- **Weaknesses for web.** Proprietary; the decoder historically shipped as a native/Unity plugin,
  not a browser-native path; per-frame meshes (no persistent topology); large files.
- **Lesson for ARES.** Adopt the "geometry rides the video decoder" idea, but do it through the
  open `WebCodecs` API with AV1/VP9, and pair it with persistent topology to cut what must be
  encoded in the first place.

### 3.2 4DViews (HOLOSYS)

Capture-stage vendor producing "lightweight" textured meshes ready for engines [37], with its own
`.4ds` sequence container and Unity/Unreal/web players.

- **Strengths.** High capture quality; established runtime players; practical LOD.
- **Weaknesses for web.** Vendor container and player; per-frame meshes; web playback is a port of
  an engine runtime rather than a browser-native design.
- **Lesson for ARES.** The *player ergonomics* (drag-in-and-play, LOD) matter as much as the codec;
  ARES must ship a great `@react-three/fiber` component, not just a spec.

### 3.3 Depthkit

Popular capture/reconstruction toolkit that emits textured meshes and a combined color+depth video
layout ("CPP" — a single video frame carrying color and packed depth) [36].

- **Strengths.** Accessible; the color+depth-in-one-video layout is elegant and web-friendly.
- **Weaknesses for web.** Still fundamentally per-frame reconstruction; quality bounded by the
  single-perspective depth layout for some captures.
- **Lesson for ARES.** Depthkit's packed color+depth frame is essentially a *geometry image*
  (§8.5). ARES generalizes this to arbitrary attribute packing and multi-view.

### 3.4 UVOL (Universal Volumetric)

Open format (Etereal/`.uvol`): each frame is a **Draco-compressed mesh** plus **KTX2/Basis
textures**, organized by a JSON manifest [4].

- **Strengths.** Open; uses GPU-friendly KTX2/Basis; a real reference for a manifest-driven web
  player.
- **Weaknesses for web.** Per-frame Draco (CPU-heavy decode); manifest + many files (request
  amplification); no temporal geometry prediction; seeking is asset lookup.
- **Lesson for ARES.** UVOL is the closest open prior art and the most honest baseline. ARES's job
  is to (a) collapse the many-files model into one chunked container, (b) replace independent Draco
  frames with I/P/B persistent-topology geometry, and (c) prefer meshopt/GPU-side dequantization
  over CPU-heavy Draco where latency matters.

### 3.5 VVglTF

A 2025 research prototype adapting glTF for **streaming** volumetric frames over HTTP: the sequence
is split into glTF "segments" streamed and rate-adapted to network conditions [16].

- **Strengths.** Demonstrates chunked streaming and frame-rate adaptation on top of the familiar
  glTF stack.
- **Weaknesses for web.** Inherits glTF's per-frame interchange overhead; still mesh-per-frame at
  heart.
- **Lesson for ARES.** Validates chunked streaming + ABR for volumetric; ARES keeps the streaming
  model but drops the glTF envelope in favor of a purpose-built chunk.

### 3.6 MPEG V-PCC and G-PCC

International standards for point-cloud compression: **V-PCC** projects a point cloud onto 2D
patches and encodes them with a conventional video codec (geometry, occupancy, attribute videos);
**G-PCC** uses octree/predictive coding for sparse clouds [39].

- **Strengths.** Rigorous, standardized; V-PCC is the canonical proof that geometry-through-video
  works and interoperates.
- **Weaknesses for web.** No browser-native decoder; a full V-PCC decoder in WASM is heavy;
  optimized for point clouds, not textured meshes or splats.
- **Lesson for ARES.** Borrow V-PCC's *atlas/patch* thinking for attribute packing
  ([§8.5](#85-video-assisted-geometry-packing-attributes-into-pixels)), but keep the reconstruction
  cheap enough for a WebGPU compute shader rather than a general standards decoder.

### 3.7 Arcturus AVV (HoloSuite)

Commercial volumetric codec (2024) advertising near-lossless compression, **multi-resolution
textures**, and **per-vertex motion vectors** for real-time playback and inter-frame
interpolation [13].

- **Strengths.** Explicitly uses motion vectors and multi-res textures — both directly relevant to
  ARES; reports compressing assets to ~25% of original with negligible loss [13].
- **Weaknesses for web.** Proprietary pipeline and runtime.
- **Lesson for ARES.** Per-vertex motion vectors are exactly the P-frame mechanism ARES formalizes;
  multi-resolution texture is the ARES texture LOD ladder ([§7.6](#76-multi-resolution-texture-ladder)).

### 3.8 Khronos glTF Volumetric subgroup and PackUV

In 2026 Khronos launched a **glTF Volumetric subgroup** to extend glTF for 3D video, explicitly
noting neural techniques including Gaussian Splatting [7][16]. Separately, Brown's **PackUV**
(CVPR 2026) maps 3D Gaussian-splat frames into 2D video tracks for compatibility with standard
codecs, and recommends splitting long sequences into short chunks to reset stream state and handle
object entry/exit [43].

- **Lesson for ARES.** The standards direction and ARES agree on the destination (glTF-adjacent
  extensions, splats, video-coded attributes). ARES's differentiator is being a *runtime-first*
  design rather than an *interchange-first* extension: ARES can be a concrete, shipping
  implementation of ideas the subgroup is standardizing, and can later expose a glTF-extension
  import/export bridge.

### 3.9 Comparative summary

Relative size is normalized to raw PLY+PNG = 100%. All non-ARES figures are drawn from vendor and
research claims [4][13][16][32]; the ARES row is a **[PROJECTED]** target
([§13](#13-benchmark-methodology-and-projected-performance)).

| Format | Rel. size | CPU decode | Browser-native | Persistent topology | Temporal geometry | Random seek | Notes |
|---|---|---|---|---|---|---|---|
| Raw PLY + PNG | 100% | Very high | n/a | — | — | — | Reference |
| OBJ / FBX / Alembic seq | 70–100% | High | No | — | — | — | Authoring formats |
| GLB (uncompressed) | 60–90% | High | Yes | No | No | Per-asset | Interchange metadata heavy |
| Draco GLB seq | 15–35% | High (Draco) | Yes | No | No | Per-asset | Common web baseline |
| Meshopt GLB seq | 20–40% | Low–med | Yes | No | No | Per-asset | Faster decode than Draco |
| Depthkit | 30–50% | Med | Partial | No | Partial (video) | Video seek | Color+depth video |
| Microsoft HoloVideo | 20–40% | Med | No (plugin) | No | Yes (video) | Video seek | Geometry-in-video |
| 4DViews (.4ds) | 20–40% | Med | Port | No | Partial | Yes | Vendor runtime |
| UVOL (.uvol) | 15–35% | Med–high | Yes | No | No | Per-asset | Draco + KTX2 + manifest |
| VVglTF | 15–30% | Med | Yes | No | No | Segment | Streaming glTF |
| Arcturus AVV | ~25% | Med | No | Partial | Yes (MV) | Yes | Motion vectors, multi-res tex |
| **ARES (target)** | **5–15%** | **Low** | **Yes** | **Yes** | **Yes (I/P/B)** | **Yes (GOP)** | **[PROJECTED]** |

### 3.10 Synthesis: the ARES thesis in one paragraph

Every mature system either (a) rides a hardware video decoder but is proprietary and per-frame
(Microsoft, 4DViews, Arcturus, Depthkit), or (b) is open and browser-native but per-frame and
CPU-heavy (UVOL, Draco-GLB, VVglTF). **No open format combines browser-native decode, persistent
topology, and temporal geometry compression.** That gap is the ARES thesis: an open, runtime-first
container that treats geometry with the same I/P/B temporal machinery the industry already trusts
for pixels.
