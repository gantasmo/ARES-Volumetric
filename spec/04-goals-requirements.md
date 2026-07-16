## 4. Design goals and requirements

The principles of [§1.4](#14-design-philosophy) become measurable requirements here. Where goals
conflict, the runtime prioritizes **runtime performance over encoding speed**, and **deployment
efficiency over authoring convenience**.

### 4.1 Design goals (ranked)

In priority order — when goals conflict, the higher one wins:

1. **Fast time-to-first-frame (TTFF).** The single most important user-facing metric. A capture must
   begin rendering from a small initial download, before the full asset is present.
2. **Tiny downloads.** Aggressive size reduction versus Draco-GLB sequences, the prevailing baseline.
3. **Extremely low CPU usage.** Decode belongs on the hardware video decoder, in Workers, or on the
   GPU — not the main thread. The render loop must never stall.
4. **GPU-first, GPU-resident frames.** The wire format maps onto GPU buffers with minimal CPU
   transformation.
5. **Progressive streaming and random seek.** Netflix-style chunked delivery, a keyframe/GOP model
   for seeking, and adaptive bitrate.
6. **React / Three.js / WebGPU native.** A drop-in component and a small, dependency-light core.
7. **Predictable, bounded memory.** A configurable cache/ring-buffer budget; no unbounded growth;
   near-zero GC during steady playback.
8. **Open, extensible, versioned format.** A published binary spec with a forward-compatible
   extension mechanism.
9. **No runtime dependence on glTF/GLB.** The runtime carries no glTF parser
   ([§4.9](#49-clarifying-zero-dependence-on-glb)).

### 4.2 Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| F1 | Decode and render a textured, deforming volumetric sequence in a WebGPU browser | MUST |
| F2 | Persistent-topology geometry stream with I (keyframe), P (predicted), and B (bidirectional) geometry frames | MUST |
| F3 | Texture delivered as a hardware-decoded video track (AV1/VP9) via `WebCodecs` | MUST |
| F4 | Single chunked container with a seekable GOP index | MUST |
| F5 | Adaptive bitrate across ≥2 quality tiers | SHOULD |
| F6 | 3D Gaussian-splat geometry profile sharing the same container/timeline | SHOULD |
| F7 | Optional audio track, sync'd to the frame timeline | SHOULD |
| F8 | WebGL2 fallback for the mesh profile (reduced features) | SHOULD |
| F9 | Optional per-frame metadata: markers, subtitles, bounding volumes | MAY |
| F10 | Extension blocks that unknown decoders skip safely | MUST |

Expanded detail on the load-bearing functional requirements:

- **F-Input (encoder).** The encoder SHALL import PLY, OBJ, GLB, Alembic, FBX animation, USD, point-cloud
  sequences, and the vendor formats Microsoft MRC, Depthkit, 4DViews, UVOL, and VVglTF. Gaussian-splat
  and neural datasets are future targets. Importers perform *format translation only*, no optimization.
- **F-Stream.** Assets SHALL support progressive loading, adaptive buffering, chunk prioritization,
  interrupted-download recovery/resume, and HTTP range requests.
- **F-Seek.** Playback SHALL seek without decoding every preceding frame; the encoder SHALL provide
  configurable keyframe intervals and SHALL bound delta chains to cap seek latency.
- **F-Extend.** The binary format SHALL permit additive extensions without modifying existing
  structures; unknown extensions SHALL be safely ignored (F10).
- **F-Meta.** Metadata (capture info, licensing, performer, timestamps, camera calibration,
  reconstruction settings, user attributes) SHALL be separable from runtime-critical data and SHALL
  NOT affect playback performance.

### 4.3 Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | Core runtime (excl. WASM codecs) ≤ ~50 KB gzipped. **[PROJECTED]** |
| N2 | No main-thread task > 8 ms during steady-state playback (headroom in a 16.6 ms frame). |
| N3 | Deterministic, bounded GPU + CPU memory given a configured cache budget; transient allocation minimized; geometry buffers resident; reusable pools over repeated allocation. |
| N4 | Graceful degradation: on decode/network stall, hold the last frame or drop to a lower tier — never crash or leak. |
| N5 | All multi-byte fields little-endian; format independent of host endianness. |
| N6 | Security: never `eval`; treat all container data as untrusted (bounds-check every offset). |
| N7 | Cross-origin isolation (COOP/COEP) required only for the `SharedArrayBuffer` fast path; a non-isolated fallback MUST exist ([§10.5](#105-threading-sharedarraybuffer-and-cross-origin-isolation)). |
| N8 | **Determinism.** Given identical encoded assets, decoder output SHALL be deterministic across supported platforms; floating-point error SHALL remain within predefined tolerances. |
| N9 | **Stability over peak.** Frame-time *variance* matters more than average frame time; stable delivery is preferred to bursts followed by stalls. |
| N10 | **Portability.** Consistent execution on Windows, Linux, macOS, Android, iOS; platform-specific optimizations remain optional. |

### 4.4 Browser requirements

The runtime SHALL account for browser subsystems that add latency uncommon in native apps, and SHALL
minimize interaction with the unpredictable ones:

- JavaScript garbage collection (minimize hot-path allocation; N3).
- Asynchronous execution and worker communication (transferables over clones).
- `SharedArrayBuffer` availability gated by cross-origin isolation (N7).
- WebGPU feature detection with WebGL2 fallback (F8).
- Browser/tab memory limits (bounded budget, N3).

### 4.5 Performance targets

All values are **[PROJECTED]** — hypotheses to be confirmed by the
[§13](#13-benchmark-methodology-and-projected-performance) methodology. They assume a mid-tier 2025
laptop / recent flagship phone, a ~30 fps human capture (~30–80k triangles/frame or ~150–400k
splats), and a good network.

| Metric | Baseline (Draco-GLB seq) | ARES target | Basis for target |
|---|---|---|---|
| Size per second @ "high" | 8–20 MB/s | **1.5–4 MB/s** | Temporal geometry + video texture |
| Time-to-first-frame (cached) | 1.5–4 s | **< 250 ms** | Small I-frame + progressive |
| Time-to-first-frame (broadband) | 2–5 s | **< 2 s** | Chunk 0 only |
| Main-thread CPU / frame | 6–15 ms | **< 3 ms** | HW decode + Worker + GPU upload |
| Steady-state FPS | 24–30 | **60** (playback ≥ capture fps) | GPU-resident triple buffer |
| GPU upload / frame | varies | **hidden behind buffering** | Delta upload, persistent buffers |
| Seek latency (to any point) | 0.5–3 s | **< 250 ms** | GOP index + keyframe fetch |
| Memory growth (continuous) | high/unbounded | **constant, bounded** | Ring buffer + budget |

> These are deliberately ambitious. The document's job is to make them *falsifiable*: each is tied
> to a mechanism, and [§13](#13-benchmark-methodology-and-projected-performance) specifies how to
> measure it. Actual targets require empirical validation.

### 4.6 Compression philosophy: per-subsystem strategy

No single algorithm is optimal across every component; compression is chosen by the **statistical
properties** of each data type ([§8](#8-compression-architecture)):

| Subsystem | Strategy |
|---|---|
| Geometry | predictive coding, delta compression, quantization, entropy coding |
| Textures | video codecs (AV1/VP9) for motion; AVIF/KTX2/Basis/WebP for static |
| Animation | motion prediction, temporal deltas, sparse updates |
| Metadata | conventional lossless (Zstd/Brotli) |

### 4.7 Temporal-coherence categorization

A foundational requirement that shapes both geometry ([§6](#6-geometry-representation)) and texture
([§7](#7-texture-and-video-encoding)): information is classified by *how fast it changes*, and each
class gets an appropriate strategy.

| Class | Examples | Strategy |
|---|---|---|
| **Static** | topology, UV layout, material definitions, vertex ordering, tangent basis | store once per GOP |
| **Slowly changing** | texture, normals, material parameters, lighting/shadow, subtle wrinkles | low-rate updates / temporal deltas |
| **Rapidly changing** | vertex displacement, visibility, local topology, facial expression, cloth motion | per-frame sparse deltas |

This taxonomy is the conceptual core of the whole design: ARES exists to stop paying static-data cost
on every frame.

### 4.8 Progressive refinement

To satisfy the TTFF goal (4.1) without sacrificing quality, an asset SHOULD become usable before it
is fully downloaded, refining in stages that each improve quality without interrupting playback:

```
1. Bounding volume        (instant placeholder)
2. Low-resolution proxy   (coarse mesh / low splat count)
3. Reduced vertex density (mid LOD)
4. Full geometry
5. High-resolution textures
```

### 4.9 Clarifying "zero dependence on GLB"

The plan states two things that appear to conflict — "zero dependence on GLB" and "convert from glTF
sequences" — which are not in conflict:

- The **runtime** MUST NOT require a glTF/GLB parser to play an ARES file. No `GLTFLoader`, no JSON
  scene graph, no Draco decoder on the critical path unless a capture explicitly uses the Draco
  fallback profile.
- The **encoder** MAY ingest glTF/GLB (and OBJ/Alembic/USD/etc.) as *source* material. Import is an
  offline concern with no runtime cost.

An OPTIONAL `glTF-extension bridge` for interop with the Khronos Volumetric subgroup is future work
([§15](#15-future-research-the-avatar-pipeline)), explicitly *not* part of the core runtime.
