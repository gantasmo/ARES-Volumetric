## 1. Introduction and problem statement

### 1.1 Motivation

Volumetric video — free-viewpoint capture of real people and objects as a time sequence of
3D frames — has matured on the capture side (Azure Kinect rigs, 4DViews HOLOSYS stages,
Depthkit, photogrammetry domes, neural reconstruction) far faster than on the *delivery* side.
Unlike conventional video, a volumetric frame contains continuously changing **geometry** in
addition to surface appearance: each frame may carry hundreds of thousands of vertices whose
positions, normals, texture coordinates, colors, and connectivity change over time. A single second
of raw capture is enormous: the Fraunhofer "Dimitri" sequence (2.8 M points at 30 fps) would
require roughly **110 Gbps uncompressed** [1]. Even after meshing, decimation, and codec
compression, volumetric sequences routinely exceed conventional video by one or more orders of
magnitude.

The web is where this content increasingly needs to live — product configurators, WebXR
experiences, live performers, virtual try-on, telepresence, digital preservation — yet the browser is
the most constrained delivery target: no arbitrary native codecs, a single-threaded main loop that
must not stall, garbage collection that must not spike, a strict memory ceiling on mobile, and users
on metered networks who abandon after a few seconds of blank screen.

### 1.2 Problem statement

Current volumetric workflows typically optimize **one stage** of the pipeline while accepting
inefficiency elsewhere:

- capture systems emphasize acquisition quality;
- DCC tools emphasize editing flexibility;
- interchange formats emphasize compatibility;
- streaming systems emphasize transport;
- rendering engines emphasize visualization.

Each optimization adds a *translation stage* before content reaches the browser. A representative
pipeline is a long chain, and every arrow is storage, time, metadata, or allocations:

```
Capture → Reconstruction → Mesh cleanup → Retopology → Texture baking → Export
        → Interchange format → Compression → Packaging → HTTP delivery
        → Browser parsing → Geometry decoding → GPU upload → Rendering
```

On top of this, the dominant web pipeline is **mesh-per-frame**: each frame becomes an independent
asset (typically a Draco-compressed glTF/GLB plus a WebP/KTX2 texture), listed in a JSON manifest and
loaded sequentially [4][16]. This is the correct baseline to beat, but it carries structural
inefficiencies that no amount of per-asset tuning removes:

1. **Every frame re-transmits its own topology and interchange metadata.** glTF is an interchange
   format; each GLB re-encodes accessors, buffer views, materials, and a scene graph that is
   *identical* across thousands of frames. [ASSERTED]
2. **Temporal redundancy is discarded.** Consecutive frames of a talking person differ by centimeters
   of surface motion, yet each is compressed *independently*. Video codecs would never do this for
   pixels; mesh-per-frame does it for geometry. [ASSERTED]
3. **CPU decode dominates.** Draco decoding is comparatively expensive and, unless carefully
   offloaded, competes with the render loop [29]. The pipeline is CPU-bound exactly where the browser
   is weakest.
4. **Repeated allocation.** New mesh objects and GPU buffers per frame drive garbage collection and
   buffer re-creation — both sources of frame-time variance.
5. **Request amplification.** Thousands of small files stress connection pools and caches and make
   smooth seeking hard; there is no codec-style keyframe/GOP structure or ABR ladder.

ARES treats browser playback as the **primary design objective**, not the final stage of an existing
production pipeline.

### 1.3 The core premise

ARES rejects the framing that a volumetric frame *is a 3D model that happens to be one of many*.
Instead:

> A volumetric frame is a **compressed set of GPU state changes** — new positions for a mostly
> unchanged vertex set, a new region of a texture atlas, a handful of topology patches — applied on
> top of the previous frame.

Under this framing the problems above dissolve into well-understood video-coding problems:

| Mesh-per-frame framing | ARES framing |
|---|---|
| Frame = independent asset | Frame = delta on prior GPU state |
| Topology re-sent every frame | Topology persists; patched rarely |
| Independent (intra) compression only | I / P / B geometry frames + motion prediction |
| CPU parses glTF + Draco | Hardware/`WebCodecs` decode → direct GPU upload |
| Thousands of files | One container, chunked like an MP4 |
| Seek = find an asset | Seek = jump to nearest geometry keyframe |

### 1.4 Design philosophy

Seven principles guide every decision in this document. They are stated here and made concrete as
requirements in [§4](#4-design-goals-and-requirements).

1. **Deployment, not interchange.** The runtime delivers finished assets; editing stays in existing
   formats. Compatibility is the *encoder's* job, achieved by conversion tools
   ([§14](#14-development-roadmap)), never by adding authoring features to the runtime.
2. **GPU-oriented data layout.** On-wire structures resemble GPU buffer layouts to minimize
   translation between storage and rendering.
3. **Temporal coherence.** Information unchanged between frames is neither re-stored nor re-decoded.
4. **Progressive streaming.** Playback begins before complete download whenever practical.
5. **Asynchronous execution.** Long-running work runs off the render thread whenever the browser
   permits.
6. **Open specification.** Fully documented; no dependence on proprietary software.
7. **Extensibility.** New geometry representations, codecs, and rendering APIs are incorporable
   without invalidating existing assets.

### 1.5 What ARES is and is not

- ARES **is** a *delivery and playback* format and runtime. Its container exists to feed a
  JavaScript/WebGPU renderer efficiently.
- ARES **is not** an interchange format and does not replace glTF, USD, Alembic, or FBX for authoring.
  Those remain the *inputs* to the ARES encoder.
- ARES **is not** tied to a single geometry representation. Textured meshes, persistent-topology
  deforming meshes, and 3D Gaussian splats are all first-class citizens of the same container
  ([§6](#6-geometry-representation)).

> **[ASSUMPTION A1]** The runtime target is WebGPU-capable browsers (Chromium, Safari, and Firefox
> all ship WebGPU by 2026), with a WebGL2 fallback path for the mesh profile. WebGPU is assumed
> available on the primary target devices.

> **[ASSUMPTION A2]** `WebCodecs` (`VideoDecoder`/`VideoEncoder`) is available and exposes
> hardware-accelerated AV1 and/or HEVC decode on the primary target devices. This is load-bearing for
> the video-assisted paths ([§7](#7-texture-and-video-encoding), [§8](#8-compression-architecture))
> and is validated in the roadmap's Phase 0.

### 1.6 Scope

**In scope:** runtime architecture; binary file format; geometry representation; texture
representation; compression research; streaming protocol; decoder architecture; JavaScript runtime;
WebGPU integration; WebGL compatibility; conversion utilities; performance benchmarking; future
extensions.

**Out of scope:** capture hardware; photogrammetry/reconstruction algorithms; neural-reconstruction
methods; animation-authoring workflows; DCC software. Neural reconstruction and the automated avatar
pipeline appear only as a **future module** ([§15](#15-future-research-the-avatar-pipeline)),
explicitly outside the v1 runtime.
