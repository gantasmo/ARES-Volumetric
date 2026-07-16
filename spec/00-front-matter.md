# ARES Runtime — Technical Specification

**A browser-first volumetric media runtime and container format**

Name: **ARES Volumetric** (short form **ARES**)
Container extension: `.ares`

| | |
|---|---|
| **Document** | ARES Runtime Technical Specification |
| **Version** | 0.2 (Draft — consolidated master) |
| **Status** | Working draft — for implementation planning |
| **Date** | July 2026 |
| **Editors** | Project ARES |
| **Intended audience** | Engine/runtime engineers, codec engineers, tooling authors |
| **Supersedes** | *ARES — Browser-First Volumetric Media Runtime, Architecture and Specification, Draft 0.1* (the 57-page §1–5 document) |

### Provenance

This is the consolidated, single-source master specification. It merges and supersedes three earlier
artifacts:

- **`ARES.pdf`** (Draft 0.1) — a 57-page formal document that fully developed §1–5 (Introduction,
  Existing Ecosystem, Design Requirements, System Architecture, Geometry) and then stopped at §5.18
  with a note requesting a dedicated chapter on treating geometry as a video-compression problem.
  That chapter is now [§8](#8-compression-architecture)/[§8.5](#85-video-assisted-geometry-packing-attributes-into-pixels).
  Draft 0.1's conceptual depth (topology classification, temporal categorization, encoder pipelines,
  cache hierarchy, determinism/portability requirements, regional and hybrid geometry) is retained
  and folded in here.
- **`ARES_Project_Implementation_Outline.pdf`** — the one-page objectives + comparison table.
- **`Volumetric Video Codecs & Formats (2023–2026).pdf`** — the cited research brief that grounds
  [§3](#3-survey-of-existing-formats) and [Appendix E](#appendix-e--references).

Relative to Draft 0.1, this master (a) completes §6–17 and the appendices, (b) adds concrete binary
layouts, a runtime API, and code, (c) updates the survey for mid-2026 developments (Arcturus AVV,
Brown "PackUV" / CVPR 2026, the Khronos glTF Volumetric subgroup), and (d) flags several technical
traps Draft 0.1 left implicit (WebCodecs vs. `<video>`, the video-geometry precision problem,
`SharedArrayBuffer` cross-origin isolation).

---

## Abstract

ARES is a browser-first volumetric media **runtime** and **container format**. Where
existing volumetric formats optimize *interchange* between digital-content-creation
(DCC) tools, ARES optimizes *delivery* to JavaScript, WebGPU, Three.js, and React
applications. The central design premise is a deliberate inversion of the mesh-per-frame
model that dominates current pipelines:

> **Stop treating every frame as a 3D model. Treat every frame as a compressed set of
> GPU instructions.**

From that premise, ARES is designed around three ideas that legacy formats — built for
DCC interchange — never fully exploited:

1. **Temporal coherence as a first-class citizen.** For most captures (especially of
   humans) 95–99% of mesh connectivity is stable frame to frame. ARES encodes a
   persistent-topology stream with I/P/B-style geometry frames, borrowing five decades of
   video-compression theory for geometry, not just texture.
2. **The hardware video decoder as a geometry engine.** Modern devices ship fixed-function
   AV1/HEVC/VP9 decoders. ARES can route both texture *and* suitably-encoded geometry
   through them via `WebCodecs`, keeping the CPU almost idle.
3. **GPU-resident frames.** The on-the-wire layout is chosen so that a decoded frame can be
   uploaded to GPU buffers with little or no CPU-side transformation, enabling triple-buffered,
   worker-driven playback.

This document specifies the container layout, the geometry and texture codecs, the streaming
model, the runtime architecture, and a conversion toolchain from existing formats (Microsoft
HoloVideo, 4DViews, Depthkit, PLY+PNG, OBJ/glTF/Alembic/USD sequences). It is written as an
engineering specification: **assertions are distinguished from projections**, assumptions are
called out explicitly, and all quantitative targets not yet measured are labeled
*Projected — pending empirical validation*.

---

## Document status and how to read this

This is a **draft specification and research agenda**, not a finished standard. It is intended
to be the canonical reference for an implementation effort, and to be revised section by section
as experiments retire the open questions in [§16](#16-open-research-questions-and-risks).

Notation used throughout:

- **[ASSERTED]** — established technique or fact with a citation or first-principles derivation.
- **[PROJECTED]** — a quantitative estimate pending measurement. Treat as a hypothesis.
- **[ASSUMPTION]** — a premise the design rests on that should be validated early.
- **[OPEN]** — an unresolved design question tracked in [§16](#16-open-research-questions-and-risks).

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**, **MAY**, and
**OPTIONAL** are to be interpreted as described in RFC 2119 / RFC 8174 when, and only when, they
appear in all capitals.

Byte layouts are little-endian unless stated otherwise. `u8/u16/u32/u64` denote unsigned
integers of that width; `f16/f32` denote IEEE-754 floats; `varint` denotes LEB128 unless noted.

---

## Table of contents

1. [Introduction and problem statement](#1-introduction-and-problem-statement)
2. [Background: volumetric representations](#2-background-volumetric-representations)
3. [Survey of existing formats](#3-survey-of-existing-formats)
4. [Design goals and requirements](#4-design-goals-and-requirements)
5. [Overall architecture](#5-overall-architecture)
6. [Geometry representation](#6-geometry-representation)
7. [Texture and video encoding](#7-texture-and-video-encoding)
8. [Compression architecture](#8-compression-architecture)
9. [Streaming architecture](#9-streaming-architecture)
10. [Runtime architecture](#10-runtime-architecture)
11. [File format specification](#11-file-format-specification)
12. [JavaScript / WebGPU implementation](#12-javascript--webgpu-implementation)
13. [Benchmark methodology and projected performance](#13-benchmark-methodology-and-projected-performance)
14. [Development roadmap](#14-development-roadmap)
15. [Future research: the avatar pipeline](#15-future-research-the-avatar-pipeline)
16. [Open research questions and risks](#16-open-research-questions-and-risks)
17. [Conclusion](#17-conclusion)
- [Appendix A — Binary layouts](#appendix-a--binary-layouts)
- [Appendix B — Pseudocode](#appendix-b--pseudocode)
- [Appendix C — Core runtime data structures](#appendix-c--core-runtime-data-structures)
- [Appendix D — Glossary](#appendix-d--glossary)
- [Appendix E — References](#appendix-e--references)
