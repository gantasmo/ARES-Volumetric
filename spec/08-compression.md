## 8. Compression architecture

This section specifies how bytes are actually saved, and — critically — subjects the plan's
"store geometry in video" idea to the precision analysis it requires before anyone builds it.

### 8.1 The compression stack (mesh profile)

```
Positions ─ quantize (14-bit/axis over GOP AABB)
          ─ spatial prediction (parallelogram / neighbor)
          ─ temporal prediction (P/B vs anchor + velocity)
          ─ meshopt vertex codec (SIMD)         ┐
Indices   ─ meshopt index codec                 ├─ per-frame geometry blob
Normals   ─ octahedral 2×10-bit (or GPU-derived)┘
Residuals ─ range/ANS entropy coding
Container ─ chunk = I-frame + P/B run + texture-video segment
```

Each stage is independently ablatable — required by Phase 0/Phase 6 benchmarking. The expected
contribution ordering (largest savings first) is: **temporal prediction ≫ quantization > entropy
coding > index coding.** **[PROJECTED]**

### 8.2 Intra (I-frame) geometry

An I-frame carries the full mesh: canonical interleaved, quantized positions/normals/UVs + meshopt
codecs (§6.7). It is the random-access point (a geometry keyframe) and the base for the GOP's deltas.
Size is dominated by vertex count × per-vertex bits; a 40k-vertex human at 14-bit positions +
octa normals + 14-bit UVs is on the order of tens of KB after meshopt. **[PROJECTED]**

### 8.3 Inter (P/B-frame) geometry

P/B frames carry quantized residuals against a prediction (§6.6). Static regions produce near-zero
residuals that entropy-code to almost nothing. The **GOP length** trades random-access granularity
(shorter = faster seek, larger) against size (longer = smaller, coarser seek). Default 30–60 frames
(1–2 s at 30 fps), matching the streaming chunk (§9) and PackUV's chunking guidance [43].

> **[SANITY CHECK — "keyframe every 60 frames, delta in between"]** The plan's instinct matches
> video GOP structure exactly and is correct. The subtlety the plan omits: an I-frame MUST also be
> forced whenever **topology changes** or **tracking error exceeds threshold** (§6.5.1), not only on
> a fixed cadence. Fixed-cadence-only keyframing would accumulate drift or break on occlusion.

### 8.4 Texture compression

Covered in [§7](#7-texture-and-video-encoding): AV1 (default) / VP9 video track over a
temporally-stable atlas, multi-resolution ladder, optional separate auxiliary track. The atlas
stability requirement (§7.4) is what makes the codec's existing motion estimation do the temporal
work for free.

### 8.5 Video-assisted geometry: packing attributes into pixels

The plan's most exciting — and most dangerous — idea: encode geometry into a video and let the
hardware decoder reconstruct it ("R = vertex x, G = vertex y, B = vertex z"). Microsoft, V-PCC, and
PackUV all prove *a* version of this works. But the naïve RGB-position mapping does **not** work, and
understanding why is essential.

#### 8.5.1 Why naïve "XYZ in RGB" fails

Four independent failure modes, each fatal on its own:

1. **Bit depth.** 8-bit video gives **256 levels per channel**. A vertex coordinate needs ~12–16
   bits. 256 positions across a body is centimeters-to-decimeters of quantization — visibly wrong.
   Even 10-bit video (1024 levels) is marginal for absolute positions. [ASSERTED]
2. **Chroma subsampling.** Standard 4:2:0 video stores full-resolution luma but **quarter-resolution
   chroma**. If X→R, Y→G, Z→B naïvely (converted to YUV), two of your three coordinates are
   spatially downsampled and cross-contaminated. Geometry would smear. [ASSERTED]
3. **Lossy DCT + inter prediction.** Video codecs are *perceptually* lossy: they discard
   high-frequency detail and quantize DCT coefficients. Applied to a "geometry image," this produces
   blocking and ringing **in the geometry** — wobbling surfaces, popping vertices. [ASSERTED]
4. **YUV color conversion.** The codec operates in YUV and applies a color transform; treating your
   packed bytes as RGB fights the codec's own colorspace handling. [ASSERTED]

> **Conclusion:** "just put XYZ in RGB and encode AV1" will yield unusable geometry. This must be
> stated plainly so no one wastes a sprint on it.

#### 8.5.2 How to do it correctly (if at all)

The techniques that make video-coded geometry actually work:

- **Encode displacement, not absolute position.** Over persistent topology, per-vertex *displacement*
  from the I-frame is small-magnitude and smooth → far more tolerant of 8–10-bit quantization and
  DCT than absolute positions. This is the ARES-native synergy: persistent topology makes the
  video-geometry path viable in the first place.
- **Use lossless / near-lossless, 4:4:4, 10–12-bit** configurations where the codec supports them,
  accepting lower ratio for correctness. AV1 supports 4:4:4 and higher bit depth.
- **Split high/low bytes across channels or tiles**, or use a **geometry image** parameterization
  (a 2D chart of the surface, à la V-PCC atlases / Depthkit depth packing) so spatial coherence in
  the map matches the codec's assumptions.
- **Keep a residual correction stream.** Decode the video-geometry to approximate positions, then
  apply a small entropy-coded correction to hit target precision — the video carries the bulk motion
  cheaply; the correction guarantees fidelity.

#### 8.5.3 ARES position on video-geometry

Video-assisted geometry is a **profile**, not the default, for v1:

- **Default mesh profile:** binary meshopt I-frames + entropy-coded P/B deltas (§8.1–8.3). Geometry
  does **not** go through the video decoder; texture does. This is the lowest-risk path to shipping.
- **Video-geometry profile (experimental):** displacement-image packing through AV1 4:4:4 + residual
  correction, evaluated against the default in Phase 2. Adopt only if it beats the default on the
  size×quality×CPU Pareto front. [OPEN]
- **Splat profile:** may use PackUV-style attribute-in-video packing [43] because splats tolerate it
  better than meshes (no connectivity to corrupt) — evaluated in Phase 2.

This keeps the exciting idea alive as a measured experiment while protecting the shipping timeline
from its risks.

### 8.6 Synchronization of geometry and texture streams

Two independently-decoded streams (geometry Worker; texture VideoDecoder) must present the *same*
frame together. This is a real hard problem the plan lists but does not solve.

- Every geometry frame and every texture frame carries a **presentation timestamp (PTS)** on the
  shared timeline (§11 chunk headers). The runtime presents a composed frame only when both the
  geometry and texture for that PTS are ready.
- A small **rebuffer/hold policy**: if one stream lags, hold the last complete composed frame rather
  than tearing (showing geometry N with texture N−1). Never present mismatched PTS.
- The texture ladder switch and geometry tier switch happen only at **GOP boundaries** so the two
  streams stay structurally aligned. [ASSERTED]

### 8.7 Neural / learned compression (future, not v1)

Gaussian-splat pruning/quantization, learned entropy models, and neural residual coding are active
research and can slot into the entropy-coding stage later. They are **out of scope for v1** (decode
cost and determinism), tracked in [§15](#15-future-research-the-avatar-pipeline) and
[§16](#16-open-research-questions-and-risks). [ASSERTED]

### 8.8 End-to-end size budget (illustrative, projected)

A 10-second, 30 fps human capture, "high" tier. **[PROJECTED] — illustrative, pending measurement.**

| Component | Per second | 10 s total | Notes |
|---|---|---|---|
| Geometry I-frames (1/s, ~40k verts) | ~40 KB | ~0.4 MB | meshopt intra |
| Geometry P/B deltas (29/s) | ~0.3–0.8 MB | ~3–8 MB | dominant geometry cost |
| Texture video (AV1, 1024²) | ~1–2.5 MB | ~10–25 MB | dominant overall cost |
| Audio (Opus) + metadata | ~20 KB | ~0.2 MB | optional |
| **Total** | **~1.5–3.5 MB/s** | **~15–35 MB** | vs ~80–200 MB Draco-GLB seq |

The texture video dominates, which is why [§7](#7-texture-and-video-encoding) (atlas stability, ABR
ladder) matters as much as the geometry cleverness. Both are needed to hit the
[§4.5](#45-performance-targets) targets.
