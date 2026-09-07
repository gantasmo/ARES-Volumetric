# RGBD rebuild pipeline: 2.5D depth-sensor captures → volumetric `.ares`

Status 2026-07-17. This documents the pipeline that turns legacy single-sensor RGBD
captures (Depthkit "combined-per-pixel" exports: color + hue-encoded depth stacked in one
video) into watchable volumetric clips. The reference material is a licensed 21-take
capture set whose subject matter is under NDA — this document therefore contains **no
imagery and no capture identifiers**; it describes the machinery. The lab scripts live in
`scratchpad/` (untracked); the encoder, viewer, and SAM service they drive are all in this
repo.

## Input format and decode

A Depthkit CPP export is one H.264 mp4 with the color image and a rainbow depth image
stacked vertically, plus a JSON metadata sidecar (depth intrinsics, principal point, near/
far clip). Depth decodes per Depthkit's own shader math: validity = `pow(V, 6) > 0.51` in
HSV, `depth01 = hue`, `z = hue·(far−near) + near`, then pinhole unprojection through the
sidecar intrinsics. Which half is depth is auto-detected by saturation census (the depth
half is the rainbow). Millimetre output; the color half becomes the texture atlas verbatim,
so the layout is stable and the encoder's image-based repack detection lets the texture
inter-code.

Three decode gates matter (found the hard way):

- **4:2:0 chroma smear** — hue lives in quarter-resolution chroma, which invents phantom
  mid-range depths along silhouettes. A saturation gate (≥ 0.3) kills the smear.
- **Hue wrap** — values near hue 1.0 alias between near and far; guard `h ≤ 0.93`.
- **Sheer-angle triangles** — the capture's own renderer discards grazing triangles
  (`|n_z| < 0.2`); the mesher must too, or silhouettes grow stretched ribbons.

## Pipeline of record (the v10 recipe)

Each stage below was validated in isolation on a 300-frame proof clip before joining the
recipe. Sizes are for that clip at vp9 2048² crf28, gop 30.

1. **SAM person mask, black background.** SAM 3 (`/sam/segment_text`, local service)
   isolates the subject; everything else is painted black and the masks are saved for
   reuse. **Law: black-mask before ANY depth estimation** — a video depth model given the
   full frame spends its relative range on the room and leaves the person in a sliver of
   values that metric fitting then amplifies into meter-scale noise.
2. **Photoreal video upscale (2×).** SeedVR2-3B (video-native diffusion upscaler) on a
   rented GPU, 15-frame GOP-aligned chunks. This buys silhouette resolution and texture
   quality that the 848-wide source lacks.
3. **Video-consistent depth estimate.** Video-Depth-Anything-Large over the masked
   frames: smooth, temporally consistent, upscaled-resolution silhouettes — but relative,
   not metric, and low on relief.
4. **Metric fusion — tiled locally-affine fit.** Per frame, the estimate is fitted to the
   decoded sensor depth (the sensor's one irreplaceable contribution is METRIC SCALE; its
   high frequency is known-garbage). One global robust fit (2-pass MAD; linear and inverse
   models compete per frame) underfits — protrusions flatten. The fix is a grid of 72 px
   tiles, each robust-fitted on its 3×3-neighbourhood samples, parameters blended toward
   the global fit by sample count, smoothed, and bilinearly interpolated per pixel
   (parameters are linear in the prediction, so blending parameters = blending
   predictions; no seams). Fit residual halved: p50 47 mm → 23.7 mm.
5. **Shading detail (normals → displacement).** Per-frame normal maps (MoGe-2) estimated
   from the photoreal frames carry the relief depth estimators smooth away — nose, brows,
   cloth folds. Integration is a screened Poisson solve on the **log-depth residual**:
   `min Σ|∇r − (g_normal − ∇log z_base)|² + λΣ|r|²`, `z' = z·exp(clamp(r))`. The screening
   term kills the normals' low frequencies, so metric shape stays sensor-anchored by
   construction and only detail rides on top (recovered field ≈ 11 mm RMS). Gradient
   constraints are dropped across depth discontinuities (> 25 mm) so detail never bleeds
   across occlusion boundaries; a temporal median-3 suppresses per-frame normal noise.
   The normal-map sign convention is verified from data each run (gradient agreement
   vote), never assumed.
6. **Full-body template for the unseen side.** SAM-Body4D (video wrapper over SAM 3D
   Body) produces a fixed-topology parametric body mesh per frame (18.4 k vertices,
   identity-consistent across the clip). It is anchored to the fused shell per frame:
   coordinate convention auto-detected from data, scaled by height, front surface
   median-matched (residual p50 23.4 mm — independently corroborating the fusion fit).
7. **Hybrid assembly.** Everything the camera sees = the detail shell **verbatim**
   (pixel-exact texture by construction — face, hair, draped strands). The template
   survives only where unseen: triangles covered by the shell are carved away (~64 % per
   frame), front-facing triangles outside the mask (silhouette overshoot) are dropped,
   and remaining vertices are pushed behind the shell so the template can never occlude
   it. Where the shell has no template behind it (hair mass, loose cloth), a backing
   surface bulges up to 120 mm and tapers to zero at the rim, wearing the continuing
   front pixels — a hair volume rather than a bald scalp.

Known open items: the template back's texture is still projected front pixels (a
"smeared" placeholder — a watertight per-take asset, e.g. PSHuman, is the planned
upgrade); the shell/template seam is visible on close inspection; head-pose tracking of
the template is coarse.

## Why projection-texturing a template alone fails

An intermediate build textured the full template body by camera projection. Result:
facial features slide (generic template face ≠ the subject's face, even perfectly
aligned), silhouette overshoot paints background onto shoulders, and hair projects onto
the chest. The lesson generalizes: **likeness demands that visible surfaces come from the
capture itself**; parametric templates are for completion, not replacement.

## Field notes (expensive lessons, kept so they are paid once)

- An empty person-mask on one frame produced a degenerate (0,0,0,0) detection box, which
  birthed NaN in the body model's camera-ray encoding and — through cross-frame shape
  sharing — poisoned **every** frame in its 64-frame batch. Proof was an A/B batch run;
  fix is backfilling empty masks from the nearest non-empty neighbour. Symptom signature:
  meshes whose PLY headers read `element vertex 0` (trimesh silently drops all-NaN
  vertices but keeps faces).
- Diagnose before touching the environment: a suspected package-version conflict was
  disproven by a single-frame probe in the unchanged environment. Never downgrade on a
  hypothesis.
- Rented-GPU containers usually ship no vendor EGL; anything that instantiates an
  offscreen GL renderer dies. Mesh export rarely needs GL — stub the visualization.
- Diffusion checkpoints are often fp32 on disk: cast to bf16 after load or eat a 43 GB
  OOM. Video diffusion VAEs can leak causal state across videos in-process — run one
  process per chunk.
- `| tail` on a multi-process training/inference launcher swallows child tracebacks;
  capture the full log to a file and grep it.

## Costs

The entire cloud footprint for developing the pipeline (upscale, depth, body model,
normals, all failure rounds included) was ≈ $6.7 across three rented-GPU sessions.
Everything else runs locally: fusion, Poisson integration, hybrid assembly, and encoding
are Node scripts that process a 300-frame clip in seconds to a few minutes each.

## What the encoder sees

Each stage emits per-frame `mesh-f#####.obj` + `atlas-f#####.png` and encodes with:

```
node packages/encoder/dist/cli.js encode <frames-dir> -o out.ares \
     --fps 30 --gop 30 --texture-codec vp9 --tex-size 2048 --crf 28 \
     --no-temporal --repack-detect image --meta-extra-file <provenance.json>
```

The fixed-topology template track points at the long-term prize: a mesh sequence with one
topology and one UV layout per take inter-codes its texture and enables temporal geometry
coding — the "dream input" that removes the per-frame-repack tax entirely.
