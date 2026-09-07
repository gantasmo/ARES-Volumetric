# Targeted temporal: motion-gated coherent spans for per-frame-reconstructed captures

Status 2026-07-17. This is the shipped alternative to classic temporal geometry coding
(spec P2) for captures whose topology resets every frame. Validated across eight
iterations on the 272-frame reference capture; the current keeper is the v11 recipe.

## The problem

Per-frame photogrammetric reconstruction gives every frame its own topology and its own
atlas layout. Two consequences: the texture must be coded all-intra (a repack every frame
defeats inter-prediction — measured ~3× the bytes per frame of a stable-layout span), and
the surface "boils" (~1 mm/frame at rest on the reference capture) in a way no bitrate,
resolution, or mild smoothing can fix — all were tried and adjudicated visually.

Two hard constraints shaped the design:

- A stable atlas LAYOUT is not enough: relaying raw geometry onto transferred UVs
  produced chart-clamped corners sampling atlas background (black triangle bites) — the
  stable-layout atlas must be produced by the bake itself, never by UV transfer.
- Full-clip coherence is not safe: forcing one template across high-motion frames spikes
  geometry error (measured 73.6 mm max; percentile metrics hid it — gate on MAX, not p95).

## The architecture

1. **Measure motion per run** (mean consecutive-surface displacement). Runs at or below
   ~2.2 mm/frame mean are "calm"; the rest are "motion".
2. **Calm runs → coherent spans.** One template mesh registered across the span
   (registration + atlas rebake), so topology and layout hold for the whole span and the
   texture inter-codes (−28 % texture on the reference clip at identical settings).
   Adjacent calm runs chain across their boundary by template hand-off; a 62-frame span
   with negative drift growth was measured this way.
3. **Motion runs → raw frames verbatim.** The owner-approved per-frame geometry plays
   untouched where it moves fast enough that boil reads as motion.
4. **Switches sit on GOP boundaries** with a 6-frame (3+3) transition straddling each
   cut: the coherent side converges onto the raw surface over its last three frames
   (gated — see below), the raw side is untouched.

## Registration: match the method to the span

The single most important lesson: **nearest-point pull + smoothing (the "A recipe")
collapses protruding features under motion or accumulated chain drift** — noses flatten.
It is scoped to STATIC spans only. Moving or long spans use ARAP-regularized registration
(as-rigid-as-possible, λ=10, correspondence gate 40 mm, 10 outer iterations), which holds
features at the price of slightly more within-span boil. Per-span choice, matched to
content, is the settled architecture:

| Span character | Method | Why |
|---|---|---|
| Static (subject at rest) | nearest-pull + Taubin (A recipe) | smoothest result, no drift to exploit its weakness |
| Moving, or > ~40 frames | ARAP registration | rigidity preserves protrusions under drift |

## Gated boundary transitions

Ungated convergence ramps caused "melting" where surfaces come close (hands, contact
regions): vertices snapped to the wrong nearby surface. The shipped ramp is gated three
ways — pull distance ≤ 10 mm, vertex-normal · target-face-normal > 0.3 (surface
agreement), and the displacement field is weld-aware smoothed before application. With
the gates, contact-region vertices are excluded automatically and the applied field
averages ≈ 0.1 mm — transitions soften without wrong-surface fusion.

## Instrumentation humility

Every distance metric in the toolchain has a ~0.5–1 mm floor from re-triangulation
discretization, and registration error metrics read ~0 for a protrusion resting on a real
surface — the flattened-nose failure was invisible to every number and caught only
visually. Metrics here are cause-finders, not verdicts; visual evaluation gates every
recipe change.

## Reusable pieces

- `tools/coherent/` — registration + rebake pre-pass (per-span windows via environment
  variables; partial re-bake of chosen frames supported).
- Encoder `--repack-detect image` — image-based atlas-repack detection (64² grayscale
  MAD, threshold 12; measured separation ~4 stable vs ~37 repack). Required for any
  stable-layout content: the topology-hash heuristic false-positives every frame there.
- The boundary-ramp gating pattern (distance cap + normal agreement + weld-aware
  smoothing) generalizes to any "converge surface A onto surface B" step, and is reused
  by the RGBD hybrid assembly (see `docs/rgbd-rebuild-pipeline.md`).
