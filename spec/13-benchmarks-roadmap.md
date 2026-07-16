## 13. Benchmark methodology and projected performance

Every performance claim in this document is a **hypothesis** until this methodology confirms it. The
point of writing the methodology first is to make the targets *falsifiable* and to prevent
cherry-picking.

### 13.1 Principles

- **Fixed corpus.** A small, public, representative corpus (below), versioned, so runs are
  comparable over time.
- **Baselines are non-negotiable.** Every ARES number is reported *next to* Draco-GLB and meshopt-GLB
  sequences produced from the *same* source, on the *same* device.
- **Report the Pareto front, not a single number.** Size, quality, CPU, and latency trade off; a win
  on size that loses on CPU must be shown as such.
- **Quality is measured, not asserted.** Geometry error (Chamfer distance, Hausdorff) and appearance
  (PSNR/SSIM/LPIPS on rendered frames vs source) both reported.

### 13.2 Corpus

| Clip | Content | Frames | Why |
|---|---|---|---|
| `talk` | Single person talking, static camera | 300 | Best case for persistent topology |
| `dance` | Full-body fast motion | 300 | Stresses temporal prediction / re-keyframing |
| `two` | Two people, occlusion, entry/exit | 300 | Stresses topology patches / GOP cuts |
| `object` | Rotating textured object | 150 | Texture-dominated; ABR ladder |
| `splat` | Photoreal capture (splat-friendly) | 150 | Splat profile vs mesh profile |

Sources include PLY+PNG pairs and at least one Depthkit and one 4DViews export, to exercise importers.

### 13.3 Metrics and how each is measured

| Metric | Instrument | Target (§4.5) |
|---|---|---|
| Download size / s | encoder output bytes ÷ duration | 1.5–4 MB/s |
| Time-to-first-frame | `performance.now()` from `create()` to first presented frame | < 500 ms |
| Main-thread CPU / frame | `performance.measure` on the render-loop task; long-task API | < 3 ms |
| Decode time / frame | Worker-side timing, geometry; `VideoFrame` cadence, texture | budgeted |
| GPU upload / frame | timestamp queries around upload | < 2 ms |
| Steady FPS | rAF delta histogram | ≥ capture fps (target 60) |
| Seek latency | time from `seek()` to presented target frame | < 250 ms |
| Peak memory | `performance.memory` (where available) + GPU allocation tracking | bounded |
| Power (mobile) | platform battery/energy where available; else CPU%+GPU% proxy | Low |
| Geometry error | Chamfer / Hausdorff vs source mesh | below per-clip threshold |
| Visual quality | PSNR / SSIM / LPIPS on rendered frames | below per-clip threshold |

### 13.4 Ablations (Phase 6 in the plan)

Each mechanism must justify its complexity by an ablation that turns it off:

- Quantization bits sweep (11 → 16) vs geometry error.
- meshopt vs Draco vs raw intra (size × decode CPU).
- Temporal prediction on/off; predictor order 0 vs 1.
- GOP length sweep (15/30/60/120) vs size × seek latency.
- Texture: AV1 vs VP9 vs KTX2 vs WebP-seq (size × quality × decode).
- Video-geometry profile vs binary-delta profile (the §8.5 experiment).
- Splat vs mesh profile on `splat` and `talk`.

#### 13.4.1 Phase 0 intra results — **[MEASURED 2026-07-08]**

First measured fill of the intra rows (harness `ares/bench`, report at `/bench/report/`).
Synthetic corpus per §13.2 taxonomy, geometry-only, intra-only; AMD Ryzen 7 6800H, Node 24
(V8), single thread, 14-bit positions. `talk` clip, 40,962 verts / 81,920 tris:

| Intra codec | KB/frame | % of raw | decode ms/frame | decoder shipped |
|---|---|---|---|---|
| raw f32 | 1,440 | 100% | ~0 | — |
| quantized binary (Opt C/D) | 1,200 | 83% | 0.3 | — |
| qbin + Brotli | 494 | 34% | 5.3 | — |
| **meshopt** | **210** | **14.6%** | **1.1** | **≈29 KB** |
| **meshopt + Brotli (q5)** | **92** | **6.4%** | **2.1** | **≈29 KB** |
| Draco (edgebreaker, cl7) | 41 | 2.9% | 5.5 | ≈279 KB wasm |

Quantization sweep 11→16 bits: RMS error halves per added bit (1.4×10⁻⁴ → 4.5×10⁻⁶ of bbox
diagonal); qbin's size is bit-independent (u16 storage), so aggressive tiers save bytes only
through the entropy-coded codecs — consistent with §8.1's expected ordering.

**§6.7 confirmed as measured:** Draco ≈2.2× smaller than meshopt+Brotli but 3–5× slower to
decode and ~10× heavier to ship → **meshopt stays the mesh intra default**, Draco remains the
optional high-ratio profile. meshopt+Brotli's 6.4% already sits inside §13.5's projected 5–15%
band *before* temporal coding — the P0 exit criterion (§14) is met. Caveats (§13.6): synthetic
stand-ins pending real captures in `bench/data/`; Draco geometry error reported as the analytic
quantization bound (edgebreaker reorders vertices); timings are Node, not yet in-browser workers.

### 13.5 Projected performance table

Normalized to raw PLY+PNG = 100%. **All ARES figures are [PROJECTED] pending §13.1–13.4.** The
non-ARES rows carry the vendor/research claims from [§3](#3-survey-of-existing-formats).

| Format | Rel. size | CPU decode | TTFF | Seek | Notes |
|---|---|---|---|---|---|
| Raw PLY + PNG | 100% | Very high | Very slow | — | Reference |
| Draco-GLB seq | 15–35% | High | 1.5–4 s | Slow | Common baseline |
| Meshopt-GLB seq | 20–40% | Low–med | 1–3 s | Slow | Faster decode |
| UVOL | 15–35% | Med–high | Med | Asset | Draco+KTX2 |
| Microsoft HoloVideo | 20–40% | Med | Med | Video | Proprietary |
| Arcturus AVV | ~25% | Med | Med | Yes | Proprietary |
| **ARES mesh (target)** | **5–15%** | **Low** | **< 0.5 s** | **< 0.25 s** | **[PROJECTED]** |
| **ARES splat (target)** | **8–18%** | **Low** | **< 0.5 s** | **< 0.25 s** | **[PROJECTED]** |

### 13.6 Honesty clause

If, after Phase 2, the video-geometry profile does not beat the binary-delta profile, it is dropped —
not shipped for novelty. If persistent-topology tracking proves impractical on the `dance`/`two`
clips, the fallback is per-GOP re-keyframing with meshopt intra, and the size targets are revised
upward accordingly. Projections are commitments to *measure*, not to *hit*.
