# ARES benchmark harness (spec §13)

Every performance claim is a hypothesis until measured here. Report ARES numbers **next to**
Draco-GLB and meshopt-GLB baselines from the same source, on the same device.

## Run it

- **Windows, 1-click:** **`ARES.vbs bench`** (repo root; `tools\launch-console.cmd bench` to
  watch it). Runs silently (~1–2 min), then opens the report page. Log: `ares/tools/launch.log`.
- **Any OS:** `cd ares && npm run build && npm run bench`
  (flags: `--frames N --bits 14 --reps 3 --clips talk,two --skip-draco --skip-sweep`)
- **Report:** `/bench/report/` on the dev server (`npm start`) — Pareto scatter, quantization
  sweep, decoder footprints, full table. Raw data: `results/intra-latest.json` + `.csv`.

## Corpus (spec §13.2)

The harness generates a **deterministic synthetic corpus** matching the spec taxonomy, so it
runs with zero downloads. Real captures matter more (§13.6): drop PLY sequences into
`data/<clip>/*.ply` (git-ignored) and they join every run automatically — ascii and
binary_little_endian are supported by the `@ares/encoder` importer.

| Clip | Content | Class (§6.4) | Why |
|---|---|---|---|
| `talk` | 41k verts, static topology, localized motion | A | best case for persistent topology |
| `dance` | 41k verts, fast large deformation | A | stresses temporal prediction / re-keyframing |
| `two` | two bodies, entry/exit, varying topology | C | stresses topology patches / GOP cuts |
| `object` | 12k verts, rigid rotation | A | texture-dominated; ABR ladder |
| `splat` | *(pending P4)* | — | splat vs mesh profile |

## Phase 0 intra results — measured 2026-07-08 (synthetic corpus)

AMD Ryzen 7 6800H, Node 24 (V8 ≈ Chrome), single thread, 14-bit positions, 30 frames/clip.
`talk` clip (40,962 verts / 81,920 tris):

| Intra codec | KB/frame | % of raw | encode ms | decode ms | note |
|---|---|---|---|---|---|
| raw f32 | 1440 | 100% | ~0 | ~0 | reference |
| raw f32 + Brotli | 706 | 49% | 36.1 | 6.6 | |
| quantized binary (Opt C/D) | 1200 | 83% | 1.6 | 0.3 | GPU-side dequant |
| qbin + Brotli | 494 | 34% | 33.1 | 5.3 | |
| **meshopt** | **210** | **14.6%** | 13.6 | **1.1** | **ARES default (§6.7) ✓** |
| **meshopt + Brotli** | **92** | **6.4%** | 18.7 | **2.1** | already in §13.5's 5–15% band, intra-only |
| Draco (edgebreaker, cl7) | 41 | 2.9% | 22.6 | 5.5 | optional high-ratio profile |

Decoder footprint shipped to clients: **meshopt ≈ 29 KB** (ESM module, wasm embedded) vs
**Draco ≈ 279 KB** (wasm alone).

**P0 exit reading (§14):** the §6.7 hypothesis holds as measured — Draco is ~2.2× smaller than
meshopt+Brotli but ~3–5× slower to decode and ~10× heavier to ship. meshopt is confirmed as the
mesh-profile intra default; Draco stays an optional profile for size-critical, latency-tolerant
captures. Numbers are geometry-only, intra-only, synthetic-corpus — temporal coding (P2) is where
the 5–15% full-pipeline target gets fought for. Caveats per §13.6: Draco error is reported as the
analytic quantization bound (edgebreaker reorders vertices); timings are Node/V8 single-thread,
not yet in-browser workers.

## Metrics (spec §13.3)

size/s · time-to-first-frame · main-thread CPU/frame · decode/frame · GPU upload/frame ·
steady FPS · seek latency · peak memory · geometry error (Chamfer/Hausdorff) · visual (PSNR/SSIM/LPIPS).
*(The intra bench covers size, encode/decode CPU, and quantization error; the rest arrive with P1's
runtime.)*

## Phase 0 ablations (spec §13.4)

- [x] intra codec: meshopt vs Draco vs raw (size × decode CPU) — **meshopt default confirmed**
- [x] quantization bits sweep 11→16 vs geometry error — RMS error halves per bit; qbin size is
      flat (u16 storage), only entropy-coded sizes track bits → aggressive tiers need entropy coding
- [ ] GOP length sweep 15/30/60/120 vs size × seek *(needs P2 temporal coding)*
- [ ] texture: AV1 vs VP9 vs KTX2 vs WebP-seq *(needs P1 texture path)*
- [x] confirm A1/A2 across the device matrix (`apps/phase0-probe`) — reference desktop: WebGPU ✓,
      HW decode all four codecs ✓, crossOriginIsolated ✓ (via `tools/serve.mjs` COOP/COEP)
