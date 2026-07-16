# Size comparison — Daniel Microsoft volcap (272 frames, 9.1 s)

**Every number here is measured on the real capture** (`Daniel_Microsoft_Volcap/Daniel_Volcap`),
with one labelled exception (4DViews). Nothing is derived from how ARES compresses — the Draco
figures come from encoding the actual frames with the real `draco3d` encoder, and the raw figures
are the source folder on disk.

Reproduce all of it:

```
node tools/measure-baselines.cjs
# options: [captureDir] [ares file] --sample N --mbps M
```

## The capture

| | value |
|---|---|
| Frames | 272 (`mesh-fNNNNN.obj` + `atlas-fNNNNN.png`) |
| Duration | 9.1 s @ 30 fps |
| Mesh | ~11,200 verts, 20,000 tris/frame, **independent topology per frame** (re-meshed) |
| Texture | per-frame re-atlased PNG (the atlas layout changes every frame) |

The independent per-frame topology + re-atlasing is *why* intra delivery is the faithful path for
this capture and temporal tracking (P2) is not.

## Delivered size — same capture, four formats

| Format | Geometry | Texture | **Total** | Files | Source |
|---|--:|--:|--:|--:|---|
| Raw OBJ + PNG | 485.7 MB | 1.11 GB | **1.58 GB** | 544 | measured |
| Draco-GLB sequence | **20.3 MB** | 1.11 GB¹ | **1.13 GB** | 272 | measured (real `draco3d`) |
| 4DViews `.4ds` | — | — | **2.3 → 135 MB** | 1 | 720p est · **HR measured**² |
| **ARES** | 52.4 MB | **14.7 MB** | **67.1 MB** | 1 | measured (from the container) |

¹ A per-frame glTF/Draco sequence ships per-frame images; Draco compresses **geometry only**, so its
texture column is the capture's own PNG atlases (measured). A JPEG/KTX2 texture would shrink this
column, but that's a texture-codec choice, not Draco.
² 4DViews `.4ds` is proprietary and can't be produced here, and it's strongly **tier-dependent** —
three reference points, the top one now **measured from a real `.4ds`**:

| 4DViews tier | Bitrate | This 9 s clip | Fidelity vs ARES | Source |
|---|--:|--:|---|---|
| 720p streaming | ~2 Mbps | ~2.3 MB | lower | est. (anecdotal, unverified) |
| Standard | ~16 Mbps | ~18 MB | near | est. (120 MB/60 s, published) |
| **DESKTOP_HR** | **125 Mbps** | **~135 MB** | **≥ ARES** | **measured** (238,222,244 B / 15.2 s @ 29.99 fps) |

The measured HR data rate comes from a real `.4ds` **by file size ÷ duration only** — no content read.
It's a *different capture*, so normalization to this clip is cross-content (±), but it's the format's
real high-fidelity data rate. **At that fidelity — the tier closest to ARES's 1024² + full
mesh — ARES (67 MB) measures ~2× smaller than 4DViews (135 MB), cross-content normalized (±).** 4DViews only undercuts ARES by
dropping to lower-fidelity streaming tiers.

## Encode revisions (2026-07-10)

The 67.1 MB figure above is the first full encode (`daniel.ares`); the ratios and 4DViews
comparisons in this document keep it as their reference point. Three measured
optimizations landed after it (details in [whitepaper.md](whitepaper.md) section 10):

| Encode | Change | Geometry | Texture | Total (file) |
|---|---|--:|--:|--:|
| `daniel.ares` (v1) | baseline (VP9, i8 normals) | 52.4 MB | 14.7 MB | 67.1 MB |
| `daniel-v2.ares` | lossless meshopt vertex reorder | 34.0 MB | 14.7 MB | 49.0 MB |
| `daniel-v4.ares` | oct16 normals (+2.9 MB, fidelity-up) + AV1 texture | 37.1 MB | 12.5 MB | 49.6 MB |
| `daniel-s0.ares` (keeper) | oct16 + AV1, smoothing off | 37.3 MB | 12.5 MB | 49.7 MB |

Totals are delivered file sizes per the encoder logs; v2's geometry+texture payload
columns sum to 48.7 MB (the recorded payload figure), with container overhead making up
the difference. Against the current keeper, the raw source is ~33x larger and Draco's
geometry advantage narrows to 20.3 vs 37.3 MB.

## Analysis

```
Geometry (9s-equiv)                Texture (9s-equiv)
  Raw OBJ     485.7 MB ██████████    Raw PNG     1135.8 MB ██████████
  ARES         52.4 MB █             4DViews HR    130 MB  █▏
  Draco        20.3 MB ▌             ARES VP9      14.7 MB ▏
  4DViews HR    5.5 MB ▏             (Draco = geometry codec, n/a)
  ← 4DViews smallest geom (temporal)   ← ARES smallest tex (video codec)

Total delivery (log scale)
  Raw OBJ+PNG   1.58 GB  ██████████████████   544 files
  Draco-GLB     1.13 GB  █████████████████    272 files
  ARES         67.1 MB  ███████               1 file
  4DViews  2.3→135 MB  ▓▓▓▓▓▓▓▓▓▓▓▓          1 file (720p est → HR measured; band spans ARES)
```

**Headline ratios (measured):**

- ARES total is **24.2× smaller than raw** and **17.2× smaller than a naive Draco-GLB sequence**.
- **Draco wins geometry, 2.6×** (20.3 vs 52.4 MB). ARES stores topology intra every frame — the known
  cost of not doing temporal geometry on this capture.
- **ARES wins texture, 77×** (VP9 14.7 MB vs PNG 1.11 GB). This is where ARES's delivery win comes from.
- **At comparable fidelity, ARES measures ~2× smaller than 4DViews (cross-content normalized).** 4DViews DESKTOP_HR is ~135 MB for this
  clip (real `.4ds`, 125 Mbps) vs ARES's 67 MB. 4DViews is only *smaller* (down to ~2.3 MB) at its
  720p streaming tier, i.e. by lowering fidelity below ARES. Their temporal codec is still the codec
  to learn from (it's why the low tier can be so small), but ARES is not behind at comparable quality.

### Assessment

ARES's delivery advantage over Draco-GLB is **texture video + single-file**, *not* geometry — on
geometry alone Draco is smaller. The obvious optimization target is therefore ARES geometry (the
52 MB intra cost), which is exactly what temporal P2 attacks; it just doesn't suit this re-atlased
capture (topology changes every frame with no correspondence), so it stays benched for this data.

**ARES and 4DViews are mirror images — and the ideal is the union of both** (byte-proven from the
`.4ds`, see next section). 4DViews DESKTOP_HR spends **4% on geometry, 96% on texture**: its mesh is
tiny (~5.5 MB/9s-equiv) because it's *temporally* compressed (30 keyframes + 425 delta frames), but
its texture is huge (~130 MB) because it stores a **per-frame** 1440² GPU-block image with *no
inter-frame* compression. ARES is the opposite: heavy intra geometry (52 MB), tiny **video** texture
(14.7 MB VP9). So each format already beats the other on exactly the axis the other neglects:

| Codec (9s-equiv) | Geometry | Texture | Approach |
|---|--:|--:|---|
| ARES | 52.4 MB | **14.7 MB** | intra mesh + VP9 **video** texture |
| Draco | 20.3 MB | — | intra mesh only |
| 4DViews HR | **5.5 MB** | 130 MB | **temporal** mesh + per-frame block texture |

- 4DViews geometry is **~9.5× smaller than ARES** and **~3.7× smaller than Draco** — hard proof that
  temporal mesh coding (ARES's P2) is the right lever; a shipping product spends only 4% there.
- ARES texture is **~9× smaller than 4DViews HR** — hard proof that a **video** texture codec beats
  per-frame block textures. 4DViews leaves this on the table.
- **The target codec = 4DViews-style temporal geometry + ARES-style video texture.** Neither ships it.

## Method / provenance

| Number | How measured |
|---|---|
| Raw geom / tex | `stat` sum of the `.obj` / `.png` files in the capture folder |
| Draco geometry | `draco3d` v1.5.7, pos 14b / uv 12b / normal 10b, edgebreaker, every frame summed |
| ARES geom / tex | walk the `.ares` chunk directory, sum payload bytes by block type (geometry vs VP9 texture) |
| 4DViews DESKTOP_HR | **measured** — real `.4ds` file **size ÷ duration** (238,222,244 B / 15.2 s = 125 Mbps), no content read; cross-content normalization to this clip |
| 4DViews 720p | estimate — ~2 Mbps streaming tier (anecdotal, unverified), no HR file at that tier to measure |

## Measuring 4DViews from real files

The HR tier is now measured from one real `.4ds` (size ÷ duration only — no content opened).
To measure another tier or tighten the range, feed the tool a file's **size** + **duration**:

```
node tools/measure-baselines.cjs --4ds-bytes <bytes> --4ds-dur <seconds> --4ds-label <tier>
```

It computes `size ÷ duration` → MB/s and normalizes to this clip. More files at more tiers = a
data-driven range instead of estimates.

## 4DViews `.4ds` container structure (structural analysis, byte-proven)

Beyond the total, the real `.4ds` was **structurally parsed** — magic, header, frame directory, and
per-block byte *lengths* only. **No mesh or texture was ever decoded, extracted, or rendered.** Run it:

```
node tools/probe-4ds.cjs <file.4ds>     # prints structure + geom/tex split; writes nothing
```

**Byte-proven (reconciles to the exact file size, residual 0 B):**

| Field | Value |
|---|---|
| Format | `4DS0` v20.30, single self-contained container (geometry **and** texture embedded) |
| Frames | 455 @ 29.999 fps (15.2 s); texture **1440×1440**, embedded per-frame |
| Geometry codec | **temporal** — 30 intra keyframes + 425 inter/P delta frames, adaptive GOP (avg 15.2 f) |
| **Geometry bytes** | **9,717,626 B (9.27 MiB, 4.08%)** — 455 small blocks (keyframes ~100 KB, P-frames ~15–21 KB) |
| **Texture bytes** | **228,496,451 B (217.9 MiB, 95.92%)** — 455 big blocks, uniform ~502 KB, **not GOP-structured** |
| Reconciliation | 9,717,626 + 228,496,451 + 8,167 overhead = 238,222,244 = file size, **exact** |

The geometry/texture *labeling* is inferred (the small-block bucket independently reproduces the
30-keyframe/425-P directory; big blocks are per-frame and uniform) — **HIGH confidence**, though not
from decoding pixels.

**What the byte statistics themselves establish about the texture track:** each frame is a single
~502 KB GPU-block image (~1.94 bpp) with **no inter-frame (video) compression** — which is exactly
why the texture is ~9× heavier than ARES's VP9.

Playback metrics (TTFF, decode/frame, FPS) are shown live in the demo HUD, measured on the running
renderer — not tabulated here because they're machine-dependent.
