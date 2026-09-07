# Gaussian splat integration notes

**Status: field notes, not a decision.** Filed under the `DECISIONS.md` §2 convention — this is evidence
gathered on another project, contestable, and scoped to exactly what was tried.

Written 2026-08-22 while building an SPZ→mesh importer for the *Lord Lyrical: The Resonant Realm* VR piece
(Unity 6.5, standalone Quest target). ARES has the splat profile **specified but unimplemented**
(`spec/06-geometry.md` §6.8, `GeometryProfile.SplatIPB = 1`, `AttrMask.Color` declared but never read), so
everything below is offered as input to whoever picks that up.

---

## 1. SPZ container format — validated byte-exact

Niantic's SPZ is what Scaniverse and **World Labs' Marble** both emit, so it is the realistic ingest format
for generated environments. It is straightforward enough that an ARES importer is a day's work, and it
slots directly into the `importers/` pattern next to `ply.ts`.

The whole file is a **gzip stream**. Inflate it and you get a 16-byte header followed by tightly packed,
**per-attribute arrays — not interleaved per point**, which is convenient: each attribute is a contiguous
run you can `subarray()` without striding.

```
header    : magic u32 = 0x5053474E ("NGSP") | version u32 | numPoints u32
            | shDegree u8 | fractionalBits u8 | flags u8 | reserved u8
positions : numPoints * 3 * int24   little-endian signed; metres = v / (1 << fractionalBits)
alphas    : numPoints * u8          alpha = a / 255
colors    : numPoints * 3 * u8      sh0 = (c/255 - 0.5) / 0.15
                                    rgb = 0.5 + 0.28209479177387814 * sh0
scales    : numPoints * 3 * u8      metres = exp(s / 16 - 10)
rotations : numPoints * 3 * u8      xyz = (r - 127.5)/127.5 ; w = sqrt(max(0, 1 - x²-y²-z²))
sh        : numPoints * shDim * u8  shDim = 0 / 9 / 24 / 45 for degree 0/1/2/3
```

`flags` bit 0 = antialiased. Reject `division`-style surprises by checking the magic first; a `.ply` from a
trainer is the common mis-feed.

**Verified** against a 27.8 MB Marble export: v2, 1,920,000 splats, shDegree 0, fractionalBits 10 →
predicted 36,480,016 bytes, actual 36,480,016, **zero remainder**. Worth keeping that assertion in the
importer — it is the cheapest possible corruption gate, and it is exactly the "validate against
independently reported ground truth" discipline `SVFFrameExporter.cs` already applies.

### Direct read on ARES's quantization

SPZ's position encoding is *the same idea* as `quant.ts`, arrived at independently:

| | ARES | SPZ |
|---|---|---|
| Position | 14-bit normalized to per-GOP AABB | int24 fixed-point, `1 << fractionalBits` divisor |
| Storage | u16 × 4 (padded to 2×u32) | 9 bytes/point, unaligned |
| Range | implicit in AABB | implicit in fractional bits (±8192 m at 10 bits) |

SPZ is **absolute**, not AABB-relative, so it does not adapt to content scale — a room-scale capture wastes
most of the int24 range. If ARES ever re-encodes splats, the existing per-GOP AABB scheme is strictly
better at the same bit budget. The bench table (`bench/results/intra-latest.json`, error halving exactly
per bit) applies unchanged: splat *positions* are just vertices.

**But note the asymmetry**: splat count is 10–100× typical mesh vertex count for the same scene (1.92 M for
one room), so the 9-bytes-unaligned choice matters more than it does for meshes. The `u16 × 4` padding
trick that makes ARES vertices 2×u32 costs 8 bytes/point here — worth re-deriving rather than inheriting.

---

## 2. The measured shape of a real capture

From the Marble chamber (16.4 × 15.9 × 26.2 m room), useful for sizing any buffer or codec work:

| Property | Value |
|---|---|
| Splats | 1,920,000 |
| SH degree | **0** — no view-dependent colour at all |
| Median splat scale | 1.3 cm (p99 = 10.5 cm) |
| Median alpha | 0.988 — the vast majority are effectively opaque |
| Median flatness (min/max axis) | **≈ 0.00–0.01** |
| Raw AABB vs. room AABB | 73 × 71 × 48 m vs 16 × 16 × 26 m |

Three consequences that surprised me and are worth carrying into the ARES splat profile:

1. **shDegree 0 is the common case, not the cheap fallback.** The §6.8 note that "most web captures use
   degree 0–1" is if anything understated for *generated* environments. Whatever attribute-stream design
   gets built, degree 0 should be the fast path, not a degenerate case of the degree-3 layout.

2. **Splats are discs, not blobs.** Flatness ≈ 0 means the covariance ellipsoid is essentially 2D, so the
   shortest ellipsoid axis is a usable **surface normal** — free, no estimation pass. That makes
   oct16 normal encoding (`geometry-encode.ts:71-86`) immediately applicable, and it is what makes
   splat→mesh viable at all.

3. **Generated captures carry enormous outlier halos.** The raw AABB is 4.5× the real room in X. Any
   per-GOP AABB quantization computed on raw input would waste ~2 bits of precision on empty space.
   **Filter before you fit the box.** Alpha > 0.3 plus a max-extent cut removed 15% of points and
   collapsed the AABB to the actual room.

---

## 3. §8.5.1 applies with full force — read it before packing splats into video

`spec/08-compression.md` §8.5.1 ("Why naïve XYZ-in-RGB fails") is the single most valuable page in the spec
for this work, and §6.8's "video-packed attributes: PackUV-style 2D tiles" is exactly the idea it warns
about. For splats specifically:

- **Positions**: same four failure modes as meshes, unchanged. 8-bit is not enough, 4:2:0 destroys two of
  three coordinates, DCT ringing becomes spatial jitter.
- **Rotations**: *worse* than positions. A quaternion is not spatially coherent across neighbouring splats,
  so the DCT has nothing to exploit and every artefact becomes a visible orientation flip. If anything is
  packed into video, rotations should be the last candidate.
- **Colour/SH**: the *only* attribute that is genuinely video-shaped. It is already perceptual, already
  8-bit-tolerant, and spatially coherent if splats are ordered by locality.

§6.8's own line — *"splats tolerate this better than meshes because there is no connectivity to corrupt"* —
is true for **colour** and misleading for **geometry**: a jittered vertex in a mesh is anchored by its
neighbours through the index buffer, whereas a jittered splat is anchored by nothing and simply flies off.
I would soften that claim in the spec.

**Suggested split**, following §8.5.2: colour/SH → video texture; position/scale/rotation/opacity →
meshopt-style entropy-coded intra streams, per-GOP AABB, with position deltas for P-frames.

---

## 4. Splat birth/death is the real temporal problem

§6.8 notes it in passing; having thought about it, it deserves to be the centre of the temporal design.

Mesh volcap has the opposite difficulty: connectivity is hard, correspondence is hard, but the *vertex
count is stable within a run*. Splats invert this — no connectivity to maintain (easy), but trainers
**add and prune splats every optimisation step**, so the point set is unstable frame to frame in a way
`tools/coherent/`'s ARAP template registration has no analogue for.

The nearest existing machinery is the `TriangleGrid` + `closestOnTriangle` correspondence search
(`temporal.ts:110-175`), but note the documented cell-size failure: `bboxDiag/48` is tuned for ~11 k
queries/frame and degenerates at 2.5 M. **A splat correspondence pass is 1.9 M queries/frame — re-tune or
use a BVH from the start.** That warning cost real time on the atlas baker; it will cost more here.

A plausible shape, offered as a starting point rather than a recommendation:
- I-frame: full splat set, sorted by Morton code for locality.
- P-frame: per-splat deltas for surviving splats + an explicit birth list + a death bitmap.
- Sort by Morton code so both the delta stream and any video-packed colour tile are spatially coherent.

---

## 5. What we actually shipped, and why it may matter to ARES

For the VR piece the conclusion was **do not render splats at runtime** — bake to mesh. The reasoning is
mostly application-specific (the show needs surface-emitted VFX, floor state changes and dynamic light,
none of which a splat cloud can host), but two parts generalise:

- **World Labs' own Unity guidance**: 2 M-splat files *crash* Quest 3 builds; 500 k is "more suitable" and
  still only reaches **~12 fps**. Any ARES splat profile targeting standalone XR needs a decimation story,
  not just a compression story.
- **At shDegree 0 a baked mesh is visually near-identical**, because there is no view-dependent term to
  lose. The usual "splats look better" intuition does not hold for generated environments.

The surfacing path, in case it is useful: alpha-weighted trilinear deposit onto a sparse voxel grid →
smoothing → **Naive Surface Nets** (chosen over Marching Cubes for uniform topology that survives
decimation and triplanar projection) → QEM decimation with **locked boundaries**.

Measured on the chamber, single-threaded C#: 1.92 M splats decoded in **1.33 s**; filtered and downsampled
to 330 k in 3.0 s; surfaced to 205 k tris in **3.9 s**; decimated to 80 k tris with a **volume ratio of
0.999**.

That last number is the ARES technique paying itself back immediately — I used `arap-poc.mjs`'s signed
**volume ratio** as the integrity gate rather than a surface-distance metric, on the strength of the
`DECISIONS.md` §2 finding that nearest-surface error is *blind*. It caught nothing this run, which is the
point: it is a smoke alarm, and a cheap one at ~15 lines.

---

## 6. Concrete suggestions for ARES

1. **Add an SPZ importer** next to `ply.ts`. The format is above; `propOffsets`-style stride plumbing maps
   over directly, and unlike the 3DGS PLY variant SPZ has a fixed layout with nothing to negotiate. This
   would let ARES ingest anything Scaniverse or Marble produces.
2. **Extend the PLY importer to read attributes at all.** It currently reads positions and indices only —
   no colour, opacity or normals (`PlyMesh` is `{positions, indices, vertexCount, faceCount}`). A 3DGS PLY
   is the other half of the splat ingest story and needs `f_dc_*`, `opacity`, `scale_*`, `rot_*`.
3. **Fix the §6.8 claim** that splats tolerate lossy packing better — true for colour, false for geometry.
4. **Make shDegree 0 the fast path.**
5. **Filter before fitting the quantization AABB**, or generated captures will waste ~2 bits on outlier haze.
6. If a splat renderer lands: the per-frame depth sort in §6.8 is the cost driver on tiled mobile GPUs, not
   the splat count. That is the number to bench first.

---

## Cross-references

- Format details verified against `DEV_Do_Not_Ship/Dark, Gothic, Cavernous Stone Chamber.spz`
  (not in the ARES repo).
- Working C# implementation:
  `STYLY-NetSync-Unity/Assets/LordLyrical/SplatKit/` in the `lord-lyrical-resonant-realm` repo —
  `SpzReader.cs` (format), `SplatCloud.cs` (filters), `SplatSurfacer.cs` (Surface Nets),
  `MeshDecimator.cs` (QEM + border lock), `MeshIntegrity.cs` (volume ratio, ported from `arap-poc.mjs`).
- ARES pages leaned on here: `spec/06-geometry.md` §6.8/§6.10, `spec/08-compression.md` §8.5,
  `packages/core/src/quant.ts`, `packages/encoder/src/importers/ply.ts`,
  `packages/encoder/src/decimate.ts`, `tools/coherent/arap-poc.mjs`, `DECISIONS.md` §2.
