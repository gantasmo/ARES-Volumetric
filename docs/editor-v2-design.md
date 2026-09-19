# Editor v2: time-ranged selection & delete for per-frame volcap

**Status:** design (2026-07-09) · **Builds on:** mesh editor v1 (6-slider crop box + `--crop` bake) · **Target content:** `daniel-s0.ares`: 272 frames @ 30 fps, ~11.2–11.4k verts (~20–22k tris) per frame, per-frame video-atlas texture.

This document specifies the v2 editing model (brush/box/SAM selection, keyframed over time, baked as triangle drops at encode), the WebGPU implementation, and the on-disk edit-list format. Every major choice cites the analogous feature in a shipping tool.

---

## 1. Constraint: no vertex correspondence

Microsoft-style volcap reconstructs **every frame independently**. Verified on this capture: vertex counts drift 11,211 → 11,397 frame to frame; topology is unrelated between frames. There is **no vertex correspondence across time**, so a "selection" stored as vertex/triangle indices is meaningless one frame later.

It is meaningless even on the *same* frame across the pipeline: the encoder's meshopt reorder (C1, `geometry-encode.ts`) renumbers vertices **and reorders triangles**, so triangle ids in the player's decoded mesh do not match triangle ids in the source OBJ the bake re-reads. An id list authored in the player would be wrong twice.

**Consequence, the core design rule:** every selection is stored as a **world-anchored REGION (volume or projected mask), keyframed on the timeline**, and is re-evaluated against whatever geometry exists at each frame. Triangle lists exist only transiently (per-frame, for preview highlight and at bake).

This is exactly how the industry tool handles it: Arcturus HoloEdit's Mesh Capture Tool removes unwanted capture geometry with **subtractive volumes**; "objects that will remove all vertices they overlap from the capture"; and "subtractive volumes can be **animated, using Unity's keyframing tools**" ([HoloEdit docs, Mesh Capture Tool](https://learn.arcturus.studio/docs/holoedit/en/v2023.2/html/External/Mesh_capture_tool.html)). An animated kill-volume evaluated per frame is the standard answer to editing geometry that has no temporal identity. Editor v2 is that idea, generalized to three region primitives (box, brush-stroke SDF, projected 2D mask) and driven from a browser timeline.

### Existing v1 foundation (reused, not replaced)

| Piece | Where | Reused as |
|---|---|---|
| Crop-box preview (fragment `discard` against a world AABB) | `packages/core/src/renderer.ts` WGSL `fs`, `player.setCrop()` | Degenerate single-volume preview; superseded by the per-triangle state buffer (§5) but kept for the simple crop path |
| Wireframe v1: line-list index expansion (3 edges → 6 indices/tri) | `renderer.ts` `setWireframe` / `uploadLineIndices`, `player.setWireframe()` | Kept as the cheap toggle; §7 compares it against the barycentric pipeline and recommends where each applies |
| `cropFrame()`: centroid-in-box triangle drop + 2-pass vertex compaction | `packages/encoder/src/crop.ts` | Generalized to `filterFrame(frame, keepTri: (t) => bool)`; the compaction pass is unchanged |
| `--crop x0,y0,z0,x1,y1,z1` CLI + `/encode` SSE bake endpoint | `packages/encoder/src/cli.ts`, `tools/serve.mjs` | `--edits <file.json>`; `--crop` becomes the degenerate edit list (§9) |
| Orbit camera (az/el/dist/target, fovY 50°, near `0.01·d`, far `20·d`, +Y up, WebGPU z∈[0,1]) | `packages/core/src/camera.ts`, `player.getCamera()` | Serialized into `mask2d` volumes; `orbitViewProj` is the projection used for all CPU-side region tests |
| Quantized positions on CPU (`curPosQ` u16 + GOP AABB, `dequantScale`) | `player.ts`, `quant.ts` | Cheap CPU dequant for x-ray (through) selection tests, no GPU readback needed on that path |
| Timeline scrubber + `onFrame` callback | `apps/demo/index.html` `#scrub`, `player.ts` | The edit-range track renders under it (§6) |

---

## 2. Research groundings

| Tool feature | What it does | What v2 takes from it |
|---|---|---|
| **HoloEdit subtractive volumes** ([Mesh Capture Tool](https://learn.arcturus.studio/docs/holoedit/en/v2023.2/html/External/Mesh_capture_tool.html)) | Kill-volumes overlap-delete capture geometry; animatable with keyframes; used for "removing props and consistent noise from a clip" | The entire storage model: time-anchored volumes, not element lists |
| **HoloEdit intervals & keyframe diamonds** ([Editing Compositions](https://arcturus.studio/docs/holoedit/en/v2021.1/html/Editing_Compositions/Editing_Compositions.html)) | A stage applies over an *interval* (time range) with draggable end tabs; **large filled diamond = user keyframe (draggable), small hollow diamond = derived keyframe (read-only)** | Range bars with end handles; the filled/hollow diamond convention for user vs. SAM-propagated keyframes (§6) |
| **HoloEdit post-removal pipeline** (same docs) | After subtractive volumes cut connected geometry: Clean Mesh → Stabilize → **Generate UVs → Texture Transfer** to rebuild a valid atlas | Warning heeded: v2 deliberately does **not** re-atlas, so no texture-transfer stage is needed (§8) |
| **4DViews 4Dfx Brushes** ([Laval Virtual interview](https://blog.laval-virtual.com/en/4dviews-more-photorealistic-volumetric-videos-than-ever-before/), [product page](https://www.4dviews.com/volumetric-software)) | Brushes "remove or add texture details … alter the meshes, or change the luminosity … **locally**"; "Interactive Mesh & Texture Editing" | Validates brush-local mesh editing as the headline volcap-editing tool. Public docs don't state per-frame vs. ranged; v2's differentiator is making every brush edit **time-ranged by construction** |
| **After Effects mask-path keyframes** ([Adobe: animating masks](https://helpx.adobe.com/after-effects/using/animating-shape-paths-masks.html)) | Keyframe the Mask Path property; AE interpolates the shape between keyframes | "Selection region interpolates between timeline keyframes" is lifted directly; AE's first-vertex correspondence problem is why v2 interpolates **fields (SDFs)**, not point lists (§6.2) |
| **AE Roto Brush** ([Adobe: Roto Brush & Refine Matte](https://helpx.adobe.com/ca/after-effects/using/roto-brush-refine-matte.html)) | Paint on a **base frame** (green box on the track), matte **propagates** frame-by-frame across a **span** (chevron pattern); corrections on later frames become new bases | The direct analog of SAM-assisted selection propagation: base-frame styling, span chevrons, correction-becomes-keyframe (§6.3, §7) |
| **Premiere Pro mask tracking** ([Adobe: track masks](https://helpx.adobe.com/premiere/desktop/add-video-effects/create-masks-and-composites/track-masks.html)) | "Track selected mask forward" analyzes motion and **writes a dense keyframe per frame** | Propagated SAM masklets are stored the same way: dense per-frame *derived* keyframes, no interpolation needed inside a propagated span |
| **Blender selection semantics** ([manual: Selecting](https://docs.blender.org/manual/en/latest/interface/selecting.html); [dev task T73479](https://developer.blender.org/T73479)) | X-Ray "impacts selection too: when enabled, selection isn't occluded by the object's geometry"; solid shading box/circle select picks **only visible** elements; wireframe shading is see-through (x-ray on by default) and selects **through**; Circle Select is a drag-paint brush | v2's mode law (§4): shaded ⇒ visible-only (depth-tested GPU ids); wireframe/x-ray ⇒ through (CPU projection, no depth) |
| **SAM 2** ([paper/site](https://ai.meta.com/research/sam3/), [repo](https://github.com/facebookresearch/sam2)) | Click/box/mask prompts on any video frame → mask; memory-attention **propagates a masklet** across frames (`propagate_in_video`); corrections refine it | The selection-assist engine; §7 |
| **SAM 2 web demo backend** ([demo/README](https://github.com/facebookresearch/sam2/blob/main/demo/README.md)) | Official localhost deployment: Flask + GraphQL backend in Docker, frontend :7262 / backend :7263 | Proof the "local SAM HTTP endpoint" is a supported, documented pattern; our dev server proxies to it (§7.1) |
| **SAM 3 / 3.1** ([Meta blog](https://ai.meta.com/blog/segment-anything-model-3/), [repo](https://github.com/facebookresearch/sam3)) | Released 2025-11-20: **text/concept prompts** ("boom mic") + exemplars, unified detect+segment+track in video; SAM 3.1 (2026-03) faster multi-object tracking | Drop-in upgrade behind the same endpoint: concept prompt → masklet for all matching objects (§7.4) |
| **SAM 3D** ([sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects), [sam-3d-body](https://github.com/facebookresearch/sam-3d-body)) | Public checkpoints + local inference (Nov 2025→): single-image → textured 3D object / body reconstruction | **Not a selection tool**; it reconstructs, it doesn't segment existing meshes. Out of scope for v2 selection; noted as a future aid for hole-filling after prop removal (§7.4) |
| **WebGPU wireframe techniques** ([xbdev WebGPU wireframe notebook](https://notebook.xbdev.net/index.php?page=webgpuwireframe), [tchayen: barycentric wireframes](https://tchayen.github.io/posts/wireframes-with-barycentric-coordinates)) | WebGPU has no polygon fill-mode; either expand to line-list (6 idx/tri) or draw triangles and edge-detect with barycentrics + `fwidth` AA in the fragment shader | §5/§7, the barycentric variant falls out of the same non-indexed "editor pipeline" that provides triangle-id picking |

---

## 3. Architecture overview

```
apps/demo (Editor tab / edit mode in Viewer)
│  timeline track: ranges ▬▬▬◆────◇────◆▬▬▬  (◆ user kf, ◇ derived kf)
│  tools: Brush · Box · SAM click / SAM text · X-ray toggle · Wireframe toggle
│  document: EditList (JSON, undo/redo command stack)
│
├── @ares/core player additions
│   ├── editor pipeline (WebGPU): non-indexed expanded draw, tri-id
│   │   ├── ID pass → r32uint triangle-id target (+ shared depth)   [picking]
│   │   ├── state buffer: 2 bits/tri (selected, deleted)            [highlight/hide preview]
│   │   └── barycentric wireframe / overlay                          [§7]
│   ├── CPU region math: dequant + project (through-selection, volume eval per frame)
│   └── evalEditList(frame) → per-tri keep/drop for live preview
│
├── tools/serve.mjs additions
│   ├── POST/GET /edits/<name>.edits.json    (sidecar save/load next to the .ares)
│   └── /sam/* → http://127.0.0.1:7263 proxy (SAM service; solves COEP, §7.1)
│
├── local SAM service (Docker: SAM 2.1 video predictor or SAM 3; GPU)
│
└── packages/encoder
    ├── edits.ts: parse EditList, per-frame volume interpolation, keepTri predicate
    └── cli.ts --edits foo.edits.json   (--crop = degenerate case)
        encode: read source OBJs → apply per-frame drops → compact → mux (unchanged)
```

Non-destructive throughout: the `.ares` is a delivery artifact; the edit list is a sidecar; the bake re-runs the encoder against the source OBJ/PNG frames (same rule as v1: "Bake: re-encode the SOURCE frames" in `apps/demo/main.js`).

---

## 4. Selection tools & semantics (Blender law)

Two authoring tools produce regions; one mode bit decides *which triangles they touch right now*.

**Mode law** (mirrors Blender exactly: [manual](https://docs.blender.org/manual/en/latest/interface/selecting.html): x-ray means "selection isn't occluded by the object's geometry"; solid shading selects only visible):

| Viewport state | Box / brush select | Implementation |
|---|---|---|
| Shaded (default) | **Visible-only**: only triangles with at least one pixel visible under the gesture | GPU triangle-id buffer, depth-tested (§5.2) |
| Wireframe toggle ON, or explicit X-ray toggle (Alt+Z analog) | **Through**: every triangle whose projection intersects the gesture region, at any depth | CPU: dequantize `curPosQ`, project centroids with `orbitViewProj`, 2D point-in-rect/circle test; no occlusion (§5.3) |

Wireframe implies x-ray (Blender's wireframe shading is see-through and selects through by default); the explicit toggle also allows x-ray selection while shaded.

### 4.1 Click-drag box select
Marquee in screen space. Immediate feedback = per-frame triangle set (highlight tint). What gets **stored** depends on mode:
- **Through:** the box is stored as a **frustum prism volume**; the 4 side planes through the rect edges (from the serialized camera). Depth-unbounded, matching what the user saw selected.
- **Visible-only:** stored as a `mask2d` volume: rect mask + camera + the depth band `[zmin, zmax]` of the selected pixels read from the depth attachment (§5.2). The band is what preserves "visible-only" at other frames without storing per-pixel depth.

### 4.2 Brush select
Circle cursor, drag to paint (Blender Circle Select; 4Dfx Brushes for the volcap precedent). Each `pointermove` sample picks the surface point under the cursor (1×1 ID+depth readback → world position) and appends a **world-space sphere** (center = surface hit, radius = brush radius scaled to world at that depth). A stroke = a polyline of spheres = a **capsule-chain SDF**. Stored as a `brushStrokes` volume, the brush *paints a volume*, not a vertex set, so it survives topology churn by construction. Selected-now = tris whose centroid is inside the union SDF (through mode) or that are additionally visible (shaded mode records the mask2d of painted pixels instead: cheaper and exact).

Wheel = radius (Blender convention), `[`/`]` alternates. Shift = subtract stroke (`op:"subtract"` on the stroke, carving the SDF union).

### 4.3 SAM click / SAM text
§7. Produces `mask2d` volumes (one per frame when propagated). From the selection system's point of view it is just a third region author.

---

## 5. WebGPU implementation

### 5.1 The editor pipeline: non-indexed expanded draw

The playback renderer keeps its indexed pipeline (storage-buffer vertex pulling + GPU dequant, `renderer.ts`). Entering edit mode switches the mesh draw to a second pipeline that renders **non-indexed, 3 × triCount vertices**, with the index buffer *also* bound as a storage buffer (add `STORAGE` usage at upload):

```wgsl
@group(0) @binding(6) var<storage, read> idx   : array<u32>;  // same bytes as the index buffer
@group(0) @binding(7) var<storage, read> tstate: array<u32>;  // 2 bits/tri: selected|deleted

@vertex fn vs_edit(@builtin(vertex_index) vi : u32) -> VSOut {
  let tri    = vi / 3u;                 // ← triangle id, for free
  let corner = vi % 3u;
  let v      = idx[vi];                 // pull the real vertex index
  // ...existing dequant path on v...
  let st = (tstate[tri >> 4u] >> ((tri & 15u) * 2u)) & 3u;
  if ((st & 2u) != 0u) { o.clip = vec4(0.0); }        // deleted → degenerate, tri vanishes
  o.bary = select(select(vec3(0.,0.,1.), vec3(0.,1.,0.), corner==1u), vec3(1.,0.,0.), corner==0u);
  o.tri  = tri + 1u;                                   // 0 = background in the ID pass
  o.sel  = f32(st & 1u);
}
```

Why this is the load-bearing choice, one pipeline family yields **four** editor features that the indexed path cannot express (an indexed VS sees only post-index `vertex_index`, so it has no triangle identity):

1. **Triangle-id picking target**: a second fragment entry writes `o.tri` to an `r32uint` attachment sharing the main depth buffer (§5.2).
2. **Deletion preview**: flip a bit in `tstate`, triangle collapses to degenerate. No index rebuild, no `discard` in the hot fragment path, ids stay stable. 20k tris = **5 KB** for the whole state buffer; updated with one `writeBuffer` per frame of preview.
3. **Selection highlight**: `sel` tints in the fragment shader (mix toward `#7fd8f0`, the UI accent).
4. **Barycentric wireframe / overlay**: `fwidth(bary)`-based AA edges (§7), including shaded-with-edges overlay mode which two-pipeline line-list rendering needs two passes for.

Cost: 60k VS invocations/frame vs ~11k (indexed, cached); vertex pulling is bandwidth-trivial here and this content already renders at 60 fps with ~0.4 ms decode; edit mode is scrub-heavy, not fill-bound. `tstate` is recomputed per presented frame by `evalEditList` (§6.4): 20k centroid-in-volume tests ≈ well under 1 ms in JS for the volume counts an edit session produces.

### 5.2 Picking: triangle-id render target + readback (chosen) vs CPU ray-cast

**Chosen: GPU ID buffer.**
- ID pass renders on demand (only during a gesture / SAM apply), same camera, into `r32uint` + depth. At interaction time only, never during playback.
- **Click pick:** `copyTextureToBuffer` of a 1×1 (padded to 256-byte row) region + `mapAsync` → triangle id + depth → world hit point. One readback per brush `pointermove` batch (coalesced per rAF).
- **Box select (visible-only):** read back the marquee rect, collect unique ids. Full-viewport worst case at 1080p = 8.3 MB, but rect-limited reads are typical; SAM mask application renders the ID pass at the mask's resolution (512² = 1 MB) instead.
- **Depth band capture:** the same readback provides the depth values that parameterize `mask2d.depth` (§4.1).

**Rejected as primary: CPU ray-cast** (Möller–Trumbore over ~20k tris after CPU dequant). It is perfectly adequate for single clicks (<1 ms) and we may still use it as the WebGL2-fallback click path, but it does not generalize: visible-only *region* selection needs occlusion resolution, which is exactly what a depth-tested ID raster gives for free, and SAM mask → triangle mapping is a per-pixel id lookup, i.e. inherently a raster operation. One mechanism (ID raster) serves click, box, brush, and mask; ray-casting serves only click.

**Through-selection stays on the CPU** (§4): project all centroids, 2D test. No readback, no occlusion, matches Blender x-ray semantics literally, and works identically in the WebGL2 fallback (where the editor pipeline may be reduced: GL2's crop preview is already a documented no-op; edit *preview* on GL2 can fall back to CPU-filtered index buffers).

### 5.3 CPU region math

`curPosQ` (u16) + GOP AABB + `dequantScale` already live on the CPU per presented frame (`player.ts` decode path). Editor v2 adds `player.getFrameGeometry()` exposing `{ positionsQ, indices, aabb }`; dequant of 11k verts is a ~0.1 ms typed-array loop. All volume evaluation (§6.2) uses these world positions, both for preview (`tstate`) and for through-selection gestures. The bake re-implements the identical predicate in the encoder against OBJ floats (§9): same math, two hosts, one shared `packages/core/src/edits.ts`, re-exported from `@ares/core` and imported by the encoder, so player preview and encoder bake cannot drift.

---

## 6. Time model: ranges, keyframes, interpolating regions

### 6.1 Ranges (HoloEdit intervals)

An **edit range** = `{ mode: "delete", startFrame, endFrame, keyframes[] }`. The operation applies only inside `[startFrame, endFrame]`: like a HoloEdit stage interval ("a specific portion of time where that stage will be applied"), drawn as a bar on the edit track with draggable end handles (HoloEdit's pink end tabs). A range ends when the deletion is no longer needed: before and after it, geometry is untouched.

### 6.2 Region keyframes and interpolation

Each keyframe pins the full region set at a frame: `{ frame, volumes: [...] }`. Between keyframes A (t=0) and B (t=1), with `t = (f - A.frame) / (B.frame - A.frame)`:

| Volume type | Interpolation | Rationale / analog |
|---|---|---|
| `box` | Lerp `min`/`max` corners | AE mask-path keyframe interpolation for the trivial case; HoloEdit's animated subtractive volume |
| `brushStrokes` | **SDF lerp:** `d(x,t) = (1−t)·d_A(x) + t·d_B(x)`, inside iff `d < 0` | Field interpolation needs **no correspondence** between stroke A and stroke B; the exact property AE lacks (its first-vertex pairing causes mask-morph artifacts) and that per-frame topology demands. Standard level-set shape morphing; degenerates to identity when the user doesn't re-stroke (hold) |
| `mask2d` | **None — a hard cut at t = 0.5** (corrected 2026-09-09; see below) | Only used when the user hand-places two sparse mask keyframes. SAM propagation instead emits **dense derived keyframes** (one per frame), like Premiere's tracker writing a keyframe every frame: inside a propagated span there is nothing to interpolate |

**What the `mask2d` row promised versus what `edits.ts` does** (this row read "camera lerp; mask via SDF lerp of the two masks' Euclidean distance transforms; depth band lerp" until 2026-09-09; none of the three was ever implemented). `prepareVolume` (`packages/core/src/edits.ts:351-398`) compiles each keyframe's bitmap against **that keyframe's own** camera matrix and **its own** depth band — there is no interpolation of either, and no distance transform anywhere in the file. The bitmap branch returns a two-valued field: `-0.5` inside, `+0.5` outside (`edits.ts:392`, commented "binary in/out; sign drives the predicate"). The range lerp `(1−t)·d_A + t·d_B` (`edits.ts:465`) applied to two such fields is therefore negative wherever A is inside and t < 0.5, and wherever B is inside and t > 0.5: a sparse pair resolves to **exactly A below the midpoint, exactly B above it, and A ∩ B at t = 0.5** (the disagreement region evaluates to exactly 0, and the predicate is `< 0`). Box and brush volumes still lerp as the rows above describe — those SDFs carry real magnitude in mm; only `mask2d` is degenerate, because a binary bitmap has no magnitude to blend.

Consequences: a propagated span must write a keyframe on **every** frame rather than sparse anchors, and the SAM section's "the range tweens between them" tooltip (`apps/demo/index.html`) describes behaviour that does not exist. Implementing the promised morph means a Euclidean distance transform at decode time — a per-keyframe EDT over a 768×432 bitmap, then a lerp of two float fields — and is only worth the decode cost if hand-placed sparse mask keyframes become a real authoring path.

Outside the keyframe span but inside the range: **hold** first/last keyframe (AE hold-keyframe semantics). Evaluation of a frame is pure: `insideRegion(worldPoint, frame)`; the same function drives preview and bake.

**Why geo+texture stay synchronized for free:** the interpolated region drops *triangles*; each triangle carries its UVs (§8). There is no separate texture mask to keep in step, the region is the single source of truth per frame, which is the coherence property the task demands.

### 6.3 Timeline UX

One **edit track** under the existing scrubber (`#scrub`):

```
frames   0 ........ 40 ........ 92 ........ 150 ........ 180 ........ 271
scrub    ────────────▮───────────────────────────────────────────────────
range r1             ▐▬▬▬◆▭▭▭▭◇▭▭▭▭◇▭▭▭▭◆▬▬▬▬▬▬▬▬▬▌          (delete: boom mic)
                        └ base kf   └ derived (SAM)  └ user correction kf
```

- **Filled diamond ◆** = user keyframe (draggable along the track, editable): HoloEdit's "large, filled diamond … can be dragged"; **hollow diamond ◇** = derived keyframe (SAM-propagated, read-only until touched): HoloEdit's "small, hollow … derived keyframe".
- A propagated span renders with a chevron/hatch fill and the base frame gets a distinct marker: AE Roto Brush's span + green base-frame convention, so users who know AE read it instantly.
- Editing the selection at a frame with no keyframe **creates** one there (AE behavior when a property already has keyframes). Editing at a derived keyframe promotes it to a user keyframe (Roto Brush "correction" semantics).
- Range bar drag = move; end-tab drag = trim (HoloEdit interval resize). Multiple ranges stack in lanes; a triangle is dropped at frame f if **any** active range's region contains it.

### 6.4 Live preview

On every presented frame in edit mode: `evalEditList(frameIndex)` → interpolate active regions → test 20k centroids → pack `tstate` bits → `writeBuffer` (5 KB). Deleted tris collapse in the VS (§5.1); selected tris tint. Scrubbing therefore shows the interpolated deletion exactly as it will bake, the property v1's live crop preview established (`crop.ts` header comment: preview "mirrors the renderer's live fragment-discard"; v2 keeps that contract with a different mechanism).

---

## 7. Wireframe rendering (WebGPU has no fill-mode)

WebGPU offers no `polygonMode`/wireframe fill state, so wireframe is synthesized. Two candidates ([xbdev WebGPU wireframe](https://notebook.xbdev.net/index.php?page=webgpuwireframe) documents exactly this pair; [tchayen](https://tchayen.github.io/posts/wireframes-with-barycentric-coordinates) details the barycentric math):

| | **Line-list expansion** (shipped, v1) | **Barycentric fragment shader** (editor pipeline, §5.1) |
|---|---|---|
| Geometry | tri (a,b,c) → edges ab,bc,ca = **6 indices/tri**; `line-list` pipeline (`renderer.ts uploadLineIndices`) | Same triangles, non-indexed 3×T draw; one-hot bary from `vi % 3` |
| Extra memory (20k tris) | 480 KB line index buffer, rebuilt per topology upload (per frame in intra mode) | 0 (reuses index-as-storage); +5 KB tstate it shares anyway |
| VS invocations | 120k | 60k |
| Line quality | 1 px, aliased (WebGPU lines are hairlines; no width control) | `fwidth`-AA, adjustable width, screen-space-constant |
| Shared interior edges | Drawn twice (harmless visually, noted in code) | Computed once per pixel |
| Overlay (shaded + edges) | Needs a second pass | Single pass: `mix(shaded, edgeColor, edgeMask)` |
| Synergy | None | Same pipeline as picking IDs, selection tint, deletion collapse |

**Recommendation for ~20k-tri meshes:** both are far below budget (this renderer does 60 fps with the whole mesh re-uploaded per frame), so performance does not decide; **capability does**. Adopt the **barycentric editor pipeline** as the edit-mode wireframe (it must exist anyway for tri-id picking and deletion preview; edges are ~10 lines of WGSL on top) and keep the v1 line-list toggle for plain playback viewing outside edit mode, where the indexed pipeline stays bound and the 480 KB buffer is already paid for. If maintaining two wireframes proves annoying, delete the line-list path once v2 lands: nothing outside the Edit toggle uses it.

X-ray display mode (see-through, matching the selection law of §4) = render the overlay with `alphaBlend` + depth-write off + a low fill alpha; selection tint stays opaque so selected-through triangles read clearly, which is the visual cue Blender gives in x-ray.

---

## 8. SAM-assisted selection

### 8.1 Service topology

A **local SAM HTTP service** runs beside the dev server; Meta's own web demo ships exactly this shape: dockerized Flask+GraphQL backend on `localhost:7263` with the SAM2 video predictor ([demo/README](https://github.com/facebookresearch/sam2/blob/main/demo/README.md)). We wrap SAM 2.1 (or SAM 3, §8.4) in a minimal FastAPI façade with three routes:

```
POST /segment        { image(png b64), points[{x,y,label}] | box | text } → { maskPng | rle, score }
POST /propagate      { frames[](png b64 | video/webm), prompts on frame 0, direction } → SSE/chunked: per-frame masks
GET  /health         → { model, device }
```

**The dev server proxies it** (`tools/serve.mjs` gains `app.use("/sam/*") → http://127.0.0.1:${SAM_PORT}`). This is not a convenience: `serve.mjs` sends `Cross-Origin-Embedder-Policy: require-corp` on every response to keep `crossOriginIsolated` true for SharedArrayBuffer (worker decode path). Under COEP, the page **cannot fetch cross-origin responses that lack CORP headers**: a stock localhost:7263 SAM container would be blocked. Same-origin proxying through `/sam/*` sidesteps the whole class of problem and keeps the SAM container stock.

### 8.2 Click → mask → triangles (single frame)

> **[SHIPPED 2026-07-10]** The SAM tool sits beside Box/Brush in the demo editor. Implementation
> matches this section with one simplification: instead of an ID-pass triangle mapping at mask
> resolution (step 3), the applied volume stores the mask itself as an RLE bitmap
> (`mask2d kind:"bitmap"` in `core/edits.ts`) plus the depth band from the pick raster over
> mask-covered pixels, the same visible-only law the rect marquee uses. Centroids are tested
> against the bitmap at preview and bake through one shared evaluator, so the stored artifact is
> the region (step 5) and bake-time re-evaluation works unchanged. Shift-click adds exclusion
> points; clicks accumulate into one refined mask before Apply.

1. Pause on frame f. User clicks (fg) / alt-clicks (bg) points on the canvas: SAM 2's native point-prompt loop.
2. Snapshot the rendered view: draw the WebGPU canvas into a 2D canvas → `toBlob` (≤1024px long side) → `POST /sam/segment` with the click points in image coordinates.
3. Response mask (RLE/PNG) → **triangle mapping**: render the ID pass (§5.2) at mask resolution from the identical camera, read back once; every mask-covered pixel contributes its triangle id. This is **depth-tested, visible-only by construction**: occluded geometry behind the person's arm is not selected, satisfying the shaded-mode law.
4. Optional dilation: grow the selection by shared-vertex adjacency once (cheap per-frame union-find on `indices`) to catch silhouette triangles that rasterized thinner than a pixel.
5. Store as a `mask2d` volume `{ camera, mask, depth: [zmin,zmax from the ID pass], visibleOnly: true }` at keyframe f. The stored artifact is the region, per §1, never the id list.

### 8.3 Temporal propagation (masklets → dense derived keyframes)

1. User confirms the frame-f selection and hits **Propagate** (forward / both: Premiere's track-forward/backward affordance).
2. The player renders frames `f..endFrame` offscreen **from the frozen camera** (auto-orbit disabled) at mask resolution and streams them to `/sam/propagate`. SAM 2's memory attention tracks the object and returns one mask per frame (the masklet; `propagate_in_video`).
3. Each returned mask becomes a **derived keyframe** (`derived: true`) with the shared frozen camera: hollow diamonds fill the span, chevron styling appears (§6.3). Per-frame masks mean per-frame regions, no interpolation error accumulates, exactly like Premiere writing tracker keyframes every frame.
4. Scrub the span; where SAM drifted, re-click on that frame → correction promotes/adds a user keyframe and (optionally) re-propagates from there: AE Roto Brush's correction loop verbatim.
5. Depth bands are captured per propagated frame during the same offscreen renders (2 floats/frame), keeping visible-only semantics as the subject moves.

**Rejected alternative: running SAM on the atlas video:** the texture track already is a video (VP9/AV1 of per-frame atlas PNGs), so "segment the atlas video" is tempting (no re-render). But volcap atlases are chart-scrambled: a person is dozens of discontiguous UV islands, re-packed per frame; and SAM's video memory assumes objects that cohere in image space. Screen-space rendering is the representation SAM was trained on; we render cheap proxy frames instead. (Noted for completeness since both videos exist in the container.)

**Bake-time re-evaluation** (why this survives per-frame topology): at bake, frame f's *source OBJ* triangles are projected into the stored camera and tested against mask f + depth band (§9). The mask constrains a region of the view frustum; whatever geometry frame f has there gets dropped. Vertex correspondence is never consulted.

### 8.4 SAM 3 / SAM 3D status (measured 2026-09-09)

Everything numbered below was produced by `tools/sam-service/track_smoke.py`, a read-only probe
that CI never runs (it loads a 3.4 GB checkpoint) and whose assertions are its own test. Run it
to reproduce any of these: `env/Scripts/python.exe track_smoke.py load | seam | measure`.

**Installed environment.** transformers **5.13.0**, torch **2.6.0+cu124**, torchvision
0.21.0+cu124, Python 3.13.7, in `tools/sam-service/env`. These are NOT the pins in
`requirements.txt` (5.16.1 / 2.14.0); every signature the propagation work cites was read from
the installed tree. The checkpoint is the `facebook/sam3` hub-cache snapshot `3c879f39…`,
`model.safetensors` 3,439,938,512 bytes (`main.py:45-80` resolves it there). All of `sam3`,
`sam3_tracker`, `sam3_tracker_video`, `sam3_video`, `sam2` and `sam2_video` are present.

**Checkpoint shape**, read out of the safetensors header without loading torch: 1797 tensors, of
which **538** under `detector_model.vision_encoder.*`, **0** under `tracker_model.vision_encoder.*`,
and 22 under `tracker_neck.*`. There is exactly one vision tower on disk.

**The load path.** `Sam3VideoModel.from_pretrained` reports **0 missing and 0 mismatched keys** —
859.92 M parameters: detector 840.38 M (its vision tower 454.04 M), tracker 11.74 M,
`tracker_neck` 7.80 M. Its `tracker_model.vision_encoder` **is `None`**, because
`modeling_sam3_video.py:512` constructs the tracker with `remove_vision_encoder=True`.
Checkpoint config as loaded: `num_maskmem` 7, `max_cond_frame_num` 4, `low_res_mask_size` 288,
tracker `image_size` 1008.

**Negative control, and a correction.** Loading the tracker alone was expected to fail loudly
(`base_model_prefix = "tracker_model"` at `modeling_sam3_tracker_video.py:703` plus
`_keys_to_ignore_on_load_unexpected = [r"^detector_model."]` at `:1602` predict 538 randomly
initialised vision tensors). Measured, it does not: on transformers 5.13.0
`Sam3TrackerVideoModel.from_pretrained` loads **845 tensors, 0 missing / 0 unexpected /
0 mismatched**, because the flexible cross-architecture loader remaps
`detector_model.vision_encoder.*` onto `vision_encoder.*` (516 trunk tensors, all bit-identical to
the checkpoint) and `tracker_neck.*` onto `vision_encoder.neck.*` (22 tensors, bit-identical).
The real cost is duplication, not corruption: **454.04 M of that model's 465.78 M parameters are a
second copy of the tower the concept model already holds**, ~908 MiB at fp16. That is the
argument for loading `Sam3VideoModel`, and `track_smoke.py load` pins the bit-identity so a
future transformers that stops remapping fails there instead of in a quietly worse mask.

**Resident-set delta of adding video tracking**, by parameter count rather than estimate. Today's
pair after the vision-tower graft (`main.py:180`) is `Sam3TrackerModel` 458.26 M + `Sam3Model`
840.38 M − the 454.04 M shared tower = **844.60 M**. The video path is `Sam3VideoModel` 859.92 M
plus the click tracker's non-vision remainder 4.22 M = **864.14 M**. **+19.54 M parameters,
37.3 MiB at fp16, for the whole memory bank.**

**The towerless-tracker seam is verified running**, not just read. `track_smoke.py seam` primes
the vision-feature cache with the library's own three-call sequence
(`modeling_sam3_video.py:1614-1632`: `detector_model.get_vision_features` →
`get_vision_features_for_tracker` → `inference_session.cache.cache_vision_features`), seeds one
object, and forwards `vm.tracker_model(inference_session=…, frame_idx=…)`. Result on CUDA in
fp16: **0 cache misses** across the seeded frame and the propagated frame, `pred_masks`
(1, 1, 288, 288), `post_process_masks` restoring frame size. `_prepare_vision_features`
(`modeling_sam3_tracker_video.py:1900`) consults the cache before it would dereference the `None`
encoder, so a forward that returns at all is itself the proof; the miss counter is the positive
evidence beside it.

**Measured propagation**, RTX 2080 Ti (sm_75, 11 GB), fp16, 24 frames at trackRes 1008, one
object, `torch.cuda.synchronize()` around each phase:

| | fp16 | bf16 |
|---|---|---|
| per frame, end to end | **291.6 ms** | 1,286.3 ms |
| vision encoder, median / max | 189.3 / 523.5 ms | 1,079.7 / 1,191.1 ms |
| tracker head, median / max | 75.6 / 184.8 ms | 197.2 / 221.9 ms |
| first frame (warm-up included) | 708.2 ms | 1,243.7 ms |
| device: weights | 1,835 MiB | 1,884 MiB |
| device: peak allocated | **2,263 MiB** | 5,789 MiB |
| device: activations + feature cache | 428 MiB | 3,905 MiB |
| device: peak reserved (what `nvidia-smi` shows) | 3,744 MiB | 7,076 MiB |
| host RSS | 3,358 MiB | 3,808 MiB |
| `from_pretrained` + `.to(cuda)` | 4.3 s | 4.1 s |

At 291.6 ms/frame a 272-frame forward pass is **≈ 79 s**, and the whole clip's masks fit inside
2.3 GiB of device allocation. Max is reported rather than p95, per the instrumentation rule in
`docs/targeted-temporal.md:27`.

**fp16 is the right dtype on this card, twice over.** `_best_dtype` (`main.py:92-106`) already
picks fp16 below sm_80 on a throughput measurement; this adds a second, independent one — bf16 is
**4.4× slower end to end, 5.7× slower in the encoder, and needs 2.6× the device memory** (the
activation footprint alone goes 428 MiB → 3,905 MiB) because pre-Ampere bf16 falls off the
tensor-core path onto fallback kernels. Note these bf16 numbers are the pre-Ampere penalty and say
nothing about bf16 on an Ampere-or-later card, where it is the documented path.

**Three things these numbers do not settle.** (a) **The fp16 accuracy gate.** Whether fp16 is safe
for the video tracker rests on its mask agreement with bf16 (every SAM 3 model-card video example
is bfloat16; no published fp16 video validation exists), and this corpus cannot answer it. The
only numbered frame sequence on this machine is the source capture's **atlas** PNGs, and §8.3
already records why an atlas is not a SAM input: chart-scrambled, so the tracker never establishes
a lock — object score falls from 19.8 logits at the seed to a 0.3–2.6 band and mask area
oscillates tenfold frame to frame, after which the two runs' memory banks diverge and stay
diverged. Across all 24 frames the IoU median is 0.9245 and the minimum 0.0, which measures the
corpus. On the one frame where **both** runs hold a confident lock the dtypes agree at **IoU
0.9867**, which is the only line here that is about numerics. `track_smoke.py measure` therefore
reports the locked series separately; re-run it over frozen-camera proxy renders once the player
can capture them, and gate on that series. (b) **VRAM on the 6 GB Ampere card `main.py` also ships
to**: it picks bf16 on the proper tensor-core path, which this sm_75 box cannot exercise. (c)
**Mask quality**, which per `docs/targeted-temporal.md:66-72` needs visual review of a full
propagated span, not a metric — every distance metric in this toolchain has a 0.5–1 mm floor and
read ~0 for the flattened-nose failure.

**The 2026-07-10 blocker is withdrawn.** It read: the SAM 3.1 video tracker "loads only via the
facebookresearch/sam3 repo path today and OOM'd a 24 GB 4090 in video mode (issue #511)". That
loader materialises the whole decoded video on the device before propagating; this path does not
exist here. `Sam3TrackerVideoInferenceSession` holds frames on `video_storage_device` and caches
**one** frame's vision features at a time (`max_vision_features_cache_size` defaults to 1,
`modeling_sam3_tracker_video.py:72-77` evicts the oldest on insert), which is why the measured
peak above is 2.3 GiB on an 11 GB card rather than 24 GB. Neither the SAM 2.1 fallback nor CPU
frame offloading is needed.

**Still true from 2026-07-10.** SAM 3 is the service's primary backend (`main.py`,
`Sam3TrackerModel` click path, `Sam3Model` concept path sharing one grafted vision tower, ViT-H
fallback); `/segment` returns 0.93-score masks in 360-550 ms per warm click on the 6 GB 3060; the
service starts from inside the app (`/sam/start` SSE, Edit panel SAM row); the click tool shipped
(§8.2) and the per-image vision-feature cache that section wanted now exists (`main.py:279-320`).
Concept (text) prompts run through the same weights.

**SAM 3D** (`sam-3d-objects`, `sam-3d-body`) does single-image 3D reconstruction, not segmentation
of existing meshes, so it plays no role in selection. Researched 2026-07-10: sam-3d-body outputs
Meta's MHR parametric body (a possible pose prior for temporal denoise, via the fast InstantHMR
distillation), and SAM 3D Objects could someday synthesize patch meshes for hole-filling: both out
of scope for v2.

---

## 9. Texture coherence

Geometry and texture stay paired **by construction**: UVs are per-vertex (`positions`+`uvs` move through `cropFrame` compaction together today), so a dropped triangle takes its atlas footprint out of sampling; nothing references those texels afterward. The v1 crop bake already proves this end-to-end (crop → fewer tris → same atlas video → correct rendering).

**Is atlas-space masking or inpainting ever needed? Mostly no, with three deliberate exceptions:**

1. **Not for correctness.** Orphaned texels remain in the encoded atlas but are unsampled. Bilinear/mip bleed at kept-chart borders is *unchanged* by deletion because we do not blank anything, the neighboring texels still hold the original capture colors. (This is why v2 explicitly does **not** re-pack the atlas: HoloEdit's subtractive-volume pipeline demands Clean Mesh → Generate UVs → **Texture Transfer** afterward precisely because it re-atlases; skipping re-atlas skips that entire failure class, including the UV-seam artifact class this project has already hit.)
2. **Bitrate reclaim (optional, later):** orphaned texels still cost bits in the VP9/AV1 encode. Measured pressure is low (texture is ~12–15 MB of a ~50 MB file) and orphan area is proportional to deleted surface. If a large standing prop is deleted for the whole clip, an optional pre-ffmpeg pass can zero orphaned texels + 4-px gutter dilation (reduces bitrate, *increases* nothing visually since the texels are unsampled). Off by default.
3. **Privacy/legal scrub (`scrubTexels: true` per range):** deletion for privacy (a face, a license plate) must remove pixels, not just references; orphaned texels are still extractable from the atlas video. The bake then rasterizes the dropped triangles' UV triangles into a per-frame mask and black-fills before texture encode. This is the one case where the geometric edit *must* be mirrored in atlas space, and the per-frame UV rasterization gives it exactly (same triangles, same frames: coherence again by construction).
4. *(Future, explicitly v3+)* 4Dfx-style **texture brushes** (recolor/luminosity) would edit atlas video content per frame: a different feature with a different pipeline (mask-guided video filtering), not required for delete.

---

## 10. Bake path

Edits compile to **per-frame triangle drops applied at encode time**, in the encoder, against the source frames, never by rewriting the `.ares`:

```
ares encode <frames-dir> -o out.ares --edits daniel.edits.json [--fps 30 ...]
```

Per frame f (in `cli.ts encode`, replacing the current `--crop` block):
1. `activeRanges = ranges.filter(r => r.startFrame <= f && f <= r.endFrame)`
2. For each, interpolate its region at f (§6.2) → `insideAny(p)` predicate. Volume math lives in `packages/core/src/edits.ts`, shared with the player preview (§5.3).
3. `filterFrame(frame, keepTri)`: generalization of `cropFrame` where `keepTri(t)` = *"centroid **not** inside any delete region"* (and `= "centroid inside the keep region"` for crop). Pass 2 (vertex compaction/remap of positions+UVs) is reused verbatim from `crop.ts`.
4. `scrubTexels` ranges: rasterize dropped tris' UVs → mask → blank atlas PNG before the ffmpeg texture pass (§9.3).
5. Log per-range drop counts like today's crop log (`tris/frame (x% removed)`).

**`--crop` is the degenerate edit list** and is reimplemented as one:

```json
{ "ranges": [ { "mode": "keep", "startFrame": 0, "endFrame": 271,
    "keyframes": [ { "frame": 0, "volumes": [ { "type": "box", "min": [x0,y0,z0], "max": [x1,y1,z1] } ] } ] } ] }
```

(`mode:"keep"` = drop everything *outside* the region; `mode:"delete"` = drop inside. v1's six sliders keep working; they just author this object now.)

**Centroid rule, kept from v1:** a triangle is dropped iff its **centroid** is inside the region (HoloEdit removes *vertices* a volume overlaps; centroid avoids the sliver/partial-triangle cases vertex rules create and never splits a UV-mapped triangle, the geo/texture pair lives or dies whole). Documented tolerance: player-preview geometry differs from source OBJs by quantization (≤ bbox/65535 per axis) and `--smooth` Taubin displacement; regions are world-space with brush radii and mask pixels orders of magnitude larger, so the same predicate lands identically. Frame alignment uses the same sorted-filename order in both hosts (`cli.ts` and the player's frame index).

---

## 11. Data model, the `.edits.json` sidecar

Saved next to the clip (`apps/demo/daniel-s0.edits.json`) via `POST /edits`; loaded by the editor on open; passed to the encoder at bake. Non-destructive by definition: deleting the sidecar restores the source.

```jsonc
{
  "aresEdits": 1,                        // schema version
  "source": "daniel-s0.ares",            // the clip this was authored against (informational)
  "fps": 30, "frameCount": 272,
  "ranges": [
    {
      "id": "r1", "label": "boom mic",
      "mode": "delete",                  // "delete" | "keep" (crop)
      "startFrame": 40, "endFrame": 180,
      "scrubTexels": false,              // §9.3 privacy blanking
      "keyframes": [
        { "frame": 40,                   // user keyframe (filled diamond)
          "volumes": [
            { "type": "box", "min": [-0.4, 1.6, -0.2], "max": [0.1, 2.2, 0.3] },
            { "type": "brushStrokes", "strokes": [
                { "op": "add", "radius": 0.06,
                  "points": [[-0.31,1.82,0.02], [-0.28,1.85,0.04], [-0.22,1.88,0.05]] } ] },
            { "type": "mask2d", "visibleOnly": true,
              "camera": { "azimuth": 0.62, "elevation": 0.18, "distance": 11.2,
                          "target": [0.0, 0.9, 0.0], "fovYDeg": 50, "aspect": 1.778 },
              "mask": { "encoding": "rle", "width": 512, "height": 512, "data": "…" },
              "depth": { "zmin": 0.912, "zmax": 0.941 } }               // NDC band from the ID pass
          ] },
        { "frame": 92,  "derived": true, "volumes": [ /* SAM masklet frame 92 */ ] },
        { "frame": 150, "volumes": [ /* user correction */ ] }
      ]
    }
  ]
}
```

Rules: keyframes sorted by frame, all within `[startFrame, endFrame]`; hold before-first/after-last; `derived` keyframes are regenerable (a propagation re-run may replace them) and render hollow; volumes within one keyframe are a union; strokes with `op:"subtract"` carve. Interpolation per §6.2. Everything is world-space or serialized-camera space: **no vertex or triangle indices appear anywhere in the format**, which is the invariant that makes it valid against source OBJs, re-encodes, and future re-captures of the same scene scale.

Undo/redo = command stack over this document; autosave debounced to the sidecar.

---

## 12. New surface area (implementation checklist)

**@ares/core**: editor pipeline (non-indexed VS variant + ID pass + tstate buffer + barycentric edges); `player.getFrameGeometry()`; `player.setEditPreview(editList | null)`; `player.renderIdPass(camera?, size?) → {ids, depth}` readback helper; offscreen frame renders for propagation; expose `project/unproject` from `camera.ts`.
**apps/demo**: Edit tab v2: tool palette (Box / Brush / SAM point / SAM text), x-ray + wireframe toggles honoring §4's law, edit track under `#scrub` (ranges, diamonds, chevrons), keyframe navigation (prev/next diamond), bake panel extended to send `--edits` (reusing the v1 SSE flow in `main.js initEditor`).
**tools/serve.mjs**: `GET/POST /edits/:name` (write next to the demo clips, path-sanitized like `/encode`); `/sam/*` reverse proxy with `SAM_URL` env (default `http://127.0.0.1:7263`), streaming pass-through for propagation.
**packages/core**: `edits.ts` (schema parse/validate, per-frame region interpolation), consumed by both hosts. **packages/encoder**: `filterFrame`; `cli.ts --edits`; `--crop` reimplemented as degenerate list; optional texel scrub pre-pass in `texture-video.ts`.
**SAM sidecar**: `tools/sam-service/` Dockerfile + FastAPI façade over SAM 2.1 video predictor (upgrade path: SAM 3 checkpoint swap), README with `docker compose up` (GPU) mirroring Meta's demo topology.

**Suggested build order:** (1) editor pipeline + ID picking + box/brush through+visible selection, single-frame delete preview → (2) ranges/keyframes/interpolation + track UI + bake via `--edits` (crop parity test: v1 sliders → identical output through the new path) → (3) SAM click single-frame → (4) propagation + derived keyframes → (5) scrubTexels + polish.

**Risks:** SAM drift on fast limbs (mitigated by the correction loop, §8.3); readback latency spikes on low-end GPUs (coalesce to ≤1 in-flight `mapAsync`); WebGL2 fallback gets a reduced editor (CPU-only selection, no ID pass): acceptable, matching the existing GL2 crop-preview no-op precedent; edit-preview divergence from bake (guarded by sharing `edits.ts` and a bake-parity test on the crop case).

---

## 13. Sources

- HoloEdit Mesh Capture Tool (subtractive volumes, keyframed): https://learn.arcturus.studio/docs/holoedit/en/v2023.2/html/External/Mesh_capture_tool.html
- HoloEdit compositions/intervals/keyframe diamonds: https://arcturus.studio/docs/holoedit/en/v2021.1/html/Editing_Compositions/Editing_Compositions.html · basics: https://learn.arcturus.studio/docs/holoedit/en/v2022.2/html/HoloEdit_Basics/HoloEdit_Basics.html
- Arcturus volumetric editing overview (SSDR segments, texture transfer): https://www.fxguide.com/fxfeatured/arcturus-volumetric-video-editing/
- 4DViews 4Dfx (Brushes): https://www.4dviews.com/volumetric-software · https://blog.laval-virtual.com/en/4dviews-more-photorealistic-volumetric-videos-than-ever-before/
- After Effects mask animation & Smart Mask Interpolation: https://helpx.adobe.com/after-effects/using/animating-shape-paths-masks.html · Roto Brush spans/base frames: https://helpx.adobe.com/ca/after-effects/using/roto-brush-refine-matte.html
- Premiere Pro mask tracking (dense auto-keyframes): https://helpx.adobe.com/premiere/desktop/add-video-effects/create-masks-and-composites/track-masks.html
- Blender selection & x-ray semantics: https://docs.blender.org/manual/en/latest/interface/selecting.html · select-through design task: https://developer.blender.org/T73479
- SAM 2 (video predictor, masklets, propagate_in_video): https://github.com/facebookresearch/sam2 · local web-demo backend: https://github.com/facebookresearch/sam2/blob/main/demo/README.md
- SAM 3 / 3.1 (concept prompts, tracking): https://ai.meta.com/blog/segment-anything-model-3/ · https://github.com/facebookresearch/sam3
- SAM 3D (reconstruction, local checkpoints): https://github.com/facebookresearch/sam-3d-objects · https://github.com/facebookresearch/sam-3d-body
- WebGPU wireframe (line-list vs barycentric): https://notebook.xbdev.net/index.php?page=webgpuwireframe · https://tchayen.github.io/posts/wireframes-with-barycentric-coordinates


## Addendum 2026-09-07: tools added after the audit

- **Lasso** (`A`): a free polygon rasterized into the same `mask2d/bitmap` volume SAM produces, so
  the evaluator, the X-ray/depth-band law and the bake path are shared, not forked.
- **Measure** (`T`): two surface picks through `pickRaster`, re-projected as the camera moves,
  read in the clip's inferred units. Camera-only; never touches the sidecar.
- **Grow / shrink / invert / mirror** on the ▶ active range: boxes per face, brush radii, marquee
  rects, and bitmaps by the matching pixel count (`growKeyframe`, `morphBitmap` in core
  `edits.ts`); invert flips `mode` between delete and keep; mirror duplicates box and brush
  regions across the clip's centre X plane (`mirrorKeyframe`; screen-space masks cannot be
  mirrored and are counted, not silently skipped).
- **Range properties**: `enabled:false` mutes a range everywhere (preview and every bake op filter
  on `isRangeEnabled`), `label` names it, `interp` selects linear / hold / smooth keyframe
  interpolation in `prepareRangeSdfAt`.
- **Sculpt** (`action:"sculpt"`, encoder `sculpt.ts`): world-anchored vertex displacement inside
  the interpolated region: move, inflate, smooth, flatten, pinch; weld-aware and feathered
  with the paint op's law. Bake-side only, like paint; the timeline shows the range.
- **Analysis views** in the shading control: normals, UV checker, depth, points (the vertices as
  a point cloud, textured, unlit, with a size slider). Both renderers.
- **Export** rail section: the presented frame to OBJ (`AresPlayer.exportFrame()`), a still to
  PNG, an 8-second turntable to WebM through `MediaRecorder` (orbit speed is restored after).
- **Camera bookmarks** (`◈`, `C` cycles): per clip in `localStorage`; Shift-click removes.
- `?` opens a shortcuts panel; the tooltips still carry the same words at each control.
- **FX** (`packages/core/src/fx.ts`): playback-time effects on meshes and splats; clip plane,
  dissolve (3D value noise + burn rim), tint, fresnel rim, scanlines, wobble, splat jitter / size /
  opacity: as one uniform block in both renderers, keyframed in `edits.fx` and evaluated per
  presented frame (`setFxTrack`), with an override layer for audio-reactive modulation
  (`setFxOverride`, `getAudioLevel`). Never baked: the encoder ignores `edits.fx`.
