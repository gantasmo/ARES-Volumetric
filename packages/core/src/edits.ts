/**
 * Edit-list model (editor v2, docs/editor-v2-design.md §6/§10/§11).
 *
 * Selections on per-frame-independent volcap are TIME-ANCHORED WORLD REGIONS — never vertex or
 * triangle ids (no frame-to-frame correspondence exists, and meshopt reorder renumbers within a
 * frame). A range holds keyframed region sets; regions INTERPOLATE between keyframes (box corner
 * lerp; brush strokes via SDF lerp — correspondence-free), and HOLD outside the keyframe span.
 * A triangle is dropped at frame f iff its CENTROID is inside any active delete region (or outside
 * every keep region). The same predicate drives the player's live preview and the encoder's bake,
 * which is the geo/texture-coherence property: a dropped triangle takes its texels with it.
 *
 * This module is pure and shared: @ares/core (preview) + @ares/encoder (bake).
 */
import { orbitViewProj } from "./camera.js";

export interface BoxVolume { type: "box"; min: [number, number, number]; max: [number, number, number]; }
export interface BrushStroke { op: "add" | "subtract"; radius: number; points: [number, number, number][]; }
export interface BrushVolume { type: "brushStrokes"; strokes: BrushStroke[]; }
/**
 * Screen-space selection serialized as camera + region (design §11) — world/camera space only, no
 * triangle ids. `kind:"rect"` = marquee box select: a point is inside when its projection under
 * the SERIALIZED camera lands in the NDC rect (and, for solid-mode visible-only selects, within the
 * captured depth band). `kind:"bitmap"` = SAM/painted masks: the projection is looked up in an
 * RLE-coded bitmap covering the full captured viewport (same NDC mapping, same depth-band law).
 */
export interface Mask2dVolume {
  type: "mask2d";
  kind?: "rect" | "bitmap";
  /** NDC-space rect [x0, y0, x1, y1], y up, each in [-1, 1], x0<x1, y0<y1 */
  rect?: [number, number, number, number];
  /** Row-major bitmap covering the captured viewport; rle alternates 0-run/1-run lengths, 0-run first. */
  mask?: { width: number; height: number; rle: number[] };
  camera?: { azimuth: number; elevation: number; distance: number; target: [number, number, number]; aspect: number };
  /** NDC z band captured from the pick buffer (visible-only approximation, design §5.2) */
  depth?: { zmin: number; zmax: number };
  [k: string]: unknown;
}

/** Encode a 0/1 bitmap as alternating run lengths (0-run first — a leading 1 yields a 0-length first run). */
export function rleEncodeMask(bits: Uint8Array): number[] {
  const rle: number[] = [];
  let cur = 0, run = 0;
  for (let i = 0; i < bits.length; i++) {
    const b = bits[i] ? 1 : 0;
    if (b === cur) run++;
    else { rle.push(run); cur = b; run = 1; }
  }
  rle.push(run);
  return rle;
}

/** Decode rleEncodeMask output back to a 0/1 Uint8Array of the given size. */
export function rleDecodeMask(rle: number[], size: number): Uint8Array {
  const bits = new Uint8Array(size);
  let i = 0, val = 0;
  for (const run of rle) {
    if (val) bits.fill(1, i, Math.min(size, i + run));
    i += run;
    val ^= 1;
    if (i >= size) break;
  }
  return bits;
}

/** Decoded-bitmap cache: prepareVolume runs once per frame per volume; the RLE decodes once per volume. */
const maskBitsCache = new WeakMap<object, Uint8Array>();
export type EditVolume = BoxVolume | BrushVolume | Mask2dVolume;

export interface EditKeyframe { frame: number; derived?: boolean; volumes: EditVolume[]; }
export interface EditRange {
  id?: string;
  label?: string;
  /** Display color (hex) assigned by the editor's segmentation palette; cosmetic only. */
  color?: string;
  mode: "delete" | "keep";
  /**
   * Bake-side edit op ("frame-to-frame copy" / "garment recolor"). Absent ⇒ "delete" — today's only behavior, and it MUST stay byte-identical:
   * every copy/recolor code path is gated on its own `action` value. `mode` is still required
   * by the schema for a copy or recolor range (author it as "delete", it is ignored) because
   * `keepPredicateAt` below excludes any non-"delete" action from BOTH the delete and keep
   * sweeps outright — a copy or recolor range is a pure no-op for delete-preview purposes, in
   * the live player AND in the encoder's own delete/keep pass. Live preview of the op itself is
   * out of scope for v1 (encoder-only: see packages/encoder/src/frame-copy.ts + texel-copy.ts
   * for copy, packages/encoder/src/recolor.ts for recolor); the demo timeline still renders the
   * range (bar/keyframes) like any other.
   */
  action?: "delete" | "copy" | "recolor";
  /**
   * action:"copy" payload: paste a region from `srcFrame` into each of `dstFrames`. Region = this
   * range's volumes evaluated ONCE at srcFrame (prepareRangeAt below — the SAME evaluator preview
   * and delete/keep bake use, not forked); mask2d volumes carry their own camera, so the region
   * re-projects onto any dst frame's mesh. `what` defaults to "both" when omitted:
   *   "both"   — replace dst triangles inside the region with srcFrame's, AND carry the region's
   *              texels along by RELOCATING them into free dst atlas space (charts translated
   *              whole where they fit, tiled where they don't; pasted UVs move with their texels
   *              — encoder texel-copy.ts). The correct mode for per-frame-packed atlases: this
   *              capture REPACKS its atlas every frame (measured: a same-coordinate copy is
   *              57.8% collateral at 30 frames' distance and 78.1% at one frame).
   *   "geo"    — geometry only, UVs unchanged: pasted triangles sample the DST atlas at their
   *              src coordinates. Only meaningful when the atlas layout is temporally coherent,
   *              which this capture's is NOT — the encoder warns loudly.
   *   "texels" — same-coordinate texel overwrite, no relocation. Gated at bake: the encoder
   *              counts dst triangles OUTSIDE the region that sample the patched area and
   *              refuses above 5% collateral (a per-frame-packed atlas fails by construction).
   */
  copy?: { srcFrame: number; dstFrames: number[]; what?: "geo" | "texels" | "both" };
  /**
   * action:"recolor" payload ("garment recolor", tier 1): a
   * deterministic, geometry-untouched masked retexture. Per frame f in [startFrame, endFrame],
   * the range's region is resolved via prepareRangeAt(r, f) — PER-FRAME keyframe interpolation
   * (unlike action:"copy", which locks the region to one srcFrame) — against that frame's OWN
   * mesh, selecting triangles by the same centroid rule delete uses; their UV footprint is
   * rasterized and recolored in that frame's OWN atlas (packages/encoder/src/recolor.ts).
   * `color`: "#rrggbb". `strength`: 0..1, default 0.8. `mode` default "tint":
   *   "tint" — move each masked texel toward `color` while PRESERVING its own luminance
   *            (fold/shading detail survives; a flat fill would be wrong for fabric).
   *   "hue"  — rotate each texel's hue toward `color`'s hue by `strength`, keeping its own
   *            saturation and lightness.
   */
  recolor?: { color: string; strength?: number; mode?: "tint" | "hue" };
  /**
   * Hole patching after deletion — a MODIFIER on a plain delete
   * range, not a new action: only meaningful when `(action ?? "delete") === "delete"` and
   * `mode === "delete"` (a copy/recolor/keep range carrying this field is a passive no-op,
   * same "excluded by construction" law as everything else non-delete). When present, the
   * ENCODER (bake-side only — see hole-patch.ts) detects the boundary loops this range's own
   * deletion opens in each frame it spans, caps them with a centroid-fan patch, and paints a
   * small solid-color tile for the new triangles' texels. `color`: "#rrggbb", used verbatim for
   * every capped loop's tile; omitted ⇒ each loop's fill is the AVERAGE of its own rim
   * vertices' live atlas texels (hole-patch.ts). The live player's preview is NOT extended for
   * this field (bake-side-first law, same as copy/recolor): scrubbing a patchHoles range shows
   * the open hole; only the baked .ares shows the cap.
   */
  patchHoles?: { color?: string };
  startFrame: number;
  endFrame: number;
  scrubTexels?: boolean;
  keyframes: EditKeyframe[];
}
export interface EditList {
  aresEdits: 1;
  source?: string;
  fps?: number;
  frameCount?: number;
  /**
   * Clip trim — NLE in/out points, INCLUSIVE frame indices in the SOURCE's own numbering. Unlike
   * every other field here, this does not select geometry within a frame; it selects which frames
   * exist at all. The encoder drops everything outside [in, out] and re-bases the ranges below onto
   * the surviving window (cli.ts `rebaseEditList`), so a trimmed bake's frame 0 is source frame
   * `in`. The live player mirrors it as its playback range (player.ts setTrim), which is what makes
   * the timeline's dimmed head/tail honest: what you play is what you ship.
   */
  trim?: { in: number; out: number };
  ranges: EditRange[];
}

/** Parse + validate an edit list (sorts keyframes, clamps ranges). Throws on malformed input. */
export function parseEditList(json: unknown): EditList {
  const o = json as EditList;
  if (!o || o.aresEdits !== 1 || !Array.isArray(o.ranges)) throw new Error("not an aresEdits v1 document");
  if (o.trim) {
    const { in: i, out } = o.trim;
    if (!Number.isInteger(i) || !Number.isInteger(out) || i < 0 || out < i)
      throw new Error(`bad trim {in:${i}, out:${out}} — expected integers with 0 <= in <= out`);
  }
  // A range the editor has *started* but not yet shaped has zero keyframes. The live preview treats
  // it as a no-op (prepareRangeAt → () => false), so the bake MUST agree instead of aborting the
  // whole encode. Drop empty-keyframe ranges rather than throw — this also avoids a latent trap: an
  // empty KEEP range would make every point "outside all keep-regions" and silently delete the frame.
  const ranges: EditRange[] = [];
  for (const r of o.ranges) {
    if (r.mode !== "delete" && r.mode !== "keep") throw new Error(`range ${r.id ?? "?"}: bad mode`);
    if (!Array.isArray(r.keyframes) || r.keyframes.length === 0) continue; // in-progress / no-op range
    r.keyframes.sort((a, b) => a.frame - b.frame);
    if (!Number.isFinite(r.startFrame) || !Number.isFinite(r.endFrame) || r.endFrame < r.startFrame)
      throw new Error(`range ${r.id ?? "?"}: bad start/end`);
    ranges.push(r);
  }
  o.ranges = ranges;
  return o;
}

/* ------------------------------- region evaluation ------------------------------- */

/** Signed distance to one brush volume: min over add-strokes of (|x−p|−r); subtract strokes carve. */
function brushSdf(v: BrushVolume, x: number, y: number, z: number): number {
  let d = Infinity, dSub = Infinity;
  for (const s of v.strokes) {
    for (const p of s.points) {
      const dist = Math.hypot(x - p[0]!, y - p[1]!, z - p[2]!) - s.radius;
      if (s.op === "subtract") { if (dist < dSub) dSub = dist; }
      else if (dist < d) d = dist;
    }
  }
  // inside = in an add-stroke and NOT in a subtract-stroke → d ∩ ¬dSub = max(d, −dSub)
  return dSub === Infinity ? d : Math.max(d, -dSub);
}

/** Signed distance to a box (negative inside). */
function boxSdf(v: BoxVolume, x: number, y: number, z: number): number {
  const dx = Math.max(v.min[0] - x, x - v.max[0]);
  const dy = Math.max(v.min[1] - y, y - v.max[1]);
  const dz = Math.max(v.min[2] - z, z - v.max[2]);
  return Math.max(dx, dy, dz);
}

type SdfFn = (x: number, y: number, z: number) => number;

/**
 * Compile one volume into an evaluator closure. mask2d precomputes its viewProj HERE — once per
 * frame evaluation, not once per point (20k centroids/frame would otherwise rebuild the matrix 20k×).
 * The mask2d "SDF" is in NDC units (sign is what matters; magnitude only shapes cross-type lerps).
 */
function prepareVolume(v: EditVolume): SdfFn | null {
  if (v.type === "box") return (x, y, z) => boxSdf(v, x, y, z);
  if (v.type === "brushStrokes") return (x, y, z) => brushSdf(v, x, y, z);
  if (v.type === "mask2d" && v.camera && (v.kind === "rect" ? v.rect : v.kind === "bitmap" && v.mask)) {
    const m = orbitViewProj(
      { azimuth: v.camera.azimuth, elevation: v.camera.elevation, distance: v.camera.distance, target: v.camera.target },
      v.camera.aspect || 1);
    const depth = v.depth;
    const r = v.rect;
    let bits: Uint8Array | null = null, mw = 0, mh = 0;
    if (v.kind === "bitmap" && v.mask) {
      mw = v.mask.width; mh = v.mask.height;
      bits = maskBitsCache.get(v.mask) ?? null;
      if (!bits) { bits = rleDecodeMask(v.mask.rle, mw * mh); maskBitsCache.set(v.mask, bits); }
    }
    return (x, y, z) => {
      const cw = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
      if (cw <= 1e-6) return 1;                             // behind the captured camera
      const inv = 1 / cw;
      const nx = (m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * inv;
      const ny = (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * inv;
      if (depth) {
        const nz = (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * inv;
        if (nz < depth.zmin || nz > depth.zmax) return 1;   // outside the visible-only depth band
      }
      if (bits) {
        // Bitmap lookup: the mask covers the full captured viewport (NDC [-1,1]², y flips to rows).
        const px = Math.floor(((nx + 1) / 2) * mw);
        const py = Math.floor(((1 - ny) / 2) * mh);
        if (px < 0 || py < 0 || px >= mw || py >= mh) return 1;
        return bits[py * mw + px] ? -0.5 : 0.5;             // binary in/out; sign drives the predicate
      }
      // rect layout is [x0, y0, x1, y1] (the field's documented contract, and what the marquee
      // authors) — this line previously read [x0, x1, y0, y1] and matched a garbage region.
      return Math.max(r![0] - nx, nx - r![2], r![1] - ny, ny - r![3]);
    };
  }
  return null; // unknown/unsupported volume — never matches
}

/** Compile a keyframe's volume UNION into one evaluator (negative = inside any volume). */
function prepareKeyframe(kf: EditKeyframe): SdfFn {
  const fns = kf.volumes.map(prepareVolume).filter((f): f is SdfFn => !!f);
  if (!fns.length) return () => Infinity;
  if (fns.length === 1) return fns[0]!;
  return (x, y, z) => {
    let d = Infinity;
    for (const f of fns) { const dv = f(x, y, z); if (dv < d) d = dv; }
    return d;
  };
}

/** The two keyframes bracketing frame f (hold semantics before-first / after-last). */
function bracket(r: EditRange, f: number): { a: EditKeyframe; b: EditKeyframe; t: number } | null {
  const kfs = r.keyframes;
  if (!kfs.length) return null;     // an authoring-in-progress range (started, nothing captured yet)
  const first = kfs[0]!, last = kfs[kfs.length - 1]!;
  if (f <= first.frame) return { a: first, b: first, t: 0 };
  if (f >= last.frame) return { a: last, b: last, t: 0 };
  for (let i = 0; i < kfs.length - 1; i++) {
    const a = kfs[i]!, b = kfs[i + 1]!;
    if (f >= a.frame && f <= b.frame) {
      return { a, b, t: b.frame === a.frame ? 0 : (f - a.frame) / (b.frame - a.frame) };
    }
  }
  return { a: last, b: last, t: 0 };
}

/**
 * Compile one range at frame f into an inside-test. Interpolation = SDF lerp between the
 * bracketing keyframes: d(x,t) = (1−t)·d_A(x) + t·d_B(x) (§6.2 — correspondence-free; for boxes
 * this closely tracks corner lerp; the design's rule for brush volumes; hold outside the span).
 * Exported so the encoder's frame-copy op (action:"copy") can resolve a range's region at one
 * fixed srcFrame and reuse it verbatim against other frames' meshes — same evaluator, not forked.
 */
export function prepareRangeAt(r: EditRange, f: number): (x: number, y: number, z: number) => boolean {
  const br = bracket(r, f);
  if (!br) return () => false;      // no keyframes yet → range matches nothing
  const { a, b, t } = br;
  const evA = prepareKeyframe(a);
  if (t === 0) return (x, y, z) => evA(x, y, z) < 0;
  const evB = prepareKeyframe(b);
  return (x, y, z) => (1 - t) * evA(x, y, z) + t * evB(x, y, z) < 0;
}

/** Single-point convenience form of prepareRangeAt (per-point use recompiles — prefer the compiled form in loops). */
export function insideRangeAt(r: EditRange, f: number, x: number, y: number, z: number): boolean {
  return prepareRangeAt(r, f)(x, y, z);
}

/**
 * Build the per-frame triangle-keep predicate over CENTROIDS (§10):
 * dropped iff inside any active delete region; when keep-ranges are active, also dropped iff
 * outside ALL of them (crop semantics). Returns null when no range is active at f (keep all).
 * All camera/bracket math is compiled ONCE here, then shared by every centroid test.
 */
export function keepPredicateAt(list: EditList, f: number): ((cx: number, cy: number, cz: number) => boolean) | null {
  // action:"copy" and action:"recolor" ranges are handled entirely by the encoder's own
  // bake-time passes (packages/encoder/src/frame-copy.ts + texel-copy.ts for copy,
  // packages/encoder/src/recolor.ts for recolor), BEFORE this function ever runs on the same
  // edit list — excluded here regardless of `mode` so neither the live preview nor the
  // encoder's own delete/keep sweep re-processes (and un-does or double-deletes) their region.
  // Gating on "is this a plain delete/keep range" (rather than "is this not a copy range") so
  // every future non-delete action is excluded by construction, not by an ever-growing
  // blocklist: behavior-identical to the old `r.action !== "copy"` for every value that existed
  // before today (undefined and "delete" both satisfy `(action ?? "delete") === "delete"` and
  // were already included; "copy" fails both forms and was already excluded).
  const isDeleteAction = (r: EditRange) => (r.action ?? "delete") === "delete";
  const del = list.ranges.filter((r) => isDeleteAction(r) && r.mode === "delete" && f >= r.startFrame && f <= r.endFrame).map((r) => prepareRangeAt(r, f));
  const keep = list.ranges.filter((r) => isDeleteAction(r) && r.mode === "keep" && f >= r.startFrame && f <= r.endFrame).map((r) => prepareRangeAt(r, f));
  if (!del.length && !keep.length) return null;
  return (cx, cy, cz) => {
    for (const inside of del) if (inside(cx, cy, cz)) return false;
    if (keep.length) {
      for (const inside of keep) if (inside(cx, cy, cz)) return true;
      return false;
    }
    return true;
  };
}

/**
 * Filter a triangle list by the predicate (world-space positions + centroid rule). Returns the
 * SAME array when nothing is dropped, else a compacted copy. Used by the player's live preview;
 * the encoder's bake additionally compacts vertices (crop.ts filterFrame).
 */
export function filterIndicesByPredicate(
  positions: { x(i: number): number; y(i: number): number; z(i: number): number },
  indices: Uint32Array,
  keep: (cx: number, cy: number, cz: number) => boolean,
): Uint32Array {
  const out = new Uint32Array(indices.length);
  let w = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]!, b = indices[t + 1]!, c = indices[t + 2]!;
    const cx = (positions.x(a) + positions.x(b) + positions.x(c)) / 3;
    const cy = (positions.y(a) + positions.y(b) + positions.y(c)) / 3;
    const cz = (positions.z(a) + positions.z(b) + positions.z(c)) / 3;
    if (keep(cx, cy, cz)) { out[w++] = a; out[w++] = b; out[w++] = c; }
  }
  return w === indices.length ? indices : out.subarray(0, w);
}
