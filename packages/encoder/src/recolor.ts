/**
 * Garment recolor, tier 1 ("garment change / masked retexture").
 * Bake-side only, geometry-untouched: a deterministic masked retint/hue-shift of a region's
 * texels, per frame. This proves the texel-resolution pipeline the future generative tier 2
 * (masked SD img2img replace) will ride, without any of that tier's atlas-coherence problems.
 *
 * Unlike action:"copy" (which locks its region to ONE srcFrame and pastes into other frames'
 * meshes/atlases), recolor resolves the range's region at EVERY frame it spans via
 * prepareRangeAt(r, f) — per-frame keyframe interpolation, the same evaluator preview and the
 * delete/keep bake use — against that frame's OWN mesh, and recolors texels in that frame's OWN
 * atlas. There is no cross-frame atlas-coherence assumption to worry about (texel-copy.ts's
 * relocation machinery exists ONLY because a copy pastes one frame's texels into another
 * frame's differently-packed atlas); recolor never crosses a frame boundary, so it is not
 * needed here and is deliberately not used.
 *
 * This module owns: the edit-list → concrete-op parsing (mirrors frame-copy.ts's
 * collectCopyOps), the pure triangle-by-centroid partition used to build the per-frame mask
 * (mirrors crop.ts's filterFrame pass 1 exactly — same rule, so the SAME triangles a `delete`
 * range would drop are the ones recolor treats as "selected"), and the pixel color math. Mask
 * RASTERIZATION reuses texel-copy.ts's rasterizeUvFootprint/dilateMask (not duplicated here);
 * that orchestration lives in cli.ts, in the same place the copy op's footprint is planned, so
 * this file has no reason to import texel-copy.ts (avoiding a needless import cycle — texel-
 * copy.ts imports the pixel-math half of THIS file to apply patches inside its one shared
 * scratch-dir pass).
 */
import type { EditRange } from "@ares/core";
import type { EncodeMeshFrame } from "./geometry-encode.js";

export interface RecolorOp {
  range: EditRange;
  startFrame: number;
  endFrame: number;
  color: [number, number, number];
  strength: number;
  mode: "tint" | "hue";
}

/** Parse "#rrggbb" (leading # optional) to 0-255 ints. Throws loudly on a malformed color —
 *  same fail-fast style as collectCopyOps's frame-index checks (a typo'd sidecar should not
 *  silently no-op or produce garbage). */
export function parseHexColor(s: string, id?: string): [number, number, number] {
  const m = /^#?([0-9a-fA-F]{6})$/.exec(s ?? "");
  if (!m) throw new Error(`range ${id ?? "?"}: recolor.color "${s}" is not a "#rrggbb" hex color`);
  const n = parseInt(m[1]!, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function hexOf(color: [number, number, number]): string {
  return "#" + color.map((c) => Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, "0")).join("");
}

/** Pull action:"recolor" ranges out of an edit list into concrete, validated ops. */
export function collectRecolorOps(ranges: EditRange[], frameCount: number): RecolorOp[] {
  const ops: RecolorOp[] = [];
  for (const r of ranges) {
    if (r.action !== "recolor") continue;
    const c = r.recolor;
    if (!c) throw new Error(`range ${r.id ?? "?"}: action "recolor" requires a "recolor" payload`);
    const color = parseHexColor(c.color, r.id);
    const strength = c.strength ?? 0.8;
    if (!(strength >= 0 && strength <= 1)) throw new Error(`range ${r.id ?? "?"}: recolor.strength ${c.strength} out of [0,1]`);
    const mode = c.mode ?? "tint";
    if (mode !== "tint" && mode !== "hue") throw new Error(`range ${r.id ?? "?"}: recolor.mode must be "tint" or "hue", got "${String(mode)}"`);
    // Clamp endFrame into the imported frame count (mirrors collectCopyOps' bounds-checking
    // intent) rather than throw: --max-frames legitimately truncates a sidecar authored
    // against the full clip, and a range merely trailing off the truncated tail is not an
    // authoring error the way an out-of-range copy.srcFrame/dstFrames index would be.
    if (r.startFrame >= frameCount) continue; // entirely past the truncated frame count — no-op
    ops.push({ range: r, startFrame: Math.max(0, r.startFrame), endFrame: Math.min(r.endFrame, frameCount - 1), color, strength, mode });
  }
  return ops;
}

/** Split one frame's triangles by the delete-style centroid rule (crop.ts filterFrame pass 1,
 *  not forked — same three-point average, same "inside ⇒ selected" sense). Returns two flat
 *  vertex-index arrays directly usable as an EncodeMeshFrame.indices (no compaction: rasterizing
 *  a UV footprint only reads uvs[index], so leaving positions/uvs un-compacted is harmless and
 *  avoids a needless remap pass here). */
export function partitionTrianglesByCentroid(
  frame: Pick<EncodeMeshFrame, "positions" | "indices">,
  inside: (x: number, y: number, z: number) => boolean,
): { sel: Uint32Array; other: Uint32Array } {
  const { positions, indices } = frame;
  const sel: number[] = [], other: number[] = [];
  for (let t = 0; t < indices.length; t += 3) {
    const ia = indices[t]!, ib = indices[t + 1]!, ic = indices[t + 2]!;
    const a = ia * 3, b = ib * 3, c = ic * 3;
    const cx = (positions[a]! + positions[b]! + positions[c]!) / 3;
    const cy = (positions[a + 1]! + positions[b + 1]! + positions[c + 1]!) / 3;
    const cz = (positions[a + 2]! + positions[b + 2]! + positions[c + 2]!) / 3;
    const bucket = inside(cx, cy, cz) ? sel : other;
    bucket.push(ia, ib, ic);
  }
  return { sel: Uint32Array.from(sel), other: Uint32Array.from(other) };
}

/* ------------------------------- pixel color math ------------------------------- */

/** Standard RGB[0,255] -> HSL(h,s,l each 0..1). */
function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const max = Math.max(rn, gn, bn), min = Math.min(rn, gn, bn);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h: number;
  if (max === rn) h = (gn - bn) / d + (gn < bn ? 6 : 0);
  else if (max === gn) h = (bn - rn) / d + 2;
  else h = (rn - gn) / d + 4;
  return [h / 6, s, l];
}

function hue2rgb(p: number, q: number, t: number): number {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}

/** HSL(h,s,l each 0..1) -> RGB[0,255] (unrounded — caller rounds once at the end). */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) { const v = l * 255; return [v, v, v]; }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3) * 255, hue2rgb(p, q, h) * 255, hue2rgb(p, q, h - 1 / 3) * 255];
}

/** BT.601 luma [0,1] — the perceptual-brightness axis "tint" mode preserves so fabric folds
 *  and shading survive a recolor instead of washing out to a flat fill. */
function luma601(r: number, g: number, b: number): number { return (0.299 * r + 0.587 * g + 0.114 * b) / 255; }

/**
 * Recolor one texel toward `target` by `strength` (task spec, editor v3 §2):
 *   "tint" — rebuild `target`'s hue/saturation AT THIS TEXEL'S OWN LUMA (so the reconstructed
 *            color carries the texel's own brightness, not the target's), then linearly mix
 *            with the original by `strength`. At strength 1 the texel becomes exactly that
 *            luma-preserving reconstruction, never a flat fill, because luma is read from the
 *            texel every time.
 *   "hue"  — rotate the texel's own hue toward target's hue by `strength` (shortest way round
 *            the hue circle), keeping the texel's own saturation and lightness untouched.
 */
export function recolorPixel(
  r: number, g: number, b: number, target: [number, number, number], strength: number, mode: "tint" | "hue",
): [number, number, number] {
  if (mode === "hue") {
    const [h0, s0, l0] = rgbToHsl(r, g, b);
    const [ht] = rgbToHsl(target[0], target[1], target[2]);
    let d = ht - h0;
    if (d > 0.5) d -= 1;
    if (d < -0.5) d += 1;
    const h = ((h0 + d * strength) % 1 + 1) % 1;
    return hslToRgb(h, s0, l0);
  }
  const l = luma601(r, g, b);
  const [ht, st] = rgbToHsl(target[0], target[1], target[2]);
  const [tr, tg, tb] = hslToRgb(ht, st, l);
  return [r + (tr - r) * strength, g + (tg - g) * strength, b + (tb - b) * strength];
}

/** One frame's queued recolor patch: `mask` (width*height, row-major, from cli.ts's planning —
 *  the selected footprint plus complement-occupancy-masked dilation) selects which pixels of
 *  `data` (tightly-packed RGBA, mutated in place) get recolorPixel applied; alpha is untouched. */
export interface RecolorPatch {
  mask: Uint8Array;
  color: [number, number, number];
  strength: number;
  mode: "tint" | "hue";
  rangeId?: string;
}

/** Apply one recolor patch to a decoded atlas image in place. Several patches (overlapping
 *  recolor ranges, or recolor over an already copy-blitted frame) apply in sequence, each
 *  painting over whatever the previous step left — the caller (texel-copy.ts
 *  buildPatchedAtlasDir) is responsible for the copy-then-recolor ordering. */
export function applyRecolorToImage(data: Uint8Array, patch: RecolorPatch): void {
  const { mask, color, strength, mode } = patch;
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p]) continue;
    const o = p * 4;
    const [r, g, b] = recolorPixel(data[o]!, data[o + 1]!, data[o + 2]!, color, strength, mode);
    data[o] = Math.max(0, Math.min(255, Math.round(r)));
    data[o + 1] = Math.max(0, Math.min(255, Math.round(g)));
    data[o + 2] = Math.max(0, Math.min(255, Math.round(b)));
  }
}
