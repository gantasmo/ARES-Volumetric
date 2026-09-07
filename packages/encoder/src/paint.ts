/**
 * Texture paint, build-order item 1 of the sculpt+paint plan (docs/reports/2026-07-16-sculpt-
 * paint-plan.md §B): a TEXEL-granular soft brush over the range's world-anchored region.
 *
 * Recolor (recolor.ts) is the hard-edged, triangle-granular ancestor: it selects whole
 * triangles by centroid and retints their entire UV footprint. Paint differs in exactly one
 * structural way — each texel gets a WEIGHT from the range's interpolated signed distance,
 * evaluated at the texel's own world position (barycentric interpolation of the triangle's
 * world vertices over its UV footprint), smoothstepped over a `feather` distance. Everything
 * else deliberately mirrors recolor: per-frame region resolution against that frame's OWN mesh
 * and OWN atlas (world-anchored strokes survive per-frame atlas repacks — the editor-v3
 * frame-copy lesson: NEVER same-coordinate texel writes), same complement-occupancy-guarded
 * 2 px dilation so chart borders don't halo, same bake-time application point.
 *
 * Brushes v1 (plan §B): "tint" (luma-preserving recolorPixel, scaled by strength×weight) and
 * "heal" (box-blur of the frame's own pre-paint atlas, mixed by strength×weight).
 */
import type { EditRange } from "@ares/core";
import type { EncodeMeshFrame } from "./geometry-encode.js";
import { parseHexColor, recolorPixel } from "./recolor.js";

export interface PaintOp {
  range: EditRange;
  startFrame: number;
  endFrame: number;
  brush: "tint" | "heal";
  color: [number, number, number] | null; // null for heal
  strength: number;
  feather: number; // mm (world units); falloff half-width outside the region surface
}

/** Pull action:"paint" ranges out of an edit list into concrete, validated ops. */
export function collectPaintOps(ranges: EditRange[], frameCount: number): PaintOp[] {
  const ops: PaintOp[] = [];
  for (const r of ranges) {
    if (r.action !== "paint") continue;
    const p = r.paint ?? {};
    const brush = p.brush ?? "tint";
    if (brush !== "tint" && brush !== "heal") throw new Error(`range ${r.id ?? "?"}: paint.brush must be "tint" or "heal", got "${String(brush)}"`);
    const color = brush === "tint" ? parseHexColor(p.color ?? "", r.id) : null;
    const strength = p.strength ?? 0.8;
    if (!(strength >= 0 && strength <= 1)) throw new Error(`range ${r.id ?? "?"}: paint.strength ${p.strength} out of [0,1]`);
    // Default feather: half the mean stroke radius of the range's brush volumes, else 20 mm.
    let feather = p.feather ?? 0;
    if (!(feather > 0)) {
      let rs = 0, n = 0;
      for (const kf of r.keyframes) for (const v of kf.volumes) {
        if (v.type === "brushStrokes") for (const s of v.strokes) { rs += s.radius; n++; }
      }
      feather = n ? (rs / n) * 0.5 : 20;
    }
    if (r.startFrame >= frameCount) continue; // past a --max-frames truncation — no-op (recolor's rule)
    ops.push({ range: r, startFrame: Math.max(0, r.startFrame), endFrame: Math.min(r.endFrame, frameCount - 1), brush, color, strength, feather });
  }
  return ops;
}

/**
 * Rasterize one frame's triangles into a per-texel WEIGHT map for a paint op.
 * For each triangle whose centroid lies within `feather` of the region (sdf < feather), its UV
 * footprint is scanned; each covered texel's world position is barycentrically interpolated and
 * weighted w = smoothstep over [-feather, +feather] of the sdf (1 deep inside, 0 at the outer
 * feather edge). Returns null when nothing is touched. The weight map is the paint analogue of
 * recolor's binary mask.
 */
export function rasterizePaintWeights(
  frame: Pick<EncodeMeshFrame, "positions" | "uvs" | "indices">,
  sdf: (x: number, y: number, z: number) => number,
  width: number,
  height: number,
  feather: number,
): Float32Array | null {
  const { positions, uvs, indices } = frame;
  if (!uvs) return null;
  const weights = new Float32Array(width * height);
  let touched = false;
  const wOf = (d: number) => {
    // smoothstep from +feather (0) down to -feather (1)
    const t = Math.max(0, Math.min(1, (feather - d) / (2 * feather)));
    return t * t * (3 - 2 * t);
  };
  for (let t = 0; t < indices.length; t += 3) {
    const ia = indices[t]!, ib = indices[t + 1]!, ic = indices[t + 2]!;
    const a3 = ia * 3, b3 = ib * 3, c3 = ic * 3;
    const cx = (positions[a3]! + positions[b3]! + positions[c3]!) / 3;
    const cy = (positions[a3 + 1]! + positions[b3 + 1]! + positions[c3 + 1]!) / 3;
    const cz = (positions[a3 + 2]! + positions[b3 + 2]! + positions[c3 + 2]!) / 3;
    if (sdf(cx, cy, cz) >= feather) continue; // whole triangle beyond the outer falloff — skip fast
    const a2 = ia * 2, b2 = ib * 2, c2 = ic * 2;
    const ax = uvs[a2]! * width, ay = uvs[a2 + 1]! * height;
    const bx = uvs[b2]! * width, by = uvs[b2 + 1]! * height;
    const cxx = uvs[c2]! * width, cyy = uvs[c2 + 1]! * height;
    const minX = Math.max(0, Math.floor(Math.min(ax, bx, cxx)));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(ax, bx, cxx)));
    const minY = Math.max(0, Math.floor(Math.min(ay, by, cyy)));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(ay, by, cyy)));
    const det = (bx - ax) * (cyy - ay) - (cxx - ax) * (by - ay);
    if (det === 0) continue;
    const inv = 1 / det;
    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const qx = px + 0.5, qy = py + 0.5;
        const w1 = ((bx - qx) * (cyy - qy) - (cxx - qx) * (by - qy)) * inv;
        const w2 = ((cxx - qx) * (ay - qy) - (ax - qx) * (cyy - qy)) * inv;
        const w3 = 1 - w1 - w2;
        const eps = -1e-4;
        if (w1 < eps || w2 < eps || w3 < eps) continue;
        const wx = w1 * positions[a3]! + w2 * positions[b3]! + w3 * positions[c3]!;
        const wy = w1 * positions[a3 + 1]! + w2 * positions[b3 + 1]! + w3 * positions[c3 + 1]!;
        const wz = w1 * positions[a3 + 2]! + w2 * positions[b3 + 2]! + w3 * positions[c3 + 2]!;
        const w = wOf(sdf(wx, wy, wz));
        if (w <= 0) continue;
        const p = py * width + px;
        if (w > weights[p]!) { weights[p] = w; touched = true; }
      }
    }
  }
  return touched ? weights : null;
}

/**
 * Extend the weight map's edge by up to `radius` px (chart-border anti-halo, recolor's dilation
 * rule): a zero-weight texel adjacent to painted texels inherits their max weight, EXCEPT where
 * `blockedMask` (the complement footprint — texels a foreign chart actually samples) is set.
 */
export function dilatePaintWeights(
  weights: Float32Array, w: number, h: number, radius: number, blockedMask: Uint8Array,
): void {
  const src = Float32Array.from(weights);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      if (src[p]! > 0 || blockedMask[p]) continue;
      let best = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const v = src[yy * w + xx]!;
          if (v > best) best = v;
        }
      }
      if (best > 0) weights[p] = best;
    }
  }
}

/** One frame's queued paint patch, applied by texel-copy.ts's shared atlas pass. */
export interface PaintPatch {
  weights: Float32Array; // width*height, 0..1
  brush: "tint" | "heal";
  color: [number, number, number] | null;
  strength: number;
  rangeId?: string;
}

const HEAL_RADIUS = 3; // px box-blur half-width for the heal brush

/** Apply one paint patch to a decoded RGBA atlas in place (alpha untouched). Heal samples a
 *  SNAPSHOT of the incoming image so already-healed texels don't feed later ones. */
export function applyPaintToImage(data: Uint8Array, patch: PaintPatch, width: number, height: number): void {
  const { weights, brush, color, strength } = patch;
  const snap = brush === "heal" ? Uint8Array.from(data) : null;
  for (let p = 0; p < weights.length; p++) {
    const w = weights[p]!;
    if (w <= 0) continue;
    const s = strength * w;
    const o = p * 4;
    if (brush === "tint") {
      const [r, g, b] = recolorPixel(data[o]!, data[o + 1]!, data[o + 2]!, color!, s, "tint");
      data[o] = Math.max(0, Math.min(255, Math.round(r)));
      data[o + 1] = Math.max(0, Math.min(255, Math.round(g)));
      data[o + 2] = Math.max(0, Math.min(255, Math.round(b)));
    } else {
      const x = p % width, y = (p / width) | 0;
      let sr = 0, sg = 0, sb = 0, n = 0;
      for (let dy = -HEAL_RADIUS; dy <= HEAL_RADIUS; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -HEAL_RADIUS; dx <= HEAL_RADIUS; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          // blur only over texels that are themselves painted-or-inside a chart the brush saw:
          // weights>0 texels are all on the region's own charts, so foreign charts never bleed in
          if (weights[yy * width + xx]! <= 0) continue;
          const q = (yy * width + xx) * 4;
          sr += snap![q]!; sg += snap![q + 1]!; sb += snap![q + 2]!; n++;
        }
      }
      if (!n) continue;
      data[o] = Math.round(data[o]! + (sr / n - data[o]!) * s);
      data[o + 1] = Math.round(data[o + 1]! + (sg / n - data[o + 1]!) * s);
      data[o + 2] = Math.round(data[o + 2]! + (sb / n - data[o + 2]!) * s);
    }
  }
}
