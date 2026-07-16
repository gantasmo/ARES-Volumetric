/**
 * Model transform — the import-time orientation fix, shared by the live preview (player) and the
 * bake (encoder), so what you set in the viewport is exactly what the .ares gets. One evaluator,
 * never forked: the same law as the edit ranges (preview == bake).
 *
 * The problem it solves: captures do not agree on which way is up or where the origin is. A
 * Microsoft/SVF export is Y-up in millimetres; DCC and scanner exports are commonly Z-up; a
 * 4DViews bake is metres. Until a clip is upright and standing on the ground plane you cannot
 * compare two captures, read the grid against it, or trust a crop plane called "the floor".
 *
 * Order is fixed and matters: **upAxis, then rotate, then scale, then centre**. Centring is last
 * because it is defined on the RESULT — "put the feet on the floor" is meaningless before you know
 * which way up the feet are. `translate` is applied after centring, as a deliberate nudge away from
 * the centred result.
 */
import type { Mat4 } from "./camera.js";
import type { Aabb } from "./quant.js";

export type UpAxis = "x" | "y" | "z";
/**
 * Where the model's origin lands.
 *  - "bottom" (default): centred in X/Z, and its LOWEST point sits exactly on y = 0 — the subject
 *    stands on the ground plane. This is the default because the ground plane is the one reference
 *    every capture shares, and the grid is drawn at y = 0.
 *  - "mass": the AABB centre goes to the origin (the subject is centred through its middle).
 *  - "none": leave the source coordinates where they are.
 */
export type CenterMode = "bottom" | "mass" | "none";

export interface ModelTransform {
  /** Which axis of the SOURCE points up. Rotated so it becomes +Y (the runtime's up). Default "y". */
  upAxis?: UpAxis;
  /** Extra rotation in DEGREES, applied as X then Y then Z after the up-axis fix. */
  rotate?: [number, number, number];
  /** Uniform scale. Use it to reconcile units (e.g. 0.001 to take a millimetre capture to metres). */
  scale?: number;
  /** Nudge, in world units, applied AFTER centring. */
  translate?: [number, number, number];
  /** Where the origin lands. Default "bottom". */
  center?: CenterMode;
}

export const IDENTITY_TRANSFORM: Required<ModelTransform> = {
  upAxis: "y", rotate: [0, 0, 0], scale: 1, translate: [0, 0, 0], center: "none",
};

/** True when the transform would leave every vertex exactly where it is (so callers can skip work). */
export function isIdentityTransform(t: ModelTransform | null | undefined): boolean {
  if (!t) return true;
  const r = t.rotate ?? [0, 0, 0];
  return (t.upAxis ?? "y") === "y" && (t.scale ?? 1) === 1 && (t.center ?? "bottom") === "none"
    && r[0] === 0 && r[1] === 0 && r[2] === 0
    && (t.translate ?? [0, 0, 0]).every((v) => v === 0);
}

/* --------------------------------- rotation basis --------------------------------- */

/**
 * The 3×3 (row-major, as 9 numbers) that carries the source's up axis onto +Y.
 *
 * These are PROPER rotations (determinant +1), not axis swaps: a swap with determinant −1 mirrors
 * the model, which flips winding and turns a tattoo into its mirror image — the exact class of bug
 * the 4DViews X-mirror hunt chased.
 */
function upAxisBasis(up: UpAxis): number[] {
  switch (up) {
    // Z-up -> Y-up: rotate −90° about X. (x, y, z) -> (x, z, −y). det = +1.
    case "z": return [1, 0, 0, 0, 0, 1, 0, -1, 0];
    // X-up -> Y-up: rotate +90° about Z. (x, y, z) -> (−y, x, z). det = +1.
    // NOT (y, −x, z) — that is the −90° turn and lands the model UPSIDE DOWN (+X -> −Y).
    case "x": return [0, -1, 0, 1, 0, 0, 0, 0, 1];
    case "y": default: return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }
}

const mul3 = (a: number[], b: number[]): number[] => {
  const o = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    o[r * 3 + c] = a[r * 3]! * b[c]! + a[r * 3 + 1]! * b[3 + c]! + a[r * 3 + 2]! * b[6 + c]!;
  }
  return o;
};

/** Euler XYZ in degrees -> row-major 3×3. */
function eulerBasis(deg: [number, number, number]): number[] {
  const [rx, ry, rz] = deg.map((d) => (d * Math.PI) / 180) as [number, number, number];
  const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry), cz = Math.cos(rz), sz = Math.sin(rz);
  const X = [1, 0, 0, 0, cx, -sx, 0, sx, cx];
  const Y = [cy, 0, sy, 0, 1, 0, -sy, 0, cy];
  const Z = [cz, -sz, 0, sz, cz, 0, 0, 0, 1];
  return mul3(Z, mul3(Y, X));
}

/** The full rotation+scale basis (row-major 3×3), excluding translation. */
export function transformBasis(t: ModelTransform): number[] {
  const b = mul3(eulerBasis(t.rotate ?? [0, 0, 0]), upAxisBasis(t.upAxis ?? "y"));
  const s = t.scale ?? 1;
  return s === 1 ? b : b.map((v) => v * s);
}

/* --------------------------------- application --------------------------------- */

/** Apply a row-major 3×3 + offset to one point. */
const applyB = (b: number[], o: [number, number, number], x: number, y: number, z: number): [number, number, number] => [
  b[0]! * x + b[1]! * y + b[2]! * z + o[0],
  b[3]! * x + b[4]! * y + b[5]! * z + o[1],
  b[6]! * x + b[7]! * y + b[8]! * z + o[2],
];

/**
 * Resolve the offset that centring implies, given the ROTATED+SCALED bounds of the whole clip.
 *
 * Note "of the whole clip": the offset must be computed once over EVERY frame's bounds and reused
 * for all of them. Centring each frame against its own bounds would re-centre the subject every
 * frame — the model would slide around under its own motion, and a walk cycle would moonwalk in
 * place. This is why callers pass a clip-wide AABB, not a per-frame one.
 */
export function centerOffset(rotatedBounds: Aabb, mode: CenterMode): [number, number, number] {
  if (mode === "none") return [0, 0, 0];
  const cx = (rotatedBounds.min[0] + rotatedBounds.max[0]) / 2;
  const cz = (rotatedBounds.min[2] + rotatedBounds.max[2]) / 2;
  if (mode === "mass") {
    const cy = (rotatedBounds.min[1] + rotatedBounds.max[1]) / 2;
    return [-cx, -cy, -cz];
  }
  // "bottom": centred in X/Z, lowest point exactly on the ground plane.
  return [-cx, -rotatedBounds.min[1], -cz];
}

/** Rotate+scale an AABB by transforming its 8 corners (a rotated box is not axis-aligned). */
export function transformAabb(box: Aabb, t: ModelTransform): Aabb {
  const b = transformBasis(t);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 8; i++) {
    const p = applyB(b, [0, 0, 0],
      i & 1 ? box.max[0] : box.min[0],
      i & 2 ? box.max[1] : box.min[1],
      i & 4 ? box.max[2] : box.min[2]);
    for (let k = 0; k < 3; k++) { if (p[k]! < min[k]!) min[k] = p[k]!; if (p[k]! > max[k]!) max[k] = p[k]!; }
  }
  return { min, max };
}

/**
 * The complete offset for a transform: centring (against the clip-wide rotated bounds) plus the
 * user's translate nudge. Compute once per clip, pass to every applyTransform call.
 */
export function resolveOffset(clipBounds: Aabb, t: ModelTransform): [number, number, number] {
  const rotated = transformAabb(clipBounds, t);
  const c = centerOffset(rotated, t.center ?? "bottom");
  const tr = t.translate ?? [0, 0, 0];
  return [c[0] + tr[0], c[1] + tr[1], c[2] + tr[2]];
}

/** Apply a transform to a flat xyz position array, in place. `offset` comes from resolveOffset. */
export function applyTransform(positions: Float32Array, t: ModelTransform, offset: [number, number, number]): Float32Array {
  const b = transformBasis(t);
  for (let i = 0; i < positions.length; i += 3) {
    const p = applyB(b, offset, positions[i]!, positions[i + 1]!, positions[i + 2]!);
    positions[i] = p[0]; positions[i + 1] = p[1]; positions[i + 2] = p[2];
  }
  return positions;
}

/**
 * Normals rotate but must NOT translate, and must not carry the scale (a uniform scale leaves
 * directions unchanged once renormalized; carrying it would just denormalize them).
 */
export function applyTransformNormals(normals: Float32Array, t: ModelTransform): Float32Array {
  const b = transformBasis({ upAxis: t.upAxis, rotate: t.rotate });   // rotation only, scale 1
  for (let i = 0; i < normals.length; i += 3) {
    const p = applyB(b, [0, 0, 0], normals[i]!, normals[i + 1]!, normals[i + 2]!);
    const l = Math.hypot(p[0], p[1], p[2]) || 1;
    normals[i] = p[0] / l; normals[i + 1] = p[1] / l; normals[i + 2] = p[2] / l;
  }
  return normals;
}

/** Column-major mat4 (the renderer's convention) for the live preview. */
export function transformMatrix(t: ModelTransform, offset: [number, number, number]): Mat4 {
  const b = transformBasis(t);
  const m = new Float32Array(16);
  // row-major 3x3 -> column-major 4x4
  m[0] = b[0]!; m[1] = b[3]!; m[2] = b[6]!;
  m[4] = b[1]!; m[5] = b[4]!; m[6] = b[7]!;
  m[8] = b[2]!; m[9] = b[5]!; m[10] = b[8]!;
  m[12] = offset[0]; m[13] = offset[1]; m[14] = offset[2];
  m[15] = 1;
  return m;
}
