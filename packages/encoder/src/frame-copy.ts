/**
 * Frame-to-frame copy — geometry half ("fix a bad frame from a good one"). Bake-side only; the texel half lives in texel-copy.ts (it needs the atlas PNGs,
 * which this module never touches).
 *
 * Region = one EditRange's volumes evaluated ONCE at srcFrame via @ares/core's prepareRangeAt —
 * the SAME evaluator the live preview and the delete/keep bake use (mask2d volumes carry their
 * own camera, so a region masked on one frame projects onto any other frame's mesh; box volumes
 * are just as portable since the SDF is pure world-space). Never forked here.
 *
 * Geo copy per dst frame: delete dst triangles inside the region, then concatenate the srcFrame's
 * inside-region fragment (both halves via crop.ts filterFrame — the same select+compact pass
 * decimate.ts's compaction mirrors). Runs BEFORE the delete/keep sweep and before decimate
 * (cli.ts), on frames as imported. For what:"both" the paste is a buildPieceFragment rebuild
 * whose UVs follow the texels' relocation into free dst atlas space (v1.1 — this capture's
 * atlas is repacked every frame, so unchanged UVs would sample unrelated dst content; see
 * texel-copy.ts header for the measured numbers).
 */
import { prepareRangeAt, type EditRange } from "@ares/core";
import { filterFrame } from "./crop.js";
import type { EncodeMeshFrame } from "./geometry-encode.js";

export interface CopyOp {
  range: EditRange;
  srcFrame: number;
  dstFrames: number[];
  what: "geo" | "texels" | "both";
}

/** Pull action:"copy" ranges out of an edit list into concrete, bounds-checked ops. Throws on a
 *  malformed copy payload (out-of-range frame refs) — same throw-on-bad-input style as
 *  parseCropBox/parseEditList; a hand-authored test sidecar with a typo'd frame index should
 *  fail loudly rather than silently no-op. */
export function collectCopyOps(ranges: EditRange[], frameCount: number): CopyOp[] {
  const ops: CopyOp[] = [];
  for (const r of ranges) {
    if (r.action !== "copy") continue;
    const c = r.copy;
    if (!c) throw new Error(`range ${r.id ?? "?"}: action "copy" requires a "copy" payload`);
    if (!Number.isInteger(c.srcFrame) || c.srcFrame < 0 || c.srcFrame >= frameCount) {
      throw new Error(`range ${r.id ?? "?"}: copy.srcFrame ${c.srcFrame} out of range [0, ${frameCount})`);
    }
    if (!Array.isArray(c.dstFrames) || !c.dstFrames.length) {
      throw new Error(`range ${r.id ?? "?"}: copy.dstFrames must be a non-empty array`);
    }
    for (const d of c.dstFrames) {
      if (!Number.isInteger(d) || d < 0 || d >= frameCount) {
        throw new Error(`range ${r.id ?? "?"}: copy.dstFrames entry ${d} out of range [0, ${frameCount})`);
      }
    }
    ops.push({ range: r, srcFrame: c.srcFrame, dstFrames: c.dstFrames.slice(), what: c.what ?? "both" });
  }
  return ops;
}

/** The region resolved to a concrete triangle fragment of srcFrame's OWN mesh (crop.ts
 *  filterFrame — select + compact, reused not forked). Shared by the geo-copy path (paste these
 *  triangles) and texel-copy.ts (rasterize this fragment's UV footprint); both must see the
 *  identical selection, so this is the one place that selection is computed. */
export function resolveSrcFragment(frames: EncodeMeshFrame[], op: CopyOp): EncodeMeshFrame {
  const inside = prepareRangeAt(op.range, op.srcFrame);
  return filterFrame(frames[op.srcFrame]!, (cx, cy, cz) => inside(cx, cy, cz));
}

/**
 * Rebuild a paste fragment from relocation pieces (v1.1 what:"both" — texel-copy.ts
 * planRegionRelocation): each piece's triangles become a compact sub-mesh whose UVs are
 * uniformly displaced by the piece's atlas translation (dx/w, dy/h). A vertex shared by
 * triangles of two pieces is DUPLICATED, one copy per piece — per-vertex offsets cannot point
 * one vertex at two destinations, and a triangle whose corners moved by different offsets would
 * shear its sampled region. Triangle count is preserved exactly (only vertices duplicate).
 */
export function buildPieceFragment(
  fragment: EncodeMeshFrame, pieces: { tris: number[]; dx: number; dy: number }[], w: number, h: number,
): EncodeMeshFrame {
  const { positions, uvs, indices } = fragment;
  const outPos: number[] = [], outUv: number[] = [], outIdx: number[] = [];
  for (const p of pieces) {
    const remap = new Map<number, number>();
    const du = p.dx / w, dv = p.dy / h;
    for (const t of p.tris) {
      for (let k = 0; k < 3; k++) {
        const v = indices[t + k]!;
        let nv = remap.get(v);
        if (nv === undefined) {
          nv = outPos.length / 3;
          outPos.push(positions[v * 3]!, positions[v * 3 + 1]!, positions[v * 3 + 2]!);
          if (uvs) outUv.push(uvs[v * 2]! + du, uvs[v * 2 + 1]! + dv);
          remap.set(v, nv);
        }
        outIdx.push(nv);
      }
    }
  }
  return { positions: new Float32Array(outPos), uvs: uvs ? new Float32Array(outUv) : undefined, indices: new Uint32Array(outIdx) };
}

/** Concatenate two ALREADY-COMPACT frames (crop.ts filterFrame output on both sides) with a
 *  plain vertex-index offset. No further remap/compaction pass is needed: neither side
 *  references the other's vertices, so the union is compact by construction (this is the
 *  "delete, concat, compact" sequence collapsed into two filterFrame calls
 *  that already compact + one cheap concatenation, rather than a third redundant compaction). */
function concatFrames(a: EncodeMeshFrame, b: EncodeMeshFrame): EncodeMeshFrame {
  const av = a.positions.length / 3, bv = b.positions.length / 3;
  const positions = new Float32Array((av + bv) * 3);
  positions.set(a.positions, 0);
  positions.set(b.positions, av * 3);
  let uvs: Float32Array | undefined;
  if (a.uvs || b.uvs) {
    uvs = new Float32Array((av + bv) * 2);
    if (a.uvs) uvs.set(a.uvs, 0);
    if (b.uvs) uvs.set(b.uvs, av * 2);
  }
  const indices = new Uint32Array(a.indices.length + b.indices.length);
  indices.set(a.indices, 0);
  for (let i = 0; i < b.indices.length; i++) indices[a.indices.length + i] = b.indices[i]! + av;
  return { positions, uvs, indices };
}

export interface GeoCopyStat { frame: number; before: number; after: number; }

export interface GeoCopyOptions {
  /** Pre-resolved src fragment (resolveSrcFragment output for this op) — pass the SAME object
   *  the texel half planned from, so both halves are guaranteed one selection. */
  fragment?: EncodeMeshFrame;
  /** Per-dst-frame replacement paste (v1.1 what:"both": buildPieceFragment output whose UVs
   *  follow that dst frame's texel relocation). Falls back to the plain fragment when absent —
   *  what:"geo" and hypothetically-coherent-atlas captures. */
  pasteFragments?: Map<number, EncodeMeshFrame>;
}

/**
 * Apply the geometry half of one copy op across its dst frames. Mutates `frames` in place
 * (replaces each dst entry); returns per-dst before/after triangle counts plus the srcFrame
 * fragment's triangle count (0 means the region matched nothing at srcFrame — dst frames only
 * had the region deleted, nothing pasted back; caller should warn).
 */
export function applyGeoCopy(frames: EncodeMeshFrame[], op: CopyOp, opts?: GeoCopyOptions): { stats: GeoCopyStat[]; srcTris: number } {
  const inside = prepareRangeAt(op.range, op.srcFrame);
  const srcFragment = opts?.fragment ?? resolveSrcFragment(frames, op);
  const stats: GeoCopyStat[] = [];
  for (const d of op.dstFrames) {
    const before = frames[d]!.indices.length / 3;
    const kept = filterFrame(frames[d]!, (cx, cy, cz) => !inside(cx, cy, cz));
    frames[d] = concatFrames(kept, opts?.pasteFragments?.get(d) ?? srcFragment);
    stats.push({ frame: d, before, after: frames[d]!.indices.length / 3 });
  }
  return { stats, srcTris: srcFragment.indices.length / 3 };
}
