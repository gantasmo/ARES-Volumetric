/**
 * Hole patching after deletion. Bake-side only, a MODIFIER on plain
 * delete ranges (`patchHoles` in @ares/core edits.ts), not a new action.
 *
 * Per frame f a `patchHoles` range spans, AFTER the delete/keep sweep has already dropped that
 * frame's triangles (cli.ts runs this block right after that sweep, before decimate — a footprint
 * planned from already-decimated geometry would no longer match what gets pasted/capped, the same
 * reasoning frame-copy.ts and recolor.ts's planning follow):
 *
 *   1. Weld the frame's vertices by EXACT position bits (weldByPosition below — the same rule
 *      temporal.ts's taubinWelded and geometry-encode.ts's computeSmoothNormals use, not a new
 *      tolerance scheme). CRITICAL: this capture's atlas duplicates vertices along every UV-chart
 *      seam (~1,200 seam groups measured), so an edge/adjacency pass over the RAW index buffer
 *      reads every seam as a false boundary (each side sees only its own duplicate's triangles).
 *      Welding first makes every duplicate act as ONE vertex, so seams disappear from the
 *      boundary-edge count and only genuine open edges (the new hole, or the capture's
 *      pre-existing scan gaps) remain.
 *   2. Find boundary loops: an undirected welded edge used by exactly one triangle is a boundary
 *      edge; walk the boundary-edge graph into simple cycles. A welded vertex with boundary-degree
 *      != 2 is a non-manifold junction — every loop that would cross it is bailed (not patched),
 *      with a console.warn naming the frame; loops that don't touch a junction are unaffected.
 *   3. Filter to loops the DELETE region actually opened (not the capture's own pre-existing open
 *      boundaries — the floor cut at the feet, scan gaps — which must never be patched): a loop
 *      passes if its centroid tests inside the range's region (prepareRangeAt, per-frame
 *      interpolated) OR at least one of its rim vertices individually does (see
 *      detectHoleLoopsForRange's own doc comment for the measured tuning story — a >50%-majority
 *      version was tried first and turned out to be fragile). The OR-with-centroid is the fix for
 *      the polarity trap — after deletion a hole's rim vertices belong to SURVIVING triangles just
 *      outside the deleted region (the centroid rule decided survival, not any per-vertex test), so
 *      individual rim vertices can straddle the region boundary either way; the loop's centroid
 *      (near the true cut surface) is the more reliable single signal, with "any rim vertex inside"
 *      as a fallback when the centroid lands exactly on/outside the region's surface.
 *   4. Cap: one new apex vertex per accepted loop at its centroid, fan-triangulated. Winding: each
 *      boundary edge has exactly one owning triangle, which traverses it in SOME direction (u,v);
 *      the "missing" partner triangle a closed edge would have had always traverses the shared
 *      edge in the OPPOSITE direction in a consistently-wound mesh, so the cap triangle uses
 *      (v, u, apex) — the reversed direction — to face outward like its neighbors.
 *   5. Texel patch: cap vertices (the apex AND duplicated copies of the rim vertices — see
 *      appendCaps) all get the SAME uv, the center of a small allocated tile, so the whole cap
 *      renders as one flat color regardless of triangle interpolation. Tile allocation (free-space
 *      scan) happens HERE at planning time (needs only UV footprint rasterization, not decoded
 *      pixels); the actual pixel fill happens later, inside texel-copy.ts's buildPatchedAtlasDir,
 *      the one shared scratch-dir decode/re-encode pass copy and recolor already use (order:
 *      copy blit -> recolor -> hole-tile fill, so a patch can sample a just-recolored rim texel).
 */
import { prepareRangeAt, type EditRange } from "@ares/core";
import type { EncodeMeshFrame } from "./geometry-encode.js";

/** Tile size (atlas px) for one capped loop's flat-fill patch. Small and constant (task spec):
 *  every cap vertex samples its tile's CENTER, so 16px gives generous bilinear margin around that
 *  one effective texel while staying cheap to place in a ~60%-occupied atlas. */
export const HOLE_TILE = 16;

/** Weld vertices by EXACT position bits (house rule — see temporal.ts's taubinWelded, not
 *  duplicated here because that function is GOP-smoothing-specific and private to temporal.ts;
 *  this is the same weld half, standalone). Returns, per vertex, its canonical representative
 *  (the first-seen original vertex index sharing its exact float bit-pattern) — NOT compacted,
 *  so canonical ids are themselves valid indices into the frame's own position/uv arrays. */
export function weldByPosition(positions: Float32Array): Int32Array {
  const n = positions.length / 3;
  const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
  const canon = new Int32Array(n);
  const map = new Map<string, number>();
  for (let v = 0; v < n; v++) {
    const key = bits[v * 3]! + "," + bits[v * 3 + 1]! + "," + bits[v * 3 + 2]!;
    const c = map.get(key);
    if (c === undefined) { map.set(key, v); canon[v] = v; }
    else canon[v] = c;
  }
  return canon;
}

/** One capped-hole boundary loop: welded (canonical) vertex ids, plus the loop's boundary edges
 *  in their OWN owning triangle's directed order (needed for correct cap winding — see module
 *  header point 4). `vertices` and `directedEdges` are parallel arrays, one entry per loop edge,
 *  in walk order (the walk direction itself is arbitrary; only each edge's STORED direction,
 *  captured from whichever single triangle owns it, matters for winding). */
export interface BoundaryLoop {
  vertices: number[];
  directedEdges: { u: number; v: number }[];
}

const edgeKey = (a: number, b: number) => (a < b ? a + "," + b : b + "," + a);

/**
 * Find every boundary loop in a (post-delete-sweep) frame. Boundary edge = a welded undirected
 * edge used by exactly one triangle. Non-manifold welded vertices (boundary-degree != 2) and any
 * loop that would walk through one are excluded — logged once per frame via console.warn, not per
 * vertex (a torso-band delete can touch a handful of such vertices; one line is enough context).
 */
export function findBoundaryLoops(
  positions: Float32Array, indices: Uint32Array, frameLabel: string,
): { loops: BoundaryLoop[]; nonManifoldVertices: number } {
  const canon = weldByPosition(positions);
  const edgeCount = new Map<string, number>();
  const edgeDir = new Map<string, { u: number; v: number }>();
  const addEdge = (a: number, b: number) => {
    const k = edgeKey(a, b);
    edgeCount.set(k, (edgeCount.get(k) ?? 0) + 1);
    if (!edgeDir.has(k)) edgeDir.set(k, { u: a, v: b });
  };
  for (let t = 0; t < indices.length; t += 3) {
    const a = canon[indices[t]!]!, b = canon[indices[t + 1]!]!, c = canon[indices[t + 2]!]!;
    addEdge(a, b); addEdge(b, c); addEdge(c, a);
  }

  const boundaryAdj = new Map<number, number[]>();
  const pushAdj = (a: number, b: number) => {
    let arr = boundaryAdj.get(a);
    if (!arr) { arr = []; boundaryAdj.set(a, arr); }
    arr.push(b);
  };
  const boundaryEdgeDir = new Map<string, { u: number; v: number }>();
  for (const [k, count] of edgeCount) {
    if (count !== 1) continue;
    const { u, v } = edgeDir.get(k)!;
    boundaryEdgeDir.set(k, { u, v });
    pushAdj(u, v);
    pushAdj(v, u);
  }

  const junction = new Set<number>();
  for (const [v, adj] of boundaryAdj) if (adj.length !== 2) junction.add(v);
  if (junction.size) {
    console.warn(`[ares] hole-patch: frame ${frameLabel}: ${junction.size} non-manifold boundary vertex/vertices (degree != 2) — any loop touching them is skipped`);
  }

  const visited = new Set<string>();
  const loops: BoundaryLoop[] = [];
  const maxSteps = boundaryEdgeDir.size + 2;
  for (const [k, { u, v }] of boundaryEdgeDir) {
    if (visited.has(k)) continue;
    if (junction.has(u) || junction.has(v)) { visited.add(k); continue; }
    const vertices: number[] = [u];
    const directedEdges: { u: number; v: number }[] = [{ u, v }];
    visited.add(k);
    let prev = u, cur = v, steps = 0, broke = false;
    while (cur !== u) {
      vertices.push(cur);
      const adj = boundaryAdj.get(cur);
      if (!adj || adj.length !== 2 || junction.has(cur)) { broke = true; break; }
      const next = adj[0] === prev ? adj[1]! : adj[0]!;
      const nk = edgeKey(cur, next);
      directedEdges.push(boundaryEdgeDir.get(nk)!);
      visited.add(nk);
      prev = cur; cur = next;
      if (++steps > maxSteps) { broke = true; break; } // safety: a bug should never spin forever
    }
    if (broke) {
      console.warn(`[ares] hole-patch: frame ${frameLabel}: a boundary walk did not close into a simple cycle — skipping that loop`);
      continue;
    }
    loops.push({ vertices, directedEdges });
  }
  return { loops, nonManifoldVertices: junction.size };
}

/**
 * Filter one frame's boundary loops to the ones `range`'s deletion actually opened (module header
 * point 3). Returns accepted loops plus a rejected count (pre-existing capture boundaries, e.g.
 * the floor cut, land here and are never touched).
 *
 * Threshold, tuned on the real capture: a loop passes if its centroid tests inside the region, OR AT LEAST ONE rim vertex
 * does. A plain ">50% of vertices" threshold (the first thing tried) turned out to be fragile —
 * a box delete's OWN boundary loop sits ON the box's cutting face by construction, so roughly
 * HALF its rim vertices land marginally inside/outside per axis-aligned face, and that fraction
 * hovers right at 0.4-0.6 and flips accept/reject frame-to-frame for the SAME genuine loop
 * (measured: a band delete's far wall and an arm-cut loop both crossed the 50% line depending on
 * frame, while sitting on the SAME range's own cut). The capture's one genuine pre-existing
 * boundary this project could construct for comparison (an artificial floor/foot crop, unrelated
 * to a chest-band delete) measured a clean 0.00 fraction in every frame — zero rim vertices
 * anywhere near the chest region, as expected for a spatially distant boundary. Zero vs.
 * "at least a few tenths" is a much larger, more reliable gap than 0.5 vs. a coin-flip, so the
 * loop-level rule is "at least one edit-adjacent vertex" rather than "a majority."
 */
export function detectHoleLoopsForRange(
  frame: Pick<EncodeMeshFrame, "positions" | "indices">, range: EditRange, frameIndex: number, frameLabel: string,
): { accepted: BoundaryLoop[]; rejectedRegion: number; nonManifoldVertices: number } {
  const { loops, nonManifoldVertices } = findBoundaryLoops(frame.positions, frame.indices, frameLabel);
  const inside = prepareRangeAt(range, frameIndex);
  const accepted: BoundaryLoop[] = [];
  let rejectedRegion = 0;
  for (const loop of loops) {
    const n = loop.vertices.length;
    let cx = 0, cy = 0, cz = 0;
    for (const v of loop.vertices) { cx += frame.positions[v * 3]!; cy += frame.positions[v * 3 + 1]!; cz += frame.positions[v * 3 + 2]!; }
    cx /= n; cy /= n; cz /= n;
    let pass = inside(cx, cy, cz);
    if (!pass) {
      for (const v of loop.vertices) {
        if (inside(frame.positions[v * 3]!, frame.positions[v * 3 + 1]!, frame.positions[v * 3 + 2]!)) { pass = true; break; }
      }
    }
    if (pass) accepted.push(loop); else rejectedRegion++;
  }
  return { accepted, rejectedRegion, nonManifoldVertices };
}

export interface AppendCapsResult {
  frame: EncodeMeshFrame;
  capTriangles: number;
  /** per accepted loop, its rim vertices' ORIGINAL (pre-append) uv — used by the caller to derive
   *  atlas pixel coordinates for average-color sampling (cli.ts, atlas dimensions live there). */
  rimUvPerLoop: [number, number][][];
}

/**
 * Cap every accepted loop: one new apex vertex at the loop centroid, fan-triangulated against
 * DUPLICATED copies of the loop's rim vertices (never the originals — those are still referenced
 * by the surviving mesh, so touching their uv would repaint the real surface). `tileUvs[i]` is the
 * allocated tile-center uv for loop i (module header point 5), or null when no atlas space could
 * be planned (falls back to uv (0,0) — "whatever texel that holds", documented limitation).
 */
export function appendCaps(
  frame: EncodeMeshFrame, loops: BoundaryLoop[], tileUvs: ([number, number] | null)[],
): AppendCapsResult {
  const { positions, uvs, indices } = frame;
  const outPos: number[] = Array.from(positions);
  const outUv: number[] | undefined = uvs ? Array.from(uvs) : undefined;
  const outIdx: number[] = Array.from(indices);
  let capTriangles = 0;
  const rimUvPerLoop: [number, number][][] = [];

  for (let li = 0; li < loops.length; li++) {
    const loop = loops[li]!;
    const tileUv = tileUvs[li] ?? null;
    const n = loop.vertices.length;
    let cx = 0, cy = 0, cz = 0;
    for (const v of loop.vertices) { cx += positions[v * 3]!; cy += positions[v * 3 + 1]!; cz += positions[v * 3 + 2]!; }
    cx /= n; cy /= n; cz /= n;

    const apexIdx = outPos.length / 3;
    outPos.push(cx, cy, cz);
    if (outUv) outUv.push(tileUv ? tileUv[0] : 0, tileUv ? tileUv[1] : 0);

    const dupMap = new Map<number, number>();
    const dup = (orig: number): number => {
      let d = dupMap.get(orig);
      if (d === undefined) {
        d = outPos.length / 3;
        outPos.push(positions[orig * 3]!, positions[orig * 3 + 1]!, positions[orig * 3 + 2]!);
        if (outUv) outUv.push(tileUv ? tileUv[0] : 0, tileUv ? tileUv[1] : 0);
        dupMap.set(orig, d);
      }
      return d;
    };

    const rimUv: [number, number][] = [];
    if (uvs) for (const v of loop.vertices) rimUv.push([uvs[v * 2]!, uvs[v * 2 + 1]!]);
    rimUvPerLoop.push(rimUv);

    for (const e of loop.directedEdges) {
      const dv = dup(e.v), du = dup(e.u);
      outIdx.push(dv, du, apexIdx); // reversed edge direction — faces outward like its neighbor (header point 4)
      capTriangles++;
    }
  }

  return {
    frame: { positions: Float32Array.from(outPos), uvs: outUv ? Float32Array.from(outUv) : undefined, indices: Uint32Array.from(outIdx) },
    capTriangles,
    rimUvPerLoop,
  };
}

/** Scan for one free HOLE_TILE x HOLE_TILE block in `occ` (1 = occupied). Grid-aligned pass first
 *  (cheap, and this capture's atlas has plenty of tile-aligned gaps), a per-pixel fallback pass
 *  second (bounded — HOLE_TILE is tiny). Caller stamps the result into `occ` before requesting the
 *  next tile so multiple loops in one frame never collide. */
export function findFreeTile(occ: Uint8Array, w: number, h: number, tile: number = HOLE_TILE): { x: number; y: number } | null {
  const fits = (x: number, y: number): boolean => {
    for (let dy = 0; dy < tile; dy++) {
      const row = (y + dy) * w;
      for (let dx = 0; dx < tile; dx++) if (occ[row + x + dx]) return false;
    }
    return true;
  };
  for (const stride of [tile, 1]) {
    for (let y = 0; y + tile <= h; y += stride) {
      for (let x = 0; x + tile <= w; x += stride) {
        if (fits(x, y)) return { x, y };
      }
    }
  }
  return null;
}

/** One frame's queued hole-tile fill: `rect` is the allocated HOLE_TILE² block; `color` (explicit
 *  patchHoles.color) wins when present, else the caller-supplied `samplePx` (atlas pixel coords of
 *  the loop's own rim vertices) are averaged from the image AT FILL TIME — i.e. after any copy
 *  blit and recolor patch already applied to the same frame (buildPatchedAtlasDir's ordering). */
export interface HoleTileFill {
  rect: { x: number; y: number; w: number; h: number };
  color?: [number, number, number];
  samplePx?: [number, number][];
  rangeId?: string;
}

/** Apply one hole-tile fill to a decoded atlas image in place — the pixel-math half texel-copy.ts
 *  imports (mirrors recolor.ts's applyRecolorToImage split: this module owns the op's math, the
 *  shared scratch-dir decode/encode pass lives in texel-copy.ts). */
export function applyHoleFillToImage(data: Uint8Array, w: number, fill: HoleTileFill): void {
  let color = fill.color;
  if (!color) {
    let r = 0, g = 0, b = 0, n = 0;
    for (const [px, py] of fill.samplePx ?? []) {
      const o = (py * w + px) * 4;
      r += data[o]!; g += data[o + 1]!; b += data[o + 2]!; n++;
    }
    color = n ? [r / n, g / n, b / n] : [128, 128, 128]; // no rim samples at all — shouldn't happen; neutral gray
  }
  const cr = Math.max(0, Math.min(255, Math.round(color[0])));
  const cg = Math.max(0, Math.min(255, Math.round(color[1])));
  const cb = Math.max(0, Math.min(255, Math.round(color[2])));
  const { x, y, w: tw, h: th } = fill.rect;
  for (let yy = y; yy < y + th; yy++) {
    const row = yy * w;
    for (let xx = x; xx < x + tw; xx++) {
      const o = (row + xx) * 4;
      data[o] = cr; data[o + 1] = cg; data[o + 2] = cb; // alpha untouched
    }
  }
}
