/**
 * Mesh-editor bake: crop a frame to an axis-aligned world box. Triangles whose CENTROID lies
 * outside the box are dropped, then unreferenced vertices are compacted away (positions + UVs;
 * normals are derived later in the muxer from the cropped mesh). Mirrors the renderer's live
 * fragment-discard preview closely enough for author-side cleanup (floor scraps, capture edges).
 */
import type { EncodeMeshFrame } from "./geometry-encode.js";

export interface CropBox { min: [number, number, number]; max: [number, number, number]; }

export function parseCropBox(s: string): CropBox {
  const v = s.split(",").map(Number);
  if (v.length !== 6 || v.some((x) => !Number.isFinite(x))) throw new Error(`bad --crop "${s}" (want x0,y0,z0,x1,y1,z1)`);
  return {
    min: [Math.min(v[0]!, v[3]!), Math.min(v[1]!, v[4]!), Math.min(v[2]!, v[5]!)],
    max: [Math.max(v[0]!, v[3]!), Math.max(v[1]!, v[4]!), Math.max(v[2]!, v[5]!)],
  };
}

/** Crop = the degenerate keep-box filter (docs/editor-v2-design.md §10). */
export function cropFrame(frame: EncodeMeshFrame, box: CropBox): EncodeMeshFrame {
  return filterFrame(frame, (cx, cy, cz) =>
    cx >= box.min[0] && cx <= box.max[0] && cy >= box.min[1] && cy <= box.max[1] && cz >= box.min[2] && cz <= box.max[2]);
}

/**
 * Generalized editor-bake filter: keep triangles whose CENTROID passes the predicate, then compact
 * away unreferenced vertices (positions + UVs; normals derive later from the filtered mesh).
 * The same centroid rule drives the player's live preview (core edits.ts), so preview == bake.
 */
export function filterFrame(frame: EncodeMeshFrame, keepTri: (cx: number, cy: number, cz: number) => boolean): EncodeMeshFrame {
  const { positions, uvs, indices } = frame;
  // Pass 1: keep triangles whose centroid passes.
  const keptTris: number[] = [];
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]! * 3, b = indices[t + 1]! * 3, c = indices[t + 2]! * 3;
    const cx = (positions[a]! + positions[b]! + positions[c]!) / 3;
    const cy = (positions[a + 1]! + positions[b + 1]! + positions[c + 1]!) / 3;
    const cz = (positions[a + 2]! + positions[b + 2]! + positions[c + 2]!) / 3;
    if (keepTri(cx, cy, cz)) keptTris.push(t);
  }

  // Pass 2: compact vertices referenced by kept triangles.
  const vcount = positions.length / 3;
  const remap = new Int32Array(vcount).fill(-1);
  let next = 0;
  const newIndices = new Uint32Array(keptTris.length * 3);
  let w = 0;
  for (const t of keptTris) {
    for (let k = 0; k < 3; k++) {
      const v = indices[t + k]!;
      if (remap[v] === -1) remap[v] = next++;
      newIndices[w++] = remap[v]!;
    }
  }
  const newPos = new Float32Array(next * 3);
  const newUv = uvs ? new Float32Array(next * 2) : undefined;
  for (let v = 0; v < vcount; v++) {
    const r = remap[v]!;
    if (r === -1) continue;
    newPos[r * 3] = positions[v * 3]!; newPos[r * 3 + 1] = positions[v * 3 + 1]!; newPos[r * 3 + 2] = positions[v * 3 + 2]!;
    if (newUv && uvs) { newUv[r * 2] = uvs[v * 2]!; newUv[r * 2 + 1] = uvs[v * 2 + 1]!; }
  }
  return { positions: newPos, uvs: newUv, indices: newIndices };
}
