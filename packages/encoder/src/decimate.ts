/**
 * Optional geometry decimation.
 *
 * meshopt simplifyWithAttributes with UVs riding as a weighted attribute and LockBorder
 * pinning topological border vertices. On an atlased volcap mesh every UV-chart seam is an
 * open edge (vertices are position-duplicated across charts), so LockBorder freezes all
 * seams: the original atlas keeps mapping verbatim and no UV re-bake is needed — the same
 * property that made per-frame remeshing a non-starter. Runs per frame BEFORE smoothing,
 * normal computation, and quantization.
 */
import { MeshoptSimplifier } from "meshoptimizer";
import type { EncodeMeshFrame } from "./geometry-encode.js";

export async function simplifierReady(): Promise<void> {
  await MeshoptSimplifier.ready;
}

/**
 * Reduce one frame to ~`ratio` of its triangles (0 < ratio < 1). Position error is capped at
 * 1% of the mesh extent, so an aggressive ratio degrades gracefully to whatever the error
 * bound allows rather than collapsing the surface. Returns a compacted frame (unreferenced
 * vertices dropped) and never mutates the input.
 */
export function decimateFrame(frame: EncodeMeshFrame, ratio: number): EncodeMeshFrame {
  const targetIndexCount = Math.max(3, Math.floor((frame.indices.length * ratio) / 3) * 3);
  const [simplified] = frame.uvs
    ? MeshoptSimplifier.simplifyWithAttributes(
        frame.indices, frame.positions, 3,
        frame.uvs, 2, [1, 1], /*vertex_lock*/ null,
        targetIndexCount, /*target_error*/ 0.01, ["LockBorder"])
    : MeshoptSimplifier.simplify(frame.indices, frame.positions, 3, targetIndexCount, 0.01, ["LockBorder"]);

  // Compact: the simplified index list references the original vertex arrays; rebuild the
  // streams with only the vertices that survived (mirrors crop.ts filterFrame's compaction).
  const vcount = frame.positions.length / 3;
  const remap = new Int32Array(vcount).fill(-1);
  let next = 0;
  const indices = new Uint32Array(simplified.length);
  for (let i = 0; i < simplified.length; i++) {
    const v = simplified[i]!;
    if (remap[v] === -1) remap[v] = next++;
    indices[i] = remap[v]!;
  }
  const positions = new Float32Array(next * 3);
  const uvs = frame.uvs ? new Float32Array(next * 2) : undefined;
  for (let v = 0; v < vcount; v++) {
    const j = remap[v]!;
    if (j === -1) continue;
    positions[j * 3] = frame.positions[v * 3]!;
    positions[j * 3 + 1] = frame.positions[v * 3 + 1]!;
    positions[j * 3 + 2] = frame.positions[v * 3 + 2]!;
    if (uvs && frame.uvs) { uvs[j * 2] = frame.uvs[v * 2]!; uvs[j * 2 + 1] = frame.uvs[v * 2 + 1]!; }
  }
  return { positions, uvs, indices };
}
