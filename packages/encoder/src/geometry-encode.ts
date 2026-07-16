/**
 * Geometry I-frame block encode (spec §11.6.1, §8.2). Quantizes positions/UVs,
 * reorders for meshopt cache+size efficiency, and meshopt-encodes each stream.
 * Mirrors @ares/core's decodeGeometryBlock exactly.
 */
import { MeshoptEncoder } from "meshoptimizer";
import {
  ByteWriter, AttrMask, quantizePositions, quantizeUVs, type Aabb,
} from "@ares/core";

export interface EncodeMeshFrame {
  positions: Float32Array; // xyz interleaved
  uvs?: Float32Array;      // uv interleaved (optional)
  normals?: Float32Array;  // xyz interleaved unit normals (optional; smooth shading)
  indices: Uint32Array;    // triangle list
}

/**
 * Area-weighted smooth per-vertex normals (spec §6.3) — makes dense meshes read smooth, not faceted.
 * WELD-AWARE: atlased meshes duplicate vertices along UV-chart seams (same position, different UV);
 * accumulating per index would give each copy only its own side's triangles, painting a visible
 * shading line along every chart boundary. Accumulating on position-welded groups (exact float bits)
 * gives all copies the same normal, so shading is continuous across atlas seams.
 */
export function computeSmoothNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const vcount = positions.length / 3;
  const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
  const canon = new Int32Array(vcount); // vertex -> first vertex with the same position
  const map = new Map<string, number>();
  for (let v = 0; v < vcount; v++) {
    const key = bits[v * 3]! + "," + bits[v * 3 + 1]! + "," + bits[v * 3 + 2]!;
    const c = map.get(key);
    if (c === undefined) { map.set(key, v); canon[v] = v; } else canon[v] = c;
  }
  const n = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i]! * 3, b = indices[i + 1]! * 3, c = indices[i + 2]! * 3;
    const e1x = positions[b]! - positions[a]!, e1y = positions[b + 1]! - positions[a + 1]!, e1z = positions[b + 2]! - positions[a + 2]!;
    const e2x = positions[c]! - positions[a]!, e2y = positions[c + 1]! - positions[a + 1]!, e2z = positions[c + 2]! - positions[a + 2]!;
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x; // 2·area·normal
    const ca = canon[indices[i]!]! * 3, cb = canon[indices[i + 1]!]! * 3, cc = canon[indices[i + 2]!]! * 3;
    n[ca] = n[ca]! + nx; n[ca + 1] = n[ca + 1]! + ny; n[ca + 2] = n[ca + 2]! + nz;
    n[cb] = n[cb]! + nx; n[cb + 1] = n[cb + 1]! + ny; n[cb + 2] = n[cb + 2]! + nz;
    n[cc] = n[cc]! + nx; n[cc + 1] = n[cc + 1]! + ny; n[cc + 2] = n[cc + 2]! + nz;
  }
  // Normalize canonical accumulators, then copy to every duplicate.
  for (let v = 0; v < vcount; v++) {
    if (canon[v] !== v) continue;
    const i = v * 3;
    const l = Math.hypot(n[i]!, n[i + 1]!, n[i + 2]!) || 1;
    n[i] = n[i]! / l; n[i + 1] = n[i + 1]! / l; n[i + 2] = n[i + 2]! / l;
  }
  for (let v = 0; v < vcount; v++) {
    const c = canon[v]!;
    if (c === v) continue;
    n[v * 3] = n[c * 3]!; n[v * 3 + 1] = n[c * 3 + 1]!; n[v * 3 + 2] = n[c * 3 + 2]!;
  }
  return n;
}

/** Superblock `normal_encoding` values (spec §11.5): 0 = i8×4 snorm (legacy), 1 = octahedral 2×i16. */
export const NORMAL_ENCODING_OCT16 = 1;

const clampI16 = (v: number) => { const r = Math.round(v); return r < -32767 ? -32767 : r > 32767 ? 32767 : r; };

/**
 * Encode unit normals as meshopt(octahedral 2×i16 snorm), stride 4 — same byte size as the legacy
 * i8×4 but ~0.01° angular precision instead of ~0.5–1°, which removes shading banding across facets.
 * Files carry superblock normal_encoding = 1; the WGSL unpack is selected from that flag.
 */
function encodeNormals(normals: Float32Array, vcount: number): Uint8Array {
  const q = new Int16Array(vcount * 2);
  for (let i = 0; i < vcount; i++) {
    const x = normals[i * 3]!, y = normals[i * 3 + 1]!, z = normals[i * 3 + 2]!;
    const s = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
    let ox = x / s, oy = y / s;
    if (z < 0) { // fold the lower hemisphere
      const tx = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1);
      const ty = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1);
      ox = tx; oy = ty;
    }
    q[i * 2] = clampI16(ox * 32767);
    q[i * 2 + 1] = clampI16(oy * 32767);
  }
  return MeshoptEncoder.encodeVertexBuffer(new Uint8Array(q.buffer), vcount, 4);
}

export async function meshoptEncoderReady(): Promise<void> {
  await MeshoptEncoder.ready;
}

/**
 * Encode one intra geometry block.
 * `reorder` runs meshopt's vertex-cache/size reorder (remapping every attribute stream). It is
 * LOSSLESS — same verts/tris, just renumbered — and only valid where the runtime re-uploads topology
 * every frame (the intra path, spec §12.3). It roughly halves the index stream and shrinks UVs/positions
 * too (measured ~−35% geometry on real captures). Keep it OFF for temporal I-frames, whose P-frame
 * deltas require the vertex order to stay stable across the GOP.
 */
export function encodeGeometryBlock(frame: EncodeMeshFrame, box: Aabb, bits: number, reorder = false): Uint8Array {
  const vcount = frame.positions.length / 3;
  let posQ = quantizePositions(frame.positions, box, bits);     // stride 4 (u16 x,y,z,pad)
  let uvQ = frame.uvs ? quantizeUVs(frame.uvs) : undefined;      // stride 2 (u16 u,v)
  let normals = frame.normals;
  // Work on a copy of the indices — reorderMesh remaps them in place.
  let indices: Uint32Array = frame.indices;

  if (reorder) {
    indices = frame.indices.slice();
    const [remap] = MeshoptEncoder.reorderMesh(indices, /*triangles*/ true, /*optsize*/ true); // remap[oldVert]=newVert
    const posR = new Uint16Array(vcount * 4);
    const uvR = uvQ ? new Uint16Array(vcount * 2) : undefined;
    const nrmR = normals ? new Float32Array(vcount * 3) : undefined;
    for (let i = 0; i < vcount; i++) {
      const j = remap![i]!;
      posR[j * 4] = posQ[i * 4]!; posR[j * 4 + 1] = posQ[i * 4 + 1]!; posR[j * 4 + 2] = posQ[i * 4 + 2]!;
      if (uvR && uvQ) { uvR[j * 2] = uvQ[i * 2]!; uvR[j * 2 + 1] = uvQ[i * 2 + 1]!; }
      if (nrmR && normals) { nrmR[j * 3] = normals[i * 3]!; nrmR[j * 3 + 1] = normals[i * 3 + 1]!; nrmR[j * 3 + 2] = normals[i * 3 + 2]!; }
    }
    posQ = posR; uvQ = uvR; normals = nrmR;
  }

  const encPos = MeshoptEncoder.encodeVertexBuffer(new Uint8Array(posQ.buffer), vcount, 8);
  const encUv = uvQ ? MeshoptEncoder.encodeVertexBuffer(new Uint8Array(uvQ.buffer), vcount, 4) : undefined;
  const encNrm = normals ? encodeNormals(normals, vcount) : undefined;
  const encIdx = MeshoptEncoder.encodeIndexBuffer(
    new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength), indices.length, 4);

  let attrMask = AttrMask.Position;
  if (encUv) attrMask |= AttrMask.UV;
  if (encNrm) attrMask |= AttrMask.Normal;

  const w = new ByteWriter(encPos.byteLength + (encUv?.byteLength ?? 0) + (encNrm?.byteLength ?? 0) + encIdx.byteLength + 64);
  w.u32(vcount).u32(indices.length).u16(attrMask);
  w.u32(encPos.byteLength).bytes(encPos);
  if (encUv) w.u32(encUv.byteLength).bytes(encUv);
  if (encNrm) w.u32(encNrm.byteLength).bytes(encNrm);
  w.u32(encIdx.byteLength).bytes(encIdx);
  return w.finish();
}

/**
 * P-frame block (spec §11.6.2): quantized position deltas vs the previous frame, meshopt-coded.
 * Topology persists from the GOP's I-frame. Optional per-frame UV and normal streams follow
 * the deltas — tracked re-atlased captures re-lay the atlas every frame, so their GOPs carry
 * per-frame UVs (see temporal.ts).
 *   vertex_count u32, delta_len u32, delta_bytes[] (meshopt vertex buffer, stride 8 = i16 x4),
 *   uv_len u32, uv_bytes[], normal_len u32, normal_bytes[]   (zero length when absent)
 */
export function encodePFrameBlock(prevPosQ: Uint16Array, curPosQ: Uint16Array, vertexCount: number, normals?: Float32Array, uvs?: Float32Array): Uint8Array {
  const delta = new Int16Array(vertexCount * 4);
  for (let i = 0; i < vertexCount * 4; i++) delta[i] = curPosQ[i]! - prevPosQ[i]!;
  const enc = MeshoptEncoder.encodeVertexBuffer(new Uint8Array(delta.buffer), vertexCount, 8);
  // per-frame UVs (tracked re-atlased captures); stride-2 u16 → stride-4 buffer via quantizeUVs
  const encUv = uvs ? MeshoptEncoder.encodeVertexBuffer(new Uint8Array(quantizeUVs(uvs).buffer), vertexCount, 4) : new Uint8Array(0);
  const encNrm = normals ? encodeNormals(normals, vertexCount) : new Uint8Array(0);
  const w = new ByteWriter(enc.byteLength + encUv.byteLength + encNrm.byteLength + 32);
  w.u32(vertexCount).u32(enc.byteLength).bytes(enc).u32(encUv.byteLength).bytes(encUv).u32(encNrm.byteLength).bytes(encNrm);
  return w.finish();
}
