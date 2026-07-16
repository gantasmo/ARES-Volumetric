/**
 * Geometry I-frame block decode (spec §11.6.1). Mirrors the encoder's writer.
 * Positions/UVs are meshopt-encoded quantized integers; the GPU dequantizes them
 * on read (spec §12.4), so this decoder only inflates compact bytes.
 *
 * Block payload layout (little-endian):
 *   vertex_count u32
 *   index_count  u32
 *   attr_mask    u16   (bit0 pos, bit2 uv)
 *   pos_len u32, pos_bytes[pos_len]   meshopt vertex buffer, stride 8 (u16 x4: x,y,z,pad)
 *   [uv]  uv_len u32, uv_bytes[uv_len]  meshopt vertex buffer, stride 4 (u16 x2: u,v)
 *   idx_len u32, idx_bytes[idx_len]   meshopt index buffer (u32 triangle list)
 */
import { MeshoptDecoder } from "meshoptimizer";
import { ByteReader } from "./bytes.js";
import { AttrMask } from "./format.js";

export interface DecodedGeometry {
  vertexCount: number;
  indexCount: number;
  /** quantized positions, stride 4 (u16 x,y,z,pad) — uploads straight to the GPU storage buffer */
  positionsQ: Uint16Array;
  /** quantized UVs, stride 2 (u16 u,v) — present iff the block carried them */
  uvsQ?: Uint16Array;
  /** snorm normals, stride 4 (i8 x,y,z,pad) — present iff the block carried them */
  normalsQ?: Int8Array;
  indices: Uint32Array;
}

let ready: Promise<void> | null = null;
export function meshoptReady(): Promise<void> {
  return (ready ??= MeshoptDecoder.ready.then(() => undefined));
}

export function decodeGeometryBlock(block: Uint8Array): DecodedGeometry {
  const r = new ByteReader(block);
  const vertexCount = r.u32();
  const indexCount = r.u32();
  const attrMask = r.u16();

  const posLen = r.u32();
  const posSrc = r.bytes(posLen);
  const positionsQ = new Uint16Array(vertexCount * 4);
  MeshoptDecoder.decodeVertexBuffer(
    new Uint8Array(positionsQ.buffer), vertexCount, 8, posSrc);

  let uvsQ: Uint16Array | undefined;
  if (attrMask & AttrMask.UV) {
    const uvLen = r.u32();
    const uvSrc = r.bytes(uvLen);
    uvsQ = new Uint16Array(vertexCount * 2);
    MeshoptDecoder.decodeVertexBuffer(
      new Uint8Array(uvsQ.buffer), vertexCount, 4, uvSrc);
  }

  let normalsQ: Int8Array | undefined;
  if (attrMask & AttrMask.Normal) {
    const nLen = r.u32();
    const nSrc = r.bytes(nLen);
    normalsQ = new Int8Array(vertexCount * 4);
    MeshoptDecoder.decodeVertexBuffer(
      new Uint8Array(normalsQ.buffer), vertexCount, 4, nSrc);
  }

  const idxLen = r.u32();
  const idxSrc = r.bytes(idxLen);
  const indices = new Uint32Array(indexCount);
  MeshoptDecoder.decodeIndexBuffer(
    new Uint8Array(indices.buffer), indexCount, 4, idxSrc);

  return { vertexCount, indexCount, positionsQ, uvsQ, normalsQ, indices };
}

/**
 * Decode a P-frame block (spec §11.6.2): reconstruct quantized positions by adding the
 * meshopt-coded deltas to the previous frame's positions. Topology/UVs persist from the I-frame.
 */
export function decodePFrameBlock(block: Uint8Array, prevPosQ: Uint16Array): { positionsQ: Uint16Array; uvsQ?: Uint16Array; normalsQ?: Int8Array } {
  const r = new ByteReader(block);
  const vertexCount = r.u32();
  const dLen = r.u32();
  const dSrc = r.bytes(dLen);
  const delta = new Int16Array(vertexCount * 4);
  MeshoptDecoder.decodeVertexBuffer(new Uint8Array(delta.buffer), vertexCount, 8, dSrc);
  const cur = new Uint16Array(vertexCount * 4);
  for (let i = 0; i < vertexCount * 4; i++) cur[i] = (prevPosQ[i]! + delta[i]!) & 0xffff;

  let uvsQ: Uint16Array | undefined;
  const uvLen = r.u32();
  if (uvLen > 0) {
    const uvSrc = r.bytes(uvLen);
    uvsQ = new Uint16Array(vertexCount * 2);
    MeshoptDecoder.decodeVertexBuffer(new Uint8Array(uvsQ.buffer), vertexCount, 4, uvSrc);
  }

  let normalsQ: Int8Array | undefined;
  const nLen = r.u32();
  if (nLen > 0) {
    const nSrc = r.bytes(nLen);
    normalsQ = new Int8Array(vertexCount * 4);
    MeshoptDecoder.decodeVertexBuffer(new Uint8Array(normalsQ.buffer), vertexCount, 4, nSrc);
  }
  return { positionsQ: cur, uvsQ, normalsQ };
}
