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
import { ByteReader, AresParseError } from "./bytes.js";
import { AttrMask } from "./format.js";
import { SPLAT_ATTR_STRIDE, shStrideBytes, type DecodedSplat } from "./splat.js";

/**
 * Allocation guards (spec §11.8 / N6): counts come straight from the file, so a crafted block
 * could ask for tens of GB before the meshopt call ever ran. These caps are far above any real
 * capture (64 M vertices, 256 M indices, 64 M splats) and turn that into a parse error instead.
 */
const MAX_VERTICES = 1 << 26;
const MAX_INDICES = 1 << 28;
const MAX_SPLATS = 1 << 26;
function guardCount(n: number, max: number, what: string): void {
  if (!(n >= 0 && n <= max)) throw new AresParseError(`${what} ${n} exceeds the ${max} cap`);
}

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
  guardCount(vertexCount, MAX_VERTICES, "vertex_count");
  guardCount(indexCount, MAX_INDICES, "index_count");

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
  guardCount(vertexCount, MAX_VERTICES, "vertex_count");
  if (vertexCount * 4 !== prevPosQ.length) throw new AresParseError(`P-frame vertex_count ${vertexCount} does not match the keyframe's ${prevPosQ.length / 4}`);
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

/**
 * Decode a splat-profile I-frame block (spec §11.6.3 as implemented — layout in splat.ts):
 *   splat_count u32, sh_degree u8, flags u8, reserved u16,
 *   pos_len u32,  pos_bytes[]   meshopt vertex buffer, stride 8  (u16 x,y,z,pad)
 *   attr_len u32, attr_bytes[]  meshopt vertex buffer, stride 12 (3 × u32, see splat.ts)
 *   [sh_len u32, sh_bytes[]]    meshopt vertex buffer, stride shStrideBytes(degree) — degree ≥ 1 only
 */
export function decodeSplatBlock(block: Uint8Array): DecodedSplat {
  const r = new ByteReader(block);
  const count = r.u32();
  const shDegree = r.u8();
  const flags = r.u8();
  r.u16();
  guardCount(count, MAX_SPLATS, "splat_count");
  if (shDegree > 3) throw new AresParseError(`splat sh_degree ${shDegree} unsupported (0–3)`);

  const posLen = r.u32();
  const posSrc = r.bytes(posLen);
  const positionsQ = new Uint16Array(count * 4);
  if (count) MeshoptDecoder.decodeVertexBuffer(new Uint8Array(positionsQ.buffer), count, 8, posSrc);

  const attrLen = r.u32();
  const attrSrc = r.bytes(attrLen);
  const attrs = new Uint32Array(count * 3);
  if (count) MeshoptDecoder.decodeVertexBuffer(new Uint8Array(attrs.buffer), count, SPLAT_ATTR_STRIDE, attrSrc);

  let sh: Uint8Array | undefined;
  if (shDegree > 0) {
    const stride = shStrideBytes(shDegree);
    const shLen = r.u32();
    const shSrc = r.bytes(shLen);
    sh = new Uint8Array(count * stride);
    if (count) MeshoptDecoder.decodeVertexBuffer(sh, count, stride, shSrc);
  }
  return { count, shDegree, flags, positionsQ, attrs, sh };
}

/**
 * Decode a splat-profile P-frame block (dynamic splat profile, spec §11.6.3 P-frames):
 *   count u32 (this frame), sh_degree u8, flags u8, reserved u16,
 *   death_count u32, death indices as delta-coded varints (ascending, over the PREVIOUS frame),
 *   survivor_count u32,
 *   delta_len u32, delta_bytes[]   meshopt stride 8: i16 x,y,z,pad position deltas, survivor order
 *   attr_len u32,  attr_bytes[]    meshopt stride 12: survivors' attrs as BYTE deltas (mod 256) vs prev
 *   [sh_len u32, sh_bytes[]]        survivors' SH as byte deltas (mod 256) vs prev (degree ≥ 1)
 * Byte deltas make an unchanged attribute a run of zeros, which the vertex codec folds to almost
 * nothing — the common case for colour, opacity, scale and SH between neighbouring frames.
 *   birth_count u32,
 *   bpos_len u32, bpos_bytes[]     meshopt stride 8: births' quantized positions (absolute)
 *   battr_len u32, battr_bytes[]   meshopt stride 12
 *   [bsh_len u32, bsh_bytes[]]
 * Survivors keep the previous frame's order (minus the dead), births append in the order given.
 */
export function decodeSplatPBlock(block: Uint8Array, prev: DecodedSplat): DecodedSplat {
  const r = new ByteReader(block);
  const count = r.u32();
  const shDegree = r.u8();
  const flags = r.u8();
  r.u16();
  guardCount(count, MAX_SPLATS, "splat_count");
  if (shDegree > 3) throw new AresParseError(`splat sh_degree ${shDegree} unsupported (0–3)`);
  if (shDegree !== prev.shDegree) throw new AresParseError("splat P-frame SH degree differs from its keyframe");
  const stride = shDegree ? shStrideBytes(shDegree) : 0;

  const deathCount = r.u32();
  guardCount(deathCount, prev.count, "death_count");
  const dead = new Uint8Array(prev.count);
  let last = -1;
  for (let i = 0; i < deathCount; i++) {
    const idx = last + 1 + r.varint();
    if (idx >= prev.count) throw new AresParseError("splat death index out of range");
    dead[idx] = 1; last = idx;
  }
  const survivors = r.u32();
  if (survivors !== prev.count - deathCount) throw new AresParseError(`splat survivor count ${survivors} ≠ prev ${prev.count} − deaths ${deathCount}`);
  const births = count - survivors;
  if (births < 0) throw new AresParseError("splat P-frame count smaller than its survivors");

  const dLen = r.u32(); const dSrc = r.bytes(dLen);
  const delta = new Int16Array(survivors * 4);
  if (survivors) MeshoptDecoder.decodeVertexBuffer(new Uint8Array(delta.buffer), survivors, 8, dSrc);
  const aLen = r.u32(); const aSrc = r.bytes(aLen);
  const sAttrs = new Uint32Array(survivors * 3);
  if (survivors) MeshoptDecoder.decodeVertexBuffer(new Uint8Array(sAttrs.buffer), survivors, SPLAT_ATTR_STRIDE, aSrc);
  let sSh: Uint8Array | undefined;
  if (shDegree) { const l = r.u32(); const src = r.bytes(l); sSh = new Uint8Array(survivors * stride); if (survivors) MeshoptDecoder.decodeVertexBuffer(sSh, survivors, stride, src); }

  const birthCount = r.u32();
  if (birthCount !== births) throw new AresParseError(`splat birth count ${birthCount} ≠ count − survivors ${births}`);
  const bpLen = r.u32(); const bpSrc = r.bytes(bpLen);
  const bPos = new Uint16Array(births * 4);
  if (births) MeshoptDecoder.decodeVertexBuffer(new Uint8Array(bPos.buffer), births, 8, bpSrc);
  const baLen = r.u32(); const baSrc = r.bytes(baLen);
  const bAttrs = new Uint32Array(births * 3);
  if (births) MeshoptDecoder.decodeVertexBuffer(new Uint8Array(bAttrs.buffer), births, SPLAT_ATTR_STRIDE, baSrc);
  let bSh: Uint8Array | undefined;
  if (shDegree) { const l = r.u32(); const src = r.bytes(l); bSh = new Uint8Array(births * stride); if (births) MeshoptDecoder.decodeVertexBuffer(bSh, births, stride, src); }

  const positionsQ = new Uint16Array(count * 4);
  const attrs = new Uint32Array(count * 3);
  const sh = shDegree ? new Uint8Array(count * stride) : undefined;
  let w = 0;
  for (let i = 0; i < prev.count; i++) {
    if (dead[i]) continue;
    const pb = i * 4, wb = w * 4;
    positionsQ[wb] = (prev.positionsQ[pb]! + delta[wb]!) & 0xffff;
    positionsQ[wb + 1] = (prev.positionsQ[pb + 1]! + delta[wb + 1]!) & 0xffff;
    positionsQ[wb + 2] = (prev.positionsQ[pb + 2]! + delta[wb + 2]!) & 0xffff;
    // attrs + SH: byte-wise delta against the previous frame's copy of this splat
    for (let k = 0; k < 3; k++) {
      const pv = prev.attrs[i * 3 + k]!, dv = sAttrs[w * 3 + k]!;
      attrs[w * 3 + k] = (((pv & 0xff) + (dv & 0xff)) & 0xff) | ((((pv >>> 8) & 0xff) + ((dv >>> 8) & 0xff)) & 0xff) << 8 |
        ((((pv >>> 16) & 0xff) + ((dv >>> 16) & 0xff)) & 0xff) << 16 | ((((pv >>> 24) & 0xff) + ((dv >>> 24) & 0xff)) & 0xff) << 24;
      attrs[w * 3 + k] = attrs[w * 3 + k]! >>> 0;
    }
    if (sh && sSh && prev.sh) for (let c = 0; c < stride; c++) sh[w * stride + c] = (prev.sh[i * stride + c]! + sSh[w * stride + c]!) & 0xff;
    w++;
  }
  positionsQ.set(bPos, survivors * 4);
  attrs.set(bAttrs, survivors * 3);
  if (sh && bSh) sh.set(bSh, survivors * stride);
  return { count, shDegree, flags, positionsQ, attrs, sh };
}
