/**
 * ARES container format constants — spec §11 / Appendix A.
 * Single source of truth shared by the demuxer (core) and the muxer (encoder).
 * All multi-byte fields are little-endian.
 */

export const MAGIC = 0x53455241; // 'ARES' as LE u32 ('A'=0x41 at byte 0)
export const MAGIC_BYTES = new Uint8Array([0x41, 0x52, 0x45, 0x53]);
export const CHUNK_MAGIC_BYTES = new Uint8Array([0x43, 0x4e, 0x4b, 0x30]); // 'CNK0'
export const VERSION_MAJOR = 0;
export const VERSION_MINOR = 2;
export const HEADER_SIZE = 64;

export enum GeometryProfile { MeshIPB = 0, SplatIPB = 1, VideoGeometry = 2 }
export enum TextureCodec { None = 0, AV1 = 1, VP9 = 2, HEVC = 3, AVC = 4 }
export enum IntraCodec { Meshopt = 0, Draco = 1, Raw = 2 }
export enum EntropyCodec { RangeANS = 0, None = 1 }
export enum TrackType { Geometry = 0, TextureColor = 1, TextureAux = 2, Audio = 3, Metadata = 4 }
export enum BlockType { GeometryI = 0, GeometryPB = 1, TextureColor = 2, TextureAux = 3, Audio = 4, Metadata = 5 }
export enum FrameType { I = 0, P = 1, B = 2 }

export const HeaderFlags = {
  Live: 1 << 0,
  HasAudio: 1 << 1,
  CrossOriginHint: 1 << 2,
  HasAux: 1 << 3,
} as const;

/** Geometry attribute presence bits in a geometry block's `attr_mask` (spec §11.6.1). */
export const AttrMask = {
  Position: 1 << 0,
  Normal: 1 << 1,
  UV: 1 << 2,
  Color: 1 << 3,
} as const;

/** Registered track/codec FourCCs (spec §11.5, §11.7). */
export const FourCC = {
  Meshopt: "MSHO",
  Draco: "DRAC",
  /** Splat-profile geometry track (spec §6.8, §11.6.3): meshopt-coded quantized splat streams. */
  Splat: "SPLT",
  AV1: "AV01",
  VP9: "VP09",
  PNGAtlas: "PNG0",
  /** Audio track: Opus packets, one block per chunk (spec §11.6). */
  Opus: "OPUS",
} as const;

/** Splat I-frame block `flags` bits (spec §11.6.3 as implemented — see splat.ts). */
export const SplatFlags = {
  /** Splats were trained with the anti-aliased (mip-splatting style) kernel; renderers may dilate less. */
  Antialiased: 1 << 0,
} as const;

/** Byte offsets of the fixed 64-byte file header (Appendix A.1). */
export const H = {
  magic: 0, versionMajor: 4, versionMinor: 5, headerFlags: 6,
  geometryProfile: 8, textureCodec: 9, intraCodec: 10, entropyCodec: 11,
  fps: 12, frameCount: 16, durationUs: 20,
  superblockOffset: 28, gopIndexOffset: 36, trackDirOffset: 44,
  firstChunkOffset: 52, headerCrc32: 60,
} as const;

/** CRC-32 (IEEE) over [start,end); used for header + optional chunk integrity (spec §11.8). */
export function crc32(buf: Uint8Array, start = 0, end = buf.length): number {
  let c = ~0;
  for (let i = start; i < end; i++) {
    c ^= buf[i]!;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (~c) >>> 0;
}
