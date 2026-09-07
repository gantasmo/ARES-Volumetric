/**
 * .ares container structures beyond the fixed 64-byte header (spec §11.3–§11.6):
 * superblock, GOP index, track directory, chunk header + block directory.
 *
 * The exact byte layout written here is mirrored by the muxer in @ares/encoder.
 * Every field is little-endian; the reader bounds-checks all input (N6).
 */
import { ByteReader } from "./bytes.js";
import { BlockType, TrackType } from "./format.js";

export interface Superblock {
  aabb: { min: [number, number, number]; max: [number, number, number] };
  quantBitsPos: number;
  quantBitsUv: number;
  normalEncoding: number;
  /** Splat profile only (spec §11.3 `sh_degree`, 0–3); 0 for mesh files. Shares the byte that mesh files wrote as reserved. */
  shDegree: number;
  gopLength: number;
  /** Static texture atlas (spec §7.7 poster/near-static); 0-length when absent. */
  texture: { offset: bigint; length: number; format: TextureBlobFormat; width: number; height: number };
  meta: Record<string, string>;
}

export enum TextureBlobFormat { None = 0, PNG = 1, VideoVP9 = 2, VideoAV1 = 3 }

/** One coded video frame inside a chunk's texture block (WebCodecs EncodedVideoChunk source). */
export interface TextureFrameRef { data: Uint8Array; isKey: boolean; frameIndex: number; }

/**
 * Parse a chunk's texture block payload (spec §11.6 texture block):
 *   frame_count u16, reserved u16, then per frame [size u32][flags u8 (bit0 key)],
 *   then the coded frame datas concatenated.
 */
export function parseTextureBlock(block: Uint8Array, frameStart: number): TextureFrameRef[] {
  const r = new ByteReader(block);
  const count = r.u16();
  r.u16();
  const sizes: number[] = [];
  const keys: boolean[] = [];
  for (let i = 0; i < count; i++) { sizes.push(r.u32()); keys.push((r.u8() & 1) === 1); }
  const out: TextureFrameRef[] = [];
  for (let i = 0; i < count; i++) {
    out.push({ data: r.bytes(sizes[i]!), isKey: keys[i]!, frameIndex: frameStart + i });
  }
  return out;
}

/** One Opus packet from a chunk's audio block, with absolute presentation time. */
export interface AudioPacketRef { data: Uint8Array; ptsUs: number; durationUs: number; }

/**
 * Parse a chunk's audio block (spec §11.6 audio block, as written by @ares/encoder audio-mux.ts):
 *   packet_count u16, reserved u16, per packet [pts_offset_us u32][duration_us u16][size u16],
 *   then the packet datas concatenated. Offsets are relative to the chunk's pts_start.
 */
export function parseAudioBlock(block: Uint8Array, chunkStartUs: number): AudioPacketRef[] {
  const r = new ByteReader(block);
  const count = r.u16();
  r.u16();
  const meta: { off: number; dur: number; size: number }[] = [];
  for (let i = 0; i < count; i++) meta.push({ off: r.u32(), dur: r.u16(), size: r.u16() });
  const out: AudioPacketRef[] = [];
  for (const m of meta) out.push({ data: r.bytes(m.size), ptsUs: chunkStartUs + m.off, durationUs: m.dur });
  return out;
}

export interface GopEntry {
  startPtsUs: bigint;
  frameStart: number;
  frameCount: number;
  flags: number;
  byteOffset: bigint;
  byteLength: number;
}

export interface Track {
  trackId: number;
  trackType: TrackType;
  codecFourcc: string;
  tier: number;
  codecConfig: Uint8Array;
}

export interface BlockDirEntry {
  type: BlockType;
  trackId: number;
  /** offset relative to the start of the chunk */
  offset: number;
  length: number;
}

export interface ChunkHeader {
  ptsStartUs: bigint;
  frameCount: number;
  flags: number;
  gopAabb: { min: [number, number, number]; max: [number, number, number] };
  blocks: BlockDirEntry[];
  /** absolute file offset where this chunk begins */
  fileOffset: number;
}

export const CHUNK_MAGIC = "CNK0";

export function parseSuperblock(buf: Uint8Array, offset: number): Superblock {
  const r = new ByteReader(buf, offset);
  const min = r.f32x3();
  const max = r.f32x3();
  const quantBitsPos = r.u8();
  const quantBitsUv = r.u8();
  const normalEncoding = r.u8();
  const shDegree = r.u8(); // spec §11.3 sh_degree (splat profile); mesh files write 0
  const gopLength = r.u16();
  const texOffset = r.u64();
  const texLength = r.u32();
  const texFormat = r.u8() as TextureBlobFormat;
  r.u8(); // reserved2
  const texW = r.u16();
  const texH = r.u16();
  const kvCount = r.u16();
  const meta: Record<string, string> = {};
  for (let i = 0; i < kvCount; i++) {
    const k = r.str();
    meta[k] = r.str();
  }
  return {
    aabb: { min, max },
    quantBitsPos, quantBitsUv, normalEncoding, shDegree, gopLength,
    texture: { offset: texOffset, length: texLength, format: texFormat, width: texW, height: texH },
    meta,
  };
}

export function parseGopIndex(buf: Uint8Array, offset: number): GopEntry[] {
  const r = new ByteReader(buf, offset);
  const count = r.u32();
  const out: GopEntry[] = [];
  for (let i = 0; i < count; i++) {
    out.push({
      startPtsUs: r.u64(),
      frameStart: r.u32(),
      frameCount: r.u16(),
      flags: r.u16(),
      byteOffset: r.u64(),
      byteLength: r.u32(),
    });
  }
  return out;
}

export function parseTrackDir(buf: Uint8Array, offset: number): Track[] {
  const r = new ByteReader(buf, offset);
  const count = r.u16();
  const out: Track[] = [];
  for (let i = 0; i < count; i++) {
    const trackId = r.u16();
    const trackType = r.u8() as TrackType;
    const codecFourcc = r.fourcc();
    const tier = r.u8();
    const cfgLen = r.u16();
    const codecConfig = r.bytes(cfgLen).slice();
    out.push({ trackId, trackType, codecFourcc, tier, codecConfig });
  }
  return out;
}

export function parseChunkHeader(buf: Uint8Array, fileOffset: number): ChunkHeader {
  const r = new ByteReader(buf, fileOffset);
  const magic = r.fourcc();
  if (magic !== CHUNK_MAGIC) throw new Error(`bad chunk magic "${magic}" at ${fileOffset}`);
  const ptsStartUs = r.u64();
  const frameCount = r.u16();
  const flags = r.u16();
  const min = r.f32x3();
  const max = r.f32x3();
  const blockCount = r.u16();
  const blocks: BlockDirEntry[] = [];
  for (let i = 0; i < blockCount; i++) {
    blocks.push({
      type: r.u8() as BlockType,
      trackId: r.u16(),
      offset: r.u32(),
      length: r.u32(),
    });
  }
  return { ptsStartUs, frameCount, flags, gopAabb: { min, max }, blocks, fileOffset };
}

/** Binary search the GOP index for the chunk covering `ptsUs` (spec §9.2). */
export function gopForPts(index: GopEntry[], ptsUs: bigint): GopEntry | null {
  if (!index.length) return null;
  let lo = 0, hi = index.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (index[mid]!.startPtsUs <= ptsUs) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return index[ans]!;
}
