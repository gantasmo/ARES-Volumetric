/** ARES demuxer — parses the container header, superblock, GOP index, track dir, and chunks. Spec §11. */
import { H, HEADER_SIZE, MAGIC_BYTES, crc32, BlockType, TrackType, HeaderFlags } from "./format.js";
import type { AresHeader } from "./types.js";
import { ByteReader, AresParseError } from "./bytes.js";
import {
  Superblock, GopEntry, Track, ChunkHeader, TextureBlobFormat, TextureFrameRef, AudioPacketRef,
  parseSuperblock, parseGopIndex, parseTrackDir, parseChunkHeader, parseTextureBlock, parseAudioBlock, gopForPts,
} from "./container.js";

export { AresParseError };

export interface AresFile {
  buf: Uint8Array;
  header: AresHeader;
  superblock: Superblock;
  gopIndex: GopEntry[];
  tracks: Track[];
}

export interface TextureAtlas {
  bytes: Uint8Array;
  format: TextureBlobFormat;
  width: number;
  height: number;
}

export class Demuxer {
  /** Parse and validate the fixed 64-byte header. Treats all input as untrusted (N6). */
  static parseHeader(buf: Uint8Array): AresHeader {
    if (buf.length < HEADER_SIZE) throw new AresParseError("buffer shorter than header");
    for (let i = 0; i < 4; i++)
      if (buf[i] !== MAGIC_BYTES[i]) throw new AresParseError("bad magic (not an .ares file)");

    const crc = crc32(buf, 0, H.headerCrc32);
    const dv = new DataView(buf.buffer, buf.byteOffset, HEADER_SIZE);
    const stored = dv.getUint32(H.headerCrc32, true);
    if (stored !== 0 && stored !== crc) throw new AresParseError("header CRC mismatch");
    if (buf[H.versionMajor]! > 0) throw new AresParseError(`unsupported version_major ${buf[H.versionMajor]}`);

    return {
      versionMajor: buf[H.versionMajor]!,
      versionMinor: buf[H.versionMinor]!,
      headerFlags: dv.getUint16(H.headerFlags, true),
      geometryProfile: buf[H.geometryProfile]!,
      textureCodec: buf[H.textureCodec]!,
      intraCodec: buf[H.intraCodec]!,
      entropyCodec: buf[H.entropyCodec]!,
      fps: dv.getFloat32(H.fps, true),
      frameCount: dv.getUint32(H.frameCount, true),
      durationUs: dv.getBigUint64(H.durationUs, true),
      superblockOffset: dv.getBigUint64(H.superblockOffset, true),
      gopIndexOffset: dv.getBigUint64(H.gopIndexOffset, true),
      trackDirOffset: dv.getBigUint64(H.trackDirOffset, true),
      firstChunkOffset: dv.getBigUint64(H.firstChunkOffset, true),
      headerCrc32: stored,
    };
  }

  /** Parse the whole container into an in-memory descriptor. */
  static parse(buf: Uint8Array): AresFile {
    const header = Demuxer.parseHeader(buf);
    const inBounds = (o: bigint) => o > 0n && o < BigInt(buf.byteLength);
    if (!inBounds(header.superblockOffset)) throw new AresParseError("bad superblock offset");
    const superblock = parseSuperblock(buf, Number(header.superblockOffset));
    const gopIndex = inBounds(header.gopIndexOffset) ? parseGopIndex(buf, Number(header.gopIndexOffset)) : [];
    const tracks = inBounds(header.trackDirOffset) ? parseTrackDir(buf, Number(header.trackDirOffset)) : [];
    return { buf, header, superblock, gopIndex, tracks };
  }

  static gopForPts(index: GopEntry[], ptsUs: bigint): GopEntry | null {
    return gopForPts(index, ptsUs);
  }

  /** Parse a chunk's header + block directory. */
  static chunkAt(file: AresFile, gop: GopEntry): ChunkHeader {
    const off = Number(gop.byteOffset);
    if (off <= 0 || off + gop.byteLength > file.buf.byteLength) throw new AresParseError("chunk out of bounds");
    return parseChunkHeader(file.buf, off);
  }

  /** Geometry block {type, data} for a chunk, in frame order (spec §11.6.1/§11.6.2). */
  static geometryBlocks(file: AresFile, chunk: ChunkHeader): { type: BlockType; data: Uint8Array }[] {
    const out: { type: BlockType; data: Uint8Array }[] = [];
    for (const b of chunk.blocks) {
      if (b.type !== BlockType.GeometryI && b.type !== BlockType.GeometryPB) continue;
      const start = chunk.fileOffset + b.offset;
      if (start < 0 || start + b.length > file.buf.byteLength) throw new AresParseError("block out of bounds");
      out.push({ type: b.type, data: file.buf.subarray(start, start + b.length) });
    }
    return out;
  }

  /** Video-texture-track info (spec §7.1), or null if the file has no video texture. */
  static textureVideo(file: AresFile): { format: TextureBlobFormat; fourcc: string; width: number; height: number } | null {
    const fmt = file.superblock.texture.format;
    if (fmt !== TextureBlobFormat.VideoVP9 && fmt !== TextureBlobFormat.VideoAV1) return null;
    const track = file.tracks.find((t) => t.trackType === 1 /* TextureColor */);
    return { format: fmt, fourcc: track?.codecFourcc ?? "VP09", width: file.superblock.texture.width, height: file.superblock.texture.height };
  }

  /** Coded texture-video frames carried by a chunk's texture block (spec §11.6). */
  static textureFrames(file: AresFile, chunk: ChunkHeader): TextureFrameRef[] {
    for (const b of chunk.blocks) {
      if (b.type !== 2 /* TextureColor */) continue;
      const start = chunk.fileOffset + b.offset;
      if (start < 0 || start + b.length > file.buf.byteLength) throw new AresParseError("texture block out of bounds");
      // frameStart derived from the GOP entry sharing this chunk offset
      const gop = file.gopIndex.find((g) => Number(g.byteOffset) === chunk.fileOffset);
      return parseTextureBlock(file.buf.subarray(start, start + b.length), gop?.frameStart ?? 0);
    }
    return [];
  }

  /** The audio track (spec §11.5 `OPUS`), or null when the file carries none. codecConfig = OpusHead. */
  static audioTrack(file: AresFile): { trackId: number; fourcc: string; codecConfig: Uint8Array; sampleRate: number; channels: number } | null {
    if (!(file.header.headerFlags & HeaderFlags.HasAudio)) return null;
    const t = file.tracks.find((x) => x.trackType === TrackType.Audio);
    if (!t) return null;
    const cfg = t.codecConfig;
    const channels = cfg.length >= 19 ? cfg[9]! : 2;
    return { trackId: t.trackId, fourcc: t.codecFourcc, codecConfig: cfg, sampleRate: 48000, channels: channels || 2 };
  }

  /** Timed Opus packets carried by a chunk's audio block (spec §11.6). */
  static audioPackets(file: AresFile, chunk: ChunkHeader): AudioPacketRef[] {
    for (const b of chunk.blocks) {
      if (b.type !== BlockType.Audio) continue;
      const start = chunk.fileOffset + b.offset;
      if (start < 0 || start + b.length > file.buf.byteLength) throw new AresParseError("audio block out of bounds");
      return parseAudioBlock(file.buf.subarray(start, start + b.length), Number(chunk.ptsStartUs));
    }
    return [];
  }

  /** The static texture atlas (spec §7.7 poster / near-static), or null if absent. */
  static textureAtlas(file: AresFile): TextureAtlas | null {
    const t = file.superblock.texture;
    if (t.format !== TextureBlobFormat.PNG || t.length === 0) return null;
    const off = Number(t.offset);
    if (off <= 0 || off + t.length > file.buf.byteLength) throw new AresParseError("texture blob out of bounds");
    const r = new ByteReader(file.buf, off);
    return { bytes: r.bytes(t.length).slice(), format: t.format, width: t.width, height: t.height };
  }
}
