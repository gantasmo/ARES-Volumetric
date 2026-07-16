/**
 * ARES muxer — assembles a full intra-only mesh container (spec §11).
 * Layout:  [header 64B][superblock][gop index][track dir][texture blob][chunks…]
 * Each chunk is one GOP of intra geometry blocks (P1 has no P/B frames).
 */
import {
  H, HEADER_SIZE, MAGIC_BYTES, VERSION_MAJOR, VERSION_MINOR, crc32,
  GeometryProfile, TextureCodec, IntraCodec, EntropyCodec, BlockType, TrackType,
  FourCC, CHUNK_MAGIC, TextureBlobFormat, ByteWriter, quantizePositions, type Aabb,
} from "@ares/core";
import { encodeGeometryBlock, encodePFrameBlock, computeSmoothNormals, meshoptEncoderReady, type EncodeMeshFrame } from "./geometry-encode.js";
import { buildTemporalGops, type TemporalOptions } from "./temporal.js";

export interface MuxTextureVideo {
  fourcc: string; // "VP09" | "AV01"
  width: number;
  height: number;
  /** one entry per GOP, aligned to the geometry GOPs; each holds that GOP's coded frames */
  gops: { frames: { data: Uint8Array; isKey: boolean }[] }[];
}

export interface MuxClip {
  fps: number;
  frames: EncodeMeshFrame[];
  gopLength?: number;        // frames per chunk (default 30)
  quantBitsPos?: number;     // default 14
  meta?: Record<string, string>;
  texture?: { png: Uint8Array; width: number; height: number };  // still atlas (spec §7.7)
  textureVideo?: MuxTextureVideo;                                 // video track (spec §7.1)
  /** P2 temporal geometry (spec §8.3): persistent topology + P-frame deltas + jitter smoothing */
  temporal?: Partial<Omit<TemporalOptions, "gopLength">>;
}

interface EncodedChunk { bytes: Uint8Array; startPts: bigint; frameStart: number; frameCount: number; }

export interface MuxResult { bytes: Uint8Array; temporalFrames: number; intraFrames: number; meanTrackError: number; }

export async function muxClip(clip: MuxClip): Promise<Uint8Array> {
  return (await muxClipWithStats(clip)).bytes;
}

export async function muxClipWithStats(clip: MuxClip): Promise<MuxResult> {
  await meshoptEncoderReady();
  const bits = clip.quantBitsPos ?? 14;
  const gopLength = clip.gopLength ?? 30;
  const frameCount = clip.frames.length;
  if (!frameCount) throw new Error("muxClip: no frames");
  const usPerFrame = 1e6 / clip.fps;

  // Temporal plan (P2). With no temporal opts + varying topology this is all-intra (P1 behaviour);
  // with stable topology or track:true it produces I-frame + P-frame GOPs.
  const gops = buildTemporalGops(clip.frames, {
    gopLength,
    track: clip.temporal?.track ?? false,
    smoothTemporal: clip.temporal?.smoothTemporal ?? 0,
    smoothSpatial: clip.temporal?.smoothSpatial ?? 0,
    forceIntra: clip.temporal?.forceIntra ?? false,
  });
  const globalBox = gops.reduce((acc, g) => acc ? unionBox(acc, g.gopBox) : g.gopBox, null as Aabb | null)!;

  let temporalFrames = 0, intraFrames = 0, errAcc = 0, errN = 0;
  const chunks: EncodedChunk[] = [];

  // Texture frames are addressed by FRAME INDEX, never by chunk ordinal.
  //
  // This used to be `clip.textureVideo.gops[chunkIdx]` — texture GOP i onto geometry chunk i — which
  // silently assumed the two chunkings agree. They only agree when geometry is chunked uniformly at
  // gopLength. The moment the temporal planner splits a GOP at a topology reset (it must: a P-frame
  // cannot cross a topology change), geometry gets MORE chunks than the texture has GOPs and every
  // pairing after the first is wrong. Measured on the 4910-frame SVF clip: 938 geometry chunks vs 164
  // texture GOPs — chunk 1 covered geometry frames 1-5 but carried texture frames 30-59, and chunks
  // 164+ got no texture at all. That is a wrong texture on essentially every frame, and it is exactly
  // why the temporal path had to be forced off with --no-temporal.
  //
  // The texture GOPs are a flat, in-order sequence covering frame 0..frameCount-1, so flattening and
  // slicing by the chunk's own frame range is correct for ANY chunking, uniform or not.
  const texFrames = clip.textureVideo ? clip.textureVideo.gops.flatMap((g) => g.frames) : null;
  if (texFrames && texFrames.length !== clip.frames.length) {
    // Don't paper over it: a mismatch here means some frame would silently ship the wrong texture.
    throw new Error(`texture/geometry frame count mismatch: ${texFrames.length} coded texture frames for ${clip.frames.length} geometry frames`);
  }

  gops.forEach((gop, chunkIdx) => {
    const start = gop.frameStart;
    const gopBox = gop.gopBox;
    const blocks: { type: BlockType; trackId: number; payload: Uint8Array }[] = [];
    let framesInChunk: number;

    if (gop.temporal && gop.framePositions && gop.uvs && gop.indices) {
      const fp = gop.framePositions;
      const fuv = gop.frameUvs; // per-frame UVs when tracked (else undefined → UVs constant)
      const vcount = fp[0]!.length / 3;
      const posQs = fp.map((p) => quantizePositions(p, gopBox, bits));
      const normals = fp.map((p) => computeSmoothNormals(p, gop.indices!)); // smooth per-frame normals
      // I-frame (full: positions[0] + uvs[0] + normals[0] + persistent indices)
      blocks.push({ type: BlockType.GeometryI, trackId: 0, payload: encodeGeometryBlock({ positions: fp[0]!, uvs: fuv ? fuv[0]! : gop.uvs, normals: normals[0], indices: gop.indices }, gopBox, bits) });
      // P-frames (position deltas + [per-frame UVs when re-atlased] + normals)
      for (let f = 1; f < posQs.length; f++) blocks.push({ type: BlockType.GeometryPB, trackId: 0, payload: encodePFrameBlock(posQs[f - 1]!, posQs[f]!, vcount, normals[f], fuv ? fuv[f]! : undefined) });
      framesInChunk = fp.length;
      temporalFrames += fp.length;
      if (gop.trackError) { errAcc += gop.trackError; errN++; }
    } else {
      const member = gop.frames!;
      // Intra path re-uploads topology every frame → meshopt vertex reorder is lossless & free here (C1).
      for (const fr of member) blocks.push({ type: BlockType.GeometryI, trackId: 0, payload: encodeGeometryBlock({ ...fr, normals: computeSmoothNormals(fr.positions, fr.indices) }, gopBox, bits, /*reorder*/ true) });
      framesInChunk = member.length;
      intraFrames += member.length;
    }

    // This chunk's OWN frames, by index — see the note above the loop.
    const tframes = texFrames ? texFrames.slice(start, start + framesInChunk) : null;
    if (tframes && tframes.length) blocks.push({ type: BlockType.TextureColor, trackId: 1, payload: buildTextureBlock(tframes) });
    void chunkIdx;
    const startPts = BigInt(Math.round(start * usPerFrame));
    chunks.push({ bytes: assembleChunk(startPts, framesInChunk, gopBox, blocks), startPts, frameStart: start, frameCount: framesInChunk });
  });

  // Sections (superblock offset patched after layout).
  const { bytes: sb, texOffsetPos } = writeSuperblock(globalBox, bits, gopLength, clip.meta ?? {}, clip.texture, clip.textureVideo);
  const trackDir = writeTrackDir(clip.texture, clip.textureVideo);
  const texBlob = clip.texture ? clip.texture.png : new Uint8Array(0);
  const gopIndexSize = 4 + chunks.length * (8 + 4 + 2 + 2 + 8 + 4);

  // Offsets.
  let offset = HEADER_SIZE;
  const superblockOffset = offset; offset += sb.length;
  const gopIndexOffset = offset; offset += gopIndexSize;
  const trackDirOffset = offset; offset += trackDir.length;
  const textureOffset = offset; offset += texBlob.length;
  const firstChunkOffset = offset;
  const chunkOffsets = chunks.map((c) => { const o = offset; offset += c.bytes.length; return o; });

  // GOP index with real chunk offsets.
  const gi = new ByteWriter(gopIndexSize);
  gi.u32(chunks.length);
  chunks.forEach((c, i) => {
    gi.u64(c.startPts).u32(c.frameStart).u16(c.frameCount).u16(0).u64(BigInt(chunkOffsets[i]!)).u32(c.bytes.length);
  });
  const gopIndexBytes = gi.finish();

  // Patch texture offset into the superblock.
  if (clip.texture) new DataView(sb.buffer, sb.byteOffset, sb.byteLength).setBigUint64(texOffsetPos, BigInt(textureOffset), true);

  const texCodec = clip.textureVideo
    ? (clip.textureVideo.fourcc === "AV01" ? TextureCodec.AV1 : TextureCodec.VP9)
    : TextureCodec.None;
  const header = writeHeader({
    fps: clip.fps, frameCount, durationUs: BigInt(Math.round(frameCount * usPerFrame)),
    superblockOffset, gopIndexOffset, trackDirOffset, firstChunkOffset, textureCodec: texCodec,
  });

  const out = new Uint8Array(offset);
  out.set(header, 0);
  out.set(sb, superblockOffset);
  out.set(gopIndexBytes, gopIndexOffset);
  out.set(trackDir, trackDirOffset);
  out.set(texBlob, textureOffset);
  chunks.forEach((c, i) => out.set(c.bytes, chunkOffsets[i]!));
  return { bytes: out, temporalFrames, intraFrames, meanTrackError: errN ? errAcc / errN : 0 };
}

function unionBox(a: Aabb, b: Aabb): Aabb {
  return {
    min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
    max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
  };
}

function assembleChunk(startPts: bigint, frameCount: number, box: Aabb, blocks: { type: BlockType; trackId: number; payload: Uint8Array }[]): Uint8Array {
  const dirSize = blocks.length * (1 + 2 + 4 + 4);
  const headerSize = 4 + 8 + 2 + 2 + 12 + 12 + 2 + dirSize;
  const w = new ByteWriter(headerSize + blocks.reduce((s, b) => s + b.payload.length, 0));
  w.fourcc(CHUNK_MAGIC).u64(startPts).u16(frameCount).u16(0);
  w.f32x3(box.min).f32x3(box.max);
  w.u16(blocks.length);
  let payloadOffset = headerSize;
  for (const b of blocks) { w.u8(b.type).u16(b.trackId).u32(payloadOffset).u32(b.payload.length); payloadOffset += b.payload.length; }
  for (const b of blocks) w.bytes(b.payload);
  return w.finish();
}

/** Texture block payload (spec §11.6): frame_count u16, reserved u16, [size u32, flags u8]×N, datas. */
function buildTextureBlock(frames: { data: Uint8Array; isKey: boolean }[]): Uint8Array {
  const total = frames.reduce((s, f) => s + f.data.byteLength, 0);
  const w = new ByteWriter(4 + frames.length * 5 + total);
  w.u16(frames.length).u16(0);
  for (const f of frames) w.u32(f.data.byteLength).u8(f.isKey ? 1 : 0);
  for (const f of frames) w.bytes(f.data);
  return w.finish();
}

function textureFormat(texture?: { png: Uint8Array }, video?: MuxTextureVideo): TextureBlobFormat {
  if (video) return video.fourcc === "AV01" ? TextureBlobFormat.VideoAV1 : TextureBlobFormat.VideoVP9;
  if (texture) return TextureBlobFormat.PNG;
  return TextureBlobFormat.None;
}

function writeSuperblock(box: Aabb, bits: number, gopLength: number, meta: Record<string, string>, texture?: { png: Uint8Array; width: number; height: number }, video?: MuxTextureVideo): { bytes: Uint8Array; texOffsetPos: number } {
  const w = new ByteWriter(256);
  w.f32x3(box.min).f32x3(box.max);
  w.u8(bits).u8(14).u8(1).u8(0);            // quant_bits_pos, quant_bits_uv, normal_encoding (1 = oct16), reserved
  w.u16(gopLength);
  const texOffsetPos = w.pos;
  w.u64(0n);                                 // texture_offset (patched after layout; 0 for video/none)
  w.u32(texture ? texture.png.byteLength : 0);
  w.u8(textureFormat(texture, video)).u8(0);
  w.u16(video?.width ?? texture?.width ?? 0).u16(video?.height ?? texture?.height ?? 0);
  const entries = Object.entries(meta);
  w.u16(entries.length);
  for (const [k, v] of entries) w.str(k).str(v);
  return { bytes: w.finish(), texOffsetPos };
}

function writeTrackDir(texture?: { png: Uint8Array }, video?: MuxTextureVideo): Uint8Array {
  const hasTex = !!(texture || video);
  const w = new ByteWriter(64);
  w.u16(hasTex ? 2 : 1);
  w.u16(0).u8(TrackType.Geometry).fourcc(FourCC.Meshopt).u8(0).u16(0);
  if (hasTex) {
    const fourcc = video ? video.fourcc : FourCC.PNGAtlas;
    w.u16(1).u8(TrackType.TextureColor).fourcc(fourcc).u8(0).u16(0);
  }
  return w.finish();
}

interface HeaderFields {
  fps: number; frameCount: number; durationUs: bigint;
  superblockOffset: number; gopIndexOffset: number; trackDirOffset: number; firstChunkOffset: number;
  textureCodec: TextureCodec;
}

function writeHeader(f: HeaderFields): Uint8Array {
  const buf = new Uint8Array(HEADER_SIZE);
  const dv = new DataView(buf.buffer);
  buf.set(MAGIC_BYTES, H.magic);
  buf[H.versionMajor] = VERSION_MAJOR;
  buf[H.versionMinor] = VERSION_MINOR;
  dv.setUint16(H.headerFlags, 0, true);
  buf[H.geometryProfile] = GeometryProfile.MeshIPB;
  buf[H.textureCodec] = f.textureCodec; // None for still-atlas (§7.7) or no texture; VP9/AV1 for video (§7.1)
  buf[H.intraCodec] = IntraCodec.Meshopt;
  buf[H.entropyCodec] = EntropyCodec.None;
  dv.setFloat32(H.fps, f.fps, true);
  dv.setUint32(H.frameCount, f.frameCount, true);
  dv.setBigUint64(H.durationUs, f.durationUs, true);
  dv.setBigUint64(H.superblockOffset, BigInt(f.superblockOffset), true);
  dv.setBigUint64(H.gopIndexOffset, BigInt(f.gopIndexOffset), true);
  dv.setBigUint64(H.trackDirOffset, BigInt(f.trackDirOffset), true);
  dv.setBigUint64(H.firstChunkOffset, BigInt(f.firstChunkOffset), true);
  dv.setUint32(H.headerCrc32, crc32(buf, 0, H.headerCrc32), true);
  return buf;
}
