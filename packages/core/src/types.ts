/** Runtime data structures — spec Appendix C / §11. */
import type { GeometryProfile, TextureCodec, IntraCodec, EntropyCodec } from "./format.js";

export interface AresHeader {
  versionMajor: number;
  versionMinor: number;
  headerFlags: number;
  geometryProfile: GeometryProfile;
  textureCodec: TextureCodec;
  intraCodec: IntraCodec;
  entropyCodec: EntropyCodec;
  fps: number;
  frameCount: number;
  durationUs: bigint;
  superblockOffset: bigint;
  gopIndexOffset: bigint;
  trackDirOffset: bigint;
  firstChunkOffset: bigint;
  headerCrc32: number;
}

// The parsed GOP index record lives in container.ts (GopEntry) — the shape actually
// read from the file. SplatBuffers/DecodedFrame below are spec-shaped types for the
// P4 splat profile; no decode path consumes them yet.

export interface SplatBuffers {
  count: number;
  position: Float32Array; // xyz * count
  scale: Float32Array;    // xyz * count
  rotation: Int8Array;    // quat * count
  opacity: Uint8Array;
  sh: Float32Array;
}

export interface DecodedFrame {
  pts: number;
  positions?: Uint16Array;      // quantized; dequantized on the GPU
  indices?: Uint32Array;        // present on I-frames only (persistent topology)
  changedIndices?: Uint32Array; // P/B sparse
  residuals?: Int16Array;       // P/B sparse
  splat?: SplatBuffers;
  texture?: VideoFrame;         // WebCodecs output; close() after GPU import
}
