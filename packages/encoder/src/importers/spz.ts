/**
 * Niantic SPZ importer/exporter (spec §14 P5; docs/splat-integration-notes.md §1).
 *
 * Versions 1–3: one gzip stream of [16-byte header][positions][alphas][colors][scales][rotations][sh].
 * Version 4:    plaintext 32-byte header, optional extension records, a TOC, then one zstd stream per
 *               attribute in the same order. Node ≥ 22.15 ships zstd in node:zlib, so no dependency.
 *
 * Per-attribute encodings (nianticlabs/spz load-spz.cc):
 *   positions  v1: float16 ×3 · v2+: int24 LE ×3, metres = v / 2^fractionalBits
 *   alphas     u8, opacity = a/255 (sigmoid already applied)
 *   colors     u8 ×3, sh0 = (c/255 − 0.5)/0.15 → display rgb = 0.5 + C0·sh0
 *   scales     u8 ×3, log scale = s/16 − 10
 *   rotations  v1–2: u8 xyz = (r − 127.5)/127.5, w = √(1 − |xyz|²) · v3+: u32 "smallest three" (splat.ts)
 *   sh         u8, coefficient-major rgb, value = (v − 128)/128
 * Coordinate system: SPZ default is RUB (right, up, back = the OpenGL/three.js frame), which is
 * ARES's Y-up world; positions are taken verbatim.
 */
import { gunzipSync, gzipSync, zstdDecompressSync } from "node:zlib";
import { SH_C0, sh0ToColor, colorToSh0, shRestCoeffs, packQuaternion, unpackQuaternion } from "@ares/core";
import { emptySplatFrame, type SplatFrame } from "../splat-frame.js";

export const SPZ_MAGIC = 0x5053474e; // "NGSP"
const COLOR_SCALE = 0.15;

function f16(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export function isSpz(buf: Uint8Array): boolean {
  if (buf.length < 4) return false;
  if (buf[0] === 0x1f && buf[1] === 0x8b) return true; // gzip (v1–3); the magic is inside
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  return dv.getUint32(0, true) === SPZ_MAGIC;
}

interface Streams { positions: Uint8Array; alphas: Uint8Array; colors: Uint8Array; scales: Uint8Array; rotations: Uint8Array; sh: Uint8Array; }

export interface SpzHeader { version: number; numPoints: number; shDegree: number; fractionalBits: number; flags: number; }

export function parseSpz(input: Uint8Array): SplatFrame {
  let buf = input;
  if (buf[0] === 0x1f && buf[1] === 0x8b) buf = new Uint8Array(gunzipSync(buf));
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (buf.length < 16 || dv.getUint32(0, true) !== SPZ_MAGIC) throw new Error("SPZ: bad magic (not an SPZ file)");
  const version = dv.getUint32(4, true);
  const numPoints = dv.getUint32(8, true);
  const shDegree = buf[12]!, fractionalBits = buf[13]!, flags = buf[14]!;
  if (version < 1 || version > 4) throw new Error(`SPZ: unsupported version ${version}`);
  if (shDegree > 3) throw new Error(`SPZ: sh degree ${shDegree} unsupported (0–3)`);
  const shDim = shRestCoeffs(shDegree) * 3;
  const posBytes = version === 1 ? 2 : 3;
  const rotBytes = version >= 3 ? 4 : 3;
  const sizes = { positions: numPoints * 3 * posBytes, alphas: numPoints, colors: numPoints * 3, scales: numPoints * 3, rotations: numPoints * rotBytes, sh: numPoints * shDim };

  let st: Streams;
  if (version >= 4) {
    const numStreams = buf[15]!;
    const toc = dv.getUint32(16, true);
    if (numStreams < 6) throw new Error(`SPZ v4: expected ≥ 6 streams, got ${numStreams}`);
    if (toc + numStreams * 16 > buf.length) throw new Error("SPZ v4: TOC out of bounds");
    let off = toc + numStreams * 16;
    const out: Uint8Array[] = [];
    for (let i = 0; i < numStreams; i++) {
      const comp = Number(dv.getBigUint64(toc + i * 16, true));
      const raw = Number(dv.getBigUint64(toc + i * 16 + 8, true));
      if (off + comp > buf.length) throw new Error(`SPZ v4: stream ${i} out of bounds`);
      const data = new Uint8Array(zstdDecompressSync(buf.subarray(off, off + comp)));
      if (data.length !== raw) throw new Error(`SPZ v4: stream ${i} decompressed to ${data.length}, expected ${raw}`);
      out.push(data);
      off += comp;
    }
    st = { positions: out[0]!, alphas: out[1]!, colors: out[2]!, scales: out[3]!, rotations: out[4]!, sh: out[5]! };
  } else {
    let off = 16;
    const take = (n: number): Uint8Array => {
      if (off + n > buf.length) throw new Error(`SPZ: truncated (need ${off + n} bytes, have ${buf.length})`);
      const v = buf.subarray(off, off + n); off += n; return v;
    };
    st = { positions: take(sizes.positions), alphas: take(sizes.alphas), colors: take(sizes.colors), scales: take(sizes.scales), rotations: take(sizes.rotations), sh: take(sizes.sh) };
    // Cheapest corruption gate there is: the declared layout must consume the file exactly.
    if (off !== buf.length) throw new Error(`SPZ: ${buf.length - off} trailing byte(s) after the declared arrays — corrupt or mis-declared header`);
  }
  for (const [k, n] of Object.entries(sizes)) if ((st as unknown as Record<string, Uint8Array>)[k]!.length !== n) throw new Error(`SPZ: ${k} stream is ${(st as unknown as Record<string, Uint8Array>)[k]!.length} bytes, expected ${n}`);

  const f = emptySplatFrame(numPoints, shDegree);
  f.antialiased = (flags & 1) !== 0;
  const scale = 1 / (1 << fractionalBits);
  const p = st.positions;
  for (let i = 0; i < numPoints * 3; i++) {
    if (version === 1) f.positions[i] = f16(p[i * 2]! | (p[i * 2 + 1]! << 8));
    else {
      let v = p[i * 3]! | (p[i * 3 + 1]! << 8) | (p[i * 3 + 2]! << 16);
      if (v & 0x800000) v |= 0xff000000;
      f.positions[i] = (v | 0) * scale;
    }
  }
  for (let i = 0; i < numPoints; i++) f.opacities[i] = st.alphas[i]! / 255;
  for (let i = 0; i < numPoints * 3; i++) f.colors[i] = sh0ToColor((st.colors[i]! / 255 - 0.5) / COLOR_SCALE);
  for (let i = 0; i < numPoints * 3; i++) f.scales[i] = Math.exp(st.scales[i]! / 16 - 10);
  const r = st.rotations;
  if (version >= 3) {
    const q: [number, number, number, number] = [0, 0, 0, 0];
    for (let i = 0; i < numPoints; i++) {
      const c = (r[i * 4]! | (r[i * 4 + 1]! << 8) | (r[i * 4 + 2]! << 16) | (r[i * 4 + 3]! << 24)) >>> 0;
      unpackQuaternion(c, q);
      f.rotations[i * 4] = q[0]; f.rotations[i * 4 + 1] = q[1]; f.rotations[i * 4 + 2] = q[2]; f.rotations[i * 4 + 3] = q[3];
    }
  } else {
    for (let i = 0; i < numPoints; i++) {
      const x = (r[i * 3]! - 127.5) / 127.5, y = (r[i * 3 + 1]! - 127.5) / 127.5, z = (r[i * 3 + 2]! - 127.5) / 127.5;
      const w = Math.sqrt(Math.max(0, 1 - x * x - y * y - z * z));
      f.rotations[i * 4] = x; f.rotations[i * 4 + 1] = y; f.rotations[i * 4 + 2] = z; f.rotations[i * 4 + 3] = w;
    }
  }
  if (shDim && f.sh) for (let i = 0; i < numPoints * shDim; i++) f.sh[i] = (st.sh[i]! - 128) / 128;
  return f;
}

export interface SpzWriteOptions {
  /** 2 = first-three rotations (widest reader support), 3 = smallest-three (default). Both gzip. */
  version?: 2 | 3;
  fractionalBits?: number;
}

/** Write an SPZ (gzip, v2 or v3). Positions are clamped to the int24 range at the chosen precision. */
export function writeSpz(f: SplatFrame, opts: SpzWriteOptions = {}): Uint8Array {
  const version = opts.version ?? 3;
  const fb = opts.fractionalBits ?? 12;
  const n = f.count;
  const shDim = shRestCoeffs(f.shDegree) * 3;
  const rotBytes = version >= 3 ? 4 : 3;
  const total = 16 + n * 9 + n + n * 3 + n * 3 + n * rotBytes + n * shDim;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, SPZ_MAGIC, true); dv.setUint32(4, version, true); dv.setUint32(8, n, true);
  out[12] = f.shDegree; out[13] = fb; out[14] = f.antialiased ? 1 : 0; out[15] = 0;
  let o = 16;
  const mul = 1 << fb, lim = (1 << 23) - 1;
  for (let i = 0; i < n * 3; i++) {
    let v = Math.round(f.positions[i]! * mul);
    v = v < -lim ? -lim : v > lim ? lim : v;
    out[o++] = v & 0xff; out[o++] = (v >> 8) & 0xff; out[o++] = (v >> 16) & 0xff;
  }
  const u8 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
  for (let i = 0; i < n; i++) out[o++] = u8(f.opacities[i]! * 255);
  for (let i = 0; i < n * 3; i++) out[o++] = u8(colorToSh0(f.colors[i]!) * (COLOR_SCALE * 255) + 127.5);
  for (let i = 0; i < n * 3; i++) out[o++] = u8((Math.log(Math.max(1e-12, f.scales[i]!)) + 10) * 16);
  for (let i = 0; i < n; i++) {
    let x = f.rotations[i * 4]!, y = f.rotations[i * 4 + 1]!, z = f.rotations[i * 4 + 2]!, w = f.rotations[i * 4 + 3]!;
    if (version >= 3) {
      const c = packQuaternion(x, y, z, w);
      out[o++] = c & 0xff; out[o++] = (c >>> 8) & 0xff; out[o++] = (c >>> 16) & 0xff; out[o++] = (c >>> 24) & 0xff;
    } else {
      const l = Math.hypot(x, y, z, w) || 1;
      x /= l; y /= l; z /= l; w /= l;
      const sgn = w < 0 ? -127.5 : 127.5;
      out[o++] = u8(x * sgn + 127.5); out[o++] = u8(y * sgn + 127.5); out[o++] = u8(z * sgn + 127.5);
    }
  }
  if (shDim && f.sh) for (let i = 0; i < n * shDim; i++) out[o++] = u8(f.sh[i]! * 128 + 128);
  return new Uint8Array(gzipSync(out));
}

export { SH_C0 };
