/**
 * Minimal RGBA PNG writer (Node zlib) + a procedural atlas generator, so the demo
 * ships a self-contained texture (spec §7.7: a still atlas is valid for near-static
 * captures and as the poster that makes TTFF instant). No image-library dependency.
 */
import { deflateSync } from "node:zlib";
import { crc32 } from "@ares/core";

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.byteLength);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.byteLength, false);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  const crc = crc32(out, 4, 8 + data.byteLength);
  dv.setUint32(8 + data.byteLength, crc, false);
  return out;
}

/** Encode an RGBA pixel buffer (width*height*4, row-major) to PNG bytes. */
export function encodePNG(rgba: Uint8Array, width: number, height: number): Uint8Array {
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width, false);
  dv.setUint32(4, height, false);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 6;  // color type RGBA
  // 10,11,12 = compression, filter, interlace = 0

  // raw scanlines with filter byte 0 per row
  const stride = width * 4;
  const raw = new Uint8Array((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0;
    raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
  }
  const idat = new Uint8Array(deflateSync(raw, { level: 9 }));

  const parts = [sig, chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", new Uint8Array(0))];
  const total = parts.reduce((s, p) => s + p.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.byteLength; }
  return out;
}

/** A legible test atlas: colored grid + diagonal gradient + labelled quadrants. */
export function proceduralAtlas(size = 512): { png: Uint8Array; width: number; height: number } {
  const rgba = new Uint8Array(size * size * 4);
  const cells = 8;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const cx = Math.floor((x / size) * cells);
      const cy = Math.floor((y / size) * cells);
      const checker = (cx + cy) & 1;
      // base gradient (warm ARES palette) so UV direction is readable
      const gx = x / size, gy = y / size;
      let r = 40 + 180 * gx;
      let g = 30 + 120 * gy;
      let b = 60 + 120 * (1 - gx);
      if (checker) { r *= 0.55; g *= 0.55; b *= 0.55; }
      // grid lines
      const onGrid = (x % (size / cells)) < 2 || (y % (size / cells)) < 2;
      if (onGrid) { r = 245; g = 238; b = 226; }
      rgba[i] = Math.min(255, r) | 0;
      rgba[i + 1] = Math.min(255, g) | 0;
      rgba[i + 2] = Math.min(255, b) | 0;
      rgba[i + 3] = 255;
    }
  }
  return { png: encodePNG(rgba, size, size), width: size, height: size };
}
