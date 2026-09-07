/**
 * antimatter15 `.splat` importer/exporter: 32 bytes per splat, no header —
 *   f32 position ×3 · f32 LINEAR scale ×3 · u8 rgba (rgb display colour, a = opacity) ·
 *   u8 rotation ×4 as (w, x, y, z), component = (byte − 128) / 128.
 * No spherical harmonics. The count is the file length / 32.
 */
import { emptySplatFrame, type SplatFrame } from "../splat-frame.js";

export const SPLAT_RECORD = 32;

export function parseSplatFile(buf: Uint8Array): SplatFrame {
  if (buf.length % SPLAT_RECORD !== 0) throw new Error(`.splat: length ${buf.length} is not a multiple of ${SPLAT_RECORD}`);
  const n = buf.length / SPLAT_RECORD;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const f = emptySplatFrame(n, 0);
  for (let i = 0; i < n; i++) {
    const o = i * SPLAT_RECORD;
    f.positions[i * 3] = dv.getFloat32(o, true); f.positions[i * 3 + 1] = dv.getFloat32(o + 4, true); f.positions[i * 3 + 2] = dv.getFloat32(o + 8, true);
    f.scales[i * 3] = dv.getFloat32(o + 12, true); f.scales[i * 3 + 1] = dv.getFloat32(o + 16, true); f.scales[i * 3 + 2] = dv.getFloat32(o + 20, true);
    f.colors[i * 3] = buf[o + 24]! / 255; f.colors[i * 3 + 1] = buf[o + 25]! / 255; f.colors[i * 3 + 2] = buf[o + 26]! / 255;
    f.opacities[i] = buf[o + 27]! / 255;
    const w = (buf[o + 28]! - 128) / 128, x = (buf[o + 29]! - 128) / 128, y = (buf[o + 30]! - 128) / 128, z = (buf[o + 31]! - 128) / 128;
    const l = Math.hypot(x, y, z, w) || 1;
    f.rotations[i * 4] = x / l; f.rotations[i * 4 + 1] = y / l; f.rotations[i * 4 + 2] = z / l; f.rotations[i * 4 + 3] = w / l;
  }
  return f;
}

export function writeSplatFile(f: SplatFrame): Uint8Array {
  const out = new Uint8Array(f.count * SPLAT_RECORD);
  const dv = new DataView(out.buffer);
  const u8 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
  for (let i = 0; i < f.count; i++) {
    const o = i * SPLAT_RECORD;
    dv.setFloat32(o, f.positions[i * 3]!, true); dv.setFloat32(o + 4, f.positions[i * 3 + 1]!, true); dv.setFloat32(o + 8, f.positions[i * 3 + 2]!, true);
    dv.setFloat32(o + 12, f.scales[i * 3]!, true); dv.setFloat32(o + 16, f.scales[i * 3 + 1]!, true); dv.setFloat32(o + 20, f.scales[i * 3 + 2]!, true);
    out[o + 24] = u8(f.colors[i * 3]! * 255); out[o + 25] = u8(f.colors[i * 3 + 1]! * 255); out[o + 26] = u8(f.colors[i * 3 + 2]! * 255);
    out[o + 27] = u8(f.opacities[i]! * 255);
    const x = f.rotations[i * 4]!, y = f.rotations[i * 4 + 1]!, z = f.rotations[i * 4 + 2]!, w = f.rotations[i * 4 + 3]!;
    out[o + 28] = u8(w * 128 + 128); out[o + 29] = u8(x * 128 + 128); out[o + 30] = u8(y * 128 + 128); out[o + 31] = u8(z * 128 + 128);
  }
  return out;
}
