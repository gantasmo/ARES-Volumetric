/**
 * Gaussian splat profile — shared definitions (spec §6.8, §11.6.3 as implemented).
 *
 * A splat frame is a set of 3D Gaussians: position, scale (3), rotation (unit quaternion),
 * opacity, base colour, and optional higher-order spherical harmonics. The container carries
 * it as meshopt-coded fixed-stride streams, exactly like mesh vertices, so the same vertex
 * codec, the same per-chunk AABB quantization and the same GPU-side dequant apply:
 *
 *   positions  stride 8   u16 x,y,z,pad         quantized over the chunk AABB (quant_bits_pos)
 *   attrs      stride 12  3 × u32, see below
 *   sh         stride 12 / 24 / 48 for degree 1 / 2 / 3, u8 per coefficient, coefficient-major rgb
 *
 * attrs word 0 : scale.x | scale.y << 8 | scale.z << 16 | alpha << 24
 *                  scale byte s → metres = exp(s / 16 − 10)   (SPZ's log encoding, ~0.9 % steps)
 *                  alpha byte a → opacity = a / 255           (already sigmoid-activated)
 * attrs word 1 : rotation, "smallest three" packed as SPZ v3 does it: bits 30–31 = index of the
 *                  largest |component| (xyzw order), then three 10-bit fields (sign + 9-bit
 *                  magnitude over [0, 1/√2]) for the remaining components in ascending index order,
 *                  first one in bits 20–29. The largest component is rebuilt positive from |q| = 1.
 * attrs word 2 : r | g << 8 | b << 16 | reserved << 24   — display-referred base colour
 *                  (0.5 + C0·sh0, clamped to [0,1], 8-bit), the KHR_gaussian_splatting
 *                  "srgb_rec709_display" convention; higher SH bands add on top in the shader.
 *
 * Why not video-pack any of this: spec §8.5.1 — positions and rotations are not spatially
 * coherent enough to survive chroma subsampling and DCT ringing; only colour is video-shaped,
 * and at shDegree 0 (the common case for generated environments) it is one byte per channel.
 */

/** Y₀₀ = ½·√(1/π). Base colour = 0.5 + SH_C0 · sh0 (3DGS / KHR_gaussian_splatting convention). */
export const SH_C0 = 0.28209479177387814;

export const SPLAT_ATTR_STRIDE = 12;

/** Number of higher-order (degree ≥ 1) SH coefficients per colour channel for a given degree. */
export function shRestCoeffs(degree: number): number {
  return degree <= 0 ? 0 : (degree + 1) * (degree + 1) - 1; // 3 / 8 / 15
}

/** Bytes per splat of the higher-order SH stream (u8 per value, coefficient-major rgb), padded to a multiple of 4. */
export function shStrideBytes(degree: number): number {
  const raw = shRestCoeffs(degree) * 3;         // 9 / 24 / 45
  return (raw + 3) & ~3;                        // 12 / 24 / 48 — meshopt vertex stride must be a multiple of 4
}

/** A decoded splat frame, in the exact byte layouts the renderers upload (see the header comment). */
export interface DecodedSplat {
  count: number;
  shDegree: number;
  flags: number;
  /** quantized positions, stride 4 (u16 x,y,z,pad) — the mesh position layout, dequantized on the GPU */
  positionsQ: Uint16Array;
  /** packed attributes, 3 u32 per splat (see header) */
  attrs: Uint32Array;
  /** higher-order SH bytes, shStrideBytes(shDegree) per splat; undefined for degree 0 */
  sh?: Uint8Array;
}

// ---- scalar encodings (shared by the encoder, the CPU decoder and tests; the shaders mirror them) ----

export function encodeScaleByte(linearScale: number): number {
  const v = Math.round((Math.log(Math.max(1e-12, linearScale)) + 10) * 16);
  return v < 0 ? 0 : v > 255 ? 255 : v;
}
export function decodeScaleByte(b: number): number { return Math.exp(b / 16 - 10); }

export function encodeShByte(v: number): number {
  const q = Math.round(v * 128) + 128;
  return q < 0 ? 0 : q > 255 ? 255 : q;
}
export function decodeShByte(b: number): number { return (b - 128) / 128; }

const SQRT1_2 = Math.SQRT1_2;

/**
 * Pack a unit quaternion (x, y, z, w) as SPZ v3 "smallest three" into one u32.
 * Normalizes first; the sign is chosen so the omitted (largest) component is positive.
 */
export function packQuaternion(x: number, y: number, z: number, w: number): number {
  const l = Math.hypot(x, y, z, w) || 1;
  const q = [x / l, y / l, z / l, w / l];
  let iLargest = 0;
  for (let i = 1; i < 4; i++) if (Math.abs(q[i]!) > Math.abs(q[iLargest]!)) iLargest = i;
  const negate = q[iLargest]! < 0 ? 1 : 0;
  let comp = iLargest;
  for (let i = 0; i < 4; i++) {
    if (i === iLargest) continue;
    const negbit = ((q[i]! < 0 ? 1 : 0) ^ negate) & 1;
    let mag = Math.floor(511 * (Math.abs(q[i]!) / SQRT1_2) + 0.5);
    if (mag > 511) mag = 511;
    comp = ((comp << 10) | (negbit << 9) | mag) >>> 0;
  }
  return comp >>> 0;
}

/** Inverse of packQuaternion → [x, y, z, w], unit length. */
export function unpackQuaternion(comp: number, out?: Float32Array | number[]): [number, number, number, number] {
  const c = comp >>> 0;
  const iLargest = c >>> 30;
  const q: [number, number, number, number] = [0, 0, 0, 0];
  let sum = 0;
  let shift = 20;
  for (let i = 0; i < 4; i++) {
    if (i === iLargest) continue;
    const mag = (c >>> shift) & 0x1ff;
    const neg = (c >>> (shift + 9)) & 1;
    const v = (mag / 511) * SQRT1_2 * (neg ? -1 : 1);
    q[i] = v;
    sum += v * v;
    shift -= 10;
  }
  q[iLargest] = Math.sqrt(Math.max(0, 1 - sum));
  if (out) { out[0] = q[0]; out[1] = q[1]; out[2] = q[2]; out[3] = q[3]; }
  return q;
}

/** Pack attrs word 0 / word 2. Colour components are 0..1 display-referred. */
export function packAttrWord0(sx: number, sy: number, sz: number, opacity: number): number {
  const a = Math.round(Math.max(0, Math.min(1, opacity)) * 255);
  return (encodeScaleByte(sx) | (encodeScaleByte(sy) << 8) | (encodeScaleByte(sz) << 16) | (a << 24)) >>> 0;
}
export function packAttrWord2(r: number, g: number, b: number): number {
  const q = (v: number) => Math.round(Math.max(0, Math.min(1, v)) * 255);
  return (q(r) | (q(g) << 8) | (q(b) << 16)) >>> 0;
}

/** Convert an SH DC coefficient to display colour (0.5 + C0·sh0) and back. */
export const sh0ToColor = (sh0: number): number => 0.5 + SH_C0 * sh0;
export const colorToSh0 = (c: number): number => (c - 0.5) / SH_C0;

/** Sigmoid / logit for 3DGS PLY opacities (stored pre-activation). */
export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
export const logit = (p: number): number => {
  const c = Math.min(1 - 1e-6, Math.max(1e-6, p));
  return Math.log(c / (1 - c));
};
