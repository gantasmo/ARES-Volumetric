/**
 * Position/UV quantization shared by the encoder (quantize) and the runtime's GPU
 * dequant params (spec §6.3, §12.4). Positions quantize to fixed-point over an AABB;
 * the GPU vertex shader dequantizes on read so the CPU only moves compact bytes.
 */

export interface Aabb {
  min: [number, number, number];
  max: [number, number, number];
}

export function computeAabb(positions: Float32Array): Aabb {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = positions[i + a]!;
      if (v < min[a]!) min[a] = v as never;
      if (v > max[a]!) max[a] = v as never;
    }
  }
  if (!Number.isFinite(min[0])) { min[0] = min[1] = min[2] = 0; max[0] = max[1] = max[2] = 0; }
  return { min, max };
}

export function unionAabb(a: Aabb, b: Aabb): Aabb {
  return {
    min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
    max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
  };
}

/**
 * Quantize interleaved xyz positions to u16 over `box`, packed at stride 4
 * (x,y,z,pad) so the result maps 1:1 onto the GPU storage layout (2 u32 / vertex).
 */
export function quantizePositions(positions: Float32Array, box: Aabb, bits = 14): Uint16Array {
  const n = positions.length / 3;
  const levels = (1 << bits) - 1;
  const out = new Uint16Array(n * 4);
  const size = [box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]];
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) {
      const s = size[a]! || 1;
      let v = Math.round(((positions[i * 3 + a]! - box.min[a]!) / s) * levels);
      out[i * 4 + a] = v < 0 ? 0 : v > levels ? levels : v;
    }
    out[i * 4 + 3] = 0;
  }
  return out;
}

/** Quantize interleaved uv to u16 over [0,1], packed at stride 2 (1 u32 / vertex). */
export function quantizeUVs(uvs: Float32Array): Uint16Array {
  const out = new Uint16Array(uvs.length);
  for (let i = 0; i < uvs.length; i++) {
    let v = Math.round(uvs[i]! * 65535);
    out[i] = v < 0 ? 0 : v > 65535 ? 65535 : v;
  }
  return out;
}

/** Dequant scale the GPU applies: world = mix(min, max, q/levels). */
export function dequantScale(bits: number): number {
  return 1 / ((1 << bits) - 1);
}
