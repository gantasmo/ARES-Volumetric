/** Mesh primitives, quantization, and error metrics for the Phase 0 bench (spec §13). */

export interface BenchMesh {
  positions: Float32Array; // xyz interleaved
  indices: Uint32Array;
}

export const vertexCount = (m: BenchMesh) => m.positions.length / 3;

/** Deterministic PRNG so corpus + results are reproducible run-to-run (spec §13.1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Unit icosphere. subdiv 5 → 10,242 verts; 6 → 40,962 verts (≈ "40k-vertex human", §8.2). */
export function icosphere(subdiv: number): BenchMesh {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts: number[][] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map(normalize);
  let faces: number[][] = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  for (let s = 0; s < subdiv; s++) {
    const cache = new Map<number, number>();
    const mid = (a: number, b: number): number => {
      const key = a < b ? a * 1e7 + b : b * 1e7 + a;
      const hit = cache.get(key);
      if (hit !== undefined) return hit;
      const va = verts[a]!, vb = verts[b]!;
      const m = normalize([(va[0]! + vb[0]!) / 2, (va[1]! + vb[1]!) / 2, (va[2]! + vb[2]!) / 2]);
      verts.push(m);
      cache.set(key, verts.length - 1);
      return verts.length - 1;
    };
    const next: number[][] = [];
    for (const [a, b, c] of faces as [number, number, number][]) {
      const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
  }
  const positions = new Float32Array(verts.length * 3);
  verts.forEach((v, i) => positions.set(v, i * 3));
  const indices = new Uint32Array(faces.length * 3);
  faces.forEach((f, i) => indices.set(f, i * 3));
  return { positions, indices };
}

function normalize(v: number[]): number[] {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!);
  return [v[0]! / l, v[1]! / l, v[2]! / l];
}

/** Torus-knot tube as a closed u×v grid (p=2, q=3): the "object" clip's base (spec §13.2). */
export function torusKnot(uSegs: number, vSegs: number, tube = 0.35): BenchMesh {
  const p = 2, q = 3;
  const positions = new Float32Array(uSegs * vSegs * 3);
  const center = (u: number): [number, number, number, number, number, number] => {
    const r = 2 + Math.cos(q * u);
    const x = r * Math.cos(p * u), y = r * Math.sin(p * u), z = Math.sin(q * u);
    // forward-difference tangent for the tube frame
    const e = 1e-3;
    const r2 = 2 + Math.cos(q * (u + e));
    return [x, y, z, r2 * Math.cos(p * (u + e)) - x, r2 * Math.sin(p * (u + e)) - y, Math.sin(q * (u + e)) - z];
  };
  for (let i = 0; i < uSegs; i++) {
    const u = (i / uSegs) * Math.PI * 2;
    const [cx, cy, cz, tx, ty, tz] = center(u);
    const tl = Math.hypot(tx, ty, tz);
    const t: [number, number, number] = [tx / tl, ty / tl, tz / tl];
    // frame: n = t × up (robust enough for this knot), b = t × n
    let n: [number, number, number] = [t[1], -t[0], 0];
    const nl = Math.hypot(n[0], n[1], n[2]) || 1;
    n = [n[0] / nl, n[1] / nl, n[2] / nl];
    const b: [number, number, number] = [
      t[1] * n[2] - t[2] * n[1], t[2] * n[0] - t[0] * n[2], t[0] * n[1] - t[1] * n[0],
    ];
    for (let j = 0; j < vSegs; j++) {
      const v = (j / vSegs) * Math.PI * 2;
      const cv = Math.cos(v) * tube, sv = Math.sin(v) * tube;
      const k = (i * vSegs + j) * 3;
      positions[k] = cx + cv * n[0] + sv * b[0];
      positions[k + 1] = cy + cv * n[1] + sv * b[1];
      positions[k + 2] = cz + cv * n[2] + sv * b[2];
    }
  }
  const indices = new Uint32Array(uSegs * vSegs * 6);
  let w = 0;
  for (let i = 0; i < uSegs; i++) {
    const i2 = (i + 1) % uSegs;
    for (let j = 0; j < vSegs; j++) {
      const j2 = (j + 1) % vSegs;
      const a = i * vSegs + j, bIdx = i2 * vSegs + j, c = i2 * vSegs + j2, d = i * vSegs + j2;
      indices[w++] = a; indices[w++] = bIdx; indices[w++] = c;
      indices[w++] = a; indices[w++] = c; indices[w++] = d;
    }
  }
  return { positions, indices };
}

export interface Aabb { min: [number, number, number]; size: [number, number, number]; diag: number; }

export function aabb(pos: Float32Array): Aabb {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = pos[i + a]!;
      if (v < min[a]!) min[a] = v as never;
      if (v > max[a]!) max[a] = v as never;
    }
  }
  const size: [number, number, number] = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  return { min, size, diag: Math.hypot(...size) };
}

export interface Quantized { q: Uint16Array; box: Aabb; bits: number; }

/** Quantize positions to `bits` fixed point over the frame AABB (spec §6.3). */
export function quantizePositions(pos: Float32Array, bits: number): Quantized {
  const box = aabb(pos);
  const levels = (1 << bits) - 1;
  const q = new Uint16Array(pos.length);
  for (let i = 0; i < pos.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const s = box.size[a]! || 1;
      let v = Math.round(((pos[i + a]! - box.min[a]!) / s) * levels);
      if (v < 0) v = 0; else if (v > levels) v = levels;
      q[i + a] = v;
    }
  }
  return { q, box, bits };
}

export function dequantizePositions(qz: Quantized): Float32Array {
  const levels = (1 << qz.bits) - 1;
  const out = new Float32Array(qz.q.length);
  for (let i = 0; i < qz.q.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      out[i + a] = qz.box.min[a]! + (qz.q[i + a]! / levels) * (qz.box.size[a]! || 1);
    }
  }
  return out;
}

/** RMS + max positional error between corresponding vertices, relative to bbox diagonal. */
export function positionError(src: Float32Array, dec: Float32Array, diag: number): { rms: number; max: number } {
  let sum = 0, max = 0;
  const n = Math.min(src.length, dec.length) / 3;
  for (let i = 0; i < n * 3; i += 3) {
    const dx = src[i]! - dec[i]!, dy = src[i + 1]! - dec[i + 1]!, dz = src[i + 2]! - dec[i + 2]!;
    const d2 = dx * dx + dy * dy + dz * dz;
    sum += d2;
    if (d2 > max) max = d2;
  }
  return { rms: Math.sqrt(sum / n) / diag, max: Math.sqrt(max) / diag };
}
