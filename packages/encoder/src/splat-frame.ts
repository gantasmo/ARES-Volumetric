/**
 * Splat-profile frame model + I-frame block encode (spec §6.8, §11.6.3; layout in @ares/core splat.ts).
 * Mirrors @ares/core's decodeSplatBlock exactly. Importers (spz/ply/splat/gltf/sog) all produce a
 * SplatFrame; the muxer quantizes it over the chunk AABB and meshopt-codes the streams.
 */
import { MeshoptEncoder } from "meshoptimizer";
import {
  ByteWriter, quantizePositions, computeAabb, packAttrWord0, packAttrWord2, packQuaternion, unpackQuaternion,
  encodeShByte, decodeShByte, decodeScaleByte, shRestCoeffs, shStrideBytes, SPLAT_ATTR_STRIDE, SplatFlags,
  type Aabb, type DecodedSplat,
} from "@ares/core";

export interface SplatFrame {
  count: number;
  /** xyz interleaved, world units */
  positions: Float32Array;
  /** xyz interleaved, LINEAR scales (world units; not log) */
  scales: Float32Array;
  /** xyzw interleaved unit quaternions */
  rotations: Float32Array;
  /** 0..1, activated (sigmoid already applied) */
  opacities: Float32Array;
  /** rgb interleaved, display-referred 0..1 (0.5 + C0·sh0; may slightly exceed the range) */
  colors: Float32Array;
  /** 0..3 */
  shDegree: number;
  /** count × shRestCoeffs(shDegree) × 3, coefficient-major rgb; absent for degree 0 */
  sh?: Float32Array;
  antialiased?: boolean;
}

export function emptySplatFrame(count: number, shDegree = 0): SplatFrame {
  const k = shRestCoeffs(shDegree);
  return {
    count, shDegree,
    positions: new Float32Array(count * 3), scales: new Float32Array(count * 3), rotations: new Float32Array(count * 4),
    opacities: new Float32Array(count), colors: new Float32Array(count * 3),
    sh: k ? new Float32Array(count * k * 3) : undefined,
  };
}

/** AABB over the splats whose opacity is at least `minAlpha` (quantization range; see the field notes on outlier haze). */
export function splatAabb(f: SplatFrame, minAlpha = 0): Aabb {
  if (minAlpha <= 0) return computeAabb(f.positions);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < f.count; i++) {
    if (f.opacities[i]! < minAlpha) continue;
    for (let a = 0; a < 3; a++) {
      const v = f.positions[i * 3 + a]!;
      if (v < min[a]!) min[a] = v;
      if (v > max[a]!) max[a] = v;
    }
  }
  if (!Number.isFinite(min[0])) return computeAabb(f.positions);
  return { min, max };
}

/** Keep the splats for which `keep(i)` is true (new frame, same layout). */
export function filterSplatFrame(f: SplatFrame, keep: (i: number) => boolean): SplatFrame {
  const idx: number[] = [];
  for (let i = 0; i < f.count; i++) if (keep(i)) idx.push(i);
  return permuteSplatFrame(f, Uint32Array.from(idx));
}

/** Reorder (or subset) a frame by an index list. */
export function permuteSplatFrame(f: SplatFrame, order: Uint32Array): SplatFrame {
  const n = order.length;
  const out = emptySplatFrame(n, f.shDegree);
  const k3 = shRestCoeffs(f.shDegree) * 3;
  for (let j = 0; j < n; j++) {
    const i = order[j]!;
    out.positions[j * 3] = f.positions[i * 3]!; out.positions[j * 3 + 1] = f.positions[i * 3 + 1]!; out.positions[j * 3 + 2] = f.positions[i * 3 + 2]!;
    out.scales[j * 3] = f.scales[i * 3]!; out.scales[j * 3 + 1] = f.scales[i * 3 + 1]!; out.scales[j * 3 + 2] = f.scales[i * 3 + 2]!;
    out.rotations[j * 4] = f.rotations[i * 4]!; out.rotations[j * 4 + 1] = f.rotations[i * 4 + 1]!; out.rotations[j * 4 + 2] = f.rotations[i * 4 + 2]!; out.rotations[j * 4 + 3] = f.rotations[i * 4 + 3]!;
    out.opacities[j] = f.opacities[i]!;
    out.colors[j * 3] = f.colors[i * 3]!; out.colors[j * 3 + 1] = f.colors[i * 3 + 1]!; out.colors[j * 3 + 2] = f.colors[i * 3 + 2]!;
    if (k3 && f.sh && out.sh) for (let c = 0; c < k3; c++) out.sh[j * k3 + c] = f.sh[i * k3 + c]!;
  }
  out.antialiased = f.antialiased;
  return out;
}

/** Spread the low 10 bits of v over 30 bits (Morton helper). */
function part1By2(v: number): number {
  let x = v & 0x3ff;
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}

/**
 * Morton (Z-order) permutation over `box`: neighbouring splats land next to each other in the
 * streams, which is what meshopt's per-byte delta prediction feeds on (~20–35 % smaller streams
 * on generated environments), and it keeps the CPU depth sort cache-friendly.
 */
export function mortonOrder(f: SplatFrame, box: Aabb): Uint32Array {
  const n = f.count;
  const keys = new Uint32Array(n);
  const sx = 1023 / ((box.max[0] - box.min[0]) || 1), sy = 1023 / ((box.max[1] - box.min[1]) || 1), sz = 1023 / ((box.max[2] - box.min[2]) || 1);
  for (let i = 0; i < n; i++) {
    const qx = Math.max(0, Math.min(1023, Math.round((f.positions[i * 3]! - box.min[0]) * sx)));
    const qy = Math.max(0, Math.min(1023, Math.round((f.positions[i * 3 + 1]! - box.min[1]) * sy)));
    const qz = Math.max(0, Math.min(1023, Math.round((f.positions[i * 3 + 2]! - box.min[2]) * sz)));
    keys[i] = (part1By2(qx) | (part1By2(qy) << 1) | (part1By2(qz) << 2)) >>> 0;
  }
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  // Stable sort by key (Array sort on a typed view of indices).
  const arr = Array.from(order);
  arr.sort((a, b) => keys[a]! - keys[b]!);
  return Uint32Array.from(arr);
}

/**
 * Quantize a frame into the exact byte streams the container carries (and the decoder returns):
 * positions over `box`, packed attrs, SH bytes capped at `shDegreeCap`. The encoder keeps this
 * state per frame so P-frame deltas are computed against what the decoder will actually hold.
 */
export function quantizeSplatFrame(f: SplatFrame, box: Aabb, bits: number, shDegreeCap = 3): DecodedSplat {
  const n = f.count;
  const positionsQ = quantizePositions(f.positions, box, bits);
  const attrs = new Uint32Array(n * 3);
  for (let i = 0; i < n; i++) {
    attrs[i * 3] = packAttrWord0(f.scales[i * 3]!, f.scales[i * 3 + 1]!, f.scales[i * 3 + 2]!, f.opacities[i]!);
    attrs[i * 3 + 1] = packQuaternion(f.rotations[i * 4]!, f.rotations[i * 4 + 1]!, f.rotations[i * 4 + 2]!, f.rotations[i * 4 + 3]!);
    attrs[i * 3 + 2] = packAttrWord2(f.colors[i * 3]!, f.colors[i * 3 + 1]!, f.colors[i * 3 + 2]!);
  }
  const degree = Math.max(0, Math.min(f.sh ? f.shDegree : 0, shDegreeCap, 3));
  let sh: Uint8Array | undefined;
  if (degree > 0 && f.sh) {
    const stride = shStrideBytes(degree);
    const keep = shRestCoeffs(degree) * 3;
    const srcK = shRestCoeffs(f.shDegree) * 3;
    sh = new Uint8Array(n * stride);
    for (let i = 0; i < n; i++) for (let c = 0; c < keep; c++) sh[i * stride + c] = encodeShByte(f.sh[i * srcK + c]!);
  }
  return { count: n, shDegree: degree, flags: f.antialiased ? SplatFlags.Antialiased : 0, positionsQ, attrs, sh };
}

/** Encode one splat I-frame block from its quantized state (layout in @ares/core splat.ts). */
export function encodeSplatStateBlock(q: DecodedSplat): Uint8Array {
  const n = q.count;
  const encPos = n ? MeshoptEncoder.encodeVertexBuffer(new Uint8Array(q.positionsQ.buffer, q.positionsQ.byteOffset, n * 8), n, 8) : new Uint8Array(0);
  const encAttr = n ? MeshoptEncoder.encodeVertexBuffer(new Uint8Array(q.attrs.buffer, q.attrs.byteOffset, n * 12), n, SPLAT_ATTR_STRIDE) : new Uint8Array(0);
  const encSh = q.sh && q.shDegree > 0 && n ? MeshoptEncoder.encodeVertexBuffer(q.sh, n, shStrideBytes(q.shDegree)) : undefined;
  const w = new ByteWriter(encPos.byteLength + encAttr.byteLength + (encSh?.byteLength ?? 0) + 64);
  w.u32(n).u8(q.shDegree).u8(q.flags & 0xff).u16(0);
  w.u32(encPos.byteLength).bytes(encPos);
  w.u32(encAttr.byteLength).bytes(encAttr);
  if (q.shDegree > 0 && encSh) w.u32(encSh.byteLength).bytes(encSh);
  return w.finish();
}

/**
 * Encode one splat I-frame block. `shDegreeCap` truncates higher SH bands (coefficient-major
 * order means lower bands come first, so truncation is a prefix copy).
 */
export function encodeSplatBlock(f: SplatFrame, box: Aabb, bits: number, shDegreeCap = 3): Uint8Array {
  return encodeSplatStateBlock(quantizeSplatFrame(f, box, bits, shDegreeCap));
}

// ---- dynamic splat profile: correspondence + P-frames ---------------------------------------

export interface SplatMatch {
  /** for each current splat: index into the previous frame, or -1 (birth) */
  prevOf: Int32Array;
  survivors: number;
  births: number;
  deaths: number;
}

export interface SplatTemporalOptions {
  /** "index": same Gaussian set in the same order every frame (trainer-stable sequences);
   *  "nn": nearest-neighbour correspondence; "auto": index when counts match and it fits, else nn. */
  mode?: "auto" | "index" | "nn" | "off";
  /** nn match radius as a fraction of the GOP box diagonal (default 0.01) */
  matchDist?: number;
  /** below this survivor fraction the frame is coded intra instead (default 0.5) */
  minSurvive?: number;
}

/** Index correspondence: every splat survives at its own index; only valid when counts are equal. */
export function matchSplatsByIndex(prev: DecodedSplat, cur: DecodedSplat): SplatMatch | null {
  if (prev.count !== cur.count) return null;
  const prevOf = new Int32Array(cur.count);
  for (let i = 0; i < cur.count; i++) prevOf[i] = i;
  return { prevOf, survivors: cur.count, births: 0, deaths: 0 };
}

/**
 * Nearest-neighbour correspondence on QUANTIZED positions (so it reflects what is stored): each
 * current splat claims the closest unclaimed previous splat within `radiusQ` levels; the rest
 * are births, and unclaimed previous splats are deaths. A uniform hash grid keeps it O(n).
 */
export function matchSplatsNearest(prev: DecodedSplat, cur: DecodedSplat, radiusQ: number): SplatMatch {
  const n = cur.count, m = prev.count;
  const prevOf = new Int32Array(n).fill(-1);
  if (!m || !n) return { prevOf, survivors: 0, births: n, deaths: m };
  const cell = Math.max(1, Math.ceil(radiusQ));
  const key = (x: number, y: number, z: number) => ((x / cell) | 0) * 73856093 ^ ((y / cell) | 0) * 19349663 ^ ((z / cell) | 0) * 83492791;
  const grid = new Map<number, number[]>();
  const pq = prev.positionsQ;
  for (let i = 0; i < m; i++) {
    const k = key(pq[i * 4]!, pq[i * 4 + 1]!, pq[i * 4 + 2]!);
    let arr = grid.get(k); if (!arr) { arr = []; grid.set(k, arr); } arr.push(i);
  }
  const claimed = new Uint8Array(m);
  const cq = cur.positionsQ;
  const r2 = radiusQ * radiusQ;
  let survivors = 0;
  for (let i = 0; i < n; i++) {
    const x = cq[i * 4]!, y = cq[i * 4 + 1]!, z = cq[i * 4 + 2]!;
    let best = -1, bestD = r2 + 1;
    const cx = (x / cell) | 0, cy = (y / cell) | 0, cz = (z / cell) | 0;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const arr = grid.get(((cx + dx) * 73856093) ^ ((cy + dy) * 19349663) ^ ((cz + dz) * 83492791));
      if (!arr) continue;
      for (const j of arr) {
        if (claimed[j]) continue;
        const ex = pq[j * 4]! - x, ey = pq[j * 4 + 1]! - y, ez = pq[j * 4 + 2]! - z;
        const d = ex * ex + ey * ey + ez * ez;
        if (d < bestD) { bestD = d; best = j; }
      }
    }
    if (best >= 0 && bestD <= r2) { prevOf[i] = best; claimed[best] = 1; survivors++; }
  }
  return { prevOf, survivors, births: n - survivors, deaths: m - survivors };
}

/**
 * Reorder `cur` into the container's P-frame order — survivors in the previous frame's order,
 * births after — and return that state plus the per-survivor previous index.
 */
export function orderForPFrame(prev: DecodedSplat, cur: DecodedSplat, match: SplatMatch): { state: DecodedSplat; survivorPrev: Int32Array; deaths: Int32Array } {
  const curOfPrev = new Int32Array(prev.count).fill(-1);
  for (let i = 0; i < cur.count; i++) if (match.prevOf[i]! >= 0) curOfPrev[match.prevOf[i]!] = i;
  const order: number[] = [];
  const survivorPrev: number[] = [];
  const deaths: number[] = [];
  for (let j = 0; j < prev.count; j++) { const i = curOfPrev[j]!; if (i >= 0) { order.push(i); survivorPrev.push(j); } else deaths.push(j); }
  for (let i = 0; i < cur.count; i++) if (match.prevOf[i]! < 0) order.push(i);
  const stride = cur.shDegree ? shStrideBytes(cur.shDegree) : 0;
  const state: DecodedSplat = {
    count: cur.count, shDegree: cur.shDegree, flags: cur.flags,
    positionsQ: new Uint16Array(cur.count * 4), attrs: new Uint32Array(cur.count * 3), sh: cur.sh ? new Uint8Array(cur.count * stride) : undefined,
  };
  order.forEach((src, dst) => {
    state.positionsQ.set(cur.positionsQ.subarray(src * 4, src * 4 + 4), dst * 4);
    state.attrs.set(cur.attrs.subarray(src * 3, src * 3 + 3), dst * 3);
    if (state.sh && cur.sh) state.sh.set(cur.sh.subarray(src * stride, (src + 1) * stride), dst * stride);
  });
  return { state, survivorPrev: Int32Array.from(survivorPrev), deaths: Int32Array.from(deaths) };
}

/** Encode a P-frame block (layout: @ares/core geometry.ts decodeSplatPBlock). `state` must be in P-frame order. */
export function encodeSplatPBlock(prev: DecodedSplat, state: DecodedSplat, survivorPrev: Int32Array, deaths: Int32Array): Uint8Array {
  const survivors = survivorPrev.length, births = state.count - survivors;
  const stride = state.shDegree ? shStrideBytes(state.shDegree) : 0;
  const delta = new Int16Array(survivors * 4);
  for (let s = 0; s < survivors; s++) {
    const j = survivorPrev[s]!;
    for (let a = 0; a < 3; a++) {
      let d = state.positionsQ[s * 4 + a]! - prev.positionsQ[j * 4 + a]!;
      if (d > 32767) d -= 65536; else if (d < -32768) d += 65536;   // decoder adds mod 2^16
      delta[s * 4 + a] = d;
    }
  }
  const enc = (buf: Uint8Array, count: number, st: number) => (count ? MeshoptEncoder.encodeVertexBuffer(buf, count, st) : new Uint8Array(0));
  const eDelta = enc(new Uint8Array(delta.buffer), survivors, 8);
  // Survivors' attrs + SH as byte-wise deltas vs the previous frame (decoder adds mod 256).
  const attrDelta = new Uint8Array(survivors * 12);
  const curA = new Uint8Array(state.attrs.buffer, state.attrs.byteOffset, state.count * 12);
  const prevA = new Uint8Array(prev.attrs.buffer, prev.attrs.byteOffset, prev.count * 12);
  for (let s = 0; s < survivors; s++) { const j = survivorPrev[s]!; for (let b = 0; b < 12; b++) attrDelta[s * 12 + b] = (curA[s * 12 + b]! - prevA[j * 12 + b]!) & 0xff; }
  const eAttr = enc(attrDelta, survivors, SPLAT_ATTR_STRIDE);
  let eSh: Uint8Array | null = null;
  if (stride) {
    const shDelta = new Uint8Array(survivors * stride);
    for (let s = 0; s < survivors; s++) { const j = survivorPrev[s]!; for (let b = 0; b < stride; b++) shDelta[s * stride + b] = (state.sh![s * stride + b]! - prev.sh![j * stride + b]!) & 0xff; }
    eSh = enc(shDelta, survivors, stride);
  }
  const eBPos = enc(new Uint8Array(state.positionsQ.buffer, state.positionsQ.byteOffset + survivors * 8, births * 8), births, 8);
  const eBAttr = enc(new Uint8Array(state.attrs.buffer, state.attrs.byteOffset + survivors * 12, births * 12), births, SPLAT_ATTR_STRIDE);
  const eBSh = stride ? enc(state.sh!.subarray(survivors * stride), births, stride) : null;
  const w = new ByteWriter(eDelta.byteLength + eAttr.byteLength + (eSh?.byteLength ?? 0) + eBPos.byteLength + eBAttr.byteLength + (eBSh?.byteLength ?? 0) + deaths.length * 3 + 64);
  w.u32(state.count).u8(state.shDegree).u8(state.flags & 0xff).u16(0);
  w.u32(deaths.length);
  let last = -1;
  for (const d of deaths) { w.varint(d - last - 1); last = d; }
  w.u32(survivors);
  w.u32(eDelta.byteLength).bytes(eDelta);
  w.u32(eAttr.byteLength).bytes(eAttr);
  if (eSh) w.u32(eSh.byteLength).bytes(eSh);
  w.u32(births);
  w.u32(eBPos.byteLength).bytes(eBPos);
  w.u32(eBAttr.byteLength).bytes(eBAttr);
  if (eBSh) w.u32(eBSh.byteLength).bytes(eBSh);
  return w.finish();
}

/** Inverse of the block encode (dequantize a decoded block back to a float SplatFrame) — export/tests. */
export function decodedSplatToFrame(d: DecodedSplat, box: Aabb, invLevels: number): SplatFrame {
  const f = emptySplatFrame(d.count, d.shDegree);
  const sx = (box.max[0] - box.min[0]) * invLevels, sy = (box.max[1] - box.min[1]) * invLevels, sz = (box.max[2] - box.min[2]) * invLevels;
  const q: [number, number, number, number] = [0, 0, 0, 0];
  const k3 = shRestCoeffs(d.shDegree) * 3;
  const stride = d.shDegree ? shStrideBytes(d.shDegree) : 0;
  for (let i = 0; i < d.count; i++) {
    f.positions[i * 3] = box.min[0] + d.positionsQ[i * 4]! * sx;
    f.positions[i * 3 + 1] = box.min[1] + d.positionsQ[i * 4 + 1]! * sy;
    f.positions[i * 3 + 2] = box.min[2] + d.positionsQ[i * 4 + 2]! * sz;
    const w0 = d.attrs[i * 3]!, w1 = d.attrs[i * 3 + 1]!, w2 = d.attrs[i * 3 + 2]!;
    f.scales[i * 3] = decodeScaleByte(w0 & 0xff);
    f.scales[i * 3 + 1] = decodeScaleByte((w0 >>> 8) & 0xff);
    f.scales[i * 3 + 2] = decodeScaleByte((w0 >>> 16) & 0xff);
    f.opacities[i] = ((w0 >>> 24) & 0xff) / 255;
    unpackQuaternion(w1, q);
    f.rotations[i * 4] = q[0]; f.rotations[i * 4 + 1] = q[1]; f.rotations[i * 4 + 2] = q[2]; f.rotations[i * 4 + 3] = q[3];
    f.colors[i * 3] = (w2 & 0xff) / 255; f.colors[i * 3 + 1] = ((w2 >>> 8) & 0xff) / 255; f.colors[i * 3 + 2] = ((w2 >>> 16) & 0xff) / 255;
    if (k3 && d.sh && f.sh) for (let c = 0; c < k3; c++) f.sh[i * k3 + c] = decodeShByte(d.sh[i * stride + c]!);
  }
  f.antialiased = (d.flags & SplatFlags.Antialiased) !== 0;
  return f;
}

/** Quaternion product a·b (xyzw). */
export function quatMul(a: ArrayLike<number>, b: ArrayLike<number>): [number, number, number, number] {
  const ax = a[0]!, ay = a[1]!, az = a[2]!, aw = a[3]!;
  const bx = b[0]!, by = b[1]!, bz = b[2]!, bw = b[3]!;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/** Rotation quaternion (xyzw) of a column-major mat4's upper 3×3, assuming uniform scale. */
export function quatFromMat4(m: ArrayLike<number>): [number, number, number, number] {
  const s0 = Math.hypot(m[0]!, m[1]!, m[2]!) || 1, s1 = Math.hypot(m[4]!, m[5]!, m[6]!) || 1, s2 = Math.hypot(m[8]!, m[9]!, m[10]!) || 1;
  const r00 = m[0]! / s0, r10 = m[1]! / s0, r20 = m[2]! / s0;
  const r01 = m[4]! / s1, r11 = m[5]! / s1, r21 = m[6]! / s1;
  const r02 = m[8]! / s2, r12 = m[9]! / s2, r22 = m[10]! / s2;
  const tr = r00 + r11 + r22;
  let x: number, y: number, z: number, w: number;
  if (tr > 0) {
    const S = Math.sqrt(tr + 1) * 2;
    w = 0.25 * S; x = (r21 - r12) / S; y = (r02 - r20) / S; z = (r10 - r01) / S;
  } else if (r00 > r11 && r00 > r22) {
    const S = Math.sqrt(1 + r00 - r11 - r22) * 2;
    w = (r21 - r12) / S; x = 0.25 * S; y = (r01 + r10) / S; z = (r02 + r20) / S;
  } else if (r11 > r22) {
    const S = Math.sqrt(1 + r11 - r00 - r22) * 2;
    w = (r02 - r20) / S; x = (r01 + r10) / S; y = 0.25 * S; z = (r12 + r21) / S;
  } else {
    const S = Math.sqrt(1 + r22 - r00 - r11) * 2;
    w = (r10 - r01) / S; x = (r02 + r20) / S; y = (r12 + r21) / S; z = 0.25 * S;
  }
  const l = Math.hypot(x, y, z, w) || 1;
  return [x / l, y / l, z / l, w / l];
}

/**
 * Apply a column-major mat4 (rotation + uniform scale + translation) to a splat frame in place:
 * positions through the matrix, rotations pre-multiplied by its rotation, scales by its uniform
 * scale. Non-uniform scale is approximated by the mean axis scale (splat ellipsoids cannot shear).
 */
export function transformSplatFrame(f: SplatFrame, m: Float32Array): void {
  const qR = quatFromMat4(m);
  const s0 = Math.hypot(m[0]!, m[1]!, m[2]!), s1 = Math.hypot(m[4]!, m[5]!, m[6]!), s2 = Math.hypot(m[8]!, m[9]!, m[10]!);
  const scale = (s0 + s1 + s2) / 3 || 1;
  for (let i = 0; i < f.count; i++) {
    const x = f.positions[i * 3]!, y = f.positions[i * 3 + 1]!, z = f.positions[i * 3 + 2]!;
    f.positions[i * 3] = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
    f.positions[i * 3 + 1] = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
    f.positions[i * 3 + 2] = m[2]! * x + m[6]! * y + m[10]! * z + m[14]!;
    const q = quatMul(qR, f.rotations.subarray(i * 4, i * 4 + 4));
    f.rotations[i * 4] = q[0]; f.rotations[i * 4 + 1] = q[1]; f.rotations[i * 4 + 2] = q[2]; f.rotations[i * 4 + 3] = q[3];
    f.scales[i * 3] = f.scales[i * 3]! * scale; f.scales[i * 3 + 1] = f.scales[i * 3 + 1]! * scale; f.scales[i * 3 + 2] = f.scales[i * 3 + 2]! * scale;
  }
}
