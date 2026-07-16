/**
 * Deterministic synthetic clips for the P1 demo (no capture required, reproducible).
 * Each frame carries full geometry (P1 is intra-only) with a temporally STABLE UV atlas
 * (spec §7.4) so the same still texture samples correctly across the animation.
 */
import type { EncodeMeshFrame } from "./geometry-encode.js";

export interface SynthClip {
  name: string;
  fps: number;
  frames: EncodeMeshFrame[];
}

type Vec3 = [number, number, number];
const norm = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};

/** Icosphere with spherical UVs. subdiv 5 → 10,242 verts; 6 → 40,962. */
function icosphere(subdiv: number): { positions: Float32Array; uvs: Float32Array; indices: Uint32Array } {
  const t = (1 + Math.sqrt(5)) / 2;
  const verts: Vec3[] = ([
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ] as Vec3[]).map(norm);
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
      verts.push(norm([(va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2, (va[2] + vb[2]) / 2]));
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
  const uvs = new Float32Array(verts.length * 2);
  verts.forEach((v, i) => {
    positions.set(v, i * 3);
    uvs[i * 2] = 0.5 + Math.atan2(v[2], v[0]) / (2 * Math.PI);
    uvs[i * 2 + 1] = 0.5 - Math.asin(Math.max(-1, Math.min(1, v[1]))) / Math.PI;
  });
  const indices = new Uint32Array(faces.length * 3);
  faces.forEach((f, i) => indices.set(f, i * 3));
  return { positions, uvs, indices };
}

/** Torus knot (p=2,q=3) as a closed u×v tube grid with natural UVs. */
function torusKnot(uSegs: number, vSegs: number, tube = 0.5): { positions: Float32Array; uvs: Float32Array; indices: Uint32Array } {
  const p = 2, q = 3;
  const positions = new Float32Array(uSegs * vSegs * 3);
  const uvs = new Float32Array(uSegs * vSegs * 2);
  for (let i = 0; i < uSegs; i++) {
    const u = (i / uSegs) * Math.PI * 2;
    const r = 2 + Math.cos(q * u);
    const cx = r * Math.cos(p * u), cy = r * Math.sin(p * u), cz = Math.sin(q * u);
    const e = 1e-3, r2 = 2 + Math.cos(q * (u + e));
    const tx = r2 * Math.cos(p * (u + e)) - cx, ty = r2 * Math.sin(p * (u + e)) - cy, tz = Math.sin(q * (u + e)) - cz;
    const tl = Math.hypot(tx, ty, tz) || 1;
    const tang: Vec3 = [tx / tl, ty / tl, tz / tl];
    let n: Vec3 = [tang[1], -tang[0], 0];
    const nl = Math.hypot(n[0], n[1], n[2]) || 1;
    n = [n[0] / nl, n[1] / nl, n[2] / nl];
    const b: Vec3 = [
      tang[1] * n[2] - tang[2] * n[1], tang[2] * n[0] - tang[0] * n[2], tang[0] * n[1] - tang[1] * n[0],
    ];
    for (let j = 0; j < vSegs; j++) {
      const v = (j / vSegs) * Math.PI * 2;
      const cv = Math.cos(v) * tube, sv = Math.sin(v) * tube;
      const k = i * vSegs + j;
      positions[k * 3] = cx + cv * n[0] + sv * b[0];
      positions[k * 3 + 1] = cy + cv * n[1] + sv * b[1];
      positions[k * 3 + 2] = cz + cv * n[2] + sv * b[2];
      uvs[k * 2] = (i / uSegs) * 4; // wrap the atlas a few times along the tube
      uvs[k * 2 + 1] = j / vSegs;
    }
  }
  const indices = new Uint32Array(uSegs * vSegs * 6);
  let w = 0;
  for (let i = 0; i < uSegs; i++) {
    const i2 = (i + 1) % uSegs;
    for (let j = 0; j < vSegs; j++) {
      const j2 = (j + 1) % vSegs;
      const a = i * vSegs + j, bb = i2 * vSegs + j, c = i2 * vSegs + j2, d = i * vSegs + j2;
      indices[w++] = a; indices[w++] = bb; indices[w++] = c;
      indices[w++] = a; indices[w++] = c; indices[w++] = d;
    }
  }
  return { positions, uvs, indices };
}

const wrapUV = (uv: Float32Array): Float32Array => {
  const out = new Float32Array(uv.length);
  for (let i = 0; i < uv.length; i++) { const f = uv[i]! - Math.floor(uv[i]!); out[i] = f; }
  return out;
};

/** Rotating + gently morphing torus knot: every vertex moves each frame (proves geometry streaming). */
function objectClip(frames: number, fps: number): SynthClip {
  const base = torusKnot(220, 40);
  const uvs = wrapUV(base.uvs);
  const out: EncodeMeshFrame[] = [];
  for (let f = 0; f < frames; f++) {
    const tt = f / fps;
    const th = 2 * Math.PI * 0.25 * tt;
    const c = Math.cos(th), s = Math.sin(th);
    const breathe = 1 + 0.06 * Math.sin(2 * Math.PI * 0.5 * tt);
    const pos = new Float32Array(base.positions.length);
    for (let i = 0; i < pos.length; i += 3) {
      const x = base.positions[i]! * breathe, y = base.positions[i + 1]! * breathe, z = base.positions[i + 2]!;
      pos[i] = x * c - z * s;
      pos[i + 1] = y;
      pos[i + 2] = x * s + z * c;
    }
    out.push({ positions: pos, uvs, indices: base.indices });
  }
  return { name: "object", fps, frames: out };
}

/** Talking-head stand-in: static topology sphere with localized pulsing. */
function talkClip(frames: number, fps: number): SynthClip {
  const base = icosphere(6);
  const out: EncodeMeshFrame[] = [];
  for (let f = 0; f < frames; f++) {
    const tt = f / fps;
    const mouth = Math.sin(2 * Math.PI * 2.2 * tt);
    const breath = 1 + 0.02 * Math.sin(2 * Math.PI * 0.3 * tt);
    const pos = new Float32Array(base.positions.length);
    for (let i = 0; i < pos.length; i += 3) {
      const x = base.positions[i]!, y = base.positions[i + 1]!, z = base.positions[i + 2]!;
      const md = y * -0.35 + z * 0.94;
      const r = (1 + 0.05 * mouth * Math.exp((md - 1) * 18)) * breath;
      pos[i] = x * r; pos[i + 1] = y * r; pos[i + 2] = z * r;
    }
    out.push({ positions: pos, uvs: base.uvs, indices: base.indices });
  }
  return { name: "talk", fps, frames: out };
}

export function synthClip(shape: "object" | "talk", frames: number, fps: number): SynthClip {
  return shape === "talk" ? talkClip(frames, fps) : objectClip(frames, fps);
}
