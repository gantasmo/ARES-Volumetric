/** Synthetic Phase 0 corpus following the spec §13.2 clip taxonomy (§6.4 classes).
 *
 * These are deterministic stand-ins so the intra bench runs with zero downloads;
 * real captures (PLY sequences) dropped into bench/data/<clip>/ take part automatically
 * and should replace these numbers before any public claim (spec §13.6 honesty clause).
 */
import { BenchMesh, icosphere, torusKnot, mulberry32 } from "./mesh.js";

export interface Clip {
  name: string;
  description: string;
  cls: "A" | "B" | "C";
  frames: BenchMesh[];
  synthetic: boolean;
}

/** Sum of k random Gaussian lobes on the unit sphere — smooth, scan-like surface detail. */
function makeLobes(rng: () => number, k: number, amp: number) {
  const cx: number[] = [], cy: number[] = [], cz: number[] = [], w: number[] = [], a: number[] = [];
  for (let i = 0; i < k; i++) {
    const u = rng() * 2 - 1, phi = rng() * Math.PI * 2;
    const r = Math.sqrt(1 - u * u);
    cx.push(r * Math.cos(phi)); cy.push(r * Math.sin(phi)); cz.push(u);
    w.push(4 + rng() * 24);
    a.push((rng() * 2 - 1) * amp);
  }
  return (x: number, y: number, z: number): number => {
    let s = 0;
    for (let i = 0; i < k; i++) {
      const d = x * cx[i]! + y * cy[i]! + z * cz[i]!; // cos(angle) on unit sphere
      s += a[i]! * Math.exp((d - 1) * w[i]!);
    }
    return s;
  };
}

/** `talk` — class A: static topology, gentle localized motion (spec: best case). */
function talkClip(frames: number): Clip {
  const base = icosphere(6); // 40,962 verts / 81,920 tris
  const rng = mulberry32(0xa11e5);
  const bump = makeLobes(rng, 32, 0.06);
  const out: BenchMesh[] = [];
  for (let f = 0; f < frames; f++) {
    const t = f / 30;
    const pos = new Float32Array(base.positions.length);
    const mouth = Math.sin(2 * Math.PI * 2.2 * t); // ~2 Hz speech-like
    const breath = 1 + 0.003 * Math.sin(2 * Math.PI * 0.3 * t);
    for (let i = 0; i < pos.length; i += 3) {
      const x = base.positions[i]!, y = base.positions[i + 1]!, z = base.positions[i + 2]!;
      // mouth region: lobe around (0, -0.35, 0.94)
      const md = x * 0 + y * -0.35 + z * 0.94;
      const r = 1 + bump(x, y, z) + 0.02 * mouth * Math.exp((md - 1) * 18) + (rng() - 0.5) * 1e-4;
      pos[i] = x * r * breath; pos[i + 1] = y * r * breath; pos[i + 2] = z * r * breath;
    }
    out.push({ positions: pos, indices: base.indices });
  }
  return { name: "talk", description: "40k verts, static topology, localized motion", cls: "A", frames: out, synthetic: true };
}

/** `dance` — class A/B: static topology, large fast deformation (stresses prediction). */
function danceClip(frames: number): Clip {
  const base = icosphere(6);
  const rng = mulberry32(0xda2ce);
  const bump = makeLobes(rng, 32, 0.05);
  const out: BenchMesh[] = [];
  for (let f = 0; f < frames; f++) {
    const t = f / 30;
    const pos = new Float32Array(base.positions.length);
    const twistA = 0.9 * Math.sin(2 * Math.PI * 0.8 * t);
    const bendA = 0.5 * Math.sin(2 * Math.PI * 0.53 * t + 1.3);
    for (let i = 0; i < pos.length; i += 3) {
      const x0 = base.positions[i]!, y = base.positions[i + 1]!, z0 = base.positions[i + 2]!;
      const r = 1 + bump(x0, y, z0) + (rng() - 0.5) * 1e-4;
      let x = x0 * r, z = z0 * r;
      const yy = y * r;
      // twist around Y proportional to height, then bend in XZ
      const th = twistA * yy;
      const c = Math.cos(th), s = Math.sin(th);
      const xt = x * c - z * s, zt = x * s + z * c;
      x = xt + bendA * (yy + 1) * 0.5;
      z = zt + 0.3 * Math.sin(2 * Math.PI * 0.67 * t) * (yy + 1) * 0.5;
      pos[i] = x; pos[i + 1] = yy; pos[i + 2] = z;
    }
    out.push({ positions: pos, indices: base.indices });
  }
  return { name: "dance", description: "40k verts, fast large deformation", cls: "A", frames: out, synthetic: true };
}

/** `two` — class C: two bodies, one progressively clipped (entry/exit → per-frame topology). */
function twoClip(frames: number): Clip {
  const sphere = icosphere(5); // 10,242 verts each
  const rng = mulberry32(0x2b0d1e5);
  const bumpA = makeLobes(rng, 24, 0.05);
  const bumpB = makeLobes(rng, 24, 0.05);
  const out: BenchMesh[] = [];
  const n = sphere.positions.length / 3;
  for (let f = 0; f < frames; f++) {
    const t = f / 30;
    const orbit = 2 * Math.PI * 0.25 * t;
    const ax = 1.4 * Math.cos(orbit), az = 1.4 * Math.sin(orbit);
    // body A: full sphere
    const posA = new Float32Array(sphere.positions.length);
    for (let i = 0; i < posA.length; i += 3) {
      const x = sphere.positions[i]!, y = sphere.positions[i + 1]!, z = sphere.positions[i + 2]!;
      const r = 1 + bumpA(x, y, z);
      posA[i] = x * r + ax; posA[i + 1] = y * r; posA[i + 2] = z * r + az;
    }
    // body B: clipped by a plane sweeping through it (enters/exits the stage)
    const clipX = -1.2 + 2.4 * Math.abs(Math.sin(2 * Math.PI * 0.18 * t));
    const keep = new Int32Array(n).fill(-1);
    const posBList: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = sphere.positions[i * 3]!, y = sphere.positions[i * 3 + 1]!, z = sphere.positions[i * 3 + 2]!;
      if (x > clipX - 1) { // keep verts on the visible side (plane in body-local space)
        const r = 1 + bumpB(x, y, z);
        keep[i] = posBList.length / 3;
        posBList.push(x * r - ax, y * r + 0.2, z * r - az);
      }
    }
    const idxB: number[] = [];
    for (let i = 0; i < sphere.indices.length; i += 3) {
      const a = keep[sphere.indices[i]!]!, b = keep[sphere.indices[i + 1]!]!, c = keep[sphere.indices[i + 2]!]!;
      if (a >= 0 && b >= 0 && c >= 0) idxB.push(a, b, c);
    }
    const offB = posA.length / 3;
    const positions = new Float32Array(posA.length + posBList.length);
    positions.set(posA);
    positions.set(posBList, posA.length);
    const indices = new Uint32Array(sphere.indices.length + idxB.length);
    indices.set(sphere.indices);
    for (let i = 0; i < idxB.length; i++) indices[sphere.indices.length + i] = idxB[i]! + offB;
    out.push({ positions, indices });
  }
  return { name: "two", description: "two bodies, occlusion/entry/exit, varying topology", cls: "C", frames: out, synthetic: true };
}

/** `object` — class A: rigid rotating object (texture-dominated in real life). */
function objectClip(frames: number): Clip {
  const base = torusKnot(256, 48); // 12,288 verts
  const out: BenchMesh[] = [];
  for (let f = 0; f < frames; f++) {
    const t = f / 30;
    const th = 2 * Math.PI * 0.4 * t;
    const c = Math.cos(th), s = Math.sin(th);
    const pos = new Float32Array(base.positions.length);
    for (let i = 0; i < pos.length; i += 3) {
      const x = base.positions[i]!, y = base.positions[i + 1]!, z = base.positions[i + 2]!;
      pos[i] = x * c - z * s;
      pos[i + 1] = y;
      pos[i + 2] = x * s + z * c;
    }
    out.push({ positions: pos, indices: base.indices });
  }
  return { name: "object", description: "12k verts, rigid rotation", cls: "A", frames: out, synthetic: true };
}

export function syntheticCorpus(frames: number): Clip[] {
  return [talkClip(frames), danceClip(frames), twoClip(frames), objectClip(frames)];
}
