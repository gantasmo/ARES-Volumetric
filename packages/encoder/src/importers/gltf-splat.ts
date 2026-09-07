/**
 * glTF 2.0 + KHR_gaussian_splatting importer/exporter (Khronos, release candidate 2026).
 *
 * Contract (extension README): a mesh primitive with `mode: 0` (POINTS) carrying
 *   POSITION                                  VEC3 float
 *   KHR_gaussian_splatting:ROTATION           VEC4 float | normalized i8/i16          (x, y, z, w)
 *   KHR_gaussian_splatting:SCALE              VEC3 float | u8/u16 (normalized or not)  LINEAR, ≥ 0
 *   KHR_gaussian_splatting:OPACITY            SCALAR float | normalized u8/u16         [0, 1]
 *   KHR_gaussian_splatting:SH_DEGREE_0_COEF_0 VEC3 float                               sh0 (colour = 0.5 + C0·sh0)
 *   KHR_gaussian_splatting:SH_DEGREE_{1..3}_COEF_{k}  VEC3 float, degrees complete and in order
 * and `extensions.KHR_gaussian_splatting = { kernel: "ellipse", colorSpace: "srgb_rec709_display" | "lin_rec709_display" }`,
 * with the extension listed in `extensionsUsed`. Reads .glb (JSON + BIN chunks) and .gltf (buffers as
 * data: URIs or files through `loadUri`). Writes .glb.
 */
import { sh0ToColor, colorToSh0, shRestCoeffs } from "@ares/core";
import { emptySplatFrame, quatMul, quatFromMat4, type SplatFrame } from "../splat-frame.js";

export const KHR_GS = "KHR_gaussian_splatting";
const GLB_MAGIC = 0x46546c67, CHUNK_JSON = 0x4e4f534a, CHUNK_BIN = 0x004e4942;

interface GltfJson {
  asset?: { version?: string };
  extensionsUsed?: string[];
  buffers?: { byteLength: number; uri?: string }[];
  bufferViews?: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
  accessors?: { bufferView?: number; byteOffset?: number; componentType: number; normalized?: boolean; count: number; type: string; sparse?: unknown }[];
  meshes?: { primitives: { attributes: Record<string, number>; mode?: number; extensions?: Record<string, { kernel?: string; colorSpace?: string }> }[] }[];
  nodes?: { mesh?: number; matrix?: number[]; translation?: number[]; rotation?: number[]; scale?: number[]; children?: number[] }[];
  scenes?: { nodes?: number[] }[];
}

const NUM_COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const COMPONENT_SIZE: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };

export interface GltfSplatOptions {
  /** Resolve a relative buffer URI (for .gltf with external .bin). Not needed for .glb or data: URIs. */
  loadUri?: (uri: string) => Uint8Array;
}

export function isGlb(buf: Uint8Array): boolean {
  return buf.length >= 12 && new DataView(buf.buffer, buf.byteOffset, buf.byteLength).getUint32(0, true) === GLB_MAGIC;
}

function splitGlb(buf: Uint8Array): { json: GltfJson; bin: Uint8Array | null } {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (dv.getUint32(4, true) !== 2) throw new Error("GLB: only version 2 is supported");
  const len = dv.getUint32(8, true);
  if (len > buf.length) throw new Error("GLB: declared length exceeds the file");
  let off = 12;
  let json: GltfJson | null = null, bin: Uint8Array | null = null;
  while (off + 8 <= len) {
    const clen = dv.getUint32(off, true), ctype = dv.getUint32(off + 4, true);
    if (off + 8 + clen > len) throw new Error("GLB: chunk out of bounds");
    const data = buf.subarray(off + 8, off + 8 + clen);
    if (ctype === CHUNK_JSON) json = JSON.parse(new TextDecoder().decode(data)) as GltfJson;
    else if (ctype === CHUNK_BIN && !bin) bin = data;
    off += 8 + clen;
  }
  if (!json) throw new Error("GLB: no JSON chunk");
  return { json, bin };
}

function decodeDataUri(uri: string): Uint8Array {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(uri);
  if (!m) throw new Error("glTF: unsupported buffer uri");
  return m[2] ? new Uint8Array(Buffer.from(m[3]!, "base64")) : new Uint8Array(Buffer.from(decodeURIComponent(m[3]!), "latin1"));
}

class Reader {
  private buffers: (Uint8Array | null)[];
  constructor(private json: GltfJson, bin: Uint8Array | null, opts: GltfSplatOptions) {
    this.buffers = (json.buffers ?? []).map((b, i) => {
      if (b.uri == null) return i === 0 ? bin : null;
      if (b.uri.startsWith("data:")) return decodeDataUri(b.uri);
      if (!opts.loadUri) throw new Error(`glTF: external buffer "${b.uri}" needs loadUri`);
      return opts.loadUri(b.uri);
    });
  }
  /** Read an accessor as floats (normalized types mapped per the glTF rules). */
  accessor(index: number): { data: Float32Array; comps: number; count: number } {
    const a = this.json.accessors?.[index];
    if (!a) throw new Error(`glTF: accessor ${index} missing`);
    if (a.sparse) throw new Error("glTF: sparse accessors are not supported");
    const comps = NUM_COMPONENTS[a.type];
    const csize = COMPONENT_SIZE[a.componentType];
    if (!comps || !csize) throw new Error(`glTF: accessor ${index} has unsupported type ${a.type}/${a.componentType}`);
    const out = new Float32Array(a.count * comps);
    if (a.bufferView == null) return { data: out, comps, count: a.count }; // all zeros per spec
    const bv = this.json.bufferViews?.[a.bufferView];
    if (!bv) throw new Error(`glTF: bufferView ${a.bufferView} missing`);
    const buf = this.buffers[bv.buffer];
    if (!buf) throw new Error(`glTF: buffer ${bv.buffer} unavailable`);
    const stride = bv.byteStride ?? comps * csize;
    const base = (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
    if (base + (a.count - 1) * stride + comps * csize > buf.byteLength) throw new Error(`glTF: accessor ${index} out of bounds`);
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const norm = !!a.normalized;
    for (let i = 0; i < a.count; i++) {
      const o = base + i * stride;
      for (let c = 0; c < comps; c++) {
        const p = o + c * csize;
        let v: number;
        switch (a.componentType) {
          case 5120: v = dv.getInt8(p); if (norm) v = Math.max(v / 127, -1); break;
          case 5121: v = dv.getUint8(p); if (norm) v /= 255; break;
          case 5122: v = dv.getInt16(p, true); if (norm) v = Math.max(v / 32767, -1); break;
          case 5123: v = dv.getUint16(p, true); if (norm) v /= 65535; break;
          case 5125: v = dv.getUint32(p, true); break;
          default: v = dv.getFloat32(p, true);
        }
        out[i * comps + c] = v;
      }
    }
    return { data: out, comps, count: a.count };
  }
}

/** World matrix (column-major) of the first node referencing `meshIndex`, or null when none / identity. */
function nodeMatrixForMesh(json: GltfJson, meshIndex: number): Float32Array | null {
  const nodes = json.nodes ?? [];
  const parent = new Map<number, number>();
  nodes.forEach((n, i) => n.children?.forEach((c) => parent.set(c, i)));
  const local = (n: GltfJson["nodes"] extends (infer T)[] | undefined ? T : never): Float32Array => {
    if (n.matrix) return Float32Array.from(n.matrix);
    const t = n.translation ?? [0, 0, 0], r = n.rotation ?? [0, 0, 0, 1], s = n.scale ?? [1, 1, 1];
    const [x, y, z, w] = r as [number, number, number, number];
    const m = new Float32Array(16);
    m[0] = (1 - 2 * (y * y + z * z)) * s[0]!; m[1] = 2 * (x * y + w * z) * s[0]!; m[2] = 2 * (x * z - w * y) * s[0]!;
    m[4] = 2 * (x * y - w * z) * s[1]!; m[5] = (1 - 2 * (x * x + z * z)) * s[1]!; m[6] = 2 * (y * z + w * x) * s[1]!;
    m[8] = 2 * (x * z + w * y) * s[2]!; m[9] = 2 * (y * z - w * x) * s[2]!; m[10] = (1 - 2 * (x * x + y * y)) * s[2]!;
    m[12] = t[0]!; m[13] = t[1]!; m[14] = t[2]!; m[15] = 1;
    return m;
  };
  const mul = (a: Float32Array, b: Float32Array): Float32Array => {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r]! * b[c * 4]! + a[4 + r]! * b[c * 4 + 1]! + a[8 + r]! * b[c * 4 + 2]! + a[12 + r]! * b[c * 4 + 3]!;
    return o;
  };
  const idx = nodes.findIndex((n) => n.mesh === meshIndex);
  if (idx < 0) return null;
  let m = local(nodes[idx]!);
  let p = parent.get(idx);
  while (p != null) { m = mul(local(nodes[p]!), m); p = parent.get(p); }
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  return I.every((v, i) => Math.abs(m[i]! - v) < 1e-7) ? null : m;
}

/** Parse a .glb or .gltf (bytes) into one SplatFrame (all KHR_gaussian_splatting primitives concatenated). */
export function parseGltfSplat(bytes: Uint8Array, opts: GltfSplatOptions = {}): SplatFrame {
  let json: GltfJson, bin: Uint8Array | null = null;
  if (isGlb(bytes)) ({ json, bin } = splitGlb(bytes));
  else json = JSON.parse(new TextDecoder().decode(bytes)) as GltfJson;
  const rd = new Reader(json, bin, opts);
  const parts: SplatFrame[] = [];
  (json.meshes ?? []).forEach((mesh, mi) => {
    const xf = nodeMatrixForMesh(json, mi);
    for (const prim of mesh.primitives) {
      const ext = prim.extensions?.[KHR_GS];
      if (!ext) continue;
      if ((prim.mode ?? 4) !== 0) throw new Error(`glTF: ${KHR_GS} primitive must use mode 0 (POINTS)`);
      if (ext.kernel && ext.kernel !== "ellipse") throw new Error(`glTF: ${KHR_GS} kernel "${ext.kernel}" unsupported`);
      const A = prim.attributes;
      const need = (name: string) => { const i = A[name]; if (i == null) throw new Error(`glTF: ${KHR_GS} primitive lacks ${name}`); return rd.accessor(i); };
      const pos = need("POSITION");
      const n = pos.count;
      const rot = need(`${KHR_GS}:ROTATION`), scl = need(`${KHR_GS}:SCALE`), opa = need(`${KHR_GS}:OPACITY`);
      const sh0i = A[`${KHR_GS}:SH_DEGREE_0_COEF_0`];
      const col0 = A["COLOR_0"];
      if (sh0i == null && col0 == null) throw new Error(`glTF: ${KHR_GS} primitive lacks SH_DEGREE_0_COEF_0`);
      const sh0 = sh0i != null ? rd.accessor(sh0i) : null;
      const color0 = sh0 ? null : rd.accessor(col0!);
      for (const acc of [rot, scl, opa]) if (acc.count !== n) throw new Error("glTF: attribute counts differ within a splat primitive");
      // Highest fully-present degree, lower degrees required.
      const perDegree = [0, 3, 5, 7];
      let degree = 0;
      for (let d = 1; d <= 3; d++) {
        const ok = Array.from({ length: perDegree[d]! }, (_, k) => A[`${KHR_GS}:SH_DEGREE_${d}_COEF_${k}`] != null).every(Boolean);
        if (ok) degree = d; else break;
      }
      const f = emptySplatFrame(n, degree);
      f.positions.set(pos.data);
      f.rotations.set(rot.data);
      f.scales.set(scl.data);
      f.opacities.set(opa.data);
      if (sh0) for (let i = 0; i < n * 3; i++) f.colors[i] = sh0ToColor(sh0.data[i]!);
      else for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) f.colors[i * 3 + c] = color0!.data[i * color0!.comps + c]!;
      if (degree > 0 && f.sh) {
        const k3 = shRestCoeffs(degree) * 3;
        let k = 0;
        for (let d = 1; d <= degree; d++) for (let c = 0; c < perDegree[d]!; c++, k++) {
          const acc = rd.accessor(A[`${KHR_GS}:SH_DEGREE_${d}_COEF_${c}`]!);
          for (let i = 0; i < n; i++) { f.sh[i * k3 + k * 3] = acc.data[i * 3]!; f.sh[i * k3 + k * 3 + 1] = acc.data[i * 3 + 1]!; f.sh[i * k3 + k * 3 + 2] = acc.data[i * 3 + 2]!; }
        }
      }
      for (let i = 0; i < n; i++) { // renormalize — readers must not assume it, writers should guarantee it
        const x = f.rotations[i * 4]!, y = f.rotations[i * 4 + 1]!, z = f.rotations[i * 4 + 2]!, w = f.rotations[i * 4 + 3]!;
        const l = Math.hypot(x, y, z, w) || 1;
        f.rotations[i * 4] = x / l; f.rotations[i * 4 + 1] = y / l; f.rotations[i * 4 + 2] = z / l; f.rotations[i * 4 + 3] = w / l;
      }
      if (ext.colorSpace === "lin_rec709_display") {
        // Stored colours are display-referred in ARES; encode linear to sRGB for the base colour.
        const toSrgb = (v: number) => (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(Math.max(0, v), 1 / 2.4) - 0.055);
        for (let i = 0; i < n * 3; i++) f.colors[i] = toSrgb(f.colors[i]!);
      }
      if (xf) {
        const qR = quatFromMat4(xf);
        const s = (Math.hypot(xf[0]!, xf[1]!, xf[2]!) + Math.hypot(xf[4]!, xf[5]!, xf[6]!) + Math.hypot(xf[8]!, xf[9]!, xf[10]!)) / 3 || 1;
        for (let i = 0; i < n; i++) {
          const x = f.positions[i * 3]!, y = f.positions[i * 3 + 1]!, z = f.positions[i * 3 + 2]!;
          f.positions[i * 3] = xf[0]! * x + xf[4]! * y + xf[8]! * z + xf[12]!;
          f.positions[i * 3 + 1] = xf[1]! * x + xf[5]! * y + xf[9]! * z + xf[13]!;
          f.positions[i * 3 + 2] = xf[2]! * x + xf[6]! * y + xf[10]! * z + xf[14]!;
          const q = quatMul(qR, f.rotations.subarray(i * 4, i * 4 + 4));
          f.rotations.set(q, i * 4);
          f.scales[i * 3] = f.scales[i * 3]! * s; f.scales[i * 3 + 1] = f.scales[i * 3 + 1]! * s; f.scales[i * 3 + 2] = f.scales[i * 3 + 2]! * s;
        }
      }
      parts.push(f);
    }
  });
  if (!parts.length) throw new Error(`glTF: no ${KHR_GS} primitives found`);
  if (parts.length === 1) return parts[0]!;
  const degree = Math.min(...parts.map((p) => p.shDegree));
  const total = parts.reduce((s, p) => s + p.count, 0);
  const out = emptySplatFrame(total, degree);
  const k3 = shRestCoeffs(degree) * 3;
  let o = 0;
  for (const p of parts) {
    out.positions.set(p.positions, o * 3); out.scales.set(p.scales, o * 3); out.rotations.set(p.rotations, o * 4);
    out.opacities.set(p.opacities, o); out.colors.set(p.colors, o * 3);
    if (k3 && p.sh && out.sh) { const pk = shRestCoeffs(p.shDegree) * 3; for (let i = 0; i < p.count; i++) for (let c = 0; c < k3; c++) out.sh[(o + i) * k3 + c] = p.sh[i * pk + c]!; }
    o += p.count;
  }
  return out;
}

/** Write a .glb with one KHR_gaussian_splatting POINTS primitive (float accessors, display colour space). */
export function writeGlbSplat(f: SplatFrame, generator = "ares-encoder"): Uint8Array {
  const n = f.count;
  const views: { data: Uint8Array; comps: number; type: string; min?: number[]; max?: number[] }[] = [];
  const attr: Record<string, number> = {};
  const push = (name: string, arr: Float32Array, comps: number, withBounds = false) => {
    const bytes = new Uint8Array(arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength));
    const v: (typeof views)[number] = { data: bytes, comps, type: comps === 1 ? "SCALAR" : `VEC${comps}` };
    if (withBounds) {
      v.min = Array.from({ length: comps }, () => Infinity); v.max = Array.from({ length: comps }, () => -Infinity);
      for (let i = 0; i < n; i++) for (let c = 0; c < comps; c++) { const x = arr[i * comps + c]!; if (x < v.min![c]!) v.min![c] = x; if (x > v.max![c]!) v.max![c] = x; }
      if (n === 0) { v.min = v.min.map(() => 0); v.max = v.max.map(() => 0); }
    }
    attr[name] = views.length;
    views.push(v);
  };
  push("POSITION", f.positions, 3, true);
  push(`${KHR_GS}:ROTATION`, f.rotations, 4);
  push(`${KHR_GS}:SCALE`, f.scales, 3);
  push(`${KHR_GS}:OPACITY`, f.opacities, 1);
  const sh0 = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) sh0[i] = colorToSh0(f.colors[i]!);
  push(`${KHR_GS}:SH_DEGREE_0_COEF_0`, sh0, 3);
  if (f.shDegree > 0 && f.sh) {
    const k3 = shRestCoeffs(f.shDegree) * 3;
    const perDegree = [0, 3, 5, 7];
    let k = 0;
    for (let d = 1; d <= f.shDegree; d++) for (let c = 0; c < perDegree[d]!; c++, k++) {
      const a = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) { a[i * 3] = f.sh[i * k3 + k * 3]!; a[i * 3 + 1] = f.sh[i * k3 + k * 3 + 1]!; a[i * 3 + 2] = f.sh[i * k3 + k * 3 + 2]!; }
      push(`${KHR_GS}:SH_DEGREE_${d}_COEF_${c}`, a, 3);
    }
  }
  // BIN layout: each view 4-byte aligned.
  let total = 0;
  const offsets = views.map((v) => { const o = total; total += (v.data.byteLength + 3) & ~3; return o; });
  const bin = new Uint8Array(total);
  views.forEach((v, i) => bin.set(v.data, offsets[i]!));
  const json = {
    asset: { version: "2.0", generator },
    extensionsUsed: [KHR_GS],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: "splats" }],
    meshes: [{ primitives: [{ attributes: attr, mode: 0, extensions: { [KHR_GS]: { kernel: "ellipse", colorSpace: "srgb_rec709_display" } } }] }],
    buffers: [{ byteLength: total }],
    bufferViews: views.map((v, i) => ({ buffer: 0, byteOffset: offsets[i], byteLength: v.data.byteLength })),
    accessors: views.map((v, i) => ({ bufferView: i, componentType: 5126, count: n, type: v.type, ...(v.min ? { min: v.min, max: v.max } : {}) })),
  };
  let jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jpad = (4 - (jsonBytes.length % 4)) % 4;
  if (jpad) { const p = new Uint8Array(jsonBytes.length + jpad); p.set(jsonBytes); p.fill(0x20, jsonBytes.length); jsonBytes = p; }
  const out = new Uint8Array(12 + 8 + jsonBytes.length + 8 + bin.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, GLB_MAGIC, true); dv.setUint32(4, 2, true); dv.setUint32(8, out.length, true);
  dv.setUint32(12, jsonBytes.length, true); dv.setUint32(16, CHUNK_JSON, true); out.set(jsonBytes, 20);
  const bo = 20 + jsonBytes.length;
  dv.setUint32(bo, bin.length, true); dv.setUint32(bo + 4, CHUNK_BIN, true); out.set(bin, bo + 8);
  return out;
}
