/**
 * PLY importer (spec §14 P5): ascii and binary_little_endian, header + body decode.
 *
 * Meshes: positions + fan-triangulated faces, plus any of normals (nx ny nz), UVs (s t | u v |
 * texture_u texture_v) and colours (red green blue [alpha], uchar or float).
 *
 * 3D Gaussian splat PLYs (the INRIA 3DGS export that every trainer and Scaniverse/Polycam/Luma
 * write) are detected from their properties — f_dc_0..2, opacity, scale_0..2, rot_0..3, optional
 * f_rest_* — and decoded into a SplatFrame:
 *   colour   = 0.5 + C0·f_dc          opacity = sigmoid(opacity)        scale = exp(scale_i)
 *   rotation = (rot_1, rot_2, rot_3, rot_0) normalized   (the PLY stores w first)
 *   f_rest   channel-major (all R coefficients, then G, then B) → coefficient-major rgb
 * SuperSplat "compressed.ply" (a `chunk` element with packed_* properties) is refused with a
 * pointer to splat-transform, whose SOG output this package reads directly.
 * Coordinates are taken verbatim; the original 3DGS convention is Y-down, Z-forward, so `--rotate
 * 180,0,0` on the encode CLI turns it into ARES's Y-up world.
 */
import { SH_C0, sh0ToColor, colorToSh0, sigmoid, logit, shRestCoeffs } from "@ares/core";
import { emptySplatFrame, type SplatFrame } from "../splat-frame.js";

export interface PlyProperty { name: string; type: string; }
export interface PlyElement { name: string; count: number; props: PlyProperty[]; listProps: { name: string; countType: string; indexType: string }[]; }
export interface PlyHeader {
  format: "ascii" | "binary_little_endian" | "binary_big_endian";
  vertexCount: number;
  faceCount: number;
  vertexProps: PlyProperty[];
  /** count/index types of the face `property list` (e.g. uchar / int). */
  faceCountType: string;
  faceIndexType: string;
  dataOffset: number; // byte offset where the element data begins
  elements: PlyElement[];
}

export interface PlyMesh {
  positions: Float32Array; // xyz interleaved
  indices: Uint32Array;    // triangulated (fan) from face lists
  vertexCount: number;
  faceCount: number;       // source polygon count (pre-triangulation)
  uvs?: Float32Array;      // uv interleaved (when the file carries s/t, u/v or texture_u/v)
  normals?: Float32Array;  // xyz interleaved
  colors?: Float32Array;   // rgb interleaved 0..1
}

const dec = new TextDecoder("ascii");

const TYPE_SIZE: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};

/** Parse a PLY header from the start of `buf`. Throws on malformed input (untrusted). */
export function parsePlyHeader(buf: Uint8Array): PlyHeader {
  const end = indexOf(buf, "end_header");
  if (end < 0) throw new Error("PLY: no end_header");
  const nl = buf.indexOf(0x0a, end);
  const dataOffset = nl + 1;
  const lines = dec.decode(buf.subarray(0, end)).split(/\r?\n/);
  if (lines[0]?.trim() !== "ply") throw new Error("PLY: missing magic");
  let format: PlyHeader["format"] = "ascii";
  const elements: PlyElement[] = [];
  let cur: PlyElement | null = null;
  for (const line of lines.slice(1)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === "format") format = t[1] as PlyHeader["format"];
    else if (t[0] === "element") { cur = { name: t[1]!, count: Number(t[2]), props: [], listProps: [] }; elements.push(cur); }
    else if (t[0] === "property" && cur) {
      if (t[1] === "list") cur.listProps.push({ name: t[t.length - 1]!, countType: t[2]!, indexType: t[3]! });
      else cur.props.push({ type: t[1]!, name: t[t.length - 1]! });
    }
  }
  const vertex = elements.find((e) => e.name === "vertex");
  const face = elements.find((e) => e.name === "face");
  if (!vertex) throw new Error("PLY: no vertex element");
  if (vertex.listProps.length) throw new Error("PLY: list property on vertex element unsupported");
  const faceList = face?.listProps[0];
  return {
    format, vertexCount: vertex.count, faceCount: face?.count ?? 0, vertexProps: vertex.props,
    faceCountType: faceList?.countType ?? "uchar", faceIndexType: faceList?.indexType ?? "int", dataOffset, elements,
  };
}

/** True when the vertex element carries the 3DGS splat attributes. */
export function isSplatPlyHeader(h: PlyHeader): boolean {
  const names = new Set(h.vertexProps.map((p) => p.name));
  return ["f_dc_0", "opacity", "scale_0", "rot_0"].every((n) => names.has(n));
}
export function isSplatPly(buf: Uint8Array): boolean {
  try { return isSplatPlyHeader(parsePlyHeader(buf)); } catch { return false; }
}

function checkLayout(h: PlyHeader): void {
  if (h.format === "binary_big_endian") throw new Error("PLY: binary_big_endian unsupported");
  if (h.elements.some((e) => e.name === "chunk")) throw new Error("PLY: SuperSplat compressed.ply is not supported — convert it to SOG with splat-transform (read natively) or to a plain 3DGS .ply");
  const vi = h.elements.findIndex((e) => e.name === "vertex");
  if (vi !== 0) throw new Error(`PLY: element "${h.elements[0]?.name}" precedes vertex — unsupported layout`);
  const fi = h.elements.findIndex((e) => e.name === "face");
  if (fi > 1) throw new Error(`PLY: element "${h.elements[1]?.name}" sits between vertex and face — unsupported layout`);
}

function readScalar(dv: DataView, offset: number, type: string): number {
  switch (type) {
    case "char": case "int8": return dv.getInt8(offset);
    case "uchar": case "uint8": return dv.getUint8(offset);
    case "short": case "int16": return dv.getInt16(offset, true);
    case "ushort": case "uint16": return dv.getUint16(offset, true);
    case "int": case "int32": return dv.getInt32(offset, true);
    case "uint": case "uint32": return dv.getUint32(offset, true);
    case "float": case "float32": return dv.getFloat32(offset, true);
    case "double": case "float64": return dv.getFloat64(offset, true);
    default: throw new Error(`PLY: unknown scalar type ${type}`);
  }
}

/** Decode the requested vertex columns (missing names are simply absent from the result). Returns the byte offset after the vertex block. */
function readColumns(buf: Uint8Array, h: PlyHeader, wanted: string[]): { cols: Map<string, Float32Array>; types: Map<string, string>; end: number } {
  const cols = new Map<string, Float32Array>();
  const types = new Map<string, string>();
  const n = h.vertexCount;
  if (h.format === "ascii") {
    const text = new TextDecoder("ascii").decode(buf.subarray(h.dataOffset));
    const tok = text.split(/\s+/).filter((s) => s.length > 0);
    const nProps = h.vertexProps.length;
    const want = new Map<number, Float32Array>();
    h.vertexProps.forEach((p, j) => { if (wanted.includes(p.name)) { const a = new Float32Array(n); cols.set(p.name, a); types.set(p.name, p.type); want.set(j, a); } });
    let p = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nProps; j++) {
        const s = tok[p++];
        const a = want.get(j);
        if (a) { const v = Number(s); if (Number.isNaN(v)) throw new Error(`PLY: non-numeric token at ${p - 1}`); a[i] = v; }
      }
    }
    // ASCII faces are read by the caller from the same token stream: stash the token cursor.
    asciiCursor = { tok, p };
    return { cols, types, end: -1 };
  }
  let stride = 0;
  const readers: { arr: Float32Array; off: number; type: string }[] = [];
  for (const p of h.vertexProps) {
    const size = TYPE_SIZE[p.type];
    if (!size) throw new Error(`PLY: unknown property type ${p.type}`);
    if (wanted.includes(p.name)) { const a = new Float32Array(n); cols.set(p.name, a); types.set(p.name, p.type); readers.push({ arr: a, off: stride, type: p.type }); }
    stride += size;
  }
  if (h.dataOffset + n * stride > buf.byteLength) throw new Error("PLY: truncated vertex data");
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let off = h.dataOffset;
  for (let i = 0; i < n; i++) {
    for (const r of readers) r.arr[i] = readScalar(dv, off + r.off, r.type);
    off += stride;
  }
  return { cols, types, end: off };
}
let asciiCursor: { tok: string[]; p: number } | null = null;

function readFacesBinary(buf: Uint8Array, h: PlyHeader, off: number): Uint32Array {
  const countSize = TYPE_SIZE[h.faceCountType];
  const idxSize = TYPE_SIZE[h.faceIndexType];
  if (!countSize || !idxSize) throw new Error("PLY: unknown face list types");
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let tris = 0;
  let scan = off;
  for (let f = 0; f < h.faceCount; f++) {
    if (scan + countSize > buf.byteLength) throw new Error("PLY: truncated face data");
    const n = readScalar(dv, scan, h.faceCountType);
    if (n < 3) throw new Error(`PLY: face ${f} has ${n} vertices`);
    tris += n - 2;
    scan += countSize + n * idxSize;
    if (scan > buf.byteLength) throw new Error("PLY: truncated face data");
  }
  const indices = new Uint32Array(tris * 3);
  let w = 0;
  for (let f = 0; f < h.faceCount; f++) {
    const n = readScalar(dv, off, h.faceCountType);
    off += countSize;
    const i0 = readScalar(dv, off, h.faceIndexType);
    let prev = readScalar(dv, off + idxSize, h.faceIndexType);
    for (let k = 2; k < n; k++) {
      const cur = readScalar(dv, off + k * idxSize, h.faceIndexType);
      indices[w++] = i0; indices[w++] = prev; indices[w++] = cur;
      prev = cur;
    }
    off += n * idxSize;
  }
  return indices;
}

function readFacesAscii(h: PlyHeader): Uint32Array {
  const c = asciiCursor;
  if (!c) return new Uint32Array(0);
  const next = () => { const v = Number(c.tok[c.p++]); if (Number.isNaN(v)) throw new Error(`PLY: non-numeric token at ${c.p - 1}`); return v; };
  const triIdx: number[] = [];
  for (let f = 0; f < h.faceCount; f++) {
    const n = next();
    if (n < 3) throw new Error(`PLY: face ${f} has ${n} vertices`);
    const i0 = next();
    let prev = next();
    for (let k = 2; k < n; k++) { const cur = next(); triIdx.push(i0, prev, cur); prev = cur; }
  }
  return Uint32Array.from(triIdx);
}

const interleave3 = (a: Float32Array, b: Float32Array, c: Float32Array): Float32Array => {
  const out = new Float32Array(a.length * 3);
  for (let i = 0; i < a.length; i++) { out[i * 3] = a[i]!; out[i * 3 + 1] = b[i]!; out[i * 3 + 2] = c[i]!; }
  return out;
};

/** Decode a mesh PLY: positions + triangulated indices, plus normals/UVs/colours when present. */
export function parsePly(buf: Uint8Array): PlyMesh {
  const h = parsePlyHeader(buf);
  checkLayout(h);
  const { cols, types, end } = readColumns(buf, h, ["x", "y", "z", "nx", "ny", "nz", "s", "t", "u", "v", "texture_u", "texture_v", "red", "green", "blue"]);
  const x = cols.get("x"), y = cols.get("y"), z = cols.get("z");
  if (!x || !y || !z) throw new Error("PLY: vertex element lacks x/y/z");
  const positions = interleave3(x, y, z);
  const indices = h.format === "ascii" ? readFacesAscii(h) : readFacesBinary(buf, h, end);
  const mesh: PlyMesh = { positions, indices, vertexCount: h.vertexCount, faceCount: h.faceCount };
  const nx = cols.get("nx"), ny = cols.get("ny"), nz = cols.get("nz");
  if (nx && ny && nz) mesh.normals = interleave3(nx, ny, nz);
  const u = cols.get("s") ?? cols.get("u") ?? cols.get("texture_u"), v = cols.get("t") ?? cols.get("v") ?? cols.get("texture_v");
  if (u && v) { const uv = new Float32Array(h.vertexCount * 2); for (let i = 0; i < h.vertexCount; i++) { uv[i * 2] = u[i]!; uv[i * 2 + 1] = v[i]!; } mesh.uvs = uv; }
  const r = cols.get("red"), g = cols.get("green"), b = cols.get("blue");
  if (r && g && b) {
    const scale = /uchar|uint8/.test(types.get("red") ?? "") ? 1 / 255 : 1;
    const c = interleave3(r, g, b);
    if (scale !== 1) for (let i = 0; i < c.length; i++) c[i] = c[i]! * scale;
    mesh.colors = c;
  }
  return mesh;
}

/** Decode a 3DGS splat PLY into a SplatFrame. */
export function parsePlySplat(buf: Uint8Array): SplatFrame {
  const h = parsePlyHeader(buf);
  checkLayout(h);
  if (!isSplatPlyHeader(h)) throw new Error("PLY: not a 3D Gaussian splat PLY (needs f_dc_0, opacity, scale_0, rot_0)");
  const restNames = h.vertexProps.map((p) => p.name).filter((n) => /^f_rest_\d+$/.test(n));
  const K = Math.floor(restNames.length / 3);
  const degree = K >= 15 ? 3 : K >= 8 ? 2 : K >= 3 ? 1 : 0;
  const wanted = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", ...restNames];
  const { cols } = readColumns(buf, h, wanted);
  const get = (n: string): Float32Array => { const c = cols.get(n); if (!c) throw new Error(`PLY: missing ${n}`); return c; };
  const n = h.vertexCount;
  const f = emptySplatFrame(n, degree);
  f.positions.set(interleave3(get("x"), get("y"), get("z")));
  const dc = [get("f_dc_0"), get("f_dc_1"), get("f_dc_2")];
  const sc = [get("scale_0"), get("scale_1"), get("scale_2")];
  const rot = [get("rot_0"), get("rot_1"), get("rot_2"), get("rot_3")];
  const op = get("opacity");
  for (let i = 0; i < n; i++) {
    f.colors[i * 3] = sh0ToColor(dc[0]![i]!); f.colors[i * 3 + 1] = sh0ToColor(dc[1]![i]!); f.colors[i * 3 + 2] = sh0ToColor(dc[2]![i]!);
    f.scales[i * 3] = Math.exp(sc[0]![i]!); f.scales[i * 3 + 1] = Math.exp(sc[1]![i]!); f.scales[i * 3 + 2] = Math.exp(sc[2]![i]!);
    f.opacities[i] = sigmoid(op[i]!);
    const w = rot[0]![i]!, x = rot[1]![i]!, y = rot[2]![i]!, z = rot[3]![i]!;
    const l = Math.hypot(x, y, z, w) || 1;
    f.rotations[i * 4] = x / l; f.rotations[i * 4 + 1] = y / l; f.rotations[i * 4 + 2] = z / l; f.rotations[i * 4 + 3] = w / l;
  }
  if (degree > 0 && f.sh) {
    const k = shRestCoeffs(degree);           // coefficients per channel actually kept
    const k3 = k * 3;
    const rest = Array.from({ length: K * 3 }, (_, j) => cols.get(`f_rest_${j}`));
    for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) for (let c = 0; c < k; c++) {
      const col = rest[ch * K + c];            // channel-major in the file
      f.sh[i * k3 + c * 3 + ch] = col ? col[i]! : 0;
    }
  }
  return f;
}

/** Write a 3DGS-layout binary PLY (x y z nx ny nz f_dc_* [f_rest_*] opacity scale_* rot_*), the format every splat tool reads. */
export function writeSplatPly(f: SplatFrame): Uint8Array {
  const n = f.count;
  const K = shRestCoeffs(f.shDegree);
  const names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", ...Array.from({ length: K * 3 }, (_, j) => `f_rest_${j}`), "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
  const header = `ply\nformat binary_little_endian 1.0\ncomment generated by ares-encoder\nelement vertex ${n}\n${names.map((p) => `property float ${p}`).join("\n")}\nend_header\n`;
  const hb = new TextEncoder().encode(header);
  const stride = names.length * 4;
  const out = new Uint8Array(hb.length + n * stride);
  out.set(hb, 0);
  const dv = new DataView(out.buffer, hb.length);
  const k3 = K * 3;
  for (let i = 0; i < n; i++) {
    let o = i * stride;
    const put = (v: number) => { dv.setFloat32(o, v, true); o += 4; };
    put(f.positions[i * 3]!); put(f.positions[i * 3 + 1]!); put(f.positions[i * 3 + 2]!);
    put(0); put(0); put(0);
    put(colorToSh0(f.colors[i * 3]!)); put(colorToSh0(f.colors[i * 3 + 1]!)); put(colorToSh0(f.colors[i * 3 + 2]!));
    for (let ch = 0; ch < 3; ch++) for (let c = 0; c < K; c++) put(f.sh ? f.sh[i * k3 + c * 3 + ch]! : 0);
    put(logit(f.opacities[i]!));
    put(Math.log(Math.max(1e-12, f.scales[i * 3]!))); put(Math.log(Math.max(1e-12, f.scales[i * 3 + 1]!))); put(Math.log(Math.max(1e-12, f.scales[i * 3 + 2]!)));
    put(f.rotations[i * 4 + 3]!); put(f.rotations[i * 4]!); put(f.rotations[i * 4 + 1]!); put(f.rotations[i * 4 + 2]!);
  }
  return out;
}

function indexOf(buf: Uint8Array, needle: string): number {
  const n = new TextEncoder().encode(needle);
  outer: for (let i = 0; i <= buf.length - n.length; i++) {
    for (let j = 0; j < n.length; j++) if (buf[i + j] !== n[j]) continue outer;
    return i;
  }
  return -1;
}

export { SH_C0 };
