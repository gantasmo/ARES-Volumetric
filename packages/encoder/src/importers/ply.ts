/** Minimal PLY importer (spec §14 P5). Header + body decode for ascii and binary_little_endian. */

export interface PlyProperty { name: string; type: string; }
export interface PlyHeader {
  format: "ascii" | "binary_little_endian" | "binary_big_endian";
  vertexCount: number;
  faceCount: number;
  vertexProps: PlyProperty[];
  /** count/index types of the face `property list` (e.g. uchar / int). */
  faceCountType: string;
  faceIndexType: string;
  dataOffset: number; // byte offset where the element data begins
}

export interface PlyMesh {
  positions: Float32Array; // xyz interleaved
  indices: Uint32Array;    // triangulated (fan) from face lists
  vertexCount: number;
  faceCount: number;       // source polygon count (pre-triangulation)
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
  // Header is ASCII up to and including the "end_header\n" line.
  const end = indexOf(buf, "end_header");
  if (end < 0) throw new Error("PLY: no end_header");
  const nl = buf.indexOf(0x0a, end);
  const dataOffset = nl + 1;
  const lines = dec.decode(buf.subarray(0, end)).split(/\r?\n/);

  if (lines[0]?.trim() !== "ply") throw new Error("PLY: missing magic");
  let format: PlyHeader["format"] = "ascii";
  let vertexCount = 0, faceCount = 0;
  let faceCountType = "uchar", faceIndexType = "int";
  const vertexProps: PlyProperty[] = [];
  let current: "vertex" | "face" | "" = "";

  for (const line of lines.slice(1)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === "format") format = t[1] as PlyHeader["format"];
    else if (t[0] === "element") {
      current = t[1] === "vertex" ? "vertex" : t[1] === "face" ? "face" : "";
      if (t[1] === "vertex") vertexCount = Number(t[2]);
      else if (t[1] === "face") faceCount = Number(t[2]);
    } else if (t[0] === "property" && current === "vertex") {
      if (t[1] === "list") throw new Error("PLY: list property on vertex element unsupported");
      vertexProps.push({ type: t[1]!, name: t[t.length - 1]! });
    } else if (t[0] === "property" && current === "face" && t[1] === "list") {
      faceCountType = t[2]!;
      faceIndexType = t[3]!;
    }
  }
  return { format, vertexCount, faceCount, vertexProps, faceCountType, faceIndexType, dataOffset };
}

/** Decode positions + triangulated indices. Supports ascii and binary_little_endian (N6: throws on malformed). */
export function parsePly(buf: Uint8Array): PlyMesh {
  const h = parsePlyHeader(buf);
  if (h.format === "binary_big_endian") throw new Error("PLY: binary_big_endian unsupported");
  return h.format === "ascii" ? parseAsciiBody(buf, h) : parseBinaryBody(buf, h);
}

function propOffsets(h: PlyHeader): { stride: number; x: number; y: number; z: number; xType: string } {
  let stride = 0;
  let x = -1, y = -1, z = -1, xType = "float";
  for (const p of h.vertexProps) {
    const size = TYPE_SIZE[p.type];
    if (!size) throw new Error(`PLY: unknown property type ${p.type}`);
    if (p.name === "x") { x = stride; xType = p.type; }
    else if (p.name === "y") y = stride;
    else if (p.name === "z") z = stride;
    stride += size;
  }
  if (x < 0 || y < 0 || z < 0) throw new Error("PLY: vertex element lacks x/y/z");
  return { stride, x, y, z, xType };
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

function parseBinaryBody(buf: Uint8Array, h: PlyHeader): PlyMesh {
  const { stride, x, y, z, xType } = propOffsets(h);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const positions = new Float32Array(h.vertexCount * 3);
  let off = h.dataOffset;
  for (let i = 0; i < h.vertexCount; i++) {
    positions[i * 3] = readScalar(dv, off + x, xType);
    positions[i * 3 + 1] = readScalar(dv, off + y, xType);
    positions[i * 3 + 2] = readScalar(dv, off + z, xType);
    off += stride;
  }

  const countSize = TYPE_SIZE[h.faceCountType];
  const idxSize = TYPE_SIZE[h.faceIndexType];
  if (!countSize || !idxSize) throw new Error("PLY: unknown face list types");
  // Two passes over variable-length face records: count triangles, then fill.
  let tris = 0;
  let scan = off;
  for (let f = 0; f < h.faceCount; f++) {
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
  return { positions, indices, vertexCount: h.vertexCount, faceCount: h.faceCount };
}

function parseAsciiBody(buf: Uint8Array, h: PlyHeader): PlyMesh {
  const text = new TextDecoder("ascii").decode(buf.subarray(h.dataOffset));
  const tok = text.split(/\s+/).filter((s) => s.length > 0);
  let p = 0;
  const next = () => {
    const v = Number(tok[p++]);
    if (Number.isNaN(v)) throw new Error(`PLY: non-numeric token at ${p - 1}`);
    return v;
  };
  const xi = h.vertexProps.findIndex((q) => q.name === "x");
  const yi = h.vertexProps.findIndex((q) => q.name === "y");
  const zi = h.vertexProps.findIndex((q) => q.name === "z");
  if (xi < 0 || yi < 0 || zi < 0) throw new Error("PLY: vertex element lacks x/y/z");
  const nProps = h.vertexProps.length;
  const positions = new Float32Array(h.vertexCount * 3);
  for (let i = 0; i < h.vertexCount; i++) {
    for (let j = 0; j < nProps; j++) {
      const v = next();
      if (j === xi) positions[i * 3] = v;
      else if (j === yi) positions[i * 3 + 1] = v;
      else if (j === zi) positions[i * 3 + 2] = v;
    }
  }
  const triIdx: number[] = [];
  for (let f = 0; f < h.faceCount; f++) {
    const n = next();
    if (n < 3) throw new Error(`PLY: face ${f} has ${n} vertices`);
    const i0 = next();
    let prev = next();
    for (let k = 2; k < n; k++) {
      const cur = next();
      triIdx.push(i0, prev, cur);
      prev = cur;
    }
  }
  return {
    positions,
    indices: Uint32Array.from(triIdx),
    vertexCount: h.vertexCount,
    faceCount: h.faceCount,
  };
}

function indexOf(buf: Uint8Array, needle: string): number {
  const n = new TextEncoder().encode(needle);
  outer: for (let i = 0; i <= buf.length - n.length; i++) {
    for (let j = 0; j < n.length; j++) if (buf[i + j] !== n[j]) continue outer;
    return i;
  }
  return -1;
}
