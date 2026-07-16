/**
 * Wavefront OBJ importer (spec §14 P5). Parses v/vt/vn + triangulated faces into an
 * indexed mesh with positions, UVs, and per-face-vertex splitting where v/vt/vn differ.
 *
 * Microsoft volumetric captures export one OBJ + one atlas PNG per frame with shared
 * v/vt/vn indices, so the dedup below collapses to the original vertex set; the general
 * path still handles OBJs that index position/uv/normal independently.
 */
export interface ObjMesh {
  positions: Float32Array; // xyz interleaved
  uvs: Float32Array;       // uv interleaved (V flipped to top-left origin)
  indices: Uint32Array;    // triangle list
  vertexCount: number;
}

export interface ObjParseOptions {
  /** OBJ UVs use bottom-left origin; flip V for top-left texture atlases (default true). */
  flipV?: boolean;
}

export function parseObj(text: string, opts: ObjParseOptions = {}): ObjMesh {
  const flipV = opts.flipV ?? true;
  const vp: number[] = [];   // positions pool (xyz)
  const vt: number[] = [];   // uv pool (uv)
  const outPos: number[] = [];
  const outUv: number[] = [];
  const outIdx: number[] = [];
  const combo = new Map<string, number>();

  const addVertex = (vi: number, ti: number): number => {
    const key = vi + "/" + ti;
    const hit = combo.get(key);
    if (hit !== undefined) return hit;
    const idx = outPos.length / 3;
    outPos.push(vp[vi * 3]!, vp[vi * 3 + 1]!, vp[vi * 3 + 2]!);
    if (ti >= 0 && ti * 2 + 1 < vt.length) {
      const u = vt[ti * 2]!;
      const v = vt[ti * 2 + 1]!;
      outUv.push(u, flipV ? 1 - v : v);
    } else {
      outUv.push(0, 0);
    }
    combo.set(key, idx);
    return idx;
  };

  // Resolve a possibly-negative (relative) OBJ index to 0-based absolute.
  const resolve = (raw: number, poolLen: number): number => (raw < 0 ? poolLen + raw : raw - 1);

  let i = 0;
  const n = text.length;
  while (i < n) {
    // read one line
    let j = text.indexOf("\n", i);
    if (j < 0) j = n;
    const line = text.slice(i, j);
    i = j + 1;
    if (line.length < 2) continue;
    const c0 = line.charCodeAt(0);
    if (c0 === 35) continue; // '#'
    const c1 = line.charCodeAt(1);

    if (c0 === 118 && c1 === 32) { // "v "
      const t = line.split(/\s+/);
      vp.push(+t[1]!, +t[2]!, +t[3]!);
    } else if (c0 === 118 && c1 === 116) { // "vt"
      const t = line.split(/\s+/);
      vt.push(+t[1]!, +t[2]!);
    } else if (c0 === 102 && c1 === 32) { // "f "
      const t = line.split(/\s+/);
      // gather the face's (vi,ti) per corner, triangulate as a fan
      const corners: [number, number][] = [];
      for (let k = 1; k < t.length; k++) {
        const tok = t[k];
        if (!tok) continue;
        const parts = tok.split("/");
        const vi = resolve(parseInt(parts[0]!, 10), vp.length / 3);
        const ti = parts[1] && parts[1].length ? resolve(parseInt(parts[1], 10), vt.length / 2) : -1;
        corners.push([vi, ti]);
      }
      for (let k = 2; k < corners.length; k++) {
        outIdx.push(
          addVertex(corners[0]![0], corners[0]![1]),
          addVertex(corners[k - 1]![0], corners[k - 1]![1]),
          addVertex(corners[k]![0], corners[k]![1]),
        );
      }
    }
    // vn ignored: the encoder recomputes weld-aware smooth normals from positions
    // (geometry-encode.ts computeSmoothNormals), so source normals are never read
  }

  return {
    positions: new Float32Array(outPos),
    uvs: new Float32Array(outUv),
    indices: new Uint32Array(outIdx),
    vertexCount: outPos.length / 3,
  };
}
