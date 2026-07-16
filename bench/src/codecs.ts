/** Intra-codec adapters for the §13.4 ablation: raw / quantized-binary (Opt C/D) /
 * meshopt (§6.7 default) / Draco (optional profile), each ± a Brotli wrap (§8.1 entropy stage).
 */
import { MeshoptEncoder, MeshoptDecoder } from "meshoptimizer";
import draco3d from "draco3d";
import type { DecoderModule, EncoderModule, Mesh as DracoMesh } from "draco3d";

// draco3d's runtime API (see draco_nodejs_example.js) that @types/draco3d 1.4.x misses.
declare module "draco3d" {
  interface MeshBuilder {
    AddFloatAttributeToMesh(
      mesh: DracoMesh, attribute: number, count: number, itemSize: number, array: Float32Array,
    ): number;
  }
}
import { brotliCompressSync, brotliDecompressSync, constants as zc } from "node:zlib";
import {
  BenchMesh, Quantized, quantizePositions, dequantizePositions, positionError, aabb,
} from "./mesh.js";

export interface Encoded {
  parts: Uint8Array[];
  /** carried so decode can reconstruct (in the container this lives in the frame header) */
  meta: {
    count: number; indexCount: number;
    box?: Quantized["box"]; bits?: number;
    srcPerm?: Uint32Array; lens?: Uint32Array;
  };
}

export interface IntraCodec {
  name: string;
  /** error reported is an analytic quantization bound, not measured per-vertex */
  analyticError?: boolean;
  encode(mesh: BenchMesh): Encoded;
  decode(enc: Encoded): BenchMesh;
  /** positions to compare against for error (identity unless the codec reorders) */
  reference?(mesh: BenchMesh, enc: Encoded): Float32Array;
}

export const totalBytes = (e: Encoded) => e.parts.reduce((s, p) => s + p.byteLength, 0);

/* ---------------------------------- raw ---------------------------------- */

const raw: IntraCodec = {
  name: "raw-f32",
  encode(mesh) {
    return {
      parts: [
        new Uint8Array(mesh.positions.buffer, mesh.positions.byteOffset, mesh.positions.byteLength),
        new Uint8Array(mesh.indices.buffer, mesh.indices.byteOffset, mesh.indices.byteLength),
      ],
      meta: { count: mesh.positions.length / 3, indexCount: mesh.indices.length },
    };
  },
  decode(enc) {
    const [p, i] = enc.parts as [Uint8Array, Uint8Array];
    return {
      positions: new Float32Array(p.buffer, p.byteOffset, p.byteLength / 4),
      indices: new Uint32Array(i.buffer, i.byteOffset, i.byteLength / 4),
    };
  },
};

/* ------------------- quantized binary (Options C/D layout) ------------------- */

function qbin(bits: number): IntraCodec {
  return {
    name: `qbin${bits}`,
    encode(mesh) {
      const qz = quantizePositions(mesh.positions, bits);
      return {
        parts: [
          new Uint8Array(qz.q.buffer, qz.q.byteOffset, qz.q.byteLength),
          new Uint8Array(mesh.indices.buffer, mesh.indices.byteOffset, mesh.indices.byteLength),
        ],
        meta: { count: mesh.positions.length / 3, indexCount: mesh.indices.length, box: qz.box, bits },
      };
    },
    decode(enc) {
      const [p, i] = enc.parts as [Uint8Array, Uint8Array];
      const q = new Uint16Array(p.buffer, p.byteOffset, p.byteLength / 2);
      return {
        positions: dequantizePositions({ q, box: enc.meta.box!, bits: enc.meta.bits! }),
        indices: new Uint32Array(i.buffer, i.byteOffset, i.byteLength / 4),
      };
    },
  };
}

/* --------------------------------- meshopt --------------------------------- */

function meshopt(bits: number): IntraCodec {
  return {
    name: `meshopt${bits}`,
    encode(mesh) {
      const count = mesh.positions.length / 3;
      const qz = quantizePositions(mesh.positions, bits);
      // meshopt wants optimized order for best ratio + cache efficiency
      const idx = mesh.indices.slice();
      const [remap] = MeshoptEncoder.reorderMesh(idx, /*triangles*/ true, /*optsize*/ true);
      // interleave to stride 8 (3×u16 + pad) in remapped order
      const vb = new Uint8Array(count * 8);
      const vbView = new Uint16Array(vb.buffer);
      for (let i = 0; i < count; i++) {
        const j = remap![i]!;
        vbView[j * 4] = qz.q[i * 3]!;
        vbView[j * 4 + 1] = qz.q[i * 3 + 1]!;
        vbView[j * 4 + 2] = qz.q[i * 3 + 2]!;
      }
      const encV = MeshoptEncoder.encodeVertexBuffer(vb, count, 8);
      const encI = MeshoptEncoder.encodeIndexBuffer(
        new Uint8Array(idx.buffer, idx.byteOffset, idx.byteLength), idx.length, 4);
      return {
        parts: [encV, encI],
        meta: { count, indexCount: mesh.indices.length, box: qz.box, bits, srcPerm: remap! },
      };
    },
    decode(enc) {
      const { count, indexCount } = enc.meta;
      const vb = new Uint8Array(count * 8);
      MeshoptDecoder.decodeVertexBuffer(vb, count, 8, enc.parts[0]!);
      const ib = new Uint8Array(indexCount * 4);
      MeshoptDecoder.decodeIndexBuffer(ib, indexCount, 4, enc.parts[1]!);
      const v16 = new Uint16Array(vb.buffer);
      const q = new Uint16Array(count * 3);
      for (let i = 0; i < count; i++) {
        q[i * 3] = v16[i * 4]!;
        q[i * 3 + 1] = v16[i * 4 + 1]!;
        q[i * 3 + 2] = v16[i * 4 + 2]!;
      }
      return {
        positions: dequantizePositions({ q, box: enc.meta.box!, bits: enc.meta.bits! }),
        indices: new Uint32Array(ib.buffer),
      };
    },
    // decoded vertex j corresponds to source vertex i where remap[i] === j
    reference(mesh, enc) {
      const remap = enc.meta.srcPerm!;
      const count = enc.meta.count;
      const ref = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        const j = remap[i]!;
        ref[j * 3] = mesh.positions[i * 3]!;
        ref[j * 3 + 1] = mesh.positions[i * 3 + 1]!;
        ref[j * 3 + 2] = mesh.positions[i * 3 + 2]!;
      }
      return ref;
    },
  };
}

/* ---------------------------------- Draco ---------------------------------- */

function makeDraco(enc: EncoderModule, dec: DecoderModule, bits: number): IntraCodec {
  return {
    name: `draco${bits}`,
    analyticError: true, // edgebreaker reorders vertices; error ≈ quantization bound (see README)
    encode(mesh) {
      const count = mesh.positions.length / 3;
      const builder = new enc.MeshBuilder();
      const m = new enc.Mesh();
      builder.AddFloatAttributeToMesh(m, enc.POSITION, count, 3, mesh.positions);
      builder.AddFacesToMesh(m, mesh.indices.length / 3, mesh.indices);
      const encoder = new enc.Encoder();
      encoder.SetAttributeQuantization(enc.POSITION, bits);
      encoder.SetSpeedOptions(3, 3); // draco_encoder CLI default (-cl 7)
      encoder.SetEncodingMethod(enc.MESH_EDGEBREAKER_ENCODING);
      const out = new enc.DracoInt8Array();
      const len = encoder.EncodeMeshToDracoBuffer(m, out);
      if (len <= 0) throw new Error("draco encode failed");
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = out.GetValue(i);
      enc.destroy(out); enc.destroy(encoder); enc.destroy(m); enc.destroy(builder);
      return { parts: [bytes], meta: { count, indexCount: mesh.indices.length } };
    },
    decode(encd) {
      const bytes = encd.parts[0]!;
      const buffer = new dec.DecoderBuffer();
      buffer.Init(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), bytes.byteLength);
      const decoder = new dec.Decoder();
      const m = new dec.Mesh();
      const status = decoder.DecodeBufferToMesh(buffer, m);
      if (!status.ok()) throw new Error(`draco decode: ${status.error_msg()}`);
      const nPts = m.num_points(), nFaces = m.num_faces();
      const attr = decoder.GetAttribute(m, decoder.GetAttributeId(m, dec.POSITION));
      const posBytes = nPts * 3 * 4;
      const posPtr = dec._malloc(posBytes);
      decoder.GetAttributeDataArrayForAllPoints(m, attr, dec.DT_FLOAT32, posBytes, posPtr);
      const positions = new Float32Array(dec.HEAPF32.buffer, posPtr, nPts * 3).slice();
      dec._free(posPtr);
      const idxBytes = nFaces * 3 * 4;
      const idxPtr = dec._malloc(idxBytes);
      decoder.GetTrianglesUInt32Array(m, idxBytes, idxPtr);
      const indices = new Uint32Array(dec.HEAPU32.buffer, idxPtr, nFaces * 3).slice();
      dec._free(idxPtr);
      dec.destroy(m); dec.destroy(decoder); dec.destroy(buffer);
      return { positions, indices };
    },
  };
}

/* ------------------------------- brotli wrap ------------------------------- */

/** General-purpose lossless wrap (§6.7 "LZ4/Zstd/Brotli MAY wrap"); q5 ≈ CDN-realistic. */
function brotli(inner: IntraCodec, quality = 5): IntraCodec {
  return {
    name: `${inner.name}+br`,
    analyticError: inner.analyticError,
    encode(mesh) {
      const e = inner.encode(mesh);
      const joined = concat(e.parts);
      const lens = new Uint32Array(e.parts.map((p) => p.byteLength));
      const packed = brotliCompressSync(joined, {
        params: { [zc.BROTLI_PARAM_QUALITY]: quality, [zc.BROTLI_PARAM_SIZE_HINT]: joined.byteLength },
      });
      return { parts: [new Uint8Array(packed)], meta: { ...e.meta, lens } };
    },
    decode(enc) {
      const joined = new Uint8Array(brotliDecompressSync(enc.parts[0]!));
      const lens = enc.meta.lens!;
      const parts: Uint8Array[] = [];
      let off = 0;
      for (const len of lens) { parts.push(joined.slice(off, off + len)); off += len; }
      return inner.decode({ parts, meta: enc.meta });
    },
    reference: inner.reference?.bind(inner),
  };
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.byteLength, 0));
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.byteLength; }
  return out;
}

/* --------------------------------- factory --------------------------------- */

export async function makeCodecs(bits: number, includeDraco: boolean): Promise<IntraCodec[]> {
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;
  const list: IntraCodec[] = [
    raw,
    brotli(raw),
    qbin(bits),
    brotli(qbin(bits)),
    meshopt(bits),
    brotli(meshopt(bits)),
  ];
  if (includeDraco) {
    const encM = await draco3d.createEncoderModule({});
    const decM = await draco3d.createDecoderModule({});
    list.push(makeDraco(encM, decM, bits));
  }
  return list;
}

/** meshopt + qbin at a given bit depth, for the quantization sweep (§13.4). */
export async function sweepCodecs(bits: number): Promise<IntraCodec[]> {
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;
  return [qbin(bits), meshopt(bits), brotli(meshopt(bits))];
}

export { positionError, aabb };
