/**
 * Main-thread client for the geometry decode worker (spec §10.7).
 *
 * `WorkerGeometryDecoder` mirrors the sync decode API (decodeGeometryBlock /
 * decodePFrameBlock) as promises backed by decode-worker.ts. Requests carry ids
 * and may overlap (simple in-flight map), so a player can request frame N+1
 * while frame N uploads/renders.
 *
 * Transfer policy (zero-copy where safe):
 * - Block payloads are COPIED once before transfer — geometry blocks are
 *   subarray views into the single fetched .ares file, and transferring their
 *   buffer would detach the whole file.
 * - `prevPosQ` for P-frames is TRANSFERRED when it exclusively owns its buffer
 *   (the normal case: it came from a previous decode) — the caller must treat
 *   it as consumed. Pass `transferPrev = false` to keep it (sends a copy).
 * - All decoded output arrays come back transferred, never copied.
 *
 * The worker URL resolves relative to the compiled module (dist/), so hosts
 * that load @ares/core from dist (importmap or bundler) get the sibling
 * dist/decode-worker.js automatically. If the worker cannot boot (e.g. a
 * browser that doesn't inherit import maps into module workers can't resolve
 * the bare "meshoptimizer" specifier), `ready` rejects and callers should fall
 * back to main-thread decode.
 */
import type { DecodeOk, DecodeRequest, DecodeResponse } from "./decode-worker.js";

export interface WorkerDecodeResult {
  /** quantized positions, stride 4 (u16 x,y,z,pad) — upload straight to writePositions() */
  positionsQ: Uint16Array;
  /** quantized UVs, stride 2 — present iff the block carried them */
  uvsQ?: Uint16Array;
  /** packed normals, 4 B/vertex (i8×4 legacy or oct16 2×i16) — present iff carried */
  normalsQ?: Int8Array;
  /** u32 triangle list — I-frame results only */
  indices?: Uint32Array;
  indexCount?: number;
  vertexCount?: number;
  /** worker-side decode time (ms) for this block */
  decodeMs: number;
}

interface Pending {
  resolve: (r: DecodeOk) => void;
  reject: (e: Error) => void;
}

export class WorkerGeometryDecoder {
  /** Resolves once the worker booted and the meshopt WASM is ready; rejects if the worker can't start. */
  readonly ready: Promise<void>;
  private readonly worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private dead: Error | null = null;

  constructor() {
    this.worker = new Worker(new URL("./decode-worker.js", import.meta.url), { type: "module", name: "ares-geom-decode" });
    this.worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data as DecodeResponse;
      const p = this.pending.get(msg.id);
      if (!p) return; // stale/unknown id (e.g. resolved after dispose)
      this.pending.delete(msg.id);
      if (msg.ok) p.resolve(msg);
      else p.reject(new Error(msg.error));
    };
    this.worker.onerror = (ev: ErrorEvent) => {
      this.failAll(new Error(ev.message || "geometry decode worker failed to load"));
    };
    this.worker.onmessageerror = () => {
      this.failAll(new Error("geometry decode worker: message deserialization failed"));
    };
    this.ready = this.send({ id: this.nextId++, kind: "init" }, []).then(() => undefined);
    this.ready.catch(() => { /* surfaced to whoever awaits `ready`; avoid an unhandled rejection */ });
  }

  /** Number of in-flight requests (the "queue"). */
  get inFlight(): number { return this.pending.size; }

  private send(req: DecodeRequest, transfer: Transferable[]): Promise<DecodeOk> {
    if (this.dead) return Promise.reject(this.dead);
    return new Promise<DecodeOk>((resolve, reject) => {
      this.pending.set(req.id, { resolve, reject });
      this.worker.postMessage(req, transfer);
    });
  }

  /**
   * Decode an I-frame geometry block. Matches decodeGeometryBlock():
   * positionsQ + indices/indexCount/vertexCount, and uvsQ/normalsQ when present.
   */
  async decodeI(block: Uint8Array): Promise<WorkerDecodeResult> {
    // Copy: geometry blocks are views into the whole file — never transfer the file's buffer.
    const payload = block.slice();
    const res = await this.send({ id: this.nextId++, kind: "i", block: payload }, [payload.buffer]);
    return toResult(res);
  }

  /**
   * Decode a P-frame block on top of `prevPosQ` (matches decodePFrameBlock()).
   * By default `prevPosQ` is transferred to the worker (zero-copy) when it owns
   * its buffer — treat it as consumed. Pass `transferPrev = false` to keep it.
   */
  async decodePB(block: Uint8Array, prevPosQ: Uint16Array, transferPrev = true): Promise<WorkerDecodeResult> {
    const payload = block.slice();
    const transfer: Transferable[] = [payload.buffer];
    const ownsBuffer = prevPosQ.byteOffset === 0 && prevPosQ.byteLength === prevPosQ.buffer.byteLength;
    const prev = transferPrev && ownsBuffer ? prevPosQ : prevPosQ.slice();
    transfer.push(prev.buffer as ArrayBuffer);
    const res = await this.send({ id: this.nextId++, kind: "pb", block: payload, prevPosQ: prev }, transfer);
    return toResult(res);
  }

  private failAll(err: Error): void {
    this.dead = err;
    for (const [, p] of this.pending) p.reject(err);
    this.pending.clear();
  }

  dispose(): void {
    this.failAll(new Error("WorkerGeometryDecoder disposed"));
    this.worker.terminate();
  }
}

function toResult(r: DecodeOk): WorkerDecodeResult {
  if (!r.positionsQ) throw new Error("decode worker returned no positions");
  return {
    positionsQ: r.positionsQ,
    uvsQ: r.uvsQ ?? undefined,
    normalsQ: r.normalsQ ?? undefined,
    indices: r.indices ?? undefined,
    indexCount: r.indices ? r.indexCount : undefined,
    vertexCount: r.vertexCount > 0 ? r.vertexCount : undefined,
    decodeMs: r.decodeMs,
  };
}
