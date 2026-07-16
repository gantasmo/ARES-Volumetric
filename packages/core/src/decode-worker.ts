/**
 * Geometry decode worker (spec §10.7) — meshopt block decode off the main thread.
 *
 * Runs the same pure decoders as the sync path (geometry.ts) inside a module
 * worker. The main-thread client (worker-decode.ts → WorkerGeometryDecoder)
 * posts one request per block; results come back as transferables, so the only
 * per-frame main-thread cost is a small compressed-block copy going out and a
 * zero-copy adoption of the decoded arrays coming back.
 *
 * Protocol (request → response, matched by `id`; requests may overlap):
 *   {id, kind:"init"}                → ack after the meshopt WASM is instantiated
 *   {id, kind:"i",  block}           → I-frame: positionsQ/uvsQ/normalsQ/indices/indexCount/vertexCount
 *   {id, kind:"pb", block, prevPosQ} → P-frame: positionsQ (prev + delta), optional uvsQ/normalsQ
 * Failures answer {id, ok:false, error} and never take the worker down.
 */
import { decodeGeometryBlock, decodePFrameBlock, meshoptReady } from "./geometry.js";

export type DecodeRequest =
  | { id: number; kind: "init" }
  | { id: number; kind: "i"; block: Uint8Array }
  | { id: number; kind: "pb"; block: Uint8Array; prevPosQ: Uint16Array };

export interface DecodeOk {
  id: number;
  ok: true;
  decodeMs: number;                 // worker-side decode time for this block
  positionsQ: Uint16Array | null;   // null only on the init ack
  uvsQ: Uint16Array | null;
  normalsQ: Int8Array | null;
  indices: Uint32Array | null;      // I-frames only (topology persists per GOP)
  indexCount: number;
  vertexCount: number;
}
export interface DecodeErr { id: number; ok: false; error: string }
export type DecodeResponse = DecodeOk | DecodeErr;

/** The package compiles against the DOM lib; type the worker-global surface we use explicitly. */
interface WorkerScope {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
}
const scope = self as unknown as WorkerScope;

const ready = meshoptReady();

function reply(res: DecodeResponse, transfer: Transferable[]): void {
  scope.postMessage(res, transfer);
}

async function handle(req: DecodeRequest): Promise<void> {
  try {
    await ready;
    if (req.kind === "init") {
      reply({ id: req.id, ok: true, decodeMs: 0, positionsQ: null, uvsQ: null, normalsQ: null, indices: null, indexCount: 0, vertexCount: 0 }, []);
      return;
    }
    const t0 = performance.now();
    if (req.kind === "i") {
      const g = decodeGeometryBlock(req.block);
      const transfer: Transferable[] = [g.positionsQ.buffer as ArrayBuffer, g.indices.buffer as ArrayBuffer];
      if (g.uvsQ) transfer.push(g.uvsQ.buffer as ArrayBuffer);
      if (g.normalsQ) transfer.push(g.normalsQ.buffer as ArrayBuffer);
      reply({
        id: req.id, ok: true, decodeMs: performance.now() - t0,
        positionsQ: g.positionsQ, uvsQ: g.uvsQ ?? null, normalsQ: g.normalsQ ?? null,
        indices: g.indices, indexCount: g.indexCount, vertexCount: g.vertexCount,
      }, transfer);
      return;
    }
    const p = decodePFrameBlock(req.block, req.prevPosQ);
    const transfer: Transferable[] = [p.positionsQ.buffer as ArrayBuffer];
    if (p.uvsQ) transfer.push(p.uvsQ.buffer as ArrayBuffer);
    if (p.normalsQ) transfer.push(p.normalsQ.buffer as ArrayBuffer);
    reply({
      id: req.id, ok: true, decodeMs: performance.now() - t0,
      positionsQ: p.positionsQ, uvsQ: p.uvsQ ?? null, normalsQ: p.normalsQ ?? null,
      indices: null, indexCount: 0, vertexCount: p.positionsQ.length / 4,
    }, transfer);
  } catch (e) {
    reply({ id: req.id, ok: false, error: e instanceof Error ? e.message : String(e) }, []);
  }
}

scope.onmessage = (ev: MessageEvent) => { void handle(ev.data as DecodeRequest); };
