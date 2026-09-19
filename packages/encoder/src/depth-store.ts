/**
 * Frame stores: a fixed-size stack of equally sized frames, addressed by frame index, that lives
 * either in one typed array or in a scratch file.
 *
 * The depth pipeline makes several whole-clip passes (align forward, smooth backward, smooth
 * forward, mesh twice). A clip is `frames x W x H` samples per pass, which for a few thousand frames
 * is gigabytes: past what one typed array can hold and far past what is sensible to keep resident.
 * Every pass is written against this interface, so the same code runs on a 12-frame test stack in
 * memory and on a 5,627-frame film on disk, and produces the same numbers in both.
 */
import { openSync, closeSync, readSync, writeSync, rmSync, ftruncateSync } from "node:fs";

export type FrameArray = Float32Array | Uint8Array;

export interface FrameStore<T extends FrameArray> {
  readonly frames: number;
  /** Samples per frame. */
  readonly frameLength: number;
  read(t: number, out: T): void;
  write(t: number, src: T): void;
  /** Release the store. A file-backed store deletes its scratch file. */
  close(): void;
}

function check(t: number, frames: number, have: number, need: number): void {
  if (!(t >= 0 && t < frames)) throw new Error(`frame store: frame ${t} is outside 0..${frames - 1}`);
  if (have < need) throw new Error(`frame store: buffer holds ${have} samples, ${need} needed`);
}

/** One typed array. `backing`, when given, is used as is (length >= frames * frameLength). */
export function memoryStore<T extends FrameArray>(make: (n: number) => T, frames: number, frameLength: number, backing?: T): FrameStore<T> & { data: T } {
  const data = backing ?? make(frames * frameLength);
  if (data.length < frames * frameLength) throw new Error(`memoryStore: backing holds ${data.length} samples, ${frames * frameLength} needed`);
  return {
    frames, frameLength, data,
    read(t, out) { check(t, frames, out.length, frameLength); (out as Uint8Array).set((data as Uint8Array).subarray(t * frameLength, (t + 1) * frameLength)); },
    write(t, src) { check(t, frames, src.length, frameLength); (data as Uint8Array).set((src as Uint8Array).subarray(0, frameLength), t * frameLength); },
    close() { /* garbage collected */ },
  };
}

/**
 * A scratch file of `frames x frameLength` samples, native byte order (it never leaves this
 * process). Reads and writes are synchronous and positional: the passes that use a store are tight
 * numeric loops, and an await per frame would cost more than the I/O it wraps.
 */
export function fileStore<T extends FrameArray>(path: string, bytesPerSample: 1 | 4, frames: number, frameLength: number): FrameStore<T> {
  const fd = openSync(path, "w+");
  const frameBytes = frameLength * bytesPerSample;
  ftruncateSync(fd, frames * frameBytes);
  let open = true;
  const view = (a: T) => new Uint8Array(a.buffer, a.byteOffset, frameBytes);
  return {
    frames, frameLength,
    read(t, out) {
      check(t, frames, out.length, frameLength);
      const b = view(out);
      for (let got = 0; got < frameBytes; ) {
        const n = readSync(fd, b, got, frameBytes - got, t * frameBytes + got);
        if (n <= 0) throw new Error(`frame store ${path}: short read at frame ${t}`);
        got += n;
      }
    },
    write(t, src) {
      check(t, frames, src.length, frameLength);
      const b = view(src);
      for (let put = 0; put < frameBytes; ) put += writeSync(fd, b, put, frameBytes - put, t * frameBytes + put);
    },
    close() {
      if (!open) return;
      open = false;
      try { closeSync(fd); } catch { /* already closed */ }
      try { rmSync(path, { force: true }); } catch { /* the temp dir sweep gets it */ }
    },
  };
}
