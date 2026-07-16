/**
 * Little-endian binary reader/writer for the .ares container (spec §11).
 * All multi-byte integers are little-endian; strings are UTF-8 with a u16 length prefix.
 * The reader treats all input as untrusted and bounds-checks every access (spec §11.8, N6).
 */

export class ByteReader {
  readonly view: DataView;
  private p: number;
  constructor(public readonly buf: Uint8Array, offset = 0) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    this.p = offset;
  }
  get pos(): number { return this.p; }
  set pos(v: number) { this.bounds(v, 0); this.p = v; }
  get remaining(): number { return this.buf.byteLength - this.p; }

  private bounds(at: number, size: number): void {
    if (at < 0 || at + size > this.buf.byteLength)
      throw new AresParseError(`read out of bounds at ${at}+${size} of ${this.buf.byteLength}`);
  }
  u8(): number { this.bounds(this.p, 1); return this.view.getUint8(this.p++); }
  u16(): number { this.bounds(this.p, 2); const v = this.view.getUint16(this.p, true); this.p += 2; return v; }
  u32(): number { this.bounds(this.p, 4); const v = this.view.getUint32(this.p, true); this.p += 4; return v; }
  u64(): bigint { this.bounds(this.p, 8); const v = this.view.getBigUint64(this.p, true); this.p += 8; return v; }
  i8(): number { this.bounds(this.p, 1); return this.view.getInt8(this.p++); }
  f32(): number { this.bounds(this.p, 4); const v = this.view.getFloat32(this.p, true); this.p += 4; return v; }
  f32x3(): [number, number, number] { return [this.f32(), this.f32(), this.f32()]; }
  bytes(len: number): Uint8Array { this.bounds(this.p, len); const v = this.buf.subarray(this.p, this.p + len); this.p += len; return v; }
  str(): string { return new TextDecoder("utf-8").decode(this.bytes(this.u16())); }
  fourcc(): string { return String.fromCharCode(this.u8(), this.u8(), this.u8(), this.u8()); }
  skip(n: number): void { this.pos = this.p + n; }
}

/** Growable little-endian writer. */
export class ByteWriter {
  private buf: Uint8Array;
  private view: DataView;
  private p = 0;
  constructor(initial = 1024) {
    this.buf = new Uint8Array(initial);
    this.view = new DataView(this.buf.buffer);
  }
  get pos(): number { return this.p; }
  private ensure(extra: number): void {
    if (this.p + extra <= this.buf.byteLength) return;
    let cap = this.buf.byteLength * 2;
    while (cap < this.p + extra) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf);
    this.buf = next;
    this.view = new DataView(this.buf.buffer);
  }
  u8(v: number): this { this.ensure(1); this.view.setUint8(this.p++, v); return this; }
  u16(v: number): this { this.ensure(2); this.view.setUint16(this.p, v, true); this.p += 2; return this; }
  u32(v: number): this { this.ensure(4); this.view.setUint32(this.p, v, true); this.p += 4; return this; }
  u64(v: bigint): this { this.ensure(8); this.view.setBigUint64(this.p, v, true); this.p += 8; return this; }
  i8(v: number): this { this.ensure(1); this.view.setInt8(this.p++, v); return this; }
  f32(v: number): this { this.ensure(4); this.view.setFloat32(this.p, v, true); this.p += 4; return this; }
  f32x3(v: ArrayLike<number>): this { return this.f32(v[0]!).f32(v[1]!).f32(v[2]!); }
  bytes(b: Uint8Array): this { this.ensure(b.byteLength); this.buf.set(b, this.p); this.p += b.byteLength; return this; }
  str(s: string): this { const b = new TextEncoder().encode(s); return this.u16(b.byteLength).bytes(b); }
  fourcc(s: string): this { for (let i = 0; i < 4; i++) this.u8(s.charCodeAt(i) & 0xff); return this; }
  /** Patch a u32 at an earlier absolute offset (for back-filling sizes/offsets). */
  patchU32(at: number, v: number): void { this.view.setUint32(at, v, true); }
  patchU64(at: number, v: bigint): void { this.view.setBigUint64(at, v, true); }
  align(n: number): this { while (this.p % n !== 0) this.u8(0); return this; }
  finish(): Uint8Array { return this.buf.subarray(0, this.p); }
}

export class AresParseError extends Error {}
