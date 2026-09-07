/**
 * Minimal Ogg demuxer + Opus packet timing (spec §11.6 audio block; RFC 3533, RFC 7845).
 *
 * ffmpeg hands the encoder an Ogg Opus stream; the container never carries Ogg — it carries the
 * raw Opus packets per chunk with explicit presentation times, so the runtime feeds WebCodecs
 * `AudioDecoder` directly. This file turns pages into packets and packets into durations.
 */

export interface OggPacket { data: Uint8Array; granule: bigint; serial: number; }

/** Split an Ogg stream into logical packets (segments joined across 255-byte lacing, continued pages included). */
export function parseOggPackets(buf: Uint8Array): OggPacket[] {
  const out: OggPacket[] = [];
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let off = 0;
  const partial = new Map<number, Uint8Array[]>();   // serial → pending segments of an unfinished packet
  while (off + 27 <= buf.length) {
    if (buf[off] !== 0x4f || buf[off + 1] !== 0x67 || buf[off + 2] !== 0x67 || buf[off + 3] !== 0x53) throw new Error(`Ogg: bad capture pattern at ${off}`);
    const version = buf[off + 4]!;
    if (version !== 0) throw new Error(`Ogg: unsupported version ${version}`);
    const granule = dv.getBigInt64(off + 6, true);
    const serial = dv.getUint32(off + 14, true);
    const nsegs = buf[off + 26]!;
    const segTable = buf.subarray(off + 27, off + 27 + nsegs);
    let p = off + 27 + nsegs;
    let pend = partial.get(serial) ?? [];
    for (let i = 0; i < nsegs; i++) {
      const len = segTable[i]!;
      if (p + len > buf.length) throw new Error("Ogg: truncated page");
      pend.push(buf.subarray(p, p + len));
      p += len;
      if (len < 255) {                       // packet ends here
        const total = pend.reduce((s, x) => s + x.length, 0);
        const data = new Uint8Array(total);
        let w = 0;
        for (const x of pend) { data.set(x, w); w += x.length; }
        out.push({ data, granule, serial });
        pend = [];
      }
    }
    partial.set(serial, pend);
    off = p;
  }
  return out;
}

export interface OpusHead { channels: number; preSkip: number; inputSampleRate: number; outputGain: number; mappingFamily: number; raw: Uint8Array; }

/** Parse the OpusHead identification header (RFC 7845 §5.1). */
export function parseOpusHead(p: Uint8Array): OpusHead {
  if (p.length < 19 || String.fromCharCode(...p.subarray(0, 8)) !== "OpusHead") throw new Error("Opus: not an OpusHead packet");
  const dv = new DataView(p.buffer, p.byteOffset, p.byteLength);
  return { channels: p[9]!, preSkip: dv.getUint16(10, true), inputSampleRate: dv.getUint32(12, true), outputGain: dv.getInt16(16, true), mappingFamily: p[18]!, raw: p.slice() };
}

/** Samples (at 48 kHz) in one Opus packet, from its TOC byte and frame-count code (RFC 6716 §3.1). */
export function opusPacketSamples(p: Uint8Array): number {
  if (!p.length) return 0;
  const toc = p[0]!;
  const config = toc >> 3;
  const code = toc & 3;
  let frameSamples: number;
  if (config < 12) frameSamples = [480, 960, 1920, 2880][config & 3]!;          // SILK NB/MB/WB: 10/20/40/60 ms
  else if (config < 16) frameSamples = [480, 960][config & 1]!;                 // hybrid: 10/20 ms
  else frameSamples = [120, 240, 480, 960][config & 3]!;                        // CELT: 2.5/5/10/20 ms
  let frames: number;
  if (code === 0) frames = 1;
  else if (code === 1 || code === 2) frames = 2;
  else frames = p.length > 1 ? (p[1]! & 0x3f) : 1;
  return frameSamples * frames;
}

export interface TimedOpusPacket { data: Uint8Array; ptsUs: number; durationUs: number; }

/**
 * Turn an Ogg Opus stream into timed packets. PTS is derived by accumulating packet durations
 * from the first audio packet, minus pre-skip (the encoder priming the decoder must discard), so
 * media time 0 is the first audible sample. Packets that end before 0 are dropped; the first
 * packet straddling 0 keeps a negative-clamped pts of 0 and the decoder's own pre-skip trims it.
 */
export function oggOpusToPackets(buf: Uint8Array): { head: OpusHead; packets: TimedOpusPacket[]; durationUs: number } {
  const pk = parseOggPackets(buf);
  if (pk.length < 2) throw new Error("Ogg Opus: stream has no header packets");
  const head = parseOpusHead(pk[0]!.data);
  const tags = pk[1]!.data;
  if (String.fromCharCode(...tags.subarray(0, 8)) !== "OpusTags") throw new Error("Ogg Opus: second packet is not OpusTags");
  const out: TimedOpusPacket[] = [];
  let sample = -head.preSkip;                   // 48 kHz sample position of the packet's first sample
  for (let i = 2; i < pk.length; i++) {
    const data = pk[i]!.data;
    const n = opusPacketSamples(data);
    if (!n) continue;
    const start = sample;
    sample += n;
    if (sample <= 0) continue;                  // entirely inside the pre-skip
    out.push({ data, ptsUs: Math.max(0, Math.round((start / 48000) * 1e6)), durationUs: Math.round((n / 48000) * 1e6) });
  }
  return { head, packets: out, durationUs: Math.max(0, Math.round((sample / 48000) * 1e6)) };
}
