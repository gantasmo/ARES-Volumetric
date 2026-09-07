/**
 * Audio track for the container (spec §11.5 `OPUS`, §11.6 audio block): transcode any input
 * ffmpeg can read to Ogg Opus at 48 kHz, split into timed packets, and lay them into the GOP
 * chunks by presentation time so every chunk stays self-contained (a seek lands on a chunk and
 * finds its own audio next to its own video).
 *
 * Audio block payload:
 *   packet_count u16, reserved u16,
 *   per packet: pts_offset_us u32 (from the chunk's pts_start), duration_us u16, size u16
 *   then the packet datas, concatenated.
 */
import { spawn } from "node:child_process";
import { ByteWriter } from "@ares/core";
import { ffmpegPath } from "./texture-video.js";
import { oggOpusToPackets, type TimedOpusPacket, type OpusHead } from "./ogg.js";

export interface AudioTrackData {
  fourcc: "OPUS";
  head: OpusHead;
  /** the OpusHead bytes — the track's codec_config (WebCodecs `description`) */
  codecConfig: Uint8Array;
  packets: TimedOpusPacket[];
  durationUs: number;
  sampleRate: 48000;
  channels: number;
}

export interface TranscodeOptions {
  /** kbit/s, default 96 */
  bitrateKbps?: number;
  /** shift the audio by this many seconds (positive = audio starts later) */
  offsetSec?: number;
  /** trim the source to this window (seconds) before encoding */
  startSec?: number;
  durationSec?: number;
  /** 1 or 2; default: keep the source's (capped at 2) */
  channels?: number;
}

/** Run ffmpeg → Ogg Opus in memory, then time the packets. Throws with ffmpeg's stderr on failure. */
export async function transcodeToOpus(input: string, opts: TranscodeOptions = {}): Promise<AudioTrackData> {
  const args = ["-v", "error", "-nostdin"];
  if (opts.startSec) args.push("-ss", String(opts.startSec));
  args.push("-i", input, "-vn", "-sn", "-dn");
  if (opts.durationSec) args.push("-t", String(opts.durationSec));
  if (opts.channels) args.push("-ac", String(Math.max(1, Math.min(2, opts.channels))));
  else args.push("-af", "aformat=channel_layouts=stereo|mono");
  args.push("-ar", "48000", "-c:a", "libopus", "-b:a", `${opts.bitrateKbps ?? 96}k`, "-vbr", "on", "-frame_duration", "20", "-application", "audio", "-f", "ogg", "pipe:1");
  const ogg: Buffer = await new Promise((resolve, reject) => {
    const p = spawn(ffmpegPath(), args, { stdio: ["ignore", "pipe", "pipe"] });
    const chunks: Buffer[] = [], err: Buffer[] = [];
    p.stdout.on("data", (c: Buffer) => chunks.push(c));
    p.stderr.on("data", (c: Buffer) => err.push(c));
    p.on("error", reject);
    p.on("close", (code) => (code === 0 ? resolve(Buffer.concat(chunks)) : reject(new Error(`ffmpeg audio transcode exited ${code}: ${Buffer.concat(err).toString("utf8").trim()}`))));
  });
  const { head, packets, durationUs } = oggOpusToPackets(new Uint8Array(ogg.buffer, ogg.byteOffset, ogg.byteLength));
  const shift = Math.round((opts.offsetSec ?? 0) * 1e6);
  const shifted = shift ? packets.map((p) => ({ ...p, ptsUs: p.ptsUs + shift })).filter((p) => p.ptsUs + p.durationUs > 0) : packets;
  return { fourcc: "OPUS", head, codecConfig: head.raw, packets: shifted, durationUs: durationUs + shift, sampleRate: 48000, channels: head.channels };
}

/** Packets whose pts lies in [startUs, endUs). */
export function packetsInWindow(packets: TimedOpusPacket[], startUs: number, endUs: number): TimedOpusPacket[] {
  return packets.filter((p) => p.ptsUs >= startUs && p.ptsUs < endUs);
}

/** Serialize one chunk's audio block. `chunkStartUs` is the chunk's pts_start. */
export function buildAudioBlock(packets: TimedOpusPacket[], chunkStartUs: number): Uint8Array {
  const total = packets.reduce((s, p) => s + p.data.byteLength, 0);
  const w = new ByteWriter(4 + packets.length * 8 + total);
  w.u16(packets.length).u16(0);
  for (const p of packets) {
    const off = Math.max(0, Math.round(p.ptsUs - chunkStartUs));
    if (p.data.byteLength > 0xffff) throw new Error(`audio: packet of ${p.data.byteLength} bytes exceeds the 65535-byte block limit`);
    w.u32(off).u16(Math.min(0xffff, p.durationUs)).u16(p.data.byteLength);
  }
  for (const p of packets) w.bytes(p.data);
  return w.finish();
}
