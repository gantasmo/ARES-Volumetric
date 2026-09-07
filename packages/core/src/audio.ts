/**
 * Runtime audio track (spec §11.5 `OPUS`, §11.6 audio block): WebCodecs `AudioDecoder` → Web
 * Audio scheduling. The container carries raw Opus packets with presentation times per chunk;
 * this class decodes a sliding window ahead of the playhead, schedules each decoded buffer on
 * the AudioContext timeline, and exposes the context clock so the player can slave video to
 * audio (the one clock that does not drift).
 *
 * Forward playback only: reverse (ping-pong) mutes, and any non-sequential jump re-anchors.
 */

import type { AudioPacketRef } from "./container.js";
export type { AudioPacketRef };

export interface AudioTrackInfo { fourcc: string; codecConfig: Uint8Array; sampleRate: number; channels: number; }

const LOOKAHEAD_US = 1.5e6;     // decode/schedule this far ahead of the playhead
const MAX_SOURCES = 400;        // scheduled-but-unplayed buffers kept (≈ 8 s of 20 ms packets)

export class AudioTrack {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private pre: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private levelBuf: Uint8Array<ArrayBuffer> | null = null;
  private decoder: AudioDecoder | null = null;
  private config: AudioDecoderConfig | null = null;
  private lastError: string | null = null;
  private volume = 1;
  private muted = false;
  /** playhead media time ↔ context time anchor while running */
  private anchorMediaUs = 0;
  private anchorCtxSec = 0;
  private running = false;
  private fedIdx = 0;               // next packet index to feed
  private packets: AudioPacketRef[];
  private scheduled: { src: AudioBufferSourceNode; endCtx: number }[] = [];
  private pendingDecodes = 0;

  constructor(private readonly track: AudioTrackInfo, packets: AudioPacketRef[]) {
    this.packets = packets.slice().sort((a, b) => a.ptsUs - b.ptsUs);
  }

  get error(): string | null { return this.lastError; }
  get available(): boolean { return typeof AudioDecoder !== "undefined" && typeof AudioContext !== "undefined" && this.packets.length > 0; }
  get durationUs(): number { const l = this.packets[this.packets.length - 1]; return l ? l.ptsUs + l.durationUs : 0; }

  static async isSupported(track: AudioTrackInfo): Promise<boolean> {
    if (typeof AudioDecoder === "undefined") return false;
    try {
      const r = await AudioDecoder.isConfigSupported({ codec: "opus", sampleRate: track.sampleRate, numberOfChannels: track.channels });
      return !!r.supported;
    } catch { return false; }
  }

  /** Create the context + decoder lazily (a user gesture is required to start audio in browsers). */
  private ensure(): boolean {
    if (this.ctx && this.decoder) return true;
    if (!this.available) return false;
    try {
      this.ctx ??= new AudioContext({ sampleRate: this.track.sampleRate });
      if (!this.gain) {
        this.gain = this.ctx.createGain(); this.gain.connect(this.ctx.destination); this.applyGain();
        // Sources → pre → gain → out; the analyser taps `pre` so a muted track still reports a level.
        this.pre = this.ctx.createGain(); this.pre.connect(this.gain);
        this.analyser = this.ctx.createAnalyser(); this.analyser.fftSize = 256; this.analyser.smoothingTimeConstant = 0.6;
        this.pre.connect(this.analyser);
      }
      this.config = { codec: "opus", sampleRate: this.track.sampleRate, numberOfChannels: this.track.channels, description: this.track.codecConfig as BufferSource };
      this.decoder = new AudioDecoder({
        output: (data) => this.onDecoded(data),
        error: (e) => { this.lastError = e.message; },
      });
      this.decoder.configure(this.config);
      return true;
    } catch (e) {
      this.lastError = e instanceof Error ? e.message : String(e);
      return false;
    }
  }

  private applyGain(): void {
    if (this.gain) this.gain.gain.value = this.muted ? 0 : Math.max(0, Math.min(1, this.volume));
  }
  setVolume(v: number): void { this.volume = v; this.applyGain(); }
  setMuted(m: boolean): void { this.muted = m; this.applyGain(); }
  get isMuted(): boolean { return this.muted; }
  get isRunning(): boolean { return this.running; }

  /** Start (or re-anchor) playback at media time `mediaUs`. Safe to call repeatedly. */
  start(mediaUs: number): void {
    if (!this.ensure() || !this.ctx) return;
    if (this.ctx.state === "suspended") void this.ctx.resume();
    this.cancelScheduled();
    this.resetDecoder();
    this.anchorMediaUs = mediaUs;
    this.anchorCtxSec = this.ctx.currentTime + 0.05;   // a little slack so the first buffers land on time
    this.running = true;
    // Feed from the first packet at or before mediaUs.
    let i = 0;
    while (i + 1 < this.packets.length && this.packets[i + 1]!.ptsUs <= mediaUs) i++;
    this.fedIdx = i;
    this.pump(mediaUs);
  }

  /** Stop scheduling and silence what is queued (pause / dispose). */
  stop(): void {
    this.running = false;
    this.cancelScheduled();
  }

  /** Instantaneous loudness 0..1 (RMS of the last analyser window, pre-gain), 0 when idle. */
  level(): number {
    if (!this.analyser || !this.running) return 0;
    if (!this.levelBuf || this.levelBuf.length !== this.analyser.fftSize) this.levelBuf = new Uint8Array(this.analyser.fftSize);
    this.analyser.getByteTimeDomainData(this.levelBuf);
    let sum = 0;
    for (let i = 0; i < this.levelBuf.length; i++) { const v = (this.levelBuf[i]! - 128) / 128; sum += v * v; }
    return Math.min(1, Math.sqrt(sum / this.levelBuf.length) * 2.5);
  }

  /** Media time implied by the audio clock while running, else null. */
  currentMediaUs(): number | null {
    if (!this.running || !this.ctx) return null;
    return this.anchorMediaUs + (this.ctx.currentTime - this.anchorCtxSec) * 1e6;
  }

  /** Keep the decode window ahead of the playhead; called every animation frame while playing. */
  pump(mediaUs: number): void {
    if (!this.running || !this.decoder || this.decoder.state !== "configured") return;
    // Drop finished sources.
    const now = this.ctx!.currentTime;
    this.scheduled = this.scheduled.filter((s) => s.endCtx > now);
    const limit = mediaUs + LOOKAHEAD_US;
    while (this.fedIdx < this.packets.length && this.packets[this.fedIdx]!.ptsUs < limit && this.pendingDecodes < 64 && this.scheduled.length < MAX_SOURCES) {
      const p = this.packets[this.fedIdx++]!;
      try {
        this.pendingDecodes++;
        this.decoder.decode(new EncodedAudioChunk({ type: "key", timestamp: p.ptsUs, duration: p.durationUs, data: p.data as BufferSource }));
      } catch (e) { this.pendingDecodes--; this.lastError = e instanceof Error ? e.message : String(e); }
    }
  }

  private onDecoded(data: AudioData): void {
    this.pendingDecodes = Math.max(0, this.pendingDecodes - 1);
    const ctx = this.ctx;
    if (!ctx || !this.gain || !this.running) { data.close(); return; }
    const frames = data.numberOfFrames, ch = data.numberOfChannels, rate = data.sampleRate;
    const buffer = ctx.createBuffer(ch, frames, rate);
    try {
      for (let c = 0; c < ch; c++) {
        const plane = new Float32Array(frames);
        data.copyTo(plane, { planeIndex: c, format: "f32-planar" });
        buffer.copyToChannel(plane, c);
      }
    } catch (e) {
      // Interleaved-only implementations: fall back to a single interleaved copy.
      try {
        const inter = new Float32Array(frames * ch);
        data.copyTo(inter, { planeIndex: 0, format: "f32" });
        for (let c = 0; c < ch; c++) { const plane = new Float32Array(frames); for (let i = 0; i < frames; i++) plane[i] = inter[i * ch + c]!; buffer.copyToChannel(plane, c); }
      } catch { this.lastError = e instanceof Error ? e.message : String(e); data.close(); return; }
    }
    const mediaUs = Number(data.timestamp);
    data.close();
    const when = this.anchorCtxSec + (mediaUs - this.anchorMediaUs) / 1e6;
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.pre ?? this.gain);
    const late = ctx.currentTime - when;
    if (late > 0) {
      if (late >= buffer.duration) return;          // already entirely in the past
      src.start(ctx.currentTime, late);              // trim the elapsed head so it stays in sync
    } else src.start(when);
    this.scheduled.push({ src, endCtx: when + buffer.duration });
  }

  private cancelScheduled(): void {
    for (const s of this.scheduled) { try { s.src.stop(); } catch { /* already ended */ } try { s.src.disconnect(); } catch { /* ignore */ } }
    this.scheduled = [];
  }

  private resetDecoder(): void {
    if (!this.decoder || !this.config) return;
    try {
      if (this.decoder.state === "configured") this.decoder.reset();
      if (this.decoder.state !== "closed") this.decoder.configure(this.config);
    } catch (e) { this.lastError = e instanceof Error ? e.message : String(e); }
    this.pendingDecodes = 0;
  }

  dispose(): void {
    this.stop();
    try { this.decoder?.close(); } catch { /* ignore */ }
    this.decoder = null;
    void this.ctx?.close().catch(() => { /* ignore */ });
    this.ctx = null;
    this.gain = null;
  }
}
