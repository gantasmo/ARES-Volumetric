/**
 * Runtime texture-video decode (spec §7.1, §12.2): pull-based WebCodecs `VideoDecoder`.
 * Coded frames are demuxed from the container's per-chunk texture blocks and fed to the
 * hardware decoder; each `VideoFrame` is imported straight into the GPU texture (§10.3).
 *
 * Feeding is incremental with a small look-ahead so only a few frames are ever buffered
 * (a whole 1024² GOP of VideoFrames would be hundreds of MB). Seeking / looping back
 * restarts feeding from the covering GOP's keyframe.
 */

export interface CodedTextureFrame { data: Uint8Array; isKey: boolean; frameIndex: number; }

export interface TextureVideoTrack {
  fourcc: string; // "VP09" | "AV01"
  width: number;
  height: number;
}

/** Candidate WebCodecs codec strings, escalating level until one is supported. */
function codecCandidates(fourcc: string): string[] {
  if (fourcc === "AV01") {
    return ["av01.0.04M.08", "av01.0.05M.08", "av01.0.08M.08", "av01.0.12M.08", "av01.0.13M.08"];
  }
  // VP9 profile 0, 8-bit, escalating level
  return ["vp09.00.10.08", "vp09.00.20.08", "vp09.00.21.08", "vp09.00.30.08",
    "vp09.00.31.08", "vp09.00.40.08", "vp09.00.41.08", "vp09.00.50.08", "vp09.00.51.08"];
}

export class TextureVideo {
  private decoder: VideoDecoder | null = null;
  private config: VideoDecoderConfig | null = null;
  private ready = new Map<number, VideoFrame>();
  private fedNext = 0;
  private lastIdx = -1;
  private lastError: string | null = null;

  constructor(
    private readonly track: TextureVideoTrack,
    private readonly frameSource: (globalFrameIndex: number) => CodedTextureFrame | null,
    private readonly gopLength: number,
    private readonly frameCount: number,
  ) {}

  get error(): string | null { return this.lastError; }

  static async isSupported(track: TextureVideoTrack): Promise<string | null> {
    if (typeof VideoDecoder === "undefined") return null;
    for (const codec of codecCandidates(track.fourcc)) {
      try {
        const r = await VideoDecoder.isConfigSupported({ codec, codedWidth: track.width, codedHeight: track.height });
        if (r.supported) return codec;
      } catch { /* try next */ }
    }
    return null;
  }

  async configure(): Promise<boolean> {
    const codec = await TextureVideo.isSupported(this.track);
    if (!codec) { this.lastError = "no supported WebCodecs config for texture track"; return false; }
    this.decoder = new VideoDecoder({
      output: (frame) => {
        const key = Number(frame.timestamp);
        const prev = this.ready.get(key);
        if (prev) prev.close();
        this.ready.set(key, frame);
        this.lastError = null;   // a successful emission clears a stale transient error (a fatal
                                 // decoder moves to state 'closed', which callers check separately)
      },
      error: (e) => { this.lastError = e.message; },
    });
    this.config = { codec, codedWidth: this.track.width, codedHeight: this.track.height, optimizeForLatency: true };
    this.decoder.configure(this.config);
    return true;
  }

  private gopStart(idx: number): number { return Math.floor(idx / this.gopLength) * this.gopLength; }

  private feedTo(idx: number, lookahead = 4): void {
    if (!this.decoder) return;
    // Re-seek only on a real discontinuity: playhead moved BACKWARD (loop/seek-back), or
    // jumped far FORWARD past what we've fed. Normal forward playback just feeds ahead —
    // sequential feeding already includes each GOP's keyframe, so continuity is preserved.
    if (idx < this.lastIdx || idx > this.fedNext + this.gopLength) {
      // Abandon queued decodes from the old position: a scrub drag re-seeks on every event,
      // and letting hundreds of stale chunks drain would delay the sought frame by seconds.
      // reset() drops the queue; feeding restarts at the covering GOP's keyframe.
      if (this.decoder.state === "configured" && this.decoder.decodeQueueSize > 0 && this.config) {
        try { this.decoder.reset(); this.decoder.configure(this.config); } catch (e) { this.lastError = (e as Error).message; }
      }
      this.closeReady();
      this.fedNext = this.gopStart(idx);
    }
    this.lastIdx = idx;
    const target = Math.min(this.frameCount - 1, idx + lookahead);
    while (this.fedNext <= target) {
      const cf = this.frameSource(this.fedNext);
      if (cf) {
        try {
          this.decoder.decode(new EncodedVideoChunk({ type: cf.isKey ? "key" : "delta", timestamp: cf.frameIndex, data: cf.data as BufferSource }));
        } catch (e) { this.lastError = (e as Error).message; }
      }
      this.fedNext++;
    }
  }

  /**
   * The freshest decoded frame at or before `idx` (managed internally — do not close it).
   * Decode emission lags feeding by ~1 frame, so falling back to the most recent past
   * frame keeps the texture ~1 frame behind geometry rather than dropping to blank.
   * Safe to call repeatedly with the same idx (the player polls while paused/scrubbing);
   * `lookahead` can grow across retries to push emission out of decoders that hold frames.
   */
  present(idx: number, lookahead = 4): VideoFrame | null {
    this.feedTo(idx, lookahead);
    let best: VideoFrame | null = null, bestK = -1;
    for (const [k, f] of this.ready) if (k <= idx && k > bestK) { bestK = k; best = f; }
    // Evict frames strictly older than the one in use, and stale in-flight emissions from
    // before a re-seek (they land after closeReady with far-future timestamps).
    for (const [k, f] of this.ready) if (k < bestK || k > idx + this.gopLength) { f.close(); this.ready.delete(k); }
    return best;
  }

  private closeReady(): void { for (const [, f] of this.ready) f.close(); this.ready.clear(); }

  dispose(): void {
    this.closeReady();
    try { this.decoder?.close(); } catch { /* already closed */ }
    this.decoder = null;
  }
}
