/**
 * BROWSER DEPTH ENGINE — a 2D clip becomes a float depth map per frame, in the tab, no install.
 *
 * Ported from VJ-9000 (github.com/gantasmo/VJ-9000) src/akvj/depthWorker.ts + src/useDepthCloud.ts,
 * 2026-09-18. This is the fallback engine of the 2D → 2.5D conversion: it needs nothing but the
 * page, so it runs on a machine that has no local CUDA depth model, and it is the reference the
 * local engine is compared against.
 *
 * What came from VJ-9000: a hidden <video> is drawn into a small canvas, the RGBA pixels go to a
 * Web Worker, and the worker answers with a depth map. What changed, and why:
 *
 * - SEEK, DO NOT SAMPLE. VJ-9000 ran a setInterval at ~8 fps against a playing element and took
 *   whatever frame happened to be on screen: at 60 fps playback and 120 ms inference it re-read the
 *   same frame twice as often as it skipped three, which is fine for a live visual and useless as a
 *   frame sequence. Here every frame is addressed: seek to (i + 0.5) / F, await `seeked`, draw.
 *   Nothing is dropped, nothing is sampled twice, and the run goes as fast as the model does rather
 *   than as fast as the clip plays.
 * - FLOAT, NOT 8-BIT. The worker returns the model's own `predicted_depth` float32 tensor. The
 *   256-level RawImage VJ-9000 consumed is gone (see apps/demo/depth-worker.js).
 * - PIPELINED. Up to MAX_INFLIGHT frames are outstanding, so the next seek+draw overlaps the
 *   current inference, while results are consumed strictly in index order.
 * - No renderer, no EMA, no pseudo-metre remap: those were VJ display choices. Depth is written out
 *   as the model produced it and the consumer decides the mapping.
 *
 * The maps stream to the dev server as they arrive (batched, in order), so a long run never holds
 * more than a batch in memory and a crash leaves a partial, cancellable job rather than nothing.
 *
 * Pure module: no DOM at module scope, no import of main.js, no window globals. The only ambient
 * dependencies are `document` (the hidden <video>) and `fetch` (the upload endpoints).
 */

/** Mirrors apps/demo/depth-worker.js. Copied rather than imported: importing anything from the
 *  worker module would evaluate the transformers.js bundle on the main thread. */
const MODEL_IDS = {
  small: "onnx-community/depth-anything-v2-small",
  base: "onnx-community/depth-anything-v2-base",
  large: "onnx-community/depth-anything-v2-large",
};

/** DINOv2 patch size: every inference dimension is a multiple of it. */
const PATCH = 14;
/** Frames outstanding at the worker. 2 is enough to hide the seek behind the inference; more only
 *  buys latency, since the worker runs one frame at a time. */
const MAX_INFLIGHT = 2;
/** Maps per POST /depth/upload. 8 at 518x294 is ~4.9 MB, and a retry costs one second of clip. */
const UPLOAD_BATCH = 8;
/** Grace for one requestVideoFrameCallback after `seeked`. The element is off-screen, so a browser
 *  that never composites it would otherwise pay this on every frame — two misses disable it. */
const RVFC_WAIT_MS = 30;
/** requestAdapter() can hang instead of rejecting (measured in headless Chrome), and a probe that
 *  never settles is worse than a negative answer. */
const ADAPTER_TIMEOUT_MS = 3000;

const snap14 = (v) => Math.max(PATCH, Math.round(v / PATCH) * PATCH);

function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  promise.catch(() => {}); // a slot nobody awaits (abort, fatal) must not print an unhandled rejection
  return { promise, resolve, reject };
}

/**
 * What this engine can actually do here, before anything is loaded.
 * @returns {Promise<{webgpu: boolean, isolated: boolean}>} webgpu: an adapter was granted (without
 * one the worker falls back to wasm q8, roughly an order of magnitude slower). isolated: the page
 * is cross-origin isolated, so SharedArrayBuffer exists and the wasm fallback can use threads.
 */
export async function probeBrowserDepth() {
  let webgpu = false;
  try {
    if (navigator.gpu?.requestAdapter) {
      webgpu = !!(await Promise.race([
        navigator.gpu.requestAdapter(),
        new Promise((r) => setTimeout(() => r(null), ADAPTER_TIMEOUT_MS)),
      ]));
    }
  } catch {
    webgpu = false;
  }
  return { webgpu, isolated: !!self.crossOriginIsolated };
}

/** Resolve when the element knows its dimensions and duration. */
function loadMetadata(el) {
  return new Promise((resolve, reject) => {
    if (el.readyState >= 1 /* HAVE_METADATA */) return resolve();
    const cleanup = () => {
      el.removeEventListener("loadedmetadata", ok);
      el.removeEventListener("error", bad);
    };
    const ok = () => {
      cleanup();
      resolve();
    };
    const bad = () => {
      cleanup();
      reject(new Error(`source failed to load: ${el.error?.message || "media error"}`));
    };
    el.addEventListener("loadedmetadata", ok);
    el.addEventListener("error", bad);
  });
}

/**
 * Seek to `t` and resolve once that frame is drawable. `seeked` is the guarantee; one
 * requestVideoFrameCallback afterwards is the belt, bounded by RVFC_WAIT_MS and dropped for the
 * rest of the run after two misses.
 */
function seekTo(el, t, state) {
  return new Promise((resolve, reject) => {
    if (Math.abs(el.currentTime - t) < 1e-6 && el.readyState >= 2) return resolve();
    const cleanup = () => {
      el.removeEventListener("seeked", ok);
      el.removeEventListener("error", bad);
    };
    const bad = () => {
      cleanup();
      reject(new Error(`seek to ${t.toFixed(3)}s failed: ${el.error?.message || "media error"}`));
    };
    const ok = () => {
      cleanup();
      if (!state.useRvfc || typeof el.requestVideoFrameCallback !== "function") return resolve();
      let settled = false;
      const done = (hit) => {
        if (settled) return;
        settled = true;
        if (hit) state.rvfcMisses = 0;
        else if (++state.rvfcMisses >= 2) state.useRvfc = false;
        resolve();
      };
      let id = 0;
      try {
        id = el.requestVideoFrameCallback(() => done(true));
      } catch {
        return resolve();
      }
      setTimeout(() => {
        try {
          el.cancelVideoFrameCallback(id);
        } catch {
          /* already fired */
        }
        done(false);
      }, RVFC_WAIT_MS);
    };
    el.addEventListener("seeked", ok);
    el.addEventListener("error", bad);
    try {
      el.currentTime = t;
    } catch (e) {
      cleanup();
      reject(e);
    }
  });
}

/**
 * Run the whole clip through the browser depth engine and stream the maps to the dev server.
 *
 * @param {Object} opts
 * @param {string} opts.video Absolute source path, recorded in depth.json and in upload/begin.
 * @param {string} opts.sourceUrl URL the hidden <video> loads (range-capable, same-origin).
 * @param {number} opts.sourceFps Probed source frame rate.
 * @param {number|null} [opts.fps] Sampling rate; null means sourceFps.
 * @param {number|null} [opts.maxFrames] Cap on frames.
 * @param {"small"|"base"|"large"} [opts.modelKey]
 * @param {"auto"|"fp16"|"fp32"|"q8"} [opts.precision]
 * @param {number} [opts.inferWidth] Inference width before snapping to a multiple of 14.
 * @param {(p: {stage: "model"|"depth", done: number, total: number, msPerFrame: number}) => void} [opts.onProgress]
 * @param {(line: string) => void} [opts.onLog]
 * @param {AbortSignal} [opts.signal] Abort stops sampling, terminates the worker, cancels the job
 *   and rejects with an AbortError.
 * @returns {Promise<{job: string, dir: string, frames: number, width: number, height: number,
 *   msPerFrame: number, device: string, dtype: string}>} width/height are the DEPTH MAP dims, which
 *   are constant for the run and may differ from the source dims.
 */
export async function runBrowserDepth({
  video,
  sourceUrl,
  sourceFps,
  fps = null,
  maxFrames = null,
  modelKey = "small",
  precision = "auto",
  inferWidth = 518,
  onProgress,
  onLog,
  signal,
} = {}) {
  const log = (line) => {
    try {
      onLog?.(line);
    } catch {
      /* a broken log sink must not kill the run */
    }
  };
  const emit = (p) => {
    try {
      onProgress?.(p);
    } catch {
      /* same */
    }
  };
  const abortError = () => new DOMException("depth run aborted", "AbortError");

  if (!sourceUrl) throw new Error("runBrowserDepth: sourceUrl is required");
  if (signal?.aborted) throw abortError();

  const modelId = MODEL_IDS[modelKey] || MODEL_IDS.small;

  // ---- worker plumbing ------------------------------------------------------
  const pending = new Map(); // frame index -> deferred depth reply
  let fatal = null;
  const readyD = deferred();
  const abortD = deferred();

  const slotFor = (i) => {
    let d = pending.get(i);
    if (!d) {
      d = deferred();
      if (fatal) d.reject(fatal);
      pending.set(i, d);
    }
    return d;
  };
  // Set once the producer loop exists: an abort has to wake it, or it sits on a slot that the
  // consumer will never release.
  let wakeProducer = () => {};
  const failAll = (err) => {
    fatal = fatal || err;
    readyD.reject(fatal);
    for (const d of pending.values()) d.reject(fatal);
    wakeProducer();
  };
  const race = (p) => Promise.race([p, abortD.promise]);

  let aborted = false;
  const onAbort = () => {
    aborted = true;
    const e = abortError();
    abortD.reject(e);
    failAll(e);
  };
  signal?.addEventListener("abort", onAbort, { once: true });

  // ---- hidden source element ------------------------------------------------
  // Off-screen rather than display:none — a video that is not rendered at all never presents a
  // frame, which would cost RVFC_WAIT_MS on every seek.
  const el = document.createElement("video");
  el.preload = "auto";
  el.muted = true;
  el.playsInline = true;
  el.setAttribute("playsinline", "");
  el.setAttribute("muted", "");
  el.style.cssText =
    "position:fixed;left:-10000px;top:0;width:2px;height:2px;opacity:0.01;pointer-events:none";
  document.body.appendChild(el);

  let worker = null;
  let job = null;
  let dir = null;
  let frames = 0;

  try {
    el.src = sourceUrl;
    await race(loadMetadata(el));
    const duration = el.duration;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("source has no finite duration");
    if (!el.videoWidth || !el.videoHeight) throw new Error("source has no video track");

    const F = Number(fps) > 0 ? Number(fps) : Number(sourceFps) > 0 ? Number(sourceFps) : 30;
    const available = Math.max(1, Math.floor(duration * F));
    const N = maxFrames ? Math.min(Math.max(1, Math.floor(maxFrames)), available) : available;
    const W = snap14(Number(inferWidth) > 0 ? Number(inferWidth) : 518);
    const H = snap14((W * el.videoHeight) / el.videoWidth);

    log(`source ${el.videoWidth}x${el.videoHeight} · ${duration.toFixed(2)} s · ${Number(sourceFps) || 0} fps`);
    log(`sample ${W}x${H} · ${F} fps · ${N} frames`);

    const canvas = new OffscreenCanvas(W, H);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2d context unavailable");

    // ---- model ---------------------------------------------------------------
    const dl = new Map(); // per-file bytes, used only until a progress_total arrives
    let dlTotal = null;
    worker = new Worker(new URL("./depth-worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (!m) return;
      if (m.type === "progress") {
        if (m.status === "progress_total") dlTotal = { done: m.loaded || 0, total: m.total || 0 };
        else if (m.file) dl.set(m.file, { loaded: m.loaded || 0, total: m.total || 0 });
        let done = 0;
        let total = 0;
        if (dlTotal) ({ done, total } = dlTotal);
        else
          for (const f of dl.values()) {
            done += f.loaded;
            total += f.total;
          }
        emit({ stage: "model", done, total, msPerFrame: 0 });
        return;
      }
      if (m.type === "ready") return readyD.resolve(m);
      if (m.type === "depth") return slotFor(m.index).resolve(m);
      if (m.type === "error") {
        const err = new Error(m.message || "depth worker error");
        if (typeof m.index === "number") slotFor(m.index).reject(err);
        else failAll(err);
      }
    };
    worker.onerror = (e) => failAll(new Error(`depth worker crashed: ${e?.message || e}`));
    worker.onmessageerror = () => failAll(new Error("depth worker message could not be deserialized"));

    worker.postMessage({ type: "init", modelKey, precision, inferWidth: W });
    const ready = await race(readyD.promise);
    log(`model ${modelId} · ${ready.device} ${ready.dtype}`);

    // ---- sampling: producer ---------------------------------------------------
    let posted = 0;
    let consumed = 0;
    let slotWaiters = [];
    const releaseSlot = () => {
      const waiting = slotWaiters;
      slotWaiters = [];
      for (const r of waiting) r();
    };
    const waitSlot = () =>
      posted - consumed < MAX_INFLIGHT ? Promise.resolve() : new Promise((r) => slotWaiters.push(r));
    wakeProducer = releaseSlot;
    const seekState = { useRvfc: typeof el.requestVideoFrameCallback === "function", rvfcMisses: 0 };

    (async () => {
      for (let i = 0; i < N; i++) {
        if (aborted || fatal) return;
        await waitSlot();
        if (aborted || fatal) return;
        await seekTo(el, Math.min((i + 0.5) / F, Math.max(0, duration - 1e-3)), seekState);
        if (aborted || fatal) return;
        ctx.drawImage(el, 0, 0, W, H);
        const px = ctx.getImageData(0, 0, W, H);
        slotFor(i); // the reply can land before the consumer asks for it
        worker.postMessage(
          { type: "frame", index: i, data: px.data.buffer, width: W, height: H },
          [px.data.buffer],
        );
        posted++;
      }
    })().catch((e) => failAll(e instanceof Error ? e : new Error(String(e))));

    // ---- consumer: depth in, uploads out ---------------------------------------
    let mapW = 0;
    let mapH = 0;
    let inferMs = 0;
    const batch = [];
    let batchStart = 0;
    const t0 = performance.now();

    const flush = async () => {
      if (!batch.length) return;
      const k = batch.length;
      const per = mapW * mapH;
      const body = new Float32Array(k * per);
      for (let j = 0; j < k; j++) body.set(batch[j], j * per);
      const index = batchStart;
      batchStart += k;
      batch.length = 0;
      const r = await fetch(`/depth/upload?job=${encodeURIComponent(job)}&index=${index}&count=${k}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: body.buffer,
        signal,
      });
      if (!r.ok) throw new Error(`upload ${index}+${k} failed: HTTP ${r.status}`);
      log(`upload ${index}+${k} · ${(body.byteLength / 1e6).toFixed(1)} MB`);
    };

    for (let i = 0; i < N; i++) {
      const reply = await race(slotFor(i).promise);
      pending.delete(i);
      consumed++;
      releaseSlot();

      if (i === 0) {
        mapW = reply.width;
        mapH = reply.height;
        // begin() only now: before the first reply the map dims are a guess, and depth.json must
        // carry the real ones.
        const r = await fetch("/depth/upload/begin", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            video,
            width: mapW,
            height: mapH,
            fps: F,
            sampling: { fps: fps ?? null, maxFrames: maxFrames ?? null },
            sourceFps,
            sourceWidth: el.videoWidth,
            sourceHeight: el.videoHeight,
            sourceDurationS: duration,
            engine: "browser",
            model: modelId,
            modelKey,
            device: ready.device,
            dtype: ready.dtype,
          }),
          signal,
        });
        if (!r.ok) throw new Error(`upload/begin failed: HTTP ${r.status}`);
        const j = await r.json();
        if (!j?.job) throw new Error("upload/begin returned no job");
        job = j.job;
        dir = j.dir || null;
        log(`job ${job} · map ${mapW}x${mapH}`);
      } else if (reply.width !== mapW || reply.height !== mapH) {
        throw new Error(
          `depth map dims changed at frame ${i}: ${reply.width}x${reply.height} after ${mapW}x${mapH}`,
        );
      }

      batch.push(new Float32Array(reply.data));
      inferMs += reply.ms || 0;
      frames++;
      if (batch.length >= UPLOAD_BATCH) await flush();
      emit({
        stage: "depth",
        done: frames,
        total: N,
        msPerFrame: (performance.now() - t0) / frames,
      });
    }
    await flush();

    // ---- finish ----------------------------------------------------------------
    const msPerFrame = (performance.now() - t0) / Math.max(1, frames);
    const fin = await fetch("/depth/upload/finish", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job, frames, msPerFrame }),
    });
    if (!fin.ok) throw new Error(`upload/finish failed: HTTP ${fin.status}`);
    const finJson = await fin.json().catch(() => ({}));
    dir = finJson.dir || dir;
    log(
      `done ${frames}/${N} frames · ${msPerFrame.toFixed(0)} ms/f wall · ` +
        `${(inferMs / Math.max(1, frames)).toFixed(0)} ms/f infer · ${ready.device} ${ready.dtype}`,
    );

    return {
      job,
      dir,
      frames,
      width: mapW,
      height: mapH,
      msPerFrame,
      device: ready.device,
      dtype: ready.dtype,
    };
  } catch (err) {
    if (job) {
      // Best effort, and deliberately unsignalled: the whole point of the call is to run after an
      // abort. A job left open would keep its partial directory forever.
      try {
        await fetch("/depth/upload/cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job }),
        });
        log(`cancel ${job} · ${frames} frames written`);
      } catch {
        /* the server will garbage-collect it */
      }
    }
    throw aborted ? abortError() : err;
  } finally {
    signal?.removeEventListener("abort", onAbort);
    try {
      worker?.terminate();
    } catch {
      /* already gone */
    }
    try {
      el.pause();
      el.removeAttribute("src");
      el.load();
      el.remove();
    } catch {
      /* the element is being dropped anyway */
    }
  }
}
