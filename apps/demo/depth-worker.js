/**
 * MONOCULAR DEPTH WORKER — Depth-Anything-V2 (ONNX) off the main thread.
 *
 * Ported from VJ-9000 (github.com/gantasmo/VJ-9000) src/akvj/depthWorker.ts + src/useDepthCloud.ts,
 * 2026-09-18. What carried over: the transformers.js model id, the WebGPU-then-wasm backend ladder
 * with its dtype rules, and the rule that inference NEVER runs on the main thread. What changed for
 * ARES:
 *
 * - FLOAT depth, not an 8-bit image. VJ-9000 used the `depth-estimation` pipeline and read
 *   `out.depth.data`, a Uint8 RawImage the pipeline produces by min/max-normalising the prediction
 *   to 0..255. That quantisation is invisible on a VJ point cloud and ruinous for a mesh: 256 depth
 *   planes is banding you can count. Here AutoModelForDepthEstimation + AutoProcessor are driven
 *   directly, and the model's own `predicted_depth` tensor is returned as float32 (converted when
 *   the session hands back float16). Values are RELATIVE inverse depth, not metres — the caller
 *   decides the near/far mapping.
 * - CONSTANT map dimensions. Every frame of a run must produce the same W'xH' or the .depth stream
 *   is not a stream. The processor's resize is switched OFF whenever the incoming frame is already a
 *   multiple of 14 (DINOv2's patch size), which the driver guarantees, so the map dims are exactly
 *   the frame dims, every frame, and no bicubic pass runs per frame. Non-conforming frames fall back
 *   to the processor's own keep_aspect_ratio resize, targeted at `inferWidth` rather than 518.
 * - Per-frame ONNX timing is reported so the caller can show a real ms/frame instead of guessing.
 *
 * The frames arrive as transferred RGBA ArrayBuffers and the maps go back as transferred float32
 * ArrayBuffers: no structured-clone copy in either direction.
 *
 * ES module worker, no bundler: apps/demo is loaded as plain ES modules, so transformers.js comes
 * from jsdelivr at a pinned version. The page is cross-origin isolated (tools/serve.mjs sends
 * COOP/COEP), and both jsdelivr and huggingface.co answer with `access-control-allow-origin: *`
 * plus `cross-origin-resource-policy: cross-origin`, so the module import, the ORT wasm binaries
 * and the model files all pass the COEP check. Isolation also buys SharedArrayBuffer, which is what
 * lets the wasm backend run multi-threaded.
 *
 * API verified against the 4.3.0 sources served by the same CDN (src/pipelines/depth-estimation.js,
 * src/models/auto/{processing,modeling}_auto.js, src/image_processors_utils.js, src/utils/hub.js,
 * src/utils/dtypes.js, src/utils/tensor.js), not from memory.
 *
 * Messages in:
 *   {type:"init", modelKey, precision, inferWidth}
 *   {type:"frame", index, data:ArrayBuffer(RGBA WxH), width, height}   (data transferred)
 *   {type:"close"}
 * Messages out:
 *   {type:"progress", file, progress, loaded, total, status}
 *   {type:"ready", device, dtype, model}
 *   {type:"depth", index, data:ArrayBuffer(float32 H'xW'), width, height, ms, prepMs}  (data transferred)
 *   {type:"error", index?, message}
 */

// Pinned: @huggingface/transformers 4.3.0 (latest on npm 2026-09-18; jsdelivr's own entrypoint for
// the package is this exact file, so the bare-specifier form in the README resolves here).
import {
  AutoModelForDepthEstimation,
  AutoProcessor,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";

// Models are fetched from the HF CDN; a local lookup would hit the dev server and get HTML back.
env.allowLocalModels = false;

/** onnx-community repos. small is Apache-2.0; base and large are CC-BY-NC-4.0 (non-commercial).
 *  Deliberately NOT exported: depth-browser.js keeps its own copy, because importing anything from
 *  this file would evaluate the transformers.js bundle on the main thread. */
const MODEL_IDS = {
  small: "onnx-community/depth-anything-v2-small",
  base: "onnx-community/depth-anything-v2-base",
  large: "onnx-community/depth-anything-v2-large",
};

/** DINOv2 patch size. Every inference dimension has to be a multiple of it. */
const PATCH = 14;
const snap14 = (v) => Math.max(PATCH, Math.round(v / PATCH) * PATCH);

let model = null;
let processor = null;
let imageProc = null;       // the DPTImageProcessor inside the AutoProcessor wrapper
let loadError = null;

// One frame at a time, in arrival order. `onmessage` is async, so without this chain a second frame
// would enter the session while the first is still awaiting it — two concurrent runs on one ORT
// session, double the peak memory, and replies out of order.
let chain = Promise.resolve();

/**
 * VJ-9000's dtype rule, kept: WebGPU runs the chosen precision (auto means fp16), while the wasm
 * fallback prefers q8 on auto because full precision on wasm exhausts the heap
 * ("Array buffer allocation failed").
 */
const dtypeFor = (p) => (p === "fp32" ? "fp32" : p === "q8" ? "q8" : "fp16");
const webgpuDtypeFor = (p) => dtypeFor(p);
const wasmDtypeFor = (p) => (p === "auto" ? "q8" : dtypeFor(p));

const post = (msg, transfer) => (transfer ? self.postMessage(msg, transfer) : self.postMessage(msg));

/**
 * `navigator.gpu` exists in more places than an adapter does, and requestAdapter() can hang
 * forever rather than reject — measured in headless Chrome, where it never settles. A pending
 * probe must not become a stuck run, so it is raced against a deadline and treated as "no GPU".
 */
const ADAPTER_TIMEOUT_MS = 3000;
async function hasWebGPU() {
  try {
    if (!self.navigator?.gpu) return false;
    const adapter = await Promise.race([
      self.navigator.gpu.requestAdapter(),
      new Promise((r) => setTimeout(() => r(null), ADAPTER_TIMEOUT_MS)),
    ]);
    return !!adapter;
  } catch {
    return false;
  }
}

/** Load the processor and the session, walking the backend ladder. */
async function load({ modelKey, precision, inferWidth }) {
  if (model) return;
  const id = MODEL_IDS[modelKey] || MODEL_IDS.small;
  const prec = precision || "auto";
  const width = snap14(Number(inferWidth) > 0 ? Number(inferWidth) : 518);

  // 4.3.0 emits a `progress_total` aggregate alongside the per-file `progress`; the aggregate is
  // the one worth a progress bar, so it is passed through with file:"" and the caller picks.
  const progress_callback = (p) => {
    if (!p) return;
    if (p.status === "progress" || p.status === "progress_total") {
      post({
        type: "progress",
        status: p.status,
        file: p.file || "",
        progress: typeof p.progress === "number" ? p.progress : 0,
        loaded: p.loaded || 0,
        total: p.total || 0,
      });
    }
  };

  processor = await AutoProcessor.from_pretrained(id, { progress_callback });
  // AutoProcessor returns a Processor wrapping components.image_processor (a DPTImageProcessor for
  // these repos, per their preprocessor_config.json).
  imageProc = processor.image_processor || processor;
  // If a frame ever does need the processor's resize, aim it at the requested inference width
  // instead of the config's 518: with keep_aspect_ratio + ensure_multiple_of 14 a square target of
  // `width` is a no-op for a frame whose long side is already `width`, and a clean downscale
  // otherwise.
  if (imageProc && imageProc.size) imageProc.size = { width, height: width };

  const attempts = [];
  if (await hasWebGPU()) attempts.push({ device: "webgpu", dtype: webgpuDtypeFor(prec) });
  attempts.push({ device: "wasm", dtype: wasmDtypeFor(prec) });

  let last = null;
  for (const a of attempts) {
    try {
      model = await AutoModelForDepthEstimation.from_pretrained(id, {
        device: a.device,
        dtype: a.dtype,
        progress_callback,
      });
      post({ type: "ready", device: a.device, dtype: a.dtype, model: id });
      return;
    } catch (e) {
      last = e;
      model = null;
    }
  }
  loadError = String(last?.message ?? last ?? "model init failed");
  post({ type: "error", message: loadError });
}

/** Run one frame and reply with its float32 depth map. */
async function infer(msg) {
  const { index, data, width, height } = msg;
  if (!model || !processor) {
    post({ type: "error", index, message: loadError || "depth model not loaded" });
    return;
  }
  try {
    const image = new RawImage(new Uint8ClampedArray(data), width, height, 4);
    // Frames the driver sends are already multiples of 14, so skip the resize entirely; anything
    // else goes through the processor's own keep_aspect_ratio path.
    if (imageProc) imageProc.do_resize = width % PATCH !== 0 || height % PATCH !== 0;

    const t0 = performance.now();
    const inputs = await processor(image);
    const t1 = performance.now();
    const out = await model(inputs);
    const t2 = performance.now();

    const tensor = out?.predicted_depth ?? out?.logits ?? null;
    if (!tensor) throw new Error("model produced no predicted_depth tensor");
    // fp16 sessions return a float16 tensor (stored in a Uint16Array where Float16Array is
    // missing); Tensor.to('float32') handles both storages.
    const f32 = tensor.type === "float32" ? tensor : tensor.to("float32");
    const dims = f32.dims;
    const h = dims[dims.length - 2];
    const w = dims[dims.length - 1];
    const copy = new Float32Array(h * w);            // batch 0, detached so it can be transferred
    copy.set(f32.data.subarray(0, h * w));

    post(
      {
        type: "depth",
        index,
        data: copy.buffer,
        width: w,
        height: h,
        ms: t2 - t1,
        prepMs: t1 - t0,
      },
      [copy.buffer],
    );
  } catch (e) {
    post({ type: "error", index, message: String(e?.message ?? e) });
  }
}

self.onmessage = (ev) => {
  const msg = ev.data;
  if (!msg) return;
  if (msg.type === "init") {
    chain = chain.then(() => load(msg)).catch((e) => {
      loadError = String(e?.message ?? e);
      post({ type: "error", message: loadError });
    });
    return;
  }
  if (msg.type === "frame") {
    chain = chain.then(() => infer(msg)).catch((e) => {
      post({ type: "error", index: msg.index, message: String(e?.message ?? e) });
    });
    return;
  }
  if (msg.type === "close") {
    chain = chain.then(async () => {
      try {
        await model?.dispose();
      } catch {
        /* the worker is about to be terminated anyway */
      }
      model = null;
    });
  }
};

self.onerror = (e) => post({ type: "error", message: String(e?.message ?? e) });
