/**
 * Convert tab: 2D video → 2.5D volumetric card.
 *
 * Ported from VJ-9000 (github.com/gantasmo/VJ-9000, src/useDepthCloud.ts: the "depthcloud" live
 * source) and turned into an offline conversion, 2026-09-18. A monocular depth model estimates a
 * depth map per frame, the encoder unprojects each map through a pinhole ray table into a relief
 * mesh and textures it with the video frame itself, and the result is an ordinary .ares clip.
 *
 * Two depth engines, one run-directory contract (docs/depth-2d-to-25d.md):
 *   service  the local Python service (tools/sam-service/depth.py): Video-Depth-Anything (32-frame
 *            windows, consistent across the clip) or Depth-Anything-V2 (per frame) on CUDA, float
 *            output, every model size, and the optional SAM 3 subject mask pass. The dev server
 *            drives it inside the same SSE stream as the encode (/depth-convert).
 *   browser  the VJ-9000 worker itself (depth-worker.js via depth-browser.js): transformers.js on
 *            WebGPU or wasm. The maps upload to the dev server (/depth/upload/*), which then runs
 *            the same encode. This is the engine for a machine without the Python environment.
 *
 * Only the log scrolls; the secondary groups are collapsed <details> so the card fits one viewport.
 */

import { accessPrompt, fetchJsonEnsuring, sseErrorData } from "./ensure.js";

const $ = (id) => document.getElementById(id);
const esc =(s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const VIDEO_EXT = /\.(mp4|m4v|mov|webm|mkv|avi|mpg|mpeg|wmv|ts)$/i;
const PICK_FILTER = "Video|*.mp4;*.m4v;*.mov;*.webm;*.mkv;*.avi;*.mpg;*.mpeg;*.wmv|All files|*.*";
// Model keys the service knows; the browser engine covers the V2 relative three.
const SERVICE_MODELS = [
  ["video-small", "Video Small: 28M"], ["video-base", "Video Base: 113M"], ["video-large", "Video Large: 382M"],
  ["base", "V2 Base: 97M"], ["small", "V2 Small: 25M"], ["large", "V2 Large: 335M"],
  ["metric-indoor-base", "Metric indoor Base"], ["metric-indoor-small", "Metric indoor Small"], ["metric-indoor-large", "Metric indoor Large"],
  ["metric-outdoor-base", "Metric outdoor Base"], ["metric-outdoor-small", "Metric outdoor Small"], ["metric-outdoor-large", "Metric outdoor Large"],
];
const BROWSER_MODELS = [["small", "V2 Small: 25M"], ["base", "V2 Base: 97M"], ["large", "V2 Large: 335M"]];
// Progress bar spans per stage (percent of the whole conversion), depth being the long one. A
// subject run splits the depth span: the mask pass first, then depth (MASKED_DEPTH).
const STAGE_SPAN = { model: [0, 4], mask: [4, 30], depth: [4, 55], rgb: [55, 59], stabilize: [59, 64], mesh: [64, 78], texture: [78, 95], mux: [95, 100] };
const MASKED_DEPTH = [30, 55];

let hooks = { addToShowcase: null, refreshHistory: null };
let engines = null;          // /depth/engines snapshot for the current card
let probe = null;            // /probe-video result for the current path
let running = null;          // { es, abort } while a conversion runs

export const isVideoName = (name) => VIDEO_EXT.test(String(name || ""));

/** convert.js hands over the two things the card cannot own: the source bar and the history list. */
export function initDepthCard(h) { hooks = { ...hooks, ...h }; }

/** Native file dialog, then the card. */
export async function pickVideoAndRender() {
  let picked;
  try { picked = await fetch("/pick?type=file&for=depthvideo&filter=" + encodeURIComponent(PICK_FILTER)).then((r) => r.json()); }
  catch { $("convertOut").innerHTML = `<div class="card"><div class="cap" style="color:var(--bad)">Picker requires the ARES dev server</div></div>`; return; }
  if (!picked || !picked.path) return; // cancelled
  await renderDepthCard(picked.path);
}

/**
 * Render the card for `path` (an absolute video path, or "" for a dropped file the browser cannot
 * locate). `opts.hint` names the dropped file; `opts.settings` re-applies a history recipe.
 */
export async function renderDepthCard(path, opts = {}) {
  const out = $("convertOut");
  if (running) stopRun("replaced");
  probe = null;
  const base = (path || opts.hint || "video").split(/[\\/]/).pop().replace(VIDEO_EXT, "");
  const name = base.replace(/[^a-z0-9_-]/gi, "_").slice(0, 60) || "video";
  const s = opts.settings || {};
  out.innerHTML = `
    <div class="card cv2" id="dcCard">
      <section class="cvcol">
        <div class="cap">Source</div>
        <div class="fld"><span class="k">video</span><span class="row">
          <input id="dcPath" class="inp" value="${esc(path)}" placeholder="${esc(opts.hint ? "full path to " + opts.hint : "path to a video file")}" style="flex:1">
          <button class="u" id="dcPickBtn" title="Pick the video with the native dialog">Browse…</button>
        </span></div>
        <div class="note2" id="dcProbeNote">${path ? "reading stream info…" : "no file selected"}</div>
        <dl class="stats" id="dcStats" style="display:none">
          <dt>stream</dt><dd id="dcStream"></dd>
          <dt>frames</dt><dd id="dcFrames"></dd>
        </dl>
        <div class="fld"><span class="k">engine</span><span class="row">
          <select id="dcEngine" class="inp" style="flex:1;max-width:190px" title="service: Video-Depth-Anything or Depth-Anything-V2 on the local CUDA service, float inference. browser: Depth-Anything-V2 in this tab through WebGPU or wasm, no Python needed.">
            <option value="auto">auto</option>
            <option value="service">service: CUDA</option>
            <option value="browser">browser: WebGPU</option>
          </select>
          <select id="dcModel" class="inp" style="flex:1;max-width:190px" title="Depth checkpoint. Video: Video-Depth-Anything, 32-frame windows with scale and shift consistent across the clip (service engine). V2: Depth-Anything-V2, each frame alone. Small is Apache-2.0; Base and Large are CC-BY-NC-4.0. Metric variants output metres and skip the near/far mapping."></select>
        </span></div>
        <div class="note2" id="dcEngineNote">checking engines…</div>
        <details class="sec"><summary>Inference</summary><div class="body">
          <div class="fld"><span class="k">width</span><span class="row">
            <select id="dcInferWidth" class="inp" title="Inference width in pixels (multiples of 14; the model's native scale is 518). Larger costs time quadratically and sharpens silhouettes.">
              <option value="322">322</option><option value="518" selected>518</option><option value="714">714</option><option value="924">924</option></select>
            <span class="k">precision</span>
            <select id="dcPrecision" class="inp" title="browser engine only: WebGPU fp16 by default, wasm q8 as the fallback">
              <option value="auto">auto</option><option value="fp16">fp16</option><option value="fp32">fp32</option><option value="q8">q8</option></select>
          </span></div>
          <div class="fld"><span class="k">sample</span><span class="row">
            <input id="dcFps" class="inp" type="number" min="0.1" max="240" step="0.001" placeholder="source" style="width:76px" title="Frame rate to sample the video at; blank keeps every source frame"><span class="suf">fps</span>
            <span class="k">frames</span><input id="dcMax" class="inp" type="number" min="1" placeholder="all" style="width:72px" title="Bound the run to the first N sampled frames">
          </span></div>
          <div class="fld"><span class="k">subject</span><span class="row">
            <input id="dcSubject" class="inp" maxlength="200" placeholder="none" style="flex:1;max-width:230px" title="SAM 3 text prompt, e.g. person. A mask pass runs before depth: the depth model sees the subject on black and the relief keeps the subject only. Service engine.">
          </span></div>
        </div></details>
      </section>

      <section class="cvcol">
        <div class="cap">Output</div>
        <div class="fld"><span class="k">name</span><span class="row">
          <input id="dcName" class="inp" value="${esc(name)}" style="flex:1;max-width:230px"><span class="suf">.ares</span></span></div>
        <div class="fld"><span class="k">texture</span><span class="row">
          <select id="dcCodec" class="inp" title="The video frame is the atlas; AV1 is smaller at equal quality"><option value="vp9">VP9</option><option value="av1">AV1</option></select>
          <select id="dcSize" class="inp"><option value="1024">1024²</option><option value="2048">2048²</option><option value="512">512²</option></select>
          <span class="k">CRF</span><input id="dcCrf" class="inp" type="number" value="30" min="0" max="63" style="width:52px">
        </span></div>
        <div class="fld"><span class="k">grid</span><span class="row">
          <select id="dcGrid" class="inp" title="Mesh columns across the frame; rows follow the aspect. 256 ≈ 37k vertices per frame at 16:9.">
            <option value="128">128</option><option value="192">192</option><option value="256" selected>256</option><option value="320">320</option><option value="384">384</option><option value="512">512</option></select>
          <span class="k">FOV</span><input id="dcFov" class="inp" type="number" value="55" min="10" max="150" step="1" style="width:56px" title="Assumed vertical field of view of the camera in degrees (the VJ-9000 default is 55)"><span class="suf">°</span>
        </span></div>
        <details class="sec"><summary>Geometry</summary><div class="body">
          <div class="fld"><span class="k">range</span><span class="row">
            <input id="dcNear" class="inp" type="number" min="0.01" step="0.1" placeholder="2" style="width:60px" title="Nearest depth in metres: the model's nearest disparity lands here. Blank: 2 for a relative model, 0.5 for a metric one. far / near is the scale of the background against the foreground seen from anywhere but the capture camera."><span class="suf">near</span>
            <input id="dcFar" class="inp" type="number" min="0.02" step="0.5" placeholder="6" style="width:60px" title="Farthest depth in metres. Blank: 6 for a relative model, 20 for a metric one; metric models clamp to the range."><span class="suf">far</span>
          </span></div>
          <div class="fld"><span class="k">edge</span><span class="row">
            <input id="dcEdge" class="inp" type="number" value="0.08" min="0" max="10" step="0.01" style="width:64px" title="Silhouette cut: a triangle whose edge spans more than this fraction of its depth is dropped, so foreground and background never join by a stretched sheet. 0 keeps everything.">
            <label class="row" style="gap:5px;cursor:pointer;color:var(--text-mid)" title="Keep the full grid every frame: identical topology per frame, so geometry codes as I+P deltas and the file is smaller. The triangles across silhouettes are discarded at draw time.">
              <input type="checkbox" id="dcSheets"><span>sheets</span></label>
            <span class="k">decimate</span><input id="dcDecimate" class="inp" type="number" min="0.01" max="1" step="0.05" placeholder="off" style="width:60px" title="Fraction of triangles kept per frame after the silhouette cut (meshoptimizer simplify). Excludes sheets.">
          </span></div>
          <div class="fld"><span class="k">surface</span><span class="row">
            <label class="row" style="gap:5px;cursor:pointer;color:var(--text-mid)" title="Image-guided resampling: joint bilateral filter, depth edges follow the colour edges of the frame">
              <input type="checkbox" id="dcGuided" checked><span>guided</span></label>
            <label class="row" style="gap:5px;cursor:pointer;color:var(--text-mid)" title="Fill layer: the background continued behind every silhouette, textured from a plate under the frame in the atlas (atlas 1.5x taller)">
              <input type="checkbox" id="dcInpaint"><span>fill</span></label>
            <input id="dcBand" class="inp" type="number" min="1" max="4096" step="1" placeholder="auto" style="width:60px" title="Fill band: grid cells the background is continued under the foreground (default 16% of the grid width)"><span class="suf">cells</span>
            <label class="row" style="gap:5px;cursor:pointer;color:var(--text-mid)" title="Ramp snap: a vertex partway down a silhouette ramp takes the depth of the side its colour matches">
              <input type="checkbox" id="dcSnap"><span>snap</span></label>
          </span></div>
          <div class="fld"><span class="k">stabilize</span><span class="row">
            <input id="dcStabilize" type="range" min="0" max="100" value="70" style="flex:1;min-width:70px" title="Temporal stabilization: per-frame scale/shift alignment plus motion-gated smoothing (static pixels smooth, moving pixels follow). 0 is off.">
            <span id="dcStabilizeVal" style="font:11px ui-monospace,monospace;color:var(--text-mid);min-width:32px">70%</span>
          </span></div>
          <div class="fld"><span class="k">origin</span><span class="row">
            <select id="dcCenter" class="inp" title="Where the clip's origin lands: bottom stands the relief on the floor grid, mass centres it, none keeps camera space (camera at the origin looking down −Z)">
              <option value="bottom">bottom: floor</option><option value="mass">mass: centre</option><option value="none">none: camera space</option></select>
            <label class="row" style="gap:5px;cursor:pointer;color:var(--text-mid)" title="Carry the video's own audio track into the clip (Opus)">
              <input type="checkbox" id="dcAudio" checked><span>audio</span></label>
            <label class="row" style="gap:5px;cursor:pointer;color:var(--text-mid)" title="Letterbox crop: rows and columns black in every keyframe are cut from the maps and the texture">
              <input type="checkbox" id="dcCrop" checked><span>letterbox</span></label>
          </span></div>
        </div></details>
      </section>

      <div class="full">
        <div class="cvfoot">
          <button class="u primary" id="dcGo" disabled>Convert</button>
          <button class="u" id="dcStop" style="display:none">Cancel</button>
          <div class="prog" id="dcProg"><div></div></div>
          <span class="note2" id="dcStage" style="min-width:120px;text-align:right"></span>
        </div>
        <details class="sec"><summary>Log</summary><div class="body"><pre id="dcLog" class="cvlog"></pre></div></details>
        <div id="dcDone"></div>
      </div>
    </div>`;

  $("dcPickBtn").onclick = async () => {
    let picked;
    try { picked = await fetch("/pick?type=file&for=depthvideo&filter=" + encodeURIComponent(PICK_FILTER)).then((r) => r.json()); }
    catch { $("dcProbeNote").innerHTML = `<span style="color:var(--bad)">Picker requires the ARES dev server</span>`; return; }
    if (!picked || !picked.path) return;
    $("dcPath").value = picked.path;
    if (!$("dcName").dataset.edited) $("dcName").value = picked.path.split(/[\\/]/).pop().replace(VIDEO_EXT, "").replace(/[^a-z0-9_-]/gi, "_").slice(0, 60) || "video";
    probePath(picked.path);
  };
  $("dcPath").addEventListener("change", () => { const p = $("dcPath").value.trim(); if (p) probePath(p); });
  $("dcName").addEventListener("input", () => { $("dcName").dataset.edited = "1"; });
  $("dcEngine").onchange = syncEngine;
  $("dcStabilize").oninput = () => { $("dcStabilizeVal").textContent = $("dcStabilize").value + "%"; };
  // Decimation re-triangulates each frame; sheets keep one topology for the clip.
  $("dcSheets").onchange = () => { $("dcDecimate").disabled = $("dcSheets").checked; };
  $("dcGo").onclick = runDepthConvert;
  $("dcStop").onclick = () => stopRun("cancelled");
  applySettings(s);
  await loadEngines();
  syncEngine();
  if (path) probePath(path);
}

function applySettings(m) {
  const set = (id, v) => { const el = $(id); if (el && v != null && v !== "") el.value = v; };
  set("dcCodec", m.codec); set("dcSize", m.texSize); set("dcCrf", m.crf); set("dcGrid", m.grid); set("dcFov", m.fov);
  set("dcNear", m.near); set("dcFar", m.far); set("dcEdge", m.edge); set("dcMax", m.maxFrames); set("dcFps", m.sampleFps);
  set("dcInferWidth", m.inferWidth); set("dcEngine", m.engine); set("dcModel", m.model);
  if (m.stabilize != null && m.stabilize !== "") { $("dcStabilize").value = Math.round(Number(m.stabilize) * 100); $("dcStabilizeVal").textContent = $("dcStabilize").value + "%"; }
  if ($("dcSheets")) $("dcSheets").checked = m.sheets === "1";
  set("dcSubject", m.subject); set("dcDecimate", m.decimate); set("dcBand", m.inpaintBand);
  if (m.inpaint != null) $("dcInpaint").checked = m.inpaint === "1";
  if (m.guided != null) $("dcGuided").checked = m.guided !== "0";
  if (m.snapRamps != null) $("dcSnap").checked = m.snapRamps === "1";
  if (m.crop != null) $("dcCrop").checked = m.crop !== "none";
  $("dcDecimate").disabled = $("dcSheets").checked;
}

async function loadEngines() {
  engines = null;
  try { engines = await fetch("/depth/engines").then((r) => r.json()); } catch { engines = null; }
  // The browser engine's own capability probe lives in the ported module; a missing module just
  // means the option is greyed, the service engine is unaffected.
  try { const mod = await import("./depth-browser.js"); engines = { ...(engines || {}), browserProbe: await mod.probeBrowserDepth() }; }
  catch (e) { engines = { ...(engines || {}), browserProbe: null, browserError: (e && e.message) || String(e) }; }
}

/** Which engine "auto" resolves to, and the status line under the selector. */
function resolveEngine() {
  const want = $("dcEngine") ? $("dcEngine").value : "auto";
  // The service engine is always eligible: /depth-convert installs the Python environment, the
  // model weights and ffmpeg inside its own stream when they are absent.
  const svcOk = !!(engines && engines.service);
  const brOk = !!(engines && engines.browserProbe);
  if (want === "service") return "service";
  if (want === "browser") return "browser";
  return svcOk ? "service" : brOk ? "browser" : "service";
}

function syncEngine() {
  if (!$("dcModel")) return;
  const eng = resolveEngine();
  const list = eng === "browser" ? BROWSER_MODELS : SERVICE_MODELS;
  const cur = $("dcModel").value;
  $("dcModel").innerHTML = list.map(([k, l]) => `<option value="${k}">${esc(l)}</option>`).join("");
  $("dcModel").value = list.some(([k]) => k === cur) ? cur : list[0][0];
  $("dcPrecision").disabled = eng !== "browser";
  $("dcSubject").disabled = eng !== "service";
  const parts = [];
  if (!engines) parts.push("dev server not reachable");
  else {
    const svc = engines.service || {};
    const depth = svc.depth || {};
    parts.push(svc.env
      ? `service: env ready${svc.running ? (depth.ready ? ` · ${depth.model} loaded` : depth.error ? " · depth error: " + esc(depth.error) : " · running") : " · starts on demand"}${engines.ffmpeg ? "" : " · ffmpeg installs on Convert"}`
      : "service: Python environment installs on Convert");
    const bp = engines.browserProbe;
    parts.push(bp ? `browser: ${bp.webgpu ? "WebGPU" : "wasm only"}${bp.isolated === false ? " · not cross-origin isolated" : ""}` : `browser: engine module unavailable${engines.browserError ? " (" + esc(engines.browserError) + ")" : ""}`);
    if (!engines.encoder) parts.push("encoder builds on Convert");
  }
  $("dcEngineNote").innerHTML = `→ ${eng}${$("dcEngine").value === "auto" ? " (auto)" : ""} · ` + parts.join(" · ");
  gate();
}

function gate() {
  const go = $("dcGo");
  if (!go) return;
  go.disabled = !!running || !probe || !!probe.error || !engines;
}

async function probePath(p) {
  const note = $("dcProbeNote"), stats = $("dcStats");
  if (!note) return;
  probe = null; gate();
  note.textContent = "reading stream info…"; note.style.color = "";
  // ffprobe absent: the probe answers { needs:["ffmpeg"] }, the install runs here with its
  // progress in this status line, and the probe repeats.
  let info;
  const status = (t) => { const n = $("dcProbeNote"); if (n && $("dcPath").value.trim() === p) n.textContent = String(t).trim().slice(0, 140); };
  try { info = await fetchJsonEnsuring("/probe-video?path=" + encodeURIComponent(p), { onLog: status, onStep: (s) => status(`${s.label}: ${s.index + 1} of ${s.total}`) }); }
  catch { note.textContent = "Probe failed: ARES dev server not reachable"; note.style.color = "var(--bad)"; return; }
  if ($("dcPath").value.trim() !== p) return; // a newer pick replaced this one
  if (!info || info.error) { note.textContent = (info && info.error) || "probe failed"; note.style.color = "var(--warn)"; stats.style.display = "none"; return; }
  probe = info;
  note.textContent = ""; stats.style.display = "";
  $("dcStream").textContent = `${info.width}×${info.height} · ${info.fps} fps · ${info.codec || "?"}${info.hasAudio ? " · audio" : " · no audio"}`;
  $("dcFrames").textContent = `${info.frames || "?"} · ${info.durationS ? info.durationS.toFixed(2) + " s" : "?"} · ${info.sizeBytes ? (info.sizeBytes / 1048576).toFixed(1) + " MB" : ""}`;
  $("dcMax").placeholder = info.frames ? `all ${info.frames}` : "all";
  $("dcFps").placeholder = String(info.fps || "source");
  $("dcAudio").checked = !!info.hasAudio; $("dcAudio").disabled = !info.hasAudio;
  gate();
}

function stopRun(why) {
  if (!running) return;
  const r = running; running = null;
  try { r.abort.abort(); } catch { /* not started */ }
  try { r.es?.close(); } catch { /* not open */ }
  if ($("dcGo")) { $("dcGo").disabled = false; $("dcGo").textContent = "Convert"; }
  if ($("dcStop")) $("dcStop").style.display = "none";
  if (why === "cancelled" && $("dcStage")) $("dcStage").textContent = "cancelled";
  gate();
}

function setProgress(stage, frac, masked = false) {
  const span = stage === "depth" && masked ? MASKED_DEPTH : STAGE_SPAN[stage];
  const prog = $("dcProg");
  if (!prog) return;
  if (!span) { prog.firstChild.style.width = "96%"; return; }
  const f = Math.max(0, Math.min(1, frac ?? 0));
  prog.firstChild.style.width = (span[0] + (span[1] - span[0]) * f).toFixed(1) + "%";
}

async function runDepthConvert() {
  const video = $("dcPath").value.trim();
  if (!video || !probe || running) return;
  const engine = resolveEngine();
  const name = ($("dcName").value.trim() || "video").replace(/[^a-z0-9_-]/gi, "_");
  const v = (id) => $(id).value.trim();
  const q = new URLSearchParams({ video, name, engine, model: v("dcModel"), textureCodec: v("dcCodec"), texSize: v("dcSize"), crf: v("dcCrf"),
    grid: v("dcGrid"), fov: v("dcFov"), edge: v("dcEdge"), stabilize: String(Number($("dcStabilize").value) / 100),
    inferWidth: v("dcInferWidth"), center: v("dcCenter") });
  // Blank near/far leave the range to the encoder, which picks it by the model's kind.
  if (v("dcNear")) q.set("near", v("dcNear"));
  if (v("dcFar")) q.set("far", v("dcFar"));
  if (v("dcFps")) q.set("fps", v("dcFps"));
  if (v("dcMax")) q.set("maxFrames", v("dcMax"));
  if ($("dcSheets").checked) q.set("sheets", "1");
  if (!$("dcAudio").checked) q.set("noAudio", "1");
  if (engine === "service" && v("dcSubject")) q.set("subject", v("dcSubject"));
  if (v("dcDecimate") && !$("dcSheets").checked && Number(v("dcDecimate")) < 1) q.set("decimate", v("dcDecimate"));
  if ($("dcInpaint").checked) { q.set("inpaint", "1"); if (v("dcBand")) q.set("inpaintBand", v("dcBand")); }
  if (!$("dcGuided").checked) q.set("noGuided", "1");
  if ($("dcSnap").checked) q.set("snapRamps", "1");
  if (!$("dcCrop").checked) q.set("crop", "none");

  const log = $("dcLog"), prog = $("dcProg"), done = $("dcDone"), go = $("dcGo"), stage = $("dcStage");
  log.style.display = "block"; const sec = log.closest("details"); if (sec) sec.open = true;
  prog.style.display = "block"; prog.firstChild.style.width = "1%"; prog.firstChild.style.background = "";
  done.innerHTML = ""; go.disabled = true; go.textContent = "Converting…"; $("dcStop").style.display = "";
  const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
  line(`\n=== ${name}.ares ← ${video} (${engine} · ${v("dcModel")} · grid ${v("dcGrid")} · fov ${v("dcFov")} · ${v("dcCodec")} ${v("dcSize")}² crf${v("dcCrf")}) ===`);
  const abort = new AbortController();
  running = { es: null, abort };
  const finish = (ok, msg) => {
    const wasRunning = !!running;
    stopRun();
    if (!wasRunning) return;
    if (ok) { prog.firstChild.style.width = "100%"; stage.textContent = "done"; }
    else { prog.firstChild.style.background = "var(--bad)"; stage.textContent = "failed"; line("✗ " + msg); done.innerHTML = `<div class="note2" style="color:var(--bad);margin-top:8px">${esc(msg)}</div>`; }
  };

  try {
    if (engine === "browser") {
      // The VJ-9000 worker runs in this tab; its maps upload as they come and the server encodes
      // afterwards from the finished job.
      stage.textContent = "model";
      const mod = await import("./depth-browser.js");
      const fpsIn = v("dcFps") ? Number(v("dcFps")) : null;
      const maxIn = v("dcMax") ? Number(v("dcMax")) : null;
      const r = await mod.runBrowserDepth({
        video, sourceUrl: "/depth/source?path=" + encodeURIComponent(video), sourceFps: probe.fps, fps: fpsIn, maxFrames: maxIn,
        modelKey: v("dcModel"), precision: v("dcPrecision"), inferWidth: Number(v("dcInferWidth")) || 518,
        onProgress: (p) => { stage.textContent = `${p.stage} ${p.done ?? ""}${p.total ? "/" + p.total : ""}${p.msPerFrame ? " · " + p.msPerFrame.toFixed(0) + " ms" : ""}`; setProgress(p.stage, p.total ? p.done / p.total : 0); },
        onLog: (l) => line("[browser] " + l),
        signal: abort.signal,
      });
      if (!running) return;
      line(`[browser] ${r.frames} depth frames ${r.width}×${r.height} · ${r.device} ${r.dtype} · ${r.msPerFrame ? r.msPerFrame.toFixed(1) + " ms/frame" : ""}`);
      q.set("depthJob", r.job);
    }
  } catch (e) {
    finish(false, `browser engine: ${(e && e.message) || e}`);
    return;
  }
  if (!running) return;

  const es = new EventSource("/depth-convert?" + q.toString());
  running.es = es;
  es.addEventListener("start", (e) => { const d = JSON.parse(e.data); line(`▶ ${d.engine} depth → ${d.out}`); });
  es.addEventListener("log", (e) => { line(JSON.parse(e.data)); });
  let masked = false;   // a subject run: the mask pass holds the first part of the depth span
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    if (d.stage === "encode") { stage.textContent = "encode"; return; }
    if (d.stage === "mask") masked = true;
    const frac = d.of ? d.frame / d.of : 0;
    setProgress(d.stage, frac, masked);
    stage.textContent = `${d.stage}${d.state && d.state !== "running" ? " · " + d.state : ""} ${d.of ? `${d.frame}/${d.of}` : d.frame || ""}${d.msPerFrame ? " · " + Number(d.msPerFrame).toFixed(0) + " ms" : ""}`;
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    finish(true);
    done.innerHTML = `<div class="note2" style="color:var(--good);margin-top:10px">wrote ${esc(d.out)} · ${d.frames ?? "?"} frames · ${d.fps ?? "?"} fps · ${d.seconds ?? "?"} s</div>
      <button class="u" id="dcOpen">Open in Viewer</button>
      <button class="u" id="dcShowcase">Add to source bar</button>`;
    const src = d.out.split("/").pop();
    $("dcOpen").onclick = () => { location.search = "?src=" + src; };
    $("dcShowcase").onclick = async (ev) => {
      const ok = hooks.addToShowcase ? await hooks.addToShowcase(src, src.replace(/\.ares$/i, "")) : false;
      ev.target.textContent = ok ? "Added" : "✗ failed"; ev.target.disabled = ok;
    };
    if (hooks.refreshHistory) hooks.refreshHistory();
  });
  es.addEventListener("error", (e) => {
    const d = sseErrorData(e);
    finish(false, (d && d.message) || "conversion failed");
    // A gated repository (SAM 3 for a subject mask) raises the access prompt; Resume repeats the run.
    if (d && d.gated) accessPrompt(done, d, () => $("dcGo") && $("dcGo").click());
  });
  es.onerror = () => { /* SSE stream closed by the server */ };
}
