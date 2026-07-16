/**
 * Compare tab — two AresPlayers rendered pixel-aligned for A/B judgment.
 *
 * SYNC DESIGN: neither player owns a clock. This module runs ONE external clock and drives both
 * players by seek() every rAF, so both compute the same frame index from the same t — frame lock is
 * exact by construction (the earlier onFrame-chaining design could drift). seek() also re-renders,
 * so camera motion stays live on both sides even while paused.
 *
 * WIPE mode (default): canvases stacked, B clipped by a draggable divider — the eye never moves,
 * the mask is the only variable. SPLIT mode: classic side-by-side. A's canvas takes the pointer
 * (orbit/pan/zoom); B mirrors A's camera every frame.
 */
import { AresPlayer } from "@ares/core";

const $ = (id) => document.getElementById(id);

/** Curated labels for the recipes worth naming — anything else falls back to "name (size MB)".
 *  This is now a label OVERRIDE table, not the source of truth for what's selectable: the actual
 *  list comes from GET /list-ares (every .ares apps/demo/ has on disk, same endpoint the Viewer's
 *  "＋ Add…" source-bar picker uses — main.js's openAddPicker), so a new bake shows up here on the
 *  next Compare tab open with no code edit. */
const CURATED_LABELS = {
  "daniel-s0.ares": "KEEPER · smooth 0 · oct16+AV1 1024²",
  "daniel-dec06.ares": "decimate 0.6 · 12k tris · 36.6 MB (eval)",
  "daniel-s0hq.ares": "s0 HQ · 2048² texture (seam-line test)",
  "daniel-s1.ares": "smooth 1 · oct16+AV1 1024²",
  "daniel-v4.ares": "smooth 2 · oct16+AV1 1024²",
  "daniel-v5.ares": "smooth 3 · oct16+AV1 1024²",
  "daniel.ares": "v1 original · VP9+i8, no reorder (67 MB)",
  "daniel-v2.ares": "legacy smooth 0 · VP9+i8+reorder",
  "daniel-v3.ares": "legacy smooth 2 · VP9+i8+reorder",
  "demo.ares": "synth clip",
};
/** The live variant list — populated from /list-ares at init (see loadVariants). */
let VARIANTS = [];
const labelOf = (src) => VARIANTS.find((v) => v.src === src)?.label ?? src;

/** Pull every known .ares from the server (same source the "＋ Add…" source-bar picker uses) and
 *  label it: curated override when the filename matches, else a plain "name (size MB)". Falls back
 *  to the curated table's own keys if the server is unreachable, so the tab still has *something*
 *  selectable rather than two empty dropdowns. */
async function loadVariants() {
  let list = [];
  try { list = await fetch("/list-ares").then((r) => r.json()); } catch { /* server down */ }
  if (!Array.isArray(list) || !list.length) {
    VARIANTS = Object.keys(CURATED_LABELS).map((src) => ({ src, label: CURATED_LABELS[src] }));
    return VARIANTS;
  }
  VARIANTS = list.map((f) => ({
    src: f.src,
    label: CURATED_LABELS[f.src] || `${f.src} (${(f.bytes / 1048576).toFixed(1)} MB)`,
  }));
  return VARIANTS;
}

let A = null, B = null;
let raf = 0, playing = true, t = 0, lastNow = 0, duration = 9.07, frameCount = 272, orbitOn = false;

function fitBoth() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  for (const [p, c] of [[A, $("cmpCanvasA")], [B, $("cmpCanvasB")]]) {
    const w = Math.max(1, (c.clientWidth * dpr) | 0), h = Math.max(1, (c.clientHeight * dpr) | 0);
    if (p) p.resize(w, h); else { c.width = w; c.height = h; }
  }
}

/** Current Compare shading mode, applied to BOTH players (and re-applied to any player created
 *  later — otherwise swapping clip A while in clay would bring the new clip back textured). */
let cmpShade = "shaded";
function applyCmpShade() {
  for (const p of [A, B]) {
    if (!p) continue;
    p.setWireframe(cmpShade === "wire");
    p.setTextured(cmpShade !== "clay");
  }
}

async function createA(src) {
  const cam = A?.getCamera();
  const old = A; A = null; old?.dispose();
  A = await AresPlayer.create({ canvas: $("cmpCanvasA"), src, loop: true, autoOrbit: false });
  window.__cmpA = A; // debug handle (matches main.js's window.__ares)
  if (cam) A.setCamera(cam);
  applyCmpShade();
  frameCount = A.getStats().frameCount;
  duration = frameCount / 30;
  // Frame-accurate transport (same law as the viewer): the scrub value IS A's frame index.
  const scrub = $("cmpScrub");
  scrub.max = String(Math.max(1, frameCount - 1));
  scrub.step = "1";
  $("cmpLabelA").textContent = "◀ A: " + labelOf(src);
  fitBoth();
}

async function createB(src) {
  const old = B; B = null; old?.dispose();
  B = await AresPlayer.create({ canvas: $("cmpCanvasB"), src, loop: true, autoOrbit: false });
  window.__cmpB = B; // debug handle
  if (A) B.setCamera(A.getCamera());
  applyCmpShade();
  $("cmpLabelB").textContent = "B: " + labelOf(src) + " ▶";
  fitBoth();
}

/** The shared clock: advance t, mirror the camera, seek BOTH players to the same instant. */
function drive(now) {
  raf = requestAnimationFrame(drive);
  const dt = Math.min(0.05, (now - lastNow) / 1000);
  lastNow = now;
  if (playing) t = (t + dt) % duration;
  if (A) {
    if (orbitOn) { const c = A.getCamera(); c.azimuth += dt * 0.2; A.setCamera(c); }
    if (B) B.setCamera(A.getCamera());
    A.seek(t);
    B?.seek(t);
    const f = Math.min(frameCount - 1, Math.floor(t * 30 + 1e-6));
    const scrub = $("cmpScrub");
    if (!scrub.matches(":active")) scrub.value = String(f);
    $("cmpTime").textContent = `${f + 1}/${frameCount} · ${t.toFixed(2)}s / ${duration.toFixed(2)}s`;
  }
}

function wireWipe() {
  const stage = $("cmpStage"), handle = $("wipeHandle");
  const setWipe = (clientX) => {
    const r = stage.getBoundingClientRect();
    const pct = Math.min(98, Math.max(2, ((clientX - r.left) / r.width) * 100));
    stage.style.setProperty("--wipe", pct.toFixed(2) + "%");
  };
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const move = (ev) => setWipe(ev.clientX);
    const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });
}

export async function initCompare() {
  const selA = $("cmpA"), selB = $("cmpB");
  const variants = await loadVariants();
  selA.innerHTML = ""; selB.innerHTML = "";
  for (const v of variants) { selA.append(new Option(v.label, v.src)); selB.append(new Option(v.label, v.src)); }
  const has = (src) => variants.some((v) => v.src === src);
  // Keep the current defaults when present: the live question was does the 2048² texture tier …
  // … remove the faint seam/shading lines? Fall back to the first two entries if this .ares set
  // doesn't have them (e.g. a fresh checkout before any bake has run).
  selA.value = has("daniel-s0.ares") ? "daniel-s0.ares" : (variants[0]?.src ?? "");
  selB.value = has("daniel-s0hq.ares") ? "daniel-s0hq.ares" : (variants[1]?.src ?? variants[0]?.src ?? "");
  selA.onchange = () => createA(selA.value).catch(showErr);
  selB.onchange = () => createB(selB.value).catch(showErr);

  $("cmpPlay").onclick = () => { playing = !playing; $("cmpPlay").textContent = playing ? "Pause" : "Play"; };
  const orbitBtn = $("cmpOrbit");
  orbitBtn.setAttribute("aria-pressed", "false");
  orbitBtn.onclick = () => { orbitOn = !orbitOn; orbitBtn.setAttribute("aria-pressed", String(orbitOn)); };
  $("cmpMode").onclick = () => {
    const stage = $("cmpStage");
    const toSplit = stage.classList.contains("wipe");
    stage.classList.toggle("wipe", !toSplit);
    stage.classList.toggle("split", toSplit);
    $("cmpMode").textContent = toSplit ? "Wipe view" : "Split view";
    fitBoth();
  };
  // Viewport shading for BOTH clips at once — an A/B is only fair if both sides shade identically.
  // Re-applied in createA/createB too, since a player created later must adopt the current mode.
  const cmpShadeSeg = $("cmpShadeSeg");
  if (cmpShadeSeg) cmpShadeSeg.onclick = (e) => {
    const b = e.target.closest("button[data-shade]");
    if (!b) return;
    cmpShade = b.dataset.shade;
    for (const x of cmpShadeSeg.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x.dataset.shade === cmpShade));
    applyCmpShade();
  };
  $("cmpScrub").oninput = (e) => {
    if (playing) { playing = false; $("cmpPlay").textContent = "Play"; }
    t = Number(e.target.value) / 30;   // value IS the frame index; core's seek guard makes k/30 exact
  };
  wireWipe();
  window.addEventListener("resize", fitBoth);

  try { await createA(selA.value); await createB(selB.value); } catch (e) { showErr(e); return; }
  cancelAnimationFrame(raf);
  lastNow = performance.now();
  drive(lastNow);
}

function showErr(e) {
  $("cmpLabelA").textContent = "load failed: " + (e && e.message ? e.message : e);
  console.error(e);
}
