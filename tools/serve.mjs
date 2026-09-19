/**
 * ARES dev server — zero-dependency static server with cross-origin isolation.
 *
 * Sends COOP/COEP on every response so `crossOriginIsolated` is true and
 * SharedArrayBuffer is available — the probe measures both (spec §10.2 assumes
 * they stay reachable), and `npx serve` / `python -m http.server` do NOT send
 * these headers, which is why the probe reported SharedArrayBuffer: no. The
 * runtime does not depend on it: worker decode transfers ArrayBuffers.
 *
 * Usage:  node tools/serve.mjs [port]     (default 8137, walks up if busy)
 * Serves: the ares/ repo root (parent of tools/), / redirects to the probe.
 */
import { createServer, request as httpRequest } from "node:http";
import { stat, readFile, writeFile, appendFile, readdir, mkdir, link, copyFile, unlink, rename, mkdtemp, rm, statfs, open } from "node:fs/promises";
import { totalmem, freemem, homedir } from "node:os";
import { statSync, existsSync, createReadStream } from "node:fs";
import { spawn } from "node:child_process";
import { join, normalize, extname, dirname, basename, sep } from "node:path";
import zlib from "node:zlib";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { catalog, encoderState, ensurePrivatePython, findFfmpeg, findGit, FORGE_PYTHON, gpuProbe, hfTokenPresent, installOne, paths as installPaths, preflight, profiles, recommend, resolve, saveHfToken } from "./installer.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url))); // ares/

// Never overwrite a clip: if the requested <name> already exists, auto-bump to the next -vN so versions
// accumulate (v1, v2, v3 … — no manual renaming). base = name minus any trailing -vN;
// a plain base.ares counts as v1. Returns the name WITHOUT extension.
async function versionedOutName(dirAbs, requested) {
  const clean = requested.replace(/\.ares$/i, "");
  const base = clean.replace(/-v\d+$/i, "");
  const esc = base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`^${esc}(?:-v(\\d+))?\\.ares$`, "i");
  let maxV = 0, plain = false, any = false, files = [];
  try { files = await readdir(dirAbs); } catch { /* dir may not exist yet */ }
  for (const f of files) { const m = re.exec(f); if (m) { any = true; if (m[1]) maxV = Math.max(maxV, +m[1]); else plain = true; } }
  if (!any) return clean;                                  // fresh name → keep exactly as requested
  return `${base}-v${Math.max(maxV, plain ? 1 : 0) + 1}`;  // collision → next version, never overwrite
}
const BASE_PORT = Number(process.argv[2] ?? process.env.ARES_PORT ?? 8137);
const PROBE = "/apps/phase0-probe/";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".map": "application/json",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".webp": "image/webp",
  ".ktx2": "image/ktx2",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".ares": "application/octet-stream",
  ".ply": "application/octet-stream",
  ".md": "text/markdown; charset=utf-8",
  ".ts": "text/plain; charset=utf-8",
};

const HEADERS = {
  // The two headers that make crossOriginIsolated true:
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Cache-Control": "no-store",
  "X-Content-Type-Options": "nosniff",
};

// ---- RunPod control (in-app Compute panel). The API key stays SERVER-SIDE: the browser only ever
// calls /runpod/*, and this process forwards to RunPod with the key. Key source order: RUNPOD_API_KEY
// env var, else the file at RUNPOD_KEY_FILE (default .runpod/runpod.key — a git-ignored dir). ----
const RUNPOD_KEY_FILE = process.env.RUNPOD_KEY_FILE || join(ROOT, ".runpod", "runpod.key");
async function readRunpodKey() {
  if (process.env.RUNPOD_API_KEY) return process.env.RUNPOD_API_KEY.trim();
  try { return (await readFile(RUNPOD_KEY_FILE, "utf8")).trim(); } catch { return null; }
}
async function runpodGraphQL(key, query, variables) {
  const r = await fetch("https://api.runpod.io/graphql", {
    method: "POST",
    headers: { Authorization: "Bearer " + key, "Content-Type": "application/json" },
    body: JSON.stringify({ query, variables }),
  });
  const j = await r.json();
  if (j.errors) throw new Error(j.errors.map((e) => e.message).join("; "));
  return j.data;
}

// Default pod image (RunPod's nix-based pytorch line — has JupyterLab + SSH auto-setup from PUBLIC_KEY).
const RUNPOD_IMAGE = process.env.RUNPOD_IMAGE || "runpod/pytorch:1.0.7-dev-nix-cu1290-torch280-ubuntu2204";
const RUNPOD_SSH_DIR = join(ROOT, ".runpod");
const RUNPOD_SSH_KEY = join(RUNPOD_SSH_DIR, "id_pod");
let _sshPubCache = null;
// The server's own keypair for driving pods: private key never leaves this machine, the public key
// rides to the pod as PUBLIC_KEY (RunPod images append it to authorized_keys). Generated once.
async function ensureSshKey() {
  if (_sshPubCache) return _sshPubCache;
  await mkdir(RUNPOD_SSH_DIR, { recursive: true });
  if (!existsSync(RUNPOD_SSH_KEY)) {
    await new Promise((resolve, reject) => {
      const p = spawn("ssh-keygen", ["-t", "ed25519", "-N", "", "-f", RUNPOD_SSH_KEY, "-C", "ares-runpod"], { stdio: "ignore", windowsHide: true });
      p.on("close", (c) => (c === 0 ? resolve() : reject(new Error("ssh-keygen exited " + c))));
      p.on("error", reject);
    });
  }
  _sshPubCache = (await readFile(RUNPOD_SSH_KEY + ".pub", "utf8")).trim();
  return _sshPubCache;
}

// Public SSH endpoint (ip+port) for a pod, once RunPod has mapped its 22/tcp port. Null while booting.
async function runpodPodSsh(key, id) {
  const d = await runpodGraphQL(key, "query($id:String!){pod(input:{podId:$id}){desiredStatus runtime{ports{ip isIpPublic privatePort publicPort type}}}}", { id });
  const pod = d.pod;
  if (!pod) return { ready: false, status: "GONE" };
  const ports = (pod.runtime && pod.runtime.ports) || [];
  const ssh = ports.find((p) => p.privatePort === 22 && p.isIpPublic && p.type === "tcp");
  return ssh ? { ready: true, ip: ssh.ip, port: ssh.publicPort, status: pod.desiredStatus } : { ready: false, status: pod.desiredStatus };
}

// Spawn `ssh` (unsandboxed — serve.mjs is a normal process) to run a command on the pod, emitting
// output line-by-line. Returns the child so callers can kill it on client disconnect.
function sshRun(ip, port, command, onLine, onDone) {
  // accept-new pins each pod's host key on first contact and refuses a changed one afterwards
  // (pods are ephemeral, so a fresh IP is a fresh key — a CHANGED key on a known IP is the alarm).
  const args = ["-i", RUNPOD_SSH_KEY, "-p", String(port), "-o", "StrictHostKeyChecking=accept-new",
    "-o", `UserKnownHostsFile=${join(RUNPOD_SSH_DIR, "known_hosts")}`, "-o", "ConnectTimeout=12", "-o", "ServerAliveInterval=15",
    "root@" + ip, command];
  const p = spawn("ssh", args, { windowsHide: true });
  let buf = "";
  const feed = (chunk) => { buf += chunk; let i; while ((i = buf.indexOf("\n")) >= 0) { onLine(buf.slice(0, i)); buf = buf.slice(i + 1); } };
  p.stdout.on("data", (d) => feed(d.toString()));
  p.stderr.on("data", (d) => feed(d.toString()));
  p.on("close", (code) => { if (buf) onLine(buf); onDone(code); });
  p.on("error", (e) => { onLine("[ssh error] " + e.message); onDone(-1); });
  return p;
}

/**
 * POST JSON to the local SD-Forge REST API. node:http (not fetch) keeps this zero-dep AND
 * un-timeouted — Forge's sync endpoints only answer when the frame is done, which for the
 * img2img tier can be minutes. onReq exposes the request so /enhance can abort on client close.
 */
function forgePost(base, apiPath, payload, onReq) {
  return new Promise((resolve, reject) => {
    let u;
    try { u = new URL(apiPath, base); } catch { reject(new Error(`bad forge URL: ${base}`)); return; }
    const body = Buffer.from(JSON.stringify(payload));
    const rq = httpRequest({
      hostname: u.hostname,
      port: u.port || 80,
      path: u.pathname,
      method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": body.length },
    }, (rs) => {
      const chunks = [];
      rs.on("data", (c) => chunks.push(c));
      rs.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        if (rs.statusCode !== 200) { reject(new Error(`forge ${u.pathname} → HTTP ${rs.statusCode}: ${text.slice(0, 400)}`)); return; }
        try { resolve(JSON.parse(text)); } catch { reject(new Error(`forge ${u.pathname} returned non-JSON`)); }
      });
    });
    rq.on("error", reject);
    if (onReq) onReq(rq);
    rq.end(body);
  });
}

// ---- native folder/file picker (Windows) --------------------------------------------------
// Opens a real OS dialog via a hidden STA PowerShell so the user never types a filesystem path.
// -STA is load-bearing: a WinForms dialog on an MTA thread hangs forever, so the timeout is the
// only safeguard against a leaked process. Returns the absolute path, or null on cancel.
function pickNative(type, { timeoutMs = 180000, filter, initialDirectory } = {}) {
  return new Promise((resolve, reject) => {
    const args = ["-NoProfile", "-STA", "-WindowStyle", "Hidden",
      "-File", join(ROOT, "tools", "pick.ps1"), "-Type", type === "file" ? "file" : "folder"];
    if (filter) args.push("-Filter", filter);
    if (initialDirectory) args.push("-InitialDirectory", initialDirectory);
    const child = spawn("powershell.exe", args, { windowsHide: true });
    let out = "", settled = false;
    const timer = setTimeout(() => { if (settled) return; settled = true; child.kill(); reject(new Error(`picker timed out after ${timeoutMs}ms`)); }, timeoutMs);
    child.stdout.on("data", (d) => (out += d));
    child.on("error", (e) => { if (settled) return; settled = true; clearTimeout(timer); reject(e); });
    child.on("close", () => { if (settled) return; settled = true; clearTimeout(timer); resolve(out.trim() || null); });
  });
}

// ---- unified volcap history + last-used folders (apps/demo/history.json) -------------------
// One store shared by the Inspect and Convert tabs: every folder analysed, file inspected,
// encode produced, and enhance run, newest first, deduped by kind+path+name, capped at 200.
// `lastDirs` remembers the last picked folder per purpose key so pickers reopen there.
const HISTORY_FILE = join(ROOT, "apps", "demo", "history.json");
async function readHistoryStore() {
  try {
    const h = JSON.parse(await readFile(HISTORY_FILE, "utf8"));
    if (h && Array.isArray(h.items) && typeof h.lastDirs === "object") return h;
  } catch { /* absent or malformed → fresh store */ }
  return { lastDirs: {}, items: [] };
}
let historyWriteChain = Promise.resolve(); // serialize writers (concurrent SSE routes record too)
function mutateHistory(fn) {
  historyWriteChain = historyWriteChain.then(async () => {
    const h = await readHistoryStore();
    fn(h);
    await writeFile(HISTORY_FILE, JSON.stringify(h, null, 1));
  }).catch(() => { /* history is best-effort — never fail the primary operation */ });
  return historyWriteChain;
}
function historyAdd(entry) {
  return mutateHistory((h) => {
    const key = (e) => `${e.kind}|${e.path || ""}|${e.name || ""}`;
    h.items = h.items.filter((e) => key(e) !== key(entry));
    h.items.unshift({ ...entry, at: new Date().toISOString() });
    if (h.items.length > 200) h.items.length = 200;
  });
}
function rememberDir(key, dir) {
  if (!key || !dir) return;
  mutateHistory((h) => { h.lastDirs[key] = dir; });
}

// ---- folder analysis (server-side parity with the browser drag-drop preview) ---------------
const OBJ_RE = /\.obj$/i, PLY_RE = /\.ply$/i, ATLAS_RE = /^atlas-.*\.png$/i, PNG_RE = /\.png$/i;
// Splat-profile frame files (spec §6.8): SPZ, .splat, SOG bundles, glTF/GLB with KHR_gaussian_splatting.
const SPLAT_RE = /\.(spz|splat|sog|glb|gltf)$/i;
const pngDimsBuf = (b) => (b && b.length > 24 && b.readUInt32BE(0) === 0x89504e47) ? [b.readUInt32BE(16), b.readUInt32BE(20)] : null;
const isFrameFile = (f) => OBJ_RE.test(f) || PLY_RE.test(f) || SPLAT_RE.test(f);
// Point-at-parent convenience: if `dir` has no meshes but a single subfolder does, use that.
async function resolveFramesDir(dir) {
  try {
    const names = await readdir(dir);
    if (names.some(isFrameFile) || names.includes("meta.json")) return dir;
    const hits = [];
    for (const n of names) {
      const p = join(dir, n);
      try { if (statSync(p).isDirectory()) { const inner = await readdir(p); if (inner.some(isFrameFile)) hits.push(p); } } catch { /* ignore */ }
    }
    return hits.length === 1 ? hits[0] : dir;
  } catch { return dir; }
}
/** 3DGS splat PLY? Only the header is read (f_dc_0 + scale_0 + rot_0 + opacity on the vertex element). */
async function isSplatPlyFile(path) {
  try {
    const fh = await open(path, "r");
    try {
      const buf = Buffer.alloc(16384);
      const { bytesRead } = await fh.read(buf, 0, buf.length, 0);
      const head = buf.subarray(0, bytesRead).toString("latin1");
      const hdr = head.slice(0, head.indexOf("end_header") >= 0 ? head.indexOf("end_header") : head.length);
      return ["f_dc_0", "scale_0", "rot_0", "opacity"].every((k) => hdr.includes("property float " + k) || hdr.includes(" " + k + "\n") || hdr.includes(" " + k + "\r"));
    } finally { await fh.close(); }
  } catch { return false; }
}
/** Splat count of one frame file, cheaply (headers only). 0 when unknown. */
async function splatCountOf(path, kind) {
  try {
    if (kind === "SPZ") {
      const fh = await open(path, "r");
      let head;
      try { const b = Buffer.alloc(65536); const { bytesRead } = await fh.read(b, 0, b.length, 0); head = b.subarray(0, bytesRead); } finally { await fh.close(); }
      if (head[0] === 0x1f && head[1] === 0x8b) head = zlib.gunzipSync(head, { finishFlush: zlib.constants.Z_SYNC_FLUSH });
      return head.readUInt32LE(0) === 0x5053474e ? head.readUInt32LE(8) : 0;
    }
    if (kind === "3DGS PLY") {
      const fh = await open(path, "r");
      try { const b = Buffer.alloc(16384); const { bytesRead } = await fh.read(b, 0, b.length, 0); const m = /element vertex (\d+)/.exec(b.subarray(0, bytesRead).toString("latin1")); return m ? Number(m[1]) : 0; } finally { await fh.close(); }
    }
    if (kind === ".splat") return Math.floor((await stat(path)).size / 32);
    if (kind === "SOG") {
      const metaPath = (await stat(path)).isDirectory() ? join(path, "meta.json") : null;
      if (metaPath) return Number(JSON.parse(await readFile(metaPath, "utf8")).count) || 0;
    }
  } catch { /* unknown */ }
  return 0;
}
async function analyseDir(dir0) {
  const dir = await resolveFramesDir(dir0);
  const names = (await readdir(dir)).sort();
  const objs = names.filter((f) => OBJ_RE.test(f)), plys = names.filter((f) => PLY_RE.test(f));
  const splatFiles = names.filter((f) => SPLAT_RE.test(f));
  const atlases = names.filter((f) => ATLAS_RE.test(f)), pngs = names.filter((f) => PNG_RE.test(f));
  let meshes = objs.length ? objs : plys;
  let kind = objs.length ? "OBJ" : plys.length ? "PLY" : "·";
  // Splat sequences (spec §6.8): a 3DGS PLY folder, or one SPZ/.splat/SOG/glTF file per frame, or
  // a single SOG directory (meta.json + webp images).
  let splat = false, splatCount = 0;
  if (!objs.length) {
    if (plys.length && await isSplatPlyFile(join(dir, plys[0]))) { splat = true; kind = "3DGS PLY"; }
    else if (splatFiles.length) {
      const ext = extname(splatFiles[0]).toLowerCase();
      kind = ext === ".spz" ? "SPZ" : ext === ".splat" ? ".splat" : ext === ".sog" ? "SOG" : "glTF splat";
      meshes = splatFiles; splat = true;
    } else if (names.includes("meta.json")) { kind = "SOG"; meshes = ["."]; splat = true; }
    if (splat) splatCount = await splatCountOf(meshes[0] === "." ? dir : join(dir, meshes[0]), kind);
  }
  let rawBytes = 0;
  for (const f of names) { try { rawBytes += (await stat(join(dir, f))).size; } catch { /* ignore */ } }
  let verts = 0, atlasDims = null;
  if (objs.length) { try { verts = ((await readFile(join(dir, objs[0]), "utf8")).match(/^v /gm) || []).length; } catch { /* ignore */ } }
  if (atlases.length) { try { atlasDims = pngDimsBuf(await readFile(join(dir, atlases[0]))); } catch { /* ignore */ } }
  return { dir, meshes: meshes.length, kind, atlases: atlases.length, pngs: pngs.length, atlasDims, verts, rawBytes, splat, splatCount };
}

// ---- standalone Real-ESRGAN (ncnn-vulkan): the no-server, low-VRAM upscale tier -----------
const REALESRGAN_DIR = join(ROOT, "tools", "bin", "realesrgan-ncnn-vulkan");
const REALESRGAN_EXE = join(REALESRGAN_DIR, "realesrgan-ncnn-vulkan.exe");
const FFMPEG = process.env.FFMPEG || "ffmpeg";

// ---- ffmpeg / ffprobe: ONE resolver --------------------------------------------------------
// installer.mjs findFfmpeg() owns the search order (FFMPEG override, tools/bin/ffmpeg, the legacy
// C:\FFmpeg\bin, PATH) and rejects a build without libvpx-vp9 / libsvtav1 / libopus. Nothing in
// this file spawns a bare "ffmpeg": the resolved absolute paths go to every child through toolEnv()
// (the encoder CLI reads FFMPEG / FFPROBE) and to the Python service in the /depth/run body.
// A miss is never cached, so the path appears the moment ensureComponents(["ffmpeg"]) installs it.
let ffToolsCache = null;
function ffTools({ fresh = false } = {}) {
  if (fresh || !ffToolsCache) ffToolsCache = findFfmpeg(ROOT);
  return ffToolsCache;
}
function toolEnv(extra = {}) {
  const ff = ffTools();
  return {
    ...process.env,
    ...(ff ? { FFMPEG: ff.ffmpeg, FFMPEG_PATH: ff.ffmpeg, FFPROBE: ff.ffprobe, FFPROBE_PATH: ff.ffprobe } : {}),
    ...extra,
  };
}

// ---- component ensure: every work route calls this before it starts ------------------------
// The rule it implements: a job never stops to tell the person that something is absent. It
// resolves the component ids the job needs through the catalog's `requires` graph, installs what
// is absent inside the route's own SSE stream ("[setup] …" log lines plus `setup` events), and
// returns so the job carries on. The one thing it cannot supply is a credential: a gated
// Hugging Face repository comes back as { gated, needsToken, url } and the client raises its
// access prompt, then re-issues the same request.
//   ensureComponents(ids: string[], send?: (event, data) => void)
//     -> { ok: true, installed: string[] }
//      | { ok: false, error, id?, label?, gated?, needsToken?, url? }
let gpuCache = null, gpuCacheAt = 0;
async function gpuCached() {
  if (!gpuCache || Date.now() - gpuCacheAt > 60000) { gpuCache = await gpuProbe(); gpuCacheAt = Date.now(); }
  return gpuCache;
}
const ensureInFlight = new Map();   // component id -> running install, shared by concurrent routes
async function ensureComponents(ids, send) {
  const say = (t) => { if (send) send("log", "[setup] " + t); };
  const want = [...new Set(ids.filter(Boolean))];
  let items = catalog(ROOT, await gpuCached());
  const unknown = want.filter((id) => !items.some((i) => i.id === id));
  if (unknown.length) return { ok: false, error: `unknown component: ${unknown.join(", ")}` };
  const installable = (it) => it.install && it.install.kind !== "route" && !it.statusOnly;
  const todo = resolve(items, want).filter((it) => installable(it) && !it.present);
  if (!todo.length) return { ok: true, installed: [] };
  say(`absent: ${todo.map((t) => t.label).join(", ")}`);
  if (send) send("setup", { state: "plan", todo: todo.map((t) => ({ id: t.id, label: t.label, sizeMB: t.sizeMB || 0 })) });
  for (const [index, it] of todo.entries()) {
    if (send) send("setup", { state: "start", id: it.id, label: it.label, index, total: todo.length, sizeMB: it.sizeMB || 0 });
    say(`${it.label}${it.sizeMB ? ` (${it.sizeMB} MB)` : ""}`);
    let job = ensureInFlight.get(it.id);
    if (job) say(`${it.label}: install already running, waiting for it`);
    else {
      job = installOne(ROOT, it, (l) => say("  " + l)).catch((e) => ({ ok: false, error: String((e && e.message) || e) }))
        .finally(() => ensureInFlight.delete(it.id));
      ensureInFlight.set(it.id, job);
    }
    const r = await job;
    if (it.id === "ffmpeg") ffTools({ fresh: true });
    if (!r.ok) {
      say(`${it.label}: ${r.error}`);
      return { ok: false, id: it.id, label: it.label, error: r.error, gated: !!r.gated, needsToken: !!r.needsToken, url: it.gated?.url || null };
    }
    if (send) send("setup", { state: "done", id: it.id, label: it.label, index: index + 1, total: todo.length });
    say(`${it.label}: ready`);
  }
  // Judge by detection, never by the installer's own word: the job is about to depend on it.
  items = catalog(ROOT, await gpuCached());
  const still = todo.filter((t) => !items.find((i) => i.id === t.id)?.present);
  if (still.length) return { ok: false, id: still[0].id, label: still[0].label, error: `${still[0].label}: not detected after installation` };
  return { ok: true, installed: todo.map((t) => t.id) };
}
/** The `error` event a route sends when ensureComponents could not finish. */
const ensureErrorPayload = (r) => ({
  message: r.error, stage: "setup", component: r.id || null, label: r.label || null,
  gated: !!r.gated, needsToken: !!r.needsToken, url: r.url || null,
});
/** ensureComponents for an SSE route: on failure the error event is sent and the stream closed.
 *  Resolves true when the job may proceed. */
async function ensureOrEnd(ids, send, res) {
  const r = await ensureComponents(ids, send);
  if (r.ok) return true;
  send("error", ensureErrorPayload(r));
  res.end();
  return false;
}

// Run a child to completion; resolve on exit 0. onChild exposes it so /enhance can cancel it.
function runProc(exe, args, opts, onChild) {
  return new Promise((resolve, reject) => {
    const c = spawn(exe, args, { windowsHide: true, ...opts });
    if (onChild) onChild(c);
    let err = "";
    if (c.stderr) c.stderr.on("data", (d) => { err = (err + d).slice(-4000); });
    c.on("error", reject);
    c.on("close", (code) => code === 0 ? resolve() : reject(new Error(`${basename(exe)} exit ${code}: ${err.trim().split(/\r?\n/).slice(-2).join(" ")}`)));
  });
}

/** Like runProc, but relays whole stdout/stderr LINES to `onLine` and resolves with the exit
 *  code instead of rejecting. Used for workers whose progress is their stdout. */
function runProcLines(exe, args, opts, onChild, onLine) {
  return new Promise((resolve) => {
    const c = spawn(exe, args, { windowsHide: true, ...opts });
    if (onChild) onChild(c);
    let buf = "";
    const feed = (d) => {
      buf += d;
      const lines = buf.split(/\r?\n/);
      buf = lines.pop() ?? "";                       // keep the partial line for the next chunk
      for (const l of lines) if (l.trim()) onLine(l.trim().slice(0, 400));
    };
    c.stdout?.on("data", feed);
    c.stderr?.on("data", feed);
    c.on("error", (e) => { onLine("ERROR " + e.message); resolve(1); });
    c.on("close", (code) => { if (buf.trim()) onLine(buf.trim().slice(0, 400)); resolve(code ?? 1); });
  });
}

// ---- Forge (generative "hero" tier only) — auto-launched headless, no terminal ------------
// Two layouts run here. A PACKAGED install (the one-click archive: system/python + webui/) is
// used as it is when FORGE_ROOT or ~/webui_forge holds one. Otherwise the catalog's `forge`
// component is a git clone under tools/ext/webui_forge, and forgeEnsure() builds that clone its
// own virtual environment on a private CPython 3.10 (Forge's pinned wheels stop at 3.10), lets
// launch.py install its packages on the first start, and marks the venv ready afterwards.
const FORGE_CLONE = join(ROOT, "tools", "ext", "webui_forge");
const FORGE_URL = process.env.FORGE_URL || "http://127.0.0.1:7861";
function forgeLayout() {
  const packaged = [process.env.FORGE_ROOT, join(homedir(), "webui_forge")].filter(Boolean)
    .find((d) => existsSync(join(d, "system", "python", "python.exe")));
  if (packaged) {
    return {
      kind: "packaged", root: packaged, cwd: join(packaged, "webui"), py: join(packaged, "system", "python", "python.exe"), ready: true,
      path: [join(packaged, "system", "git", "bin"), join(packaged, "system", "python"), join(packaged, "system", "python", "Scripts")],
      ckptDir: process.env.FORGE_CKPT_DIR || join(packaged, "webui", "models", "Stable-diffusion"),
    };
  }
  const root = process.env.FORGE_ROOT && existsSync(join(process.env.FORGE_ROOT, "launch.py")) ? process.env.FORGE_ROOT : FORGE_CLONE;
  const git = findGit(ROOT);
  return {
    kind: "clone", root, cwd: root, py: join(root, "venv", "Scripts", "python.exe"),
    ready: existsSync(join(root, "venv", ".ares-ready")),
    path: [join(root, "venv", "Scripts"), ...(git ? [dirname(git)] : [])],
    ckptDir: process.env.FORGE_CKPT_DIR || join(root, "models", "Stable-diffusion"),
  };
}
let forgeChild = null;
function forgeHealthy(timeoutMs = 1500) {
  return new Promise((resolve) => {
    let u; try { u = new URL("/sdapi/v1/sd-models", FORGE_URL); } catch { resolve(false); return; }
    const rq = httpRequest({ hostname: u.hostname, port: u.port || 80, path: u.pathname, method: "GET", timeout: timeoutMs }, (rs) => { rs.resume(); resolve(rs.statusCode === 200); });
    rq.on("error", () => resolve(false));
    rq.on("timeout", () => { rq.destroy(); resolve(false); });
    rq.end();
  });
}
// Ensure Forge's API is up: health-check, else install what is absent, spawn it hidden and poll.
// `send` streams SSE status. Resolves true, or { error, ...ensure failure fields }.
async function forgeEnsure(send) {
  const log = (t) => { if (send) send("log", t); };
  if (await forgeHealthy()) return true;
  let lay = forgeLayout();
  if (lay.kind === "clone" && !existsSync(join(lay.root, "launch.py"))) {
    const r = await ensureComponents(["forge"], send);
    if (!r.ok) return r;
    lay = forgeLayout();
  }
  if (lay.kind === "clone" && !existsSync(lay.py)) {
    const py = await ensurePrivatePython(ROOT, (l) => log("[setup]   " + l), FORGE_PYTHON);
    if (py.error) return { ok: false, error: py.error };
    log("[setup] Forge: creating its virtual environment");
    const code = await runProcLines(py.exe, ["-m", "venv", join(lay.root, "venv")], { cwd: lay.root }, null, (l) => log("[setup]   " + l));
    if (code !== 0 || !existsSync(lay.py)) return { ok: false, error: `Forge venv creation failed (exit ${code})` };
  }
  const firstRun = !lay.ready;
  if (!forgeChild || forgeChild.exitCode !== null) {
    log(firstRun ? "Forge: first start, installing its packages (several GB, tens of minutes)…" : "Forge: starting (headless API, cold start 30 to 60 s)…");
    const args = ["launch.py", "--nowebui", ...(firstRun ? [] : ["--skip-install"])];
    if (existsSync(lay.ckptDir)) args.push("--ckpt-dir", lay.ckptDir);
    const env = { ...process.env, PATH: lay.path.join(";") + ";" + process.env.PATH };
    // Forge pins a cu121 torch, which has no kernels for Blackwell (sm_120): give it a build that does.
    const gpu = await gpuCached();
    if (firstRun && gpu.cc >= 12) env.TORCH_COMMAND = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128";
    if (firstRun) {
      // Piped so the package install is visible in the job log; the reader stays attached for the
      // life of the child, whether or not the browser is still listening.
      forgeChild = spawn(lay.py, args, { cwd: lay.cwd, env, windowsHide: true });
      const relay = (d) => String(d).split(/\r?\n/).forEach((l) => l.trim() && log("[forge] " + l.trim().slice(0, 300)));
      forgeChild.stdout.on("data", relay);
      forgeChild.stderr.on("data", relay);
      forgeChild.on("error", () => { /* surfaced by the health deadline below */ });
    } else {
      forgeChild = spawn(lay.py, args, { cwd: lay.cwd, env, detached: true, stdio: "ignore", windowsHide: true });
      forgeChild.on("error", () => { /* surfaced by the health deadline below */ });
      forgeChild.unref();
    }
  } else log("Forge: start already in progress, waiting");
  const limitS = firstRun ? 3600 : 180;
  const deadline = Date.now() + limitS * 1000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    if (await forgeHealthy()) {
      if (firstRun && lay.kind === "clone") { try { await writeFile(join(lay.root, "venv", ".ares-ready"), new Date().toISOString()); } catch { /* next start repeats the install check */ } }
      log("Forge API ready");
      return true;
    }
    if (forgeChild && forgeChild.exitCode !== null) return { ok: false, error: `Forge exited with code ${forgeChild.exitCode} before its API answered` };
  }
  return { ok: false, error: `Forge API not ready after ${limitS} s` };
}

// ---- SAM segmentation service — auto-launched hidden from the app, no terminal ------------
// Mirrors forgeEnsure: health-check, else spawn the hidden launcher and poll. The service
// binds its port immediately and reports {loading:true} during the ~40 s ViT-H load, so
// concurrent ensures/launchers converge on one instance instead of racing the bind.
const SAM_URL = process.env.SAM_URL || "http://127.0.0.1:7263";
const SAM_PS1 = join(ROOT, "tools", "sam-service", "run-sam-service.ps1");
let samChild = null;
function samHealth(timeoutMs = 1500) {
  return new Promise((resolve) => {
    let u; try { u = new URL("/health", SAM_URL); } catch { resolve(null); return; }
    const rq = httpRequest({ hostname: u.hostname, port: u.port || 80, path: u.pathname, method: "GET", timeout: timeoutMs }, (rs) => {
      let body = "";
      rs.on("data", (d) => { body += d; });
      rs.on("end", () => { try { resolve(JSON.parse(body)); } catch { resolve(null); } });
    });
    rq.on("error", () => resolve(null));
    rq.on("timeout", () => { rq.destroy(); resolve(null); });
    rq.end();
  });
}
/** The last lines of the service transcript, so a failed start is explained in the job log. */
async function samLogTail(n = 12) {
  try {
    const txt = await readFile(join(ROOT, "tools", "sam-service", "sam-service.log"), "utf8");
    return txt.replace(/\0/g, "").split(/\r?\n/).map((l) => l.trim()).filter(Boolean).slice(-n);
  } catch { return []; }
}
/** Stop the service that owns SAM_URL's port. Used when weights arrive after a start that latched
 *  a load failure: the service loads its model once, so new weights need a new process. Only a
 *  python process is ever stopped. */
async function samStop(send) {
  let port = 7263; try { port = Number(new URL(SAM_URL).port) || 80; } catch { /* default */ }
  const ps = `$c = Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; ` +
    `if ($c) { $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue; if ($p -and $p.ProcessName -like 'python*') { Stop-Process -Id $p.Id -Force } }`;
  await runProcLines("powershell.exe", ["-NoProfile", "-Command", ps], {}, null, () => {});
  samChild = null;
  for (let i = 0; i < 10 && (await samHealth(800)); i++) await new Promise((r) => setTimeout(r, 500));
  if (send) send("log", "SAM service stopped for a model reload");
}
/**
 * samEnsure(send, { purpose, components })
 *   purpose "sam"   (default) a segmentation backend must load: SAM 3, else ViT-H
 *   purpose "track" the video tracker, which is SAM 3 only
 *   purpose "depth" the depth routes only: a latched SAM load failure does not block them
 *   components      extra catalog ids the job needs (a depth model, SAM 3 for a subject mask)
 * Installs whatever is absent first (Python environment, weights), then health-checks, spawns the
 * hidden launcher and polls. Resolves true, or { ok:false, error, gated?, needsToken?, url? }.
 */
let samRestarted = false;   // one automatic restart per latched failure, never a loop
async function samEnsure(send, { purpose = "sam", components = [] } = {}) {
  const log = (t) => { if (send) send("log", t); };
  const need = ["python-env", ...components];
  if (purpose === "track") need.push("sam3");
  if (purpose === "sam") {
    const items = catalog(ROOT, await gpuCached());
    const have = (id) => !!items.find((i) => i.id === id)?.present;
    // Any loadable backend is enough to start. With none on disk: SAM 3 when a Hugging Face token
    // is stored (it is gated), otherwise the ungated ViT-H so the first click works without one.
    if (!have("sam3") && !have("vit-h")) need.push(hfTokenPresent() ? "sam3" : "vit-h");
  }
  const ens = await ensureComponents(need, send);
  if (!ens.ok) return ens;

  let h = await samHealth();
  // A latched load failure with weights now on disk (installed just now, or from Settings while
  // the service was up) is stale: the service loads once, so it is restarted to pick them up.
  if (h && h.error && purpose !== "depth") {
    const items = catalog(ROOT, await gpuCached());
    if (["sam3", "vit-h"].some((id) => items.find((i) => i.id === id)?.present) && !samRestarted) {
      samRestarted = true;
      await samStop(send);
      h = null;
    }
  }
  const usable = (x) => !!x && (x.ok || (purpose === "depth" && !x.loading));
  if (usable(h)) return true;
  const failed = async (x) => ({ ok: false, error: `SAM model load failed: ${x.error}`, log: await samLogTail() });
  if (h && h.error) return failed(h);
  if (!h) {
    if (!existsSync(SAM_PS1)) return { ok: false, error: `SAM launcher absent from this checkout: ${SAM_PS1}` };
    if (!samChild || samChild.exitCode !== null) {
      log("starting SAM service (model load ~40 s on first start)…");
      // NOT detached. Measured 2026-09-18 (Windows PowerShell 5.1.26100): powershell.exe spawned
      // with `detached: true` exits 0 within a second without running the script at all, whatever
      // the window style or stdio (five variants tried, tools/sam-service/sam-service.log never
      // even gained a new transcript), so every auto-start silently timed out after 180 s. Without
      // `detached` the same command binds the port in ~3 s and reports ready in ~13 s. Windows does
      // not kill a child when its parent exits, so the service still outlives this process; it now
      // shares this process's (hidden) console instead of a new process group.
      // toolEnv(): the service inherits the resolved FFMPEG / FFPROBE, so depth.py finds them
      // even when a request omits the `ffmpeg` field.
      samChild = spawn("powershell.exe",
        ["-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", SAM_PS1],
        { stdio: "ignore", windowsHide: true, env: toolEnv() });
      samChild.unref();
    } else {
      log("waiting for the SAM service to finish starting…");
    }
  } else {
    log("SAM service is loading the model…");
  }
  const deadline = Date.now() + 180000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    h = await samHealth();
    if (h && h.ok) { samRestarted = false; log(`SAM ready (${h.model} on ${h.device})`); return true; }
    if (usable(h)) { log("SAM service up (depth routes)"); return true; }
    if (h && h.error) return failed(h);
  }
  return { ok: false, error: "SAM service not ready after 180 s", log: await samLogTail() };
}
/** Send a samEnsure / forgeEnsure failure down an SSE stream: the transcript tail as log lines,
 *  then the error event (gated fields included, so the client can raise its access prompt). */
function sendEnsureFailure(send, r) {
  for (const l of r.log || []) send("log", "[service] " + l);
  send("error", ensureErrorPayload(r));
}

// ---- 4DViews .4ds codec (Task I: Convert tab wiring for tools/4ds/decode_4ds.py) -----------
// Two pieces must both be present: the SAM-service Python env (has numpy+Pillow, reused rather
// than a dedicated venv) and the proprietary BridgeCodec4DS.dll (used under your own 4DViews SDK
// license — never committed; copy it into tools/4ds/bin/ or point FOURDS_DLL at it).
const FOURDS_PY = join(ROOT, "tools", "sam-service", "env", "Scripts", "python.exe");
const FOURDS_SCRIPT = join(ROOT, "tools", "4ds", "decode_4ds.py");
const FOURDS_DLL_LOCAL = join(ROOT, "tools", "4ds", "bin", "BridgeCodec4DS.dll");
const FOURDS_DLL_FALLBACK = process.env.FOURDS_DLL || "";
function fourdsDllPath() {
  if (existsSync(FOURDS_DLL_LOCAL)) return FOURDS_DLL_LOCAL;
  if (existsSync(FOURDS_DLL_FALLBACK)) return FOURDS_DLL_FALLBACK;
  return null;
}
/** What a .4ds job still lacks, in the two forms the client acts on by itself: `needs` =
 *  component ids it installs through /install before retrying, `needsFile` = a file only the
 *  person has (the licensed codec DLL), collected with the native file dialog and copied into
 *  place by POST /setup/4ds-codec. FOURDS_DLL still points at one in place. */
function fourdsNeeds() {
  return {
    needs: existsSync(FOURDS_PY) ? [] : ["python-env"],
    needsFile: fourdsDllPath() ? null : {
      id: "4ds-codec", label: "BridgeCodec4DS.dll (4DViews SDK)", route: "/setup/4ds-codec",
      filter: "4DViews codec (BridgeCodec4DS.dll)|BridgeCodec4DS.dll|DLL files (*.dll)|*.dll",
    },
  };
}
// Run `decode_4ds.py --info` (no decode) and parse its JSON. Used by /probe-4ds and to pick the
// bake fps in /convert-4ds.
function run4dsInfo(inputPath) {
  return new Promise((resolve, reject) => {
    const child = spawn(FOURDS_PY, [FOURDS_SCRIPT, inputPath, "--info"], { windowsHide: true });
    let out = "", err = "";
    child.stdout.on("data", (d) => { out += d; });
    child.stderr.on("data", (d) => { err = (err + d).slice(-4000); });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) { reject(new Error(err.trim().split(/\r?\n/).slice(-4).join(" ") || `decode_4ds.py --info exited ${code}`)); return; }
      try { resolve(JSON.parse(out)); } catch { reject(new Error("decode_4ds.py --info returned non-JSON: " + out.slice(0, 300))); }
    });
  });
}

/**
 * Stream one file, honouring a single `Range: bytes=` request (206, or 416 when unsatisfiable) and
 * HEAD. Throws when the file cannot be stat'ed, before any header is written.
 */
async function sendFileRange(req, res, file, type) {
  const st = await stat(file);
  if (!st.isFile()) throw new Error(`not a file: ${file}`);
  const size = st.size;
  const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || "");
  let start = 0, end = size - 1;
  if (m && (m[1] || m[2])) {
    if (m[1]) { start = Number(m[1]); end = m[2] ? Math.min(Number(m[2]), size - 1) : size - 1; }
    else { start = Math.max(0, size - Number(m[2])); }
    if (!(start <= end && start < size)) { res.writeHead(416, { ...HEADERS, "Content-Range": `bytes */${size}` }); res.end(); return; }
  }
  const partial = !!(m && (m[1] || m[2]));
  res.writeHead(partial ? 206 : 200, {
    ...HEADERS, "Content-Type": type, "Accept-Ranges": "bytes", "Content-Length": size ? end - start + 1 : 0,
    ...(partial ? { "Content-Range": `bytes ${start}-${end}/${size}` } : {}),
  });
  if (req.method === "HEAD" || !size) { res.end(); return; }
  const rs = createReadStream(file, { start, end });
  rs.on("error", () => { try { res.destroy(); } catch { /* gone */ } });
  res.on("close", () => rs.destroy());
  rs.pipe(res);
}

// Routes that do work, spend money, open dialogs or write files. Browsers stamp Sec-Fetch-Site on
// every request, so a page on any other origin (or a same-site page on another port) is refused
// here even though the server only listens on loopback: EventSource/GET side effects were the
// audit's CSRF finding. Non-browser callers (curl, scripts) send no such header and pass.
const GUARDED = /^\/(encode|convert-4ds|enhance|sam\/start|forge\/start|setup\/|runpod\/(launch|stop|action|logs)|pick|log|edits\/|showcase|deps\/)/;
const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]", "::1"]);
function sameOrigin(req) {
  const sfs = req.headers["sec-fetch-site"];
  if (sfs && sfs !== "same-origin" && sfs !== "none") return false;
  const origin = req.headers.origin || (req.headers.referer ? (() => { try { return new URL(req.headers.referer).origin; } catch { return "bad"; } })() : null);
  if (origin) { try { return LOOPBACK.has(new URL(origin).hostname); } catch { return false; } }
  return true;
}

async function handle(req, res) {
  const url = new URL(req.url, "http://localhost");
  let path = decodeURIComponent(url.pathname);

  // Host pinning (DNS rebinding): the only names this loopback server answers to are its own.
  const hostName = String(req.headers.host || "").replace(/:\d+$/, "").toLowerCase();
  if (hostName && !LOOPBACK.has(hostName)) { res.writeHead(421, { "Content-Type": "text/plain" }); res.end("wrong host"); return; }
  if (GUARDED.test(path) && !sameOrigin(req)) { res.writeHead(403, { ...HEADERS, "Content-Type": "text/plain" }); res.end("cross-origin request refused"); return; }

  if (path === "/__ares") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    res.end(JSON.stringify({ server: "ares-dev", root: ROOT, pid: process.pid, encode: true }));
    return;
  }

  // RunPod account snapshot for the Compute panel: balance, live spend, and current pods.
  if (path === "/runpod/status") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    const key = await readRunpodKey();
    if (!key) { res.end(JSON.stringify({ ok: false, error: "no RunPod key (set RUNPOD_API_KEY or save it to " + RUNPOD_KEY_FILE + ")" })); return; }
    try {
      const d = await runpodGraphQL(key, "query{myself{clientBalance currentSpendPerHr pods{id name desiredStatus costPerHr gpuCount machine{gpuDisplayName} runtime{uptimeInSeconds}}}}");
      const me = d.myself;
      res.end(JSON.stringify({
        ok: true,
        balance: me.clientBalance,
        spendPerHr: me.currentSpendPerHr,
        pods: (me.pods || []).map((p) => ({
          id: p.id, name: p.name, status: p.desiredStatus, costPerHr: p.costPerHr,
          gpuCount: p.gpuCount, gpu: p.machine && p.machine.gpuDisplayName,
          uptimeS: p.runtime && p.runtime.uptimeInSeconds,
        })),
      }));
    } catch (e) { res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); }
    return;
  }

  // Launch a pod (the ONLY spend — the UI confirms before calling this). Community RTX 3090 by default.
  if (path === "/runpod/launch" && req.method === "POST") {
    if (!/^application\/json/i.test(String(req.headers["content-type"] || ""))) {
      res.writeHead(415, { ...HEADERS, "Content-Type": "application/json" }); res.end(JSON.stringify({ ok: false, error: "expected application/json" })); return;
    }
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    const key = await readRunpodKey();
    if (!key) { res.end(JSON.stringify({ ok: false, error: "no RunPod key" })); return; }
    let body = ""; req.on("data", (d) => { body += d; if (body.length > 100000) req.destroy(); });
    await new Promise((r) => req.on("end", r));
    let opts = {}; try { opts = body ? JSON.parse(body) : {}; } catch { /* defaults */ }
    try {
      const pub = await ensureSshKey();
      const input = {
        // Secure Cloud by default: RunPod-managed datacenter hosts stay on current NVIDIA drivers,
        // so a modern-CUDA image (cu12.9) actually starts — community hosts are driver-roulette.
        cloudType: opts.cloudType || "SECURE", gpuCount: 1,
        gpuTypeId: opts.gpu || "NVIDIA GeForce RTX 3090",
        name: (opts.name || "ares-sam3d").slice(0, 40),
        imageName: opts.image || RUNPOD_IMAGE,
        containerDiskInGb: 40, volumeInGb: 30, volumeMountPath: "/workspace",
        ports: "22/tcp,8888/http",
        env: [{ key: "PUBLIC_KEY", value: pub }],
      };
      const d = await runpodGraphQL(key, "mutation($input:PodFindAndDeployOnDemandInput){podFindAndDeployOnDemand(input:$input){id name imageName costPerHr}}", { input });
      res.end(JSON.stringify({ ok: true, pod: d.podFindAndDeployOnDemand }));
    } catch (e) { res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); }
    return;
  }

  // Terminate a pod (fully stops billing). UI confirms.
  if (path === "/runpod/stop" && req.method === "POST") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    const key = await readRunpodKey();
    if (!key) { res.end(JSON.stringify({ ok: false, error: "no RunPod key" })); return; }
    let body = ""; req.on("data", (d) => { body += d; }); await new Promise((r) => req.on("end", r));
    let id; try { id = JSON.parse(body || "{}").id; } catch { /* */ }
    if (!id) { res.end(JSON.stringify({ ok: false, error: "no pod id" })); return; }
    try {
      await runpodGraphQL(key, "mutation($id:String!){podTerminate(input:{podId:$id})}", { id });
      res.end(JSON.stringify({ ok: true }));
    } catch (e) { res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); }
    return;
  }

  // Live logs: SSH to the pod and tail the run log. SSE. Waits (polling) for the pod to expose SSH.
  if (path === "/runpod/logs") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const key = await readRunpodKey();
    const id = url.searchParams.get("id");
    if (!key || !id) { send("error", { message: "missing key or pod id" }); res.end(); return; }
    let child = null, closed = false;
    req.on("close", () => { closed = true; if (child) child.kill(); });
    try {
      let ep = null;
      for (let i = 0; i < 40 && !closed; i++) {          // up to ~2 min for SSH to come up
        ep = await runpodPodSsh(key, id);
        if (ep.ready) break;
        send("log", `waiting for pod SSH… (${ep.status || "starting"})`);
        await new Promise((r) => setTimeout(r, 3000));
      }
      if (closed) { res.end(); return; }
      if (!ep || !ep.ready) { send("error", { message: "pod SSH not ready (still " + (ep && ep.status) + ")" }); res.end(); return; }
      send("log", `connected · ${ep.ip}:${ep.port}`);
      child = sshRun(ep.ip, ep.port, "touch /workspace/run.log; tail -n 400 -F /workspace/run.log",
        (line) => send("log", line), (code) => { send("done", { code }); if (!res.writableEnded) res.end(); });
    } catch (e) { send("error", { message: String((e && e.message) || e) }); res.end(); }
    return;
  }

  // Run a named server-defined script on the pod (probe | setup), streamed + appended to run.log.
  // GET so the browser's EventSource can drive it (params in the query string, no body).
  if (path === "/runpod/action") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const key = await readRunpodKey();
    const id = url.searchParams.get("id");
    const which = url.searchParams.get("script") || "probe";
    const POD_SCRIPTS = {
      probe: "echo '== pod probe =='; hostname; uname -sr; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python -c 'import torch;print(\"torch\",torch.__version__,\"cuda_ok\",torch.cuda.is_available())' 2>&1 | head -1; echo '== probe done =='",
      setup: "set -o pipefail; cd /workspace; echo '== SAM-3D-Body setup =='; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; if [ ! -d sam-3d-body ]; then echo 'cloning facebookresearch/sam-3d-body…'; git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git 2>&1 | tail -3; fi; cd sam-3d-body && echo 'installing (few min)…' && pip install -q -e . 2>&1 | tail -6; echo '== setup step done (exit '$?') =='",
    };
    const script = POD_SCRIPTS[which];
    if (!key || !id || !script) { send("error", { message: "missing key/id or unknown script" }); res.end(); return; }
    let child = null, closed = false;
    req.on("close", () => { closed = true; if (child) child.kill(); });
    try {
      const ep = await runpodPodSsh(key, id);
      if (!ep.ready) { send("error", { message: "pod SSH not ready (" + ep.status + ")" }); res.end(); return; }
      // tee so the logs view (tail -F run.log) mirrors this run too.
      const wrapped = "mkdir -p /workspace; { " + script + " ; } 2>&1 | tee -a /workspace/run.log";
      child = sshRun(ep.ip, ep.port, wrapped, (line) => send("log", line), (code) => { send("done", { code }); if (!res.writableEnded) res.end(); });
    } catch (e) { send("error", { message: String((e && e.message) || e) }); res.end(); }
    return;
  }

  // Start the SAM service from the app — SSE status, no terminal (mirrors /forge/start).
  if (path === "/sam/start") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    // ?for=track additionally requires SAM 3 (the video tracker has no ViT-H path).
    const r = await samEnsure(send, { purpose: url.searchParams.get("for") === "track" ? "track" : "sam" });
    if (r === true) send("done", { url: SAM_URL }); else sendEnsureFailure(send, r);
    res.end();
    return;
  }

  // Same-origin reverse proxy to the local SAM service (editor v2, design §8.1). This server's
  // COEP header blocks direct cross-origin fetches from the app, so the page calls /sam/* and we
  // forward to SAM_URL (default http://127.0.0.1:7263), streaming the response straight through.
  if (path === "/sam" || path.startsWith("/sam/")) {
    const sam = new URL(SAM_URL);
    const fwd = { ...req.headers, host: sam.host };
    for (const h of ["connection", "keep-alive", "upgrade", "te", "trailer", "transfer-encoding"]) delete fwd[h];
    const up = httpRequest({
      hostname: sam.hostname,
      port: sam.port || 80,
      path: (path.slice("/sam".length) || "/") + url.search,
      method: req.method,
      headers: fwd,
    }, (ur) => {
      const out = { ...HEADERS, "Content-Type": ur.headers["content-type"] ?? "application/octet-stream" };
      if (ur.headers["content-length"]) out["Content-Length"] = ur.headers["content-length"];
      res.writeHead(ur.statusCode ?? 502, out);
      ur.pipe(res);
    });
    // Tell the SAM service when the browser walked away. pipe() only UNPIPES a dead client socket;
    // it never destroys the upstream request, so the proxy keeps the upstream response and its
    // socket open and the service goes on computing for a reader that will never come back.
    // Measured on a 60-frame /sam/track/run abandoned after 4 masks: without this the service ran
    // the whole span (16.7 s of GPU) before anything noticed; with it the run stalls at once and
    // the service's own idle-stream reaper cancels it and frees the inference session. It does NOT
    // end the upstream generator by itself — starlette cancels its send task and leaves the sync
    // generator parked, which is exactly the case tools/sam-service/track.py's reaper exists for.
    // Guarded on writableFinished so a normally completed response is left alone.
    res.on("close", () => { if (!res.writableFinished) up.destroy(); });
    up.on("error", (e) => {
      // Also bail once the response is gone: the close handler above destroys `up` deliberately,
      // and answering a 502 into a dead ServerResponse raises on the write instead.
      if (res.headersSent || res.destroyed || res.writableEnded) { res.destroy(); return; }
      const refused = e.code === "ECONNREFUSED";
      // Auto-start on real work (POST /segment); plain health GETs stay passive so
      // status polling never spawns anything.
      if (refused && req.method === "POST") samEnsure(null, { purpose: path.startsWith("/sam/track/") ? "track" : path.startsWith("/sam/depth/") ? "depth" : "sam" }).catch(() => {});
      res.writeHead(refused ? 503 : 502, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify(refused
        ? { error: "sam service not running", starting: req.method === "POST", start: "/sam/start" }
        : { error: "sam proxy failed", detail: e.message }));
    });
    req.pipe(up);
    return;
  }

  // Editor v2 sidecars: GET/POST /edits/<clipname> ⇄ apps/demo/<clipname>.edits.json (non-destructive
  // edit lists, docs/editor-v2-design.md §11). Name sanitized the same way as /encode outputs.
  if (path.startsWith("/edits/")) {
    const name = path.slice("/edits/".length).replace(/\.ares$/i, "").replace(/[^a-z0-9._-]/gi, "_");
    const file = join(ROOT, "apps", "demo", name + ".edits.json");
    if (req.method === "GET") {
      try {
        const body = await readFile(file);
        res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
        res.end(body);
      } catch {
        // No sidecar yet — a normal state, not an error (keeps the browser console clean).
        res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
        res.end("null");
      }
      return;
    }
    if (req.method === "POST") {
      // 16 MB, not 4, because one propagated range writes a bitmap keyframe per FRAME and the demo
      // saves pretty-printed (main.js:1057 — JSON.stringify(edits, null, 1)). rleEncodeMask
      // (edits.ts:50) emits alternating run lengths over the FLAT row-major bitmap, so a silhouette
      // costs two runs per edge crossing per covered row: a standing figure in a 768 px long-side
      // mask (768x432) covers 390 rows and measures 1,185 runs, mean run 280 — 3 digits. Every run
      // is its own ARRAY ELEMENT nine levels down (ranges/keyframes/volumes/mask/rle), so `null, 1`
      // spends ",\n" plus 9 spaces plus the digits on each: 13.7 bytes measured, 16,290 bytes for
      // the keyframe. x272 frames = 4.4 MB, over the old cap on ONE object with no user keyframes
      // and nothing else in the document. Headroom the 16 MB buys: maskRes 1008 is ~5.8 MB, two
      // tracked objects at 768 ~8.9 MB, a 500-frame clip at 768 ~8.1 MB. The same document
      // serialized compact is 1.2 MB, so the cap stays generous even after the writer stops
      // pretty-printing tracked documents.
      //
      // And overflow ANSWERS now. `req.destroy()` sent no response at all, which doSave's
      // .catch(() => {}) (main.js:1057) swallows whole: the sidecar silently stopped being written
      // and the track was gone on the next reload with nothing logged anywhere.
      const CAP = 16_000_000;
      const chunks = [];
      let bytes = 0, over = false;
      req.on("data", (d) => {
        bytes += d.length;
        if (bytes > CAP) {
          if (!over) {
            over = true;
            chunks.length = 0;                  // drop what was buffered; keep draining so the 413 flushes
            const declared = Number(req.headers["content-length"]);
            res.writeHead(413, { ...HEADERS, "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "edit list too large", bytes: declared > bytes ? declared : bytes, cap: CAP }));
          }
          return;
        }
        chunks.push(d);
      });
      req.on("end", async () => {
        if (over) return;                       // the 413 already answered
        // Buffer.concat, not `body += d`: the cap is a BYTE cap, and per-chunk toString() turns a
        // chunk boundary landing mid-sequence in a non-ASCII range label into U+FFFD.
        const body = Buffer.concat(chunks).toString("utf8");
        try {
          JSON.parse(body); // must at least be JSON
          // One-deep backup before every overwrite: a stale browser tab's debounced autosave can
          // clobber newer edits (it DID, 2026-07-11 — lost a saved range). The .bak
          // always holds the previous state, so a clobber is recoverable by one file copy.
          try { await copyFile(file, file + ".bak"); } catch { /* no prior file — first save */ }
          await writeFile(file, body);
          res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, file: `apps/demo/${name}.edits.json` }));
        } catch (e) {
          res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" });
          res.end("bad edit list: " + e.message);
        }
      });
      return;
    }
  }

  // ---- GLOBAL ACTIVITY LOG: everything that happens goes to a toggleable, append-only JSONL log on
  // the SERVER, not localStorage, so it survives tab switches, reloads, and a different browser — an
  // audit trail of what actually happened. Three producers write here: the app (UI actions, console,
  // errors) and the pipeline (bake/encode stages).
  // GET ?tail=N returns the last N entries so a fresh tab restores the full history.
  if (path === "/log") {
    const file = join(ROOT, "apps", "demo", ".ares-activity.jsonl");
    if (req.method === "GET") {
      const tail = Math.min(5000, Math.max(1, Number(url.searchParams.get("tail")) || 800));
      let lines = [];
      try {
        const txt = await readFile(file, "utf8");
        lines = txt.split("\n").filter(Boolean).slice(-tail);
      } catch { /* no log yet */ }
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end("[" + lines.join(",") + "]");
      return;
    }
    if (req.method === "POST") {
      let body = "";
      req.on("data", (d) => { body += d; if (body.length > 400_000) req.destroy(); });
      req.on("end", async () => {
        try {
          const arr = JSON.parse(body);
          const items = Array.isArray(arr) ? arr : [arr];
          const out = items.slice(0, 500).map((e) => JSON.stringify({
            t: Number.isFinite(e.t) ? e.t : Date.now(),
            src: String(e.src || "app").slice(0, 16),        // app | bake | claude | error
            lvl: ["info", "warn", "error", "act"].includes(e.lvl) ? e.lvl : "info",
            msg: String(e.msg ?? "").slice(0, 2000),
            data: e.data === undefined ? undefined : String(typeof e.data === "string" ? e.data : JSON.stringify(e.data)).slice(0, 2000),
          })).join("\n") + "\n";
          await appendFile(file, out);
          // Rotate at ~8 MB so an always-on log can't fill the disk the diag bar is reporting on.
          try {
            const st = await stat(file);
            if (st.size > 8 * 1024 * 1024) {
              const txt = await readFile(file, "utf8");
              await writeFile(file, txt.split("\n").filter(Boolean).slice(-4000).join("\n") + "\n");
            }
          } catch { /* rotation is best-effort */ }
          res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, n: items.length }));
        } catch (e) { res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" }); res.end("bad log: " + e.message); }
      });
      return;
    }
    if (req.method === "DELETE") {
      try { await writeFile(file, ""); } catch { /* */ }
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true, cleared: true }));
      return;
    }
  }

  // Source-bar showcase: the user's editable set of {label, src} buttons. GET returns null when
  // unseeded (client seeds from its defaults); POST replaces it. Persisted as apps/demo/showcase.json.
  if (path === "/showcase") {
    const file = join(ROOT, "apps", "demo", "showcase.json");
    if (req.method === "GET") {
      let body;
      try { body = await readFile(file); } catch { body = "null"; }
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end(body);
      return;
    }
    if (req.method === "POST") {
      let body = "";
      req.on("data", (d) => { body += d; if (body.length > 200_000) req.destroy(); });
      req.on("end", async () => {
        try {
          const arr = JSON.parse(body);
          if (!Array.isArray(arr)) throw new Error("expected an array of {label, src}");
          // Extended library item: label (renamable), src, fav (like), folder (group), addedAt
          // (import time, for the default "recent" sort). Thumbnails live as files (see /save-thumb),
          // not inline, so this stays small. Backward-compatible: old {label,src} entries still load.
          const clean = arr.filter((e) => e && typeof e.label === "string" && typeof e.src === "string")
            .map((e) => ({
              label: e.label.slice(0, 80),
              src: basename(e.src),
              fav: !!e.fav,
              folder: (typeof e.folder === "string" && e.folder.trim()) ? e.folder.trim().slice(0, 40) : null,
              addedAt: Number.isFinite(e.addedAt) ? e.addedAt : Date.now(),
            }));
          await writeFile(file, JSON.stringify(clean, null, 1));
          res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, count: clean.length }));
        } catch (e) { res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" }); res.end("bad showcase: " + e.message); }
      });
      return;
    }
  }

  // Unified volcap history (Inspect + Convert tabs). GET /history → the whole store;
  // POST /history/add → one entry {kind, path?, name?, out?, meta?} (server stamps the time).
  if (path === "/history") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    res.end(JSON.stringify(await readHistoryStore()));
    return;
  }
  if (path === "/history/add" && req.method === "POST") {
    let body = "";
    req.on("data", (d) => { body += d; if (body.length > 100_000) req.destroy(); });
    req.on("end", async () => {
      try {
        const e = JSON.parse(body);
        if (!e || typeof e.kind !== "string") throw new Error("entry needs a kind");
        await historyAdd({
          kind: e.kind.slice(0, 20),
          path: typeof e.path === "string" ? e.path.slice(0, 500) : "",
          name: typeof e.name === "string" ? e.name.slice(0, 200) : "",
          out: typeof e.out === "string" ? e.out.slice(0, 500) : undefined,
          meta: e.meta && typeof e.meta === "object" ? e.meta : undefined,
        });
        res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
      } catch (err) {
        res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" });
        res.end("bad history entry: " + err.message);
      }
    });
    return;
  }

  // List the .ares files available in apps/demo/ (for the "Add to source bar" picker).
  if (path === "/list-ares") {
    const dir = join(ROOT, "apps", "demo");
    try {
      const names = (await readdir(dir)).filter((f) => /\.ares$/i.test(f)).sort();
      const items = [];
      for (const f of names) {
        try {
          const st = await stat(join(dir, f));
          const safe = f.replace(/\.ares$/i, "").replace(/[^a-z0-9._-]/gi, "_");
          // mtime = "last made/imported/edited" (re-encode/re-bake bumps it) → drives the default sort.
          items.push({ src: f, bytes: st.size, mtime: st.mtimeMs, meta: existsSync(join(dir, f + ".meta.json")), thumb: existsSync(join(dir, ".thumbs", safe + ".jpg")) });
        } catch { /* ignore */ }
      }
      items.sort((a, b) => b.mtime - a.mtime); // most-recent first so every dropdown/list defaults to newest-on-top
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify(items));
    } catch (e) { res.writeHead(500, { ...HEADERS, "Content-Type": "application/json" }); res.end(JSON.stringify({ error: String((e && e.message) || e) })); }
    return;
  }

  // Provenance sidecar for a clip: GET /clip-meta?name=foo.ares → the <name>.ares.meta.json the
  // encoder wrote (the full conversion/creation recipe, bound to the clip), or {} if none exists.
  // Read-only, confined to apps/demo. Feeds the ARES readouts' provenance line + the "⧉ Use these
  // settings" transpose (re-apply one clip's recipe to another conversion).
  if (path === "/clip-meta") {
    const safe = basename(url.searchParams.get("name") || "").replace(/\.meta\.json$/i, "").replace(/\.ares$/i, "").replace(/[^a-z0-9._-]/gi, "_");
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    if (!safe) { res.end("{}"); return; }
    try { res.end(await readFile(join(ROOT, "apps", "demo", safe + ".ares.meta.json"), "utf8")); }
    catch { res.end("{}"); }
    return;
  }

  // Clip thumbnail: POST /save-thumb?name=foo.ares  body {dataURI:"data:image/jpeg;base64,…"} writes
  // apps/demo/.thumbs/<safe>.jpg (served statically as .thumbs/<safe>.jpg); body {clear:true} removes it.
  // The client captures the live canvas or an uploaded image — the app owner is the only one who sees it.
  if (path === "/save-thumb" && req.method === "POST") {
    const safe = basename(url.searchParams.get("name") || "").replace(/\.ares$/i, "").replace(/[^a-z0-9._-]/gi, "_");
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    if (!safe) { res.end(JSON.stringify({ ok: false, error: "no name" })); return; }
    let body = ""; req.on("data", (d) => { body += d; if (body.length > 4_000_000) req.destroy(); });
    await new Promise((r) => req.on("end", r));
    try {
      const thumbsDir = join(ROOT, "apps", "demo", ".thumbs");
      const target = join(thumbsDir, safe + ".jpg");
      let payload; try { payload = JSON.parse(body); } catch { payload = { dataURI: body }; }
      if (payload && payload.clear) { try { await unlink(target); } catch { /* already gone */ } res.end(JSON.stringify({ ok: true, cleared: true })); return; }
      const m = /^data:image\/\w+;base64,(.+)$/s.exec((payload && payload.dataURI) || "");
      if (!m) { res.end(JSON.stringify({ ok: false, error: "expected a data:image/*;base64 URI" })); return; }
      await mkdir(thumbsDir, { recursive: true });
      await writeFile(target, Buffer.from(m[1], "base64"));
      res.end(JSON.stringify({ ok: true, thumb: `.thumbs/${safe}.jpg` }));
    } catch (e) { res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); }
    return;
  }

  // Delete a clip from the library (apps/demo/*.ares) + drop it from the source bar. Sanitized and
  // confined to apps/demo; the UI confirms first. This is the owner's "delete from library" option.
  if (path === "/delete-ares" && req.method === "POST") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    let body = ""; req.on("data", (d) => { body += d; if (body.length > 10000) req.destroy(); });
    await new Promise((r) => req.on("end", r));
    let name; try { name = JSON.parse(body || "{}").name; } catch { /* */ }
    const safe = String(name || "").replace(/\.ares$/i, "").replace(/[^a-z0-9._-]/gi, "_");
    if (!safe) { res.end(JSON.stringify({ ok: false, error: "no name" })); return; }
    const target = join(ROOT, "apps", "demo", safe + ".ares");
    try {
      if (!existsSync(target)) { res.end(JSON.stringify({ ok: false, error: "not found" })); return; }
      await unlink(target);
      // clean up the clip's sidecars too (provenance + thumbnail) so nothing dangles
      for (const side of [safe + ".ares.meta.json", join(".thumbs", safe + ".jpg")]) { try { await unlink(join(ROOT, "apps", "demo", side)); } catch { /* optional */ } }
      try {   // also drop it from the persisted source bar so it doesn't dangle
        const scPath = join(ROOT, "apps", "demo", "showcase.json");
        const sc = JSON.parse(await readFile(scPath, "utf8"));
        if (Array.isArray(sc)) { const next = sc.filter((s) => s.src !== safe + ".ares"); if (next.length !== sc.length) await writeFile(scPath, JSON.stringify(next, null, 1)); }
      } catch { /* showcase optional */ }
      res.end(JSON.stringify({ ok: true, deleted: safe + ".ares" }));
    } catch (e) { res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); }
    return;
  }

  // Import a pre-existing .ares into the library (apps/demo). The Import button and the library's
  // drag-drop target both land here. Two shapes, one endpoint:
  //   POST /import-ares              body {"path":"<abs>"}   copy a file already on this machine
  //   POST /import-ares?name=x.ares  body = the raw bytes    stream an upload (drag-drop, file input)
  // The path form is the good one on Windows: the server copies straight from disk, so a 300 MB
  // capture imports at disk speed instead of round-tripping through the browser. Both verify the
  // 'ARES' magic BEFORE anything joins the library (a mis-picked .mp4 fails here, not later as an
  // unplayable row), never overwrite (versionedOutName bumps to -vN), and carry the clip's
  // <name>.ares.meta.json provenance sidecar across when the source has one.
  if (path === "/import-ares" && req.method === "POST") {
    const demoDir = join(ROOT, "apps", "demo");
    const IMPORT_MAX = 4 * 1024 * 1024 * 1024; // captures run to hundreds of MB; this is just a ceiling
    const isAres = (b) => b.length >= 4 && b[0] === 0x41 && b[1] === 0x52 && b[2] === 0x45 && b[3] === 0x53;
    const sameDir = (a, b) => (process.platform === "win32" ? a.toLowerCase() === b.toLowerCase() : a === b);
    const clipName = (n) => basename(String(n)).replace(/\.ares$/i, "").replace(/[^a-z0-9._-]/gi, "_") || "imported";
    const jsonOut = (o) => { res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" }); res.end(JSON.stringify(o)); };
    const uploadName = url.searchParams.get("name");
    try {
      // --- copy from a path on this machine (what the native picker hands us) ---
      if (!uploadName) {
        let body = ""; req.on("data", (d) => { body += d; if (body.length > 10000) req.destroy(); });
        await new Promise((r) => req.on("end", r));
        let srcPath = ""; try { srcPath = String(JSON.parse(body || "{}").path || "").trim(); } catch { /* bad JSON → no path */ }
        if (!srcPath) { jsonOut({ ok: false, error: "no path" }); return; }
        const abs = normalize(srcPath);
        let st; try { st = await stat(abs); } catch { jsonOut({ ok: false, error: "not found: " + abs }); return; }
        if (!st.isFile()) { jsonOut({ ok: false, error: "not a file" }); return; }
        if (st.size > IMPORT_MAX) { jsonOut({ ok: false, error: "larger than the 4 GB import limit" }); return; }
        const head = Buffer.alloc(4);
        const fh = await open(abs, "r");
        try { await fh.read(head, 0, 4, 0); } finally { await fh.close(); }
        if (!isAres(head)) { jsonOut({ ok: false, error: "not an .ares file (bad magic)" }); return; }
        // Already in the library: adopt it where it lies rather than making a second copy of it.
        if (sameDir(dirname(abs), demoDir)) { jsonOut({ ok: true, src: basename(abs), bytes: st.size, already: true }); return; }
        const want = clipName(abs);
        const outName = await versionedOutName(demoDir, want);
        await copyFile(abs, join(demoDir, outName + ".ares"));
        try { await copyFile(abs + ".meta.json", join(demoDir, outName + ".ares.meta.json")); } catch { /* no sidecar to carry */ }
        jsonOut({ ok: true, src: outName + ".ares", bytes: st.size, renamed: outName !== want });
        return;
      }
      // --- streamed upload (drag-drop, the file input, any non-Windows client) ---
      // Streamed to a .part file rather than buffered: an import is allowed to be bigger than RAM.
      // .part never matches the /\.ares$/ filter, so a half-written import cannot show up as a clip.
      const want = clipName(uploadName);
      await mkdir(demoDir, { recursive: true });
      const tmpPath = join(demoDir, `.${want}.${Date.now()}.part`);
      let total = 0, head = Buffer.alloc(0), err = null;
      const fh = await open(tmpPath, "w");
      try {
        for await (const chunk of req) {
          if (head.length < 4) head = Buffer.concat([head, chunk]).subarray(0, 4);
          if (head.length >= 4 && !isAres(head)) { err = "not an .ares file (bad magic)"; break; }
          total += chunk.length;
          if (total > IMPORT_MAX) { err = "larger than the 4 GB import limit"; break; }
          await fh.write(chunk);
        }
      } finally { await fh.close(); }
      if (!err && !total) err = "empty upload";
      if (!err && !isAres(head)) err = "not an .ares file (bad magic)";
      if (err) {
        try { await unlink(tmpPath); } catch { /* nothing landed */ }
        jsonOut({ ok: false, error: err });
        req.destroy();   // answer first, then stop the rest of a rejected upload from streaming in
        return;
      }
      const outName = await versionedOutName(demoDir, want);
      await rename(tmpPath, join(demoDir, outName + ".ares"));
      jsonOut({ ok: true, src: outName + ".ares", bytes: total, renamed: outName !== want });
    } catch (e) {
      jsonOut({ ok: false, error: String((e && e.message) || e) });
    }
    return;
  }

  // System diagnostics feed for the dashboard rail: GPU (util/temp/VRAM), RAM, disk, the coherent
  // bake stage+ETA, and RunPod on/off — one snapshot every 2s over SSE. RunPod is refreshed on a
  // 15s cadence (cached) so the panel never rate-limits the API; all local reads are cheap.
  if (path === "/diagnostics") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const logPath = join(ROOT, "apps", "demo", ".coherent-bake.log");
    let closed = false, timer = null; const rpCache = { at: 0, data: null };
    req.on("close", () => { closed = true; if (timer) clearInterval(timer); });
    const gpuSnap = () => new Promise((resolve) => {
      const p = spawn("nvidia-smi", ["--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], { windowsHide: true });
      let out = ""; p.stdout.on("data", (d) => (out += d)); p.on("error", () => resolve(null));
      p.on("close", () => {
        const l = out.trim().split("\n")[0];
        if (!l) return resolve(null);
        const [name, util, temp, memU, memT] = l.split(",").map((s) => s.trim());
        resolve({ name, util: +util, temp: +temp, vramUsedMB: +memU, vramTotalMB: +memT });
      });
    });
    const snap = async () => {
      if (closed) return;
      let gpu = null, disk = null, bake = null;
      try { gpu = await gpuSnap(); } catch { /* no nvidia-smi */ }
      try { const s = await statfs(ROOT); const totalGB = (s.blocks * s.bsize) / 1073741824, freeGB = (s.bavail * s.bsize) / 1073741824; disk = { totalGB: +totalGB.toFixed(1), freeGB: +freeGB.toFixed(1), usedGB: +(totalGB - freeGB).toFixed(1) }; } catch { /* */ }
      const ram = { usedMB: Math.round((totalmem() - freemem()) / 1048576), totalMB: Math.round(totalmem() / 1048576) };
      try {
        const txt = await readFile(logPath, "utf8");
        let prog = null, done = null, stage = null;
        for (const ln of txt.split(/\r?\n/)) {
          if (ln.startsWith("[PROGRESS] ")) { try { prog = JSON.parse(ln.slice(11)); } catch { /* */ } }
          else if (ln.startsWith("[DONE] ")) { try { done = JSON.parse(ln.slice(7)); } catch { /* */ } }
          else if (ln.startsWith("[STAGE] ")) { try { stage = JSON.parse(ln.slice(8)); } catch { /* */ } }
        }
        bake = { prog, done, stage };
      } catch { /* no bake log */ }
      if (Date.now() - rpCache.at > 15000) {
        rpCache.at = Date.now();
        try {
          const key = await readRunpodKey();
          if (!key) rpCache.data = { error: "no key" };
          else { const d = await runpodGraphQL(key, "query{myself{clientBalance pods{desiredStatus costPerHr machine{gpuDisplayName}}}}"); rpCache.data = { balance: d.myself.clientBalance, pods: (d.myself.pods || []).map((p) => ({ status: p.desiredStatus, gpu: p.machine && p.machine.gpuDisplayName, costPerHr: p.costPerHr })) }; }
        } catch (e) { rpCache.data = { error: String((e && e.message) || e).slice(0, 60) }; }
      }
      send("diag", { gpu, ram, disk, bake, runpod: rpCache.data, at: Date.now() });
    };
    timer = setInterval(snap, 2000); snap();
    return;
  }

  // Dependency status for the Settings tab: what is installed, what each piece enables, and
  // where to get it. Presence checks only (fast stats, no directory walks) — the app must open
  // and switch tabs regardless of what is missing; features warn at the point of use.
  // Component catalog + hardware-derived recommendations for the Settings tab. tools/installer.mjs
  // owns the catalog, the GPU probe, and the profile maths; this route only serves it. `deps` keeps
  // the field names the old route used, so convert.js's depStatus() gate keeps working unchanged.
  if (path === "/deps") {
    try {
      const gpu = await gpuProbe();
      const items = catalog(ROOT, gpu);
      const profs = profiles(items, gpu);
      const deps = items.map((it) => ({
        ...it,
        // Back-compat shape: one `action` per row. A gated model points at its licence page, a
        // plain download at its source, and anything this server can install points at /install.
        action: it.gated ? { kind: "link", url: it.gated.url, note: it.gated.why }
          : it.link ? { kind: "link", url: it.link, note: it.why }
          : it.install && !it.statusOnly ? { kind: "sse", route: "/install?ids=" + it.id, label: "Install", note: it.why }
          : { kind: "cmd", note: it.why },
      }));
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ deps, gpu, profiles: profs, recommended: recommend(profs), preflight: await preflight(ROOT) }));
    } catch (e) {
      res.writeHead(500, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String((e && e.message) || e) }));
    }
    return;
  }

  // Store a Hugging Face access token so gated repos can download.
  //   POST /hf-token  {"token":"hf_…"}  ->  { ok, user } | { ok:false, error }
  // The token arrives once, is handed to huggingface_hub on STDIN (never argv, which any process
  // on the machine can read), and is never echoed back, logged, or written to history. This
  // exists so a gated model is a dialog inside the app rather than a trip to a terminal to run
  // `hf auth login`.
  if (path === "/hf-token" && req.method === "POST") {
    let body = "";
    req.on("data", (d) => { body += d; if (body.length > 8192) req.destroy(); });
    req.on("end", async () => {
      const jsonOut = (o, code = 200) => { res.writeHead(code, { ...HEADERS, "Content-Type": "application/json" }); res.end(JSON.stringify(o)); };
      try {
        const { token } = JSON.parse(body || "{}");
        const r = await saveHfToken(ROOT, token);
        // Deliberately no token in the log line — only whether it worked.
        console.log(`[ares-dev] hf token ${r.ok ? "stored" + (r.user ? ` for ${r.user}` : "") : "rejected"}`);
        jsonOut(r, r.ok ? 200 : 400);
      } catch { jsonOut({ ok: false, error: "malformed request" }, 400); }
    });
    return;
  }

  // Install one or more components: GET /install?ids=a,b,c — SSE, one line at a time.
  // `resolve` expands each id through its `requires` graph and orders dependencies first, so
  // asking for a model that needs the Python env installs the env without being told to.
  // Anything already present is skipped rather than re-fetched, which makes a re-run after a
  // failure cheap and safe: it picks up exactly where it stopped.
  if (path === "/install") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}
data: ${JSON.stringify(data)}

`); };
    const line = (t) => send("log", String(t));
    let cancelled = false;
    req.on("close", () => { cancelled = true; });
    try {
      const gpu = await gpuProbe();
      const items = catalog(ROOT, gpu);
      const wanted = (url.searchParams.get("ids") || "").split(",").map((s) => s.trim()).filter(Boolean);
      if (!wanted.length) { send("error", { message: "no ids given" }); res.end(); return; }
      const plan = resolve(items, wanted).filter((it) => !it.statusOnly && it.install);
      const todo = plan.filter((it) => !it.present);
      const skipped = plan.filter((it) => it.present);
      for (const it of skipped) line(`✓ ${it.label}: already installed, skipping`);
      if (!todo.length) { send("done", { message: "everything requested is already installed" }); res.end(); return; }
      line(`installing ${todo.length} component${todo.length > 1 ? "s" : ""}: ${todo.map((t) => t.label).join(", ")}`);
      send("plan", { todo: todo.map((t) => ({ id: t.id, label: t.label, sizeMB: t.sizeMB })), skipped: skipped.map((t) => t.id) });
      let done = 0;
      for (const it of todo) {
        if (cancelled) return;
        send("step", { id: it.id, label: it.label, index: done, total: todo.length });
        line(`── ${it.label} (${it.sizeMB || "?"} MB) ──`);
        const r = await installOne(ROOT, it, line);
        if (it.id === "ffmpeg") ffTools({ fresh: true });
        if (!r.ok) {
          line(`✗ ${it.label}: ${r.error}`);
          send("error", { message: r.error, id: it.id, component: it.id, label: it.label, gated: !!r.gated, needsToken: !!r.needsToken, url: it.gated?.url });
          res.end();
          return;
        }
        done++;
        line(`✓ ${it.label} ready`);
        send("stepDone", { id: it.id, index: done, total: todo.length });
      }
      send("done", { message: `${done} component${done > 1 ? "s" : ""} installed` });
    } catch (e) {
      send("error", { message: String((e && e.message) || e) });
    }
    res.end();
    return;
  }

  // Guided installs (Settings tab). Each route streams SSE progress and goes through
  // ensureComponents, so it installs only what is absent and a repeat is a no-op.
  if (path === "/setup/sam-env") {
    // The environment is the catalog's `python-env`: interpreter discovery (or the private
    // CPython), the venv, and the PyTorch build chosen for THIS GPU all live in installer.mjs.
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    if (!(await ensureOrEnd(["python-env"], send, res))) return;
    send("done", { message: "Python environment ready" });
    res.end();
    return;
  }
  // The licensed 4DViews codec: POST /setup/4ds-codec {"path":"<abs BridgeCodec4DS.dll>"} copies
  // the picked file into tools/4ds/bin. The path comes from the native dialog (/pick), never typed.
  if (path === "/setup/4ds-codec" && req.method === "POST") {
    let body = ""; req.on("data", (d) => { body += d; if (body.length > 10000) req.destroy(); });
    await new Promise((r) => req.on("end", r));
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    let src = ""; try { src = String(JSON.parse(body || "{}").path || "").trim(); } catch { /* malformed */ }
    try {
      const st = src ? await stat(src) : null;
      if (!st || !st.isFile() || !/\.dll$/i.test(src)) { res.end(JSON.stringify({ ok: false, error: "not a DLL file" })); return; }
      const head = Buffer.alloc(2);
      const fh = await open(src, "r");
      try { await fh.read(head, 0, 2, 0); } finally { await fh.close(); }
      if (head.toString("latin1") !== "MZ") { res.end(JSON.stringify({ ok: false, error: "not a Windows executable image" })); return; }
      await mkdir(dirname(FOURDS_DLL_LOCAL), { recursive: true });
      await copyFile(src, FOURDS_DLL_LOCAL);
      res.end(JSON.stringify({ ok: true, path: FOURDS_DLL_LOCAL }));
    } catch (e) { res.end(JSON.stringify({ ok: false, error: String((e && e.message) || e) })); }
    return;
  }
  if (path === "/setup/demo-clip") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const out = join(ROOT, "apps", "demo", "demo.ares");
    if (existsSync(out)) { send("done", { message: "demo.ares already present", out: "/apps/demo/demo.ares" }); res.end(); return; }
    if (!(await ensureOrEnd(["encoder"], send, res))) return;
    const cli = encoderState(ROOT).cli;
    const child = spawn(process.execPath, [cli, "synth", "-o", out], { cwd: ROOT, windowsHide: true, env: toolEnv() });
    const relay = (d) => String(d).split(/\r?\n/).forEach((l) => l.trim() && send("log", l.trim()));
    child.stdout.on("data", relay);
    child.stderr.on("data", relay);
    child.on("error", (e) => { send("error", { message: e.message }); res.end(); });
    child.on("close", (code) => { send(code === 0 ? "done" : "error", code === 0 ? { message: "demo.ares written", out: "/apps/demo/demo.ares" } : { message: "encoder exited " + code }); res.end(); });
    req.on("close", () => { try { child.kill(); } catch { /* ignore */ } });
    return;
  }

  // Native OS folder/file picker (Windows) — so the user never types a filesystem path.
  //   GET /pick?type=folder|file[&dir=<initial>][&for=<key>][&filter=<ofd filter>]  → { path | null }
  // `for` keys a remembered last-used folder (history.json lastDirs): the dialog reopens there,
  // and a successful pick updates it.
  if (path === "/pick") {
    const type = url.searchParams.get("type") === "file" ? "file" : "folder";
    const forKey = (url.searchParams.get("for") || "").replace(/[^a-z0-9_-]/gi, "").slice(0, 40);
    try {
      let initial = url.searchParams.get("dir") || undefined;
      if (!initial && forKey) {
        const remembered = (await readHistoryStore()).lastDirs[forKey];
        if (remembered && existsSync(remembered)) initial = remembered;
      }
      const picked = await pickNative(type, { filter: url.searchParams.get("filter") || undefined, initialDirectory: initial });
      if (picked && forKey) rememberDir(forKey, type === "file" ? dirname(picked) : picked);
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ path: picked }));
    } catch (e) {
      res.writeHead(500, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String((e && e.message) || e) }));
    }
    return;
  }

  // Server-side folder analysis (parity with the browser drag-drop preview for a picked folder).
  //   GET /analyse?dir=<abs>  → { dir, meshes, kind, atlases, atlasDims, verts, rawBytes }
  if (path === "/analyse") {
    const dir = url.searchParams.get("dir") || "";
    let ok = false; try { ok = !!dir && statSync(dir).isDirectory(); } catch { ok = false; }
    if (!ok) { res.writeHead(400, { ...HEADERS, "Content-Type": "application/json" }); res.end(JSON.stringify({ error: `not a directory: ${dir}` })); return; }
    try {
      const info = await analyseDir(dir);
      if (info.meshes) historyAdd({ kind: "analyse", path: info.dir, name: basename(info.dir), meta: { meshes: info.meshes, kind: info.kind, atlases: info.atlases, rawBytes: info.rawBytes, verts: info.verts } });
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify(info));
    } catch (e) {
      res.writeHead(500, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String((e && e.message) || e) }));
    }
    return;
  }

  // Pre-warm / start Forge (generative enhance tier): SSE status, no terminal.
  if (path === "/forge/start") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const r = await forgeEnsure(send);
    if (r === true) send("done", { url: FORGE_URL }); else sendEnsureFailure(send, r);
    res.end();
    return;
  }

  // GUI-triggered local encode (Convert tab). Runs the real encoder CLI on this machine and streams
  // its progress back as Server-Sent Events, the user never opens a terminal. localhost-only.
  // coherent=1 (DEFAULT; visual verdict 2026-07-13: "Coherent A by far the best"): a stable-template
  // pre-pass (tools/coherent/coherent-clip.mjs) rewrites the clip into per-GOP shared-topology
  // frames with atlases rebaked into the template's UVs, THEN the normal encoder runs on that
  // temp frames-dir: same two-stage-inside-one-SSE-stream shape as /convert-4ds. NOT used by
  // /convert-4ds itself (4DViews topology resets need cross-reset correspondence: future work).
  if (path === "/encode") {
    const q = url.searchParams;
    const dir = q.get("dir") || "";
    let ok = false;
    try { ok = !!dir && statSync(dir).isDirectory(); } catch { ok = false; }
    if (!ok) { res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" }); res.end(`not a directory: ${dir}`); return; }
    const reqName = (q.get("name") || "converted").replace(/[^a-z0-9._-]/gi, "_");
    const name = await versionedOutName(join(ROOT, "apps", "demo"), reqName); // auto -vN, never overwrite
    const outRel = `apps/demo/${name}.ares`;
    let coherent = q.get("coherent") === "1";
    // Splat sequences skip the mesh-only coherent pre-pass and the texture flags (the encoder
    // ignores them for splat input anyway); their own flags ride through below.
    let isSplatDir = false;
    try { isSplatDir = !!(await analyseDir(dir)).splat; } catch { /* treat as mesh */ }
    if (coherent && isSplatDir) coherent = false;
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const relay = (d) => String(d).split(/\r?\n/).forEach((l) => l.trim() && send("log", l));
    let child = null;
    req.on("close", () => { try { child?.kill(); } catch { /* ignore */ } });
    // One stage = one child process whose output relays to the same SSE stream.
    const runStage = (stageArgs) => new Promise((resolve, reject) => {
      child = spawn(process.execPath, stageArgs, { cwd: ROOT, windowsHide: true, env: toolEnv() });
      child.stdout.on("data", relay);
      child.stderr.on("data", relay);
      child.on("error", reject);
      child.on("close", resolve);
    });
    (async () => {
      let tmpDir = null;
      try {
        // The encoder build (rebuilt when its sources are newer) and ffmpeg (texture video, audio)
        // are installed here, inside this stream, when absent.
        if (!(await ensureOrEnd(["encoder", "ffmpeg"], send, res))) return;
        let encDir = dir;
        if (coherent) {
          // The runner needs the actual frames dir (it does not descend like the encoder does)
          // and mesh-fNNNNN.obj + atlas-fNNNNN.png naming: it fails loudly (relayed) otherwise.
          const framesSrc = await resolveFramesDir(dir);
          tmpDir = await mkdtemp(join(tmpdir(), "ares-coherent-"));
          const cohFrames = join(tmpDir, "frames");
          const cohArgs = [join(ROOT, "tools/coherent/coherent-clip.mjs"), framesSrc, cohFrames, "--ckpt", join(tmpDir, "ckpt")];
          const cohPass = (k, f) => { const v = q.get(k); if (v != null && v !== "") cohArgs.push(f, v); };
          cohPass("gop", "--gop"); cohPass("maxFrames", "--max-frames"); // geometry-GOP length intentionally matches the texture --gop default (30)
          cohPass("bake", "--bake"); // absent → runner default "exact" (the fast bake smears)
          send("start", { dir, out: "/" + outRel, args: "coherent pre-pass → encode", stage: "coherent" });
          send("log", "[coherent] stable-template pre-pass (registration + atlas rebake): this is the long stage…");
          if (q.get("decimate")) send("log", "[coherent] note: decimate re-collapses each frame independently and can undo the shared topology (frames may fall back to intra)");
          const code = await runStage(cohArgs);
          if (code !== 0) { send("error", { code, stage: "coherent", message: `coherent pre-pass exited ${code}; see log above` }); res.end(); return; }
          encDir = cohFrames;
          send("log", "[coherent] pre-pass done: encoding coherent frames…");
        }
        // Provenance enrichment: stamp the encoder's <out>.ares.meta.json sidecar with which pipeline
        // produced this clip and (coherent) the resolved registration recipe from the pre-pass manifest.
        let metaExtraArgs = [];
        try {
          const metaExtra = { pipeline: coherent ? "coherent" : "native", source: { dir } };
          if (coherent && tmpDir) { try { metaExtra.coherent = JSON.parse(await readFile(join(tmpDir, "coherent-manifest.json"), "utf8")).clip; } catch { /* manifest optional */ } }
          const mePath = join(tmpdir(), `ares-meta-extra-${Date.now()}.json`);
          await writeFile(mePath, JSON.stringify(metaExtra));
          metaExtraArgs = ["--meta-extra-file", mePath];
        } catch { /* best-effort provenance */ }

        const args = [join(ROOT, "packages/encoder/dist/cli.js"), "encode", encDir, "-o", join(ROOT, outRel), ...metaExtraArgs];
        const pass = (k, f) => { const v = q.get(k); if (v != null && v !== "") args.push(f, v); };
        pass("fps", "--fps"); pass("maxFrames", "--max-frames"); pass("gop", "--gop");
        pass("textureCodec", "--texture-codec"); pass("texSize", "--tex-size"); pass("crf", "--crf");
        pass("smooth", "--smooth"); pass("crop", "--crop"); pass("decimate", "--decimate");
        // Model transform (Edit ▸ Transform): bakes the up-axis / centring / scale into the
        // geometry. The encoder validates each of these strictly and refuses garbage, so a typo
        // stops the bake instead of silently encoding the untransformed clip.
        pass("upAxis", "--up-axis"); pass("center", "--center"); pass("scale", "--scale");
        pass("rotate", "--rotate"); pass("translate", "--translate");
        // Splat profile (spec §6.8): SH cap, outlier-alpha filter, position quantization bits.
        pass("shDegree", "--sh-degree"); pass("splatMinAlpha", "--splat-min-alpha"); pass("quantBits", "--quant-bits");
        // Audio track (spec §11.5): any file ffmpeg reads, transcoded to Opus and laid into the chunks.
        const audioPath = q.get("audio");
        if (audioPath) {
          let okA = false; try { okA = statSync(audioPath).isFile(); } catch { okA = false; }
          if (!okA) { send("error", { message: `audio file not found: ${audioPath}` }); res.end(); return; }
          args.push("--audio", audioPath);
          pass("audioOffset", "--audio-offset"); pass("audioBitrate", "--audio-bitrate");
        }
        if (isSplatDir) send("log", "[encode] splat sequence detected: encoding as the Gaussian splat profile (coherent pre-pass and texture flags do not apply)");
        // editsName → the sidecar saved via POST /edits/<name> (path stays server-side, sanitized).
        const editsName = q.get("editsName");
        if (editsName) args.push("--edits", join(ROOT, "apps", "demo", editsName.replace(/[^a-z0-9._-]/gi, "_") + ".edits.json"));
        if (q.get("noTexture") === "1") args.push("--no-texture");
        if (!coherent) send("start", { dir, out: "/" + outRel, args: args.slice(1).join(" ") });
        const code = await runStage(args);
        if (code === 0) {
          // meta records the FULL recipe, the history panel's "⧉ Use settings" re-applies it to
          // another clip's conversion.
          historyAdd({ kind: "encode", path: dir, name: name + ".ares", out: "/" + outRel, meta: {
            codec: q.get("textureCodec") || "vp9", texSize: q.get("texSize") || "", crf: q.get("crf") || "",
            smooth: q.get("smooth") || "0", coherent: coherent ? "1" : "", fps: q.get("fps") || "",
            decimate: q.get("decimate") || "", maxFrames: q.get("maxFrames") || "", noTexture: q.get("noTexture") || "",
            profile: isSplatDir ? "splat" : "mesh", shDegree: q.get("shDegree") || "", splatMinAlpha: q.get("splatMinAlpha") || "", quantBits: q.get("quantBits") || "",
            audio: q.get("audio") || "", audioOffset: q.get("audioOffset") || "",
          } });
        }
        send(code === 0 ? "done" : "error", { code, out: "/" + outRel });
        res.end();
      } catch (e) {
        send("error", { message: e?.message || String(e) });
        res.end();
      } finally {
        if (tmpDir) rm(tmpDir, { recursive: true, force: true }).catch(() => { /* best-effort temp cleanup */ });
      }
    })();
    return;
  }

  // 4DViews .4ds codec info (Task I). GET /probe-4ds?path=<abs .4ds> → decode_4ds.py --info's
  // JSON {nbFrames, framerate, maxVertices, maxTriangles, textureSize, textureEncoding}, no
  // decode, just a CreateSequence + query. Richer than the browser's byte-level structural probe
  // (probe.js) because it comes straight from the codec. Used by the Convert tab to size the
  // max-frames default and gate the Convert button before any decode work starts.
  if (path === "/probe-4ds") {
    const p = url.searchParams.get("path") || "";
    let ok = false;
    try { ok = !!p && /\.4ds$/i.test(p) && statSync(p).isFile(); } catch { ok = false; }
    if (!ok) { res.writeHead(400, { ...HEADERS, "Content-Type": "application/json" }); res.end(JSON.stringify({ error: `not a .4ds file: ${p}` })); return; }
    // Not an error to read: `needs` is installed by the client through /install and `needsFile`
    // is collected with the file dialog, then this request is repeated (apps/demo/ensure.js).
    const lack = fourdsNeeds();
    if (lack.needs.length || lack.needsFile) {
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: lack.needsFile ? "4DViews codec DLL not located" : "Python environment absent", ...lack }));
      return;
    }
    try {
      const info = await run4dsInfo(p);
      res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify(info));
    } catch (e) {
      res.writeHead(500, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String((e && e.message) || e) }));
    }
    return;
  }

  // GUI-triggered .4ds → .ares conversion (Task I, Convert tab). GET + SSE like /encode and
  // /enhance above (EventSource can't POST, so this follows their query-string convention rather
  // than the JSON-body shape sketched in the task brief). Pipeline: decode_4ds.py writes a
  // per-frame OBJ+PNG frames-dir into a fresh OS-temp directory, then the real encoder CLI bakes
  // that dir into apps/demo/<name>.ares, the same two tools Task H already verified standalone,
  // just chained and streamed. The temp frames dir is ALWAYS removed in a finally, mirroring
  // cli.ts's own scratch-atlas-dir cleanup (`.finally(() => cleanupTexDir?.())`), success or fail.
  if (path === "/convert-4ds") {
    const q = url.searchParams;
    const srcPath = q.get("path") || "";
    const rawName = q.get("name") || "";
    const maxFramesArg = q.get("maxFrames") || "";
    const mirrorX = q.get("mirrorX") === "1";

    // Optional codec-quality overrides (Task J fix 3). Default for .4ds converts is now the
    // SOURCE atlas's NATIVE texture size (not a hardcoded 1024) and crf 28 (not 32): measured:
    // native size + crf28 cuts end-to-end UV-seam MAE from 8.14 to 5.75 (-29%) for +5% file size,
    // vs downscaling to 1024 which concentrates lanczos error on the seams before the codec even
    // runs. Query params are validated integers; a present-but-invalid value 400s rather than
    // silently falling back to some other size/quality than what was asked for.
    const parseIntParam = (name, min, max) => {
      const raw = q.get(name);
      if (raw == null || raw === "") return { present: false, value: null };
      const n = Number(raw);
      if (!Number.isInteger(n) || n < min || n > max) return { present: true, value: undefined };
      return { present: true, value: n };
    };
    const texSizeReq = parseIntParam("texSize", 64, 4096);
    const crfReq = parseIntParam("crf", 0, 63);
    if (texSizeReq.present && texSizeReq.value === undefined) {
      res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" });
      res.end(`bad texSize (integer 64-4096): "${q.get("texSize")}"`);
      return;
    }
    if (crfReq.present && crfReq.value === undefined) {
      res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" });
      res.end(`bad crf (integer 0-63): "${q.get("crf")}"`);
      return;
    }

    let srcOk = false;
    try { srcOk = !!srcPath && /\.4ds$/i.test(srcPath) && statSync(srcPath).isFile(); } catch { srcOk = false; }
    if (!srcOk) { res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" }); res.end(`not a .4ds file: ${srcPath}`); return; }
    if (!/^[a-z0-9_-]+$/i.test(rawName)) { res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" }); res.end(`bad output name (letters/digits/-/_ only): "${rawName}"`); return; }
    const encoderCli = join(ROOT, "packages", "encoder", "dist", "cli.js");

    const name = await versionedOutName(join(ROOT, "apps", "demo"), rawName); // auto -vN, never overwrite
    const outRel = `apps/demo/${name}.ares`;
    const outAbs = join(ROOT, outRel);   // was undeclared: every .4ds convert died at the encode stage with "outAbs is not defined"

    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    let closed = false, child = null, tmpDir = null;
    req.on("close", () => { closed = true; try { if (child) child.kill(); } catch { /* ignore */ } });

    try {
      send("start", { path: srcPath, name, out: "/" + outRel, maxFrames: maxFramesArg || null, mirrorX });
      if (!(await ensureOrEnd(["python-env", "encoder", "ffmpeg"], send, res))) return;
      // The codec DLL is licensed and cannot be fetched: the client collects it with a file dialog
      // (needsFile) and re-issues this request. /probe-4ds asks first, so this is the backstop.
      const lack = fourdsNeeds();
      if (lack.needsFile) { send("error", { message: "4DViews codec DLL not located", stage: "setup", needsFile: lack.needsFile }); res.end(); return; }
      tmpDir = await mkdtemp(join(tmpdir(), "ares-4ds-"));

      // --- decode phase: decode_4ds.py prints "[decode_4ds] N/M frames (...)" progress lines:
      // forward every line as a log event and pull frame/of out of the matching ones as progress.
      const decodeArgs = [FOURDS_SCRIPT, srcPath, "-o", tmpDir];
      if (maxFramesArg) decodeArgs.push("--max-frames", maxFramesArg);
      if (mirrorX) decodeArgs.push("--mirror-x");
      send("log", `[server] decoding (this can take a while: ~2 fps): ${basename(FOURDS_PY)} decode_4ds.py ${basename(srcPath)} -o <tmp>${maxFramesArg ? " --max-frames " + maxFramesArg : ""}${mirrorX ? " --mirror-x" : ""}`);
      await new Promise((resolve, reject) => {
        const c = spawn(FOURDS_PY, decodeArgs, { windowsHide: true, env: toolEnv() });
        child = c;
        let buf = "";
        const onData = (d) => {
          buf += d;
          const lines = buf.split(/\r?\n/);
          buf = lines.pop() ?? "";
          for (const l of lines) {
            if (!l.trim()) continue;
            send("log", l);
            const m = l.match(/(\d+)\/(\d+) frames/);
            if (m) send("progress", { stage: "decode", frame: Number(m[1]), of: Number(m[2]) });
          }
        };
        c.stdout.on("data", onData);
        c.stderr.on("data", onData);
        c.on("error", reject);
        c.on("close", (code) => { child = null; code === 0 ? resolve() : reject(new Error(`decode_4ds.py exited ${code}`)); });
      });
      if (closed) return;

      // fps for the bake comes from the codec itself (manifest.json), rounded to an integer:
      // more reliable than trusting the .4ds probe's float fps for --fps. Same manifest also
      // reports the SOURCE atlas's native textureSize, the new default (fix 3) absent an explicit
      // texSize= override.
      let fps = 30, framesDecoded = null, nativeTexSize = null;
      try {
        const manifest = JSON.parse(await readFile(join(tmpDir, "manifest.json"), "utf8"));
        fps = Math.round(manifest.fps) || 30;
        framesDecoded = manifest.nbFramesDecoded;
        nativeTexSize = Number(manifest.textureSize) || null;
      } catch { /* keep default fps=30 if the manifest is somehow unreadable */ }
      const texSize = texSizeReq.value ?? nativeTexSize ?? 1024;
      const crf = crfReq.value ?? 28;
      send("log", `[server] decode complete: ${framesDecoded ?? "?"} frame(s); encoding at ${fps}fps, ${texSize}px crf${crf}…`);
      send("progress", { stage: "encode" });

      // --- encode phase: the same encoder CLI /encode already spawns; .4ds converts now default
      // to native texture size + crf 28 (fix 3) instead of the encoder's own generic 1024/crf32.
      const encodeArgs = [encoderCli, "encode", tmpDir, "-o", outAbs, "--fps", String(fps), "--tex-size", String(texSize), "--crf", String(crf)];
      await new Promise((resolve, reject) => {
        const c = spawn(process.execPath, encodeArgs, { cwd: ROOT, windowsHide: true, env: toolEnv() });
        child = c;
        const relay = (d) => String(d).split(/\r?\n/).forEach((l) => l.trim() && send("log", l));
        c.stdout.on("data", relay);
        c.stderr.on("data", relay);
        c.on("error", reject);
        c.on("close", (code) => { child = null; code === 0 ? resolve() : reject(new Error(`encoder exited ${code}`)); });
      });
      if (closed) return;

      historyAdd({
        kind: "encode", path: srcPath, name: name + ".ares", out: "/" + outRel,
        meta: { codec: "vp9", texSize: String(texSize), crf: String(crf), source: "4ds", frames: framesDecoded, fps, mirrorX: mirrorX || undefined },
      });
      send("done", { out: "/" + outRel, frames: framesDecoded, fps });
    } catch (e) {
      send("error", { message: String((e && e.message) || e) });
    } finally {
      // Always reclaim the temp frames dir: success or failure; same discipline as cli.ts's
      // scratch atlas-dir cleanup. A full-length decode can be gigabytes of OBJ+PNG.
      if (tmpDir) { try { await rm(tmpDir, { recursive: true, force: true }); } catch { /* best-effort */ } }
    }
    res.end();
    return;
  }

  // GUI-triggered generative texture enhance (Convert tab).
  // Streams SSE progress (mirroring /encode) while each atlas-*.png is POSTed to a local SD-Forge
  // instance. tier "resrgan": extras API; upscaler_2 (R-ESRGAN 4x+) blended over upscaler_1
  // (Lanczos) at `strength` via extras_upscaler_2_visibility, so the strength dial runs inside
  // Forge (parameter names verified against Forge's modules/api/models.py, ExtrasBaseRequest).
  // tier "sd": script-less img2img at source resolution, denoise = strength*0.5 (hero frames:
  // minutes/frame). Output goes to a SIBLING folder that /encode can consume directly.
  if (path === "/enhance") {
    const q = url.searchParams;
    const dir = q.get("dir") || "";
    let ok = false;
    try { ok = !!dir && statSync(dir).isDirectory(); } catch { ok = false; }
    if (!ok) { res.writeHead(400, { ...HEADERS, "Content-Type": "text/plain" }); res.end(`not a directory: ${dir}`); return; }
    // tiers: "ncnn" = standalone Real-ESRGAN (no server, default) · "sd" = Forge img2img (generative).
    // Legacy tier names (resrgan/forge) fold into the no-server ncnn path.
    const tier = q.get("tier") === "sd" ? "sd" : "ncnn";
    const strength = Math.min(1, Math.max(0, Number(q.get("strength") ?? 1) || 0));
    const scale = Math.min(4, Math.max(1, Number(q.get("scale") ?? 2) || 2));
    const model = /anime/i.test(q.get("model") || "") ? "realesrgan-x4plus-anime" : "realesrgan-x4plus";
    const explicitOut = q.get("outDirName");
    const mf = Number(q.get("maxFrames") || 0);
    const maxFrames = mf >= 1 ? Math.floor(mf) : Infinity;

    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    let closed = false, inflight = null, child = null;
    req.on("close", () => { closed = true; try { if (inflight) inflight.destroy(new Error("client closed")); } catch { /* ignore */ } try { if (child) child.kill(); } catch { /* ignore */ } });
    const atlasesIn = (names) => names.filter((f) => /^atlas-.*\.png$/i.test(f));

    try {
      // Frames dir: allow pointing at a parent that holds a single subfolder of atlas-*.png frames
      // (mirrors the CLI encode descent) so one path works for both enhance and the follow-up convert.
      let srcDir = dir;
      let all = (await readdir(srcDir)).sort();
      let atlases = atlasesIn(all);
      if (!atlases.length) {
        const hits = [];
        for (const name of all) {
          const p = join(srcDir, name);
          try { if (statSync(p).isDirectory() && atlasesIn(await readdir(p)).length) hits.push(p); }
          catch { /* ignore unreadable entries */ }
        }
        if (hits.length === 1) {
          srcDir = hits[0];
          all = (await readdir(srcDir)).sort();
          atlases = atlasesIn(all);
        } else if (hits.length > 1) {
          send("error", { message: `multiple subfolders under ${dir} contain atlas-*.png: point enhance at the specific frames folder`, hint: hits.map((p) => basename(p)).join(", ") });
          res.end(); return;
        }
      }
      if (!atlases.length) { send("error", { message: `no atlas-*.png files in ${srcDir}` }); res.end(); return; }
      const outName = (explicitOut || basename(srcDir) + "-enhanced").replace(/[\\/:*?"<>|]/g, "_");
      const outDir = join(dirname(srcDir), outName);
      const todo = atlases.slice(0, maxFrames);
      await mkdir(outDir, { recursive: true });
      // Make the out folder a drop-in /encode source: hard-link (same volume — it's a sibling;
      // copy as fallback) the meshes and any atlas frames beyond maxFrames.
      const carry = async (f) => {
        try { await link(join(srcDir, f), join(outDir, f)); }
        catch (e) { if (e.code !== "EEXIST") { try { await copyFile(join(srcDir, f), join(outDir, f)); } catch { /* ignore */ } } }
      };
      for (const f of all) if (!/\.png$/i.test(f)) await carry(f);
      for (const f of atlases.slice(todo.length)) await carry(f);
      send("start", { dir: srcDir, out: outDir, frames: todo.length, of: atlases.length, tier, strength, scale, via: tier === "sd" ? "SD img2img (Forge)" : "Real-ESRGAN (ncnn, no server)" });

      // Tier prechecks: sd auto-launches Forge; ncnn just needs the vendored binary present.
      if (tier === "sd") {
        const f = await forgeEnsure(send);
        if (f !== true) { sendEnsureFailure(send, f); res.end(); return; }
      } else if (!existsSync(REALESRGAN_EXE)) {
        send("error", { message: "Real-ESRGAN upscaler missing", hint: "expected tools/bin/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan.exe (re-run the vendor step)" }); res.end(); return;
      }

      const t0 = Date.now();
      for (let i = 0; i < todo.length; i++) {
        if (closed) return;
        const f = todo[i];
        const srcPng = join(srcDir, f);
        const t = Date.now();
        if (tier === "sd") {
          const buf = await readFile(srcPng);
          const isPng = buf.length > 24 && buf.readUInt32BE(0) === 0x89504e47; // PNG IHDR w/h
          const w = isPng ? buf.readUInt32BE(16) : 1024, h = isPng ? buf.readUInt32BE(20) : 1024;
          const r = await forgePost(FORGE_URL, "/sdapi/v1/img2img", {
            init_images: [buf.toString("base64")],
            denoising_strength: Math.min(0.5, strength * 0.5),
            prompt: "high detail photographic texture, sharp fabric weave, skin pores",
            steps: 20, width: w, height: h,
          }, (rq) => { inflight = rq; });
          inflight = null;
          if (closed) return;
          const outB64 = r.images && r.images[0];
          if (!outB64) throw new Error("Forge returned no image (no checkpoint loaded)");
          await writeFile(join(outDir, f), Buffer.from(outB64, "base64"));
        } else {
          // ncnn upscales to 2× → tmp; an optional ffmpeg pass handles scale=1 (downscale to source
          // for detail-only) and/or the strength blend: out = lanczos(orig)*(1-S) + esrgan*S — the same
          // Lanczos/R-ESRGAN blend Forge's extras tier did, but with no server.
          const tmpPng = join(outDir, "._up_" + f);
          await runProc(REALESRGAN_EXE, ["-i", srcPng, "-o", tmpPng, "-n", model, "-s", "2", "-t", "256", "-g", "0"], { cwd: REALESRGAN_DIR }, (c) => { child = c; });
          child = null;
          if (closed) { try { await unlink(tmpPng); } catch { /* ignore */ } return; }
          const dims = pngDimsBuf(await readFile(srcPng));
          const outPng = join(outDir, f);
          if ((strength < 0.99 || scale < 2) && dims) {
            const tw = scale < 2 ? dims[0] : dims[0] * 2, th = scale < 2 ? dims[1] : dims[1] * 2;
            await runProc(FFMPEG, ["-y", "-loglevel", "error", "-i", srcPng, "-i", tmpPng,
              "-filter_complex", `[0:v]scale=${tw}:${th}:flags=lanczos[a];[1:v]scale=${tw}:${th}:flags=lanczos[b];[a][b]blend=all_expr='A*(1-${strength})+B*${strength}'`,
              "-frames:v", "1", outPng], {}, (c) => { child = c; });
            child = null;
            try { await unlink(tmpPng); } catch { /* ignore */ }
          } else {
            try { await rename(tmpPng, outPng); } catch { await copyFile(tmpPng, outPng); try { await unlink(tmpPng); } catch { /* ignore */ } }
          }
          if (closed) return;
        }
        send("progress", { frame: i + 1, of: todo.length, file: f, ms: Date.now() - t });
      }
      historyAdd({ kind: "enhance", path: srcDir, name: basename(outDir), out: outDir, meta: { frames: todo.length, tier, strength, scale } });
      send("done", { out: outDir, frames: todo.length, ms: Date.now() - t0 });
    } catch (e) {
      send("error", { message: String((e && e.message) || e) });
    }
    res.end();
    return;
  }
  if (path === "/") {
    res.writeHead(302, { ...HEADERS, Location: PROBE });
    res.end();
    return;
  }

  // Resolve inside ROOT only (reject traversal).
  const fsPath = normalize(join(ROOT, path));
  if (fsPath !== ROOT && !fsPath.startsWith(ROOT.endsWith(sep) ? ROOT : ROOT + sep)) {
    res.writeHead(403, HEADERS);
    res.end("forbidden");
    return;
  }

  try {
    let target = fsPath;
    const s = await stat(target);
    if (s.isDirectory()) {
      if (!path.endsWith("/")) {
        res.writeHead(302, { ...HEADERS, Location: path + "/" });
        res.end();
        return;
      }
      target = join(target, "index.html");
    }
    // Streamed, never read whole: a long .ares clip runs to gigabytes, and readFile stops at 2 GiB.
    await sendFileRange(req, res, target, MIME[extname(target).toLowerCase()] ?? "application/octet-stream");
  } catch {
    res.writeHead(404, { ...HEADERS, "Content-Type": "text/plain" });
    res.end(`404 ${path}`);
  }
}

function listen(port, attemptsLeft) {
  // Wrap the async handler so a rejection returns 500 instead of crashing the whole dev server.
  const server = createServer((req, res) => {
    Promise.resolve(handle(req, res)).catch((e) => {
      console.error("[ares] handler error:", (e && e.stack) || e);
      try { if (!res.headersSent) { res.writeHead(500, { ...HEADERS, "Content-Type": "text/plain" }); res.end("server error"); } else res.end(); } catch { /* ignore */ }
    });
  });
  server.on("error", (e) => {
    if (e.code === "EADDRINUSE" && attemptsLeft > 0) listen(port + 1, attemptsLeft - 1);
    else {
      console.error(`[ares-dev] cannot bind: ${e.message}`);
      process.exit(1);
    }
  });
  server.listen(port, "127.0.0.1", () => {
    console.log(`[ares-dev] serving ${ROOT}`);
    console.log(`[ares-dev] http://127.0.0.1:${port}${PROBE}`);
  });
}

listen(BASE_PORT, 10);
