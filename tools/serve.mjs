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
import { statSync, existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { join, normalize, extname, dirname, basename, sep } from "node:path";
import zlib from "node:zlib";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

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
  let kind = objs.length ? "OBJ" : plys.length ? "PLY" : "—";
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

// ---- Forge (generative "hero" tier only) — auto-launched headless, no terminal ------------
const FORGE_ROOT = process.env.FORGE_ROOT || join(homedir(), "webui_forge");            // set FORGE_ROOT to your install
const FORGE_PY = join(FORGE_ROOT, "system", "python", "python.exe");
const FORGE_WEBUI = join(FORGE_ROOT, "webui");
const FORGE_CKPT_DIR = process.env.FORGE_CKPT_DIR || join(FORGE_ROOT, "webui", "models", "Stable-diffusion");
const FORGE_URL = process.env.FORGE_URL || "http://127.0.0.1:7861";
const FORGE_PATH_PREPEND = [join(FORGE_ROOT, "system", "git", "bin"), join(FORGE_ROOT, "system", "python"), join(FORGE_ROOT, "system", "python", "Scripts")].join(";");
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
// Ensure Forge's API is up: health-check, else spawn it detached+hidden and poll. `send` streams
// SSE status. Mirrors the repo's hidden-PowerShell launcher path (tools/launch.ps1).
async function forgeEnsure(send) {
  if (await forgeHealthy()) return true;
  if (!existsSync(FORGE_PY)) { send && send("log", `Forge not found at ${FORGE_ROOT} — set FORGE_ROOT to your webui_forge install`); return false; }
  if (!forgeChild || forgeChild.exitCode !== null) {
    send && send("log", "starting Forge (headless API, cold start ~30–60 s)…");
    const args = ["launch.py", "--nowebui", "--skip-install"];
    if (existsSync(FORGE_CKPT_DIR)) args.push("--ckpt-dir", FORGE_CKPT_DIR);
    forgeChild = spawn(FORGE_PY, args, {
      cwd: FORGE_WEBUI,
      env: { ...process.env, PATH: FORGE_PATH_PREPEND + ";" + process.env.PATH },
      detached: true, stdio: "ignore", windowsHide: true,
    });
    forgeChild.unref();
  } else {
    send && send("log", "waiting for Forge to finish starting…");
  }
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    if (await forgeHealthy()) { send && send("log", "Forge API ready"); return true; }
  }
  send && send("log", "Forge did not become ready within 120 s (check tools/… or start it once manually)");
  return false;
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
async function samEnsure(send) {
  let h = await samHealth();
  if (h && h.ok) return true;
  if (h && h.error) { send && send("log", `SAM model load failed: ${h.error} — see tools/sam-service/sam-service.log`); return false; }
  if (!h) {
    if (!existsSync(SAM_PS1)) { send && send("log", `SAM launcher missing: ${SAM_PS1}`); return false; }
    if (!samChild || samChild.exitCode !== null) {
      send && send("log", "starting SAM service (model load ~40 s on first start)…");
      samChild = spawn("powershell.exe",
        ["-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", SAM_PS1],
        { detached: true, stdio: "ignore", windowsHide: true });
      samChild.unref();
    } else {
      send && send("log", "waiting for the SAM service to finish starting…");
    }
  } else {
    send && send("log", "SAM service is loading the model…");
  }
  const deadline = Date.now() + 180000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    h = await samHealth();
    if (h && h.ok) { send && send("log", `SAM ready (${h.model} on ${h.device})`); return true; }
    if (h && h.error) { send && send("log", `SAM model load failed: ${h.error} — see tools/sam-service/sam-service.log`); return false; }
  }
  send && send("log", "SAM did not become ready within 180 s — see tools/sam-service/sam-service.log");
  return false;
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
function fourdsMissing() {
  const missing = [];
  if (!existsSync(FOURDS_PY)) missing.push("SAM-service Python env (tools/sam-service/env/Scripts/python.exe) — see ⚙ Settings → SAM service Python env");
  if (!fourdsDllPath()) missing.push("BridgeCodec4DS.dll — copy it from your 4DViews SDK into tools\\4ds\\bin\\ (or set FOURDS_DLL)");
  return missing;
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
    const ok = await samEnsure(send);
    send(ok ? "done" : "error", ok ? { url: SAM_URL } : { message: "SAM service could not be started — see tools/sam-service/sam-service.log" });
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
    up.on("error", (e) => {
      if (res.headersSent) { res.destroy(); return; }
      const refused = e.code === "ECONNREFUSED";
      // Auto-start on real work (POST /segment); plain health GETs stay passive so
      // status polling never spawns anything.
      if (refused && req.method === "POST") samEnsure(null);
      res.writeHead(refused ? 503 : 502, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify(refused
        ? { error: "sam service not running", starting: req.method === "POST", hint: "GET /sam/start streams launch progress" }
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
      let body = "";
      req.on("data", (d) => { body += d; if (body.length > 4_000_000) req.destroy(); });
      req.on("end", async () => {
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
  if (path === "/deps") {
    const REPO = dirname(ROOT);
    const sizeOf = (p) => { try { return Math.round(statSync(p).size / 1048576) || 1; } catch { return null; } };
    const samPy = join(ROOT, "tools", "sam-service", "env", "Scripts", "python.exe");
    const sam3W = join(REPO, "sam3", "model.safetensors");
    const sam31W = join(REPO, "sam3.1", "sam3.1_multiplex.pt");
    const vitH = process.env.SAM_CKPT || join(ROOT, "tools", "sam-service", "models", "sam_vit_h_4b8939_fp16.safetensors");
    const keeper = join(ROOT, "apps", "demo", "daniel-s0.ares");
    const synth = join(ROOT, "apps", "demo", "demo.ares");
    const capture = join(REPO, "Daniel_Microsoft_Volcap", "Daniel_Volcap");
    const deps = [
      { id: "encoder", label: "Encoder build (tsc output)", present: existsSync(join(ROOT, "packages", "encoder", "dist", "cli.js")),
        enables: "Convert tab encodes, editor Bake", sizeNote: "rebuilds in seconds",
        action: { kind: "cmd", note: "run: npm install && npx tsc -b (in ares/)" } },
      { id: "sam3-weights", label: "SAM 3 weights (sam3/model.safetensors)", present: existsSync(sam3W), sizeMB: sizeOf(sam3W), sizeNote: "~3.3 GB",
        enables: "SAM click-to-select in the editor (primary backend)", path: sam3W,
        action: { kind: "link", url: "https://huggingface.co/facebook/sam3", note: "gated repo — accept the license, download the transformers snapshot into <repo>/sam3/" } },
      { id: "sam-env", label: "SAM service Python env", present: existsSync(samPy), sizeNote: "~4.8 GB (torch cu124 + transformers)",
        enables: "runs the local segmentation service", path: dirname(dirname(samPy)),
        action: { kind: "sse", route: "/setup/sam-env", label: "Create env", note: "downloads ~2.5 GB of wheels; takes several minutes" } },
      { id: "vith", label: "SAM ViT-H fallback checkpoint", present: existsSync(vitH), sizeNote: "~1.2 GB", optional: true,
        enables: "fallback segmentation backend when SAM 3 weights are absent", path: vitH,
        action: { kind: "link", url: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", note: "optional — only used when sam3/ is missing; set SAM_CKPT to its path" } },
      { id: "sam31", label: "SAM 3.1 multiplex checkpoint", present: existsSync(sam31W), sizeMB: sizeOf(sam31W), sizeNote: "~3.3 GB", optional: true,
        enables: "nothing yet — held until HF transformers ships SAM 3.1 support", path: sam31W,
        action: { kind: "link", url: "https://huggingface.co/facebook/sam3.1", note: "no current code path loads it; safe to delete to reclaim 3.3 GB" } },
      { id: "realesrgan", label: "Real-ESRGAN (ncnn-vulkan)", present: existsSync(REALESRGAN_EXE), sizeNote: "~50 MB",
        enables: "Convert tab's Fast texture-enhance tier", path: REALESRGAN_DIR,
        action: { kind: "link", url: "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases", note: "unzip into ares/tools/bin/realesrgan-ncnn-vulkan/" } },
      { id: "forge", label: "SD-Forge (generative enhance)", present: existsSync(FORGE_PY), sizeNote: "external install", optional: true,
        enables: "Convert tab's Generative (img2img) enhance tier", path: FORGE_ROOT,
        action: { kind: "link", url: "https://github.com/lllyasviel/stable-diffusion-webui-forge", note: "external app; set FORGE_ROOT if installed elsewhere" } },
      { id: "keeper-clip", label: "Reference clip (daniel-s0.ares)", present: existsSync(keeper), sizeMB: sizeOf(keeper), sizeNote: "~50 MB", optional: true,
        enables: "the Viewer's default source and Compare presets", path: keeper,
        action: { kind: "cmd", note: "re-encode from the capture folder via the Convert tab" } },
      { id: "synth-clip", label: "Synthetic demo clip (demo.ares)", present: existsSync(synth), sizeMB: sizeOf(synth), sizeNote: "~3 MB",
        enables: "a from-nothing playable clip (no capture data needed)", path: synth,
        action: { kind: "sse", route: "/setup/demo-clip", label: "Generate", note: "runs the encoder's synth generator locally, ~seconds" } },
      { id: "capture", label: "Source capture frames (Daniel_Volcap)", present: existsSync(capture), sizeNote: "~1.6 GB", optional: true,
        enables: "re-encoding, editor Bake, enhance experiments", path: capture,
        action: { kind: "cmd", note: "local dataset — any per-frame OBJ/PLY + atlas PNG folder works via the Convert tab" } },
      { id: "4ds-codec", label: "4DViews codec (BridgeCodec4DS.dll)", present: !!fourdsDllPath() && existsSync(FOURDS_PY), optional: true,
        sizeNote: "your licensed 4DViews SDK install", path: fourdsDllPath() || FOURDS_DLL_LOCAL,
        enables: "Convert tab's .4ds → .ares conversion (DXT1 desktop captures, ~2 fps decode)",
        action: { kind: "cmd", note: "not an installer — copy your own licensed DLL from your 4DViews SDK into tools\\4ds\\bin\\ (also needs the SAM-service Python env above, for numpy+Pillow)" } },
    ];
    res.writeHead(200, { ...HEADERS, "Content-Type": "application/json" });
    res.end(JSON.stringify({ deps }));
    return;
  }

  // Guided installs (Settings tab). Each route streams SSE progress and refuses to touch
  // anything that already exists — a failed or repeated install can never break a working state.
  if (path === "/setup/sam-env") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const svcDir = join(ROOT, "tools", "sam-service");
    const envPy = join(svcDir, "env", "Scripts", "python.exe");
    if (existsSync(envPy)) { send("done", { message: "env already present — nothing to do" }); res.end(); return; }
    const reqs = join(svcDir, "requirements.txt");
    if (!existsSync(reqs)) { send("error", { message: "requirements.txt missing in tools/sam-service/" }); res.end(); return; }
    send("log", "creating venv (python -m venv env)…");
    const script = `Set-Location '${svcDir}'; python -m venv env; if ($LASTEXITCODE -ne 0) { exit 1 }; ` +
      `.\\env\\Scripts\\python.exe -m pip install --upgrade pip; ` +
      `.\\env\\Scripts\\pip.exe install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`;
    const child = spawn("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], { windowsHide: true });
    const relay = (d) => String(d).split(/\r?\n/).forEach((l) => l.trim() && send("log", l.trim().slice(0, 300)));
    child.stdout.on("data", relay);
    child.stderr.on("data", relay);
    child.on("error", (e) => { send("error", { message: e.message }); res.end(); });
    child.on("close", (code) => { send(code === 0 ? "done" : "error", code === 0 ? { message: "env ready — press Start SAM in the editor" } : { message: "pip exited " + code }); res.end(); });
    req.on("close", () => { try { child.kill(); } catch { /* ignore */ } });
    return;
  }
  if (path === "/setup/demo-clip") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const out = join(ROOT, "apps", "demo", "demo.ares");
    if (existsSync(out)) { send("done", { message: "demo.ares already present", out: "/apps/demo/demo.ares" }); res.end(); return; }
    const cli = join(ROOT, "packages", "encoder", "dist", "cli.js");
    if (!existsSync(cli)) { send("error", { message: "encoder not built — run npx tsc -b in ares/ first" }); res.end(); return; }
    const child = spawn(process.execPath, [cli, "synth", "-o", out], { cwd: ROOT, windowsHide: true });
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

  // Pre-warm / start Forge (generative enhance tier) — SSE status, no terminal.
  if (path === "/forge/start") {
    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    const ok = await forgeEnsure(send);
    send(ok ? "done" : "error", ok ? { url: FORGE_URL } : { message: "Forge could not be started" });
    res.end();
    return;
  }

  // GUI-triggered local encode (Convert tab). Runs the real encoder CLI on this machine and streams
  // its progress back as Server-Sent Events — the user never opens a terminal. localhost-only.
  // coherent=1 (DEFAULT; visual verdict 2026-07-13: "Coherent A by far the best"): a stable-template
  // pre-pass (tools/coherent/coherent-clip.mjs) rewrites the clip into per-GOP shared-topology
  // frames with atlases rebaked into the template's UVs, THEN the normal encoder runs on that
  // temp frames-dir — same two-stage-inside-one-SSE-stream shape as /convert-4ds. NOT used by
  // /convert-4ds itself (4DViews topology resets need cross-reset correspondence — future work).
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
      child = spawn(process.execPath, stageArgs, { cwd: ROOT, windowsHide: true });
      child.stdout.on("data", relay);
      child.stderr.on("data", relay);
      child.on("error", reject);
      child.on("close", resolve);
    });
    (async () => {
      let tmpDir = null;
      try {
        let encDir = dir;
        if (coherent) {
          // The runner needs the actual frames dir (it does not descend like the encoder does)
          // and mesh-fNNNNN.obj + atlas-fNNNNN.png naming — it fails loudly (relayed) otherwise.
          const framesSrc = await resolveFramesDir(dir);
          tmpDir = await mkdtemp(join(tmpdir(), "ares-coherent-"));
          const cohFrames = join(tmpDir, "frames");
          const cohArgs = [join(ROOT, "tools/coherent/coherent-clip.mjs"), framesSrc, cohFrames, "--ckpt", join(tmpDir, "ckpt")];
          const cohPass = (k, f) => { const v = q.get(k); if (v != null && v !== "") cohArgs.push(f, v); };
          cohPass("gop", "--gop"); cohPass("maxFrames", "--max-frames"); // geometry-GOP length intentionally matches the texture --gop default (30)
          cohPass("bake", "--bake"); // absent → runner default "exact" (the fast bake smears)
          send("start", { dir, out: "/" + outRel, args: "coherent pre-pass → encode", stage: "coherent" });
          send("log", "[coherent] stable-template pre-pass (registration + atlas rebake) — this is the long stage…");
          if (q.get("decimate")) send("log", "[coherent] note: decimate re-collapses each frame independently and can undo the shared topology (frames may fall back to intra)");
          const code = await runStage(cohArgs);
          if (code !== 0) { send("error", { code, stage: "coherent", message: `coherent pre-pass exited ${code} — see log above` }); res.end(); return; }
          encDir = cohFrames;
          send("log", "[coherent] pre-pass done — encoding coherent frames…");
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
        if (isSplatDir) send("log", "[encode] splat sequence detected — encoding as the Gaussian splat profile (coherent pre-pass and texture flags do not apply)");
        // editsName → the sidecar saved via POST /edits/<name> (path stays server-side, sanitized).
        const editsName = q.get("editsName");
        if (editsName) args.push("--edits", join(ROOT, "apps", "demo", editsName.replace(/[^a-z0-9._-]/gi, "_") + ".edits.json"));
        if (q.get("noTexture") === "1") args.push("--no-texture");
        if (!coherent) send("start", { dir, out: "/" + outRel, args: args.slice(1).join(" ") });
        const code = await runStage(args);
        if (code === 0) {
          // meta records the FULL recipe — the history panel's "⧉ Use settings" re-applies it to
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
  // JSON {nbFrames, framerate, maxVertices, maxTriangles, textureSize, textureEncoding} — no
  // decode, just a CreateSequence + query. Richer than the browser's byte-level structural probe
  // (probe.js) because it comes straight from the codec. Used by the Convert tab to size the
  // max-frames default and gate the Convert button before any decode work starts.
  if (path === "/probe-4ds") {
    const p = url.searchParams.get("path") || "";
    let ok = false;
    try { ok = !!p && /\.4ds$/i.test(p) && statSync(p).isFile(); } catch { ok = false; }
    if (!ok) { res.writeHead(400, { ...HEADERS, "Content-Type": "application/json" }); res.end(JSON.stringify({ error: `not a .4ds file: ${p}` })); return; }
    const missing = fourdsMissing();
    if (missing.length) {
      res.writeHead(503, { ...HEADERS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "4DViews codec unavailable", missing }));
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
  // that dir into apps/demo/<name>.ares — the same two tools Task H already verified standalone,
  // just chained and streamed. The temp frames dir is ALWAYS removed in a finally, mirroring
  // cli.ts's own scratch-atlas-dir cleanup (`.finally(() => cleanupTexDir?.())`), success or fail.
  if (path === "/convert-4ds") {
    const q = url.searchParams;
    const srcPath = q.get("path") || "";
    const rawName = q.get("name") || "";
    const maxFramesArg = q.get("maxFrames") || "";
    const mirrorX = q.get("mirrorX") === "1";

    // Optional codec-quality overrides (Task J fix 3). Default for .4ds converts is now the
    // SOURCE atlas's NATIVE texture size (not a hardcoded 1024) and crf 28 (not 32) — measured:
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
    const missing = fourdsMissing();
    if (missing.length) { res.writeHead(503, { ...HEADERS, "Content-Type": "text/plain" }); res.end("4DViews codec unavailable: " + missing.join("; ")); return; }
    const encoderCli = join(ROOT, "packages", "encoder", "dist", "cli.js");
    if (!existsSync(encoderCli)) { res.writeHead(503, { ...HEADERS, "Content-Type": "text/plain" }); res.end("encoder not built — run npx tsc -b in ares/ first"); return; }

    const name = await versionedOutName(join(ROOT, "apps", "demo"), rawName); // auto -vN, never overwrite
    const outRel = `apps/demo/${name}.ares`;

    res.writeHead(200, { ...HEADERS, "Content-Type": "text/event-stream", Connection: "keep-alive" });
    const send = (ev, data) => { if (!res.writableEnded) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`); };
    let closed = false, child = null, tmpDir = null;
    req.on("close", () => { closed = true; try { if (child) child.kill(); } catch { /* ignore */ } });

    try {
      tmpDir = await mkdtemp(join(tmpdir(), "ares-4ds-"));
      send("start", { path: srcPath, name, out: "/" + outRel, maxFrames: maxFramesArg || null, mirrorX, tmp: tmpDir });

      // --- decode phase: decode_4ds.py prints "[decode_4ds] N/M frames (...)" progress lines —
      // forward every line as a log event and pull frame/of out of the matching ones as progress.
      const decodeArgs = [FOURDS_SCRIPT, srcPath, "-o", tmpDir];
      if (maxFramesArg) decodeArgs.push("--max-frames", maxFramesArg);
      if (mirrorX) decodeArgs.push("--mirror-x");
      send("log", `[server] decoding (this can take a while — ~2 fps): ${basename(FOURDS_PY)} decode_4ds.py ${basename(srcPath)} -o <tmp>${maxFramesArg ? " --max-frames " + maxFramesArg : ""}${mirrorX ? " --mirror-x" : ""}`);
      await new Promise((resolve, reject) => {
        const c = spawn(FOURDS_PY, decodeArgs, { windowsHide: true });
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

      // fps for the bake comes from the codec itself (manifest.json), rounded to an integer —
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
      send("log", `[server] decode complete — ${framesDecoded ?? "?"} frame(s); encoding at ${fps}fps, ${texSize}px crf${crf}…`);
      send("progress", { stage: "encode" });

      // --- encode phase: the same encoder CLI /encode already spawns; .4ds converts now default
      // to native texture size + crf 28 (fix 3) instead of the encoder's own generic 1024/crf32.
      const encodeArgs = [encoderCli, "encode", tmpDir, "-o", outAbs, "--fps", String(fps), "--tex-size", String(texSize), "--crf", String(crf)];
      await new Promise((resolve, reject) => {
        const c = spawn(process.execPath, encodeArgs, { cwd: ROOT, windowsHide: true });
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
      // Always reclaim the temp frames dir — success or failure — same discipline as cli.ts's
      // scratch atlas-dir cleanup. A full-length decode can be gigabytes of OBJ+PNG.
      if (tmpDir) { try { await rm(tmpDir, { recursive: true, force: true }); } catch { /* best-effort */ } }
    }
    res.end();
    return;
  }

  // GUI-triggered generative texture enhance (Convert tab).
  // Streams SSE progress (mirroring /encode) while each atlas-*.png is POSTed to a local SD-Forge
  // instance. tier "resrgan": extras API — upscaler_2 (R-ESRGAN 4x+) blended over upscaler_1
  // (Lanczos) at `strength` via extras_upscaler_2_visibility, so the strength dial runs inside
  // Forge (parameter names verified against Forge's modules/api/models.py, ExtrasBaseRequest).
  // tier "sd": script-less img2img at source resolution, denoise = strength*0.5 (hero frames —
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
          send("error", { message: `multiple subfolders under ${dir} contain atlas-*.png — point enhance at the specific frames folder`, hint: hits.map((p) => basename(p)).join(", ") });
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
        if (!(await forgeEnsure(send))) { send("error", { message: "Forge unavailable for the generative tier", hint: "use the Fast (Real-ESRGAN) tier — it needs no server — or set FORGE_ROOT to your webui_forge install" }); res.end(); return; }
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
          if (!outB64) throw new Error("Forge returned no image (is a checkpoint loaded?)");
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
    const body = await readFile(target);
    const type = MIME[extname(target).toLowerCase()] ?? "application/octet-stream";
    res.writeHead(200, { ...HEADERS, "Content-Type": type, "Content-Length": body.length });
    res.end(body);
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
