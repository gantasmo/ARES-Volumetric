/**
 * ARES component installer — the catalog behind the Settings tab.
 *
 * One source of truth for every optional model, tool, and Python package the app can use:
 * where it might already be, what it enables, what it needs first, what it costs to download,
 * and how much VRAM it wants at runtime. The Settings tab renders this; /install executes it.
 *
 * Three ideas carry the whole file:
 *
 *  1. DETECTION LOOKS EVERYWHERE A THING COULD ALREADY BE. The old check tested one hardcoded
 *     path per component, so a model downloaded the ordinary way (`hf download facebook/sam3`,
 *     which lands in the shared HF cache) read as "not installed" and invited a second 6.5 GB
 *     download. Every model here resolves through the HF cache as well as the repo-local paths.
 *
 *  2. RECOMMENDATIONS COME FROM THE MEASURED MACHINE, NOT FROM A GUESS. gpuProbe() reads the
 *     real GPUs; profiles are then filtered by actual VRAM against each component's measured
 *     footprint. A component is only dropped from a tier when the hardware genuinely cannot
 *     hold it — never because a card "looks old".
 *
 *  3. A COMPONENT DECLARES WHAT IT NEEDS. `requires` is a real graph, so asking for SAM 3 pulls
 *     in the Python env (and the env pulls in a CUDA wheel index chosen for THIS GPU) without
 *     the user knowing any of that exists.
 */
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream, existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { mkdir, readdir, readFile, rename, rm, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// ---------------------------------------------------------------- paths ----

export function paths(ROOT) {
  const svc = join(ROOT, "tools", "sam-service");
  return {
    ROOT,
    REPO: dirname(ROOT),
    svc,
    envPy: join(svc, "env", "Scripts", "python.exe"),
    envDir: join(svc, "env"),
    models: join(svc, "models"),
    reqs: join(svc, "requirements.txt"),
    bin: join(ROOT, "tools", "bin"),
    esrganDir: join(ROOT, "tools", "bin", "realesrgan-ncnn-vulkan"),
    ffmpegDir: join(ROOT, "tools", "bin", "ffmpeg"),     // pinned static build, never on PATH
    gitDir: join(ROOT, "tools", "bin", "git"),           // MinGit, used when the machine has no git
    encoderStamp: join(ROOT, "tools", "bin", ".encoder-build"),
    ext: join(ROOT, "tools", "ext"),          // git clones of third-party source trees live here
  };
}

// ------------------------------------------------------- tool resolvers ----
// Every external tool resolves through one function, and every one of them accepts a private
// copy under tools/bin. That is what lets the installer fetch a tool with no administrator
// rights and no PATH edit: the server hands the resolved absolute path to whatever it spawns.

/** The encoders the app uses: VP9 and AV1 texture video, Opus audio. A build without all three
 *  is rejected, so a job never starts against an ffmpeg that fails halfway through. */
export const FFMPEG_ENCODERS = ["libvpx-vp9", "libsvtav1", "libopus"];
const ffProbeCache = new Map();   // "<exe>|<mtime>" -> missing encoder names, or null when it does not run
function ffmpegMissingEncoders(exe) {
  let key = exe;
  try { key = exe + "|" + statSync(exe).mtimeMs; } catch { /* bare command name */ }
  if (ffProbeCache.has(key)) return ffProbeCache.get(key);
  let missing = null;
  try {
    const r = spawnSync(exe, ["-hide_banner", "-encoders"], { windowsHide: true, encoding: "utf8", timeout: 10000 });
    if (r.status === 0 && r.stdout) missing = FFMPEG_ENCODERS.filter((c) => !r.stdout.split(/\r?\n/).some((l) => l.split(/\s+/).includes(c)));
  } catch { missing = null; }
  ffProbeCache.set(key, missing);
  return missing;
}

/** ffmpeg.exe inside the private install. The release zip wraps everything in one versioned
 *  folder; installZip hoists it, and this still looks one level down in case it did not. */
function localFfmpeg(P) {
  const exe = process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg";
  const spots = [join(P.ffmpegDir, "bin", exe), join(P.ffmpegDir, exe)];
  try {
    for (const e of readdirSync(P.ffmpegDir, { withFileTypes: true })) if (e.isDirectory()) spots.push(join(P.ffmpegDir, e.name, "bin", exe));
  } catch { /* not installed */ }
  return spots.find((p) => existsSync(p)) || null;
}

/**
 * ffmpeg + ffprobe, in this order: FFMPEG / FFMPEG_PATH override, tools/bin/ffmpeg, the legacy
 * C:\FFmpeg\bin, PATH. A candidate is accepted only when it runs, carries every encoder in
 * FFMPEG_ENCODERS, and has an ffprobe. Returns { ffmpeg, ffprobe, source } with absolute paths,
 * or null. `findFfmpeg.rejected` holds the reason each skipped candidate was skipped.
 */
export function findFfmpeg(ROOT) {
  const P = paths(ROOT);
  const abs = (v) => (!v ? null : /[\\/]/.test(v) ? (existsSync(v) ? v : null) : whichSync(v.replace(/\.exe$/i, "")));
  const envFf = process.env.FFMPEG || process.env.FFMPEG_PATH || "";
  const legacy = "C:\\FFmpeg\\bin\\ffmpeg.exe";
  const candidates = [
    { source: "FFMPEG override", exe: abs(envFf), asked: envFf },
    { source: "tools/bin/ffmpeg", exe: localFfmpeg(P) },
    { source: "C:\\FFmpeg\\bin", exe: process.platform === "win32" && existsSync(legacy) ? legacy : null },
    { source: "PATH", exe: whichSync("ffmpeg") },
  ];
  const rejected = [];
  for (const c of candidates) {
    if (!c.exe) { if (c.asked) rejected.push(`${c.source}: ${c.asked} does not exist`); continue; }
    const missing = ffmpegMissingEncoders(c.exe);
    if (missing === null) { rejected.push(`${c.source}: ${c.exe} does not run`); continue; }
    if (missing.length) { rejected.push(`${c.source}: ${c.exe} lacks ${missing.join(", ")}`); continue; }
    const envProbe = process.env.FFPROBE || process.env.FFPROBE_PATH || "";
    const sibling = c.exe.replace(/ffmpeg(\.exe)?$/i, (m, ext) => "ffprobe" + (ext || ""));
    const ffprobe = (c.source === "FFMPEG override" && abs(envProbe))
      || (sibling !== c.exe && existsSync(sibling) ? sibling : null) || whichSync("ffprobe");
    if (!ffprobe) { rejected.push(`${c.source}: no ffprobe beside ${c.exe}`); continue; }
    findFfmpeg.rejected = rejected;
    return { ffmpeg: c.exe, ffprobe, source: c.source };
  }
  findFfmpeg.rejected = rejected;
  return null;
}
findFfmpeg.rejected = [];

/** git: PATH first, then the private MinGit under tools/bin/git. */
export function findGit(ROOT) {
  const local = join(paths(ROOT).gitDir, "cmd", "git.exe");
  return whichSync("git") || (existsSync(local) ? local : null);
}

/** npm without relying on PATH: the server already runs under node, and every official Node
 *  distribution ships npm beside it. Returns { cmd, prefix } to spawn, or null. */
export function findNpm() {
  const cli = join(dirname(process.execPath), "node_modules", "npm", "bin", "npm-cli.js");
  if (existsSync(cli)) return { cmd: process.execPath, prefix: [cli] };
  const onPath = whichSync("npm");
  return onPath ? { cmd: onPath, prefix: [] } : null;
}

/** vcvars64.bat is the proof that the C++ workload is installed. The BuildTools folder alone is
 *  not: the bootstrapper creates it before any compiler exists. */
function findVcvars() {
  for (const root of ["C:\\Program Files\\Microsoft Visual Studio", "C:\\Program Files (x86)\\Microsoft Visual Studio"]) {
    let years = [];
    try { years = readdirSync(root); } catch { continue; }
    for (const year of years) for (const ed of ["BuildTools", "Community", "Professional", "Enterprise"]) {
      const p = join(root, year, ed, "VC", "Auxiliary", "Build", "vcvars64.bat");
      if (existsSync(p)) return p;
    }
  }
  return null;
}

/** Encoder build state. Stale means a source file under packages/encoder/src or packages/core/src
 *  is newer than both dist/cli.js and the last build attempt, so a route can rebuild before it
 *  spawns the CLI. A tree with no sources (a release bundle) is never stale. */
export function encoderState(ROOT) {
  const cli = join(ROOT, "packages", "encoder", "dist", "cli.js");
  if (!existsSync(cli)) return { built: false, stale: false, cli };
  let built = 0;
  for (const f of [cli, paths(ROOT).encoderStamp]) { try { built = Math.max(built, statSync(f).mtimeMs); } catch { /* no stamp yet */ } }
  let newest = 0;
  const walk = (d) => {
    let entries = [];
    try { entries = readdirSync(d, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const p = join(d, e.name);
      if (e.isDirectory()) walk(p);
      else if (/\.ts$/.test(e.name)) { try { newest = Math.max(newest, statSync(p).mtimeMs); } catch { /* raced */ } }
    }
  };
  for (const pkg of ["encoder", "core"]) walk(join(ROOT, "packages", pkg, "src"));
  return { built: true, stale: newest > built, cli };
}

/** True when a Hugging Face token is available to the downloaders. */
export function hfTokenPresent() {
  if (process.env.HF_TOKEN) return true;
  try { return readFileSync(hfTokenFile(), "utf8").trim().length > 0; } catch { return false; }
}

/** PATH lookup without spawning anything, so the catalog can stay synchronous. */
export function whichSync(exe) {
  const isWin = process.platform === "win32";
  const exts = isWin ? [".exe", ".cmd", ".bat", ""] : [""];
  for (const dir of (process.env.PATH || "").split(isWin ? ";" : ":")) {
    if (!dir) continue;
    for (const e of exts) {
      const full = join(dir.replace(/^"|"$/g, ""), exe + e);
      try { if (existsSync(full) && statSync(full).isFile()) return full; } catch { /* unreadable PATH entry */ }
    }
  }
  return null;
}

/** Where huggingface_hub keeps the CLI token. HF_TOKEN in the environment wins over it. */
export function hfTokenFile() {
  if (process.env.HF_HOME) return join(process.env.HF_HOME, "token");
  return join(homedir(), ".cache", "huggingface", "token");
}

/** The shared Hugging Face cache — the same one `hf download`, transformers, and diffusers use.
 *  HF_HOME wins, then HUGGINGFACE_HUB_CACHE, else the documented ~/.cache/huggingface/hub. */
export function hfCacheDir() {
  if (process.env.HUGGINGFACE_HUB_CACHE) return process.env.HUGGINGFACE_HUB_CACHE;
  if (process.env.HF_HOME) return join(process.env.HF_HOME, "hub");
  return join(homedir(), ".cache", "huggingface", "hub");
}

/** Resolve a repo id to its cached snapshot directory, or null. `refs/main` names the current
 *  revision; a snapshot is only accepted when the files the loader actually needs are in it, so
 *  an interrupted or metadata-only download does not read as installed. */
export function hfSnapshot(repoId, needFiles = ["config.json"]) {
  const dir = join(hfCacheDir(), "models--" + repoId.replace(/\//g, "--"));
  if (!existsSync(dir)) return null;
  const snapRoot = join(dir, "snapshots");
  if (!existsSync(snapRoot)) return null;
  let revs = [];
  try { revs = readdirSync(snapRoot); } catch { return null; }
  // Prefer the revision refs/main points at; fall back to any complete snapshot.
  let preferred = null;
  try { preferred = readFileSync(join(dir, "refs", "main"), "utf8").trim(); } catch { /* no ref */ }
  const ordered = preferred && revs.includes(preferred) ? [preferred, ...revs.filter((r) => r !== preferred)] : revs;
  for (const rev of ordered) {
    const snap = join(snapRoot, rev);
    if (needFiles.every((f) => existsSync(join(snap, f)))) return snap;
  }
  return null;
}

/** Total bytes of a cached snapshot, following the blob links. */
export function dirBytes(dir) {
  let total = 0;
  const walk = (d) => {
    let entries = [];
    try { entries = readdirSync(d, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const p = join(d, e.name);
      if (e.isDirectory()) walk(p);
      else { try { total += statSync(p).size; } catch { /* dangling link */ } }
    }
  };
  walk(dir);
  return total;
}

const mb = (bytes) => Math.round(bytes / 1048576);

// ------------------------------------------------------------------ GPU ----

/** Marketing names change every generation; compute capability does not. Everything downstream
 *  (dtype, CUDA wheel index, tier feasibility) keys off the number, never off the name. */
function archOf(cc) {
  const n = Number(cc);
  if (!Number.isFinite(n)) return { name: "unknown", cc: 0 };
  if (n >= 12.0) return { name: "Blackwell", cc: n };
  if (n >= 10.0) return { name: "Blackwell (datacenter)", cc: n };
  if (n >= 9.0) return { name: "Hopper", cc: n };
  if (n >= 8.9) return { name: "Ada Lovelace", cc: n };
  if (n >= 8.0) return { name: "Ampere", cc: n };
  if (n >= 7.5) return { name: "Turing", cc: n };
  if (n >= 7.0) return { name: "Volta", cc: n };
  if (n >= 6.0) return { name: "Pascal", cc: n };
  return { name: "pre-Pascal", cc: n };
}

/** Read the real GPUs. nvidia-smi answers in milliseconds and reports compute capability
 *  directly, so this never has to import torch just to describe the machine. */
export function gpuProbe(timeoutMs = 4000) {
  return new Promise((resolve) => {
    const done = (gpus) => resolve(summarize(gpus));
    const p = spawn("nvidia-smi",
      ["--query-gpu=index,name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"],
      { windowsHide: true });
    let out = "";
    const timer = setTimeout(() => { try { p.kill(); } catch { /* gone */ } done([]); }, timeoutMs);
    p.stdout.on("data", (d) => (out += d));
    p.on("error", () => { clearTimeout(timer); done([]); });   // no NVIDIA driver at all
    p.on("close", () => {
      clearTimeout(timer);
      const gpus = out.trim().split(/\r?\n/).filter(Boolean).map((line) => {
        const [index, name, memTotal, cc, driver] = line.split(",").map((s) => s.trim());
        return { index: +index, name, vramMB: +memTotal, cc, arch: archOf(cc).name, driver };
      }).filter((g) => g.name);
      done(gpus);
    });
  });
}

function summarize(gpus) {
  if (!gpus.length) {
    return {
      gpus: [], count: 0, vramMB: 0, cc: 0, arch: "none", label: "no NVIDIA GPU detected",
      // Everything still installs; it just runs on the CPU, so say so rather than blocking.
      dtype: "float32", dtypeWhy: "no CUDA device: models run on the CPU in float32",
      cudaIndex: null, cudaWhy: "no CUDA device detected",
    };
  }
  const cc = Math.min(...gpus.map((g) => Number(g.cc) || 0));   // the weakest card sets the floor
  const vramMB = Math.max(...gpus.map((g) => g.vramMB));        // a model runs on ONE card
  const arch = archOf(cc);
  // fp16 vs bf16 is a HARDWARE question. Native bf16 tensor cores arrive with Ampere (sm_80).
  // Below that bf16 falls off the tensor-core path and runs several times slower than fp16 at
  // identical memory cost — measured 43.0 vs 7.3 TFLOP/s on Turing — so fp16 is simply correct
  // there. It is not a downgrade and it costs no quality for inference at this size.
  const bf16Native = cc >= 8.0;
  const same = gpus.every((g) => g.name === gpus[0].name);
  return {
    gpus, count: gpus.length, vramMB, cc, arch: arch.name,
    label: (gpus.length > 1 && same ? `${gpus.length}× ` : "") +
           (same ? gpus[0].name : gpus.map((g) => g.name).join(" + ")) +
           ` · ${Math.round(vramMB / 1024)} GB · ${arch.name}`,
    dtype: bf16Native ? "bfloat16" : "float16",
    dtypeWhy: bf16Native
      ? `${arch.name} (sm_${String(cc).replace(".", "")}) has native bf16 tensor cores`
      : `${arch.name} (sm_${String(cc).replace(".", "")}) has no native bf16 path: fp16 is the fast one here, at the same VRAM`,
    // PyTorch's own build table (.ci/manywheel/build_env_setup.py): cu126 builds SASS for
    // {50,60,70,75,80,86,90}, cu130 for {75,80,86,90,100,120}. Anything Turing or newer takes
    // cu130 (it also covers Blackwell); older cards have to stay on cu126, which still has them.
    cudaIndex: cc >= 7.5 ? "cu130" : "cu126",
    cudaWhy: cc >= 7.5
      ? `sm_${String(cc).replace(".", "")} is in PyTorch's cu130 build set (75–120)`
      : `sm_${String(cc).replace(".", "")} was dropped by CUDA 13: cu126 still ships it`,
  };
}

// ------------------------------------------------------------- catalog ----

/**
 * Every component. `vramMB` is what it actually costs on the GPU when loaded — measured, not
 * inferred from file size, because the two differ a lot: SAM 3 is a 3.3 GB download that peaks
 * at 1.65 GB of VRAM (the tracker and concept models share one vision tower).
 */
export function catalog(ROOT, gpu) {
  const P = paths(ROOT);
  const idx = gpu?.cudaIndex || "cu130";

  // `sizeMB` on each item is what it COSTS TO DOWNLOAD and is declared, never overwritten here.
  // What detection measures is separate (`onDiskMB`): for SAM 3 the two legitimately differ, since
  // a cache that also holds sam3.pt is bigger than what this installer would fetch.
  const found = (present, path, bytes) => (present
    ? { present: true, path, ...(bytes ? { onDiskMB: mb(bytes) } : {}) }
    : { present: false });
  const fileFound = (p) => (existsSync(p) ? found(true, p, statSync(p).size) : found(false));

  const sam3Snap = hfSnapshot("facebook/sam3", ["config.json", "model.safetensors"]);
  // Both of the places a hand-cloned model plausibly sits: beside the repo (what the original
  // code assumed) and INSIDE it, which is what `git clone .../facebook/sam3` from the repo root
  // gives you and is the more natural of the two. Checking only one of them is how a finished
  // 6.5 GB download gets reported as missing.
  const sam3Legacy = [join(P.ROOT, "sam3"), join(P.REPO, "sam3")]
    .find((d) => existsSync(join(d, "config.json")) && existsSync(join(d, "model.safetensors")));
  const liteSnap = hfSnapshot("vil-uob/sam3-litetext-s0", ["config.json"]);
  const dreamSnap = hfSnapshot("Lykon/dreamshaper-8", ["model_index.json"]);
  // SAM 3D Body ships as a gated Meta repo plus an ungated community mirror of the same weights
  // (validated 2026-07-13, and re-checked against the Hub 2026-09-08: model.ckpt 2.1 GB +
  // assets/mhr_model.pt 696 MB). Either satisfies the component, so probe both.
  // Name the WEIGHTS, not the config: model_config.yaml is 1.5 KB and lands in the first second
  // of a 2.8 GB download, so probing for it alone reports an interrupted fetch as installed.
  const SAM3D_BODY_FILES = ["model_config.yaml", "model.ckpt", "assets/mhr_model.pt"];
  const sam3dBodySnap = hfSnapshot("facebook/sam-3d-body-dinov3", SAM3D_BODY_FILES)
    || hfSnapshot("jetjodh/sam-3d-body-dinov3", SAM3D_BODY_FILES);
  const sam3dObjSnap = hfSnapshot("facebook/sam-3d-objects",
    ["checkpoints/pipeline.yaml", "checkpoints/slat_generator.ckpt", "checkpoints/ss_generator.ckpt"]);
  const gitExe = findGit(ROOT);
  const ff = findFfmpeg(ROOT);
  const enc = encoderState(ROOT);
  // The env counts as installed only once the packages the service imports are in it: a venv whose
  // pip step was interrupted has python.exe and nothing else.
  const envReady = existsSync(P.envPy)
    && ["torch", "transformers", "uvicorn"].every((m) => existsSync(join(P.envDir, "Lib", "site-packages", m)));

  const items = [
    {
      id: "git",
      group: "Runtime",
      label: "Git",
      enables: "cloning the source trees the components below install from",
      why: "MinGit 2.55.0 unpacked into tools/bin/git when the machine has no git: no administrator, no PATH edit",
      sizeMB: 37,
      ...(gitExe ? found(true, gitExe) : found(false)),
      install: {
        kind: "zip", into: P.gitDir,
        url: "https://github.com/git-for-windows/git/releases/download/v2.55.0.windows.5/MinGit-2.55.0.5-64-bit.zip",
        sha256: "56d7b226b7693196cfc71fef26568f536c4a021ab6c37ff2db4287bed908e96e",
      },
    },
    {
      id: "ffmpeg",
      group: "Runtime",
      label: "FFmpeg 9.0.1 (libvpx-vp9, libsvtav1, libopus)",
      enables: "texture video, audio mux, video probing, depth frame extraction",
      // The FULL build on purpose: gyan.dev's "essentials" build ships libaom but not libsvtav1
      // (checked on 8.0 and 9.0.1), and the AV1 texture path encodes with libsvtav1. The shared
      // variant carries the same encoders in 93 MB where the static one is 240 MB.
      why: "gyan.dev full shared build (GPL-3.0) unpacked into tools/bin/ffmpeg: no administrator, no PATH edit",
      sizeMB: 93,
      ...(ff ? found(true, ff.ffmpeg) : found(false)),
      ...(ff ? { source: ff.source } : {}),
      install: {
        kind: "zip", into: P.ffmpegDir, flatten: true,
        url: "https://github.com/GyanD/codexffmpeg/releases/download/9.0.1/ffmpeg-9.0.1-full_build-shared.zip",
        sha256: "6fd54b3b4f49117a307877b570f5e1659090f178973298658b41f5c559b5b5ab",
      },
    },
    {
      id: "python-env",
      group: "Runtime",
      label: "Python environment (PyTorch + transformers)",
      enables: "every local model: segmentation, upscaling, detail",
      why: `PyTorch built for ${idx}: ${gpu?.cudaWhy || "default CUDA build"}. CPython ${PRIVATE_PYTHON.version} is unpacked into tools/bin when the machine has no usable interpreter`,
      sizeMB: 3400,
      ...(envReady ? found(true, P.envDir) : found(false)),
      install: { kind: "python-env", cudaIndex: idx },
    },
    {
      id: "sam3",
      group: "Segmentation",
      label: "SAM 3 (facebook/sam3)",
      enables: "click-to-select and text-prompted selection in the editor, the primary backend",
      why: "the full model; what the editor uses by default",
      sizeMB: 3281,          // model.safetensors only — sam3.pt is a second copy the loader never reads
      vramMB: 1653,          // measured: tracker 914 MB + concept sharing the vision tower
      requires: ["python-env"],
      gated: { url: "https://huggingface.co/facebook/sam3", why: "Meta gates this repo: accept the licence once, with the same account your HF token belongs to" },
      ...(sam3Legacy ? found(true, sam3Legacy, dirBytes(sam3Legacy))
        : sam3Snap ? found(true, sam3Snap, dirBytes(sam3Snap))
        : found(false)),
      // A snapshot pulled with a plain `hf download facebook/sam3` also carries sam3.pt, the
      // original research checkpoint. transformers reads model.safetensors and never opens it, so
      // it is dead weight worth naming rather than leaving as unexplained disk use.
      ...(() => {
        const at = sam3Legacy || sam3Snap;
        const pt = at && join(at, "sam3.pt");
        return pt && existsSync(pt)
          ? { diskNote: `includes sam3.pt (${mb(statSync(pt).size)} MB), which the transformers loader never reads: safe to delete` }
          : {};
      })(),
      install: {
        kind: "hf", repo: "facebook/sam3",
        // sam3.pt is the original research checkpoint; the transformers loader reads
        // model.safetensors. Fetching both doubles a 3.3 GB download for nothing.
        allow: ["*.json", "*.txt", "*.safetensors"], ignore: ["sam3.pt", "*.pt"],
      },
    },
    {
      id: "sam3-lite",
      group: "Segmentation",
      label: "SAM 3 LiteText S0 (ungated)",
      enables: "text-prompted selection without Meta's licence: click-to-select still needs SAM 3 or ViT",
      why: "Apache-2.0, no sign-up; the fallback when the gated repo is not an option",
      sizeMB: 2022,
      vramMB: 1100,
      optional: true,
      requires: ["python-env"],
      ...(liteSnap ? found(true, liteSnap, dirBytes(liteSnap)) : found(false)),
      install: { kind: "hf", repo: "vil-uob/sam3-litetext-s0" },
    },
    {
      id: "vit-h",
      group: "Segmentation",
      label: "SAM ViT-H (fallback backend)",
      enables: "click-to-select when SAM 3 is unavailable",
      why: "the original SAM; runs when sam3 is missing or fails to load",
      sizeMB: 2446,
      vramMB: 2600,
      optional: true,
      requires: ["python-env"],
      ...fileFound(process.env.SAM_CKPT || join(P.models, "sam_vit_h_4b8939.pth")),
      install: { kind: "url", url: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", into: P.models, as: "sam_vit_h_4b8939.pth" },
    },
    {
      id: "vit-b",
      group: "Segmentation",
      label: "SAM ViT-B (small fallback)",
      enables: "click-to-select on a small GPU",
      why: "same job as ViT-H at a seventh of the size and a fraction of the quality",
      sizeMB: 358,
      vramMB: 900,
      optional: true,
      requires: ["python-env"],
      ...fileFound(join(P.models, "sam_vit_b_01ec64.pth")),
      install: { kind: "url", url: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", into: P.models, as: "sam_vit_b_01ec64.pth" },
    },
    {
      id: "esrgan-x4plus",
      group: "Texture enhance",
      label: "Real-ESRGAN x4plus",
      enables: "the GPU /upscale pass on atlases",
      why: "the full-quality 4× upscaler the service loads by default",
      sizeMB: 64,
      vramMB: 1000,          // tiled inference holds it near 1 GB regardless of atlas size
      requires: ["python-env"],
      ...fileFound(process.env.UPSCALE_CKPT || join(P.models, "RealESRGAN_x4plus.pth")),
      install: { kind: "url", url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", into: P.models, as: "RealESRGAN_x4plus.pth" },
    },
    {
      id: "esrgan-general",
      group: "Texture enhance",
      label: "Real-ESRGAN general x4 v3 (tiny)",
      enables: "the same upscale pass, far lighter",
      why: "4.7 MB — for machines that cannot spare the VRAM or the disk",
      sizeMB: 5,
      vramMB: 400,
      optional: true,
      requires: ["python-env"],
      ...fileFound(join(P.models, "realesr-general-x4v3.pth")),
      install: { kind: "url", url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", into: P.models, as: "realesr-general-x4v3.pth" },
    },
    {
      id: "esrgan-ncnn",
      group: "Texture enhance",
      label: "Real-ESRGAN ncnn-vulkan (CLI)",
      enables: "the Convert tab's Fast enhance tier",
      why: "standalone Vulkan binary — no Python, works on any GPU including AMD/Intel",
      sizeMB: 43,
      aliases: ["realesrgan"],   // convert.js's Fast-tier gate asks for this id
      ...(existsSync(join(P.esrganDir, "realesrgan-ncnn-vulkan.exe")) ? found(true, P.esrganDir) : found(false)),
      install: { kind: "zip", url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip", into: P.esrganDir, flatten: true },
    },
    {
      id: "dreamshaper",
      group: "Texture enhance",
      label: "DreamShaper 8 (generative detail)",
      enables: "the /detail SD 1.5 img2img pass",
      why: "fp16 weights only, the fp32 copies and the disabled safety checker are 5.8 GB the code never loads",
      sizeMB: 2034,
      vramMB: 2600,
      optional: true,
      requires: ["python-env"],
      ...(dreamSnap ? found(true, dreamSnap, dirBytes(dreamSnap)) : found(false)),
      install: {
        kind: "hf", repo: "Lykon/dreamshaper-8",
        allow: ["*.json", "*.txt", "*fp16.safetensors", "tokenizer/*", "scheduler/*"],
        ignore: ["safety_checker/*", "*.ckpt", "*.bin", "*nonema*"],
      },
    },
  ];

  // ---- status only: nothing here is downloadable, but each one gates a feature, so the tab
  // still has to be able to answer "why can't I do X". They render in a separate, quieter list.
  const fourdsDll = [join(P.ROOT, "tools", "4ds", "bin", "BridgeCodec4DS.dll"), process.env.FOURDS_DLL].filter(Boolean).find(existsSync);
  const forgeRoot = process.env.FORGE_ROOT || join(homedir(), "webui_forge");
  // ---- SAM 3D ---------------------------------------------------------------------------
  // Body is the completion prior the RGBD pipeline already fits per frame (fixed-topology MHR,
  // 18,439 vertices, identity-consistent across a take). Objects reconstructs props and sets but
  // wants ~32 GB of VRAM, so on most machines the profile maths drops it — correctly.
  const sam3d = [
    {
      id: "sam3d-body",
      group: "SAM 3D",
      label: "SAM 3D Body weights (MHR)",
      enables: "the full-body template that completes the unseen side of a capture",
      why: "2.8 GB: DINOv3-H+ checkpoint plus the Momentum Human Rig asset. Falls back to the ungated mirror, so no licence is needed.",
      sizeMB: 2800,
      vramMB: 6000,
      optional: true,
      requires: ["python-env"],
      ...(sam3dBodySnap ? found(true, sam3dBodySnap) : found(false)),
      install: { kind: "hf", repo: "jetjodh/sam-3d-body-dinov3" },
    },
    {
      id: "sam3d-body-code",
      group: "SAM 3D",
      label: "SAM 3D Body source",
      enables: "running the body model locally instead of on a rented GPU",
      why: "shallow git clone of facebookresearch/sam-3d-body into tools/ext",
      sizeMB: 80,
      optional: true,
      requires: ["git"],
      ...(existsSync(join(P.ext, "sam-3d-body", ".git")) ? found(true, join(P.ext, "sam-3d-body")) : found(false)),
      install: { kind: "git", url: "https://github.com/facebookresearch/sam-3d-body.git", into: join(P.ext, "sam-3d-body") },
    },
    {
      id: "sam3d-body-deps",
      group: "SAM 3D",
      label: "SAM 3D Body Python dependencies",
      enables: "the body model's own inference entry points",
      // Kept separate from the clone on purpose: detectron2 compiles from source and needs a C++
      // toolchain, so it is the one step here that can genuinely fail on a clean Windows box.
      // Cloning stays useful on its own, and this row is the part you opt into.
      why: "builds detectron2 from source with MSVC Build Tools (installed first when absent), about 10 minutes",
      sizeMB: 400,
      optional: true,
      requires: ["sam3d-body-code", "python-env", "msvc-build-tools"],
      ...(existsSync(join(P.envDir, "Lib", "site-packages", "detectron2")) ? found(true, join(P.envDir, "Lib", "site-packages", "detectron2")) : found(false)),
      install: { kind: "git", url: "https://github.com/facebookresearch/sam-3d-body.git", into: join(P.ext, "sam-3d-body"), pip: ["-e", "."] },
    },
    {
      id: "msvc-build-tools",
      group: "Runtime",
      label: "MSVC Build Tools",
      enables: "compiling Python packages that ship C++ (detectron2)",
      why: "C++ workload with the Windows SDK, through winget or the Microsoft bootstrapper; Windows shows one elevation prompt",
      sizeMB: 2500,
      optional: true,
      ...(findVcvars() ? found(true, findVcvars()) : found(false)),
      install: {
        kind: "winget", id: "Microsoft.VisualStudio.2022.BuildTools", verify: "vcvars",
        // Without the workload the package installs a bootstrapper and no compiler.
        override: "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended",
        bootstrapper: "https://aka.ms/vs/17/release/vs_BuildTools.exe",
      },
    },
    {
      id: "sam3d-objects",
      group: "SAM 3D",
      label: "SAM 3D Objects",
      enables: "single-image reconstruction of props and sets",
      why: "~14 GB of checkpoints and about 32 GB of VRAM: a rented-GPU component on most machines",
      sizeMB: 14000,
      vramMB: 32000,
      optional: true,
      requires: ["python-env"],
      gated: { url: "https://huggingface.co/facebook/sam-3d-objects", why: "Meta gates this repo: accept the licence once, with the account your HF token belongs to" },
      ...(sam3dObjSnap ? found(true, sam3dObjSnap) : found(false)),
      install: { kind: "hf", repo: "facebook/sam-3d-objects" },
    },
  ];

  const status = [
    {
      id: "encoder", group: "Project", label: "Encoder build (tsc output)",
      enables: "Convert tab encodes, editor Bake",
      why: enc.stale ? "sources are newer than the build: rebuilt before the next job" : "TypeScript build of packages/core and packages/encoder, rebuilt when the sources change",
      sizeMB: 0,
      ...(enc.built && !enc.stale ? found(true, join(P.ROOT, "packages", "encoder", "dist")) : found(false)),
      ...(enc.stale ? { stale: true } : {}),
      install: { kind: "npm", args: ["run", "build"], cwd: P.ROOT },
    },
    {
      id: "synth-clip", group: "Project", label: "Synthetic demo clip (demo.ares)", statusOnly: true,
      enables: "a playable clip with no capture data at all",
      why: "generated locally by the encoder",
      ...fileFound(join(P.ROOT, "apps", "demo", "demo.ares")),
      install: { kind: "route", route: "/setup/demo-clip", label: "Generate" },
    },
    {
      id: "keeper-clip", group: "Project", label: "Reference clip (daniel-s0.ares)", statusOnly: true, optional: true,
      enables: "the Viewer's default source and the Compare presets",
      why: "re-encode it from the capture folder in the Convert tab",
      ...fileFound(join(P.ROOT, "apps", "demo", "daniel-s0.ares")),
    },
    {
      id: "capture", group: "Project", label: "Source capture frames", statusOnly: true, optional: true,
      enables: "re-encoding, editor Bake, enhance experiments",
      why: "your own data: any per-frame OBJ/PLY + atlas PNG folder works",
      ...(existsSync(join(P.REPO, "Daniel_Microsoft_Volcap", "Daniel_Volcap")) ? found(true, join(P.REPO, "Daniel_Microsoft_Volcap", "Daniel_Volcap")) : found(false)),
    },
    {
      id: "4ds-codec", group: "External", label: "4DViews codec (BridgeCodec4DS.dll)", statusOnly: true, optional: true,
      enables: "Convert tab's .4ds → .ares conversion",
      why: "licensed 4DViews SDK file: located once with the file dialog in the Convert tab and copied into tools/4ds/bin",
      requires: ["python-env"],
      ...(fourdsDll ? found(true, fourdsDll, statSync(fourdsDll).size) : found(false)),
    },
    {
      id: "forge", group: "External", label: "SD-Forge (generative enhance)", optional: true,
      enables: "the Convert tab's generative img2img tier",
      why: "cloned into tools/ext/webui_forge; the first start builds its own virtual environment on CPython 3.10",
      sizeMB: 120,
      requires: ["git"],
      ...(existsSync(join(forgeRoot, "system", "python", "python.exe")) ? found(true, forgeRoot)
        : existsSync(join(P.ext, "webui_forge", ".git")) ? found(true, join(P.ext, "webui_forge")) : found(false)),
      install: { kind: "git", url: "https://github.com/lllyasviel/stable-diffusion-webui-forge.git", into: join(P.ext, "webui_forge") },
    },
    {
      id: "sam31", group: "External", label: "SAM 3.1 checkpoint", statusOnly: true, optional: true,
      enables: "nothing yet",
      // Re-checked 2026-09-08: facebook/sam3.1 is still published as library `checkpoint`, and
      // transformers 5.16 ships sam3 / sam3_tracker / sam3_lite_text but no sam3_1. Still parked.
      why: "no code path loads it: transformers has no SAM 3.1 architecture yet",
      ...fileFound(join(P.REPO, "sam3.1", "sam3.1_multiplex.pt")),
    },
  ];

  return [...items, ...sam3d, ...status];
}

// ------------------------------------------------------------ profiles ----

/**
 * The three one-click tiers. Membership is decided by what the detected hardware can actually
 * hold: `vramMB` is compared against the largest single card, because a model runs on one GPU.
 * A tier never silently swaps in a smaller model — if something does not fit it is dropped and
 * the reason is reported, so nobody is quietly given a lesser model than their box can run.
 */
export function profiles(items, gpu) {
  const byId = Object.fromEntries(items.map((i) => [i.id, i]));
  const vram = gpu?.vramMB || 0;
  const fits = (id) => {
    const it = byId[id];
    if (!it) return false;
    if (!it.vramMB) return true;                 // CPU-side or negligible
    if (!vram) return it.vramMB <= 1200;         // CPU-only box: only the light things are sane
    return it.vramMB + 800 <= vram;              // 800 MB headroom for activations + the viewer
  };

  const defs = [
    {
      id: "best", label: "Best",
      blurb: "Everything, at full quality. Pick this when the GPU has the room.",
      want: ["python-env", "sam3", "vit-h", "esrgan-x4plus", "esrgan-ncnn", "dreamshaper"],
    },
    {
      id: "balanced", label: "Balanced",
      blurb: "The full segmentation and upscale models, without the generative extras.",
      want: ["python-env", "sam3", "esrgan-x4plus", "esrgan-ncnn"],
    },
    {
      id: "smallest", label: "Smallest",
      blurb: "Ungated and light, no licence to accept, least disk and VRAM.",
      want: ["python-env", "sam3-lite", "esrgan-general", "esrgan-ncnn"],
    },
  ];

  return defs.map((d) => {
    const included = d.want.filter(fits);
    const dropped = d.want.filter((id) => !fits(id)).map((id) => ({ id, label: byId[id]?.label, needMB: byId[id]?.vramMB }));
    const missing = included.filter((id) => !byId[id]?.present);
    const downloadMB = missing.reduce((s, id) => s + (byId[id]?.sizeMB || 0), 0);
    const peakVramMB = Math.max(0, ...included.map((id) => byId[id]?.vramMB || 0));
    return { ...d, included, dropped, missing, downloadMB, peakVramMB, complete: missing.length === 0 };
  });
}

/** Which tier to put the badge on: the richest one whose components all fit this machine. */
export function recommend(profs) {
  return (profs.find((p) => p.id === "best" && !p.dropped.length)
    || profs.find((p) => p.id === "balanced" && !p.dropped.length)
    || profs[2]).id;
}

/** Transitive `requires`, de-duplicated, dependencies before dependents. */
export function resolve(items, wanted) {
  const byId = Object.fromEntries(items.map((i) => [i.id, i]));
  const out = [], seen = new Set(), visiting = new Set();
  const visit = (id) => {
    if (seen.has(id) || visiting.has(id)) return;   // visiting-guard: a cycle stops, never hangs
    const it = byId[id];
    if (!it) return;
    visiting.add(id);
    for (const dep of it.requires || []) visit(dep);
    visiting.delete(id);
    seen.add(id);
    out.push(it);
  };
  for (const id of wanted) visit(id);
  return out;
}

// ------------------------------------------------------------ installers ----

const sh = (cmd, args, opts, onLine, stdin) => new Promise((done) => {
  const p = spawn(cmd, args, { windowsHide: true, ...opts });
  const relay = (d) => String(d).split(/\r?\n/).forEach((l) => l.trim() && onLine(l.trim().slice(0, 300)));
  p.stdout?.on("data", relay);
  p.stderr?.on("data", relay);
  p.on("error", (e) => { onLine("ERROR: " + e.message); done(1); });
  p.on("close", (c) => done(c ?? 1));
  // Secrets go in on stdin, never in argv — argv is visible to every process on the machine.
  if (stdin != null) { try { p.stdin.write(stdin); p.stdin.end(); } catch { /* child already gone */ } }
  return p;
});

/** Run a command and return { code, lines } without relaying anything. */
const capture = async (cmd, args, opts = {}) => {
  const lines = [];
  const code = await sh(cmd, args, opts, (l) => lines.push(l));
  return { code, lines };
};

/**
 * The CPython this installer unpacks when the machine has no usable interpreter. It is the
 * official python.org build as published on nuget.org: a plain zip with the full standard library
 * (venv and ensurepip included, which the "embeddable" zip lacks), so it needs no installer run,
 * no administrator and no PATH edit. 3.10 exists only for SD-Forge, whose pinned wheels stop there.
 */
export const PRIVATE_PYTHON = {
  version: "3.12.10", dir: "python-3.12", sizeMB: 14,
  url: "https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg",
  sha256: "0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8",
};
export const FORGE_PYTHON = {
  version: "3.10.11", dir: "python-3.10", sizeMB: 16,
  url: "https://api.nuget.org/v3-flatcontainer/python/3.10.11/python.3.10.11.nupkg",
  sha256: "7c6f99b160a36a7e09492dfcff2b0a3a60bb5229ca44cdcc3ecb32871a6144d0",
};
const privatePythonExe = (ROOT, spec) => join(paths(ROOT).bin, spec.dir, "tools", "python.exe");

/** Unpack a private CPython under tools/bin when it is not already there. Resolves with the
 *  interpreter path, or { error }. */
export async function ensurePrivatePython(ROOT, onLine, spec = PRIVATE_PYTHON) {
  const exe = privatePythonExe(ROOT, spec);
  if (existsSync(exe)) return { exe };
  onLine(`CPython ${spec.version}: fetching the python.org build from nuget.org (${spec.sizeMB} MB)`);
  const r = await unpackZip(ROOT, { url: spec.url, sha256: spec.sha256, into: join(paths(ROOT).bin, spec.dir), tag: spec.dir }, onLine);
  if (!r.ok) return { error: `CPython ${spec.version}: ${r.error}` };
  const v = await capture(exe, ["--version"]);
  if (v.code !== 0) return { error: `CPython ${spec.version}: unpacked interpreter does not run (exit ${v.code})` };
  onLine(`CPython ${spec.version} ready: ${exe}`);
  return { exe };
}

/** Find an interpreter to build the venv with. `py -3` is the Windows launcher and is the most
 *  reliable when several Pythons are installed; plain `python` on Windows may be the Store stub,
 *  which exits non-zero here and is skipped. Only 3.11 to 3.13 with a working venv module is
 *  accepted, because those are the versions every pinned wheel in requirements.txt ships for;
 *  anything else falls through to the private interpreter. */
async function findPython(ROOT, onLine) {
  const probe = "import sys,venv,ensurepip;print(sys.executable);print('%d.%d'%sys.version_info[:2])";
  const tries = [["py", ["-3"]], ["python", []], ["python3", []], [privatePythonExe(ROOT, PRIVATE_PYTHON), []]];
  for (const [cmd, prefix] of tries) {
    if (/[\\/]/.test(cmd) && !existsSync(cmd)) continue;
    const r = await capture(cmd, [...prefix, "-c", probe]);
    if (r.code !== 0 || r.lines.length < 2) continue;
    const [exe, ver] = r.lines.slice(-2);
    const minor = Number((/^3\.(\d+)$/.exec(ver) || [])[1]);
    if (!(minor >= 11 && minor <= 13)) { onLine(`${cmd}: CPython ${ver} is outside 3.11 to 3.13, skipped`); continue; }
    onLine(`interpreter: ${exe} (CPython ${ver})`);
    return { cmd, prefix, exe, version: ver };
  }
  return null;
}

async function installPythonEnv(ROOT, item, onLine) {
  const P = paths(ROOT);
  const idx = item.install.cudaIndex;
  if (!existsSync(P.envPy)) {
    let py = await findPython(ROOT, onLine);
    if (!py) {
      onLine("no usable CPython on this machine: unpacking a private one");
      const priv = await ensurePrivatePython(ROOT, onLine);
      if (priv.error) return { ok: false, error: priv.error };
      py = { cmd: priv.exe, prefix: [] };
    }
    onLine("creating the virtual environment…");
    const code = await sh(py.cmd, [...py.prefix, "-m", "venv", P.envDir], { cwd: P.svc }, onLine);
    if (code !== 0) return { ok: false, error: "venv creation failed (exit " + code + ")" };
  } else onLine("virtual environment present: completing its packages");

  onLine("upgrading pip…");
  await sh(P.envPy, ["-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check"], { cwd: P.svc }, onLine);
  onLine(`installing PyTorch (${idx}) and the service packages: several minutes…`);
  const code = await sh(P.envPy, ["-m", "pip", "install", "-r", P.reqs, "--extra-index-url", `https://download.pytorch.org/whl/${idx}`], { cwd: P.svc }, onLine);
  if (code !== 0) return { ok: false, error: "pip install failed (exit " + code + ")" };
  return { ok: true };
}

/** Download through huggingface_hub so the stored token, resume, and xet transfer all apply,
 *  and the files land in the shared cache where every other tool already looks for them. */
async function installHf(ROOT, item, onLine) {
  const P = paths(ROOT);
  if (!existsSync(P.envPy)) return { ok: false, error: "Python environment absent (dependency order error)" };
  const { repo, allow, ignore } = item.install;
  // A JS null stringifies to `null`, which is a NameError in Python — absent patterns must be None.
  const pyList = (a) => (a && a.length ? JSON.stringify(a) : "None");
  const code = await sh(P.envPy, ["-c", `
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
try:
    p = snapshot_download(${JSON.stringify(repo)},
        allow_patterns=${pyList(allow)},
        ignore_patterns=${pyList(ignore)})
    print("SNAPSHOT " + p)
except GatedRepoError:
    print("GATED"); sys.exit(3)
except HfHubHTTPError as e:
    code = getattr(getattr(e, "response", None), "status_code", None)
    print("HTTP %s: %s" % (code, e)); sys.exit(4 if code in (401, 403) else 1)
`.trim()], { cwd: P.svc }, onLine);
  if (code === 3 || code === 4) {
    // The caller turns this into the in-app access prompt (token field + licence link) and
    // resumes the same job afterwards, so the message only names the state.
    const needsToken = !hfTokenPresent();
    return {
      ok: false, gated: true, needsToken,
      error: needsToken ? `${repo}: gated repository, access token required` : `${repo}: gated repository, licence not accepted for the stored token`,
    };
  }
  if (code !== 0) return { ok: false, error: `download failed (exit ${code})` };
  return { ok: true };
}

/** Stream a URL to disk. Writes to .part and renames on success, so an interrupted download can
 *  never be mistaken for an installed component by the detection above. A pinned `sha256` is
 *  checked before the rename: a binary that does not match its pin never reaches its final path. */
async function download(url, dest, onLine, sha256) {
  await mkdir(dirname(dest), { recursive: true });
  const part = dest + ".part";
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  let seen = 0, lastPct = -1;
  const hash = sha256 ? createHash("sha256") : null;
  const out = createWriteStream(part);
  for await (const chunk of res.body) {
    seen += chunk.length;
    if (hash) hash.update(chunk);
    if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
    if (total) {
      const pct = Math.floor((seen / total) * 100 / 5) * 5;
      if (pct !== lastPct) { lastPct = pct; onLine(`  ${pct}%  (${mb(seen)} / ${mb(total)} MB)`); }
    }
  }
  await new Promise((r) => out.end(r));
  if (hash) {
    const got = hash.digest("hex");
    if (got !== sha256.toLowerCase()) {
      try { await unlink(part); } catch { /* already gone */ }
      throw new Error(`sha256 mismatch for ${url}: expected ${sha256}, got ${got}`);
    }
    onLine("  sha256 verified");
  }
  try { await unlink(dest); } catch { /* not there */ }
  await rename(part, dest);
  return seen;
}

async function installUrl(ROOT, item, onLine) {
  const { url, into, as, sha256 } = item.install;
  const dest = join(into, as);
  onLine(`downloading ${as} (${item.sizeMB} MB)…`);
  try { const n = await download(url, dest, onLine, sha256); onLine(`saved ${mb(n)} MB → ${dest}`); return { ok: true }; }
  catch (e) { return { ok: false, error: String(e.message || e) }; }
}

/**
 * Download a zip and unpack it into `into`. The archive is extracted beside the target and
 * swapped in only when complete, so detection never sees a half-unpacked tool and a re-install
 * replaces a broken one. bsdtar (System32\tar.exe, Windows 10 1803+) reads zip and is many times
 * faster than Expand-Archive, which stays as the fallback.
 */
async function unpackZip(ROOT, { url, into, flatten, sha256, tag = "zip" }, onLine) {
  const P = paths(ROOT);
  const stamp = `${tag}-${Date.now().toString(36)}`;
  const tmp = join(P.bin, `_dl-${stamp}.zip`);
  const staging = into + ".new-" + stamp;
  const drop = async (p) => { try { await rm(p, { recursive: true, force: true }); } catch { /* best effort */ } };
  try { await download(url, tmp, onLine, sha256); } catch (e) { return { ok: false, error: String(e.message || e) }; }
  await mkdir(staging, { recursive: true });
  onLine("extracting…");
  const sysTar = join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe");
  let code = existsSync(sysTar) ? await sh(sysTar, ["-xf", tmp, "-C", staging], {}, onLine) : 1;
  if (code !== 0) {
    code = await sh("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
      `Expand-Archive -LiteralPath '${tmp}' -DestinationPath '${staging}' -Force`], {}, onLine);
  }
  await drop(tmp);
  if (code !== 0) { await drop(staging); return { ok: false, error: "extract failed (exit " + code + ")" }; }
  let from = staging;
  if (flatten) {
    // Some releases wrap everything in one top-level folder; use its contents as the root so the
    // exe path is stable across versions.
    try {
      const entries = await readdir(staging, { withFileTypes: true });
      const dirs = entries.filter((e) => e.isDirectory());
      if (dirs.length === 1 && !entries.some((e) => e.isFile() && e.name.endsWith(".exe"))) {
        from = join(staging, dirs[0].name);
        onLine(`root folder: ${dirs[0].name}/`);
      }
    } catch { /* layout was already flat */ }
  }
  const old = into + ".old-" + stamp;
  try {
    if (existsSync(into)) await rename(into, old);
    await rename(from, into);
  } catch (e) {
    if (existsSync(old) && !existsSync(into)) { try { await rename(old, into); } catch { /* leave .old in place */ } }
    await drop(staging);
    return { ok: false, error: `could not move the unpacked files into ${into}: ${e.code || e.message}` };
  }
  await drop(old);
  if (from !== staging) await drop(staging);
  return { ok: true };
}

async function installZip(ROOT, item, onLine) {
  onLine(`downloading (${item.sizeMB} MB)…`);
  return unpackZip(ROOT, { ...item.install, tag: item.id }, onLine);
}

/** Re-read the machine+user PATH from the registry.
 *  winget puts a newly installed tool on the MACHINE PATH, but this process captured PATH when
 *  the server started, so without this the very next step still cannot see the tool. */
async function refreshPath(onLine) {
  let out = "";
  const code = await sh("powershell.exe", ["-NoProfile", "-Command",
    "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"],
    {}, (l) => { out += l; });
  if (code === 0 && out.trim().length > 10) {
    process.env.PATH = out.trim();
    onLine("PATH refreshed from the registry");
  }
}

/** Install a system tool with winget, or with the vendor bootstrapper when winget is absent or
 *  did not produce the tool. Success is judged by PROBING afterwards, never by an exit code:
 *  winget reports a non-zero "no applicable update" when the package is already present. */
async function installWinget(ROOT, item, onLine) {
  const { id, probe, verify, override, bootstrapper } = item.install;
  const verified = () => (verify === "vcvars" ? !!findVcvars() : probe ? !!whichSync(probe) : null);
  if (verified() === true) { onLine(`${item.label}: already present`); return { ok: true }; }
  let wingetCode = null;
  if (whichSync("winget")) {
    onLine(`winget install ${id} …`);
    const args = ["install", "--id", id, "-e", "--source", "winget",
      "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity"];
    if (override) args.push("--override", override);
    wingetCode = await sh("winget", args, {}, onLine);
    await refreshPath(onLine);
    if (verified() !== false) { onLine(`${id} installed`); return { ok: true }; }
  } else onLine("winget absent on this machine");
  if (bootstrapper) {
    const exe = join(paths(ROOT).bin, `_${item.id}-setup.exe`);
    onLine(`${item.label}: running the vendor bootstrapper`);
    try { await download(bootstrapper, exe, onLine); } catch (e) { return { ok: false, error: String(e.message || e) }; }
    const code = await sh(exe, (override || "").split(/\s+/).filter(Boolean), {}, onLine);
    try { await unlink(exe); } catch { /* still locked: left for the next run to overwrite */ }
    await refreshPath(onLine);
    // 3010 = installed, reboot pending. The compiler is on disk either way.
    if (verified() !== false) { onLine(`${item.label} installed${code === 3010 ? " (Windows restart pending)" : ""}`); return { ok: true }; }
    return { ok: false, error: `${item.label}: bootstrapper exit ${code}, component not detected afterwards` };
  }
  return { ok: false, error: `${id}: not detected after installation${wingetCode === null ? " (winget absent, no bootstrapper defined)" : ` (winget exit ${wingetCode})`}` };
}

/** Shallow-clone a source tree, optionally installing it into the Python environment.
 *  A repo that is already cloned is fetched rather than re-cloned, so a re-run is cheap. */
async function installGit(ROOT, item, onLine) {
  const P = paths(ROOT);
  const { url, into, ref, recurse, pip } = item.install;
  const git = findGit(ROOT);
  if (!git) return { ok: false, error: "git absent (dependency order error)" };
  if (existsSync(join(into, ".git"))) {
    onLine("already cloned: fetching the latest commit…");
    await sh(git, ["-C", into, "fetch", "--depth", "1", "origin"], {}, onLine);
    await sh(git, ["-C", into, "reset", "--hard", "FETCH_HEAD"], {}, onLine);
  } else {
    await mkdir(dirname(into), { recursive: true });
    const args = ["clone", "--depth", "1"];
    if (ref) args.push("--branch", ref);
    if (recurse) args.push("--recurse-submodules", "--shallow-submodules");
    args.push(url, into);
    onLine(`git clone ${url} …`);
    const code = await sh(git, args, {}, onLine);
    if (code !== 0) return { ok: false, error: `git clone failed (exit ${code})` };
  }
  if (pip) {
    if (!existsSync(P.envPy)) return { ok: false, error: "Python environment absent (dependency order error)" };
    onLine("installing it into the Python environment…");
    // A source build has to find the private git and the compiler without any PATH edit.
    const env = { ...process.env, PATH: dirname(git) + ";" + (process.env.PATH || "") };
    const code = await sh(P.envPy, ["-m", "pip", "install", ...pip], { cwd: into, env }, onLine);
    if (code !== 0) return { ok: false, error: `pip install failed (exit ${code})` };
  }
  return { ok: true };
}

/** The TypeScript build (`npm run build` = `tsc -b`). tsc runs straight through this node
 *  process, so npm is only needed when node_modules is missing, and then it is the npm that
 *  ships beside node. A failed rebuild keeps an existing dist in use: another engineer's
 *  half-written source must not take the Convert tab down. */
async function installNpm(ROOT, item, onLine) {
  const P = paths(ROOT);
  const cwd = item.install.cwd || ROOT;
  const tsc = join(cwd, "node_modules", "typescript", "bin", "tsc");
  if (!existsSync(tsc)) {
    const npm = findNpm();
    if (!npm) return { ok: false, error: "npm not found beside node.exe: node_modules cannot be restored" };
    onLine("node_modules absent: npm install …");
    const code = await sh(npm.cmd, [...npm.prefix, "install", "--no-audit", "--no-fund"], { cwd }, onLine);
    if (code !== 0 || !existsSync(tsc)) return { ok: false, error: `npm install failed (exit ${code})` };
  }
  onLine("tsc -b …");
  const code = await sh(process.execPath, [tsc, "-b"], { cwd }, onLine);
  try { await mkdir(P.bin, { recursive: true }); await writeFile(P.encoderStamp, new Date().toISOString()); } catch { /* stamp is an optimisation */ }
  if (code === 0) return { ok: true };
  if (encoderState(ROOT).built) { onLine(`tsc exit ${code}: the existing build stays in use`); return { ok: true, warning: `tsc exit ${code}` }; }
  return { ok: false, error: `encoder build failed (tsc exit ${code})` };
}

export async function installOne(ROOT, item, onLine) {
  switch (item.install?.kind) {
    case "python-env": return installPythonEnv(ROOT, item, onLine);
    case "hf": return installHf(ROOT, item, onLine);
    case "url": return installUrl(ROOT, item, onLine);
    case "zip": return installZip(ROOT, item, onLine);
    case "winget": return installWinget(ROOT, item, onLine);
    case "git": return installGit(ROOT, item, onLine);
    case "npm": return installNpm(ROOT, item, onLine);
    default: return { ok: false, error: "no installer for " + item.id };
  }
}

/** Store a Hugging Face token the way `hf auth login` does, so every downloader picks it up.
 *  Written through huggingface_hub when the venv exists (it validates the token and writes the
 *  same file the library reads); otherwise straight to the token file. The token is never
 *  logged, echoed, or returned. */
export async function saveHfToken(ROOT, token, onLine = () => {}) {
  const t = String(token || "").trim();
  if (!/^hf_[A-Za-z0-9]{20,}$/.test(t)) {
    return { ok: false, error: "invalid token format (expected hf_…)" };
  }
  const P = paths(ROOT);
  if (existsSync(P.envPy)) {
    let who = "", failed = false;
    const code = await sh(P.envPy, ["-c",
      "import sys;from huggingface_hub import HfApi,login;t=sys.stdin.read().strip();"
      + "u=HfApi().whoami(token=t);login(token=t,add_to_git_credential=False);print('USER '+u.get('name','?'))"],
      { stdio: ["pipe", "pipe", "pipe"] }, (l) => { if (l.startsWith("USER ")) who = l.slice(5); else if (/error|Error|Invalid/.test(l)) failed = true; },
      t + "\n");
    if (code === 0 && who) { onLine(`signed in as ${who}`); return { ok: true, user: who }; }
    if (failed || code !== 0) return { ok: false, error: "token rejected by Hugging Face" };
  }
  const file = hfTokenFile();
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, t, "utf8");
  return { ok: true, user: null };
}

// ----------------------------------------------------------- preflight ----

/** The things that block an install before it starts, answered once so the UI can say what is
 *  wrong instead of failing halfway through a 3 GB download. */
export async function preflight(ROOT) {
  const P = paths(ROOT);
  const out = { python: null, envPresent: existsSync(P.envPy), hfToken: false, hfUser: null };
  const tokenFile = process.env.HF_TOKEN ? null : join(hfCacheDir(), "..", "token");
  out.hfToken = !!process.env.HF_TOKEN || existsSync(tokenFile);
  if (out.hfToken && !process.env.HF_TOKEN) {
    try { out.hfUser = (await readFile(tokenFile, "utf8")).trim() ? "stored token" : null; } catch { /* unreadable */ }
  }
  const py = await findPython(ROOT, () => {});
  out.python = py ? py.cmd : null;
  return out;
}
