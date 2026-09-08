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
import { spawn } from "node:child_process";
import { createWriteStream, existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { mkdir, readdir, readFile, rename, unlink } from "node:fs/promises";
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
  };
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
      dtype: "float32", dtypeWhy: "no CUDA device — models run on the CPU in float32",
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
      : `${arch.name} (sm_${String(cc).replace(".", "")}) has no native bf16 path — fp16 is the fast one here, at the same VRAM`,
    // PyTorch's own build table (.ci/manywheel/build_env_setup.py): cu126 builds SASS for
    // {50,60,70,75,80,86,90}, cu130 for {75,80,86,90,100,120}. Anything Turing or newer takes
    // cu130 (it also covers Blackwell); older cards have to stay on cu126, which still has them.
    cudaIndex: cc >= 7.5 ? "cu130" : "cu126",
    cudaWhy: cc >= 7.5
      ? `sm_${String(cc).replace(".", "")} is in PyTorch's cu130 build set (75–120)`
      : `sm_${String(cc).replace(".", "")} was dropped by CUDA 13 — cu126 still ships it`,
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

  const items = [
    {
      id: "python-env",
      group: "Runtime",
      label: "Python environment (PyTorch + transformers)",
      enables: "every local model: segmentation, upscaling, detail",
      why: `PyTorch built for ${idx} — ${gpu?.cudaWhy || "default CUDA build"}`,
      sizeMB: 3400,
      ...(existsSync(P.envPy) ? found(true, P.envDir) : found(false)),
      install: { kind: "python-env", cudaIndex: idx },
    },
    {
      id: "sam3",
      group: "Segmentation",
      label: "SAM 3 (facebook/sam3)",
      enables: "click-to-select and text-prompted selection in the editor — the primary backend",
      why: "the full model; what the editor uses by default",
      sizeMB: 3281,          // model.safetensors only — sam3.pt is a second copy the loader never reads
      vramMB: 1653,          // measured: tracker 914 MB + concept sharing the vision tower
      requires: ["python-env"],
      gated: { url: "https://huggingface.co/facebook/sam3", why: "Meta gates this repo — accept the licence once, with the same account your HF token belongs to" },
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
          ? { diskNote: `includes sam3.pt (${mb(statSync(pt).size)} MB), which the transformers loader never reads — safe to delete` }
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
      enables: "text-prompted selection without Meta's licence — click-to-select still needs SAM 3 or ViT",
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
      why: "fp16 weights only — the fp32 copies and the disabled safety checker are 5.8 GB the code never loads",
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
  const status = [
    {
      id: "encoder", group: "Project", label: "Encoder build (tsc output)", statusOnly: true,
      enables: "Convert tab encodes, editor Bake",
      why: "rebuilt by `npm run build` — seconds, no download",
      ...(existsSync(join(P.ROOT, "packages", "encoder", "dist", "cli.js")) ? found(true, join(P.ROOT, "packages", "encoder", "dist")) : found(false)),
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
      why: "your own data — any per-frame OBJ/PLY + atlas PNG folder works",
      ...(existsSync(join(P.REPO, "Daniel_Microsoft_Volcap", "Daniel_Volcap")) ? found(true, join(P.REPO, "Daniel_Microsoft_Volcap", "Daniel_Volcap")) : found(false)),
    },
    {
      id: "4ds-codec", group: "External", label: "4DViews codec (BridgeCodec4DS.dll)", statusOnly: true, optional: true,
      enables: "Convert tab's .4ds → .ares conversion",
      why: "licensed SDK — copy your own DLL into tools/4ds/bin/; no installer can fetch it",
      requires: ["python-env"],
      ...(fourdsDll ? found(true, fourdsDll, statSync(fourdsDll).size) : found(false)),
    },
    {
      id: "forge", group: "External", label: "SD-Forge (generative enhance)", statusOnly: true, optional: true,
      enables: "the Convert tab's Generative img2img tier",
      why: "a separate application — set FORGE_ROOT if it lives elsewhere",
      ...(existsSync(join(forgeRoot, "system", "python", "python.exe")) ? found(true, forgeRoot) : found(false)),
      link: "https://github.com/lllyasviel/stable-diffusion-webui-forge",
    },
    {
      id: "sam31", group: "External", label: "SAM 3.1 checkpoint", statusOnly: true, optional: true,
      enables: "nothing yet",
      // Re-checked 2026-09-08: facebook/sam3.1 is still published as library `checkpoint`, and
      // transformers 5.16 ships sam3 / sam3_tracker / sam3_lite_text but no sam3_1. Still parked.
      why: "no code path loads it — transformers has no SAM 3.1 architecture yet",
      ...fileFound(join(P.REPO, "sam3.1", "sam3.1_multiplex.pt")),
      link: "https://huggingface.co/facebook/sam3.1",
    },
  ];

  return [...items, ...status];
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
      blurb: "Ungated and light — no licence to accept, least disk and VRAM.",
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

const sh = (cmd, args, opts, onLine) => new Promise((done) => {
  const p = spawn(cmd, args, { windowsHide: true, ...opts });
  const relay = (d) => String(d).split(/\r?\n/).forEach((l) => l.trim() && onLine(l.trim().slice(0, 300)));
  p.stdout?.on("data", relay);
  p.stderr?.on("data", relay);
  p.on("error", (e) => { onLine("ERROR: " + e.message); done(1); });
  p.on("close", (c) => done(c ?? 1));
  return p;
});

/** Find an interpreter to build the venv with. `py -3` is the Windows launcher and is the most
 *  reliable when several Pythons are installed; plain `python` on Windows may be the Store stub. */
async function findPython(onLine) {
  for (const [cmd, args] of [["py", ["-3", "-c", "import sys;print(sys.executable)"]], ["python", ["-c", "import sys;print(sys.executable)"]], ["python3", ["-c", "import sys;print(sys.executable)"]]]) {
    let out = "";
    const code = await sh(cmd, args, {}, (l) => { out += l; });
    if (code === 0 && out.trim()) { onLine(`using ${cmd} → ${out.trim()}`); return { cmd, prefix: args.slice(0, args.length - 2) }; }
  }
  return null;
}

async function installPythonEnv(ROOT, item, onLine) {
  const P = paths(ROOT);
  const idx = item.install.cudaIndex;
  if (!existsSync(P.envPy)) {
    const py = await findPython(onLine);
    if (!py) {
      onLine("no Python found on PATH.");
      return { ok: false, error: "Python 3.11+ is required and was not found. Install it (winget install Python.Python.3.13) and run this again." };
    }
    onLine("creating the virtual environment…");
    const code = await sh(py.cmd, [...py.prefix, "-m", "venv", P.envDir], { cwd: P.svc }, onLine);
    if (code !== 0) return { ok: false, error: "venv creation failed (exit " + code + ")" };
  } else onLine("virtual environment already present — installing into it");

  onLine("upgrading pip…");
  await sh(P.envPy, ["-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check"], { cwd: P.svc }, onLine);
  onLine(`installing PyTorch (${idx}) and the rest — this is the long one, several minutes…`);
  const code = await sh(P.envPy, ["-m", "pip", "install", "-r", P.reqs, "--extra-index-url", `https://download.pytorch.org/whl/${idx}`], { cwd: P.svc }, onLine);
  if (code !== 0) return { ok: false, error: "pip install failed (exit " + code + ")" };
  return { ok: true };
}

/** Download through huggingface_hub so the stored token, resume, and xet transfer all apply,
 *  and the files land in the shared cache where every other tool already looks for them. */
async function installHf(ROOT, item, onLine) {
  const P = paths(ROOT);
  if (!existsSync(P.envPy)) return { ok: false, error: "the Python environment must be installed first" };
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
    return { ok: false, gated: true, error: `${repo} is gated. Click "Accept licence", approve it with the account your HF token belongs to, then run this again.` };
  }
  if (code !== 0) return { ok: false, error: `download failed (exit ${code})` };
  return { ok: true };
}

/** Stream a URL to disk. Writes to .part and renames on success, so an interrupted download can
 *  never be mistaken for an installed component by the detection above. */
async function download(url, dest, onLine) {
  await mkdir(dirname(dest), { recursive: true });
  const part = dest + ".part";
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  let seen = 0, lastPct = -1;
  const out = createWriteStream(part);
  for await (const chunk of res.body) {
    seen += chunk.length;
    if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
    if (total) {
      const pct = Math.floor((seen / total) * 100 / 5) * 5;
      if (pct !== lastPct) { lastPct = pct; onLine(`  ${pct}%  (${mb(seen)} / ${mb(total)} MB)`); }
    }
  }
  await new Promise((r) => out.end(r));
  try { await unlink(dest); } catch { /* not there */ }
  await rename(part, dest);
  return seen;
}

async function installUrl(ROOT, item, onLine) {
  const { url, into, as } = item.install;
  const dest = join(into, as);
  onLine(`downloading ${as} (${item.sizeMB} MB)…`);
  try { const n = await download(url, dest, onLine); onLine(`saved ${mb(n)} MB → ${dest}`); return { ok: true }; }
  catch (e) { return { ok: false, error: String(e.message || e) }; }
}

async function installZip(ROOT, item, onLine) {
  const { url, into, flatten } = item.install;
  const tmp = join(paths(ROOT).bin, "_download.zip");
  onLine(`downloading (${item.sizeMB} MB)…`);
  try { await download(url, tmp, onLine); } catch (e) { return { ok: false, error: String(e.message || e) }; }
  await mkdir(into, { recursive: true });
  onLine("extracting…");
  // Expand-Archive is present on every supported Windows and needs no bundled unzip.
  const code = await sh("powershell.exe",
    ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
     `Expand-Archive -LiteralPath '${tmp}' -DestinationPath '${into}' -Force`], {}, onLine);
  try { await unlink(tmp); } catch { /* leave it */ }
  if (code !== 0) return { ok: false, error: "extract failed (exit " + code + ")" };
  if (flatten) {
    // Some releases wrap everything in one top-level folder; hoist it so the exe path is stable.
    try {
      const entries = await readdir(into, { withFileTypes: true });
      const dirs = entries.filter((e) => e.isDirectory());
      const hasExe = entries.some((e) => e.isFile() && e.name.endsWith(".exe"));
      if (!hasExe && dirs.length === 1) {
        const inner = join(into, dirs[0].name);
        for (const f of await readdir(inner)) await rename(join(inner, f), join(into, f));
        onLine(`flattened ${dirs[0].name}/`);
      }
    } catch { /* layout was already flat */ }
  }
  return { ok: true };
}

export async function installOne(ROOT, item, onLine) {
  switch (item.install?.kind) {
    case "python-env": return installPythonEnv(ROOT, item, onLine);
    case "hf": return installHf(ROOT, item, onLine);
    case "url": return installUrl(ROOT, item, onLine);
    case "zip": return installZip(ROOT, item, onLine);
    default: return { ok: false, error: "no installer for " + item.id };
  }
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
  const py = await findPython(() => {});
  out.python = py ? py.cmd : null;
  return out;
}
