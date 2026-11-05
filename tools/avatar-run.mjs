/**
 * 2D video → one volumetric clip per person, driven from the Convert tab.
 *
 * The chain, per run:
 *   mask      the depth service's SAM 3 subject pass over the whole clip, which writes one tracked
 *             id per person per pixel (mask-ids.u8, tools/sam-service/depth.py)
 *   people    avatar.py refs + tracks: each person's best reference frame cut out at source
 *             resolution, and their own fixed 9:16 clip of `frames` frames
 *   generate  4DAnyone (tools/ext/4danyone): synchronized target-view videos around that person,
 *             the fp16 two-card configuration of tools/4danyone/README.md
 *   mesh      avatar_mesh.py: a textured mesh per frame from those views (visual hull, one
 *             unwrap per GOP, deformed onto the following frames), as a capture folder
 *   encode    the ARES encoder on that folder → apps/demo/<name>-p<id>.ares
 *
 * The caller owns the mask run directory and the SAM service; this module spawns the three local
 * Python and Node steps and reports progress through `send`.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

export const AVATAR_DEFAULTS = {
  views: 6,            // per pitch layer; the total must divide by 6
  pitch: 15,           // camera pitch in degrees, positive above the subject
  frames: 45,          // generated window; 61 and up exhaust an 11 GB card at 704x1280 (measured)
  turbo: true,
  voxel: 0.01,         // visual-hull voxel, metres
  faces: 40000,
  tex: 1024,
  rekey: 0.02,         // metres of fit residual above which a frame is meshed afresh
  height: 1280,
  width: 704,
};

export function avatarPaths(ROOT) {
  const fd = join(ROOT, "tools", "ext", "4danyone");
  return {
    fd,
    fdPython: join(fd, "venv", "Scripts", "python.exe"),
    fdInference: join(fd, "inference.py"),
    fdModels: join(fd, "models"),
    servicePython: join(ROOT, "tools", "sam-service", "env", "Scripts", "python.exe"),
    avatar: join(ROOT, "tools", "sam-service", "avatar.py"),
    avatarMesh: join(ROOT, "tools", "sam-service", "avatar_mesh.py"),
  };
}

/** Component ids this run needs that the machine does not have, as {id, why} rows. */
export function avatarMissing(ROOT) {
  const P = avatarPaths(ROOT);
  const missing = [];
  if (!existsSync(P.fdInference)) missing.push({ id: "4danyone-code", why: "tools/ext/4danyone is not cloned" });
  else if (!existsSync(P.fdPython)) missing.push({ id: "4danyone-env", why: "tools/ext/4danyone/venv is not built" });
  if (!existsSync(join(P.fdModels, "4danyone", "model.safetensors"))) missing.push({ id: "4danyone-weights", why: "the 4DAnyone checkpoint is not downloaded" });
  if (!existsSync(join(P.fdModels, "body_models", "smplx", "SMPLX_NEUTRAL.npz"))) missing.push({ id: "4danyone-smplx", why: "SMPLX_NEUTRAL.npz is not installed" });
  if (!existsSync(P.servicePython)) missing.push({ id: "python-env", why: "the SAM service environment is not built" });
  return missing;
}

/** Spawn one child, stream its lines to `send`, resolve on exit 0. `onLine` may swallow a line. */
function run(cmd, args, { cwd, env, send, tag, onLine, onChild }) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, env: { ...process.env, ...env }, windowsHide: true });
    if (onChild) onChild(child);
    let buf = "";
    const feed = (d) => {
      buf += d;
      const lines = buf.split(/\r?\n/);
      buf = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        if (onLine && onLine(line)) continue;
        send("log", `[${tag}] ${line}`);
      }
    };
    child.stdout.on("data", feed);
    child.stderr.on("data", feed);
    child.on("error", reject);
    child.on("close", (code) => {
      if (buf.trim()) send("log", `[${tag}] ${buf}`);
      code === 0 ? resolve() : reject(new Error(`${tag} exited ${code}`));
    });
  });
}

/**
 * Write each person's reference cutout and 9:16 clip from a finished mask run.
 * Returns the people that got a clip: [{id, start, frames, personHeightPx, dir}].
 */
export async function avatarPeople({ ROOT, runDir, frames, minHeight = 0, send, onChild }) {
  const P = avatarPaths(ROOT);
  await run(P.servicePython, [P.avatar, "refs", runDir], { cwd: ROOT, send, tag: "people", onChild });
  await run(P.servicePython, [P.avatar, "tracks", runDir, "--frames", String(frames), "--min-height", String(minHeight)],
    { cwd: ROOT, send, tag: "people", onChild });
  const people = [];
  let entries = [];
  try { entries = await readdir(join(runDir, "avatar")); } catch { /* no people */ }
  for (const name of entries) {
    const dir = join(runDir, "avatar", name);
    if (!existsSync(join(dir, "track.json"))) continue;
    const info = JSON.parse(await readFile(join(dir, "track.json"), "utf8"));
    let ref = null, thumb = null;
    try { ref = JSON.parse(await readFile(join(dir, "ref.json"), "utf8")); } catch { /* no reference */ }
    // A 96 px cutout inlined into the stream, so the card can show who it found without a route
    // that serves files out of a temporary run directory.
    try { thumb = "data:image/png;base64," + (await readFile(join(dir, "thumb.png"))).toString("base64"); } catch { /* no thumbnail */ }
    people.push({ ...info, dir, ref, thumb });
  }
  people.sort((a, b) => b.personHeightPx - a.personHeightPx);
  return people;
}

/** Generate the target views around one person. Returns the result directory. */
export async function avatarGenerate({ ROOT, person, knobs, send, onChild }) {
  const P = avatarPaths(ROOT);
  const out = join(person.dir, "gen");
  const env = {
    PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True",
    FDANYONE_FRAMES: String(knobs.frames),
    FDANYONE_HEIGHT: String(knobs.height),
    FDANYONE_WIDTH: String(knobs.width),
    FDANYONE_TILED_VAE: "1",
    // The DiT's internal chunk budgets: 1536 MB in the release, which overflows an 11 GB card.
    FDANYONE_TEMP_BUDGET_MB: "256",
    FDANYONE_DIT_DTYPE: "float16",
    FDANYONE_VAE_DTYPE: "float16",
    FDANYONE_SPLIT_DEVICE: "cuda:1",
    FDANYONE_POSE_BATCH: "3",
  };
  const args = [P.fdInference, "--video_path", join(person.dir, "track.mp4"), "--output_dir", out,
    "--views_per_layer", String(knobs.views), "--layer_pitches", `[${knobs.pitch}]`,
    "--enable_turbo", knobs.turbo ? "True" : "False", "--gpu_ids", "[0]", "--attention_backend", "sdpa"];
  await run(P.fdPython, ["-u", ...args], {
    cwd: P.fd, env, send, tag: `generate p${person.id}`, onChild,
    onLine: (line) => {
      const m = /"fraction":\s*([0-9.]+)/.exec(line);
      if (m) { send("progress", { stage: "generate", person: person.id, fraction: Number(m[1]) }); return false; }
      return false;
    },
  });
  return out;
}

/** One textured mesh per frame from generated views. Returns the capture folder. */
export async function avatarMesh({ ROOT, person, result, knobs, send, onChild }) {
  const P = avatarPaths(ROOT);
  const out = join(person.dir, "frames");
  await run(P.fdPython, ["-u", P.avatarMesh, result, out, "--voxel", String(knobs.voxel), "--faces", String(knobs.faces),
    "--tex", String(knobs.tex), "--rekey", String(knobs.rekey)], {
    cwd: P.fd, send, tag: `mesh p${person.id}`, onChild,
    onLine: (line) => {
      const m = /^\[mesh] frame (\d+): (\w+)/.exec(line);
      if (m) send("progress", { stage: "mesh", person: person.id, frame: Number(m[1]), kind: m[2] });
      return false;
    },
  });
  return out;
}

/** Encode a capture folder into apps/demo/<name>.ares. Returns the repo-relative output path. */
export async function avatarEncode({ ROOT, dir, name, knobs, send, onChild, encoderArgs }) {
  const outRel = `apps/demo/${name}.ares`;
  await run(process.execPath, encoderArgs(dir, join(ROOT, outRel)), {
    cwd: ROOT, send, tag: `encode ${name}`, onChild,
    onLine: (line) => {
      const m = /^\[ares] progress (\w+) (\d+)\/(\d+)/.exec(line);
      if (m) { send("progress", { stage: "encode", name, frame: Number(m[2]), of: Number(m[3]) }); return true; }
      return false;
    },
  });
  return outRel;
}
