/**
 * Multiview SAM-3D-Body pod orchestration (launch → setup → upload → infer → fetch → stop).
 * Reuses serve.mjs's RunPod pattern (GraphQL + our own ssh keypair). Key comes from RUNPOD_API_KEY's
 * file twin: RUNPOD_KEY_FILE, default .runpod/runpod.key (git-ignored).
 *
 *   node mv_run.mjs launch            # L40S Secure pod; saves {id,ip,port} → .runpod/mv-pod.json
 *   node mv_run.mjs setup             # clone repo + deps (with the validated gotcha fixes) + weights
 *   node mv_run.mjs upload <dir> <f1> <f2> ...   # scp scripts + mesh-/atlas- files for those frames
 *   node mv_run.mjs infer <views>     # run mv_render_infer.py (default 8 views), tee /workspace/run.log
 *   node mv_run.mjs fetch <localOut>  # scp out/*.json back
 *   node mv_run.mjs stop              # TERMINATE the pod (stops billing) — run this when done!
 *   node mv_run.mjs status            # balance + running pods
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const KEY_FILE = process.env.RUNPOD_KEY_FILE || join(ROOT, ".runpod", "runpod.key");
const SSH_DIR = join(ROOT, ".runpod");
const SSH_KEY = join(SSH_DIR, "id_pod");
const STATE = join(SSH_DIR, "mv-pod.json");
const HF_REPO = "jetjodh/sam-3d-body-dinov3";       // ungated mirror (validated 2026-07-13)
const IMAGE = "runpod/pytorch:1.0.7-dev-nix-cu1290-torch280-ubuntu2204";

const key = (await readFile(KEY_FILE, "utf8")).trim();
async function gql(query, variables) {
  const r = await fetch("https://api.runpod.io/graphql", {
    method: "POST", headers: { "Content-Type": "application/json", Authorization: "Bearer " + key },
    body: JSON.stringify({ query, variables }),
  });
  const j = await r.json();
  if (j.errors) throw new Error(JSON.stringify(j.errors));
  return j.data;
}
async function ensureSshKey() {
  await mkdir(SSH_DIR, { recursive: true });
  if (!existsSync(SSH_KEY)) await new Promise((res, rej) => {
    const p = spawn("ssh-keygen", ["-t", "ed25519", "-N", "", "-f", SSH_KEY, "-C", "ares-runpod"], { stdio: "ignore" });
    p.on("close", (c) => (c === 0 ? res() : rej(new Error("ssh-keygen " + c))));
  });
  return (await readFile(SSH_KEY + ".pub", "utf8")).trim();
}
async function podSsh(id) {
  const d = await gql("query($id:String!){pod(input:{podId:$id}){desiredStatus runtime{ports{ip isIpPublic privatePort publicPort type}}}}", { id });
  const pod = d.pod; if (!pod) return { ready: false, status: "GONE" };
  const ssh = ((pod.runtime && pod.runtime.ports) || []).find((p) => p.privatePort === 22 && p.isIpPublic && p.type === "tcp");
  return ssh ? { ready: true, ip: ssh.ip, port: ssh.publicPort, status: pod.desiredStatus } : { ready: false, status: pod.desiredStatus };
}
// Connect via RunPod's SSH PROXY (ssh.runpod.io, username = pod id) — works without a public TCP port,
// authenticates against the account-level key. scp/sftp aren't supported over the proxy, so files move by
// tar-over-SSH through the command channel (private: bytes go straight to your pod, no third party).
// Connect DIRECT (root@ip -p port, uses the pod's PUBLIC_KEY authorized_keys) when the state has a public
// TCP endpoint; else fall back to the RunPod PROXY (id@ssh.runpod.io, account-key auth). Files move by
// tar-over-SSH either way (private: straight to your pod, no third party; works on both paths).
const SSH_OPTS = ["-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"];
const sshBase = (s) => s.ip ? [...SSH_OPTS, "-p", String(s.port), `root@${s.ip}`] : [...SSH_OPTS, `${s.id}@ssh.runpod.io`];
function sh(s, command) {
  return new Promise((res) => {
    const p = spawn("ssh", [...sshBase(s), command], { stdio: ["ignore", "inherit", "inherit"] });
    p.on("close", (c) => res(c));
  });
}
// Upload: pipe `tar cz` of `files` (relative to `baseDir`) into `tar xz` on the pod at remoteDir.
function uploadTar(s, baseDir, files, remoteDir) {
  return new Promise((res, rej) => {
    const tar = spawn("tar", ["czf", "-", "-C", baseDir, ...files], { stdio: ["ignore", "pipe", "inherit"] });
    const ssh = spawn("ssh", [...sshBase(s), `mkdir -p ${remoteDir} && tar xzf - --no-same-owner -C ${remoteDir}`], { stdio: [tar.stdout, "inherit", "inherit"] });
    ssh.on("close", (c) => (c === 0 ? res() : rej(new Error("uploadTar " + c))));
    tar.on("error", rej);
  });
}
// Download: pipe `tar cz` of remoteDir from the pod into `tar xz` locally.
function fetchTar(s, remoteDir, localDir) {
  return new Promise((res, rej) => {
    const ssh = spawn("ssh", [...sshBase(s), `tar czf - -C ${remoteDir} .`], { stdio: ["ignore", "pipe", "inherit"] });
    const tar = spawn("tar", ["xzf", "-", "-C", localDir], { stdio: [ssh.stdout, "inherit", "inherit"] });
    tar.on("close", (c) => (c === 0 ? res() : rej(new Error("fetchTar " + c))));
    ssh.on("error", rej);
  });
}
const loadState = async () => JSON.parse(await readFile(STATE, "utf8"));

const cmd = process.argv[2];
const rest = process.argv.slice(3);

if (cmd === "status") {
  const d = await gql("query{myself{clientBalance currentSpendPerHr pods{id name desiredStatus costPerHr machine{gpuDisplayName}}}}");
  console.log("balance $" + d.myself.clientBalance, "| spend/hr $" + d.myself.currentSpendPerHr);
  for (const p of d.myself.pods || []) console.log(`  ${p.id} ${p.name} ${p.desiredStatus} ${p.machine?.gpuDisplayName} $${p.costPerHr}/hr`);

} else if (cmd === "launch") {
  const pub = await ensureSshKey();
  const input = {
    cloudType: "SECURE", gpuCount: 1, gpuTypeId: rest[0] || "NVIDIA L40S",
    name: "ares-mv-sam3d", imageName: IMAGE,
    containerDiskInGb: 60, volumeInGb: 40, volumeMountPath: "/workspace",
    ports: "22/tcp,8888/http", env: [{ key: "PUBLIC_KEY", value: pub }],
  };
  const d = await gql("mutation($input:PodFindAndDeployOnDemandInput){podFindAndDeployOnDemand(input:$input){id name costPerHr}}", { input });
  const pod = d.podFindAndDeployOnDemand;
  console.log("launched", pod.id, "$" + pod.costPerHr + "/hr — waiting for SSH…");
  let ep = null;
  for (let i = 0; i < 60; i++) { ep = await podSsh(pod.id); if (ep.ready) break; process.stdout.write(`.${ep.status || ""}`); await new Promise((r) => setTimeout(r, 4000)); }
  if (!ep?.ready) { console.log("\nSSH not up yet; re-run `status`/`launch`-poll. pod id:", pod.id); await writeFile(STATE, JSON.stringify({ id: pod.id })); process.exit(1); }
  await writeFile(STATE, JSON.stringify({ id: pod.id, ip: ep.ip, port: ep.port, costPerHr: pod.costPerHr }));
  console.log(`\nREADY ${ep.ip}:${ep.port} — state → ${STATE}. Remember: \`node mv_run.mjs stop\` when done ($${pod.costPerHr}/hr).`);

} else if (cmd === "setup") {
  const s = await loadState();
  // idempotent: clone repo, apply the validated gotcha fixes, install multiview render deps, prefetch weights.
  // Per the repo INSTALL.md (no `pip install -e .` — the package is used via sys.path) + the validated
  // gotcha fixes (setuptools<81 for pkg_resources, PyOpenGL==3.1.7 for EGL, drop xtcocotools). MoGe/SAM3
  // are skipped (multiview discards the monocular depth guess). libegl1 supplies libEGL.so.1 (the vendor-
  // neutral loader the image lacks) so headless pyrender/EGL can load.
  const DEPS = "pytorch-lightning pyrender trimesh pillow opencv-python-headless yacs scikit-image einops "
    + "timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump "
    + "'networkx==3.2.1' roma joblib seaborn appdirs cython jsonlines loguru optree fvcore pycocotools "
    + "tensorboard huggingface_hub";
  const script = [
    "set -e; cd /workspace",
    "export DEBIAN_FRONTEND=noninteractive",
    "apt-get update -qq >/dev/null 2>&1 || true",
    "apt-get install -y -qq libegl1 libgles2 libglvnd0 libglib2.0-0 >/dev/null 2>&1 || true",
    "pip -q install 'setuptools<81'",
    "[ -d sam-3d-body ] || git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git",
    `pip -q install ${DEPS}`,
    "pip -q install 'PyOpenGL==3.1.7'",
    "pip -q install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps",
    "PYOPENGL_PLATFORM=egl python -c \"import pyrender,trimesh; print('render deps ok')\"",
    `python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_REPO}'); print('weights cached')"`,
    "python -c \"import sys; sys.path.insert(0,'/workspace/sam-3d-body'); from notebook.utils import setup_sam_3d_body; print('repo import ok')\"",
    "echo SETUP_OK",
  ].join(" && ");
  process.exit(await sh(s, `{ ${script} ; } 2>&1 | tee -a /workspace/run.log`));

} else if (cmd === "upload") {
  const s = await loadState();
  const dir = rest[0], frames = rest.slice(1);
  await sh(s, "mkdir -p /workspace/mv/frames /workspace/mv/out");
  // scripts (tar-over-ssh; scp isn't available on the proxy)
  await uploadTar(s, join(ROOT, "tools", "sam3d"), ["mv_render_infer.py"], "/workspace/mv");
  // per-frame mesh + atlas
  const files = [];
  for (const f of frames) { const stub = "f" + String(f).padStart(5, "0"); files.push(`mesh-${stub}.obj`, `atlas-${stub}.png`); }
  await uploadTar(s, dir, files, "/workspace/mv/frames");
  await writeFile(join(ROOT, ".runpod", "jobs.json"), JSON.stringify({ frames: frames.map(Number) }));
  await uploadTar(s, join(ROOT, ".runpod"), ["jobs.json"], "/workspace/mv");
  console.log(`uploaded ${frames.length} frame(s) + scripts`);

} else if (cmd === "infer") {
  const s = await loadState();
  const views = rest[0] || "8";
  const repo = "/workspace/sam-3d-body";
  const script = `cd /workspace/mv && PYOPENGL_PLATFORM=egl python mv_render_infer.py --manifest jobs.json --repo ${repo} --hf ${HF_REPO} --frames-dir frames --out out --views ${views} --elev 10 --res 1024`;
  process.exit(await sh(s, `{ ${script} ; } 2>&1 | tee -a /workspace/run.log`));

} else if (cmd === "setup-raft") {
  const s = await loadState();
  // ADDITIVE to `setup` — removes nothing. `setup` (SAM-3D repo, detectron2, HF weights, pyrender,
  // PyOpenGL 3.1.7, libegl1) stays exactly as validated 2026-07-13, so the skeleton stack remains
  // available on the same pod; this only adds what RAFT needs on top. The image ships torch 2.8 +
  // cu129 and torchvision provides RAFT-large (~20MB) — prefetch so the timed run isn't downloading.
  const script = [
    "set -e; cd /workspace",
    "python -c \"import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())\"",
    "python -c \"from torchvision.models.optical_flow import raft_large, Raft_Large_Weights; raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False); print('RAFT weights cached')\"",
    "echo SETUP_RAFT_OK",
  ].join(" && ");
  process.exit(await sh(s, `{ ${script} ; } 2>&1 | tee -a /workspace/run.log`));

} else if (cmd === "upload-flow") {
  const s = await loadState();
  const dir = rest[0], f0 = Number(rest[1]), f1 = Number(rest[2]);
  if (!dir || !Number.isInteger(f0) || !Number.isInteger(f1)) throw new Error("usage: upload-flow <frames-dir> <from> <to>");
  await sh(s, "mkdir -p /workspace/mv/frames /workspace/mv/flow-samples");
  await uploadTar(s, join(ROOT, "tools", "sam3d"), ["mv_flow_capture.py"], "/workspace/mv");
  // INCLUSIVE of f1: mv_flow_capture iterates pairs f -> f+1 for f in [from, to), so it reads frame f1 too.
  const files = [];
  for (let f = f0; f <= f1; f++) { const stub = "f" + String(f).padStart(5, "0"); files.push(`mesh-${stub}.obj`, `atlas-${stub}.png`); }
  await uploadTar(s, dir, files, "/workspace/mv/frames");
  console.log(`uploaded mv_flow_capture.py + frames ${f0}..${f1} (${files.length} files)`);

} else if (cmd === "flow") {
  const s = await loadState();
  const f0 = rest[0], f1 = rest[1], views = rest[2] || "6", backend = rest[3] || "raft";
  if (!f0 || !f1) throw new Error("usage: flow <from> <to> [views=6] [raft|farneback]");
  const script = `cd /workspace/mv && PYOPENGL_PLATFORM=egl python -u mv_flow_capture.py --frames-dir frames`
    + ` --from ${f0} --to ${f1} --views ${views} --res 1024 --tex-thresh 45 --samples-per-view 6000`
    + ` --flow-backend ${backend} --out flow-samples`;
  process.exit(await sh(s, `{ ${script} ; } 2>&1 | tee -a /workspace/run.log`));

} else if (cmd === "fetch-flow") {
  const s = await loadState();
  const localOut = rest[0] || join(ROOT, "sam3d-results", "flow-samples-raft");
  await mkdir(localOut, { recursive: true });
  await fetchTar(s, "/workspace/mv/flow-samples", localOut);
  console.log("fetched → " + localOut);

} else if (cmd === "fetch") {
  const s = await loadState();
  const localOut = rest[0] || join(ROOT, "sam3d-results", "mv-out");
  await mkdir(localOut, { recursive: true });
  await fetchTar(s, "/workspace/mv/out", localOut);
  console.log("fetched → " + localOut);

} else if (cmd === "stop") {
  const s = await loadState();
  await gql("mutation($id:String!){podTerminate(input:{podId:$id})}", { id: s.id });
  console.log("terminated", s.id);

} else {
  console.log("commands: status | launch [gpuTypeId] | setup | upload <dir> <frames...> | infer [views] | fetch [dir] | stop");
}
