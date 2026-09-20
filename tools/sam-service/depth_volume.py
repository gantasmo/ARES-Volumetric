# Volumetric phases for the ARES depth engine: the job-side half of 2D video -> full volumetric.
#
# A depth job with `volumetric: true` (which requires `subject`) runs two more phases after its mask
# and depth passes, each on every CUDA device at once through volume_worker.py:
#   geometry  MoGe-2 on the unmasked frame -> metric.f32, normals.i8, intrinsics.f32; depth.json
#             gets "intrinsics" (clip median), "metric" and "normals"
#   body      SAM 3D Body from the subject box -> body.f32, body-faces.u32, body-valid.u8 (plus the
#             boxes it ran on, body-boxes.f32); depth.json gets "body"
# Every file is frames-major in the depth maps' frame order, preallocated at full size here, and
# filled positionally by workers that own disjoint contiguous shards. depth.json is rewritten
# (tmp + os.replace, depth.py _write_manifest) after each phase, and a key is added only once its
# files are complete, so a job killed at any point leaves a valid partial run.
#
# ONE PROCESS PER GPU. SAM 3D Body hard-codes the device "cuda" (volume_worker.py header), so each
# card gets its own worker with CUDA_VISIBLE_DEVICES=k. The phases run one after the other, every
# card on the same phase. Measured on this machine 2026-09-19 (2x RTX 2080 Ti 11 GB, 90 frames of a
# 1920x1080 clip, map 518x294):
#   body, "full"   one worker 0.64 frames/s (1545 ms/frame), two workers 1.03 frames/s (1761 and
#                  1910 ms/frame per card: the second card drives the display, and both workers
#                  share the CPU for decode and crop), 1.60x; outputs agree to 0.53 mm
#   geometry       two workers 3.86 frames/s at fp32 (504 ms/frame per card); fp16 runs 193 ms/frame
#   peaks          body 3470 MiB, geometry 2312 MiB (fp32) / 2978 MiB (fp16) per worker process
#
# LOCKING. The job already holds depth.py's _job_lock (one job at a time). main.py's GPU lock is held
# while the resident depth model is freed and until every worker reports its model loaded, the same
# rule depth.py applies to its own loads ("the load allocates; share SAM's GPU lock for it"); the
# per-frame work runs in other processes and does not take it, so /segment stays interactive.
#
# A LOST WORKER COSTS ITS OWN FRAMES ONLY. A worker that reports an error or exits before "done" (an
# out-of-memory on the card that drives the display, or on GPU 0 next to SAM 3 and a /detail run)
# leaves the frames it emitted on disk; [first frame it did not emit, end of its shard) is queued and
# runs on another GPU once that GPU's own worker has exited. The other workers keep running. A shard
# lost twice, or lost with no other GPU left, fails the job with "worker-failed".
#
# THE BODY BOX. The subject mask is walked in frame order here, once, before the body workers start:
# per frame the 8-connected component overlapping the previous frame's chosen component most is
# chosen (the first frame, and any frame overlapping nothing, takes the largest), and its bounds
# padded by BOX_PAD per side are the box. Done at map size and scaled: nearest upsampling of a mask
# keeps its components and scales their bounds, so the box equals the decode-size one to within a
# decode pixel, and the chain of choices stays sequential across shard boundaries.
#
# BACKFILL LAW (the Body4D field note, patch-backfill.py, 2026-07-17): a frame with no box, no
# output, a non-finite value, the wrong vertex count or a vertex behind the camera takes the mesh of
# the NEAREST valid frame (leading frames therefore borrow the next valid one) and keeps 0 in
# body-valid.u8. Every frame of body.f32 holds a usable mesh after the phase.
#
# CONVENTION CHECK. The first valid frame of the clip is projected with the MAP intrinsics; the IoU
# of its dilated vertex footprint against the subject mask inside the box must reach
# MIN_CONVENTION_IOU or the job fails with errorCode "body-convention" (an axis flip, a focal in the
# wrong units or a mis-scaled box all land far below it).

import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from queue import Empty, Queue

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARES_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
WORKER = os.path.join(HERE, "volume_worker.py")
SAM3D_BODY_CODE = os.environ.get("SAM3D_BODY_CODE_DIR") or os.path.join(ARES_ROOT, "tools", "ext", "sam-3d-body")

MOGE_REPO = "Ruicheng/moge-2-vitl-normal"
# The ungated mirror first: it is what the installer fetches. The Meta repo holds the same files.
BODY_REPOS = ("jetjodh/sam-3d-body-dinov3", "facebook/sam-3d-body-dinov3")
BODY_FILES = ("model_config.yaml", "model.ckpt", os.path.join("assets", "mhr_model.pt"))
BODY_VERTICES = 18439
BODY_FACES = 36874
# Decode sizes, long side. MoGe-2 resizes to its own token budget internally, so past 1280 a larger
# frame costs decode and post-processing time only. SAM 3D Body crops the box to 512x512, and the
# hand crops of the "full" pass come from the same frame, so it gets the source up to 1080p.
GEOMETRY_EDGE = int(os.environ.get("DEPTH_GEOMETRY_EDGE", "1280"))
BODY_EDGE = int(os.environ.get("DEPTH_BODY_EDGE", "1920"))
# fp16 autocast: 2.7x faster than fp32 at 0.05 % median depth difference (volume_worker.py header).
GEOMETRY_FP16 = os.environ.get("DEPTH_GEOMETRY_FP16", "1").strip() in ("1", "true", "on")
# "full" (body decoder, both hand crops, body decoder re-prompted) or "body" (the first body
# decoder pass alone): 3.7x faster, and a different mesh on 8 of 20 measured frames.
BODY_INFERENCE = os.environ.get("DEPTH_BODY_INFERENCE", "full").strip()
if BODY_INFERENCE not in ("full", "body"):
    # Not raised: an import failure here would take every /depth route down with it.
    print(f"[ares-sam] depth: DEPTH_BODY_INFERENCE must be full or body, got {BODY_INFERENCE!r}; using full")
    BODY_INFERENCE = "full"
BOX_PAD = 0.05
MIN_CONVENTION_IOU = 0.3
LOAD_WAIT_S = float(os.environ.get("DEPTH_VOLUME_LOAD_WAIT_S", "600"))
STDERR_KEEP = 40
# A failed import probe is run again on the first /depth/health after this many seconds: a package
# installed into the env without changing the component state is then seen.
PROBE_RETRY_S = 120.0

RUNTIME_MODULES = ("cv2", "roma", "yacs", "omegaconf", "pytorch_lightning", "timm", "braceexpand",
                   "termcolor", "iopath", "fvcore", "submitit")
# Component ids, the installer catalog's (tools/installer.mjs VOLUMETRIC_COMPONENTS and its
# `requires` graph). A 409 from /depth/run lists these.
C_MOGE_CODE = "moge-code"
C_MOGE_WEIGHTS = "moge-2"
C_BODY_CODE = "sam3d-body-code"
C_BODY_WEIGHTS = "sam3d-body"
C_BODY_RUNTIME = "sam3d-body-runtime"
C_DINOV3_HUB = "dinov3-hub"

GEOMETRY_FILES = ("metric.f32", "normals.i8", "intrinsics.f32")
BODY_OUT_FILES = ("body.f32", "body-faces.u32", "body-valid.u8", "body-boxes.f32")
FILES = GEOMETRY_FILES + BODY_OUT_FILES


def _no_window() -> dict:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags} if flags else {}


# ---------------------------------------------------------------- components ----


def _hub_cache_dir() -> str:
    return os.environ.get("HUGGINGFACE_HUB_CACHE") or (
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME")
        else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))


def _snapshot(repo_id: str, files) -> str | None:
    """The cached snapshot holding every one of `files`, refs/main first, or None."""
    repo = os.path.join(_hub_cache_dir(), "models--" + repo_id.replace("/", "--"))
    snaps = os.path.join(repo, "snapshots")
    try:
        revs = sorted(os.listdir(snaps))
    except OSError:
        return None
    try:
        with open(os.path.join(repo, "refs", "main"), encoding="utf-8") as fh:
            head = fh.read().strip()
        if head in revs:
            revs.insert(0, revs.pop(revs.index(head)))
    except OSError:
        pass
    for rev in revs:
        snap = os.path.join(snaps, rev)
        if all(os.path.isfile(os.path.join(snap, f)) for f in files):
            return snap
    return None


def moge_weights() -> str | None:
    snap = _snapshot(MOGE_REPO, ("model.pt",))
    return os.path.join(snap, "model.pt") if snap else None


def body_weights() -> tuple[str, str, str] | None:
    """(repo id, model.ckpt, mhr_model.pt) of the first complete snapshot, or None."""
    for repo in BODY_REPOS:
        snap = _snapshot(repo, BODY_FILES)
        if snap:
            return repo, os.path.join(snap, "model.ckpt"), os.path.join(snap, "assets", "mhr_model.pt")
    return None


def _torch_hub_dir() -> str:
    """torch.hub.get_dir() without importing torch: $TORCH_HOME/hub, else $XDG_CACHE_HOME/torch/hub,
    else ~/.cache/torch/hub (torch/hub.py _get_torch_home)."""
    home = os.environ.get("TORCH_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "torch")
    return os.path.join(home, "hub")


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing() -> list[str]:
    """Component ids a volumetric job needs and the machine lacks. Filesystem and import-spec
    lookups only: nothing is imported, nothing is loaded."""
    importlib.invalidate_caches()          # a package installed while the service runs is seen
    gone = []
    if not _module_present("moge"):
        gone.append(C_MOGE_CODE)
    if moge_weights() is None:
        gone.append(C_MOGE_WEIGHTS)
    if not os.path.isfile(os.path.join(SAM3D_BODY_CODE, "sam_3d_body", "__init__.py")):
        gone.append(C_BODY_CODE)
    if body_weights() is None:
        gone.append(C_BODY_WEIGHTS)
    if not all(_module_present(m) for m in RUNTIME_MODULES):
        gone.append(C_BODY_RUNTIME)
    if not os.path.isfile(os.path.join(_torch_hub_dir(), "facebookresearch_dinov3_main", "hubconf.py")):
        gone.append(C_DINOV3_HUB)
    return gone


def devices() -> list[dict]:
    """CUDA devices as this process numbers them, with the id a child's CUDA_VISIBLE_DEVICES needs to
    select the same card (the parent's own CUDA_VISIBLE_DEVICES list, when it has one)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        ids = [s.strip() for s in vis.split(",") if s.strip()] if vis else None
        out = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({"index": i, "name": p.name, "totalMiB": round(p.total_memory / 2**20),
                        "cc": f"{p.major}.{p.minor}",
                        "visibleId": ids[i] if ids and i < len(ids) else str(i)})
        return out
    except Exception:
        return []


def selected_devices() -> list[dict]:
    """Every device, or the DEPTH_VOLUME_GPUS subset (comma list of indices, in that order)."""
    devs = devices()
    want = os.environ.get("DEPTH_VOLUME_GPUS", "").strip()
    if want:
        by = {d["index"]: d for d in devs}
        devs = [by[int(x)] for x in want.split(",") if x.strip().isdigit() and int(x) in by]
    return devs


def no_device_reason() -> str:
    """Why selected_devices() is empty: a DEPTH_VOLUME_GPUS filter that matches no device, or none."""
    want = os.environ.get("DEPTH_VOLUME_GPUS", "").strip()
    devs = devices()
    if want and devs:
        return (f"DEPTH_VOLUME_GPUS {want!r} selects none of the CUDA devices "
                f"{', '.join(str(d['index']) for d in devs)}")
    return "the volumetric phases need a CUDA device; torch reports none in this service"


# The import probe runs volume_worker.py --probe in a child: importing sam_3d_body pulls
# pytorch_lightning, timm and cv2, which the service process itself never needs. It runs once per
# change of the component state and /depth/health reports the cached answer.
_probe_lock = threading.Lock()
_probe = {"state": "idle", "key": None, "result": {}, "at": None}


def _run_probe(key) -> None:
    try:
        out = subprocess.run([sys.executable, WORKER, "--probe"], capture_output=True, timeout=300,
                             cwd=HERE, **_no_window())
        res = {}
        for line in out.stdout.decode("utf-8", "replace").splitlines():
            if line.startswith("ARES-VOLUME "):
                res = json.loads(line[len("ARES-VOLUME "):])
                res.pop("ev", None)
        if not res:
            tail = out.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            res = {"moge": False, "sam3dBody": False, "error": " | ".join(tail)[-600:] or f"exit {out.returncode}"}
    except Exception as e:
        res = {"moge": False, "sam3dBody": False, "error": f"{type(e).__name__}: {e}"}
    with _probe_lock:
        if _probe["key"] == key:
            _probe.update(state="done", result=res, at=time.time())


def _probe_ok(res: dict) -> bool:
    return res.get("moge") is True and res.get("sam3dBody") is True


def health() -> dict:
    """For /depth/health. Filesystem checks inline; the import check is the cached child probe,
    started in the background on the first call, again whenever the component state changes, and
    again PROBE_RETRY_S after a failed answer."""
    gone = missing()
    key = tuple(gone)
    with _probe_lock:
        stale = (_probe["state"] == "done" and not _probe_ok(_probe["result"])
                 and time.time() - (_probe["at"] or 0) > PROBE_RETRY_S)
        if _probe["key"] != key or stale:
            _probe.update(state="running", key=key, result={}, at=None)
            threading.Thread(target=_run_probe, args=(key,), daemon=True, name="volume-probe").start()
        state, res = _probe["state"], dict(_probe["result"])
    devs, sel = devices(), selected_devices()
    return {
        "available": not gone and _probe_ok(res) and bool(sel),
        "missing": gone,
        "probe": state,
        "moge": res.get("moge"),
        "sam3dBody": res.get("sam3dBody"),
        "importErrors": {k: v for k, v in res.items() if k.endswith("Error") or k == "error"},
        "devices": devs,
        "workers": len(sel),
        **({} if sel else {"deviceError": no_device_reason()}),
        "bodyInference": BODY_INFERENCE,
        "geometryDtype": "fp16" if GEOMETRY_FP16 else "fp32",
    }


def import_failure() -> dict | None:
    """The 409 body for a volumetric request whose components are on disk but do not import, from
    the cached probe when it has answered for the current component state. None otherwise: a probe
    still running is no refusal (a job whose imports fail still ends at "worker-failed")."""
    key = tuple(missing())
    with _probe_lock:
        if _probe["key"] != key or _probe["state"] != "done":
            return None
        res = dict(_probe["result"])
    gone = ([C_MOGE_CODE] if res.get("moge") is not True else []) + \
        ([C_BODY_RUNTIME] if res.get("sam3dBody") is not True else [])
    if not gone:
        return None
    errs = "; ".join(f"{k} {v}" for k, v in res.items() if k.endswith("Error") or k == "error")
    return {"error": f"volumetric components installed but not importable: {errs or ', '.join(gone)}",
            "missing": gone}


# ------------------------------------------------------------------- workers ----


class _Worker:
    """One volume_worker.py child. Its stdout protocol lines go to the phase's queue as (k, event);
    its stderr is kept as a tail for the error report. An "exit" event follows stdout's end."""

    def __init__(self, k: int, dev: dict, spec: dict, q: Queue):
        self.k, self.dev = k, dev
        self.err: deque[str] = deque(maxlen=STDERR_KEEP)
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=dev["visibleId"], PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([sys.executable, WORKER], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=HERE, env=env,
                                     **_no_window())
        self._err_t = threading.Thread(target=self._drain_err, daemon=True, name=f"volume-err-{k}")
        self._err_t.start()
        threading.Thread(target=self._read_out, args=(q,), daemon=True, name=f"volume-out-{k}").start()
        try:
            self.proc.stdin.write(json.dumps(spec).encode("utf-8"))
            self.proc.stdin.close()
        except OSError:
            pass                      # the child already died; its exit event carries the reason

    def _drain_err(self) -> None:
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.err.append(line[:300])

    def _read_out(self, q: Queue) -> None:
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("ARES-VOLUME "):
                try:
                    q.put((self.k, json.loads(line[len("ARES-VOLUME "):])))
                except ValueError:
                    self.err.append("unparsed: " + line[:280])
            elif line:
                self.err.append(line[:300])
        code = self.proc.wait()
        self._err_t.join(timeout=5)
        q.put((self.k, {"ev": "exit", "code": code}))

    def tail(self, n: int = 8) -> str:
        lines = [l for l in self.err if not l.startswith("missing keys")][-n:]
        return " | ".join(lines)[-1500:]

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.kill()      # its ffmpeg child ends on the broken pipe
            except OSError:
                pass


def _shards(n_frames: int, n_workers: int) -> list[tuple[int, int]]:
    base, extra = divmod(n_frames, n_workers)
    out, s = [], 0
    for k in range(n_workers):
        e = s + base + (1 if k < extra else 0)
        out.append((s, e))
        s = e
    return out


def _shard_args(args: list[str], end: int, last: bool) -> list[str]:
    """The contract chain for one shard. The last shard decodes the chain exactly (it has to see the
    stream end to prove parity); every other one stops at its end frame with -frames:v, which cuts
    the SAME sequence short and selects nothing differently."""
    a = list(args)
    if last:
        return a
    if "-frames:v" in a:
        a[a.index("-frames:v") + 1] = str(int(end))
    else:
        i = a.index("-f")
        a[i:i] = ["-frames:v", str(int(end))]
    return a


def _decode_size(src_w: int, src_h: int, edge: int) -> tuple[int, int]:
    s = min(1.0, edge / float(max(src_w, src_h)))
    return max(2, int(round(src_w * s))), max(2, int(round(src_h * s)))


def _prealloc(path: str, size: int) -> None:
    with open(path, "wb") as fh:
        fh.truncate(size)


def _run_phase(job, hooks, phase: str, specs: list[dict], devs: list[dict], n_frames: int):
    """Spawn one worker per spec (spec k on devs[k]) and relay progress into job.vphases[phase].
    A lost worker's remaining frames go to another GPU (the header). Returns the phase summary:
    "workers", one record per worker process (retries included), and the totals over every frame
    ("inferMs", and for the body "valid", "iouSum", "iouN", "iouMin", "invalid"); None when the job
    was cancelled."""
    st = job.vphases[phase]
    st.update(done=0, total=n_frames, msPerFrame=None, gpus=len(specs), framesPerSecond=None)
    q: Queue = Queue()
    workers: list[_Worker] = []
    recs: list[dict] = []            # per worker process
    shard_of: list[int] = []         # worker -> the shard (index into specs) it works on
    loaded: set[int] = set()
    ended: set[int] = set()          # workers whose process is over, or that were given up
    lost: set[int] = set()
    finished: set[int] = set()       # shards whose every frame has been emitted
    checks: dict[int, tuple[int, float]] = {}
    pending: list[tuple[int, int, int]] = []     # (shard, start, end) waiting for a GPU
    retried: set[int] = set()
    bad_devs: set[int] = set()
    tot = {"inferMs": 0.0, "valid": 0, "iouSum": 0.0, "iouN": 0, "iouMin": None, "invalid": {}}
    decided = None
    locked = False
    t0 = t_lock = time.time()
    t_run = None

    def lock():
        nonlocal locked, t_lock
        t_lock = time.time()
        if not locked:
            hooks.lock.acquire()
            locked = True
            job.state = "loading"

    def maybe_unlock():
        nonlocal locked, t_run
        if locked and all(w in loaded or w in ended for w in range(len(workers))):
            hooks.lock.release()
            locked = False
            job.state = "running"
            if t_run is None:
                t_run = time.time()

    def spawn(k: int, dev: dict, start: int, retry: bool):
        lock()
        spec = dict(specs[k], start=start, gpu=str(dev["index"]))
        wi = len(workers)
        workers.append(_Worker(wi, dev, spec, q))
        shard_of.append(k)
        recs.append({"gpu": dev["index"], "start": start, "end": spec["end"], "frames": 0, "next": start,
                     "retry": retry, "lost": None, "inferMs": 0.0, "done": None, "loadMs": None})
        job.workers = [w.proc for w in workers]

    def fail(code: str, msg: str):
        for w in workers:
            w.kill()
        raise hooks.error(code, msg)

    def dispatch():
        while pending:
            busy = {workers[i].dev["index"] for i in range(len(workers)) if i not in ended}
            free = [d for d in devs if d["index"] not in bad_devs and d["index"] not in busy]
            if not free:
                return
            k, s, e = pending.pop(0)
            job.say(f"{phase} gpu{free[0]['index']}: frames {s}-{e - 1} of shard {k}, re-run")
            spawn(k, free[0], s, True)

    def lose(wi: int, why: str):
        w, r, k = workers[wi], recs[wi], shard_of[wi]
        w.kill()
        ended.add(wi)
        lost.add(wi)
        r["lost"] = why[:300]
        bad_devs.add(w.dev["index"])
        where = f"{phase} worker gpu{w.dev['index']} (frames {r['start']}-{r['end'] - 1})"
        if r["next"] >= r["end"]:
            finished.add(k)                      # every frame is on disk; only its report is missing
            job.say(f"{where}: lost after its last frame ({why[:200]})")
            decide()
        else:
            alive = [d for d in devs if d["index"] not in bad_devs]
            if k in retried or not alive:
                fail("worker-failed", f"{where}: {why}; stderr: {w.tail()}"
                     + ("" if alive else "; no other GPU to re-run its frames on")
                     + ("; the shard was already a re-run" if k in retried else ""))
            retried.add(k)
            pending.append((k, r["next"], r["end"]))
            job.say(f"{where}: lost at frame {r['next']} ({why[:200]}); frames {r['next']}-{r['end'] - 1} "
                    f"queued for gpu{', gpu'.join(str(d['index']) for d in alive)}")
        maybe_unlock()
        dispatch()

    def decide():
        """The convention check belongs to the clip's first valid frame: the lowest shard that has
        one, once every shard before it has finished with none."""
        nonlocal decided
        if decided is not None:
            return
        for k in range(len(specs)):
            if k in checks:
                decided = checks[k]
                i, iou = decided
                job.volume_check = {"frame": i, "iou": round(iou, 4)}
                job.say(f"body convention: frame {i}, footprint IoU {iou:.3f} against the subject mask")
                if iou < MIN_CONVENTION_IOU:
                    fail("body-convention",
                         f"body footprint IoU {iou:.3f} at frame {i} is below {MIN_CONVENTION_IOU}: the "
                         f"mesh does not project onto the subject mask with the map intrinsics")
                return
            if k not in finished:
                return

    try:
        for k, (spec, dev) in enumerate(zip(specs, devs)):
            spawn(k, dev, int(spec["start"]), False)
        while len(finished) < len(specs):
            if job.cancel:
                for w in workers:
                    w.kill()
                return None
            try:
                wi, ev = q.get(timeout=0.25)
            except Empty:
                if locked and time.time() - t_lock > LOAD_WAIT_S:
                    fail("worker-failed", f"{phase} workers did not load within {LOAD_WAIT_S:.0f}s")
                continue
            if job.cancel:
                # /depth/cancel kills the workers itself, so their exit events can arrive before this
                # loop has read the flag: a cancel is never reported as a worker failure.
                for w in workers:
                    w.kill()
                return None
            if wi in lost:
                continue                          # a given-up worker's trailing events
            w, r = workers[wi], recs[wi]
            kind = ev.get("ev")
            if kind == "load":
                loaded.add(wi)
                r["loadMs"] = ev.get("ms")
                job.say(f"{phase} gpu{w.dev['index']}: loaded in {ev.get('ms', 0) / 1000:.1f}s "
                        f"({ev.get('device')}, decode {ev.get('decode')})")
                maybe_unlock()
            elif kind == "frame":
                ms = float(ev.get("ms") or 0.0)
                r["next"] = int(ev.get("i", r["next"])) + 1
                r["frames"] += 1
                r["inferMs"] += ms
                tot["inferMs"] += ms
                st["done"] += 1
                st["msPerFrame"] = round(tot["inferMs"] / st["done"], 1)
                if t_run is not None and time.time() > t_run:
                    st["framesPerSecond"] = round(st["done"] / (time.time() - t_run), 3)
                if phase == "body":
                    if ev.get("valid"):
                        tot["valid"] += 1
                        if ev.get("iou") is not None:
                            iou = float(ev["iou"])
                            tot["iouSum"] += iou
                            tot["iouN"] += 1
                            tot["iouMin"] = iou if tot["iouMin"] is None else min(tot["iouMin"], iou)
                    else:
                        why = str(ev.get("why") or "invalid")
                        tot["invalid"][why] = tot["invalid"].get(why, 0) + 1
            elif kind == "check":
                checks.setdefault(shard_of[wi], (int(ev["i"]), float(ev["iou"])))
                decide()
            elif kind == "done":
                r["done"] = ev
                finished.add(shard_of[wi])
                decide()
            elif kind == "error":
                lose(wi, str(ev.get("message")))
            elif kind == "exit":
                if r["done"] is None:
                    lose(wi, f"exited {ev.get('code')} before finishing its shard")
                else:
                    ended.add(wi)                 # its GPU is free for a queued re-run
                    dispatch()
    finally:
        if locked:
            hooks.lock.release()
        for w in workers:
            w.kill()
        job.workers = []
    wall = time.time() - (t_run or t0)
    st["framesPerSecond"] = round(n_frames / wall, 3) if wall > 0 else None
    records = []
    for wi, w in enumerate(workers):
        r, d = recs[wi], recs[wi]["done"] or {}
        n = max(1, r["frames"])
        job.say(f"{phase} gpu{w.dev['index']}: frames {r['start']}-{r['end'] - 1}{' (re-run)' if r['retry'] else ''}, "
                f"{r['frames']} done, {r['inferMs'] / n:.1f} ms/frame, load {(r['loadMs'] or 0) / 1000:.1f}s, "
                f"shard {d.get('wallMs', 0) / 1000:.1f}s, peak {d.get('peakMiB')} MiB (reserved {d.get('reservedMiB')} MiB)"
                + (f", lost: {r['lost'][:160]}" if r["lost"] else ""))
        records.append({"gpu": r["gpu"], "start": r["start"], "end": r["end"], "frames": r["frames"],
                        "retry": r["retry"], "lost": r["lost"],
                        "msPerFrame": round(r["inferMs"] / n, 1) if r["frames"] else None,
                        "loadMs": r["loadMs"], "wallMs": d.get("wallMs"), "peakMiB": d.get("peakMiB"),
                        "reservedMiB": d.get("reservedMiB")})
    gpus = len({r["gpu"] for r in records if r["frames"]})
    job.say(f"{phase} done: {n_frames} frames on {gpus} GPU{'s' if gpus != 1 else ''}"
            + (f" ({len(lost)} worker{'s' if len(lost) != 1 else ''} lost, frames re-run)" if lost else "")
            + f", {wall:.1f}s wall after load, {st['framesPerSecond']} frames/s")
    job.volume_stats[phase] = {"workers": records, "wallS": round(wall, 2), "framesPerSecond": st["framesPerSecond"]}
    return {"workers": records, **tot}


# ------------------------------------------------------------------- geometry ----


def _geometry(job, hooks, src: dict, n: int, devs: list[dict]) -> bool:
    H, W = int(job.height), int(job.width)
    run = job.out
    _prealloc(os.path.join(run, "metric.f32"), n * H * W * 4)
    _prealloc(os.path.join(run, "normals.i8"), n * H * W * 3)
    _prealloc(os.path.join(run, "intrinsics.f32"), n * 16)
    wd, hd = _decode_size(int(src["width"]), int(src["height"]), GEOMETRY_EDGE)
    args, vf = hooks.decode_args(wd, hd)
    weights = moge_weights() or MOGE_REPO
    shards = _shards(n, len(devs))
    specs = [{"phase": "geometry", "worker": k, "workers": len(devs), "gpu": str(d["index"]), "run": run,
              "start": s, "end": e, "frames": n, "mapWidth": W, "mapHeight": H,
              "decode": {"args": _shard_args(args, e, e == n), "width": wd, "height": hd},
              "geometry": {"weights": weights, "fp16": GEOMETRY_FP16}}
             for k, ((s, e), d) in enumerate(zip(shards, devs))]
    job.say(f"geometry: {MOGE_REPO} {'fp16' if GEOMETRY_FP16 else 'fp32'}, ffmpeg -vf {vf}, "
            + ", ".join(f"gpu{d['index']} frames {s}-{e - 1}" for (s, e), d in zip(shards, devs)))
    job.phase = "geometry"
    ph = _run_phase(job, hooks, "geometry", specs, devs, n)
    if ph is None:
        return False

    K = np.fromfile(os.path.join(run, "intrinsics.f32"), dtype="<f4").reshape(n, 4).astype(np.float64)
    good = np.isfinite(K).all(axis=1) & (K[:, 0] > 0) & (K[:, 1] > 0)
    if not good.any():
        raise hooks.error("geometry-failed", "MoGe-2 returned no finite intrinsics on any frame")
    fx, fy, cx, cy = (float(v) for v in np.median(K[good], axis=0))
    fov_y = math.degrees(2.0 * math.atan(0.5 / fy))
    fov_x = math.degrees(2.0 * math.atan(0.5 / fx))
    p10, p90 = np.percentile(K[good, 1], [10, 90])
    job.say(f"intrinsics: fx {fx:.4f} fy {fy:.4f} cx {cx:.4f} cy {cy:.4f} (normalized, clip median of "
            f"{int(good.sum())} frames), fovY {fov_y:.2f} deg, fovX {fov_x:.2f} deg, fy p10-p90 "
            f"{p10:.4f}-{p90:.4f}")

    # One streaming pass for the manifest's statistics: one frame of each file resident at a time.
    medians, valid_sum, nz_sum, nz_n = [], 0.0, 0.0, 0
    with open(os.path.join(run, "metric.f32"), "rb") as fm, \
            open(os.path.join(run, "normals.i8"), "rb") as fn, \
            open(os.path.join(run, "mask.u8"), "rb") as fk:
        for _ in range(n):
            d = np.frombuffer(fm.read(H * W * 4), dtype="<f4").reshape(H, W)
            nz = np.frombuffer(fn.read(H * W * 3), dtype=np.int8).reshape(H, W, 3)[..., 2]
            m = np.frombuffer(fk.read(H * W), dtype=np.uint8).reshape(H, W) > 0
            ok = d > 0
            valid_sum += float(ok.mean())
            inside = m & ok
            if inside.any():
                medians.append(float(np.median(d[inside])))
                nz_sum += float(nz[inside].astype(np.float64).sum()) / 127.0
                nz_n += int(inside.sum())
    med = float(np.median(medians)) if medians else None
    mean_nz = nz_sum / nz_n if nz_n else None
    geo_ms = round(ph["inferMs"] / max(1, n), 1)
    job.volume["intrinsics"] = {"fx": round(fx, 6), "fy": round(fy, 6), "cx": round(cx, 6),
                                "cy": round(cy, 6), "source": "moge-2", "fovY": round(fov_y, 3),
                                "fovX": round(fov_x, 3), "file": "intrinsics.f32",
                                "frames": int(good.sum())}
    job.volume["metric"] = {"file": "metric.f32", "model": MOGE_REPO, "units": "m",
                            "dtype": "fp16" if GEOMETRY_FP16 else "fp32", "msPerFrame": geo_ms,
                            "validFraction": round(valid_sum / n, 5),
                            "medianInMask": round(med, 4) if med is not None else None,
                            "workers": ph["workers"]}
    job.volume["normals"] = {"file": "normals.i8", "space": "opencv-camera",
                             "meanZInMask": round(mean_nz, 4) if mean_nz is not None else None}
    job.say(f"geometry: metric depth median in mask {med if med is None else round(med, 3)} m, "
            f"valid {valid_sum / n:.4f}, mean normal z in mask "
            f"{mean_nz if mean_nz is None else round(mean_nz, 3)}")
    hooks.write_manifest(False)
    return True


# ----------------------------------------------------------------------- body ----


def subject_boxes(mask_path: str, n: int, H: int, W: int, out_path: str,
                  detected_path: str | None = None) -> tuple[int, int, int]:
    """body-boxes.f32: n x 4 float32 LE, (x0, y0, x1, y1) in map pixels (continuous, edges), NaN for
    a frame with no subject, and NaN for a frame whose mask the mask pass copied from a neighbour
    (byte 0 in `detected_path`, depth.py's mask-detected.u8): SAM 3D Body returns a mesh for any box
    it is given, and a copied mask on a frame the subject has left holds no one. Such a frame is
    then "no-box" in the body phase, gets the nearest fitted mesh with body-valid 0, and is never an
    anchor or the convention frame. Returns (frames with a box, frames where the chosen component
    had no overlap with the previous choice and the largest one was taken instead, copied frames)."""
    import cv2

    flags = None
    if detected_path:
        size = os.path.getsize(detected_path)
        if size != n:
            raise RuntimeError(f"{detected_path}: {size} bytes, the run has {n} frames")
        flags = np.fromfile(detected_path, dtype=np.uint8)
    prev = None
    boxed, jumps, copied = 0, 0, 0
    with open(mask_path, "rb") as fm, open(out_path + ".tmp", "wb") as fo:
        for t in range(n):
            m = (np.frombuffer(fm.read(H * W), dtype=np.uint8).reshape(H, W) > 0).astype(np.uint8)
            row = np.full(4, np.nan, dtype="<f4")
            if flags is not None and not flags[t]:
                copied += 1
                fo.write(row.tobytes())
                continue
            count, labels, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
            if count > 1:
                pick = None
                if prev is not None:
                    ov = np.bincount(labels[prev], minlength=count)
                    ov[0] = 0
                    if ov.max() > 0:
                        pick = int(np.argmax(ov))
                    else:
                        jumps += 1
                if pick is None:
                    pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y = float(stats[pick, cv2.CC_STAT_LEFT]), float(stats[pick, cv2.CC_STAT_TOP])
                w, h = float(stats[pick, cv2.CC_STAT_WIDTH]), float(stats[pick, cv2.CC_STAT_HEIGHT])
                row[:] = (max(0.0, x - BOX_PAD * w), max(0.0, y - BOX_PAD * h),
                          min(float(W), x + w + BOX_PAD * w), min(float(H), y + h + BOX_PAD * h))
                prev = labels == pick
                boxed += 1
            fo.write(row.tobytes())
    os.replace(out_path + ".tmp", out_path)
    return boxed, jumps, copied


def _backfill(run: str, n: int, V: int) -> tuple[int, list[int]]:
    """Copy the nearest valid frame's mesh into every invalid one. Returns (valid count, invalid
    frame indices). One frame of vertices resident at a time."""
    valid = np.fromfile(os.path.join(run, "body-valid.u8"), dtype=np.uint8)[:n] == 1
    good = np.flatnonzero(valid)
    bad = np.flatnonzero(~valid)
    if not len(good) or not len(bad):
        return int(len(good)), [int(i) for i in bad]
    plane = V * 12
    with open(os.path.join(run, "body.f32"), "r+b") as fb:
        for k in bad:
            j = int(np.searchsorted(good, k))
            cand = [good[j - 1]] if j > 0 else []
            if j < len(good):
                cand.append(good[j])
            src = int(min(cand, key=lambda g: (abs(int(g) - int(k)), int(g))))
            fb.seek(src * plane)
            data = fb.read(plane)
            fb.seek(int(k) * plane)
            fb.write(data)
    return int(len(good)), [int(i) for i in bad]


def _body(job, hooks, src: dict, n: int, devs: list[dict]) -> bool:
    H, W = int(job.height), int(job.width)
    run = job.out
    weights = body_weights()
    if weights is None:
        raise hooks.error("missing-components", "SAM 3D Body weights are not installed", [C_BODY_WEIGHTS])
    repo, ckpt, mhr = weights
    intr = job.volume["intrinsics"]
    t_box = time.time()
    det = (getattr(job, "mask_info", None) or {}).get("detected")
    boxed, jumps, copied = subject_boxes(os.path.join(run, "mask.u8"), n, H, W, os.path.join(run, "body-boxes.f32"),
                                         os.path.join(run, det) if det else None)
    job.say(f"body boxes: {boxed}/{n} frames, {copied} copied-mask frames without a box, {jumps} component "
            f"jumps, {time.time() - t_box:.2f}s")
    if boxed == 0:
        raise hooks.error("body-not-found", "the subject mask has no detected connected component on any frame")
    _prealloc(os.path.join(run, "body.f32"), n * BODY_VERTICES * 12)
    _prealloc(os.path.join(run, "body-valid.u8"), n)
    try:
        os.remove(os.path.join(run, "body-faces.u32"))
    except OSError:
        pass
    wd, hd = _decode_size(int(src["width"]), int(src["height"]), BODY_EDGE)
    args, vf = hooks.decode_args(wd, hd)
    shards = _shards(n, len(devs))
    specs = [{"phase": "body", "worker": k, "workers": len(devs), "gpu": str(d["index"]), "run": run,
              "start": s, "end": e, "frames": n, "mapWidth": W, "mapHeight": H,
              "decode": {"args": _shard_args(args, e, e == n), "width": wd, "height": hd},
              "body": {"checkpoint": ckpt, "mhr": mhr, "inference": BODY_INFERENCE,
                       "intrinsics": [intr["fx"], intr["fy"], intr["cx"], intr["cy"]],
                       "boxes": "body-boxes.f32", "writeFaces": k == 0, "vertices": BODY_VERTICES}}
             for k, ((s, e), d) in enumerate(zip(shards, devs))]
    job.say(f"body: {repo} fp16, inference {BODY_INFERENCE}, ffmpeg -vf {vf}, "
            + ", ".join(f"gpu{d['index']} frames {s}-{e - 1}" for (s, e), d in zip(shards, devs)))
    job.phase = "body"
    job.volume_check = None
    ph = _run_phase(job, hooks, "body", specs, devs, n)
    if ph is None:
        return False
    reasons = ph["invalid"]

    faces = os.path.join(run, "body-faces.u32")
    if not os.path.isfile(faces) or os.path.getsize(faces) != BODY_FACES * 12:
        raise hooks.error("worker-failed", f"body-faces.u32 is missing or not {BODY_FACES} x 3 uint32")
    valid_n, bad = _backfill(run, n, BODY_VERTICES)
    if valid_n == 0:
        raise hooks.error("body-not-found", f"SAM 3D Body produced no valid mesh on any of {n} frames "
                                            f"({', '.join(f'{k} {v}' for k, v in reasons.items()) or 'no reason'})")
    iou_sum, iou_n, iou_min = ph["iouSum"], ph["iouN"], ph["iouMin"]
    body_ms = round(ph["inferMs"] / max(1, n - reasons.get("no-box", 0)), 1)
    chk = job.volume_check or {}
    job.volume["body"] = {"file": "body.f32", "faces": "body-faces.u32", "valid": "body-valid.u8",
                          "boxes": "body-boxes.f32", "vertices": BODY_VERTICES, "faceCount": BODY_FACES,
                          "units": "m", "space": "opencv-camera", "engine": "sam-3d-body", "model": repo,
                          "fp16": True, "inference": BODY_INFERENCE, "validFrames": valid_n,
                          "backfilled": len(bad), "msPerFrame": body_ms,
                          "conventionFrame": chk.get("frame"), "conventionIoU": chk.get("iou"),
                          "iouMean": round(iou_sum / iou_n, 4) if iou_n else None,
                          "iouMin": round(iou_min, 4) if iou_min is not None else None,
                          "copiedMaskFrames": copied, "workers": ph["workers"]}
    job.say(f"body: {valid_n}/{n} valid, {len(bad)} backfilled"
            + (f" ({', '.join(f'{k} {v}' for k, v in reasons.items())})" if reasons else "")
            + (f", footprint IoU mean {iou_sum / iou_n:.3f} min {iou_min:.3f}" if iou_n else ""))
    hooks.write_manifest(False)
    return True


# ------------------------------------------------------------------------ run ----


class Hooks:
    """What the phases need from depth.py, passed in so this module never imports it (depth.py
    imports this one): main.py's GPU lock, the resident-model unload, the contract decode chain at a
    given size, the manifest writer and the job error class."""

    def __init__(self, *, lock, unload, decode_args, write_manifest, error):
        self.lock, self.unload, self.decode_args = lock, unload, decode_args
        self.write_manifest, self.error = write_manifest, error


def remove_outputs(run: str) -> None:
    """A previous run's volume files must not outlive a run that does not rewrite them."""
    for f in FILES:
        for p in (os.path.join(run, f), os.path.join(run, f + ".tmp")):
            try:
                os.remove(p)
            except OSError:
                pass


def run(job, hooks: Hooks, src: dict, n: int) -> bool:
    """Geometry then body, after the depth pass has written `n` frames. False when cancelled."""
    devs = selected_devices()
    if not devs:
        raise hooks.error("no-cuda", no_device_reason())
    devs = devs[:max(1, n)]
    gone = missing()
    if gone:
        raise hooks.error("missing-components", f"components not installed: {', '.join(gone)}", gone)
    with hooks.lock:
        # The existing unload convention (depth.py _unload_model): the depth model leaves the card
        # and the allocator's cache is returned, so the workers see the memory SAM 3 does not hold.
        hooks.unload()
    job.say("volumetric: " + ", ".join(f"gpu{d['index']} {d['name']} {d['totalMiB']} MiB" for d in devs))
    if not _geometry(job, hooks, src, n, devs):
        return False
    if job.cancel:
        return False
    return _body(job, hooks, src, n, devs)
