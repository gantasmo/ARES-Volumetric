# ARES depth engine — the server-side half of 2D video -> 2.5D volumetric conversion.
#
# Four routes under /depth/* (mounted by main.py, reached by the browser as /sam/depth/* through
# the dev server's proxy). A job takes a video path and leaves a RUN DIRECTORY on disk:
#   depth.json   the manifest below, rewritten after every batch so a crash leaves a valid run
#   depth.f32    frames x height x width float32 LE, frame-major then row-major, row 0 = image top
# The Node CLI and the browser engine read that pair; nothing here is called at bake time.
#
# THE DECODE FILTER CHAIN IS A CONTRACT, not an implementation detail:
#   ffmpeg -hide_banner -loglevel error -i <video> -vf "<fps=F,>scale=W:H:flags=area"
#          [-frames:v N] -f rawvideo -pix_fmt rgb24 -
# The consumer runs the SAME chain for the colour frames, so depth map i and texture frame i are
# the same source frame. Change the fps expression, the scaler, the flags or the argument order
# here and the two sequences silently drift apart on any clip whose frame rate is not an integer.
#
# NO PER-FRAME NORMALIZATION. The relative models emit affine-invariant inverse depth (larger =
# nearer, arbitrary scale); the metric ones emit metres (larger = farther). Both go to disk raw.
# Min/max-scaling each frame — what the transformers.js depthcloud worker this replaces did, to fit
# 8-bit — makes depth flicker frame to frame on any clip whose nearest or farthest object changes,
# and it is unrecoverable once written.
#
# ASPECT. W and H are both rounded to multiples of 14 (DINOv2 patch size); the few pixels of
# distortion that costs are accepted and undone by the consumer, which knows the source size from
# the manifest. Padding instead would put black bars inside the model's receptive field.
#
# MEASURED ON THIS MACHINE (RTX 2080 Ti, sm_75, fp16, torch 2.6.0+cu124, transformers 5.13.0;
# 90 frames of 640x360 at 518x294, batch 8, weights already in the HF cache):
#   small                4.66 ms/frame inference, 1.6 s wall,  peak +467 MiB,   95 MB on disk
#   base                 7.99 ms/frame inference, 1.9 s wall,  peak +782 MiB,  372 MB
#   large               19.79 ms/frame inference, 3.5 s wall,  peak +1528 MiB, 1280 MB
#   metric-indoor-small  3.98 ms/frame inference, 1.2 s wall,  peak +407 MiB,   95 MB
# Wall covers the whole job including the model switch, and is DECODE-bound below Large: 90 frames
# in 1.6 s is 56 fps end to end while the card is busy for 0.42 s of it. The transformers.js
# depthcloud worker this replaces runs Small at 320 px, ~8 fps (125 ms/frame), 8-bit.
#
# FAULT ISOLATION IS THE LAW, same as track.py: the model loads LAZILY on the first job and the
# failure is latched, so a depth failure 503s here alone and can never touch /segment.
#
# LOCKING. One background worker thread runs jobs strictly one at a time; it takes main.py's model
# lock per BATCH, never for a whole job, so an interactive /segment click interleaves at one
# batch's latency (39-157 ms at batch 8, measured above) instead of blocking for the minutes a long
# clip takes.
#
# THREE MODEL FAMILIES. small/base/large and metric-* are per-frame transformers checkpoints
# (manifest "temporal": "none"). video-small/base/large are Video-Depth-Anything (depth_vda.py):
# upstream code under tools/ext, checkpoints under models/, 32-frame windows stitched as they
# arrive, scale and shift consistent across the clip at the model level ("temporal": "model", so the
# encoder can skip its per-frame scale/shift alignment). A missing component answers 409 with
# {"error", "missing": [...]} before a job exists.
#
# SUBJECT MASK. `subject` (a text prompt) runs a SAM 3 pass first (depth_subject.py): mask.u8 is
# written next to depth.f32, and the depth pass sees every frame with the background painted black.
# Two decodes of the same frame set, mask pass then depth pass, so a prompt that matches nothing
# fails before any depth is computed and the two models are never active on the card together.
#
# VOLUMETRIC. `volumetric: true` (requires `subject`) adds two phases after the depth pass, run by
# depth_volume.py as one volume_worker.py process per CUDA device: "geometry" (MoGe-2 metric depth,
# normals and intrinsics from the unmasked frame) then "body" (a SAM 3D Body mesh per frame from the
# subject box). Their files join depth.f32 in the run directory and their keys join depth.json
# (intrinsics, metric, normals, body); `done` stays false until both phases have finished.

import json
import os
import re
import shutil
import subprocess
import threading
import time
from queue import Queue

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import depth_subject as _subject
import depth_vda as _vda
import depth_volume as _volume

router = APIRouter(prefix="/depth")

# Every model is a public, ungated, Apache-2.0 or CC-BY-NC-4.0 transformers-format repo, so
# from_pretrained downloads it without a token. Value is (repo id, kind). kind is what the manifest
# reports and what tells the consumer which way the numbers run.
DEPTH_MODELS: dict[str, tuple[str, str]] = {
    "small":                ("depth-anything/Depth-Anything-V2-Small-hf",                "relative-disparity"),
    "base":                 ("depth-anything/Depth-Anything-V2-Base-hf",                 "relative-disparity"),
    "large":                ("depth-anything/Depth-Anything-V2-Large-hf",                "relative-disparity"),
    "metric-indoor-small":  ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",  "metric-depth"),
    "metric-indoor-base":   ("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",   "metric-depth"),
    "metric-indoor-large":  ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",  "metric-depth"),
    "metric-outdoor-small": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", "metric-depth"),
    "metric-outdoor-base":  ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",  "metric-depth"),
    "metric-outdoor-large": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "metric-depth"),
}



def _is_vda(key: str) -> bool:
    return key in _vda.VDA_MODELS


def _model_info(key: str) -> tuple[str, str, str]:
    """(model id for the manifest, kind, temporal). Video-Depth-Anything emits relu'd
    affine-invariant inverse depth, the same convention as the relative V2 checkpoints
    (depth_vda.py header), and is consistent in scale and shift across the clip."""
    if _is_vda(key):
        return _vda.VDA_MODELS[key]["repo"], "relative-disparity", "model"
    repo_id, kind = DEPTH_MODELS[key]
    return repo_id, kind, "none"


ALL_MODELS = list(DEPTH_MODELS) + list(_vda.VDA_MODELS)

DEPTH_PATCH = 14                 # DINOv2 patch size; every input edge must be a multiple of it
DEPTH_DEFAULT_WIDTH = 518        # the checkpoints' training edge (preprocessor_config size 518)
DEPTH_MAX_WIDTH = int(os.environ.get("DEPTH_MAX_WIDTH", "1540"))
DEPTH_MAX_BATCH = int(os.environ.get("DEPTH_MAX_BATCH", "64"))
# A finished job is kept this long so a client that lost the response can still read its status,
# then dropped — the log and the counters are all that is retained, never the maps.
DEPTH_JOB_TTL_S = float(os.environ.get("DEPTH_JOB_TTL_S", "600"))
# How long the worker waits for main.py's background loader before failing the job. Sized past the
# 30-60 s the launcher documents for the SAM load (run-sam-service.ps1:80), because the alternative
# is two multi-gigabyte loads racing for the same card.
DEPTH_LOAD_WAIT_S = float(os.environ.get("DEPTH_LOAD_WAIT_S", "180"))
DEPTH_LOG_KEEP = 200             # per job; /depth/status returns the last 20
# One byte per frame beside mask.u8: 1 where the tracker found the subject, 0 where the mask pass
# copied a neighbouring frame's mask (see _run_mask_pass).
MASK_DETECTED_FILE = "mask-detected.u8"
# frames x H x W uint8 beside mask.u8: the SAM 3 tracker's object id on each pixel of the mask, 0
# elsewhere; a copied frame copies its ids too. One id per tracked person until the tracker loses
# every object and reseeds, after which the same person comes back under a new id.
MASK_IDS_FILE = "mask-ids.u8"

# Set by configure(), called from main.py. This module never imports main — main imports it — so it
# stays importable, and py_compile-able, on a machine with no GPU and no weights.
DEVICE = "cpu"
DEPTH_DTYPE = None
_model_lock = threading.Lock()
_load_done: threading.Event | None = None


def configure(*, device, dtype, load_done, model_lock=None) -> None:
    """Hand this module main.py's resolved device, dtype, loader Event and GPU lock.

    dtype is forced to fp32 off CUDA: main._best_dtype already does that, but a caller passing fp16
    to a CPU service would get a model that runs at a few seconds per frame rather than failing.

    model_lock defaults to a private lock so the module works standalone (a test harness importing
    depth.py alone), but main.py passes ITS lock, which is the point: depth, SAM, /upscale and
    /detail then never hold the card at the same time."""
    global DEVICE, DEPTH_DTYPE, _model_lock, _load_done
    import torch

    DEVICE = device
    DEPTH_DTYPE = dtype if device == "cuda" else torch.float32
    if model_lock is not None:
        _model_lock = model_lock
    _load_done = load_done


def _dtype_name() -> str:
    """The manifest's three-value dtype vocabulary, not torch's spelling: the consumer switches on
    it and "torch.float16" is not one of the values the schema allows."""
    return {"torch.float16": "fp16", "torch.bfloat16": "bf16",
            "torch.float32": "fp32"}.get(str(DEPTH_DTYPE), str(DEPTH_DTYPE).replace("torch.", ""))


# ------------------------------------------------------------------- weight cache ----
# Same resolution order as main.py's _resolve_sam3_dir (:45): whatever `hf download` or a previous
# from_pretrained put in the shared cache is used in place, and only a genuinely missing snapshot
# is downloaded. Resolving to the snapshot PATH rather than the repo id also means a run works with
# no network at all once the weights are there.


def _hub_cache_dir() -> str:
    return os.environ.get("HUGGINGFACE_HUB_CACHE") or (
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME")
        else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))


def _cached_snapshot(repo_id: str) -> str | None:
    """The local snapshot dir for a repo, or None. Requires the WEIGHTS, not just config.json: an
    interrupted download leaves the small files behind, and reporting that as cached turns a
    "ready" badge into a surprise 400 MB download."""
    repo = os.path.join(_hub_cache_dir(), "models--" + repo_id.replace("/", "--"))
    snaps = os.path.join(repo, "snapshots")
    if not os.path.isdir(snaps):
        return None
    try:
        revs = sorted(os.listdir(snaps))
    except OSError:
        return None
    try:    # refs/main names the current revision; prefer it when it is there
        with open(os.path.join(repo, "refs", "main"), encoding="utf-8") as fh:
            head = fh.read().strip()
        if head in revs:
            revs.insert(0, revs.pop(revs.index(head)))
    except OSError:
        pass
    for rev in revs:
        snap = os.path.join(snaps, rev)
        if not os.path.isfile(os.path.join(snap, "config.json")):
            continue
        if any(os.path.isfile(os.path.join(snap, w))
               for w in ("model.safetensors", "pytorch_model.bin")):
            return snap
    return None


# ------------------------------------------------------------------------- model ----

_model = None            # the resident AutoModelForDepthEstimation
_model_key = None        # its registry key, or None
_model_mean = None       # (1, 3, 1, 1) device tensors from the checkpoint's own image processor
_model_std = None
_model_rescale = 1.0 / 255.0
_depth_error = None      # latched: set once, never retried, reported on /health
_vda_error = None        # the same latch for the Video-Depth-Anything family, kept apart so a broken
                         # clone under tools/ext cannot disable the transformers checkpoints
_depth_loading = False


def _unload_model() -> None:
    """One model resident at a time. Large is 335.3 M parameters (+1528 MiB peak measured at batch
    8), so switching keys without freeing first would put two of them on an 11 GB card that is
    already holding SAM 3's ~1.6 GiB."""
    global _model, _model_key, _model_mean, _model_std
    import torch

    _model, _model_key, _model_mean, _model_std = None, None, None, None
    import gc
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _ensure_model(key: str, log):
    """Lazy load with a latched error, following _ensure_upscaler (main.py:597) exactly.

    Runs on the worker thread, AFTER the wait on main.py's _load_done, which is also what closes
    the transformers-import race track.py:235 has to loop around: main.py's loader thread is the
    one first executing transformers/__init__.py, and by the time it sets that Event it has
    returned from its own `from transformers import ...`."""
    global _model, _model_key, _model_mean, _model_std, _model_rescale, _depth_error, _depth_loading
    if _model is not None and _model_key == key:
        return _model
    if _is_vda(key):
        return _ensure_vda(key, log)
    if _depth_error is not None:
        raise RuntimeError(f"depth model failed to load: {_depth_error}")

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    repo_id, _kind = DEPTH_MODELS[key]
    src = _cached_snapshot(repo_id) or repo_id
    _depth_loading = True
    try:
        if _model is not None:
            log(f"unloading {_model_key}")
            _unload_model()
        t0 = time.time()
        if src == repo_id:
            log(f"downloading {repo_id} (not in the HF cache)")
        # The processor is read for its normalization constants ONLY (image_mean/image_std/
        # rescale_factor). Its resize path is never run: ffmpeg's area scaler already produced the
        # exact tensor size, and routing every frame through PIL would cost more than the forward.
        proc = AutoImageProcessor.from_pretrained(src)
        model = AutoModelForDepthEstimation.from_pretrained(src, dtype=DEPTH_DTYPE)
        model = model.to(DEVICE).eval()
        mean = torch.tensor(list(proc.image_mean), dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor(list(proc.image_std), dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
        _model, _model_key, _model_mean, _model_std = model, key, mean, std
        _model_rescale = float(getattr(proc, "rescale_factor", 1.0 / 255.0))
        params = sum(p.numel() for p in model.parameters())
        msg = (f"loaded {repo_id} ({params / 1e6:.1f} M params) on {DEVICE} ({_dtype_name()}) in "
               f"{time.time() - t0:.1f}s")
        log(msg)
        print(f"[ares-sam] depth: {msg}")
        return _model
    except Exception as e:
        # An OOM is a transient collision, not a broken install; latching it would 503 /depth/* for
        # the life of the process because a job started while something else was allocating.
        # torch.cuda.OutOfMemoryError subclasses RuntimeError, so it has to be named to be caught.
        if type(e).__name__ == "OutOfMemoryError":
            _unload_model()
            raise RuntimeError("depth model out of device memory: retry once the card is free")
        _depth_error = f"{type(e).__name__}: {e}"
        print(f"[ares-sam] depth model failed to load: {_depth_error} — /depth/run will fail")
        raise RuntimeError(f"depth model failed to load: {_depth_error}")
    finally:
        _depth_loading = False


def _ensure_vda(key: str, log):
    """The Video-Depth-Anything load. fp32 weights on the device, forward under autocast: upstream's
    recipe (depth_vda.py header). Normalization is ImageNet mean/std on x/255, upstream
    video_depth.py:87."""
    global _model, _model_key, _model_mean, _model_std, _model_rescale, _vda_error, _depth_loading
    if _vda_error is not None:
        raise RuntimeError(f"video depth model failed to load: {_vda_error}")
    gone = _vda.missing(key)
    if gone:
        raise RuntimeError(f"video depth components missing: {', '.join(gone)}")

    import torch

    _depth_loading = True
    try:
        if _model is not None:
            log(f"unloading {_model_key}")
            _unload_model()
        t0 = time.time()
        model = _vda.build(key).to(DEVICE)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
        _model, _model_key, _model_mean, _model_std, _model_rescale = model, key, mean, std, 1.0 / 255.0
        params = sum(p.numel() for p in model.parameters())
        msg = (f"loaded {_vda.VDA_MODELS[key]['repo']} ({params / 1e6:.1f} M params) on {DEVICE} "
               f"(fp32 weights, {_dtype_name()} autocast) in {time.time() - t0:.1f}s")
        log(msg)
        print(f"[ares-sam] depth: {msg}")
        return _model
    except Exception as e:
        if type(e).__name__ == "OutOfMemoryError":
            _unload_model()
            raise RuntimeError("depth model out of device memory: retry once the card is free")
        _vda_error = f"{type(e).__name__}: {e}"
        print(f"[ares-sam] video depth model failed to load: {_vda_error}")
        raise RuntimeError(f"video depth model failed to load: {_vda_error}")
    finally:
        _depth_loading = False


# --------------------------------------------------------------------- ffmpeg ----


def _tools(hint: str | None) -> tuple[str, str | None]:
    """(ffmpeg, ffprobe). ffprobe is looked for NEXT TO the given ffmpeg first: the Node side passes
    the binary it bundled or detected, and a build directory's own ffprobe is the one guaranteed to
    agree with it about the container. PATH is the fallback, and None means _probe parses
    `ffmpeg -i` stderr instead."""
    ff = hint or shutil.which("ffmpeg") or "ffmpeg"
    if hint and os.path.dirname(hint):
        cand = os.path.join(os.path.dirname(os.path.abspath(hint)),
                            os.path.basename(hint).replace("ffmpeg", "ffprobe"))
        if os.path.isfile(cand):
            return ff, cand
    return ff, shutil.which("ffprobe")


def _no_window() -> dict:
    """Windows spawns a console for every child of a windowless parent; the dev server hit the same
    thing (commit 0e4dbd2) polling every two seconds. Harmless but visible, so suppressed."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags} if flags else {}


def _ratio(text) -> float | None:
    """r_frame_rate arrives as "30000/1001", not a float."""
    if not text:
        return None
    try:
        s = str(text).strip()
        if "/" in s:
            num, den = s.split("/", 1)
            den = float(den)
            return float(num) / den if den else None
        return float(s)
    except (TypeError, ValueError):
        return None


def _probe(video: str, ffmpeg: str, ffprobe: str | None) -> dict:
    """Source geometry and timing. Every field is optional: a stream with no nb_frames and no
    duration (a raw pipe-ish container) still runs, it just cannot report a `total` up front."""
    if ffprobe:
        try:
            out = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate,nb_frames,duration", "-show_entries",
                 "format=duration", "-of", "json", video],
                capture_output=True, timeout=60, **_no_window())
            info = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
            streams = info.get("streams") or []
            if streams:
                s = streams[0]
                dur = s.get("duration") or (info.get("format") or {}).get("duration")
                return {
                    "width": int(s["width"]) if s.get("width") else None,
                    "height": int(s["height"]) if s.get("height") else None,
                    "fps": _ratio(s.get("r_frame_rate")),
                    "frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
                    "duration": float(dur) if dur not in (None, "N/A") else None,
                }
        except Exception as e:
            print(f"[ares-sam] depth: ffprobe failed ({type(e).__name__}: {e}); parsing ffmpeg -i")

    # ffmpeg -i with no output is an error exit that prints the stream table to stderr. It carries
    # size, rate and duration but never a frame count, which is why it is the fallback and not the
    # primary: `total` then has to come from duration alone.
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", video],
                         capture_output=True, timeout=60, **_no_window())
    err = out.stderr.decode("utf-8", "replace")
    m = re.search(r"Video:.*?,\s*(\d+)x(\d+)", err, re.S)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    m = re.search(r"([\d.]+)\s*fps", err)
    fps = float(m.group(1)) if m else None
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    dur = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else None
    if w is None:
        raise RuntimeError(f"could not read the video's geometry: {err.strip()[-400:]}")
    return {"width": w, "height": h, "fps": fps, "frames": None, "duration": dur}


def _fmt_fps(v: float) -> str:
    """The fps filter argument, formatted so the Node consumer's chain is byte-identical. repr, not
    %g: %g rounds 30000/1001 to "29.97", which is a DIFFERENT sampling grid from 29.970029970029973
    and drifts by a frame every ~33 s against the colour pass."""
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


def _infer_size(infer_width: int, src_w: int, src_h: int) -> tuple[int, int]:
    w = max(DEPTH_PATCH, round(infer_width / DEPTH_PATCH) * DEPTH_PATCH)
    h = max(DEPTH_PATCH, round((w * src_h / src_w) / DEPTH_PATCH) * DEPTH_PATCH)
    return int(w), int(h)


def _read_exact(stream, n: int) -> bytes:
    """A pipe read returns what is available, not what was asked for; a frame split across two
    reads would otherwise be handed to the model as a torn image and silently produce garbage."""
    chunks, got = [], 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


# ---------------------------------------------------------------------- jobs ----


class DepthJobError(RuntimeError):
    """A job failure the client can switch on: `code` goes to /depth/status as errorCode and
    `missing` lists component ids, the same vocabulary as the 409 body of /depth/run."""

    def __init__(self, code: str, message: str, missing: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.missing = missing or []


class DepthJob:
    """One conversion. Everything /depth/status reports, plus the cancel flag and the ffmpeg child
    the cancel has to kill — a decoder blocked writing into a pipe nobody drains does not notice a
    flag."""

    def __init__(self, jid: str, req: "DepthRunRequest", width: int, batch: int):
        self.id = jid
        self.video = req.video
        self.out = req.out
        self.model = req.model
        self.fps = req.fps
        self.max_frames = req.maxFrames
        self.infer_width = width
        self.batch = batch
        self.ffmpeg = req.ffmpeg
        self.subject = (req.subject or "").strip() or None
        self.state = "queued"        # queued | loading | running | done | error | cancelled
        # Which pass is on the card. "mask" exists only for a subject run and always comes first.
        self.phase: str | None = None
        self.done = 0
        self.total: int | None = None
        self.mask_done = 0
        self.mask_total: int | None = None
        self.mask_ms_per_frame: float | None = None
        self.mask_info: dict | None = None       # the manifest's "mask" object, once the pass ends
        self.ms_per_frame: float | None = None
        self.error: str | None = None
        self.error_code: str | None = None
        self.missing: list[str] = []
        self.width: int | None = None
        self.height: int | None = None
        self.out_fps: float | None = None
        self.cancel = False
        self.proc: subprocess.Popen | None = None
        # Volumetric phases (depth_volume.py): per-phase counters, the manifest keys each finished
        # phase adds, per-worker statistics, and the worker processes a cancel has to kill.
        self.volumetric = bool(req.volumetric)
        self.vphases: dict[str, dict] = {
            p: {"done": 0, "total": None, "msPerFrame": None, "gpus": None, "framesPerSecond": None}
            for p in ("geometry", "body")}
        self.volume: dict = {}
        self.volume_stats: dict = {}
        self.volume_check: dict | None = None
        self.workers: list[subprocess.Popen] = []
        self.log: list[str] = []
        self.created = time.time()
        self.finished: float | None = None

    def say(self, line: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')} {line}")
        del self.log[:-DEPTH_LOG_KEEP]

    def status(self) -> dict:
        """done/total/msPerFrame stay the DEPTH pass's, as they were before the mask pass existed;
        `phase` says which pass is running and `phases` carries done/total for each, so a client
        draws one bar per pass or one bar over both."""
        phases = {"depth": {"done": self.done, "total": self.total, "msPerFrame": self.ms_per_frame}}
        if self.subject:
            phases = {"mask": {"done": self.mask_done, "total": self.mask_total,
                               "msPerFrame": self.mask_ms_per_frame}, **phases}
        if self.volumetric:
            phases.update({p: dict(v) for p, v in self.vphases.items()})
        return {
            "state": self.state, "phase": self.phase, "phases": phases,
            "done": self.done, "total": self.total,
            "msPerFrame": self.ms_per_frame, "error": self.error,
            "errorCode": self.error_code, "missing": self.missing,
            "width": self.width, "height": self.height, "fps": self.out_fps,
            "model": self.model, "subject": self.subject, "mask": self.mask_info,
            "volumetric": self.volumetric, "volume": self.volume or None,
            "log": self.log[-20:],
        }


_jobs: dict[str, DepthJob] = {}
_jobs_lock = threading.Lock()      # guards _jobs only, never held across ffmpeg or GPU work
_queue: "Queue[str]" = Queue()
# One job at a time. The single worker thread below already serializes, but the lock is what makes
# that a property of this file rather than of how many threads happen to be started: a second
# worker would otherwise put two decodes and two models on the card with no error anywhere.
_job_lock = threading.Lock()
_worker_started = False


def _new_job_id() -> str:
    return "dep_" + os.urandom(4).hex()


def _reap_jobs() -> None:
    """Finished jobs are dropped DEPTH_JOB_TTL_S after they end. Called from the two routes that
    already hold no lock but _jobs_lock, so no timer thread is needed for a dict of small records."""
    now = time.time()
    for jid, job in list(_jobs.items()):
        if job.finished is not None and now - job.finished > DEPTH_JOB_TTL_S:
            _jobs.pop(jid, None)


def _start_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    threading.Thread(target=_worker_loop, daemon=True, name="depth-worker").start()


def _worker_loop() -> None:
    while True:
        jid = _queue.get()
        with _jobs_lock:
            job = _jobs.get(jid)
        if job is None:
            continue
        if job.cancel:                      # cancelled while queued: never touch the GPU for it
            job.state = "cancelled"
            job.finished = time.time()
            continue
        with _job_lock:
            try:
                _run_job(job)
            except Exception as e:
                job.state = "error"
                job.error = f"{type(e).__name__}: {e}"
                if isinstance(e, DepthJobError):
                    job.error, job.error_code, job.missing = str(e), e.code, list(e.missing)
                job.say(f"error: {job.error}")
                print(f"[ares-sam] depth job {job.id} failed: {job.error}")
            finally:
                job.finished = time.time()
                if job.proc is not None:
                    job.proc = None


def _write_manifest(job: DepthJob, src: dict, frames: int, done: bool) -> None:
    """Rewritten after every batch. A job killed mid-run therefore leaves a manifest whose `frames`
    matches the bytes actually in depth.f32 (done:false), which the consumer can play as-is instead
    of reading past the end of the file."""
    repo_id, kind, temporal = _model_info(job.model)
    manifest = {
        "schema": "ares-depth/1",
        "engine": "service",
        "model": repo_id,
        "modelKey": job.model,
        "kind": kind,
        # "model": scale and shift are consistent across the clip as written (Video-Depth-Anything),
        # so the consumer skips its per-frame alignment. "none": every frame stands alone.
        "temporal": temporal,
        "width": job.width, "height": job.height,
        "frames": frames,
        "fps": job.out_fps,
        "sampling": {"fps": job.fps, "maxFrames": job.max_frames},
        "video": os.path.abspath(job.video),
        "sourceFps": src.get("fps"), "sourceWidth": src.get("width"),
        "sourceHeight": src.get("height"), "sourceFrames": src.get("frames"),
        "sourceDurationS": src.get("duration"),
        "msPerFrame": job.ms_per_frame,
        "device": DEVICE, "dtype": _dtype_name(),
        "done": done,
    }
    if job.mask_info is not None:        # absent without `subject`: the key's presence IS the flag
        manifest["mask"] = job.mask_info
    # intrinsics / metric / normals / body, each added once its phase has finished writing its files
    manifest.update(job.volume)
    tmp = os.path.join(job.out, "depth.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, os.path.join(job.out, "depth.json"))


def _decode_args(job: DepthJob, ffmpeg: str, width: int, height: int) -> tuple[list[str], str]:
    """THE contract chain from the header, at the given output size. The mask pass calls it with
    1008x1008 and the depth pass with the map size; everything that selects frames (`fps=`,
    `-frames:v`) is identical between the two, so frame i is the same source frame in both."""
    vf = (f"fps={_fmt_fps(job.out_fps)}," if job.fps else "") + f"scale={width}:{height}:flags=area"
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", job.video, "-vf", vf]
    if job.max_frames:
        args += ["-frames:v", str(int(job.max_frames))]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return args, vf


class _Decoder:
    """One ffmpeg rawvideo child and its pipe. read(n) returns up to n whole frames; fewer means
    the stream ended. close() reaps the child and returns (returncode, stderr text)."""

    def __init__(self, job: DepthJob, args: list[str], width: int, height: int):
        self.job, self.width, self.height = job, width, height
        self.frame_bytes = width * height * 3
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.frame_bytes * 2, **_no_window())
        job.proc = self.proc
        self._err: list[bytes] = []
        # stderr is drained on its own thread: -loglevel error usually leaves it empty, but a source
        # ffmpeg complains about fills the 64 KB pipe buffer and deadlocks the decode against a
        # reader that is only ever reading stdout.
        proc = self.proc
        threading.Thread(target=lambda: self._err.append(proc.stderr.read()),
                         daemon=True, name="depth-ffmpeg-err").start()

    def read(self, n: int) -> list[np.ndarray]:
        frames = []
        while len(frames) < n:
            raw = _read_exact(self.proc.stdout, self.frame_bytes)
            if len(raw) < self.frame_bytes:          # end of stream, or a torn trailing frame
                break
            frames.append(np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3))
        return frames

    def close(self) -> tuple[int | None, str]:
        proc = self.proc
        if self.job.cancel and proc.poll() is None:
            proc.kill()                  # a decoder blocked writing into the pipe ignores flags
        try:
            proc.stdout.close()      # EPIPE ends a decoder still writing, e.g. after an OOM
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Raising here would replace whatever exception put us in this finally with a timeout,
            # and leave the child running for the life of the service either way.
            proc.kill()
            print(f"[ares-sam] depth job {self.job.id}: ffmpeg did not exit; killed")
        self.job.proc = None
        err = (self._err[0] or b"").decode("utf-8", "replace").strip() if self._err else ""
        return proc.returncode, err


def _check_decode(job: DepthJob, code: int | None, err: str, frames: int) -> None:
    if job.cancel:
        return
    if code not in (0, None):
        raise RuntimeError(f"ffmpeg exited {code}: {err[-400:] or 'no stderr'}")
    if frames == 0:
        raise RuntimeError(f"ffmpeg produced no frames: {err[-400:] or 'no stderr'}")


# ----------------------------------------------------------------- mask pass ----


def _run_mask_pass(job: DepthJob, ffmpeg: str) -> bool:
    """Write <run>/mask.u8: frames x H x W uint8, 0 or 255, the depth map's size, order and
    orientation. Returns False when the job was cancelled.

    A frame where the tracker reports the subject absent repeats the last mask, and the frames
    before the first detection take the first one: an empty mask would hand the depth model an
    all-black frame, which is the failure docs/rgbd-rebuild-pipeline.md's field notes paid for.
    `filled` in the manifest counts those frames, and <run>/mask-detected.u8 (one byte per frame,
    1 = the tracker found the subject, 0 = a copied mask) says which they are: the body phase of a
    volumetric job fits no mesh on a copied mask, which may hold no one. <run>/mask-ids.u8 holds
    the tracker's object id per pixel (MASK_IDS_FILE), and `objects` in the manifest lists each
    id's detected frames, first and last frame and mean coverage."""
    import torch

    job.phase = "mask"
    job.mask_total = job.total
    masker = _subject.SubjectMasker(job.subject, job.height, job.width, job.say)
    try:
        masker.prewarm()
        with _model_lock:              # the tracker load allocates; share SAM's GPU lock for it
            masker.load()
    except _subject.SubjectUnavailable as e:
        raise DepthJobError("missing-components", str(e), e.missing)

    edge = _subject.SAM_EDGE
    args, vf = _decode_args(job, ffmpeg, edge, edge)
    job.say("mask pass: ffmpeg -vf " + vf)
    dec = _Decoder(job, args, edge, edge)
    job.state = "running"
    path = os.path.join(job.out, "mask.u8")
    det_path = os.path.join(job.out, MASK_DETECTED_FILE)
    ids_path = os.path.join(job.out, MASK_IDS_FILE)
    n, filled, lead, inside, gpu_s = 0, 0, 0, 0.0, 0.0
    last: np.ndarray | None = None
    last_ids: np.ndarray | None = None
    objects: dict[int, dict] = {}      # id -> {frames, first, last, cover}
    t_pass = time.time()
    try:
        with open(path, "wb") as out_f, open(det_path, "wb") as det_f, open(ids_path, "wb") as ids_f:
            while not job.cancel:
                got = dec.read(1)
                if not got:
                    break
                t0 = time.time()
                with _model_lock:      # per FRAME, so /segment interleaves at one frame's latency
                    step = masker.step(n, got[0])
                gpu_s += time.time() - t0
                m, ids = step if step is not None else (None, None)
                if m is None or not m.any():
                    m, ids = None, None
                det_f.write(b"\x00" if m is None else b"\x01")
                if ids is not None:
                    present, counts = np.unique(ids[ids > 0], return_counts=True)
                    for k, c in zip(present.tolist(), counts.tolist()):
                        o = objects.setdefault(k, {"frames": 0, "first": n, "last": n, "cover": 0.0})
                        o["frames"] += 1
                        o["last"] = n
                        o["cover"] += c / ids.size
                if m is None and last is None:
                    lead += 1          # nothing found yet: a placeholder, rewritten below
                    out_f.write(bytes(job.height * job.width))
                    ids_f.write(bytes(job.height * job.width))
                else:
                    if m is None:
                        m, ids, filled = last, last_ids, filled + 1
                    last, last_ids = m, ids
                    inside += float(np.count_nonzero(m)) / m.size
                    out_f.write(np.ascontiguousarray(m).tobytes())
                    ids_f.write(np.ascontiguousarray(ids, dtype=np.uint8).tobytes())
                n += 1
                job.mask_done = n
                if job.mask_total is not None and n > job.mask_total:
                    job.mask_total = n
                job.mask_ms_per_frame = round(gpu_s * 1000.0 / n, 2)
            if last is not None and lead and not job.cancel:
                # The frames before the first detection take the first mask. It is `lead` frames
                # back in the file, so it is read from there rather than kept in memory.
                out_f.flush()
                ids_f.flush()
                for p in (ids_path, path):
                    with open(p, "r+b") as fix:
                        fix.seek(lead * job.height * job.width)
                        first = fix.read(job.height * job.width)
                        fix.seek(0)
                        for _ in range(lead):
                            fix.write(first)
                inside += lead * (np.count_nonzero(np.frombuffer(first, np.uint8)) / len(first))
                filled += lead
    finally:
        code, err = dec.close()
        stats = (masker.encoder_s, masker.track_s, masker.detect_s, masker.detections, masker.reseeds,
                 masker.next_obj - 1)
        with _model_lock:
            masker.close()             # frees the session's memory bank before the depth model loads
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    if job.cancel:
        return False
    _check_decode(job, code, err, n)
    job.mask_total = n
    coverage = inside / n
    if last is None or coverage < _subject.MIN_COVERAGE:
        for p in (path, det_path, ids_path):     # never leave a run whose every frame is black
            try:
                os.remove(p)
            except OSError:
                pass
        raise DepthJobError("subject-not-found",
                            f"subject '{job.subject}' was not found in the clip "
                            f"(mean coverage {coverage:.5f} over {n} frames)")
    job.mask_info = {"file": "mask.u8", "prompt": job.subject, "engine": _subject.ENGINE,
                     "coverage": round(coverage, 5), "frames": n, "filled": filled,
                     "detected": MASK_DETECTED_FILE, "ids": MASK_IDS_FILE,
                     "objects": [{"id": k, "frames": o["frames"], "first": o["first"], "last": o["last"],
                                  "coverage": round(o["cover"] / max(1, o["frames"]), 5)}
                                 for k, o in sorted(objects.items())],
                     "msPerFrame": job.mask_ms_per_frame}
    enc, trk, det, ndet, reseeds, nobj = stats
    job.say(f"mask pass done: {n} frames, {job.mask_ms_per_frame} ms/frame (encoder "
            f"{enc * 1000 / n:.1f}, tracker {trk * 1000 / n:.1f}, detector {det * 1000 / n:.1f} over "
            f"{ndet} detections), coverage {coverage:.4f}, {nobj} instance{'s' if nobj != 1 else ''}, "
            f"{filled} filled, {reseeds} reseeds, {time.time() - t_pass:.1f}s wall")
    return True


class _MaskReader:
    """mask.u8 read back in step with the depth decode. A short read repeats the last mask: the two
    passes decode the same frame set, so that is a guard, not a path."""

    def __init__(self, path: str, width: int, height: int):
        self.fh = open(path, "rb")
        self.size = width * height
        self.shape = (height, width, 1)
        self.last: np.ndarray | None = None

    def apply(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for f in frames:
            raw = self.fh.read(self.size)
            if len(raw) == self.size:
                self.last = np.frombuffer(raw, np.uint8).reshape(self.shape)
            out.append(np.where(self.last > 0, f, 0).astype(np.uint8) if self.last is not None else f)
        return out

    def close(self) -> None:
        self.fh.close()


# ---------------------------------------------------------------- depth pass ----


def _to_input(frames: list[np.ndarray]):
    """uint8 HWC frames -> normalized float32 NCHW on the device. Normalized in fp32 with the
    checkpoint's own constants (read from its image processor at load) and cast by the caller."""
    import torch
    x = torch.from_numpy(np.stack(frames)).to(DEVICE)
    return x.permute(0, 3, 1, 2).float().mul_(_model_rescale).sub_(_model_mean).div_(_model_std)


def _autocast():
    import torch
    return torch.autocast(device_type="cuda", dtype=DEPTH_DTYPE, enabled=(DEVICE == "cuda"))


def _frames_per_frame_model(job: DepthJob, model, read):
    """The per-frame family: batches of job.batch, each frame on its own. Yields (array, seconds)."""
    import torch

    while not job.cancel:
        frames = read(job.batch)
        if not frames:
            return
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with _model_lock:          # per BATCH, so /segment interleaves at one batch's latency
            depth = model(pixel_values=_to_input(frames).to(DEPTH_DTYPE)).predicted_depth
            if depth.ndim == 4:                       # (B, 1, H', W') on some heads
                depth = depth.squeeze(1)
            if depth.shape[-2:] != (job.height, job.width):
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1).float(), size=(job.height, job.width),
                    mode="bilinear", align_corners=False).squeeze(1)
            depth = depth.float()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            arr = depth.cpu().numpy()
        yield arr, dt


def _frames_vda(job: DepthJob, model, read):
    """Video-Depth-Anything, streamed: upstream's 32-frame windows and stitching (depth_vda.py
    header), with one window of inputs, the previous window's inputs and 8 held-back depth frames
    resident whatever the clip length. Yields (array of FINAL frames, seconds)."""
    import torch

    n, step, overlap = _vda.INFER_LEN, _vda.STEP, _vda.OVERLAP
    aligner = _vda.WindowAligner()
    prev = None
    start, real, emitted, eof = 0, 0, 0, False
    while not job.cancel:
        # Upstream's `for frame_id in range(0, N, 22)`: a window starts only on a real frame.
        if start and eof and real <= start:
            break
        want, base = (n, 0) if start == 0 else (step, overlap)
        new = [] if eof else read(want)
        eof = eof or len(new) < want
        if start == 0 and not new:
            return
        real += len(new)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with _model_lock:          # per WINDOW
            x = torch.empty((n, 3, job.height, job.width), dtype=torch.float32, device=DEVICE)
            if start:
                x[:overlap] = prev[_vda.KEYFRAMES]    # the previous window's INPUTS, not its frames
            if new:
                x[base:base + len(new)] = _to_input(new)
            k = base + len(new)
            if k < n:
                x[k:] = x[k - 1]      # upstream pads the clip with its last frame; slot 9 is that
            with _autocast():         # frame whenever a window has no new frames at all
                depth = model(x).float()
            if not bool(torch.isfinite(depth).all()):
                raise RuntimeError(f"video depth model produced non-finite values in the window at "
                                   f"frame {start} ({_dtype_name()} autocast)")
            final = aligner.add(depth)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            arr = final.cpu().numpy()
        prev = x
        if eof:
            arr = arr[:max(0, real - emitted)]        # drop what was only padding
        emitted += len(arr)
        start += step
        if len(arr):
            yield arr, dt
    if job.cancel:
        return
    tail = aligner.flush()
    if tail is not None and real > emitted:
        yield tail[:real - emitted].cpu().numpy(), 0.0


def _run_job(job: DepthJob) -> None:
    import torch

    t_start = time.time()
    # Peak device allocation is reported as this job's OWN addition, not the process figure: SAM 3
    # is resident on the same card (1,654 MiB of the 2,121 MiB process peak on a Small run), so the
    # raw max_memory_allocated says nothing about what a depth model costs or whether a bigger one
    # still fits.
    base_alloc = 0
    if _model is not None and _model_key != job.model:
        # A different checkpoint is resident and is about to be replaced anyway. Dropping it BEFORE
        # the baseline is read keeps "peak +N MiB" this job's own cost (it read +0 MiB for a Base
        # run that followed a Large one), and keeps it off the card during a mask pass.
        with _model_lock:
            job.say(f"unloading {_model_key}")
            _unload_model()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
    job.state = "loading"
    ffmpeg, ffprobe = _tools(job.ffmpeg)
    src = _probe(job.video, ffmpeg, ffprobe)
    if not src.get("width") or not src.get("height"):
        raise RuntimeError("the source has no decodable video stream")
    job.width, job.height = _infer_size(job.infer_width, src["width"], src["height"])
    job.out_fps = float(job.fps) if job.fps else src.get("fps")
    # Best estimate up front, corrected to the true count when the pipe ends. maxFrames is exact;
    # duration x rate is not (a VFR source, or a container whose duration rounds), so the consumer
    # treats it as a progress denominator and the manifest's `frames` as the truth.
    if job.max_frames:
        job.total = int(job.max_frames)
    elif src.get("duration") and job.out_fps:
        job.total = max(1, int(round(src["duration"] * job.out_fps)))
    elif src.get("frames") and not job.fps:
        job.total = int(src["frames"])
    job.say(f"{src['width']}x{src['height']} @ {src.get('fps')} fps -> {job.width}x{job.height} "
            f"@ {job.out_fps} fps, ~{job.total} frames")

    if _load_done is not None and not _load_done.wait(DEPTH_LOAD_WAIT_S):
        raise RuntimeError("sam model still loading: retry shortly")
    os.makedirs(job.out, exist_ok=True)
    for f in ("mask.u8", MASK_DETECTED_FILE, MASK_IDS_FILE):   # a previous run's mask must not outlive a run without one
        try:
            os.remove(os.path.join(job.out, f))
        except OSError:
            pass
    _volume.remove_outputs(job.out)    # nor its volume files a run that does not rewrite them
    _write_manifest(job, src, 0, False)

    mask_peak = ""
    if job.subject:
        if not _run_mask_pass(job, ffmpeg):
            job.state = "cancelled"
            job.say("cancelled during the mask pass")
            return
        if DEVICE == "cuda":
            mask_peak = (f", mask pass peak +{(torch.cuda.max_memory_allocated() - base_alloc) / (1 << 20):.0f}"
                         f" MiB")
            torch.cuda.reset_peak_memory_stats()
        job.state = "loading"

    job.phase = "depth"
    vda = _is_vda(job.model)
    with _model_lock:                  # the load allocates 0.2-1.5 GiB; share SAM's GPU lock for it
        model = _ensure_model(job.model, job.say)
        # One throwaway forward at this job's exact tensor shape, before the clock starts. cuDNN
        # picks its convolution algorithms on the first call for a given shape, and on a short clip
        # that one-time cost IS the measurement: the first batch of a 90-frame Small run measured
        # 90.9 ms/frame, which alone put the job average at 14.15 against the 4.87 it reports with
        # this warm-up in place. It also keeps the stall off the first batch the client waits on.
        t_warm = time.time()
        with torch.inference_mode():
            if vda:
                with _autocast():
                    model(torch.zeros((_vda.INFER_LEN, 3, job.height, job.width),
                                      dtype=torch.float32, device=DEVICE))
            else:
                model(pixel_values=torch.zeros((job.batch, 3, job.height, job.width),
                                               dtype=DEPTH_DTYPE, device=DEVICE))
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        shape = f"{_vda.INFER_LEN if vda else job.batch}x{job.width}x{job.height}"
        job.say(f"warmed {shape} in {time.time() - t_warm:.1f}s")
    if job.cancel:
        job.state = "cancelled"
        return

    args, vf = _decode_args(job, ffmpeg, job.width, job.height)
    job.say("ffmpeg -vf " + vf)
    infer_s, written = 0.0, 0
    masks = _MaskReader(os.path.join(job.out, "mask.u8"), job.width, job.height) if job.subject else None
    dec = _Decoder(job, args, job.width, job.height)
    read = (lambda k: masks.apply(dec.read(k))) if masks is not None else dec.read
    job.state = "running"

    try:
        with open(os.path.join(job.out, "depth.f32"), "wb") as out_f, torch.inference_mode():
            for arr, dt in (_frames_vda if vda else _frames_per_frame_model)(job, model, read):
                infer_s += dt
                # RAW values, no per-frame scaling — see the header. "<f4" is explicit rather than
                # trusting the host's endianness, because the manifest promises little-endian.
                out_f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
                out_f.flush()
                written += len(arr)
                job.done = written
                if job.total is not None and written > job.total:
                    job.total = written          # the estimate was low; never report done > total
                job.ms_per_frame = round(infer_s * 1000.0 / written, 2)
                _write_manifest(job, src, written, False)
    finally:
        code, err = dec.close()
        if masks is not None:
            masks.close()

    if job.cancel:
        job.state = "cancelled"
        job.total = written
        _write_manifest(job, src, written, False)
        job.say(f"cancelled after {written} frames")
        return
    _check_decode(job, code, err, written)
    if job.mask_info is not None and job.mask_info["frames"] != written:
        raise RuntimeError(f"the mask pass decoded {job.mask_info['frames']} frames and the depth "
                           f"pass {written}: the two must be the same frame set")

    job.total = written                  # the estimate is replaced by the count that was written
    peak = ""
    if DEVICE == "cuda":
        peak = (f", peak +{(torch.cuda.max_memory_allocated() - base_alloc) / (1 << 20):.0f} MiB "
                f"(process {torch.cuda.max_memory_allocated() / (1 << 20):.0f} MiB){mask_peak}")
    if job.volumetric:
        _write_manifest(job, src, written, False)    # depth complete; the volume keys follow
        job.say(f"depth pass done: {written} frames, {job.ms_per_frame} ms/frame inference, "
                f"{time.time() - t_start:.1f}s wall{peak}")
        hooks = _volume.Hooks(lock=_model_lock, unload=_unload_model,
                              decode_args=lambda w, h: _decode_args(job, ffmpeg, w, h),
                              write_manifest=lambda done: _write_manifest(job, src, written, done),
                              error=DepthJobError)
        if not _volume.run(job, hooks, src, written):
            job.state = "cancelled"
            job.say(f"cancelled during the {job.phase} phase")
            return
    job.state = "done"
    _write_manifest(job, src, written, True)
    job.say(f"done: {written} frames, {job.ms_per_frame} ms/frame inference, "
            f"{time.time() - t_start:.1f}s wall{peak}")
    print(f"[ares-sam] depth job {job.id} {job.model}: {written} frames at {job.width}x{job.height}, "
          f"{job.ms_per_frame} ms/frame, {time.time() - t_start:.1f}s wall{peak}")


# --------------------------------------------------------------------- routes ----


@router.get("/health")
def depth_health():
    """Loads nothing and takes no lock. `cached` is a filesystem scan of the HF cache, so the UI can
    say which keys are one click away and which are a download before anyone starts a job."""
    with _jobs_lock:
        jobs = sum(1 for j in _jobs.values() if j.state in ("queued", "loading", "running"))
    return {
        "ready": _model is not None,
        "loading": _depth_loading,
        "error": _depth_error,
        "model": _model_key,
        "device": DEVICE,
        "dtype": _dtype_name(),
        # Per key: can it run without fetching anything. For the video keys that is the clone plus
        # the checkpoint, which this service never downloads.
        "cached": {**{k: _cached_snapshot(v[0]) is not None for k, v in DEPTH_MODELS.items()},
                   **{k: not _vda.missing(k) for k in _vda.VDA_MODELS}},
        # Video-Depth-Anything by component id, the same ids /depth/run's 409 body lists.
        "vda": _vda.installed(),
        "vdaError": _vda_error,
        "temporal": {k: _model_info(k)[2] for k in ALL_MODELS},
        # `subject` needs SAM 3's weights on disk; "sam3" is its component id in a 409 body.
        "subject": {"available": _subject.sam3_weights_present(), "engine": _subject.ENGINE},
        # `volumetric`: component ids absent, whether moge and sam_3d_body import (a cached child
        # probe, "probe": "running" until its first answer) and the CUDA devices a job would use.
        "volume": _volume.health(),
        "jobs": jobs,
        "models": ALL_MODELS,
    }


class DepthRunRequest(BaseModel):
    video: str                       # absolute path to the source
    out: str                         # absolute path to the run directory, created if missing
    model: str = "base"
    fps: float | None = None         # resample rate; null keeps every source frame
    maxFrames: int | None = None
    inferWidth: int = DEPTH_DEFAULT_WIDTH
    batch: int = 8
    ffmpeg: str | None = None        # the binary the Node side resolved; null falls back to PATH
    # Text prompt naming the subject ("person"). Set, a SAM 3 pass writes mask.u8 and the depth
    # model sees the background black. Null or empty changes nothing.
    subject: str | None = None
    # The geometry and body phases after the depth pass (depth_volume.py). Needs `subject`.
    volumetric: bool = False


def _missing_components(req: "DepthRunRequest") -> list[str]:
    """Component ids this request needs and the machine does not have. Filesystem checks only."""
    gone = _vda.missing(req.model) if _is_vda(req.model) else []
    if (req.subject or "").strip():
        # Weights absent, or present but the service came up without text segmentation (SAM_TEXT=0,
        # the vit_h fallback, a failed concept load): either way SAM 3 is what is missing. While
        # main.py's loader is still running the second test cannot be answered yet, and the job
        # itself fails with the same component list if it turns out that way.
        if not _subject.sam3_weights_present():
            gone.append("sam3")
        elif _load_done is not None and _load_done.is_set():
            trk = _subject._track_module()
            if trk is None or trk._get_concept_model() is None:
                gone.append("sam3")
    if req.volumetric:
        gone += [c for c in _volume.missing() if c not in gone]
    return gone


@router.post("/run", status_code=202)
def depth_run(req: DepthRunRequest):
    """Accepted, not performed. A first use of a key downloads 95 MB to 1.28 GB before a frame runs
    (Large measured at 104 s here) and a feature-length source runs for minutes; both are past any
    reasonable request timeout. The job id is the handle for /depth/status and /depth/cancel."""
    if req.model not in ALL_MODELS:
        raise HTTPException(status_code=400,
                            detail=f"unknown depth model '{req.model}': one of {', '.join(ALL_MODELS)}")
    if not req.video or not os.path.isfile(req.video):
        raise HTTPException(status_code=400, detail=f"video not found: {req.video}")
    if not req.out:
        raise HTTPException(status_code=400, detail="out must be a directory path")
    try:
        os.makedirs(req.out, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"cannot create out dir {req.out}: {e}")
    if req.fps is not None and not (0 < float(req.fps) <= 240):
        raise HTTPException(status_code=400, detail="fps must be in 0..240, or null")
    if req.maxFrames is not None and int(req.maxFrames) < 1:
        raise HTTPException(status_code=400, detail="maxFrames must be >= 1, or null")

    if req.subject is not None and len(req.subject) > 200:
        raise HTTPException(status_code=400, detail="subject must be at most 200 characters")
    if req.volumetric and not (req.subject or "").strip():
        raise HTTPException(status_code=400, detail="volumetric requires subject: the body phase runs "
                                                    "on the subject mask's box")
    if req.volumetric and DEVICE != "cuda":
        raise HTTPException(status_code=400, detail=f"volumetric requires a CUDA device, service device "
                                                    f"is {DEVICE}")
    if req.volumetric and not _volume.selected_devices():
        # Refused here, not after the mask and depth passes have run.
        raise HTTPException(status_code=400, detail=_volume.no_device_reason())
    # Fail fast and machine-readably: the caller maps these ids onto its installer catalog.
    gone = _missing_components(req)
    if gone:
        return JSONResponse(status_code=409, content={
            "error": f"components not installed: {', '.join(gone)}", "missing": gone})
    if req.volumetric:
        # Files on disk that do not import (a missing transitive package): the cached probe's answer,
        # when it has one for the current component state.
        bad = _volume.import_failure()
        if bad:
            return JSONResponse(status_code=409, content=bad)

    width = max(DEPTH_PATCH, min(DEPTH_MAX_WIDTH, int(req.inferWidth or DEPTH_DEFAULT_WIDTH)))
    batch = max(1, min(DEPTH_MAX_BATCH, int(req.batch or 8)))
    _start_worker()
    job = DepthJob(_new_job_id(), req, width, batch)
    with _jobs_lock:
        _reap_jobs()
        _jobs[job.id] = job
    _queue.put(job.id)
    return {"job": job.id}


@router.get("/status")
def depth_status(job: str = Query(...)):
    with _jobs_lock:
        _reap_jobs()
        j = _jobs.get(job)
    if j is None:
        raise HTTPException(status_code=404, detail=f"unknown depth job: {job}")
    return j.status()


class DepthJobRef(BaseModel):
    job: str


@router.post("/cancel")
def depth_cancel(req: DepthJobRef):
    """Cooperative: the flag is read between batches, and the ffmpeg child is killed so a decoder
    parked on a full pipe stops too. Frames already written stay on disk as a valid short run."""
    with _jobs_lock:
        j = _jobs.get(req.job)
    if j is None:
        raise HTTPException(status_code=404, detail=f"unknown depth job: {req.job}")
    j.cancel = True
    # The volume workers are killed here as well as by the phase loop, which notices the flag within
    # 0.25 s: a worker mid-forward holds its card for up to 1.6 s otherwise.
    for proc in [j.proc, *list(j.workers)]:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
    return {"ok": True}


def health_fields() -> dict:
    """Merged into GET /health by main.py. Loads nothing and takes no lock."""
    with _jobs_lock:
        jobs = sum(1 for j in _jobs.values() if j.state in ("queued", "loading", "running"))
    return {
        "depthReady": _model is not None,
        "depthLoading": _depth_loading,
        "depthModel": _model_key,
        "depthError": _depth_error,
        "depthJobs": jobs,
        "depthVda": _vda.installed(),
    }
