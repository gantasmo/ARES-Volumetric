# ARES SAM 3 video-tracker propagation — step 4 of docs/sam-propagation-plan.md.
#
# Five routes under /track/* (the browser reaches them as /sam/track/*: the dev server's proxy at
# tools/serve.mjs:669 strips the /sam prefix). A selection made once on one frame follows the
# subject through the clip, and the masks come back ALREADY in the edit list's own RLE convention
# — alternating run lengths, 0-run first, packages/core/src/edits.ts:50 rleEncodeMask — so the
# browser stores the array verbatim into the sidecar and the offline encoder bake needs no change
# at all. Nothing here runs at bake time: the network runs once, at authoring time, and its output
# is a binary bitmap on disk.
#
# THE SEAM. facebook/sam3's checkpoint stores exactly ONE vision tower (detector_model.
# vision_encoder.*, 538 tensors; ZERO under tracker_model.vision_encoder.*). Sam3VideoModel builds
# its tracker with remove_vision_encoder=True (modeling_sam3_video.py:512), so the towerless
# tracker is fed by priming its vision-feature cache — the three-call sequence the library itself
# runs per frame at modeling_sam3_video.py:1613-1632, and the one tools/sam-service/track_smoke.py
# measured at 0 cache misses across both the seeded and the propagated forward. Loading
# Sam3TrackerVideoModel instead loads CLEAN on transformers 5.13.0 (measured; the plan's "538
# missing keys" claim is false) but 454.04 M of its 465.78 M parameters are a SECOND copy of the
# tower the concept model already holds — ~908 MiB of idle fp16 VRAM. That duplication, not
# corruption, is why this loads Sam3VideoModel and grafts the resident detector in below.
#
# THE GRAFT SHARES THE WHOLE DETECTOR, not just its tower. Sam3VideoModel's detector_model is a
# Sam3Model (config.detector_config.model_type == "sam3", built at modeling_sam3_video.py:511) —
# the same class, from the same checkpoint, as the concept model main.py already holds resident.
# track_smoke.py measured it at 840.38 M parameters of which 454.04 M are the vision tower, so
# sharing the tower alone still moves 386.34 M non-vision parameters (DETR encoder/decoder, text
# tower, geometry encoder, mask decoder) onto the card at vm.to(DEVICE) — 736.9 MiB of fp16 weights
# this file never forwards, against the 2,473 MiB main.py:113-116 measured free on the 6 GB card it
# ships on. The only detector call here is get_vision_features (:296), which touches vision_encoder
# alone (modeling_sam3.py:2254), so the whole module is shared and nothing of it is duplicated.
#
# MEASURED ON THIS MACHINE (track_smoke.py, RTX 2080 Ti, sm_75, fp16, transformers 5.13.0 / torch
# 2.6.0+cu124): 291.6 ms/frame (encoder 189.3 median, head 75.6 median), first frame 708 ms, model
# load 4.3 s, peak device allocation 2,263 MiB. bf16 on this card is 4.4x slower for 2.6x the
# memory, which is why main.py's _best_dtype picks fp16 below Ampere.
#
# FAULT ISOLATION IS THE LAW. The model loads LAZILY on the first /track/open and latches its
# failure, exactly as _ensure_upscaler (main.py:587) and _ensure_detail (main.py:674) do: a
# tracker failure 503s here in isolation and can never touch /segment or /segment_text, which are
# the shipped interactive paths.
#
# LOCKING. A run holds _track_lock for its whole duration (one run at a time) but takes main.py's
# model lock only around each single-frame forward, so an interactive /segment click interleaves
# at one frame's latency — about 0.3 s — instead of blocking for the two minutes a 272-frame clip
# takes.

import io
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

router = APIRouter(prefix="/track")

# Two live sessions, because each one's memory bank is the expensive part: per object per frame the
# tracker stores maskmem_features 5184x64 bf16 (hardcoded at modeling_sam3_tracker_video.py:2701),
# maskmem_pos_enc, pred_masks and high_res_masks — about 3.36 MiB, so about 914 MiB for one object
# over 272 frames. Two of those is the ceiling beside SAM 3's own weights.
TRACK_MAX_SESSIONS = int(os.environ.get("TRACK_MAX_SESSIONS", "2"))
TRACK_MAX_FRAMES = int(os.environ.get("TRACK_MAX_FRAMES", "4096"))
# A session nobody has touched is reaped; its JPEGs are retired for the same window so a Retrack
# under the same clip + camera + trackRes skips the browser's capture sweep entirely.
TRACK_SESSION_IDLE_S = float(os.environ.get("TRACK_SESSION_IDLE_S", "600"))
TRACK_FRAME_TTL_S = float(os.environ.get("TRACK_FRAME_TTL_S", "600"))
# A RUNNING session whose stream has had no reader for this long is cancelled and its lock broken.
# This is the tab-reload case: the browser closes the EventSource, the dev server's /sam proxy
# (serve.mjs:659-687) registers no req.on("close") so it never tells us, and the generator ends up
# blocked inside a yield nobody will ever drain — holding _track_lock for the rest of the process.
TRACK_STREAM_IDLE_S = float(os.environ.get("TRACK_STREAM_IDLE_S", "30"))
TRACK_REAP_TICK_S = 5.0
# Replay page size. 272 frames at 768x432 is roughly 1,700 runs per frame; a whole run in one JSON
# body is several megabytes, so /track/results pages and hands back the next cursor.
TRACK_REPLAY_PAGE = 256
# How long a /track/open will wait for `from transformers import ...` to become answerable while
# main.py's loader thread is still executing transformers/__init__.py. Measured below; the window
# is a few seconds, and 10 s is the same order as the 4.3 s the model load itself costs.
TRACK_IMPORT_WAIT_S = float(os.environ.get("TRACK_IMPORT_WAIT_S", "10"))
# How long a /track/open will wait for main.py's background loader to finish before giving up and
# answering a RETRYABLE 503. Sized past the 30-60 s the launcher documents for the model load
# (run-sam-service.ps1:80), because the alternative is two multi-gigabyte loads racing on one card.
TRACK_LOAD_WAIT_S = float(os.environ.get("TRACK_LOAD_WAIT_S", "180"))

# Set by configure(), called from main.py. This module never imports main — main imports it, so an
# import back would be circular — and keeping it one-way means track.py stays importable, and
# py_compile-able, on a machine with no weights and no GPU.
DEVICE = "cpu"
SAM_DTYPE = None
SAM3_DIR = ""
_model_lock = threading.Lock()
_load_done: threading.Event | None = None


def _no_concept_model():
    return None


_get_concept_model = _no_concept_model


def configure(*, device, dtype, sam3_dir, model_lock, get_concept_model=None, load_done=None) -> None:
    """Hand this module main.py's resolved device, dtype, checkpoint path, GPU lock and loader Event.

    get_concept_model is read LATE, never captured as a value: main.py loads the concept model on a
    background thread (main.py:281), so at import time it is still None and the graft below would
    silently never happen.

    load_done is that thread's completion Event. _ensure_tracker_video waits on it before allocating
    anything, because main.py's _loader never takes the model lock (it is not a request path), so
    without the wait a /track/open landing inside the 30-60 s the launcher warns the load takes
    (run-sam-service.ps1:80) allocates ~1.6 GiB concurrently with _load_sam3's ~3 GB on the same
    card — and the concept model is not resident yet either, so the graft below silently misses and
    loads a second 908 MiB tower on top."""
    global DEVICE, SAM_DTYPE, SAM3_DIR, _model_lock, _get_concept_model, _load_done
    DEVICE, SAM_DTYPE, SAM3_DIR, _model_lock = device, dtype, sam3_dir, model_lock
    if get_concept_model is not None:
        _get_concept_model = get_concept_model
    _load_done = load_done


# ------------------------------------------------------------------------ mask RLE ----
# The one thing in this file that has to be byte-exact against another language. Both functions
# are pure numpy — no torch, no model, no service — so packages/encoder/test/mask-rle-parity.test.mjs
# can pin them to the TypeScript pair with a golden vector on a machine with no GPU.


def mask_to_rle(mask) -> list[int]:
    """Encode a 2-D 0/1 bitmap the way packages/core/src/edits.ts:50 rleEncodeMask does.

    Alternating run lengths, row-major, 0-RUN FIRST: the TS loop starts at cur=0/run=0, so a bitmap
    whose first pixel is set pushes a ZERO-LENGTH run before the first 1-run. The final run is
    always closed by the push after the loop, which is what lets validateMasks (edits.ts:301-306)
    assert the runs sum to exactly width*height — a short sum there means a truncated write.

    Non-zero is 1: the TS side reads `bits[i] ? 1 : 0`, so a 0/255 mask and a 0/1 mask must encode
    identically, and comparing raw values instead of truthiness would split a 1 from a 2.
    """
    bits = (np.ascontiguousarray(np.asarray(mask)).reshape(-1) != 0).astype(np.uint8)
    if bits.size == 0:
        return [0]                                   # the TS loop never runs and pushes run=0
    edges = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    runs = np.diff(np.concatenate(([0], edges, [bits.size])))
    rle = [int(r) for r in runs]
    if bits[0]:
        rle.insert(0, 0)
    return rle


def rle_to_mask(rle, width: int, height: int) -> np.ndarray:
    """Inverse of mask_to_rle, matching rleDecodeMask (packages/core/src/edits.ts:63) including its
    tolerances: the cursor advances by the FULL run before the bound check, so an over-long final
    run decodes rather than raising. Mask-seeded prompts arrive in sidecar form and come through
    here."""
    size = max(0, int(width) * int(height))
    bits = np.zeros(size, dtype=np.uint8)
    i, val = 0, 0
    for run in rle:
        run = int(run)
        if val and run > 0:
            bits[i:min(size, i + run)] = 1
        i += run
        val ^= 1
        if i >= size:
            break
    return bits.reshape(int(height), int(width))


# -------------------------------------------------------------------------- model ----

_video_model = None
_track_processor = None
_track_error = None      # latched: set once, never retried, reported on /health
_track_loading = False


def _assert_detector_shared(vm, concept) -> None:
    """The graft is only worth doing if it actually shares GPU allocations. Same guard shape as
    main.py:198 _assert_vision_encoder_shared and for the same reason: a silently duplicated
    detector shows up as an out-of-memory three weeks later, not as an error here."""
    if vm.detector_model is not concept:
        raise RuntimeError("video detector/concept MODULE identity check FAILED")
    for vp, cp in zip(vm.detector_model.vision_encoder.parameters(), concept.vision_encoder.parameters()):
        if vp.data_ptr() != cp.data_ptr():
            raise RuntimeError("video detector/concept vision_encoder tensors are not the same "
                               "GPU allocation (data_ptr mismatch) after sharing")


def _graft_resident_detector(vm) -> bool:
    """Reuse the resident concept model as the video model's detector instead of putting a second
    copy on the card.

    vm.detector_model IS a Sam3Model (asserted by track_smoke.py's load probe) built from the same
    checkpoint's detector_config as main.py's concept model, so the two are the same module class
    holding bit-identical weights — measured at 840.38 M parameters, 454.04 M of them the vision
    tower. Sharing the TOWER alone would still hand vm.to(DEVICE) the other 386.34 M to move, and
    this file never forwards one of them: get_vision_features (:296) is the only detector call and
    it touches vision_encoder alone (modeling_sam3.py:2254). So the whole module is shared.

    Grafting BEFORE vm.to(DEVICE) is what makes it free: the concept model is already on the
    device, nn.Module._apply leaves a parameter whose device and dtype already match untouched (so
    data_ptr survives and the assertion above holds), and vm's own freshly loaded detector is
    dropped on the HOST where it never cost VRAM at all. main.py:185 does the same share for the
    click path but pays a transient ~+0.9 GB because it grafts after the move.

    Returns False and logs rather than raising when the shapes do not line up: no graft is heavier,
    not wrong, and this must never be the thing that fails a track.
    """
    concept = _get_concept_model()
    if concept is None:
        return False
    try:
        own = list(vm.detector_model.vision_encoder.parameters())
        theirs = list(concept.vision_encoder.parameters())
        if type(vm.detector_model) is not type(concept) or len(own) != len(theirs) \
                or any(a.shape != b.shape for a, b in zip(own, theirs)):
            print(f"[ares-sam] track: detector shapes differ ({type(vm.detector_model).__name__}/"
                  f"{len(own)} params vs {type(concept).__name__}/{len(theirs)}) — loading a second "
                  f"copy rather than grafting")
            return False
        vm.detector_model = concept
        return True
    except Exception as e:
        print(f"[ares-sam] track: detector graft skipped ({type(e).__name__}: {e})")
        return False


def _import_transformers():
    """`from transformers import ...`, waited out rather than latched.

    main.py starts its own model load on a BACKGROUND thread at import (main.py:281) and that
    thread is what first executes transformers/__init__.py. A `from transformers import X` issued
    from a request thread while that is still running gets handed the partially initialised module
    and raises ImportError — measured on this machine: a /track/open in the first seconds of
    process life fails with "cannot import name 'Sam3TrackerVideoProcessor' from 'transformers'",
    and the same call 6 s later resolves both names and gets all the way to from_pretrained. The
    real cold start is worse than the measurement, because the browser's first track request lands
    inside the 30-60 s the launcher already warns the model load takes (run-sam-service.ps1:80).

    That is a startup race, not a load failure, so it must never reach _track_error: latching it
    would 503 /track/* for the life of the process because a click was early.
    """
    deadline = time.time() + TRACK_IMPORT_WAIT_S
    while True:
        try:
            from transformers import Sam3TrackerVideoProcessor, Sam3VideoModel
            return Sam3TrackerVideoProcessor, Sam3VideoModel
        except ImportError:
            if time.time() >= deadline:
                raise
            time.sleep(0.25)


_tf_classes = None


def _prewarm() -> None:
    """Wait out transformers' import and main.py's loader thread BEFORE the model lock is taken.

    Neither wait needs the GPU, and both are long: the import loop above runs up to
    TRACK_IMPORT_WAIT_S and the loader up to TRACK_LOAD_WAIT_S. Holding main.py's _lock across
    either of them stalls /segment (main.py:511) and /segment_text (main.py:552) for the whole
    window, for a request that has not allocated a byte yet. So every /track/* route that goes on
    to call _ensure_tracker_video calls this first, unlocked.

    Both timeouts answer a RETRYABLE 503 and are never latched into _track_error: a click that
    landed early must not disable /track/* for the life of the process."""
    global _tf_classes
    if _video_model is not None:
        return
    if _tf_classes is None:
        try:
            _tf_classes = _import_transformers()
        except ImportError as e:
            raise HTTPException(status_code=503,
                                detail=f"video tracker still starting: {type(e).__name__}: {e}")
    if _load_done is not None and not _load_done.wait(TRACK_LOAD_WAIT_S):
        raise HTTPException(status_code=503, detail="sam model still loading: retry shortly")


def _ensure_tracker_video():
    """Lazy load with a latched error, following _ensure_upscaler (main.py:587) exactly.

    Call _prewarm() before taking the model lock; this half is the part that allocates."""
    global _video_model, _track_processor, _track_error, _track_loading
    if _video_model is not None:
        return _video_model, _track_processor
    if _track_error is not None:
        raise HTTPException(status_code=503, detail=f"video tracker failed to load: {_track_error}")
    classes = _tf_classes or _import_transformers()
    _track_loading = True
    try:
        import torch
        Sam3TrackerVideoProcessor, Sam3VideoModel = classes

        t0 = time.time()
        processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_DIR)
        vm = Sam3VideoModel.from_pretrained(SAM3_DIR, dtype=SAM_DTYPE)
        shared = _graft_resident_detector(vm)
        vm = vm.to(DEVICE).eval()
        if shared:
            _assert_detector_shared(vm, _get_concept_model())
            import gc
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        if vm.tracker_model.vision_encoder is not None:
            raise RuntimeError("tracker_model.vision_encoder is not None — the towerless seam this "
                               "service primes (modeling_sam3_video.py:512) is gone from this transformers")
        _video_model, _track_processor = vm, processor
        print(f"[ares-sam] video tracker loaded from {SAM3_DIR} on {DEVICE} ({SAM_DTYPE}) in "
              f"{time.time() - t0:.1f}s, input edge {processor.target_size}, detector "
              f"{'SHARED with the concept model' if shared else 'own copy (concept model not resident)'}")
        return _video_model, _track_processor
    except Exception as e:
        # An OOM is a transient collision, not a broken install, and latching it would 503 /track/*
        # for the life of the process because a track started while something else was allocating.
        # torch.cuda.OutOfMemoryError subclasses RuntimeError, so it has to be named to be caught.
        if type(e).__name__ == "OutOfMemoryError":
            print(f"[ares-sam] video tracker load ran out of device memory: {e} — retryable")
            raise HTTPException(status_code=503,
                                detail="video tracker out of device memory: retry once the card is free")
        _track_error = f"{type(e).__name__}: {e}"
        print(f"[ares-sam] video tracker failed to load: {_track_error} — /track/* will 503")
        raise HTTPException(status_code=503, detail=f"video tracker failed to load: {_track_error}")
    finally:
        _track_loading = False


def _preprocess_frame(processor, jpeg: bytes):
    """JPEG bytes -> (1, 3, target, target) float CPU tensor, the same call init_video_session makes
    (processing_sam3_tracker_video.py:551) with a one-frame video. Runs on the prefetch thread:
    12-20 ms of decode and resize per frame, which hides entirely under the 291.6 ms the GPU spends
    on the previous frame. Kept on the HOST — moving it to the device is the caller's job, inside
    the model lock, so the prefetch never allocates VRAM behind a running forward."""
    img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    out = processor.video_processor(videos=img[None, ...], device="cpu", return_tensors="pt")
    return out.pixel_values_videos[0]


def _prime_vision_cache(vm, isess, frame_idx: int, px) -> None:
    """The three-call sequence from modeling_sam3_video.py:1613-1632. The detector's tower runs ONCE
    per frame and its output feeds both the detector head and, through tracker_neck plus the mask
    decoder's conv_s0/conv_s1 projections, the towerless tracker. _prepare_vision_features
    (modeling_sam3_tracker_video.py:1900) reads inference_session.cache BEFORE it would dereference
    that None encoder, so this priming IS the seam — and _batch_encode_memories (:2706) asks for the
    same frame's features a second time, which is why they go in the cache rather than a variable."""
    vision_embeds = vm.detector_model.get_vision_features(pixel_values=px)
    feats, pos = vm.get_vision_features_for_tracker(vision_embeds=vision_embeds)
    isess.cache.cache_vision_features(frame_idx, {"vision_feats": feats, "vision_pos_embeds": pos})


# ------------------------------------------------------------------ session store ----


class TrackSession:
    """One tracked clip: the uploaded proxy renders, the library's inference session, and every
    mask this run has already emitted. Everything a fresh tab needs to reattach after a reload."""

    def __init__(self, sid: str, clip: str, cam_key: str, frames: int, track_res: int, mask_res: int,
                 capture: tuple[int, int] = (0, 0)):
        self.id = sid
        self.clip = clip
        self.cam_key = cam_key
        self.frames = frames
        self.track_res = track_res
        self.mask_res = mask_res
        self.capture = capture                          # (width, height) the client declared at open
        self.jpegs: dict[int, bytes] = {}
        self.frame_hw: tuple[int, int] | None = None    # (height, width) of the uploaded renders
        self.isess = None                               # Sam3TrackerVideoInferenceSession
        self.obj_ids: list[int] = []
        self.seeds: list[dict] = []
        self.events: list[dict] = []                    # every mask/gap emitted, in emission order
        self.run_id = 0                                 # bumped per run; /track/results is per-run
        self.gen = None                                 # the live _propagate generator, for close()
        self.close_after: bool | None = None            # keepFrames of a /track/close during a run
        self.state = "idle"                             # idle | running | stranded
        self.cancel = False
        self.at_frame: int | None = None                # the frame the live run has reached
        self.stopped_at: int | None = None
        self.created = time.time()
        self.touched = time.time()
        # stage/stage_at are what the reaper reads. "gpu" and "work" are healthy however long they
        # last (a /detail pass can hold the model lock for minutes); "start" and "yield" are the two
        # states in which the only thing we can be waiting for is a reader that may never come back.
        self.stage = "idle"
        self.stage_at = time.time()

    def mark(self, stage: str) -> None:
        self.stage, self.stage_at = stage, time.time()

    def touch(self) -> None:
        self.touched = time.time()

    def store_bytes(self) -> int:
        return sum(len(b) for b in self.jpegs.values())


_sessions: dict[str, TrackSession] = {}
_retired: dict[str, dict] = {}          # "clip|camKey|trackRes" -> {"jpegs", "hw", "at"}
_store_lock = threading.Lock()          # guards _sessions and _retired, never held across GPU work

# One run at a time. Acquired non-blocking in the /track/run precheck so a second run answers 409
# BEFORE any SSE head is written (the hard law for SSE routes here), released by the generator's
# finally — or broken by the reaper, which is legal because a threading.Lock is not owner-bound.
# The token is what stops a stranded generator from later releasing a lock the reaper has already
# handed to somebody else.
_track_lock = threading.Lock()
_run_guard = threading.Lock()
# "sess" is the running session OBJECT, not just its id. The reaper's lock recovery must not depend
# on the session still being REGISTERED: _drop_session pops it out of _sessions (:498) and
# /track/close calls that on a session whose run has no SSE reader, so a lookup by id would find
# nothing, skip the recovery branch, and wedge _track_lock for the life of the process — every
# later GET /track/run answering 409 forever, on exactly the tab-reload flow the reaper exists for.
_track_run: dict[str, Any] = {"session": None, "sess": None, "token": 0}
_reaper_started = False


def _acquire_run(sess: "TrackSession") -> int | None:
    if not _track_lock.acquire(blocking=False):
        return None
    with _run_guard:
        _track_run["token"] += 1
        _track_run["session"] = sess.id
        _track_run["sess"] = sess
        return int(_track_run["token"])


def _release_run(token: int) -> bool:
    with _run_guard:
        if _track_run["session"] is None or _track_run["token"] != token:
            return False
        _track_run["session"] = None
        _track_run["sess"] = None
        _track_lock.release()
        return True


def _close_run_generator(sess: "TrackSession") -> bool:
    """Raise GeneratorExit inside a generator parked at a `yield` nobody will drain again.

    Without this a stranded run leaks everything its frame locals hold — the inference session's
    memory bank (~914 MiB over 272 frames), the JPEG frame store (~60 MB) and a non-daemon
    ThreadPoolExecutor thread — for the life of the process, because _drop_session only rebinds
    sess.isess / sess.jpegs (:511-512) and the generator's own references are what keep them
    alive. It also never sees GeneratorExit on its own: the dev server's /sam proxy (serve.mjs:659-687) registers
    no req/res close handler, so the upstream response is never destroyed and starlette's send()
    blocks forever.

    A generator SUSPENDED at a yield accepts close() from another thread — GeneratorExit is raised
    at the yield point, the `finally` runs (pool shutdown, session drop, lock release) and the frame
    locals die. A generator that happens to be RUNNING at that instant raises ValueError instead,
    which is why this reports rather than raises: the next reaper tick tries again."""
    gen = sess.gen
    if gen is None:
        return False
    try:
        gen.close()
    except Exception as e:            # "generator already executing" — it resumed under us
        print(f"[ares-sam] track: could not close {sess.id}'s stream ({type(e).__name__}: {e})")
        return False
    sess.gen = None
    return True


def _new_session_id() -> str:
    return "trk_" + os.urandom(4).hex()


def _frame_key(clip: str, cam_key: str, track_res: int, capture: tuple[int, int]) -> str:
    """The retired frame store's key. The CAPTURE SIZE is part of it, and has to be: trackRes is
    pinned to the checkpoint's native edge (:611) so it cannot discriminate, and track_open restores
    sess.frame_hw out of the store, which _mask_dims turns into every emitted bitmap's w/h. Without
    it, tracking at 1024x576, resizing the window and re-opening reports reusedFrames > 0, the
    client skips the capture sweep, and every mask comes back at the OLD aspect — which
    validateMasks (packages/core/src/edits.ts:307-309) then rejects for the whole tracked range."""
    return f"{clip}|{cam_key}|{track_res}|{capture[0]}x{capture[1]}"


def _require(sid: str) -> TrackSession:
    sess = _sessions.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"unknown track session: {sid}")
    sess.touch()
    return sess


def _drop_session(sid: str, keep_frames: bool) -> float:
    """Free a session's inference state and report the device memory it actually gave back, rather
    than an estimate: the memory bank is the only thing here big enough to be worth reporting."""
    sess = _sessions.pop(sid, None)
    if sess is None:
        return 0.0
    if keep_frames and sess.jpegs:
        _retired[_frame_key(sess.clip, sess.cam_key, sess.track_res, sess.capture)] = {
            "jpegs": sess.jpegs, "hw": sess.frame_hw, "at": time.time()}
    freed = 0.0
    try:
        import gc

        import torch
        before = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0
        sess.isess = None
        sess.jpegs = {}
        sess.events = []
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            freed = max(0.0, (before - torch.cuda.memory_allocated()) / (1 << 20))
    except Exception:
        sess.isess = None
        sess.jpegs = {}
    return round(freed, 1)


def _reap() -> None:
    now = time.time()
    run = dict(_track_run)
    sess = run.get("sess")
    if sess is not None:
        # A generator blocked inside `yield` cannot see its own cancel flag, so the lock has to be
        # taken off it from out here. Every mask already emitted stays in sess.events, so the client
        # that comes back can replay the run through GET /track/results instead of losing it — it
        # pins the run by `run` (the id in the `start` event), because the next run resets the list.
        if sess.state == "running" and sess.stage in ("start", "yield") \
                and now - sess.stage_at > TRACK_STREAM_IDLE_S:
            idle = now - sess.stage_at    # read BEFORE the close: the generator's finally marks
            sess.cancel = True            # a new stage, which would make this read as -0s
            sess.state = "stranded"
            _release_run(int(run["token"]))
            closed = _close_run_generator(sess)
            print(f"[ares-sam] track: {sess.id} had no stream reader for {idle:.0f}s "
                  f"— cancelled, lock released, stream {'closed' if closed else 'still open'}, "
                  f"{len(sess.events)} events kept for replay")
    with _store_lock:
        for sid, sess in list(_sessions.items()):
            if sess.state == "running":
                continue
            if now - sess.touched > TRACK_SESSION_IDLE_S:
                freed = _drop_session(sid, keep_frames=True)
                print(f"[ares-sam] track: reaped idle session {sid} ({freed} MiB, frames retained)")
        for key, store in list(_retired.items()):
            if now - store["at"] > TRACK_FRAME_TTL_S:
                del _retired[key]


def _reaper_loop() -> None:
    while True:
        time.sleep(TRACK_REAP_TICK_S)
        try:
            _reap()
        except Exception as e:     # a reaper that dies silently is worse than one that logs
            print(f"[ares-sam] track reaper: {type(e).__name__}: {e}")


def _start_reaper() -> None:
    """Started by the first /track/open, not at import: a service that never tracks should not
    carry a timer thread for the life of the process."""
    global _reaper_started
    if _reaper_started:
        return
    _reaper_started = True
    threading.Thread(target=_reaper_loop, daemon=True, name="sam-track-reaper").start()


# ------------------------------------------------------------------------- open ----


class TrackOpenRequest(BaseModel):
    clip: str = ""
    camKey: str = ""
    frames: int
    trackRes: int = 0            # 0 -> the checkpoint's native input edge
    maskRes: int = 768
    # The proxy render's pixel size. Part of the retired frame store's key (_frame_key): the same
    # clip under the same frozen camera at a different viewport size is a DIFFERENT frame store, and
    # reusing the old one hands every mask the old aspect.
    captureW: int = 0
    captureH: int = 0


@router.post("/open")
def track_open(req: TrackOpenRequest):
    _start_reaper()
    frames = int(req.frames)
    if frames < 1 or frames > TRACK_MAX_FRAMES:
        raise HTTPException(status_code=400, detail=f"frames must be 1..{TRACK_MAX_FRAMES}")
    mask_res = max(64, min(2048, int(req.maskRes or 768)))
    capture = (max(0, int(req.captureW)), max(0, int(req.captureH)))

    t0 = time.time()
    _prewarm()                   # unlocked: pure import and loader waiting, no allocation
    with _model_lock:            # the load allocates ~1.6 GiB in 4.3 s; share SAM's GPU lock for it
        _, processor = _ensure_tracker_video()

    # trackRes is NOT a free parameter on this seam, whatever the plan's draft tier says. The
    # tracker views the detector's tokens through prompt_encoder.image_embedding_size, which
    # modeling_sam3_tracker_video.py:1232 fixes at image_size // patch_size = 1008 // 14 = 72 when
    # the model is CONSTRUCTED. Feed a 560 px frame and get_vision_features_for_tracker's view()
    # (modeling_sam3_video.py:549) is handed 40x40 tokens for a 72x72 grid and raises. A draft tier
    # needs position-embedding interpolation, not a smaller resize.
    native = int(processor.target_size)
    track_res = int(req.trackRes) or native
    if track_res != native:
        raise HTTPException(status_code=400, detail=f"trackRes must be {native}: the tracker's token "
                                                    f"grid is fixed at construction (image_size // patch_size)")

    with _store_lock:
        if len(_sessions) >= TRACK_MAX_SESSIONS:
            oldest = min(_sessions.values(), key=lambda s: s.created)
            raise HTTPException(status_code=409,
                                detail=f"track session limit reached: close {oldest.id} first")
        sess = TrackSession(_new_session_id(), req.clip or "", req.camKey or "", frames, track_res,
                            mask_res, capture)
        # A frame store retired by the reaper (or by close with keepFrames) under the same clip,
        # camera, input edge and capture size is still the same pixels, so a Retrack skips the sweep.
        store = _retired.pop(_frame_key(sess.clip, sess.cam_key, track_res, capture), None)
        if store is not None:
            sess.jpegs = {f: b for f, b in store["jpegs"].items() if f < frames}
            sess.frame_hw = store["hw"]
        _sessions[sess.id] = sess

    return {
        "session": sess.id,
        "trackRes": track_res,
        "maskRes": mask_res,
        "dtype": str(SAM_DTYPE).replace("torch.", ""),
        "device": DEVICE,
        "checkpoint": SAM3_DIR,
        "frames": frames,
        "reusedFrames": len(sess.jpegs),
        "loadMs": round((time.time() - t0) * 1000, 1),
    }


# ----------------------------------------------------------------------- frames ----


@router.post("/frames")
async def track_frames(request: Request, session: str):
    """Frame upload. The body is a bare stream of records with no envelope:
         [u32le frameIdx][u32le byteLen][byteLen bytes of JPEG] ...
    JPEG rather than PNG because canvas.toDataURL costs 15-30 ms of main-thread deflate per frame
    while loopback bytes are free — about 60 MB for a 272-frame clip, under 0.2 s of wire time."""
    sess = _require(session)
    body = await request.body()
    n = len(body)
    off, stored, added, first = 0, 0, 0, None
    while off < n:
        if off + 8 > n:
            raise HTTPException(status_code=400, detail=f"truncated frame record at offset {off}")
        idx = int.from_bytes(body[off:off + 4], "little")
        size = int.from_bytes(body[off + 4:off + 8], "little")
        if size == 0 or off + 8 + size > n:
            raise HTTPException(status_code=400, detail=f"truncated frame record at offset {off}")
        if idx >= sess.frames:
            raise HTTPException(status_code=400,
                                detail=f"frame {idx} is outside the session's 0..{sess.frames - 1}")
        sess.jpegs[idx] = body[off + 8:off + 8 + size]
        if first is None:
            first = idx
        off += 8 + size
        stored += 1
        added += size
    if first is not None:
        # Header only, no decode — but a well-framed record whose payload is not an image still
        # reaches PIL, and an uncaught UnidentifiedImageError here would answer 500 with a stack
        # trace where every other caller mistake on this route answers 400 with a reason.
        try:
            w, h = Image.open(io.BytesIO(sess.jpegs[first])).size
        except Exception as e:
            raise HTTPException(status_code=400,
                                detail=f"frame {first} is not a decodable image: {type(e).__name__}")
        # The session's frame size is set ONCE and every emitted bitmap's w/h is derived from it
        # (_mask_dims). A second upload at a different size — a viewport resize mid-sweep, or a
        # retired store reused under a stale key — would silently mask at the old aspect.
        if sess.frame_hw is None:
            sess.frame_hw = (int(h), int(w))
        elif sess.frame_hw != (int(h), int(w)):
            raise HTTPException(status_code=400, detail=f"frame {first} is {w}x{h}, but the session's "
                                                        f"frames are {sess.frame_hw[1]}x{sess.frame_hw[0]}")
    sess.touch()
    return {"stored": stored, "total": len(sess.jpegs), "bytes": sess.store_bytes(), "added": added}


# ----------------------------------------------------------------------- prompt ----


class TrackObjectPrompt(BaseModel):
    objId: int
    points: list[list[float]] | None = None     # proxy-render pixels, one [x, y] per label
    labels: list[int] | None = None             # 1 foreground, 0 background
    box: list[float] | None = None              # [x0, y0, x1, y1], proxy-render pixels
    mask: dict[str, Any] | None = None          # {width, height, rle} in the sidecar's convention


class TrackPromptRequest(BaseModel):
    session: str
    frame: int
    clearOldInputs: bool = True
    objects: list[TrackObjectPrompt]


def _ensure_inference_session(sess: TrackSession, processor):
    """One Sam3TrackerVideoInferenceSession per track session, built with NO video: frames stream in
    one at a time. init_video_session(video=...) preprocesses the whole clip into a single
    (frames, 3, 1008, 1008) tensor — 1.66 GiB at fp16 for 272 frames — which is exactly the
    allocation the JPEG frame store exists to avoid."""
    if sess.isess is not None:
        return sess.isess
    if sess.frame_hw is None:
        raise HTTPException(status_code=400, detail="no frames uploaded yet")
    h, w = sess.frame_hw
    # inference_state_device is the memory bank's home AND the vision cache's — one knob, two very
    # different bills. On the host it costs a ~111 MiB round trip per frame (the cache holds
    # vision_feats and vision_pos_embeds over the 288/144/72 pyramid at 256 channels, cache size 1);
    # on the device it costs ~3.36 MiB per frame per object of resident memory bank, ~914 MiB over
    # 272 frames on top of the 2,263 MiB peak track_smoke.py measured. The 6 GB card main.py already
    # ships on cannot spare that, so the host wins by default and TRACK_STATE_DEVICE buys the
    # round trip back on a big card.
    state_device = os.environ.get("TRACK_STATE_DEVICE", "cpu")
    sess.isess = processor.init_video_session(
        video=None, inference_device=DEVICE, inference_state_device=state_device,
        video_storage_device=DEVICE, dtype=SAM_DTYPE)
    sess.isess.video_height, sess.isess.video_width = h, w
    return sess.isess


def _forward_frame(vm, processor, sess: TrackSession, isess, frame: int, reverse: bool, px_cpu=None):
    """Prime the vision cache, run one frame, and drop the preprocessed pixels again. Called with
    main.py's model lock HELD.

    `frame=` is passed to the tracker deliberately: it puts the forward on the library's streaming
    path (modeling_sam3_tracker_video.py:1765/:1818), which does NOT bound the object-pointer window
    by inference_session.num_frames (:2354). That matters because processed_frames is popped after
    every frame to keep exactly one 6.1 MiB tensor resident — under the non-streaming path num_frames
    would then read as 1 and _get_object_pointers would break out of its loop on the first
    iteration, silently discarding the memory the whole track runs on."""
    import torch
    if px_cpu is None:
        px_cpu = _preprocess_frame(processor, sess.jpegs[frame])
    with torch.no_grad():
        px = px_cpu.to(DEVICE, dtype=SAM_DTYPE, non_blocking=True)
        _prime_vision_cache(vm, isess, frame, px)
        out = vm.tracker_model(inference_session=isess, frame_idx=frame, frame=px, reverse=reverse)
    if isess.processed_frames is not None:
        isess.processed_frames.pop(frame, None)
    return out


@router.post("/prompt")
def track_prompt(req: TrackPromptRequest):
    """Seed or nudge, then run ONE forward at that frame to consume the prompt.

    EVERY object of a frame must arrive in one call. add_inputs_to_inference_session ends with
    `inference_session.obj_with_new_inputs = obj_ids` — an ASSIGNMENT, not an append, at
    processing_sam3_tracker_video.py:736 for points/boxes and :800 for masks. The forward gates on
    `obj_id in inference_session.obj_with_new_inputs` (modeling_sam3_tracker_video.py:1784), so a
    second call for the same frame leaves the first object's stored prompt on the floor and tracks
    it as though nothing had been clicked.

    Coordinates are proxy-render pixels; original_size is passed explicitly so the processor scales
    them to the tracker's square input itself (processing_sam3_tracker_video.py:199-202).

    A run must not be live while this happens. _propagate releases the model lock after every
    single-frame forward, so a prompt slips into that gap and mutates the very inference session the
    generator is iterating: obj_id_to_idx (modeling_sam3_tracker_video.py:184-203) appends a new
    object id to inference_session.obj_ids, the run's forward then loops over it (:1782) and the
    stream starts emitting mask events for an objId the already-sent `start` event never announced.
    The library has no per-object removal (only reset_tracking_data, :324), so a run that has been
    mutated this way cannot be put back."""
    sess = _require(req.session)
    if sess.state == "running":
        raise HTTPException(status_code=409, detail="track busy: cancel the run before seeding")
    if not req.objects:
        raise HTTPException(status_code=400, detail="objects must be non-empty")
    frame = int(req.frame)
    if frame < 0 or frame >= sess.frames:
        raise HTTPException(status_code=400, detail=f"frame {frame} is outside 0..{sess.frames - 1}")
    if frame not in sess.jpegs:
        raise HTTPException(status_code=400, detail=f"frame {frame} has not been uploaded")

    mask_objs: list[TrackObjectPrompt] = []
    point_objs: list[TrackObjectPrompt] = []
    for o in req.objects:
        has_pt = bool(o.points) or bool(o.box)
        # Verbatim from processing_sam3_tracker_video.py:611-613 and :702-707 — both are ValueErrors
        # inside the library, which would surface as a 500; they are the caller's mistakes, so they
        # answer 400 here instead.
        if o.mask is not None and has_pt:
            raise HTTPException(status_code=400,
                                detail="mask prompts cannot be combined with points or boxes on one object")
        if o.mask is None and not has_pt:
            raise HTTPException(status_code=400, detail=f"object {o.objId} carries no points, box or mask")
        if bool(o.points) != bool(o.labels) or (o.points and len(o.points) != len(o.labels or [])):
            raise HTTPException(status_code=400,
                                detail="points and labels must be provided together and match in length")
        if o.box is not None:
            if len(o.box) != 4:
                raise HTTPException(status_code=400, detail="box must be [x0, y0, x1, y1]")
            if not req.clearOldInputs:
                raise HTTPException(status_code=400, detail="box prompt requires clearOldInputs true")
        if o.mask is not None:
            # Same well-formedness the sidecar side asserts, in the same words (validateMasks,
            # packages/core/src/edits.ts:303-306). Without it a mask missing width/height decodes to
            # a (0, 0) array and process_new_mask_for_video_frame's F.interpolate
            # (processing_sam3_tracker_video.py:784-791) raises a 500 for a caller mistake, and an
            # rle whose runs fall short is silently zero-padded by rle_to_mask's tolerant decode —
            # the service accepting a truncated mask the sidecar validator would reject.
            mw_in, mh_in = int(o.mask.get("width", 0) or 0), int(o.mask.get("height", 0) or 0)
            runs = o.mask.get("rle")
            if mw_in < 1 or mh_in < 1:
                raise HTTPException(status_code=400,
                                    detail=f"object {o.objId} mask needs a positive width and height")
            if not isinstance(runs, list):
                raise HTTPException(status_code=400, detail=f"object {o.objId} mask has no rle")
            total = sum(int(r) for r in runs)
            if total != mw_in * mh_in:
                raise HTTPException(status_code=400, detail=f"object {o.objId} mask rle runs sum to "
                                                            f"{total}, not {mw_in}x{mh_in} = {mw_in * mh_in}")
        (mask_objs if o.mask is not None else point_objs).append(o)
    if point_objs and any(o.box for o in point_objs) and not all(o.box for o in point_objs):
        # input_boxes is checked against len(obj_ids) at processing_sam3_tracker_video.py:698, so a
        # single call cannot carry a box for some objects and only points for others.
        raise HTTPException(status_code=400, detail="a box prompt needs every object of the call to carry one")

    _prewarm()                   # unlocked: pure import and loader waiting, no allocation
    with _model_lock:
        vm, processor = _ensure_tracker_video()
        isess = _ensure_inference_session(sess, processor)
        h, w = sess.frame_hw
        obj_ids: list[int] = []
        if point_objs:
            ids = [int(o.objId) for o in point_objs]
            any_points = any(o.points for o in point_objs)
            any_boxes = any(o.box for o in point_objs)
            # Nesting is (batch, object, point, xy): the processor checks shape[1] against the object
            # count (processing_sam3_tracker_video.py:681-684) and pads ragged point lists itself —
            # but only ones it can MEASURE. _validate_single_input reads the nesting depth by
            # descending data[0] alone (_get_nesting_level, :361-372, which returns 1 for an empty
            # list), so a leading object with no points makes a 4-level structure measure as 3 and
            # :422-425 raises ValueError — an opaque 500, and one that goes away if the same two
            # objects are reordered. Pad here instead, to point_pad_value, which is exactly what
            # _pad_nested_list (:284-302) produces for the non-ragged case and what the prompt
            # encoder zeroes back out at modeling_sam3_tracker_video.py:1253-1257.
            pad = float(processor.point_pad_value)
            wide = max((len(o.points or []) for o in point_objs), default=0)
            pts = [[[[float(p[0]), float(p[1])] for p in (o.points or [])]
                    + [[pad, pad]] * (wide - len(o.points or [])) for o in point_objs]]
            lbl = [[[int(v) for v in (o.labels or [])]
                    + [int(pad)] * (wide - len(o.labels or [])) for o in point_objs]]
            box = [[[float(v) for v in (o.box or [])] for o in point_objs]]
            processor.add_inputs_to_inference_session(
                inference_session=isess, frame_idx=frame, obj_ids=ids,
                input_points=pts if any_points else None,
                input_labels=lbl if any_points else None,
                input_boxes=box if any_boxes else None,
                original_size=(h, w), clear_old_inputs=bool(req.clearOldInputs))
            obj_ids.extend(ids)
        if mask_objs:
            ids = [int(o.objId) for o in mask_objs]
            masks = [rle_to_mask(o.mask.get("rle", []), int(o.mask.get("width", 0)),
                                 int(o.mask.get("height", 0))).astype(bool) for o in mask_objs]
            processor.add_inputs_to_inference_session(
                inference_session=isess, frame_idx=frame, obj_ids=ids, input_masks=masks)
            obj_ids.extend(ids)
        # Masks dispatch down a different branch than points (add_inputs_to_inference_session:615-616),
        # so a frame carrying both kinds costs two calls and the SECOND assignment to
        # obj_with_new_inputs drops the first group. The per-object inputs themselves are already
        # stored (add_point_inputs / add_mask_inputs, modeling_sam3_tracker_video.py:215/:229) — only
        # this pending list was clobbered, so restoring the union is exactly what the single-call
        # rule buys and nothing more.
        if mask_objs and point_objs:
            isess.obj_with_new_inputs = list(obj_ids)
        out = _forward_frame(vm, processor, sess, isess, frame, reverse=False)
        # Read the answer out of the session instead of asserting it. A SEED lands in
        # cond_frame_outputs, because is_init_cond_frame is `frame_idx not in
        # frames_tracked_per_obj[obj_idx]` (modeling_sam3_tracker_video.py:1801) and an untracked
        # frame satisfies it. A NUDGE on a frame the run has already tracked does not: it stores as
        # NON-conditioning (:1820-1821), and re-running FROM that frame then re-predicts it from the
        # neighbouring memory alone (:1794-1797 takes the else branch with point_inputs None) and
        # overwrites both the stored output and its maskmem with the drifted result — the correction
        # is gone from the emitted mask and from the memory bank. So the client must start the
        # re-propagation at the NEXT frame whenever this comes back false.
        non_cond = [oid for oid in obj_ids
                    if frame not in isess.output_dict_per_obj[isess.obj_id_to_idx(int(oid))]["cond_frame_outputs"]]

    sess.obj_ids = [int(v) for v in out.object_ids]
    sess.seeds.append({
        "frame": frame, "objIds": obj_ids, "clearOldInputs": bool(req.clearOldInputs),
        "kinds": sorted({"mask" if o.mask is not None else ("box" if o.box else "points")
                         for o in req.objects}),
    })
    return {"ok": True, "objIds": list(sess.obj_ids), "conditioning": not non_cond,
            "nonConditioning": non_cond, "consumedAt": frame}


# -------------------------------------------------------------------------- run ----


def _mask_dims(sess: TrackSession) -> tuple[int, int]:
    """Stored bitmap size: maskRes on the long side, never upsampled past the capture. The aspect
    has to survive, because validateMasks (packages/core/src/edits.ts:307-309) rejects a bitmap
    whose width/height disagrees with the volume's camera.aspect by more than 1%."""
    h, w = sess.frame_hw
    scale = min(1.0, sess.mask_res / max(1, max(w, h)))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _sse(event: str, data) -> str:
    """The house SSE frame, byte-identical to serve.mjs:599-600 and the nine other SSE routes — no id:, no retry:, no heartbeat,
    matching every other SSE route in the tree."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _sigmoid(x: float) -> float:
    x = min(30.0, max(-30.0, x))          # object scores measured up to 19.8 logits; exp() overflows
    return 1.0 / (1.0 + math.exp(-x))


@router.get("/run")
def track_run(session: str, start: int = 0, direction: str = Query("forward", alias="dir"),
              limit: int = Query(0, alias="max")):
    """SSE over frames the session already holds, so a correction costs inference and zero upload.
    Every precheck that can fail does so BEFORE the head, as 400/503/409 JSON."""
    sess = _require(session)
    direction = (direction or "forward").lower()
    if direction not in ("forward", "reverse"):
        raise HTTPException(status_code=400, detail="dir must be forward or reverse")
    if sess.isess is None or not sess.obj_ids:
        raise HTTPException(status_code=400, detail="no objects seeded: POST /track/prompt first")
    if start < 0 or start >= sess.frames:
        raise HTTPException(status_code=400, detail=f"start {start} is outside 0..{sess.frames - 1}")

    # The span is built here rather than by propagate_in_video_iterator, which cannot be used at all
    # on this seam: it owns the frame loop, and this loop has to prime the vision cache between
    # frames. Building it explicitly also forces the start frame to be explicit — the iterator
    # defaults start_frame_idx to a min() over every conditioning frame
    # (modeling_sam3_tracker_video.py:2792-2806), which after a correction silently re-runs the
    # whole clip. Reverse is INCLUSIVE of the start frame, matching :2810-2816.
    span = limit if limit and limit > 0 else sess.frames
    if direction == "forward":
        order = list(range(start, min(sess.frames, start + span)))
    else:
        order = list(range(start, max(-1, start - span), -1))
    if not order:
        raise HTTPException(status_code=400, detail="empty span")
    missing = [f for f in order if f not in sess.jpegs]
    if missing:
        raise HTTPException(status_code=400, detail=f"{len(missing)} frames of the span have not "
                                                    f"been uploaded, first {missing[0]}")

    _prewarm()                  # unlocked, and before the run token: a 503 here must not hold it
    token = _acquire_run(sess)
    if token is None:
        raise HTTPException(status_code=409, detail="track busy: one run at a time")
    try:
        with _model_lock:
            vm, processor = _ensure_tracker_video()
    except BaseException:
        _release_run(token)     # nothing has been written yet, so the 503 still reaches the client
        raise

    sess.cancel = False
    sess.stopped_at = None
    sess.at_frame = None
    sess.events = []
    sess.run_id += 1
    sess.state = "running"
    sess.mark("start")
    # The inference session and the frame store are captured HERE, in the route, not read off sess
    # inside the generator. _propagate is a generator function, so nothing in its body runs until
    # the first next() — which happens after starlette has accepted the response — and a
    # /track/close landing in that window runs _drop_session first, so the generator would bind the
    # already-emptied dict and the first prefetch would raise KeyError out of pending.result(). The
    # except there turns that into `event: error`, i.e. a deliberate close reads to the client as a
    # track failure. Bound before the response exists, the cooperative cancel /track/close sets does
    # its job instead: the in-flight frame finishes, the next iteration breaks, and `done` carries
    # cancelled/stoppedAt. The store's lifetime is unchanged: with keepFrames it is the same dict
    # _retired now owns, and without it this reference is the last one and dies with the generator.
    gen = _propagate(sess, vm, processor, sess.isess, sess.jpegs, order,
                     direction == "reverse", token)
    sess.gen = gen              # so the reaper can close a stream nobody will ever drain again
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


def _propagate(sess: TrackSession, vm, processor, isess, jpegs: dict[int, bytes],
               order: list[int], reverse: bool, token: int):
    """Generator of SSE frames. Runs on starlette's threadpool (StreamingResponse wraps a sync
    iterator in iterate_in_threadpool, responses.py:236), so every blocking call here is legal."""
    mw, mh = _mask_dims(sess)
    t_run = time.time()
    masks_out = gaps_out = done = 0
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam-track-pre")

    def emit(event: str, data):
        """Build the frame and declare the stage, in that order: `yield emit(...)` evaluates this
        first, so the session is marked "yield" for exactly the window in which the generator is
        parked waiting for a reader — which is the one window the reaper is allowed to kill."""
        chunk = _sse(event, data)
        sess.mark("yield")
        return chunk

    try:
        yield emit("start", {"session": sess.id, "run": sess.run_id, "start": order[0],
                             "dir": "reverse" if reverse else "forward", "max": len(order),
                             "objIds": list(sess.obj_ids), "trackRes": sess.track_res,
                             "maskRes": sess.mask_res, "frames": sess.frames, "w": mw, "h": mh})
        sess.mark("work")
        yield emit("log", f"[track] sam3 video, {str(SAM_DTYPE).replace('torch.', '')}, {DEVICE}, "
                          f"{len(sess.obj_ids)} object{'s' if len(sess.obj_ids) != 1 else ''}, "
                          f"{'reverse' if reverse else 'forward'} {order[0]} to {order[-1]}")
        sess.mark("work")

        pending = pool.submit(_preprocess_frame, processor, jpegs[order[0]])
        for i, f in enumerate(order):
            px_cpu = pending.result()
            if i + 1 < len(order):
                pending = pool.submit(_preprocess_frame, processor, jpegs[order[i + 1]])
            # The RUN TOKEN, not just the flag. sess.cancel is reset to False by the next
            # track_run (:971), so a generator the reaper stranded and a client later un-stalled
            # would see the flag cleared, resume, and forward into the same inference session as
            # the new run — two propagation loops interleaving single-frame forwards, both
            # appending to sess.events and both writing sess.at_frame. The token is stale the
            # instant anybody else acquires the run, which is exactly the condition.
            if sess.cancel or _track_run["token"] != token:
                sess.stopped_at = f
                break
            sess.at_frame = f
            t0 = time.time()
            sess.mark("gpu")
            with _model_lock:   # taken per FRAME, never for the run: a click interleaves at ~0.3 s
                out = _forward_frame(vm, processor, sess, isess, f, reverse, px_cpu=px_cpu)
                low = out.pred_masks.float().cpu()
                logits = out.object_score_logits.detach().float().cpu().reshape(-1)
                obj_ids = [int(v) for v in out.object_ids]
            sess.mark("work")
            # Deliberately outside the model lock, and deliberately straight to the STORED
            # resolution: post_process_masks bilinearly resamples the 288x288 logits and thresholds
            # after (image_processing_sam3.py:672-678), so going to maskRes in one step keeps the
            # sub-pixel information a threshold-then-box-downsample would have already thrown away.
            # It takes a LIST whose element i is the full 4-D tensor — passing the tensor itself
            # makes masks[0] 3-D and F.interpolate rejects it.
            frame_masks = processor.post_process_masks([low], [[mh, mw]], binarize=True)[0]
            arr = frame_masks.numpy().reshape((-1, mh, mw))
            for oi, obj_id in enumerate(obj_ids):
                logit = float(logits[oi]) if logits.numel() > oi else float(logits[0])
                bits = arr[oi]
                area = float(bits.mean())
                # A gap is emitted, never an empty mask: which of the two policies applies is the
                # client's call, because it depends on the range's mode (an empty keep mask would
                # delete the entire frame through keepPredicateAt, edits.ts:582-586).
                if logit <= 0.0:
                    name, data = "gap", {"frame": f, "objId": obj_id, "reason": "occluded"}
                    gaps_out += 1
                elif area <= 0.0:
                    name, data = "gap", {"frame": f, "objId": obj_id, "reason": "empty"}
                    gaps_out += 1
                else:
                    name = "mask"
                    data = {"frame": f, "objId": obj_id, "w": mw, "h": mh, "rle": mask_to_rle(bits),
                            "score": round(_sigmoid(logit), 4), "area": round(area, 5)}
                    masks_out += 1
                sess.events.append({"event": name, **data})
                yield emit(name, data)
                sess.mark("work")
            done += 1
            ms = (time.time() - t0) * 1000.0
            yield emit("progress", {"done": done, "of": len(order), "msPerFrame": round(ms, 1),
                                    "etaS": round((len(order) - done) * ms / 1000.0, 1)})
            sess.mark("work")
            sess.touch()
        # `frames` is what was actually forwarded, not len(order): a cancel — the user's, or the
        # reaper's on a stall — breaks the loop early, and reporting the full span there is how a
        # silently truncated track reads to the client as a complete one.
        yield emit("done", {"frames": done, "of": len(order), "masks": masks_out, "gaps": gaps_out,
                            "ms": round((time.time() - t_run) * 1000, 1),
                            "cancelled": bool(sess.cancel), "stoppedAt": sess.stopped_at})
    except Exception as e:
        print(f"[ares-sam] track {sess.id} failed: {type(e).__name__}: {e}")
        yield emit("error", {"message": f"{type(e).__name__}: {e}"})   # emit, so a client that is
        # already gone leaves the session in "yield" and the reaper can still take the lock back
    finally:
        if sess.state == "running":
            sess.state = "idle"
        sess.mark("idle")
        sess.touch()
        sess.gen = None
        pool.shutdown(wait=False)
        _release_run(token)
        # A /track/close that arrived mid-run deferred the drop to here, because the memory it
        # reports freeing is held by THIS generator's frame locals until it ends. Doing it here is
        # what makes freedMB honest and what keeps the session counted against TRACK_MAX_SESSIONS
        # for as long as its memory bank is actually resident.
        if sess.close_after is not None:
            keep, sess.close_after = sess.close_after, None
            # Drop the generator's OWN references first. _drop_session measures
            # torch.cuda.memory_allocated() around clearing sess.isess, and these frame locals are
            # the other half of what pins the memory bank.
            del isess, jpegs
            with _store_lock:
                freed = _drop_session(sess.id, keep_frames=keep)
            print(f"[ares-sam] track: {sess.id} closed after its run ({freed} MiB, frames "
                  f"{'retained' if keep else 'dropped'})")


# ------------------------------------------------------- cancel, close, replay ----


class TrackSessionRef(BaseModel):
    session: str


@router.post("/cancel")
def track_cancel(req: TrackSessionRef):
    """Cooperative flag, checked between frames. Sent BEFORE the client closes its EventSource,
    because whether the dev server's /sam proxy propagates a client disconnect to FastAPI's
    request.is_disconnected() is not something this tree has ever exercised. Every mask already
    emitted is already a keyframe on the client, so a cancel leaves a shorter but valid track."""
    sess = _require(req.session)
    sess.cancel = True
    # at_frame, not stopped_at: the in-flight frame finishes before the flag is read, so the honest
    # answer to "where did it stop" at cancel time is the frame currently under the GPU.
    return {"ok": True, "stoppedAt": sess.at_frame, "events": len(sess.events)}


class TrackCloseRequest(BaseModel):
    session: str
    keepFrames: bool = True


@router.post("/close")
def track_close(req: TrackCloseRequest):
    """Free a session's inference state. On a session whose run is still streaming this only ARMS
    the drop: the memory bank is held by the generator's frame locals until it ends, so dropping
    now would report freedMB 0.0, free nothing, and stop the session counting against
    TRACK_MAX_SESSIONS while its ~914 MiB is still resident — enough for three memory banks to sit
    on a card documented to hold two. The cooperative cancel ends the run within one frame and the
    generator's finally does the drop."""
    sess = _require(req.session)
    sess.cancel = True
    if sess.state == "running":
        sess.close_after = bool(req.keepFrames)
        return {"ok": True, "pending": True, "framesKept": bool(req.keepFrames)}
    with _store_lock:
        freed = _drop_session(sess.id, keep_frames=bool(req.keepFrames))
    return {"ok": True, "pending": False, "freedMB": freed, "framesKept": bool(req.keepFrames)}


@router.get("/sessions")
def track_sessions():
    """Session enumeration, so a tab that reloaded mid-run can find the run it lost instead of
    opening a second one against a service that only holds two."""
    now = time.time()
    return {
        "sessions": [{
            "session": s.id, "clip": s.clip, "camKey": s.cam_key, "frames": s.frames,
            "stored": len(s.jpegs), "bytes": s.store_bytes(), "trackRes": s.track_res,
            "maskRes": s.mask_res, "state": s.state, "objIds": list(s.obj_ids),
            "run": s.run_id, "events": len(s.events), "seeds": s.seeds,
            "ageS": round(now - s.created, 1), "idleS": round(now - s.touched, 1),
        } for s in sorted(_sessions.values(), key=lambda s: s.created)],
        "limit": TRACK_MAX_SESSIONS,
        "retained": [{"key": k, "frames": len(v["jpegs"]), "ageS": round(now - v["at"], 1)}
                     for k, v in _retired.items()],
    }


@router.get("/results")
def track_results(session: str, cursor: int = Query(0, alias="from"),
                  page: int = Query(TRACK_REPLAY_PAGE, alias="limit")):
    """Replay of the masks and gaps a run has already emitted, for a client that lost the stream.

    `from` is a 0-based index into THIS RUN'S emitted events, not a frame number. A reverse run
    emits frames in decreasing order and a multi-object run emits one event per object per frame,
    so a frame number cannot order a replay; `next` is the cursor to ask for after this page.

    `run` is the run id the `start` event carried. track_run clears sess.events in place, so a
    client that starts its next run mid-replay would otherwise watch `total` drop to 0 and read the
    resulting empty page as a finished replay. Pin the id and a wipe surfaces as a changed id."""
    sess = _require(session)
    cursor = max(0, int(cursor))
    page = max(1, min(4096, int(page)))
    events = sess.events[cursor:cursor + page]
    nxt = cursor + len(events)
    return {
        "session": sess.id, "run": sess.run_id, "state": sess.state, "from": cursor, "next": nxt,
        "total": len(sess.events), "more": nxt < len(sess.events),
        "running": sess.state == "running", "cancelled": bool(sess.cancel),
        "stoppedAt": sess.stopped_at, "events": events,
    }


def health_fields() -> dict:
    """Merged into GET /health by main.py. Loads nothing and takes no lock."""
    return {
        "trackReady": _video_model is not None,
        "trackLoading": _track_loading,
        "trackError": _track_error,
        "trackSessions": len(_sessions),
        "trackRes": int(_track_processor.target_size) if _track_processor is not None else None,
        "trackMaxSessions": TRACK_MAX_SESSIONS,
    }
