# ARES local SAM service (editor v2, docs/editor-v2-design.md section 8.1).
#
# FastAPI facade over two selectable segmentation backends, same /segment API for both:
#   sam3   (primary)  — Meta SAM 3 via HF transformers (Sam3TrackerModel, SAM1-style click
#                       prompts, bf16 on CUDA). Weights: transformers-format facebook/sam3
#                       snapshot at <repo-root>/sam3 (override with SAM3_DIR).
#   vit_h  (fallback) — SAM v1 ViT-H fp16 (segment-anything + safetensors, fp32 promote,
#                       fp16 autocast predict), the pre-2026-07 backend.
# SAM_BACKEND=auto|sam3|vit_h picks the order; auto tries sam3 then vit_h.
#
# TEXT/CONCEPT prompts (e.g. "the skateboard"): a second model, Sam3Model (+
# Sam3Processor), loaded ONLY on the sam3 backend, ONLY after the tracker above has loaded —
# it shares the tracker's vision_encoder submodule BY REFERENCE (same checkpoint dir, same
# tensor group on disk: `detector_model.vision_encoder.*` is the ONLY copy in sam3/model.
# safetensors, confirmed by header inspection) so the 6 GB card never holds two ~908 MB bf16
# vision towers. POST /segment_text {image, text, maxInstances?, scoreThreshold?}. Gate:
# SAM_TEXT=0 disables it (tracker/click-select is unaffected either way).
#
# Run: the dev server auto-starts this on demand (serve.mjs samEnsure, /sam/start SSE, and
# the editor's SAM row); `ARES.vbs sam` and tools/sam-service/run-sam-service.ps1 remain as
# manual alternatives. Direct command (dedicated env, torch cu124 + transformers 5):
#   tools/sam-service/env/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 7263
#
# The ARES dev server reverse-proxies this under /sam/* (COEP blocks direct cross-origin
# fetches from the app), so the browser calls /sam/segment, /sam/segment_text and /sam/health.

import base64
import hashlib
import io
import os
import threading
import time
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def _resolve_sam3_dir():
    """Where the SAM 3 weights actually are, in the order a user is likely to have put them.

    `hf download facebook/sam3` — the documented way to get this model, and what the Settings
    tab now runs — puts it in the shared HF cache, NOT in a folder next to the repo. Defaulting
    to <repo>/sam3 meant a perfectly good multi-gigabyte download sat there unseen while the
    service reported "weights missing". Checks, in order: an explicit override, the repo-local
    folder (how this was originally set up), the shared cache, and finally the bare repo id so
    transformers can fetch it itself.
    """
    env = os.environ.get("SAM3_DIR")
    if env:
        return env
    _ARES_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for local in (os.path.join(_ARES_ROOT, "sam3"), os.path.join(_REPO_ROOT, "sam3")):
        if os.path.isfile(os.path.join(local, "config.json")):
            return local
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE") or (
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME")
        else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    repo = os.path.join(hub, "models--facebook--sam3")
    snaps = os.path.join(repo, "snapshots")
    if os.path.isdir(snaps):
        revs = sorted(os.listdir(snaps))
        try:    # refs/main names the current revision; prefer it when it is there
            with open(os.path.join(repo, "refs", "main"), encoding="utf-8") as fh:
                head = fh.read().strip()
            if head in revs:
                revs.insert(0, revs.pop(revs.index(head)))
        except OSError:
            pass
        for rev in revs:
            snap = os.path.join(snaps, rev)
            if os.path.isfile(os.path.join(snap, "config.json")):
                return snap
    return "facebook/sam3"   # let transformers resolve + download it (uses the stored HF token)


SAM3_DIR = _resolve_sam3_dir()
VITH_CHECKPOINT = os.environ.get(
    "SAM_CKPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sam_vit_h_4b8939_fp16.safetensors"),
)
BACKEND_PREF = os.environ.get("SAM_BACKEND", "auto")  # auto | sam3 | vit_h
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _best_dtype():
    """fp16 or bf16 — decided by the GPU, not by habit.

    Native bf16 tensor cores start at Ampere (sm_80). Below that bf16 drops off the tensor-core
    path entirely: measured on a 2080 Ti (sm_75) it runs 43.0 TFLOP/s in fp16 against 7.3 in
    bf16 — slower even than fp32 — while allocating byte-for-byte the same VRAM (1653 MB either
    way for tracker+concept). So on pre-Ampere cards fp16 is a straight ~6x win for no cost.
    Ampere and newer keep bf16 for its wider exponent range.
    """
    if DEVICE != "cuda":
        return torch.float32
    major = torch.cuda.get_device_properties(0).major
    return torch.bfloat16 if major >= 8 else torch.float16


SAM_DTYPE = _best_dtype()
# Default ON — VRAM bracket measured on an RTX 3060 6 GB:
# the concept (text) model shares the tracker's vision tower, so it adds only ~+226 MB whole-GPU
# (peak 3671 MB combined, 2473 MB left free for the viewer — >2× the 1.2 GB floor). Set SAM_TEXT=0
# to disable; the tracker/click tool is unaffected either way (concept load runs after + independently).
SAM_TEXT = os.environ.get("SAM_TEXT", "1").strip().lower() not in ("0", "false", "off", "no")

app = FastAPI(title="ares-sam", version="2.0")

# ---------------------------------------------------------------- model ----

_backend = None          # "sam3" | "vit_h" once loaded
_sam3_model = None
_sam3_processor = None
_predictor = None        # vit_h SamPredictor
_load_error = None
_lock = threading.Lock()  # both backends hold single-image state; serialize requests.

# ---- TEXT/CONCEPT model — loaded after _sam3_model, sharing its vision tower.
_sam3_concept_model = None
_sam3_concept_processor = None
_concept_load_error = None


def _load_sam3():
    """SAM 3 point-prompted image segmentation via transformers (Sam3TrackerModel)."""
    global _sam3_model, _sam3_processor
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    if not os.path.isdir(SAM3_DIR):
        raise FileNotFoundError(f"SAM3 weights dir not found: {SAM3_DIR}")
    t0 = time.time()
    _sam3_processor = Sam3TrackerProcessor.from_pretrained(SAM3_DIR)
    dtype = SAM_DTYPE
    _sam3_model = Sam3TrackerModel.from_pretrained(SAM3_DIR, dtype=dtype).to(DEVICE).eval()
    print(f"[ares-sam] sam3 tracker loaded from {SAM3_DIR} on {DEVICE} ({dtype}) in {time.time() - t0:.1f}s")


def _load_sam3_concept():
    """SAM 3 TEXT/concept-prompted segmentation (Sam3Model + Sam3Processor).
    MUST run after _load_sam3() (the tracker) has succeeded: the vision tower is shared BY
    REFERENCE, not reloaded, to avoid a second ~908 MB bf16 copy on the 6 GB card. Confirmed
    by reading sam3/model.safetensors' header: the checkpoint stores exactly ONE vision-encoder
    tensor group, under `detector_model.vision_encoder.*` (538 tensors) — there is no separate
    `tracker_model.vision_encoder.*`. transformers' flexible loader (the "you are using a model
    of type sam3_video to instantiate sam3_tracker" path) maps those SAME on-disk tensors into
    whichever of Sam3Model / Sam3TrackerModel is instantiated — verified live: a freshly loaded
    Sam3TrackerModel's vision_encoder weight is bit-identical to the checkpoint's
    detector_model.vision_encoder tensor (torch.equal, max abs diff 0.0). So both models load
    numerically-identical but SEPARATELY ALLOCATED copies unless we intervene here."""
    global _sam3_concept_model, _sam3_concept_processor
    from transformers import Sam3Model, Sam3Processor

    if _sam3_model is None:
        raise RuntimeError("sam3 tracker must load before the concept model (vision-encoder sharing needs it first)")
    if not os.path.isdir(SAM3_DIR):
        raise FileNotFoundError(f"SAM3 weights dir not found: {SAM3_DIR}")
    t0 = time.time()
    processor = Sam3Processor.from_pretrained(SAM3_DIR)
    dtype = SAM_DTYPE
    concept = Sam3Model.from_pretrained(SAM3_DIR, dtype=dtype)
    # SHARE DIRECTION MATTERS (bug found 2026-07-13): the checkpoint stores exactly ONE tower
    # (detector_model.vision_encoder.*, 538 tensors; NO tracker_model.vision_encoder.*), and
    # Sam3Model loads it NATIVELY — the tracker gets it through transformers' flexible
    # cross-architecture loader, which mis-maps enough of it that the DETR presence head
    # collapses: person scores 0.977 (fresh tower) → 0.336 (tracker's tower grafted in), i.e.
    # below threshold → /segment_text returned ZERO instances for everything, always. The
    # robust promptable CLICK path shrugged the same corruption off, which hid the bug.
    # Fix: graft the CONCEPT model's
    # natively-loaded tower INTO the tracker (reverse of the original swap) — still exactly one
    # tower on the GPU, but it's the true checkpoint tower for BOTH models. The tracker's
    # flexible-loaded copy is dropped and freed.
    concept = concept.to(DEVICE).eval()
    _sam3_model.vision_encoder = concept.vision_encoder
    _assert_vision_encoder_shared(concept)
    # Free the tracker's dropped (mis-mapped) tower right away — during the graft both towers
    # sit on the GPU transiently (~+0.9 GB); without this the cached blocks linger.
    import gc as _gc
    _gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _sam3_concept_model, _sam3_concept_processor = concept, processor
    print(f"[ares-sam] sam3 concept (text) model loaded from {SAM3_DIR} on {DEVICE} ({dtype}) in "
          f"{time.time() - t0:.1f}s (vision encoder SHARED with tracker, not duplicated)")


def _assert_vision_encoder_shared(concept_model) -> None:
    """Fails LOUDLY — raised here, printed by the caller, and every /segment_text call 503s
    with the reason — instead of silently running with a duplicated ~900 MB vision tower. This
    is the guard against a future transformers rename/refactor of this attribute breaking the
    share under our feet without anyone noticing until VRAM runs out."""
    tracker_vis = _sam3_model.vision_encoder
    if concept_model.vision_encoder is not tracker_vis:
        raise RuntimeError(
            "sam3 concept/tracker vision_encoder MODULE identity check FAILED — sharing did not "
            "take; refusing to run with a silently duplicated vision tower")
    t_params = list(tracker_vis.parameters())
    c_params = list(concept_model.vision_encoder.parameters())
    if not t_params or len(t_params) != len(c_params):
        raise RuntimeError(
            f"sam3 vision_encoder parameter-list mismatch after sharing ({len(t_params)} vs "
            f"{len(c_params)}) — refusing to start")
    for tp, cp in zip(t_params, c_params):
        if tp.data_ptr() != cp.data_ptr():
            raise RuntimeError(
                "sam3 vision_encoder parameter TENSORS are not the same GPU allocation after "
                "sharing (data_ptr mismatch) — refusing to run with a silently duplicated vision tower")


def _load_vith():
    """SAM v1 ViT-H fp16 — plain safetensors load (no mmgp/accelerate needed)."""
    global _predictor
    from safetensors.torch import load_file
    from segment_anything import sam_model_registry, SamPredictor

    if not os.path.isfile(VITH_CHECKPOINT):
        raise FileNotFoundError(f"SAM ViT-H checkpoint not found: {VITH_CHECKPOINT}")
    t0 = time.time()
    model = sam_model_registry["vit_h"](checkpoint=None)
    model.load_state_dict(load_file(VITH_CHECKPOINT))
    model.to(torch.float32)  # fp16 storage -> fp32 weights (precision), predict under autocast
    model.to(device=DEVICE)
    model.eval()
    _predictor = SamPredictor(model)
    print(f"[ares-sam] vit_h loaded from {VITH_CHECKPOINT} on {DEVICE} in {time.time() - t0:.1f}s")


def _loader():
    global _backend, _load_error, _concept_load_error
    order = {"auto": ["sam3", "vit_h"], "sam3": ["sam3"], "vit_h": ["vit_h"]}.get(BACKEND_PREF, ["sam3", "vit_h"])
    errors = []
    for name in order:
        try:
            (_load_sam3 if name == "sam3" else _load_vith)()
            _backend = name
            break
        except Exception as e:  # try the next backend; surface everything via /health
            errors.append(f"{name}: {type(e).__name__}: {e}")
            print(f"[ares-sam] backend {name} failed: {errors[-1]}")
    else:
        _load_error = " | ".join(errors)
        return

    # Text/concept-prompted segmentation: only meaningful on the sam3 backend
    # (vit_h has no text path). Gated by SAM_TEXT so a VRAM-tight machine can disable it
    # without losing the click-to-select tool — a concept-model failure never takes the
    # tracker down (it already finished loading above).
    if _backend == "sam3" and SAM_TEXT:
        try:
            _load_sam3_concept()
        except Exception as e:
            _concept_load_error = f"{type(e).__name__}: {e}"
            print(f"[ares-sam] concept model (text prompts) failed to load: {_concept_load_error} — /segment_text will 503")


# Load on a BACKGROUND thread so uvicorn binds the port within ~1 s of process start.
# Loading at import kept the port unbound for the whole model load; a second launcher
# health-probed into that silence, started another instance, and the loser died on bind
# (WinError 10048 — the 2026-07-10 startup failure). With bind-first, /health answers
# {"loading": true} immediately and duplicate launchers exit instead of racing.
threading.Thread(target=_loader, daemon=True, name="sam-load").start()


def _ready():
    """503 while loading or after a failed load; returns the active backend name."""
    if _load_error is not None:
        raise HTTPException(status_code=503, detail=f"SAM model failed to load: {_load_error}")
    if _backend is None:
        raise HTTPException(status_code=503, detail="SAM model still loading")
    return _backend


# ------------------------------------------------------- embedding cache ----
# Both backends cache the vision-encoder output keyed by the image bytes' sha256, so
# refining clicks on the SAME held frame skip the encoder entirely:
#   vit_h — predictor.set_image() (~0.6-1.2 s/frame on the 3060) -> ~10-20 ms repeats.
#   sam3  — get_image_embeddings() (the bulk of the 360-550 ms/click) -> prompt+decoder only.

_EMBED_CAP = int(os.environ.get("SAM_EMBED_CACHE", "6"))
_embeds: "OrderedDict[str, tuple]" = OrderedDict()
_sam3_embeds: "OrderedDict[str, object]" = OrderedDict()
_sam3_embed_broken = False  # set on first failure; falls back to per-call encoding

# Concept-model (text) cache is PARALLEL to _sam3_embeds, not shared with it: the tracker's
# get_image_embeddings() reshapes the vision-encoder output per backbone_feature_sizes and
# folds in a "no memory" token (tracker-specific, for click-refinement across calls) — the
# result isn't the plain Sam3VisionEncoderOutput shape Sam3Model.forward(vision_embeds=...)
# expects. Confirmed by reading both call sites, not guessed. Sam3Model.get_vision_features()
# returns the right shape for THIS model's own reuse path, so it gets its own small cache.
_sam3_concept_embeds: "OrderedDict[str, object]" = OrderedDict()
_sam3_concept_embed_broken = False


def _lru_put(cache: OrderedDict, key: str, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _EMBED_CAP:
        cache.popitem(last=False)


def _vith_set_image_cached(predictor, raw: bytes, img: np.ndarray) -> None:
    key = hashlib.sha256(raw).hexdigest()
    hit = _embeds.get(key)
    if hit is not None:
        _embeds.move_to_end(key)
        predictor.features, predictor.original_size, predictor.input_size = hit
        predictor.is_image_set = True
        return
    predictor.set_image(img)
    _embeds[key] = (predictor.features, predictor.original_size, predictor.input_size)
    _embeds.move_to_end(key)
    while len(_embeds) > _EMBED_CAP:
        _embeds.popitem(last=False)


# ---------------------------------------------------------------- routes ----


@app.get("/health")
def health():
    return {
        "ok": _backend is not None,
        "loading": _backend is None and _load_error is None,
        "error": _load_error,
        "device": DEVICE,
        "backend": _backend,
        "model": {"sam3": f"sam3_tracker_{str(SAM_DTYPE).replace('torch.', '')}",
                  "vit_h": "sam_vit_h_fp16"}.get(_backend or "", "loading"),
        "dtype": str(SAM_DTYPE).replace("torch.", ""),
        "weights": SAM3_DIR if _backend == "sam3" else VITH_CHECKPOINT,
        # Text/concept prompts — independent of the tracker above.
        "textEnabled": SAM_TEXT,
        "textReady": _sam3_concept_model is not None,
        "textLoading": SAM_TEXT and _backend == "sam3" and _sam3_concept_model is None and _concept_load_error is None,
        "textError": _concept_load_error,
    }


class SegmentRequest(BaseModel):
    image: str                      # base64 PNG or JPEG (data: URL prefix tolerated)
    points: List[List[float]]      # [[x, y], ...] in image pixel coordinates
    labels: List[int]              # 1 = foreground, 0 = background, one per point


def _decode_image(req) -> tuple:
    b64 = req.image.split(",", 1)[1] if req.image.startswith("data:") else req.image
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image is not valid base64")
    try:
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image is not decodable PNG/JPEG: {e}")
    return raw, img


def _segment_sam3(img: np.ndarray, coords: list, labels: list, img_key: str) -> tuple:
    global _sam3_embed_broken
    processor, model = _sam3_processor, _sam3_model
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="sam3 backend not loaded")
    # SAM1-style single-object prompt: one point group holding all clicks.
    inputs = processor(
        images=Image.fromarray(img),
        input_points=[[coords]],
        input_labels=[[labels]],
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        # Vision-feature cache: the encoder dominates the click latency; refining clicks on
        # the same frame reuse its output. Any failure marks the cache broken and falls back
        # to per-call encoding so segmentation never breaks on a transformers API change.
        outputs = None
        if not _sam3_embed_broken:
            try:
                emb = _sam3_embeds.get(img_key)
                if emb is None:
                    emb = model.get_image_embeddings(inputs["pixel_values"])
                    _lru_put(_sam3_embeds, img_key, emb)
                else:
                    _sam3_embeds.move_to_end(img_key)
                prompt_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
                outputs = model(**prompt_inputs, image_embeddings=emb, multimask_output=True)
            except Exception as e:
                _sam3_embed_broken = True
                print(f"[ares-sam] sam3 feature cache disabled ({type(e).__name__}: {e}); per-call encoding")
        if outputs is None:
            outputs = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(
        outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu()
    )[0]  # (point_batches, num_masks, H, W) or (num_masks, H, W)
    scores = outputs.iou_scores.float().cpu().numpy().reshape(-1)
    m = masks.numpy()
    m = m.reshape((-1,) + m.shape[-2:])  # -> (num_masks, H, W)
    best = int(np.argmax(scores))
    return (m[best] > 0.5).astype(np.uint8), float(scores[best])


def _segment_text_sam3(img: np.ndarray, text: str, score_threshold: float, img_key: str) -> tuple:
    """Returns (masks, scores) — both instance arrays sorted by score DESC; the endpoint slices
    to maxInstances. Mirrors _segment_sam3's cache/fallback structure: any cache failure marks
    it broken and falls back to per-call encoding so text segmentation never breaks on a
    transformers API change. No autocast (matching _segment_sam3, not _segment_vith): the
    concept model, like the tracker, is loaded natively in bf16, so no fp32->fp16 promotion
    step is needed here."""
    global _sam3_concept_embed_broken
    processor, model = _sam3_concept_processor, _sam3_concept_model
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="sam3 concept (text) backend not loaded" +
                             (f": {_concept_load_error}" if _concept_load_error else ""))
    inputs = processor(images=Image.fromarray(img), text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = None
        if not _sam3_concept_embed_broken:
            try:
                vis = _sam3_concept_embeds.get(img_key)
                if vis is None:
                    vis = model.get_vision_features(inputs["pixel_values"])
                    _lru_put(_sam3_concept_embeds, img_key, vis)
                else:
                    _sam3_concept_embeds.move_to_end(img_key)
                text_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
                outputs = model(vision_embeds=vis, **text_inputs)
            except Exception as e:
                _sam3_concept_embed_broken = True
                print(f"[ares-sam] sam3 concept feature cache disabled ({type(e).__name__}: {e}); per-call encoding")
        if outputs is None:
            outputs = model(**inputs)
    h, w = img.shape[:2]
    results = processor.post_process_instance_segmentation(outputs, threshold=score_threshold, target_sizes=[(h, w)])[0]
    scores = results["scores"].float().cpu().numpy()
    masks = results["masks"].cpu().numpy().astype(np.uint8)  # (num_kept, H, W) in {0,1}, already at img size
    order = np.argsort(-scores)
    return masks[order], scores[order]


def _segment_vith(raw: bytes, img: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> tuple:
    predictor = _predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="vit_h backend not loaded")
    _vith_set_image_cached(predictor, raw, img)
    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                masks, scores, _ = predictor.predict(
                    point_coords=coords, point_labels=labels, multimask_output=True)
        else:
            masks, scores, _ = predictor.predict(
                point_coords=coords, point_labels=labels, multimask_output=True)
    best = int(np.argmax(scores))
    return masks[best].astype(np.uint8), float(scores[best])


@app.post("/segment")
def segment(req: SegmentRequest):
    if not req.points or len(req.points) != len(req.labels):
        raise HTTPException(status_code=400, detail="points and labels must be non-empty and the same length")
    if any(len(p) != 2 for p in req.points):
        raise HTTPException(status_code=400, detail="each point must be [x, y]")
    raw, img = _decode_image(req)

    t0 = time.time()
    with _lock:
        backend = _ready()
        if backend == "sam3":
            mask, score = _segment_sam3(
                img, [list(map(float, p)) for p in req.points], list(req.labels),
                hashlib.sha256(raw).hexdigest())
        else:
            mask, score = _segment_vith(
                raw, img,
                np.asarray(req.points, dtype=np.float32),
                np.asarray(req.labels, dtype=np.int32))

    buf = io.BytesIO()
    Image.fromarray(mask * 255, mode="L").save(buf, "PNG")     # single-channel PNG
    return {
        "mask": base64.b64encode(buf.getvalue()).decode("ascii"),
        "score": score,
        "backend": backend,
        "ms": round((time.time() - t0) * 1000, 1),
    }


class SegmentTextRequest(BaseModel):
    image: str                      # base64 PNG or JPEG (data: URL prefix tolerated)
    text: str                       # concept prompt, e.g. "the skateboard"
    maxInstances: int = 8
    scoreThreshold: float = 0.5


@app.post("/segment_text")
def segment_text(req: SegmentTextRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    if not SAM_TEXT:
        raise HTTPException(status_code=503, detail="text-prompted segmentation is disabled on this service (SAM_TEXT=0)")
    max_instances = max(1, min(64, int(req.maxInstances or 8)))
    score_threshold = min(0.999, max(0.0, float(req.scoreThreshold)))
    raw, img = _decode_image(req)

    t0 = time.time()
    with _lock:
        if _backend is None and _load_error is None:
            raise HTTPException(status_code=503, detail="SAM model still loading")
        if _sam3_concept_model is None:
            raise HTTPException(status_code=503, detail="sam3 concept (text) model not loaded" +
                                 (f": {_concept_load_error}" if _concept_load_error else ""))
        masks, scores = _segment_text_sam3(img, text, score_threshold, hashlib.sha256(raw).hexdigest())
    masks, scores = masks[:max_instances], scores[:max_instances]

    mask_b64 = []
    for m in masks:
        buf = io.BytesIO()
        Image.fromarray((m * 255).astype(np.uint8), mode="L").save(buf, "PNG")     # single-channel PNG
        mask_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return {
        "masks": mask_b64,
        "scores": [float(s) for s in scores],
        "ms": round((time.time() - t0) * 1000, 1),
    }


# =========================================================== UPSCALE (Real-ESRGAN) ==
# Added for Fauxtofab. GPU super-resolution via spandrel. Loads LAZILY on the first
# /upscale call so it never slows SAM startup or holds VRAM unless used; a load failure
# is isolated (503 on /upscale, reported on /health) and never touches segmentation.
# Tiled inference keeps peak VRAM ~1 GB, so it coexists with SAM 3 on a 6 GB card.
UPSCALE_CKPT = os.environ.get("UPSCALE_CKPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "RealESRGAN_x4plus.pth"))
UPSCALE_TILE = int(os.environ.get("UPSCALE_TILE", "256"))
UPSCALE_PAD  = int(os.environ.get("UPSCALE_PAD", "16"))

_upscaler = None
_upscale_error = None


def _ensure_upscaler():
    global _upscaler, _upscale_error
    if _upscaler is not None:
        return _upscaler
    if _upscale_error is not None:
        raise HTTPException(status_code=503, detail=f"upscaler failed to load: {_upscale_error}")
    try:
        from spandrel import ModelLoader
        if not os.path.isfile(UPSCALE_CKPT):
            raise FileNotFoundError(f"upscale weight not found: {UPSCALE_CKPT}")
        t0 = time.time()
        _upscaler = ModelLoader().load_from_file(UPSCALE_CKPT).to(DEVICE).eval()
        print(f"[ares-sam] upscaler loaded ({_upscaler.architecture.name} x{_upscaler.scale}) "
              f"from {UPSCALE_CKPT} on {DEVICE} in {time.time() - t0:.1f}s")
        return _upscaler
    except Exception as e:
        _upscale_error = f"{type(e).__name__}: {e}"
        print(f"[ares-sam] upscaler failed to load: {_upscale_error}")
        raise HTTPException(status_code=503, detail=f"upscaler failed to load: {_upscale_error}")


def _run_sr(img, tile, pad):
    """Padded-tile super-resolution: each output tile keeps only its centre (the pad gives
    seam-free context), accumulated on CPU so peak GPU memory stays ~one tile."""
    scale = _upscaler.scale
    h, w, _ = img.shape
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255).to(DEVICE)
    out = np.zeros((h * scale, w * scale, 3), np.uint8)
    for ty in range(0, h, tile):
        for tx in range(0, w, tile):
            y0, x0 = max(0, ty - pad), max(0, tx - pad)
            y1, x1 = min(h, ty + tile + pad), min(w, tx + tile + pad)
            sub = t[:, :, y0:y1, x0:x1]
            with torch.no_grad():
                if DEVICE == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        sr = _upscaler(sub)
                else:
                    sr = _upscaler(sub)
            ky1, kx1 = min(h, ty + tile), min(w, tx + tile)
            oy, ox = (ty - y0) * scale, (tx - x0) * scale
            kh, kw = (ky1 - ty) * scale, (kx1 - tx) * scale
            region = sr[:, :, oy:oy + kh, ox:ox + kw].clamp(0, 1).mul(255).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
            out[ty * scale:ky1 * scale, tx * scale:kx1 * scale] = region
    del t
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


class UpscaleRequest(BaseModel):
    image: str            # base64 PNG/JPEG (data: URL tolerated)
    tile: int = 0         # 0 -> UPSCALE_TILE


@app.post("/upscale")
def upscale(req: UpscaleRequest):
    raw, img = _decode_image(req)
    t0 = time.time()
    with _lock:  # share the GPU lock with SAM so the two never run concurrently (OOM guard)
        _ensure_upscaler()
        tile = req.tile if req.tile and req.tile > 0 else UPSCALE_TILE
        out = _run_sr(img, tile, UPSCALE_PAD)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, "PNG")
    return {
        "image": base64.b64encode(buf.getvalue()).decode("ascii"),
        "scale": int(_upscaler.scale),
        "ms": round((time.time() - t0) * 1000, 1),
    }


# ============================================================== ADD DETAIL (SD 1.5) ==
# Added for Fauxtofab. Tiled SD 1.5 img2img detail pass (DreamShaper 8) via diffusers.
# Loads LAZILY on first /detail; model_cpu_offload + vae tiling + attention slicing keep
# peak GPU ~2 GB so it coexists with SAM 3 on a 6 GB card. Tiled with seam-free feather
# blending so full resolution is preserved. Isolated failure (503), never touches SAM.
DETAIL_MODEL    = os.environ.get("DETAIL_MODEL", "Lykon/dreamshaper-8")
DETAIL_TILE     = int(os.environ.get("DETAIL_TILE", "448"))
DETAIL_OVERLAP  = int(os.environ.get("DETAIL_OVERLAP", "64"))
DETAIL_STEPS    = int(os.environ.get("DETAIL_STEPS", "20"))
DETAIL_GUIDANCE = float(os.environ.get("DETAIL_GUIDANCE", "6.0"))

_detail_pipe = None
_detail_error = None


def _ensure_detail():
    global _detail_pipe, _detail_error
    if _detail_pipe is not None:
        return _detail_pipe
    if _detail_error is not None:
        raise HTTPException(status_code=503, detail=f"detail model failed to load: {_detail_error}")
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        t0 = time.time()
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            DETAIL_MODEL, torch_dtype=(torch.float16 if DEVICE == "cuda" else torch.float32), safety_checker=None)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        if DEVICE == "cuda":
            pipe.enable_vae_tiling()
            pipe.enable_model_cpu_offload()   # stream to GPU -> coexists with SAM on 6 GB
        else:
            pipe = pipe.to(DEVICE)
        _detail_pipe = pipe
        print(f"[ares-sam] detail model loaded ({DETAIL_MODEL}) on {DEVICE} in {time.time() - t0:.1f}s")
        return _detail_pipe
    except Exception as e:
        _detail_error = f"{type(e).__name__}: {e}"
        print(f"[ares-sam] detail model failed to load: {_detail_error}")
        raise HTTPException(status_code=503, detail=f"detail model failed to load: {_detail_error}")


def _detail_ramp(n, ov):
    r = np.ones(n, np.float32)
    ov = min(ov, n // 2)
    if ov > 0:
        r[:ov] = np.linspace(0.0, 1.0, ov, endpoint=False)
        r[-ov:] = np.linspace(1.0, 0.0, ov, endpoint=False)
    return np.maximum(r, 0.02)  # floor so image-border pixels never divide to black


def _run_detail(img, prompt, negative, strength, steps, guidance, tile, overlap):
    h, w, _ = img.shape
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 1), np.float32)
    step = max(8, tile - overlap)
    ys = sorted(set(list(range(0, max(1, h - overlap), step)) + [max(0, h - tile)]))
    xs = sorted(set(list(range(0, max(1, w - overlap), step)) + [max(0, w - tile)]))
    gen = torch.Generator("cpu").manual_seed(0)
    ntiles = 0
    for y0 in ys:
        for x0 in xs:
            th = min(tile, h - y0) // 8 * 8
            tw = min(tile, w - x0) // 8 * 8
            if th < 8 or tw < 8:
                continue
            sub = img[y0:y0 + th, x0:x0 + tw]
            res = _detail_pipe(prompt=prompt, negative_prompt=negative, image=Image.fromarray(sub),
                               strength=strength, num_inference_steps=steps, guidance_scale=guidance,
                               generator=gen).images[0]
            rn = np.asarray(res, np.float32)
            wt = (_detail_ramp(th, overlap)[:, None] * _detail_ramp(tw, overlap)[None, :])[:, :, None]
            acc[y0:y0 + th, x0:x0 + tw] += rn * wt
            wsum[y0:y0 + th, x0:x0 + tw] += wt
            ntiles += 1
    wsum[wsum == 0] = 1.0
    return (acc / wsum).clip(0, 255).astype(np.uint8), ntiles


class DetailRequest(BaseModel):
    image: str
    prompt: str = "highly detailed, sharp focus, fine texture, natural detail, high quality"
    negative: str = "blurry, low quality, deformed, oversharpened, artifacts"
    strength: float = 0.30
    tile: int = 0


@app.post("/detail")
def detail(req: DetailRequest):
    raw, img = _decode_image(req)
    strength = min(0.95, max(0.05, float(req.strength)))
    tile = req.tile if req.tile and req.tile > 0 else DETAIL_TILE
    t0 = time.time()
    with _lock:  # share the GPU lock with SAM/upscale
        _ensure_detail()
        try:
            out, ntiles = _run_detail(img, req.prompt, req.negative, strength, DETAIL_STEPS, DETAIL_GUIDANCE, tile, DETAIL_OVERLAP)
        except getattr(torch.cuda, "OutOfMemoryError", RuntimeError):  # CPU-only torch has no cuda.OutOfMemoryError
            torch.cuda.empty_cache()
            out, ntiles = _run_detail(img, req.prompt, req.negative, strength, DETAIL_STEPS, DETAIL_GUIDANCE, 320, 48)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, "PNG")
    return {
        "image": base64.b64encode(buf.getvalue()).decode("ascii"),
        "tiles": ntiles,
        "ms": round((time.time() - t0) * 1000, 1),
    }
