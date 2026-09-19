# ARES SAM 3 video-tracker probe — step 2 of docs/sam-propagation-plan.md.
#
# Read-only and NEVER imported by main.py or run by CI: every verb here loads a 3.4 GB
# checkpoint. Its whole job is to settle by measurement the three facts the mask-propagation
# design rests on, so docs/editor-v2-design.md 8.4 can quote numbers taken on this machine
# instead of the estimates it carried while the feature was unbuilt.
#
#   load      The checkpoint stores exactly ONE vision tower (detector_model.vision_encoder.*,
#             538 tensors; ZERO under tracker_model.vision_encoder.*, read straight out of the
#             safetensors header here), and Sam3VideoModel is the load that keeps it that way.
#             The negative control is Sam3TrackerVideoModel, which loads a SECOND full copy of
#             that same tower — see the comment on it below, which corrects what the plan
#             predicted with what this probe measures.
#   seam      The towerless tracker (modeling_sam3_video.py:512 builds it with
#             remove_vision_encoder=True, so tracker_model.vision_encoder is None) is fed by
#             priming its vision-feature cache — the three-call sequence the library itself runs
#             at modeling_sam3_video.py:1614-1632. _prepare_vision_features
#             (modeling_sam3_tracker_video.py:1900) consults inference_session.cache BEFORE it
#             would ever dereference that None, so a forward that RETURNS is itself the proof
#             the cached branch was taken; the hit counter is the positive evidence beside it.
#   measure   Per-frame timings split into encoder and head, CUDA peak, host RSS, and the
#             fp16-vs-bf16 mask IoU. fp16 is off Meta's documented path for the video tracker
#             (every model-card video example is bfloat16) and no published fp16 video
#             validation exists, so on a pre-Ampere card — where main.py:92-106 picks fp16
#             because bf16 falls off the tensor-core path entirely — that diff is the gate.
#             Needs a directory of numbered frames; it will not invent one.
#
# Every signature below was read from the INSTALLED tree, transformers 5.13.0 / torch
# 2.6.0+cu124, which is NOT what requirements.txt pins (5.16.1 / 2.14.0). Confirm with:
#   tools/sam-service/env/Scripts/python.exe -c "import transformers,torch;print(transformers.__version__,torch.__version__)"
#
# Run (no argparse — three positional verbs, everything else an env var):
#   env/Scripts/python.exe track_smoke.py load
#   env/Scripts/python.exe track_smoke.py seam [frames-dir]
#   env/Scripts/python.exe track_smoke.py measure <frames-dir> <x> <y> [count]
# SAM3_DIR overrides the checkpoint, SAM_DEVICE the device, TRACK_SMOKE_OUT the output dir
# (default <temp>/ares-track-smoke — never the repo, so a run leaves the tree untouched).
#
# Exit code is the number of failed assertions, so a shell can gate on it.

import json
import os
import struct
import sys
import time

import numpy as np
import torch

# main.py's _resolve_sam3_dir() does this properly, and this is NOT a refactor of it: importing
# main.py starts its background loader at module scope (main.py:267), which would put a second
# 3.4 GB checkpoint on the card the moment this file is touched. So the probe resolves the
# checkpoint itself, in main.py's order, and stays importable by nothing.
def _resolve_sam3_dir() -> str:
    env = os.environ.get("SAM3_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for local in (os.path.join(here, "..", "..", "sam3"), os.path.join(here, "..", "..", "..", "sam3")):
        if os.path.isfile(os.path.join(local, "config.json")):
            return os.path.normpath(local)
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE") or (
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME")
        else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    snaps = os.path.join(hub, "models--facebook--sam3", "snapshots")
    if os.path.isdir(snaps):
        for rev in sorted(os.listdir(snaps)):
            snap = os.path.join(snaps, rev)
            if os.path.isfile(os.path.join(snap, "config.json")):
                return snap
    raise SystemExit("[track-smoke] no facebook/sam3 checkpoint found; set SAM3_DIR")


SAM3_DIR = _resolve_sam3_dir()
DEVICE = os.environ.get("SAM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.environ.get("TRACK_SMOKE_OUT") or os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMPDIR") or ".", "ares-track-smoke")

_fails: list = []


def _best_dtype(device: str) -> torch.dtype:
    """Same rule as main.py:92-106, restated here rather than imported (see _resolve_sam3_dir):
    bf16 tensor cores start at Ampere (sm_80); below that this repo measured 43.0 TFLOP/s fp16
    against 7.3 bf16 on a 2080 Ti, so fp16 is a ~6x win for byte-identical VRAM."""
    if device != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16


def _check(label: str, got, want) -> bool:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f"   expected {want!r}"))
    if not ok:
        _fails.append(label)
    return ok


def _rss_mb() -> float:
    # psutil is NOT in requirements.txt — it only arrives transitively through `accelerate`, which
    # that file lists as optional. RSS is instrumentation; it must never be the thing that fails a
    # probe whose contract is "exit code is the number of failed assertions", and both call sites
    # are the last statement of a verb, after every _check has already run.
    try:
        import psutil
    except ImportError:
        return 0.0
    return psutil.Process().memory_info().rss / (1 << 20)


def _cuda_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1 << 20) if DEVICE == "cuda" else 0.0


# ------------------------------------------------------------------ part 1: load ----


def _safetensors_header_counts() -> dict:
    """The verdict that chose this design was reached by reading model.safetensors' header, so
    the probe reproduces it on demand instead of trusting a comment. No torch, no load: just the
    u64 header length and the JSON tensor index at the front of the file."""
    path = os.path.join(SAM3_DIR, "model.safetensors")
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    keys = [k for k in header if k != "__metadata__"]
    return {
        "bytes": os.path.getsize(path),
        "tensors": len(keys),
        "detector_vision": sum(1 for k in keys if k.startswith("detector_model.vision_encoder.")),
        "tracker_vision": sum(1 for k in keys if k.startswith("tracker_model.vision_encoder.")),
        "tracker_neck": sum(1 for k in keys if k.startswith("tracker_neck.")),
    }


def _m(params) -> float:
    """Parameter count in millions, 2 dp — the resolution the VRAM arithmetic in
    docs/editor-v2-design.md 8.4 is quoted at, and stable enough to assert on."""
    return round(sum(p.numel() for p in params) / 1e6, 2)


def _tower_provenance(tracker_only) -> tuple:
    """How many of a standalone Sam3TrackerVideoModel's vision tensors are bit-identical to the
    checkpoint tensors the loader must have remapped them from: the trunk from
    `detector_model.vision_encoder.*`, the neck from `tracker_neck.*` (the tracker's FPN neck is
    stored top-level in this checkpoint because Sam3VideoModel holds it as its own module,
    modeling_sam3_video.py:541). Reads the safetensors file directly rather than a second model."""
    from safetensors import safe_open

    fh = safe_open(os.path.join(SAM3_DIR, "model.safetensors"), framework="pt")
    keys = set(fh.keys())
    sd = tracker_only.state_dict()
    trunk = neck = 0
    for k, v in sd.items():
        if not k.startswith("vision_encoder."):
            continue
        if k.startswith("vision_encoder.neck."):
            src, bump = "tracker_neck." + k.split("vision_encoder.neck.", 1)[1], "neck"
        else:
            src, bump = "detector_model." + k, "trunk"
        if src not in keys:
            continue
        t = fh.get_tensor(src)
        if t.shape == v.shape and torch.equal(t.float(), v.float()):
            if bump == "neck":
                neck += 1
            else:
                trunk += 1
    return trunk, neck


def probe_load() -> None:
    """Structural assertions only, so they run on CPU: none of them is device-dependent, and
    keeping the 3.4 GB negative control off the card means this verb still answers on a machine
    whose GPU is busy rendering."""
    from transformers import Sam3TrackerVideoModel, Sam3VideoModel

    print(f"[track-smoke] load probe — {SAM3_DIR}")
    h = _safetensors_header_counts()
    print(f"  checkpoint {h['bytes']:,} bytes")
    _check("safetensors tensors", h["tensors"], 1797)
    _check("detector_model.vision_encoder.* tensors", h["detector_vision"], 538)
    _check("tracker_model.vision_encoder.* tensors", h["tracker_vision"], 0)
    _check("tracker_neck.* tensors", h["tracker_neck"], 22)

    t0 = time.time()
    vm, info = Sam3VideoModel.from_pretrained(SAM3_DIR, dtype=torch.float32, output_loading_info=True)
    print(f"  Sam3VideoModel.from_pretrained on cpu in {time.time() - t0:.1f}s")
    _check("Sam3VideoModel missing_keys", len(info["missing_keys"]), 0)
    _check("Sam3VideoModel mismatched_keys", len(info["mismatched_keys"]), 0)
    _check("tracker_model.vision_encoder", vm.tracker_model.vision_encoder, None)
    _check("type(detector_model).__name__", type(vm.detector_model).__name__, "Sam3Model")
    _check("tracker_model.num_maskmem", vm.tracker_model.num_maskmem, 7)
    _check("tracker_model.config.max_cond_frame_num", vm.tracker_model.config.max_cond_frame_num, 4)
    _check("low_res_mask_size", vm.low_res_mask_size, 288)
    _check("tracker image_size", vm.tracker_model.image_size, 1008)
    _check("Sam3VideoModel parameters (M)", _m(vm.parameters()), 859.92)
    _check("detector_model parameters (M)", _m(vm.detector_model.parameters()), 840.38)
    _check("detector vision tower parameters (M)", _m(vm.detector_model.vision_encoder.parameters()), 454.04)
    _check("tracker_model parameters (M)", _m(vm.tracker_model.parameters()), 11.74)
    _check("tracker_neck parameters (M)", _m(vm.tracker_neck.parameters()), 7.80)

    del vm   # freed before the negative control so its ~1.9 GB is not double-counted

    # NEGATIVE CONTROL — and the plan this probe implements predicted the WRONG failure for it.
    # docs/sam-propagation-plan.md's verdict expected Sam3TrackerVideoModel.from_pretrained to
    # report 538 missing vision_encoder keys and silently random-init its whole tower, reasoning
    # from modeling_sam3_tracker_video.py:703 (base_model_prefix = "tracker_model") and :1602
    # (_keys_to_ignore_on_load_unexpected = [r"^detector_model."]). Measured on transformers
    # 5.13.0 it loads CLEAN: 845 tensors, 0 missing / 0 unexpected / 0 mismatched, because the
    # flexible cross-architecture loader remaps `detector_model.vision_encoder.*` onto
    # `vision_encoder.*` (516 trunk tensors) and `tracker_neck.*` onto `vision_encoder.neck.*`
    # (22 tensors), all of them bit-identical to the checkpoint.
    #
    # So the cost is not corruption, it is DUPLICATION: 454.04 M of this model's 465.78 M
    # parameters are a second copy of the tower the concept model already holds — ~908 MiB at
    # fp16, on a card main.py already fits inside 3,671 MB. THAT is what makes Sam3VideoModel
    # the right load, and it is what these assertions pin. The bit-identity checks are also the
    # regression trap for the plan's scenario: if a future transformers stops remapping and
    # random-inits the tower instead, they fail here rather than in a silently worse mask.
    # Kept on CPU and freed immediately.
    t0 = time.time()
    tm, tinfo = Sam3TrackerVideoModel.from_pretrained(SAM3_DIR, dtype=torch.float32, output_loading_info=True)
    print(f"  Sam3TrackerVideoModel.from_pretrained on cpu in {time.time() - t0:.1f}s (negative control)")
    _check("Sam3TrackerVideoModel missing_keys", len(tinfo["missing_keys"]), 0)
    _check("Sam3TrackerVideoModel unexpected_keys", len(tinfo["unexpected_keys"]), 0)
    _check("Sam3TrackerVideoModel parameters (M)", _m(tm.parameters()), 465.78)
    _check("duplicated vision tower parameters (M)", _m(tm.vision_encoder.parameters()), 454.04)
    trunk, neck = _tower_provenance(tm)
    _check("trunk tensors remapped from detector_model.vision_encoder.*", trunk, 516)
    _check("neck tensors remapped from tracker_neck.*", neck, 22)
    del tm


# ------------------------------------------------------------------ part 2: seam ----


def _synthetic_clip(n: int = 2, h: int = 270, w: int = 480) -> np.ndarray:
    """A disc translating across a gradient. The seam assertion is about WHICH BRANCH runs, not
    about mask quality, so a deterministic pair of frames keeps the verb runnable on a machine
    with no capture on it — unlike `measure`, which refuses to invent a corpus."""
    vid = np.zeros((n, h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    vid[:, :, :, 2] = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    for f in range(n):
        cx, cy, r = w // 3 + f * 12, h // 2, min(h, w) // 5
        vid[f][(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = (240, 240, 230)
    return vid


def _load_frames(frames_dir: str, count: int) -> tuple:
    """Numbered stills, sorted by the digits in the name rather than lexically, so f9 sorts
    before f10 whatever the zero-padding is."""
    import re

    from PIL import Image

    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    names = [n for n in os.listdir(frames_dir) if n.lower().endswith(exts)]
    if not names:
        raise SystemExit(f"[track-smoke] no images in {frames_dir}")

    def key(n):
        digits = re.findall(r"\d+", n)
        return (int(digits[-1]) if digits else 0, n)

    names.sort(key=key)
    names = names[:count]
    frames = [np.asarray(Image.open(os.path.join(frames_dir, n)).convert("RGB")) for n in names]
    return np.stack(frames), names


def _prime_and_count(vm, session, frame_idx: int) -> None:
    """The three-call vision-feature cache priming sequence, copied from the library's own
    per-frame path at modeling_sam3_video.py:1614-1632. The detector's tower runs ONCE per frame
    and its output feeds both the detector head and — through tracker_neck plus the mask
    decoder's conv_s0/conv_s1 projections — the towerless tracker. That single-encode property
    is why adding video tracking costs +19.5 M params rather than a second vision tower."""
    px = session.get_frame(frame_idx).unsqueeze(0)
    vision_embeds = vm.detector_model.get_vision_features(pixel_values=px)
    feats, pos = vm.get_vision_features_for_tracker(vision_embeds=vision_embeds)
    session.cache.cache_vision_features(
        frame_idx, {"vision_feats": feats, "vision_pos_embeds": pos})


def _count_cache_reads(session) -> dict:
    """Wrap the session cache's reader so the probe can say the cached branch was taken rather
    than infer it. A miss cannot be silent here — vision_encoder is None, so the else branch of
    _prepare_vision_features (modeling_sam3_tracker_video.py:1900-1918) would raise on the call
    into get_image_features — but counting is the positive evidence, not the absence of a crash."""
    tally = {"hit": 0, "miss": 0}
    inner = session.cache.get_vision_features

    def counted(frame_idx):
        got = inner(frame_idx)
        tally["hit" if got is not None else "miss"] += 1
        return got

    session.cache.get_vision_features = counted
    return tally


def probe_seam(frames_dir: str | None = None) -> None:
    from transformers import Sam3TrackerVideoProcessor, Sam3VideoModel

    dtype = _best_dtype(DEVICE)
    print(f"[track-smoke] seam probe — {DEVICE} / {str(dtype).replace('torch.', '')}")
    processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_DIR)
    _check("processor target_size", processor.target_size, 1008)

    t0 = time.time()
    vm = Sam3VideoModel.from_pretrained(SAM3_DIR, dtype=dtype).to(DEVICE).eval()
    print(f"  Sam3VideoModel loaded on {DEVICE} in {time.time() - t0:.1f}s")
    _check("tracker_model.vision_encoder", vm.tracker_model.vision_encoder, None)

    if frames_dir:
        video, names = _load_frames(frames_dir, 2)
        print(f"  frames {names}")
    else:
        video = _synthetic_clip(2)
        print("  frames: synthetic (a translating disc; the assertion is structural)")
    h, w = video.shape[1:3]

    session = processor.init_video_session(
        video=video, inference_device=DEVICE, inference_state_device=DEVICE, dtype=dtype)
    tally = _count_cache_reads(session)

    with torch.no_grad():
        _prime_and_count(vm, session, 0)
        # EVERY object of a frame in ONE call: add_inputs_to_inference_session ends with
        # `inference_session.obj_with_new_inputs = obj_ids` (processing_sam3_tracker_video.py:736
        # for points/boxes, :800 for masks) — an ASSIGNMENT, so a second call for the same frame
        # drops the first object's pending prompt on the floor.
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=[1],
            input_points=[[[[w * 0.36, h * 0.5]]]], input_labels=[[[1]]],
            original_size=(h, w))
        out0 = vm.tracker_model(inference_session=session, frame_idx=0)
        _check("cache misses through the seeded forward", tally["miss"], 0)
        _check("cache hits > 0", tally["hit"] > 0, True)
        _check("seed frame object_ids", list(out0.object_ids), [1])
        _check("seed frame is a conditioning frame",
               0 in session.output_dict_per_obj[0]["cond_frame_outputs"], True)

        # Frame 1 is the propagation step proper: prime, then forward with the memory bank live.
        before = dict(tally)
        _prime_and_count(vm, session, 1)
        out1 = vm.tracker_model(inference_session=session, frame_idx=1, reverse=False)
        _check("cache misses on the propagated frame", tally["miss"] - before["miss"], 0)

    print(f"  pred_masks {tuple(out1.pred_masks.shape)} {out1.pred_masks.dtype}, "
          f"object_score_logits {out1.object_score_logits.flatten().float().tolist()}")
    _check("pred_masks at low_res_mask_size", tuple(out1.pred_masks.shape[-2:]),
           (vm.low_res_mask_size, vm.low_res_mask_size))
    # post_process_masks indexes `masks[i]` per original_size and hands each entry straight to
    # F.interpolate, so element i must be the FULL 4-D (num_objects, 1, 288, 288) tensor
    # (image_processing_sam3.py:667-673). Passing the tensor itself makes masks[0] 3-D and
    # interpolate rejects it — worth pinning here because track.py will make this call per frame.
    masks = processor.post_process_masks([out1.pred_masks.float().cpu()], [[h, w]], binarize=True)[0]
    _check("post_process_masks restores frame size", tuple(masks.shape[-2:]), (h, w))
    if DEVICE == "cuda":
        print(f"  cuda peak {_cuda_peak_mb():.0f} MiB allocated, host rss {_rss_mb():.0f} MiB")


# --------------------------------------------------------------- part 3: measure ----


def _run_pass(vm, processor, video, seed_xy, dtype, out_dir, tag):
    """One full propagation over `video`, timed per frame with the encoder and head split. The
    split matters because it is the only thing the resolution tier control can trade: the
    encoder scales as trackRes^2, the head does not scale with it at all."""
    h, w = video.shape[1:3]
    n = video.shape[0]
    session = processor.init_video_session(
        video=video, inference_device=DEVICE, inference_state_device=DEVICE, dtype=dtype)
    tally = _count_cache_reads(session)
    enc_ms, head_ms, scores, areas, masks = [], [], [], [], []

    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # reset_peak_memory_stats leaves the peak sitting at the CURRENT allocation, so the weights
    # already on the card are inside every number below. Recording that baseline separately is
    # what lets the doc say how much of the peak is activations rather than weights.
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    weights_mb = torch.cuda.memory_allocated() / (1 << 20) if DEVICE == "cuda" else 0.0
    t_all = time.time()
    with torch.no_grad():
        for f in range(n):
            sync(); t0 = time.time()
            _prime_and_count(vm, session, f)
            sync(); t1 = time.time()
            if f == 0:
                processor.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=0, obj_ids=[1],
                    input_points=[[[list(seed_xy)]]], input_labels=[[[1]]], original_size=(h, w))
            out = vm.tracker_model(inference_session=session, frame_idx=f, reverse=False)
            sync(); t2 = time.time()
            enc_ms.append((t1 - t0) * 1000.0)
            head_ms.append((t2 - t1) * 1000.0)
            scores.append(float(out.object_score_logits.flatten()[0]))
            m = processor.post_process_masks([out.pred_masks.float().cpu()], [[h, w]], binarize=True)[0]
            m = m.numpy().reshape((-1, h, w))[0].astype(bool)
            areas.append(float(m.mean()))
            masks.append(m)
    total_s = time.time() - t_all

    from PIL import Image
    for f, m in enumerate(masks):
        Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{tag}-{f:05d}.png"))

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "device": DEVICE,
        "frames": n,
        "frameSize": [int(w), int(h)],
        "trackRes": int(processor.target_size),
        "totalS": round(total_s, 2),
        "msPerFrame": round(total_s * 1000.0 / max(1, n), 1),
        "encoderMsMedian": round(float(np.median(enc_ms)), 1),
        "headMsMedian": round(float(np.median(head_ms)), 1),
        # MAX, not p95: docs/targeted-temporal.md:27 is this repo's own rule that a percentile
        # hides exactly the outlier a per-frame budget has to survive.
        "encoderMsMax": round(float(np.max(enc_ms)), 1),
        "headMsMax": round(float(np.max(head_ms)), 1),
        "firstFrameMs": round(enc_ms[0] + head_ms[0], 1),
        "cudaWeightsMB": round(weights_mb, 1),
        "cudaPeakMB": round(_cuda_peak_mb(), 1),
        "cudaActivationsMB": round(_cuda_peak_mb() - weights_mb, 1),
        "cudaReservedPeakMB": round(
            torch.cuda.max_memory_reserved() / (1 << 20) if DEVICE == "cuda" else 0.0, 1),
        "rssMB": round(_rss_mb(), 1),
        "cacheHits": tally["hit"],
        "cacheMisses": tally["miss"],
        "objectScoreLogits": [round(s, 3) for s in scores],
        "maskAreaFraction": [round(a, 5) for a in areas],
    }, masks


def probe_measure(frames_dir: str, x: float, y: float, count: int):
    from transformers import Sam3TrackerVideoProcessor, Sam3VideoModel

    os.makedirs(OUT_DIR, exist_ok=True)
    video, names = _load_frames(frames_dir, count)
    n, h, w = video.shape[0], video.shape[1], video.shape[2]
    print(f"[track-smoke] measure — {n} frames {w}x{h} from {frames_dir}")
    print(f"  first {names[0]}  last {names[-1]}  seed click ({x:.0f}, {y:.0f})  out {OUT_DIR}")
    processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_DIR)

    # fp16 vs bf16 over the SAME frames and the SAME seed. The memory bank is stored bf16
    # regardless of session dtype (modeling_sam3_tracker_video.py:2701) and cast back at
    # consumption, so the only thing this diff can move is one elementwise store cast per frame
    # plus the encoder's own accumulation — which is exactly why it needs measuring rather than
    # arguing about.
    runs, stacks = {}, {}
    for dtype in (torch.float16, torch.bfloat16):
        tag = str(dtype).replace("torch.", "")
        if DEVICE != "cuda" and dtype is torch.bfloat16:
            print(f"  skipping {tag}: cpu run, no tensor-core path to compare")
            continue
        t0 = time.time()
        vm = Sam3VideoModel.from_pretrained(SAM3_DIR, dtype=dtype).to(DEVICE).eval()
        load_s = time.time() - t0
        print(f"  {tag}: loaded in {load_s:.1f}s, propagating {n} frames...")
        stats, masks = _run_pass(vm, processor, video, (x, y), dtype, OUT_DIR, tag)
        stats["loadS"] = round(load_s, 1)
        runs[tag], stacks[tag] = stats, masks
        print(f"  {tag}: {stats['msPerFrame']} ms/frame "
              f"(encoder {stats['encoderMsMedian']} median / {stats['encoderMsMax']} max, "
              f"head {stats['headMsMedian']} / {stats['headMsMax']}), "
              f"cuda peak {stats['cudaPeakMB']} MiB, rss {stats['rssMB']} MiB")
        _check(f"{tag} cache misses", stats["cacheMisses"], 0)
        del vm
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # fp16-vs-bf16 IoU, reported TWICE. The raw per-frame series answers "did the two runs end up
    # in the same place", which on a corpus the tracker cannot lock onto is a statement about the
    # corpus, not about the dtype: once the two runs' memory banks disagree at frame k they
    # disagree forever, and a wandering tracker disagrees with itself. The LOCKED series is the
    # dtype question proper — only frames where BOTH runs report a confident object score, so
    # the comparison is numerics rather than compounded drift.
    LOCK_LOGIT = 5.0
    iou, locked = [], []
    if len(stacks) == 2:
        sa = runs["float16"]["objectScoreLogits"]
        sb = runs["bfloat16"]["objectScoreLogits"]
        for f, (ma, mb) in enumerate(zip(stacks["float16"], stacks["bfloat16"])):
            inter = np.count_nonzero(ma & mb)
            union = np.count_nonzero(ma | mb)
            v = 1.0 if union == 0 else inter / union     # both empty = agreement, not 0/0
            iou.append(v)
            if sa[f] > LOCK_LOGIT and sb[f] > LOCK_LOGIT:
                locked.append(v)
        print(f"  fp16 vs bf16 IoU, all {len(iou)} frames: min {min(iou):.4f}  "
              f"median {float(np.median(iou)):.4f}  mean {float(np.mean(iou)):.4f}")
        if locked:
            print(f"  fp16 vs bf16 IoU, {len(locked)} frames locked above logit {LOCK_LOGIT}: "
                  f"min {min(locked):.4f}  median {float(np.median(locked)):.4f}")
        else:
            print(f"  NO frame was locked above logit {LOCK_LOGIT} in both runs — this corpus "
                  f"does not gate fp16; re-run over proxy renders")

    report = {
        "framesDir": frames_dir,
        "frameNames": [names[0], names[-1]],
        "seed": [x, y],
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "computeCapability": list(torch.cuda.get_device_capability(0)) if DEVICE == "cuda" else None,
        "runs": runs,
        "fp16VsBf16Iou": {
            "perFrame": [round(v, 4) for v in iou],
            "min": round(min(iou), 4) if iou else None,
            "median": round(float(np.median(iou)), 4) if iou else None,
            "lockLogit": LOCK_LOGIT,
            "lockedFrames": len(locked),
            "lockedMin": round(min(locked), 4) if locked else None,
            "lockedMedian": round(float(np.median(locked)), 4) if locked else None,
        },
    }
    path = os.path.join(OUT_DIR, "timings.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"  wrote {path} and {sum(len(s) for s in stacks.values())} mask PNGs")


# ------------------------------------------------------------------------ main ----

USAGE = """usage: track_smoke.py <verb>
  load                                 checkpoint shape, 0-missing-keys load, negative control (cpu)
  seam [frames-dir]                    prime the vision-feature cache and drive the towerless tracker
  measure <frames-dir> <x> <y> [count] timings, peak memory, fp16-vs-bf16 IoU over real frames
"""

if __name__ == "__main__":
    verb = sys.argv[1] if len(sys.argv) > 1 else ""
    if verb == "load":
        probe_load()
    elif verb == "seam":
        probe_seam(sys.argv[2] if len(sys.argv) > 2 else None)
    elif verb == "measure":
        if len(sys.argv) < 5:
            raise SystemExit(USAGE)
        probe_measure(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]),
                      int(sys.argv[5]) if len(sys.argv) > 5 else 24)
    else:
        raise SystemExit(USAGE)
    print(f"[track-smoke] {verb}: {'OK' if not _fails else str(len(_fails)) + ' FAILED: ' + ', '.join(_fails)}")
    sys.exit(len(_fails))
