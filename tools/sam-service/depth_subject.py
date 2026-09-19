# Subject mask for the ARES depth engine: "mask before depth" (docs/rgbd-rebuild-pipeline.md, the
# law in stage 1). A text prompt names the subject; every frame of the clip gets a binary mask; the
# depth model then sees the subject on black and spends its whole relative range on it.
#
# HOW THE MASK IS MADE. SAM 3 has two halves resident in this service and both are used, off ONE
# vision-encoder pass per frame:
#   detector (Sam3Model, main.py's concept model)  text -> instance masks on a single image
#   tracker  (Sam3VideoModel.tracker_model, track.py's lazy load, detector shared by graft)
#            a mask seeded on one frame -> the same object on the following frames, through its
#            memory bank, which is what makes the masks temporally stable
# The text detector seeds the tracker on the first frame where the prompt scores, the tracker
# carries each instance forward, and the detector runs again every REDETECT frames (and on every
# frame while nothing is tracked) to pick up instances that entered later or were lost. The encoder
# is ~2/3 of a frame's cost and is shared, so re-detection costs the DETR head only.
# Per-frame text segmentation alone was the alternative: same encoder cost per frame, and each
# frame's instance set is thresholded independently, so a subject that dips under the score
# threshold for one frame turns that frame entirely black.
#
# STREAMING. The library's session keeps every tracked frame's memory (maskmem features + position
# encoding + masks, 3.36 MiB per frame per object, track.py:64-67): 18.5 GiB for the 5,627-frame
# acceptance clip. The tracker READS only the last num_maskmem-1 = 6 frames of memory and the last
# max_object_pointers-1 = 15 object pointers (modeling_sam3_tracker_video.py:2225-2366), plus the
# conditioning frames. So everything non-conditioning older than KEEP frames is deleted after each
# forward, which is lossless for forward propagation and holds the session at a constant size.
#
# INPUT. Frames arrive as 1008x1008 rgb24 straight from ffmpeg's area scaler: the checkpoint's
# processor squashes to that square anyway (processor_config.json: size 1008x1008, mean = std = 0.5,
# rescale 1/255), so doing it in the decoder removes PIL and a CPU resize from the per-frame path.
# The frame SET is the depth pass's exactly: same `fps=` filter, same `-frames:v`.
#
# This module never imports main. It reaches SAM 3 through track.py, which main.py has already
# configured with the checkpoint dir, dtype, GPU lock and the late-bound concept-model getter.

import os
import sys

REDETECT = int(os.environ.get("DEPTH_SUBJECT_REDETECT", "24"))
MAX_OBJECTS = int(os.environ.get("DEPTH_SUBJECT_MAX_OBJECTS", "4"))
SCORE_THRESHOLD = float(os.environ.get("DEPTH_SUBJECT_SCORE", "0.5"))
KEEP = int(os.environ.get("DEPTH_SUBJECT_KEEP", "24"))        # > max_object_pointers (16)
# A detected instance is NEW when less than this fraction of it lies inside what is already tracked.
NEW_OVERLAP = 0.5
# Mean coverage under this over the whole clip is "the prompt matched nothing": 2e-4 of a 518x294
# map is 30 pixels.
MIN_COVERAGE = float(os.environ.get("DEPTH_SUBJECT_MIN_COVERAGE", "0.0002"))
SAM_EDGE = 1008
ENGINE = "sam3-text-tracker"


def _track_module():
    """track.py as main.py configured it, or None. Read from sys.modules FIRST: under uvicorn it is
    already imported and configured, and a fresh import would be an unconfigured twin."""
    mod = sys.modules.get("track")
    if mod is None:
        try:
            import track as mod      # noqa: PLC0415  (standalone harness)
        except Exception:
            return None
    return mod


def sam3_dir() -> str:
    mod = _track_module()
    return getattr(mod, "SAM3_DIR", "") if mod is not None else ""


def sam3_weights_present() -> bool:
    """Filesystem only. main.py's resolver hands back the bare repo id "facebook/sam3" when no
    local copy exists, which is not a directory, so that reads as absent here."""
    d = sam3_dir()
    if not d or not os.path.isfile(os.path.join(d, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(d, f))
               for f in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"))


class SubjectUnavailable(RuntimeError):
    """SAM 3 cannot serve a subject mask. `missing` is the component list for the caller."""

    def __init__(self, message: str, missing: list[str] | None = None):
        super().__init__(message)
        self.missing = missing or []


class SubjectMasker:
    """One clip's mask pass. step() is called once per frame, in order, with the model lock held by
    the caller only for the duration of that call."""

    def __init__(self, prompt: str, out_h: int, out_w: int, log):
        self.prompt = prompt
        self.out_h, self.out_w = int(out_h), int(out_w)
        self.log = log
        self.vm = None
        self.concept = None
        self.isess = None
        self.text = None
        self.text_mask = None
        self.next_obj = 1
        self.detections = 0
        self.reseeds = 0
        self.encoder_s = 0.0
        self.detect_s = 0.0
        self.track_s = 0.0

    # ------------------------------------------------------------------ load ----

    def prewarm(self) -> None:
        """Unlocked waits (transformers import, main.py's loader). No allocation."""
        trk = _track_module()
        if trk is None or not sam3_weights_present():
            raise SubjectUnavailable("SAM 3 weights are not installed", ["sam3"])
        try:
            trk._prewarm()
        except Exception as e:                       # HTTPException(503): still loading
            raise RuntimeError(str(getattr(e, "detail", e)))

    def load(self) -> None:
        """Call with the model lock HELD: the tracker load allocates."""
        import torch

        trk = _track_module()
        concept = trk._get_concept_model()
        if concept is None:
            raise SubjectUnavailable("SAM 3 text segmentation is not loaded in this service", ["sam3"])
        try:
            vm, processor = trk._ensure_tracker_video()
        except Exception as e:
            raise RuntimeError(str(getattr(e, "detail", e)))
        if vm.detector_model is not concept:
            # No graft happened (shapes differed): the video model carries its own detector, and
            # that one is then the module whose vision features feed its tracker.
            concept = vm.detector_model
        self.trk, self.vm, self.concept, self.processor = trk, vm, concept, processor
        self.device, self.dtype = trk.DEVICE, trk.SAM_DTYPE
        if int(processor.target_size) != SAM_EDGE:
            raise RuntimeError(f"tracker input edge is {processor.target_size}, expected {SAM_EDGE}")

        from transformers import Sam3Processor
        tok = Sam3Processor.from_pretrained(trk.SAM3_DIR)
        t = tok(text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.text = concept.get_text_features(input_ids=t["input_ids"],
                                                  attention_mask=t.get("attention_mask"),
                                                  return_dict=True)
        self.text_mask = t.get("attention_mask")
        self._post = tok
        self._new_session()

    def _new_session(self) -> None:
        self.isess = self.processor.init_video_session(
            video=None, inference_device=self.device, inference_state_device=self.device,
            video_storage_device=self.device, dtype=self.dtype)
        self.isess.video_height, self.isess.video_width = self.out_h, self.out_w

    def close(self) -> None:
        import gc

        import torch
        self.isess = None
        self.text = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ frame ----

    def _detect(self, vision_embeds):
        """Instance masks for the prompt on this frame: (n, 1008, 1008) bool, best score first."""
        import torch

        out = self.concept(vision_embeds=vision_embeds, text_embeds=self.text, attention_mask=self.text_mask)
        res = self._post.post_process_instance_segmentation(
            out, threshold=SCORE_THRESHOLD, target_sizes=[(SAM_EDGE, SAM_EDGE)])[0]
        masks, scores = res["masks"], res["scores"]
        if masks.shape[0] == 0:
            return masks.bool()
        order = torch.argsort(scores.float(), descending=True)
        return masks[order].bool()

    def _forward(self, idx: int, px):
        out = self.vm.tracker_model(inference_session=self.isess, frame_idx=idx, frame=px, reverse=False)
        if self.isess.processed_frames is not None:
            self.isess.processed_frames.pop(idx, None)
        return out

    def _prune(self, idx: int) -> None:
        old = idx - KEEP
        for obj_idx, store in self.isess.output_dict_per_obj.items():
            non_cond = store["non_cond_frame_outputs"]
            for f in [f for f in non_cond if f < old]:
                del non_cond[f]
            tracked = self.isess.frames_tracked_per_obj[obj_idx]
            for f in [f for f in tracked if f < old]:
                del tracked[f]
            self.isess.mask_inputs_per_obj[obj_idx].pop(idx, None)

    def step(self, idx: int, frame_u8):
        """frame_u8: (1008, 1008, 3) uint8 numpy. Returns (out_h, out_w) uint8 numpy of 0/255."""
        import time

        import torch
        import torch.nn.functional as F

        dev = self.device
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
        with torch.no_grad():
            t0 = time.time()
            px = torch.from_numpy(frame_u8).to(dev).permute(2, 0, 1).unsqueeze(0)
            px = px.float().div_(127.5).sub_(1.0).to(self.dtype)          # (x/255 - 0.5) / 0.5
            vision = self.concept.get_vision_features(pixel_values=px)
            feats, pos = self.vm.get_vision_features_for_tracker(vision_embeds=vision)
            self.isess.cache.cache_vision_features(idx, {"vision_feats": feats, "vision_pos_embeds": pos})
            sync()
            t1 = time.time()
            self.encoder_s += t1 - t0

            n_obj = self.isess.get_obj_num()
            union = None
            if n_obj:
                out = self._forward(idx, px)
                union = self._union(out)
                sync()
            t2 = time.time()
            self.track_s += t2 - t1

            lost = union is None or not bool(union.any())
            if lost or idx % REDETECT == 0:
                inst = self._detect(vision)
                self.detections += 1
                if lost and n_obj and inst.shape[0]:
                    # Every tracked object is gone and the prompt is visible again. The library has
                    # no per-object removal, so the session restarts from this frame.
                    self._new_session()
                    self.isess.cache.cache_vision_features(
                        idx, {"vision_feats": feats, "vision_pos_embeds": pos})
                    n_obj = 0
                    self.reseeds += 1
                new = []
                if inst.shape[0]:
                    big = None
                    if union is not None and not lost:
                        big = F.interpolate(union[None, None].float(), size=(SAM_EDGE, SAM_EDGE),
                                            mode="nearest")[0, 0] > 0.5
                    for m in inst:
                        if n_obj + len(new) >= MAX_OBJECTS:
                            break
                        area = int(m.sum())
                        if area == 0:
                            continue
                        if big is not None and int((m & big).sum()) / area >= NEW_OVERLAP:
                            continue
                        new.append(m)
                if new:
                    ids = list(range(self.next_obj, self.next_obj + len(new)))
                    self.next_obj += len(new)
                    self.processor.add_inputs_to_inference_session(
                        inference_session=self.isess, frame_idx=idx, obj_ids=ids, input_masks=new)
                    out = self._forward(idx, px)
                    union = self._union(out)
                    if n_obj == 0:
                        self.log(f"subject '{self.prompt}': {len(new)} instance"
                                 f"{'s' if len(new) != 1 else ''} seeded at frame {idx}")
                sync()
                self.detect_s += time.time() - t2
            self._prune(idx)
            if union is None:
                return None
            return (union.to(torch.uint8) * 255).cpu().numpy()

    def _union(self, out):
        """Union of every object the tracker scores as present, at the depth map's size. Bilinear
        on the LOGITS and thresholded after, as post_process_masks does."""
        import torch.nn.functional as F

        low = out.pred_masks.float()
        if low.ndim == 3:
            low = low.unsqueeze(1)
        present = out.object_score_logits.float().reshape(-1) > 0
        if not bool(present.any()):
            return None
        m = F.interpolate(low[present], size=(self.out_h, self.out_w), mode="bilinear",
                          align_corners=False) > 0
        return m.any(dim=0)[0]
