# Video-Depth-Anything for the ARES depth engine: the third model family of depth.py.
#
# THE INSTALL CONTRACT (the installer catalog is written against it):
#   code         <ares>/tools/ext/video-depth-anything        git clone of DepthAnything/Video-Depth-Anything
#   checkpoints  <ares>/tools/sam-service/models/video_depth_anything_{vits,vitb,vitl}.pth
# Component ids reported to the caller when something is absent: vda-code, vda-small, vda-base,
# vda-large. Nothing here downloads anything; a missing component is a 409 on /depth/run.
#
# WHAT THE NETWORK EMITS. relu(affine-invariant INVERSE depth): larger = nearer, arbitrary scale and
# shift, the same convention as Depth-Anything-V2's relative checkpoints. Upstream's own evaluation
# fits the prediction to 1/gt (benchmark/eval/eval.py:88-94), which is what settles it. So the
# manifest `kind` is "relative-disparity".
#
# THE CLONE IS NEVER EDITED AND NEVER PUT ON sys.path. Its model class lives in
# video_depth_anything/video_depth.py, which imports cv2 and a top-level package called `utils` at
# module scope, for its own whole-clip inference helper. Neither is wanted in this process: cv2 is
# not a dependency of the service, and a bare `utils` on sys.path shadows anything else of that name
# for the life of the process. So the two upstream building blocks that carry every weight,
# DINOv2 and DPTHeadTemporal, are imported through an explicitly-pathed package spec and composed
# here under the SAME attribute names (`pretrained`, `head`); load_state_dict(strict=True) is the
# proof the composition is the upstream network, tensor for tensor. forward() below is upstream's
# five lines (video_depth.py:61-68) plus one change: the per-frame encoder runs in chunks, since
# only the head's temporal modules need all 32 frames at once.
#
# xformers IS OPTIONAL. Upstream guards every xformers import with try/except. Its fallback DINOv2
# attention materializes the (tokens x tokens) matrix per head; at import time that one method is
# replaced by torch's scaled_dot_product_attention (same maths, the memory-efficient kernel on
# sm_75). easydict, a 30-line attribute-dict upstream uses to pass six kwargs, is stood in for by a
# dict subclass when it is not installed.
#
# PRECISION. Upstream's recipe exactly: fp32 weights, forward under torch.autocast. DINOv2 in PURE
# fp16 overflows in its later blocks; autocast keeps LayerNorm and softmax in fp32, and upstream's
# head already runs its last conv in fp32 by hand (dpt_temporal.py:104-105).
#
# STREAMING. Upstream's infer_video_depth takes every frame of the clip, keeps every window's depth
# in a list and aligns at the end: O(clip) memory, 12 GB of float32 for the 5,627-frame acceptance
# clip at 1080p. Upstream also ships video_depth_stream.py, a one-frame-at-a-time mode that reuses
# cached temporal hidden states; its README calls it experimental and reports d1 on ScanNet falling
# from 0.926 to 0.836 because the model was never trained that way. So this file keeps upstream's
# OFFLINE scheme and only changes when things are held:
#   windows of INFER_LEN=32 inputs, stepping 22 frames; slots 0-9 of every window after the first
#   are the previous window's INPUTS at KEYFRAMES=[0,12,24..31] (so slot 0 is always clip frame 0,
#   slot 1 a keyframe 10 frames before the overlap, slots 2-9 the 8 overlapping frames);
#   a least-squares scale+shift maps the window's slots 0-1 onto the already-aligned depth of those
#   same two images; slots 2-9 are cross-faded linearly into the previous window's last 8 frames;
#   slots 10-31 are appended.
# Resident state is one window of inputs, the previous window's inputs, 8 held-back depth frames and
# 2 reference maps, whatever the clip length. Output is identical to upstream's (the test in
# depth.py's verification compared them), except the alignment sums run in float64.

import importlib
import importlib.machinery
import importlib.util
import os
import sys

INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8
STEP = INFER_LEN - OVERLAP              # 22 new frames per window
ALIGN_LEN = OVERLAP - INTERP_LEN        # 2 reference slots

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARES_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
CODE_DIR = os.environ.get("VDA_CODE_DIR") or os.path.join(_ARES_ROOT, "tools", "ext", "video-depth-anything")
MODELS_DIR = os.environ.get("VDA_MODELS_DIR") or os.path.join(_HERE, "models")
ENCODER_CHUNK = int(os.environ.get("VDA_ENCODER_CHUNK", "8"))

# key -> (component id, checkpoint file, Hub repo, constructor kwargs from upstream run.py:45-49)
VDA_MODELS: dict[str, dict] = {
    "video-small": {"component": "vda-small", "file": "video_depth_anything_vits.pth",
                    "repo": "depth-anything/Video-Depth-Anything-Small",
                    "cfg": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}},
    "video-base":  {"component": "vda-base", "file": "video_depth_anything_vitb.pth",
                    "repo": "depth-anything/Video-Depth-Anything-Base",
                    "cfg": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]}},
    "video-large": {"component": "vda-large", "file": "video_depth_anything_vitl.pth",
                    "repo": "depth-anything/Video-Depth-Anything-Large",
                    "cfg": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}},
}
_INTERMEDIATE = {"vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}
_PKG = "video_depth_anything"


def code_present() -> bool:
    pkg = os.path.join(CODE_DIR, _PKG)
    return all(os.path.isfile(os.path.join(pkg, f)) for f in ("dinov2.py", "dpt_temporal.py", "dpt.py"))


def checkpoint_path(key: str) -> str:
    return os.path.join(MODELS_DIR, VDA_MODELS[key]["file"])


def checkpoint_present(key: str) -> bool:
    """Larger than 1 MiB, so the HTML error page or the empty file an interrupted download leaves
    behind does not read as an installed checkpoint (the smallest real one is 116 MB)."""
    try:
        return os.path.getsize(checkpoint_path(key)) > (1 << 20)
    except OSError:
        return False


def missing(key: str) -> list[str]:
    """Component ids absent for this key, code first. Empty means the key can run."""
    out = []
    if not code_present():
        out.append("vda-code")
    if not checkpoint_present(key):
        out.append(VDA_MODELS[key]["component"])
    return out


def installed() -> dict:
    """For /depth/health: a filesystem check, loads nothing."""
    return {"vda-code": code_present(),
            **{m["component"]: checkpoint_present(k) for k, m in VDA_MODELS.items()}}


# ------------------------------------------------------------------- import ----


def _import_upstream():
    """(DINOv2, DPTHeadTemporal) from the clone, without touching sys.path.

    video_depth_anything/ has no __init__.py, so a package module is registered by hand with its
    search location pinned to the clone; the submodules' relative imports then resolve normally."""
    if _PKG not in sys.modules or not getattr(sys.modules[_PKG], "__ares_vda__", False):
        spec = importlib.machinery.ModuleSpec(_PKG, None, is_package=True)
        spec.submodule_search_locations = [os.path.join(CODE_DIR, _PKG)]
        pkg = importlib.util.module_from_spec(spec)
        pkg.__ares_vda__ = True
        sys.modules[_PKG] = pkg
    try:
        import easydict  # noqa: F401
    except ImportError:
        import types

        class EasyDict(dict):
            """Upstream only ever does EasyDict(k=v, ...) and **unpacks it (dpt_temporal.py:36-52)."""
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        stub = types.ModuleType("easydict")
        stub.EasyDict = EasyDict
        sys.modules["easydict"] = stub
    dinov2 = importlib.import_module(_PKG + ".dinov2")
    head = importlib.import_module(_PKG + ".dpt_temporal")
    _patch_attention(importlib.import_module(_PKG + ".dinov2_layers.attention"))
    return dinov2.DINOv2, head.DPTHeadTemporal


def _patch_attention(attn_mod) -> None:
    """Replace the no-xformers DINOv2 attention with SDPA. Applied to the imported class, never to
    the file. With xformers installed upstream's own path is left alone."""
    if getattr(attn_mod, "XFORMERS_AVAILABLE", False) or getattr(attn_mod, "__ares_sdpa__", False):
        return
    import torch.nn.functional as F

    def forward(self, x, attn_bias=None):
        assert attn_bias is None, "nested-tensor attention needs xformers"
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)      # default scale is head_dim ** -0.5
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    attn_mod.MemEffAttention.forward = forward
    attn_mod.__ares_sdpa__ = True


# -------------------------------------------------------------------- model ----


def build(key: str):
    """The upstream network on the CPU, weights loaded strictly. The caller moves it."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    DINOv2, DPTHeadTemporal = _import_upstream()
    cfg = VDA_MODELS[key]["cfg"]

    class VideoDepthAnything(nn.Module):
        def __init__(self, encoder, features, out_channels):
            super().__init__()
            self.encoder = encoder
            self.pretrained = DINOv2(model_name=encoder)
            self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, False,
                                        out_channels=out_channels, use_clstoken=False,
                                        num_frames=INFER_LEN, pe="ape")

        def forward(self, x):
            """x: (T, 3, H, W) normalized. Returns (T, H, W). Upstream video_depth.py:61-68 with
            B = 1 and the encoder chunked; the head sees all T frames in one call."""
            T, _, H, W = x.shape
            layers = _INTERMEDIATE[self.encoder]
            parts = [self.pretrained.get_intermediate_layers(x[i:i + ENCODER_CHUNK], layers,
                                                             return_class_token=True)
                     for i in range(0, T, ENCODER_CHUNK)]
            feats = tuple((torch.cat([p[l][0] for p in parts]), torch.cat([p[l][1] for p in parts]))
                          for l in range(len(layers)))
            del parts
            depth = self.head(feats, H // 14, W // 14, T)[0]
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
            return F.relu(depth).squeeze(1)

    model = VideoDepthAnything(**cfg)
    state = torch.load(checkpoint_path(key), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


# ----------------------------------------------------------------- alignment ----


class WindowAligner:
    """Upstream's post-hoc window stitching (video_depth.py:120-160), run as the windows arrive.

    add(depth) takes one window's (32, H, W) float32 output and returns the frames that are now
    FINAL; the last 8 aligned frames are held back because the next window cross-fades into them.
    flush() returns those 8 at the end of the clip. The caller truncates to the real frame count:
    like upstream, the last window is padded with copies of the last frame."""

    def __init__(self):
        self.ref = None      # [depth of clip frame 0, aligned depth of the latest keyframe slot 12]
        self.tail = None     # (8, H, W): aligned, not yet emitted
        self.scale = 1.0
        self.shift = 0.0

    def add(self, depth):
        import torch

        if self.ref is None:
            self.ref = [depth[KEYFRAMES[0]].clone(), depth[KEYFRAMES[1]].clone()]
            self.tail = depth[INFER_LEN - INTERP_LEN:].clone()
            return depth[:INFER_LEN - INTERP_LEN]
        # Least squares for (scale, shift) over every pixel of the two reference slots: upstream's
        # compute_scale_and_shift_full with an all-ones mask, in float64 because the sums run over
        # ~3e5 float32 values each.
        p = depth[:ALIGN_LEN].double().reshape(-1)
        t = torch.stack(self.ref).double().reshape(-1)
        a00, a01, a11 = (p * p).sum(), p.sum(), float(p.numel())
        b0, b1 = (p * t).sum(), t.sum()
        det = a00 * a11 - a01 * a01
        if float(det) != 0.0:
            self.scale = float((a11 * b0 - a01 * b1) / det)
            self.shift = float((-a01 * b0 + a00 * b1) / det)
        else:
            self.scale, self.shift = 1.0, 0.0
        aligned = (depth * self.scale + self.shift).clamp_(min=0)
        w = torch.linspace(0.0, 1.0, INTERP_LEN, device=depth.device).view(-1, 1, 1)
        blended = self.tail * (1.0 - w) + aligned[ALIGN_LEN:OVERLAP] * w
        rest = aligned[OVERLAP:]
        self.ref = [self.ref[0], aligned[KEYFRAMES[1]].clone()]
        self.tail = rest[-INTERP_LEN:].clone()
        return torch.cat([blended, rest[:-INTERP_LEN]])

    def flush(self):
        tail, self.tail = self.tail, None
        return tail
