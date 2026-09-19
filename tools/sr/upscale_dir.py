#!/usr/bin/env python
"""
Batch super-resolution over a folder of atlas frames.

Why this exists as a worker instead of one `realesrgan-ncnn-vulkan.exe` call per frame:
the per-frame process paid a full Vulkan init + 33 MB weight load for every atlas, which
dominated the run. Measured on 2 x RTX 2080 Ti over a 2048 sq atlas -> 8192 sq:

    ncnn x4plus, one exe per frame (the old path)   21908 ms/frame
    ncnn x4plus, directory mode, both GPUs           9974 ms/frame
    CUDA x4plus fp16 here                           12300 ms/frame
    CUDA realesr-general-x4v3 fp16 here               674 ms/frame

Loading is amortized, the GPUs are fed in parallel (one worker thread and one model replica
per device), and the resample + strength blend happen on the GPU, so the 56 MB intermediate
PNG and the second ffmpeg decode both disappear.

Weights load through spandrel, which infers the architecture from the checkpoint. Any
architecture spandrel knows therefore works with no code change here - ESRGAN/RRDBNet,
RealESRGAN Compact, SPAN, DAT, HAT, SwinIR, RealPLKSR and so on - which is what makes
"drop in a newer model" a catalog entry rather than a patch.

Progress is one line per frame on stdout so the caller can stream it:

    LOAD <arch> x<scale> <device> <ms>
    PROGRESS <done> <total> <filename> <ms>
    DONE <count> <ms>
    ERROR <message>
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image

import warnings
# PIL hands back a read-only array; we always copy before writing, so the warning is noise that
# would otherwise print two lines of traceback into the user-facing progress log.
warnings.filterwarnings("ignore", message=".*non-writable tensors.*")

Image.MAX_IMAGE_PIXELS = None          # atlases are far past PIL's decompression-bomb default

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [
    os.path.join(HERE, "..", "sam-service", "models"),
    os.path.join(HERE, "models"),
]


def emit(*parts):
    """One record per line, flushed - the caller reads this as a stream."""
    print(" ".join(str(p) for p in parts), flush=True)


def resolve_ckpt(name):
    """A bare filename resolves against the model folders; a path is taken as given."""
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    for d in MODEL_DIRS:
        p = os.path.normpath(os.path.join(d, name))
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"checkpoint not found: {name} (looked in {', '.join(MODEL_DIRS)})")


def load_model(ckpt, device):
    from spandrel import ModelLoader
    t0 = time.time()
    d = ModelLoader().load_from_file(ckpt)
    arch, scale = d.architecture.name, d.scale
    model = d.to(device).eval()
    return model, arch, scale, (time.time() - t0) * 1000


def sr_tiled(model, t, scale, tile, pad):
    """Padded-tile inference. Each output tile keeps only its centre; the pad supplies the
    context that makes the seams invisible. Peak VRAM is one tile, not one atlas."""
    _, _, h, w = t.shape
    out = torch.empty((3, h * scale, w * scale), dtype=torch.float32, device=t.device)
    for ty in range(0, h, tile):
        for tx in range(0, w, tile):
            y0, x0 = max(0, ty - pad), max(0, tx - pad)
            y1, x1 = min(h, ty + tile + pad), min(w, tx + tile + pad)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=t.device.type == "cuda"):
                sr = model(t[:, :, y0:y1, x0:x1])
            sr = sr.float()
            ky1, kx1 = min(h, ty + tile), min(w, tx + tile)
            oy, ox = (ty - y0) * scale, (tx - x0) * scale
            kh, kw = (ky1 - ty) * scale, (kx1 - tx) * scale
            out[:, ty * scale:ky1 * scale, tx * scale:kx1 * scale] = sr[0, :, oy:oy + kh, ox:ox + kw]
    return out


def process(model, scale, src_path, dst_path, args, device):
    im = Image.open(src_path)
    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    alpha = np.asarray(im.convert("RGBA"))[:, :, 3] if has_alpha else None
    img = np.asarray(im.convert("RGB"))
    h, w, _ = img.shape
    tw, th = int(round(w * args.scale)), int(round(h * args.scale))

    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    t = t.to(device, non_blocking=True).float().div_(255)
    out = sr_tiled(model, t, scale, args.tile, args.pad)

    # The net runs at its NATIVE ratio; the requested ratio is a resample of that result.
    # (Asking a fixed-4x net for a different ratio is what shredded atlases into displaced
    # tiles before - the ratio is never handed to the model.)
    if out.shape[-2:] != (th, tw):
        out = torch.nn.functional.interpolate(out.unsqueeze(0), size=(th, tw),
                                              mode="bicubic", align_corners=False,
                                              antialias=True).squeeze(0)
    # Strength dial: out = resampled_source * (1 - S) + sr * S, on the GPU.
    if args.strength < 0.999:
        base = torch.nn.functional.interpolate(t, size=(th, tw), mode="bicubic",
                                               align_corners=False, antialias=True).squeeze(0)
        out = base.mul_(1.0 - args.strength).add_(out.mul_(args.strength))

    arr = out.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    del t, out

    # A silently-black frame is the failure mode that matters here: the old ncnn path emitted
    # them on VRAM pressure with a zero exit code, and they reached the encoder unnoticed.
    if int(arr.max()) - int(arr.min()) < 2:
        raise RuntimeError(f"upscaler returned a blank frame for {os.path.basename(src_path)}")

    if alpha is not None:
        a = Image.fromarray(alpha).resize((tw, th), Image.LANCZOS)
        Image.merge("RGBA", (*Image.fromarray(arr).split(), a)).save(dst_path, "PNG")
    else:
        Image.fromarray(arr).save(dst_path, "PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="realesr-general-x4v3.pth")
    ap.add_argument("--scale", type=float, default=2.0, help="OUTPUT ratio; resampled from the net's native ratio")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--devices", default="", help="comma list, e.g. cuda:0,cuda:1; default = every CUDA device")
    ap.add_argument("--pattern", default="atlas-")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith(".png") and f.startswith(args.pattern))
    if args.max_frames > 0:
        names = names[:args.max_frames]
    if not names:
        emit("ERROR", f"no {args.pattern}*.png frames in {args.src}")
        return 2
    os.makedirs(args.out, exist_ok=True)

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    elif torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        # Falling back to CPU is a ~30x slowdown. Silently taking it is how a "2 s/frame" tier
        # turns into 22 s/frame with nothing in the log to explain it, so say it plainly.
        devices = ["cpu"]
        emit("WARN", "no CUDA device visible to torch - running on the CPU, which is roughly "
                     "30x slower per frame. Check the Python environment's PyTorch build.")
    emit("LOG", "devices:", ",".join(devices))

    ckpt = resolve_ckpt(args.model)
    work = queue.Queue()
    for i, n in enumerate(names):
        work.put((i, n))

    lock = threading.Lock()
    state = {"done": 0, "failed": None}
    t_all = time.time()

    def worker(device):
        try:
            model, arch, scale, load_ms = load_model(ckpt, device)
        except Exception as e:                                   # noqa: BLE001 - reported, not raised
            with lock:
                state["failed"] = state["failed"] or f"{device}: {type(e).__name__}: {e}"
            return
        with lock:
            emit("LOAD", arch.replace(" ", "_"), f"x{scale}", device, f"{load_ms:.0f}")
        while state["failed"] is None:
            try:
                _, name = work.get_nowait()
            except queue.Empty:
                break
            t0 = time.time()
            try:
                process(model, scale, os.path.join(args.src, name), os.path.join(args.out, name), args, device)
            except Exception as e:                               # noqa: BLE001
                with lock:
                    state["failed"] = f"{name}: {type(e).__name__}: {e}"
                break
            with lock:
                state["done"] += 1
                emit("PROGRESS", state["done"], len(names), name, f"{(time.time() - t0) * 1000:.0f}")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    threads = [threading.Thread(target=worker, args=(d,), daemon=True) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if state["failed"]:
        emit("ERROR", state["failed"])
        return 1
    emit("DONE", state["done"], f"{(time.time() - t_all) * 1000:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
