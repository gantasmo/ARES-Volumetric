# Per-person avatars for a volumetric run: one reference cutout per person, multi-view images
# generated from it, a 3D model reconstructed from those views, and the model written out.
#
#   python avatar.py refs <run-dir> [--service http://127.0.0.1:7263]
#
# refs: for every object id in <run>/mask-ids.u8 (depth.py's mask pass), score the frames where
# the tracker detected that id and cut the best one out of the source video at full resolution:
#   score = area * edge * clear * sqrt(sharpness)
#     area       the id's pixel count on the map
#     edge       0.25 when the id's box touches the map border (the person is cut off), else 1
#     clear      1 - 2 x the fraction of the id's box (padded 5 %) covered by other ids, floor 0.1
#     sharpness  variance of the Laplacian of the source crop at 512 px tall (motion blur), taken
#                for the 8 best frames by area * edge * clear only
# The cutout's alpha is the SAM 3 "person" instance (the service's /segment_text on the source
# crop) that overlaps the upsampled id region most; below IoU 0.5 the upsampled region is used.
# Written per id to <run>/avatar/p<id>/: ref.png (RGBA, the person centred in a square with 10 %
# margin, 1024 px), ref-white.png (the same on white), ref.json. <run>/avatar/avatar.json lists them.
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

REF_EDGE = 1024
THUMB_EDGE = 96          # the Convert card shows this beside each person
MARGIN = 0.10
SHARP_CANDIDATES = 8


def load_run(run: str):
    meta = json.load(open(os.path.join(run, "depth.json"), encoding="utf-8"))
    mask = meta.get("mask") or {}
    if not mask.get("ids"):
        raise SystemExit(f"{run}: depth.json names no mask ids (run the mask pass with a subject)")
    H, W, N = int(meta["height"]), int(meta["width"]), int(meta["frames"])
    ids = np.memmap(os.path.join(run, mask["ids"]), np.uint8, "r", shape=(N, H, W))
    det = np.ones(N, np.uint8)
    if mask.get("detected"):
        det = np.fromfile(os.path.join(run, mask["detected"]), np.uint8)[:N]
    return meta, ids, det


def decode_frame(meta: dict, t: int, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Frame t of the run's frame set at source resolution: the same `fps=` selection as the
    depth service's decode (depth.py _decode_args), then the t-th frame."""
    samp = (meta.get("sampling") or {}).get("fps")
    vf = (f"fps={meta['fps']}," if samp else "") + f"select=eq(n\\,{t})"
    W, H = int(meta["sourceWidth"]), int(meta["sourceHeight"])
    out = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", meta["video"], "-vf", vf,
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    if len(out) < W * H * 3:
        raise RuntimeError(f"frame {t}: ffmpeg returned {len(out)} bytes, expected {W * H * 3}")
    return np.frombuffer(out[: W * H * 3], np.uint8).reshape(H, W, 3)


def box_of(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def square_crop(box, W: int, H: int, margin: float):
    """Square around box (source pixels), padded by `margin` of its long side; may extend past the
    frame (the caller pads)."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))


def take(img: np.ndarray, x: int, y: int, side: int, fill) -> np.ndarray:
    """img[y:y+side, x:x+side] with out-of-frame pixels set to `fill`."""
    H, W = img.shape[:2]
    out = np.empty((side, side) + img.shape[2:], img.dtype)
    out[...] = fill
    sx0, sy0, sx1, sy1 = max(0, x), max(0, y), min(W, x + side), min(H, y + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = img[sy0:sy1, sx0:sx1]
    return out


def segment_person(service: str, rgb: np.ndarray) -> list[np.ndarray]:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    body = json.dumps({"image": base64.b64encode(buf.getvalue()).decode("ascii"), "text": "person",
                       "maxInstances": 8, "scoreThreshold": 0.3}).encode()
    rq = urllib.request.Request(service.rstrip("/") + "/segment_text", body, {"Content-Type": "application/json"})
    res = json.load(urllib.request.urlopen(rq, timeout=120))
    out = []
    for inst in res.get("instances") or res.get("masks") or []:
        b64 = inst["mask"] if isinstance(inst, dict) else inst
        out.append(np.array(Image.open(io.BytesIO(base64.b64decode(b64)))) > 127)
    return out


def refs(run: str, service: str) -> dict:
    meta, ids, det = load_run(run)
    N, H, W = ids.shape
    SW, SH = int(meta["sourceWidth"]), int(meta["sourceHeight"])
    sx, sy = SW / W, SH / H
    objects = {o["id"]: o for o in (meta["mask"].get("objects") or [])}
    root = os.path.join(run, "avatar")
    os.makedirs(root, exist_ok=True)
    index = []
    for k in sorted(objects) or sorted(set(np.unique(ids).tolist()) - {0}):
        t0 = time.time()
        cands = []
        for t in range(N):
            if not det[t]:
                continue
            m = ids[t] == k
            a = int(m.sum())
            if a == 0:
                continue
            x0, y0, x1, y1 = box_of(m)
            edge = 0.25 if (x0 <= 1 or y0 <= 1 or x1 >= W - 1 or y1 >= H - 1) else 1.0
            px, py = int((x1 - x0) * 0.05) + 1, int((y1 - y0) * 0.05) + 1
            win = ids[t, max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)]
            other = float(((win > 0) & (win != k)).mean())
            clear = max(0.1, 1.0 - 2.0 * other)
            cands.append({"t": t, "area": a, "box": [x0, y0, x1, y1], "edge": edge, "clear": clear,
                          "pre": a * edge * clear})
        if not cands:
            continue
        cands.sort(key=lambda c: -c["pre"])
        frames = {}
        for c in cands[:SHARP_CANDIDATES]:
            img = frames[c["t"]] = decode_frame(meta, c["t"])
            x0, y0, x1, y1 = c["box"]
            crop = img[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)].mean(axis=2)
            if crop.shape[0] > 8:
                h = 512
                w = max(8, int(round(crop.shape[1] * h / crop.shape[0])))
                crop = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((w, h), Image.BILINEAR))
            c["sharp"] = laplacian_var(crop)
            c["score"] = c["pre"] * c["sharp"] ** 0.5
        best = max(cands[:SHARP_CANDIDATES], key=lambda c: c["score"])
        img = frames[best["t"]]
        x0, y0, x1, y1 = best["box"]
        sbox = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        cx, cy, side = square_crop(sbox, SW, SH, MARGIN)
        rgb = take(img, cx, cy, side, 0)
        # The id region at source resolution, in the crop's frame.
        region = np.asarray(Image.fromarray((ids[best["t"]] == k).astype(np.uint8) * 255)
                            .resize((SW, SH), Image.BILINEAR)) > 127
        region = take(region, cx, cy, side, False)
        alpha, iou = region, None
        try:
            inst = segment_person(service, rgb)
            best_iou = 0.0
            for m in inst:
                if m.shape != region.shape:
                    continue
                inter = float((m & region).sum())
                union = float((m | region).sum()) or 1.0
                if inter / union > best_iou:
                    best_iou, alpha_c = inter / union, m
            iou = round(best_iou, 4)
            if best_iou >= 0.5:
                alpha = alpha_c
        except Exception as e:           # the service is optional: the upsampled region stands in
            print(f"[avatar] p{k}: /segment_text failed ({e}); using the upsampled id region", file=sys.stderr)
        # Recentre on the final alpha so the person sits in the middle of the square.
        ax0, ay0, ax1, ay1 = box_of(alpha)
        cx2, cy2, side2 = square_crop((cx + ax0, cy + ay0, cx + ax1, cy + ay1), SW, SH, MARGIN)
        full_alpha = np.zeros((SH, SW), bool)
        fx0, fy0 = max(0, cx), max(0, cy)
        fx1, fy1 = min(SW, cx + side), min(SH, cy + side)
        full_alpha[fy0:fy1, fx0:fx1] = alpha[fy0 - cy:fy1 - cy, fx0 - cx:fx1 - cx]
        rgb2 = take(img, cx2, cy2, side2, 0)
        a2 = take(full_alpha, cx2, cy2, side2, False)
        rgba = np.dstack([rgb2, a2.astype(np.uint8) * 255])
        d = os.path.join(root, f"p{k}")
        os.makedirs(d, exist_ok=True)
        im = Image.fromarray(rgba, "RGBA").resize((REF_EDGE, REF_EDGE), Image.LANCZOS)
        im.save(os.path.join(d, "ref.png"))
        white = Image.new("RGB", im.size, (255, 255, 255))
        white.paste(im, mask=im.split()[3])
        white.save(os.path.join(d, "ref-white.png"))
        im.resize((THUMB_EDGE, THUMB_EDGE), Image.LANCZOS).save(os.path.join(d, "thumb.png"))
        info = {"id": k, "frame": best["t"], "sourceCrop": [cx2, cy2, side2], "sourcePixels": side2,
                "personHeightPx": int(round((y1 - y0) * sy)), "mapBox": best["box"], "area": best["area"],
                "edge": best["edge"], "clear": round(best["clear"], 4), "sharpness": round(best["sharp"], 1),
                "segmentIoU": iou, "alpha": "sam3" if alpha is not region else "id-region",
                "candidates": len(cands), "ms": round((time.time() - t0) * 1000)}
        json.dump(info, open(os.path.join(d, "ref.json"), "w"), indent=1)
        index.append({"id": k, "dir": f"p{k}", "frame": best["t"], "personHeightPx": info["personHeightPx"]})
        print(f"[avatar] p{k}: frame {best['t']}, person {info['personHeightPx']} px tall at source, "
              f"alpha {info['alpha']} (IoU {iou}), sharpness {info['sharpness']}")
    json.dump({"schema": "ares-avatar/1", "people": index}, open(os.path.join(root, "avatar.json"), "w"), indent=1)
    return {"people": index}


TRACK_MIN_FRAMES = 121      # 4DAnyone's clip length
TRACK_ASPECT = 16 / 9       # height / width of its 704x1280 input


def tracks(run: str, frames: int = TRACK_MIN_FRAMES, ffmpeg: str = "ffmpeg", min_height: int = 0) -> list[dict]:
    """One fixed-camera portrait clip per person: `frames` consecutive frames of the run, cropped at
    source resolution to a 9:16 window around the union of that person's boxes over those frames
    (10 % margin, clamped into the frame). The window is the one that maximizes the sum over its
    frames of the person's box height, counting a frame only where the person is present and the
    box touches neither the top nor the bottom of the map (the whole body is in view). Written to
    <run>/avatar/p<id>/track.mp4 (H.264 CRF 12, the run's frame rate) with track.json. A person
    present on under 90 % of the window, or shorter than `min_height` source pixels, gets no clip."""
    meta, ids, det = load_run(run)
    N, H, W = ids.shape
    SW, SH = int(meta["sourceWidth"]), int(meta["sourceHeight"])
    sx, sy = SW / W, SH / H
    fps = float(meta["fps"])
    root = os.path.join(run, "avatar")
    out = []
    if N < frames:
        print(f"[avatar] the run has {N} frames: no {frames}-frame clip")
        return out
    for o in meta["mask"].get("objects") or []:
        k = int(o["id"])
        per = [None] * N
        gain = np.zeros(N)
        for t in range(N):
            m = ids[t] == k
            if m.any():
                per[t] = box_of(m)
                x0_, y0_, x1_, y1_ = per[t]
                if y0_ > 1 and y1_ < H - 1:
                    gain[t] = y1_ - y0_
        csum = np.concatenate([[0.0], np.cumsum(gain)])
        start = int(np.argmax(csum[frames:] - csum[:-frames]))
        boxes = [per[t] for t in range(start, start + frames) if per[t] is not None]
        whole = int((gain[start:start + frames] > 0).sum())
        if len(boxes) < frames * 0.9:
            print(f"[avatar] p{k}: present on {len(boxes)} of {frames} frames at best: no clip")
            continue
        b = np.array(boxes, np.float64)
        x0, y0 = b[:, 0].min() * sx, b[:, 1].min() * sy
        x1, y1 = b[:, 2].max() * sx, b[:, 3].max() * sy
        person_h = float(np.median(b[:, 3] - b[:, 1])) * sy
        if person_h < min_height:
            print(f"[avatar] p{k}: {person_h:.0f} px tall at source (< {min_height}): no clip")
            continue
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
        ch = max(y1 - y0, (x1 - x0) * TRACK_ASPECT)
        ch = min(ch, SH, SW * TRACK_ASPECT)
        cw = ch / TRACK_ASPECT
        cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
        cx = int(round(min(max((x0 + x1) / 2 - cw / 2, 0), SW - cw)))
        cy = int(round(min(max((y0 + y1) / 2 - ch / 2, 0), SH - ch)))
        d = os.path.join(root, f"p{k}")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "track.mp4")
        samp = (meta.get("sampling") or {}).get("fps")
        vf = (f"fps={meta['fps']}," if samp else "") + \
            f"select=between(n\\,{start}\\,{start + frames - 1}),setpts=N/({fps}*TB),crop={cw}:{ch}:{cx}:{cy}"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", meta["video"], "-vf", vf,
                        "-frames:v", str(frames), "-r", f"{fps}", "-an", "-c:v", "libx264", "-crf", "12",
                        "-pix_fmt", "yuv420p", dst], check=True)
        info = {"id": k, "start": start, "frames": frames, "fps": fps, "crop": [cx, cy, cw, ch],
                "personHeightPx": round(person_h), "wholeBodyFrames": whole, "file": "track.mp4"}
        json.dump(info, open(os.path.join(d, "track.json"), "w"), indent=1)
        out.append(info)
        print(f"[avatar] p{k}: track.mp4 {cw}x{ch}, frames {start}..{start + frames - 1}, person "
              f"{person_h:.0f} px tall, whole body in view on {whole} of {frames}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refs")
    r.add_argument("run")
    r.add_argument("--service", default=os.environ.get("SAM_URL", "http://127.0.0.1:7263"))
    t = sub.add_parser("tracks")
    t.add_argument("run")
    t.add_argument("--frames", type=int, default=TRACK_MIN_FRAMES)
    t.add_argument("--min-height", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "refs":
        refs(a.run, a.service)
    elif a.cmd == "tracks":
        tracks(a.run, a.frames, min_height=a.min_height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
