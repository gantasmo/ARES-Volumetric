"""Synthetic 4DAnyone-style result: a known figure ray-traced through 4DAnyone's own camera ring.

Body: ellipsoid centre (0, 0.95, 0) radii (0.26, 0.52, 0.18); head: sphere centre (0, 1.62, 0)
r 0.13. Colour: 8 vertical stripes by world azimuth plus a horizontal band every 20 cm, so a
mis-mapped texture shows up as broken stripes. Writes <out>/cameras.json and <out>/videos/NN.mp4.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FD = Path(__file__).resolve().parents[2] / "ext" / "4danyone"
sys.path.insert(0, str(FD))

from fdanyone.config import CAMERA  # noqa: E402
from fdanyone.geometry.cameras import camera_ring, reference_intrinsics  # noqa: E402

OUT = Path(sys.argv[1])
VIEWS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
H, W = 1280, 704
BODY_C, BODY_R = np.array([0.0, 0.95, 0.0]), np.array([0.26, 0.52, 0.18])
HEAD_C, HEAD_R = np.array([0.0, 1.55, 0.0]), 0.13


def trace(K, c2w, t):
    """RGB uint8 (H, W, 3): the figure on black, moved 0.05 m per frame along +x."""
    shift = np.array([0.05 * t, 0.0, 0.0])
    ys, xs = np.mgrid[0:H, 0:W]
    pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], -1).reshape(-1, 3).astype(np.float64)
    dirs = pix @ np.linalg.inv(K).T
    dirs = dirs @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origin = c2w[:3, 3]
    best_t = np.full(len(dirs), np.inf)
    hit = np.zeros((len(dirs), 3))
    for centre, radii in ((BODY_C + shift, BODY_R), (HEAD_C + shift, np.array([HEAD_R] * 3))):
        o = (origin - centre) / radii
        d = dirs / radii
        a = (d * d).sum(1)
        b = 2 * (o * d).sum(1)
        c = (o * o).sum() - 1
        disc = b * b - 4 * a * c
        ok = disc > 0
        tt = np.full(len(dirs), np.inf)
        tt[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
        tt[tt < 0] = np.inf
        closer = tt < best_t
        best_t[closer] = tt[closer]
        hit[closer] = origin + dirs[closer] * tt[closer, None]
    rgb = np.zeros((len(dirs), 3))
    on = np.isfinite(best_t)
    p = hit[on] - shift
    azimuth = (np.arctan2(p[:, 2], p[:, 0]) + np.pi) / (2 * np.pi)
    stripe = (azimuth * 8).astype(int) % 2
    band = ((p[:, 1] * 5).astype(int) % 2)
    rgb[on] = np.stack([
        np.where(stripe == 0, 235, 40) * np.where(band == 0, 1.0, 0.55),
        np.where(band == 0, 90, 210) * 1.0,
        np.where(stripe == 0, 60, 225) * np.where(band == 0, 1.0, 0.7),
    ], 1)
    return rgb.reshape(H, W, 3).clip(0, 255).astype(np.uint8)


def main() -> None:
    K = reference_intrinsics(H, W)
    spec = type(CAMERA)(**{**CAMERA.__dict__, "count": VIEWS})
    cams = camera_ring(center=np.zeros(3), front_direction=np.array([0.0, 0.0, 1.0]), K=K,
                       image_height=H, image_width=W, spec=spec)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    records = []
    for cam in cams:
        c2w = np.asarray(cam.camera_to_world)
        path = OUT / "videos" / f"{cam.camera_id:02d}.mp4"
        proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                 "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                                 "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(path)],
                                stdin=subprocess.PIPE)
        for t in range(FRAMES):
            proc.stdin.write(trace(np.asarray(cam.K), c2w, t).tobytes())
        proc.stdin.close()
        proc.wait()
        records.append({"camera_id": cam.camera_id, "layer_index": 0, "pitch": cam.pitch_degrees,
                        "yaw": cam.yaw_degrees, "K": cam.K, "camera_to_world": cam.camera_to_world,
                        "image_width": W, "image_height": H, "video": f"videos/{cam.camera_id:02d}.mp4"})
    (OUT / "cameras.json").write_text(json.dumps(
        {"camera_model": "OPENCV", "camera_frame": "opencv_camera", "front_camera_ids": [0],
         "cameras": records}, indent=1))
    print(f"wrote {len(records)} views x {FRAMES} frames to {OUT}")


main()
