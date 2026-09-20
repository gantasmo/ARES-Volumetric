"""Compare a reconstructed frame against the synthetic figure it was rendered from."""
import sys

import numpy as np
from PIL import Image

BODY_C, BODY_R = np.array([0.0, 0.95, 0.0]), np.array([0.26, 0.52, 0.18])
HEAD_C, HEAD_R = np.array([0.0, 1.55, 0.0]), 0.13
FLOOR = BODY_C[1] - BODY_R[1]


def read_obj(path):
    v, vt, f = [], [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            f.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.array(v), np.array(vt), np.array(f)


def surface_distance(p):
    """Distance to the union surface, by projecting onto each shape (Newton on the ellipsoid)."""
    out = []
    for centre, radii in ((BODY_C, BODY_R), (HEAD_C, np.array([HEAD_R] * 3))):
        q = p - centre
        # closest point on an ellipsoid: scale-space bisection on the Lagrange multiplier
        lo, hi = -radii.min() ** 2 + 1e-9, np.linalg.norm(q, axis=1) * radii.max()
        for _ in range(60):
            mid = (lo + hi) / 2
            f = ((radii[None] ** 2 * q ** 2) / (radii[None] ** 2 + mid[:, None]) ** 2).sum(1) - 1
            hi = np.where(f < 0, mid, hi)
            lo = np.where(f >= 0, mid, lo)
        t = (lo + hi) / 2
        closest = radii[None] ** 2 * q / (radii[None] ** 2 + t[:, None])
        out.append(np.linalg.norm(q - closest, axis=1))
    return np.minimum(*out)


def colour_at(p):
    az = (np.arctan2(p[:, 2], p[:, 0]) + np.pi) / (2 * np.pi)
    stripe = (az * 8).astype(int) % 2
    band = (p[:, 1] * 5).astype(int) % 2
    return np.stack([
        np.where(stripe == 0, 235, 40) * np.where(band == 0, 1.0, 0.55),
        np.where(band == 0, 90, 210) * 1.0,
        np.where(stripe == 0, 60, 225) * np.where(band == 0, 1.0, 0.7),
    ], 1)


obj, atlas_path = sys.argv[1], sys.argv[2]
SHIFT = np.array([0.05 * float(sys.argv[3]), 0.0, 0.0]) if len(sys.argv) > 3 else np.zeros(3)
v, vt, f = read_obj(obj)
atlas = np.asarray(Image.open(atlas_path).convert("RGB"))
world = v + np.array([0.0, FLOOR, 0.0]) - SHIFT   # undo the floor shift and the scene motion
d = surface_distance(world)
print(f"vertices {len(v)}, faces {len(f)}")
print(f"bounds x [{v[:, 0].min():+.3f}, {v[:, 0].max():+.3f}]  y [{v[:, 1].min():+.3f}, {v[:, 1].max():+.3f}]"
      f"  z [{v[:, 2].min():+.3f}, {v[:, 2].max():+.3f}]   (expected x +-0.26, y 0..1.32, z +-0.18)")
print(f"distance to the true surface: p50 {np.percentile(d, 50) * 1000:.1f} mm, "
      f"p95 {np.percentile(d, 95) * 1000:.1f} mm, max {d.max() * 1000:.1f} mm")
size = atlas.shape[0]
px = np.clip((vt[:, 0] * size).astype(int), 0, size - 1)
py = np.clip(((1 - vt[:, 1]) * size).astype(int), 0, size - 1)
got = atlas[py, px].astype(float)
want = colour_at(world)
err = np.abs(got - want).mean(1)
print(f"atlas colour at the vertices: mean |error| {err.mean():.1f}/255, p90 {np.percentile(err, 90):.1f}, "
      f"share over 60 (wrong stripe) {float((err > 60).mean()) * 100:.1f} %")
