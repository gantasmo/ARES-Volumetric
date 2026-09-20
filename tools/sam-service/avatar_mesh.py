# Models from generated views: one textured mesh per frame from a 4DAnyone result (N synchronized
# target-view videos around one person plus their OpenCV camera rig), written as a capture folder
# the ARES encoder converts: mesh-fNNNNN.obj + atlas-fNNNNN.png per frame.
#
#   <4danyone venv>/python avatar_mesh.py <result-dir> <out-dir> [--voxel 0.01] [--faces 40000]
#       [--tex 1024] [--start 0] [--count N] [--tolerance 1] [--device cuda:0]
#       [--view-power 3.0] [--top-views 0]
#
# Runs in the 4DAnyone environment (tools/ext/4danyone/venv): torch, its BiRefNet copy and its
# camera helpers. Per frame:
#   masks     BiRefNet (the model 4DAnyone ships for its own framing) on every view, > 0.5
#   hull      voxels of `voxel` metres in a 2.5 m cube around the rig's look-at point; a voxel is
#             kept when it projects inside the mask in all but `tolerance` of the views that see it
#   surface   marching cubes on the kept voxels blurred by one voxel, the largest connected
#             piece, decimated to `faces` triangles (fast-simplification)
#   uv        xatlas
#   texture   each texel's surface point is coloured from the views that see it: visible when its
#             depth is within 2 voxels of the nearest texel point splatted into that view at
#             quarter resolution, weighted by max(0, n . to-camera)^`view-power`; texels no view
#             sees take the nearest coloured texel. `top-views` above 0 keeps, per texel, only the
#             `top-views` highest-weight visible views and renormalises the weighted mean over
#             those; 0 keeps every visible view
# World: 4DAnyone's canonical frame, metric and Y-up (fdanyone/nerfstudio/cameras.py), with the
# floor moved to y = 0 by the lowest vertex of the first frame.
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FDANYONE = Path(__file__).resolve().parents[1] / "ext" / "4danyone"
sys.path.insert(0, str(FDANYONE))

HALF_EXTENT = 1.25
BIREFNET_SIZE = (1024, 1024)


class Foreground:
    """BiRefNet loaded once (fdanyone.foreground reloads it per call)."""

    def __init__(self, device: str):
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        from fdanyone.assets import resolve_foreground_model

        path = resolve_foreground_model(str(FDANYONE / "models"))
        self.model = AutoModelForImageSegmentation.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=True).eval().half().to(device)
        self.device = device
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.torch = torch

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as F

        h, w = images[0].shape[:2]
        x = torch.from_numpy(np.stack(images)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=BIREFNET_SIZE, mode="bilinear", align_corners=False)
        x = self.norm(x).half()
        out = []
        with torch.inference_mode():
            for i in range(0, len(x), 4):
                p = self.model(x[i:i + 4])[-1].sigmoid()
                out.append(F.interpolate(p.float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0])
        return (torch.cat(out) > 0.5).cpu().numpy()


def read_rig(result: Path):
    rig = json.loads((result / "cameras.json").read_text())
    cams = rig["cameras"]
    K = np.stack([np.asarray(c["K"], np.float64) for c in cams])
    c2w = np.stack([np.asarray(c["camera_to_world"], np.float64) for c in cams])
    w2c = np.linalg.inv(c2w)
    P = K @ w2c[:, :3, :]                                   # (N, 3, 4) world -> pixel
    from fdanyone.nerfstudio.cameras import visual_hull_center
    center = visual_hull_center(c2w)
    videos = [result / c["video"] for c in cams]
    return cams, P, c2w, center, videos


def _inside_count(pts, m, Pt, n, h, w, cam):
    """Views that see `pts` inside the mask, minus the views that see them outside it."""
    import torch

    hom = torch.cat([pts, torch.ones_like(pts[:, :1])], 1)
    proj = torch.einsum("nij,pj->npi", Pt, hom)
    z = proj[..., 2]
    u = (proj[..., 0] / z.clamp_min(1e-6)).long()
    v = (proj[..., 1] / z.clamp_min(1e-6)).long()
    seen = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    inside = torch.zeros_like(seen)
    inside[seen] = m.expand(n, h, w)[cam.expand_as(u)[seen], v[seen], u[seen]]
    return (~inside).sum(0)


def carve(masks, P, center, voxel: float, tolerance: int, device: str, coarse: int = 4):
    """Occupancy at `voxel` metres over a 2.5 m cube: a voxel is kept when at most `tolerance`
    views see it outside the mask. Carved at `coarse` x the voxel size first; only the fine voxels
    under a kept coarse voxel or its neighbours are tested, which is where the speed comes from."""
    import torch
    import torch.nn.functional as F

    n, h, w = masks.shape
    m = torch.from_numpy(masks).to(device)
    Pt = torch.from_numpy(P.astype(np.float32)).to(device)
    axis = torch.arange(-HALF_EXTENT, HALF_EXTENT, voxel, device=device)
    g = len(axis)
    cam = torch.arange(n, device=device).view(-1, 1)
    c = torch.tensor(center, dtype=torch.float32, device=device)
    gc = (g + coarse - 1) // coarse
    ci = torch.arange(gc, device=device) * coarse
    ci = torch.minimum(ci + coarse // 2, torch.tensor(g - 1, device=device))
    grid = torch.stack(torch.meshgrid(axis[ci], axis[ci], axis[ci], indexing="ij"), -1).reshape(-1, 3) + c
    keep_c = torch.zeros(gc ** 3, dtype=torch.bool, device=device)
    for s in range(0, gc ** 3, 1 << 21):
        sl = slice(s, min(s + (1 << 21), gc ** 3))
        keep_c[sl] = _inside_count(grid[sl], m, Pt, n, h, w, cam) <= tolerance
    # A coarse voxel is only a hint, so dilate by one before refining.
    block = F.max_pool3d(keep_c.view(1, 1, gc, gc, gc).float(), 3, 1, 1)[0, 0] > 0
    fine = block.repeat_interleave(coarse, 0).repeat_interleave(coarse, 1).repeat_interleave(coarse, 2)[:g, :g, :g]
    occ = torch.zeros(g ** 3, dtype=torch.bool, device=device)
    idx_all = torch.nonzero(fine.reshape(-1), as_tuple=False).squeeze(1)
    for s in range(0, len(idx_all), 1 << 21):
        idx = idx_all[s:s + (1 << 21)]
        pts = torch.stack([axis[idx // (g * g)], axis[(idx // g) % g], axis[idx % g]], 1) + c
        occ[idx] = _inside_count(pts, m, Pt, n, h, w, cam) <= tolerance
    return occ.view(g, g, g).cpu().numpy(), axis[0].item()


def signed_distance(occ: np.ndarray, voxel: float, margin: int = 40):
    """Metres to the occupancy boundary, negative outside the surface, positive inside, over the
    occupied box grown by `margin` voxels (the rest of the cube is empty and costs nothing).
    Returns (field, offset in voxels of the field's origin within the full grid)."""
    from scipy import ndimage

    idx = np.nonzero(occ)
    lo = np.maximum(np.array([i.min() for i in idx]) - margin, 0)
    hi = np.minimum(np.array([i.max() for i in idx]) + margin + 1, occ.shape)
    sub = occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    inside = ndimage.distance_transform_edt(sub)
    outside = ndimage.distance_transform_edt(~sub)
    return (inside - outside).astype(np.float32) * voxel, lo


def occupancy_centroid(occ: np.ndarray, origin: float, voxel: float, center) -> np.ndarray:
    idx = np.nonzero(occ)
    return np.array([i.mean() for i in idx]) * voxel + origin + np.asarray(center)


def vertex_normals(verts, tris):
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def taubin(verts, tris, passes: int = 1, lam: float = 0.5, mu: float = -0.53):
    nbr_sum = np.zeros_like(verts)
    deg = np.zeros(len(verts))
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    for _ in range(passes):
        for step in (lam, mu):
            nbr_sum[:] = 0.0
            deg[:] = 0.0
            np.add.at(nbr_sum, edges[:, 0], verts[edges[:, 1]])
            np.add.at(deg, edges[:, 0], 1.0)
            avg = nbr_sum / np.maximum(deg, 1)[:, None]
            verts = verts + step * (avg - verts)
    return verts


def wrap(verts, tris, sdf, offset, origin: float, voxel: float, center, shift, iterations: int = 12):
    """Move a mesh onto this frame's hull, keeping its vertex indices and faces (so the caller's UVs,
    indexed the same way, still apply to the result). Called on the welded mesh, so every edge,
    including a UV seam, is a normal interior edge and the fit has no seam to tear along. The mesh
    is first translated by `shift` (the hull's centroid motion, which a normal-direction fit cannot
    follow), then each vertex steps along its own normal onto the zero level of the distance field,
    capped at two voxels a step so a vertex cannot jump across a limb. Returns (verts, p95 residual
    metres)."""
    from scipy.ndimage import map_coordinates

    base = np.asarray(center)[None] + origin + offset[None] * voxel
    v = verts + np.asarray(shift)[None]
    cap = 2 * voxel

    def distance(points):
        return map_coordinates(sdf, ((points - base) / voxel).T[[0, 1, 2]], order=1, mode="nearest")

    # A normal step only converges when the normals point out of the surface: check against the
    # field itself (inside is positive, so moving along an outward normal must lower the distance).
    n = vertex_normals(v, tris)
    probe = 2 * voxel
    sign = -1.0 if distance(v + n * probe).mean() > distance(v - n * probe).mean() else 1.0
    for _ in range(iterations):
        n = vertex_normals(v, tris) * sign
        v = v + n * np.clip(distance(v), -cap, cap)[:, None]
        v = taubin(v, tris, passes=1)
    # A vertex that ends up beyond `stray` metres of the surface has left the field's gradient and
    # cannot walk back: it takes the mean position of its edge-connected neighbours that are not
    # stray, one ring at a time so an interior stray vertex fills in only once the rim around it has
    # a position, until a sweep resolves no further vertex or 50 sweeps pass. A stray vertex with no
    # path to a non-stray vertex is left at its position. The mesh then fits again with the rest.
    stray = np.abs(distance(v)) > 0.03
    if stray.any() and (~stray).sum() > 100:
        edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        edges = np.concatenate([edges, edges[:, ::-1]])
        available = ~stray
        for _ in range(50):
            nbr_sum = np.zeros_like(v)
            deg = np.zeros(len(v))
            src = available[edges[:, 1]]
            np.add.at(nbr_sum, edges[src, 0], v[edges[src, 1]])
            np.add.at(deg, edges[src, 0], 1.0)
            fillable = stray & ~available & (deg > 0)
            if not fillable.any():
                break
            v[fillable] = nbr_sum[fillable] / deg[fillable][:, None]
            available[fillable] = True
        for _ in range(4):
            n = vertex_normals(v, tris) * sign
            v = v + n * np.clip(distance(v), -cap, cap)[:, None]
            v = taubin(v, tris, passes=1)
    d = np.abs(distance(v))
    return v, float(np.percentile(d, 95))


def surface(occ: np.ndarray, origin: float, voxel: float, center, faces: int):
    import fast_simplification
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from skimage.measure import marching_cubes

    field = ndimage.gaussian_filter(occ.astype(np.float32), 1.0)
    verts, tris, _, _ = marching_cubes(field, 0.5)
    verts = verts * voxel + origin + np.asarray(center)[None]
    # largest connected piece
    nv = len(verts)
    rows = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    cols = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    ncomp, label = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nv, nv)), directed=False)
    if ncomp > 1:
        # Keep every piece holding at least 5 % of the biggest piece's triangles: a person plus what
        # they carry, without the specks a mask error leaves behind.
        counts = np.bincount(label[tris[:, 0]], minlength=ncomp)
        keep = np.flatnonzero(counts >= max(1, counts.max() * 0.05))
        tris = tris[np.isin(label[tris[:, 0]], keep)]
        used = np.unique(tris)
        remap = -np.ones(nv, np.int64)
        remap[used] = np.arange(len(used))
        verts, tris = verts[used], remap[tris]
    if len(tris) > faces:
        verts, tris = fast_simplification.simplify(verts.astype(np.float32), tris.astype(np.int32),
                                                   target_reduction=1 - faces / len(tris))
    verts, tris = np.asarray(verts, np.float64), np.asarray(tris, np.int64)
    # Orient outward: the signed volume of a closed mesh is positive when its triangles wind
    # counter-clockwise seen from outside.
    a_, b_, c_ = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if float(np.einsum("ij,ij->i", a_, np.cross(b_, c_)).sum()) < 0:
        tris = tris[:, ::-1].copy()
    return verts, tris


def unwrap(verts, tris):
    """UV-parametrize with xatlas, which cuts the mesh along its chart seams and duplicates every
    seam vertex once per chart it borders, so the returned tris index a vertex list with no shared
    edges across a seam. Returns the split mesh xatlas produces (split verts, tris in split
    indices, uvs) for baking and writing, plus the welded mesh it was cut from (the input verts,
    unchanged, and the input tris re-indexed as `vmap[idx]`) and `vmap` itself, which maps a split
    vertex to the welded vertex it was copied from. A deformation pass runs on the welded mesh,
    where every seam is a normal interior edge, and the caller re-splits the result with
    `welded_verts[vmap]` to get back split verts for the UV-indexed output."""
    import xatlas

    vmap, idx, uvs = xatlas.parametrize(verts.astype(np.float32), tris.astype(np.uint32))
    idx = idx.astype(np.int64)
    vmap = vmap.astype(np.int64)
    return verts[vmap], idx, uvs.astype(np.float64), verts, vmap[idx], vmap


def rasterize_uv(uvs, tris, size: int):
    """Face id per texel (-1 = none) and barycentrics, UV origin bottom-left, row 0 at the top."""
    import cv2

    fid = np.full((size, size), -1, np.int32)
    px = np.stack([uvs[:, 0] * size - 0.5, (1 - uvs[:, 1]) * size - 0.5], 1)
    for f, (a, b, c) in enumerate(tris):
        poly = np.round(px[[a, b, c]] * 16).astype(np.int32)
        cv2.fillConvexPoly(fid, poly, int(f), lineType=cv2.LINE_8, shift=4)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    p = np.stack([xs, ys], 1).astype(np.float64)
    A, B, C = px[tris[f, 0]], px[tris[f, 1]], px[tris[f, 2]]
    v0, v1, v2 = B - A, C - A, p - A
    d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    d[np.abs(d) < 1e-12] = 1e-12
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / d
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / d
    bary = np.clip(np.stack([1 - l1 - l2, l1, l2], 1), 0, 1)
    bary /= bary.sum(1, keepdims=True)
    return ys, xs, f, bary


def bake(verts, tris, uvs, images, masks, P, c2w, size: int, voxel: float, device: str, raster=None,
         view_power: float = 3.0, top_views: int = 0):
    import cv2
    import torch

    ys, xs, f, bary = rasterize_uv(uvs, tris, size) if raster is None else raster
    tri = verts[tris]
    pts = np.einsum("tk,tkj->tj", bary, tri[f])
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    nrm = fn[f]
    n, h, w = masks.shape
    dev = torch.device(device)
    pt = torch.from_numpy(pts).float().to(dev)
    nt = torch.from_numpy(nrm).float().to(dev)
    n_texels = len(pts)
    hs, ws = h // 4, w // 4
    if top_views > 0:
        # per-view weight and colour kept whole so the top-k views per texel can be picked after
        # the loop; float16/uint8 keeps this to a few hundred MB at n_views ~ 48, n_texels ~ 1e6
        view_wgt = torch.zeros((n, n_texels), dtype=torch.float16, device=dev)
        view_col = torch.zeros((n, n_texels, 3), dtype=torch.uint8, device=dev)
    else:
        acc = torch.zeros((n_texels, 3), device=dev)
        wsum = torch.zeros(n_texels, device=dev)
    for k in range(n):
        Pk = torch.from_numpy(P[k].astype(np.float32)).to(dev)
        proj = pt @ Pk[:, :3].T + Pk[:, 3]
        z = proj[:, 2]
        u, v = proj[:, 0] / z, proj[:, 1] / z
        ok = (z > 0) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        # visibility: nearest splatted depth per quarter-resolution cell
        cu, cv_ = (u / 4).long().clamp(0, ws - 1), (v / 4).long().clamp(0, hs - 1)
        cell = cv_ * ws + cu
        zbuf = torch.full((hs * ws,), float("inf"), device=dev)
        zbuf.scatter_reduce_(0, cell[ok], z[ok], reduce="amin")
        vis = ok & (z <= zbuf[cell] + 2 * voxel)
        cam_pos = torch.from_numpy(c2w[k, :3, 3].astype(np.float32)).to(dev)
        to_cam = cam_pos - pt
        to_cam = to_cam / to_cam.norm(dim=1, keepdim=True)
        wgt = (nt * to_cam).sum(1).clamp_min(0) ** view_power
        img = torch.from_numpy(images[k]).to(dev).float()
        mk = torch.from_numpy(masks[k]).to(dev)
        ui, vi = u.round().long().clamp(0, w - 1), v.round().long().clamp(0, h - 1)
        vis = vis & mk[vi, ui] & (wgt > 1e-3)
        # bilinear colour
        u0, v0 = u.floor().long().clamp(0, w - 2), v.floor().long().clamp(0, h - 2)
        fu, fv = (u - u0).clamp(0, 1)[:, None], (v - v0).clamp(0, 1)[:, None]
        col = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv)
               + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)
        wv = torch.where(vis, wgt, torch.zeros_like(wgt))
        if top_views > 0:
            view_wgt[k] = wv.half()
            view_col[k] = col.clamp(0, 255).byte()
        else:
            acc += col * wv[:, None]
            wsum += wv
    if top_views > 0:
        k = min(top_views, n)
        top_wgt, top_idx = torch.topk(view_wgt, k, dim=0)  # (k, n_texels), view axis
        top_col = torch.gather(view_col, 0, top_idx[:, :, None].expand(-1, -1, 3))  # (k, n_texels, 3)
        wsum = top_wgt.float().sum(0)
        colour = (top_col.float() * top_wgt.float()[:, :, None]).sum(0) / wsum.clamp_min(1e-6)[:, None]
        colour = colour.clamp(0, 255).byte().cpu().numpy()
    else:
        colour = (acc / wsum.clamp_min(1e-6)[:, None]).clamp(0, 255).byte().cpu().numpy()
    seen = (wsum > 1e-6).cpu().numpy()
    atlas = np.zeros((size, size, 3), np.uint8)
    have = np.zeros((size, size), bool)
    atlas[ys[seen], xs[seen]] = colour[seen]
    have[ys[seen], xs[seen]] = True
    # every texel without a colour takes the nearest coloured texel (inside charts and a gutter)
    from scipy import ndimage
    if have.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~have, return_indices=True)
        atlas = atlas[iy, ix]
    return cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR), float(seen.mean())


def write_obj(path: Path, verts, uvs, tris) -> None:
    with open(path, "w", newline="\n") as fh:
        fh.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in verts))
        fh.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in uvs))
        fh.write("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in tris + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--faces", type=int, default=40000)
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--view-power", type=float, default=3.0,
                    help="exponent on the cosine (normal, to-camera) bake weight")
    ap.add_argument("--top-views", type=int, default=0,
                    help="keep only the N highest-weight visible views per texel and renormalise; "
                         "0 keeps every visible view")
    ap.add_argument("--mask", choices=("birefnet", "luma"), default="birefnet",
                    help="luma keeps pixels brighter than 12/255: for synthetic views on black")
    ap.add_argument("--rekey", type=float, default=0.02,
                    help="metres of p95 fit residual above which a frame is meshed as a new keyframe; "
                         "0 meshes and unwraps every frame (no shared topology)")
    ap.add_argument("--iterations", type=int, default=12, help="deformation steps per frame")
    a = ap.parse_args()

    import cv2

    from fdanyone.video import iter_rgb_video

    result, out = Path(a.result), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams, P, c2w, center, videos = read_rig(result)
    fg = Foreground(a.device) if a.mask == "birefnet" else \
        (lambda images: np.stack([im.max(axis=2) > 12 for im in images]))
    streams = [iter_rgb_video(v) for v in videos]
    floor = None
    stats: list[dict] = []
    rekeys: list[int] = []
    keyed = None
    last_centroid = np.zeros(3)
    t = -1
    while True:
        try:
            images = [next(s) for s in streams]
        except StopIteration:
            break
        t += 1
        if t < a.start:
            continue
        if a.count and t >= a.start + a.count:
            break
        t0 = time.time()
        masks = fg(images)
        occ, origin = carve(masks, P, center, a.voxel, a.tolerance, a.device)
        if occ.sum() < 1000:
            print(f"[mesh] frame {t}: visual hull holds {int(occ.sum())} voxels; skipped", flush=True)
            continue
        t_carve = time.time()
        # One topology per GOP: the keyframe is meshed and unwrapped, every following frame moves
        # the keyframe's WELDED mesh (no UV seams, so a deformation step has no chart boundary to
        # tear along) onto its own hull and re-splits the result for baking and writing, so the
        # split vertex indices and UVs persist and the encoder codes positions as deltas
        # (packages/encoder/src/temporal.ts). A frame the keyframe cannot reach (p95 residual over
        # --rekey metres) becomes the next keyframe.
        kind, residual = "key", 0.0
        centroid = occupancy_centroid(occ, origin, a.voxel, center)
        if keyed is None or a.rekey <= 0:
            verts, tris = surface(occ, origin, a.voxel, center, a.faces)
            verts, tris, uvs, welded_verts, welded_tris, vmap = unwrap(verts, tris)
            raster = rasterize_uv(uvs, tris, a.tex)
        else:
            verts, tris, uvs, raster, welded_verts, welded_tris, vmap = keyed
            sdf, offset = signed_distance(occ, a.voxel)
            moved_welded, residual = wrap(welded_verts, welded_tris, sdf, offset, origin, a.voxel,
                                          center, centroid - last_centroid, a.iterations)
            if residual > a.rekey:
                verts, tris = surface(occ, origin, a.voxel, center, a.faces)
                verts, tris, uvs, welded_verts, welded_tris, vmap = unwrap(verts, tris)
                raster = rasterize_uv(uvs, tris, a.tex)
                rekeys.append(t)
            else:
                welded_verts = moved_welded
                verts, kind = welded_verts[vmap], "deformed"
        keyed = (verts, tris, uvs, raster, welded_verts, welded_tris, vmap)
        last_centroid = centroid
        t_geom = time.time()
        atlas, coverage = bake(verts, tris, uvs, images, masks, P, c2w, a.tex, a.voxel, a.device, raster,
                                a.view_power, a.top_views)
        if floor is None:
            floor = float(verts[:, 1].min())
        n = t - a.start + 1
        write_obj(out / f"mesh-f{n:05d}.obj", verts - np.array([center[0], floor, center[2]]), uvs, tris)
        cv2.imwrite(str(out / f"atlas-f{n:05d}.png"), atlas)
        ms = (time.time() - t0) * 1000
        stats.append({"frame": t, "kind": kind, "voxels": int(occ.sum()), "vertices": len(verts),
                      "faces": len(tris), "texelsSeen": round(coverage, 4), "residualMm": round(residual * 1000, 1),
                      "ms": round(ms), "carveMs": round((t_carve - t0) * 1000),
                      "geometryMs": round((t_geom - t_carve) * 1000), "bakeMs": round((time.time() - t_geom) * 1000)})
        print(f"[mesh] frame {t}: {kind}, {len(tris)} faces, {coverage * 100:.1f} % of texels seen"
              + (f", fit {residual * 1000:.0f} mm" if kind != "key" else "")
              + f", {ms:.0f} ms (carve {(t_carve - t0) * 1000:.0f}, geometry {(t_geom - t_carve) * 1000:.0f},"
                f" bake {(time.time() - t_geom) * 1000:.0f})", flush=True)
    (out / "avatar-mesh.json").write_text(json.dumps({
        "source": str(result), "views": len(cams), "voxel": a.voxel, "faces": a.faces, "tex": a.tex,
        "tolerance": a.tolerance, "rekey": a.rekey, "viewPower": a.view_power, "topViews": a.top_views,
        "iterations": a.iterations, "rekeyFrames": rekeys,
        "world": "4DAnyone canonical, metric, Y-up, floor at y=0", "frames": stats}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
