#!/usr/bin/env python3
"""
Multiview scene-flow capture (pod-side) — produces the dense correspondence the registration will consume.

For each consecutive frame pair (f, f+1) over a range, and each of N camera views: render both frames
(RGB + depth), optical flow f→f+1, unproject both depths to world XYZ, and emit texture-tracked scene-flow
samples (src_world_XYZ → dst_world_XYZ) — filtered to samples where the flow lands on the surface AND is
texture-consistent (rejects unreliable flow, e.g. the fastest/blurriest pixels). Views are merged so the
whole surface is covered. Output per pair = compact JSON of [sx,sy,sz,dx,dy,dz] samples in ARES mm.

Downstream (JS ARAP): for template vertex v at its current position, its registration TARGET for frame f+1
is v + (interpolated scene-flow of nearby samples) — appearance-tracked, so no wrong-surface snap. Falls
back to nearest-point where no confident flow sample is near (occlusion).

  PYOPENGL_PLATFORM=egl python mv_flow_capture.py --frames-dir frames --from 140 --to 169 \
      --views 6 --res 1024 --tex-thresh 45 --samples-per-view 6000 --out flow-samples
"""
import os, sys, argparse, math, json
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender, cv2
from PIL import Image


def load_mesh(obj, atlas):
    m = trimesh.load(obj, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    try:
        uv = getattr(m.visual, "uv", None)
        if uv is not None and atlas and os.path.exists(atlas):
            m.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.open(atlas).convert("RGB"))
    except Exception:
        pass
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up); s /= np.linalg.norm(s) + 1e-12
    u = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M


def render(r, mesh_t, cam_pose, yfov, W, H):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(pyrender.Mesh.from_trimesh(mesh_t, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W / H)
    scene.add(cam, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=cam_pose)
    color, depth = r.render(scene)
    return np.ascontiguousarray(color[:, :, :3]), depth


def unproject(depth, cam_pose, yfov, W, H):
    fy = (H / 2.0) / math.tan(yfov / 2.0); fx = fy; cx = W / 2.0; cy = H / 2.0
    ys, xs = np.mgrid[0:H, 0:W]
    d = depth
    cam = np.stack([(xs - cx) * d / fx, -(ys - cy) * d / fy, -d, np.ones_like(d)], axis=-1)
    return (cam @ cam_pose.T)[:, :, :3]


_RAFT = {"net": None, "dev": None}


def raft_flow(rgbA, rgbB):
    """Learned optical flow (RAFT-large, torchvision weights ~20MB).

    Replaces cv2.calcOpticalFlowFarneback. Farneback was measured INSUFFICIENT: its outliers dominate
    the inverse-distance average, and gating them out collapses coverage to 8-26% exactly on the fast
    limbs where correspondence is needed, so it converges to the nearest-point baseline at best.
    RAFT handles large motion with far fewer outliers, and consumes RGB rather than the grayscale
    Farneback needed — colour is signal here, since we are tracking texture.
    """
    import torch
    if _RAFT["net"] is None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _RAFT["dev"] = "cuda" if torch.cuda.is_available() else "cpu"
        _RAFT["net"] = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(_RAFT["dev"]).eval()
        print(f"[flow-cap] RAFT-large loaded on {_RAFT['dev']}", flush=True)
    dev = _RAFT["dev"]
    prep = lambda im: (torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float()[None] / 255.0 * 2.0 - 1.0).to(dev)
    a, b = prep(rgbA), prep(rgbB)
    H, W = a.shape[-2:]
    ph, pw = (-H) % 8, (-W) % 8                      # RAFT requires dims divisible by 8
    if ph or pw:
        a = torch.nn.functional.pad(a, (0, pw, 0, ph), mode="replicate")
        b = torch.nn.functional.pad(b, (0, pw, 0, ph), mode="replicate")
    with torch.no_grad():
        out = _RAFT["net"](a, b)[-1]                 # iterative refinement -> last prediction is the flow
    return out[0, :, :H, :W].permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--from", dest="f0", type=int, required=True)
    ap.add_argument("--to", dest="f1", type=int, required=True)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--elev", type=float, default=10.0)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--tex-thresh", type=float, default=45.0, help="reject flow samples with color err above this")
    ap.add_argument("--samples-per-view", type=int, default=6000)
    ap.add_argument("--out", default="flow-samples")
    ap.add_argument("--flow-backend", choices=["raft", "farneback"], default="raft",
                    help="raft = learned (needs torch+torchvision); farneback = the measured-insufficient CPU baseline")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp = lambda i: (os.path.join(args.frames_dir, f"mesh-f{i:05d}.obj"), os.path.join(args.frames_dir, f"atlas-f{i:05d}.png"))
    W = int(args.res * 0.75); H = args.res; yfov = math.radians(50.0)
    r = pyrender.OffscreenRenderer(W, H)
    az_list = [i * 360.0 / args.views for i in range(args.views)]
    el = math.radians(args.elev)
    rng = np.random.RandomState(0)

    meshes = {}
    def get(i):
        if i not in meshes: meshes[i] = load_mesh(*fp(i))
        return meshes[i]

    for f in range(args.f0, args.f1):
        mA, mB = get(f), get(f + 1)
        ctr = mA.bounds.mean(axis=0); height = float(mA.bounds[1][1] - mA.bounds[0][1])
        dist = (height / 2.0) / math.tan(yfov / 2.0) * 1.25
        samples = []
        for az_deg in az_list:
            az = math.radians(az_deg)
            eye = ctr + np.array([dist * math.cos(el) * math.sin(az), dist * math.sin(el), dist * math.cos(el) * math.cos(az)])
            cp = look_at(eye, ctr)
            rgbA, dA = render(r, mA, cp, yfov, W, H); rgbB, dB = render(r, mB, cp, yfov, W, H)
            xyzA = unproject(dA, cp, yfov, W, H); xyzB = unproject(dB, cp, yfov, W, H)
            maskA = dA > 1e-6; maskB = dB > 1e-6
            if args.flow_backend == "raft":
                flow = raft_flow(rgbA, rgbB)
            else:
                grayA = cv2.cvtColor(rgbA, cv2.COLOR_RGB2GRAY); grayB = cv2.cvtColor(rgbB, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(grayA, grayB, None, 0.5, 5, 31, 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            ys, xs = np.nonzero(maskA)
            if len(ys) == 0: continue
            sel = rng.choice(len(ys), size=min(args.samples_per_view, len(ys)), replace=False)
            ys, xs = ys[sel], xs[sel]
            tx = np.clip(np.round(xs + flow[ys, xs, 0]).astype(int), 0, W - 1)
            ty = np.clip(np.round(ys + flow[ys, xs, 1]).astype(int), 0, H - 1)
            good = maskB[ty, tx]
            terr = np.linalg.norm(rgbA[ys, xs].astype(float) - rgbB[ty, tx].astype(float), axis=1)
            good &= terr < args.tex_thresh                          # keep only texture-consistent (reliable) flow
            src = xyzA[ys[good], xs[good]]; dst = xyzB[ty[good], tx[good]]
            for s, d in zip(src, dst):
                samples.append([round(float(s[0]), 2), round(float(s[1]), 2), round(float(s[2]), 2),
                                round(float(d[0]), 2), round(float(d[1]), 2), round(float(d[2]), 2)])
        json.dump({"from": f, "to": f + 1, "n": len(samples), "samples": samples},
                  open(os.path.join(args.out, f"pair_{f:05d}.json"), "w"))
        print(f"[flow-cap] pair f{f}->f{f+1}: {len(samples)} texture-consistent scene-flow samples ({args.views} views)", flush=True)
    r.delete()
    print("[flow-cap] DONE", flush=True)


if __name__ == "__main__":
    main()
