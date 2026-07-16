#!/usr/bin/env python3
"""
Optical-flow correspondence POC (pod-side) — the decisive test for the motion-robust fix.

HYPOTHESIS: on fast motion, geometric nearest-point registration snaps a surface point to the WRONG body
part (the gate problem behind every deformation attempt), while TEXTURE flow tracks the SAME material point
correctly. We already have the 3D + textures, so: render frame N and N+1 from a fixed camera → optical flow
(texture-tracked correspondence) → unproject both depths to world XYZ → each surface pixel of N gets a
flow-based 3D target on N+1. Compare that to the geometric NEAREST-POINT target (what ARAP uses).

KEY METRIC — `snap_gap` = ||flow_target − nearest_point_target||. Where it's large (> the 40mm gate), the
nearest-point method would snap to the wrong surface but flow gets it right. If snap_gap is frequently large
on a fast pair (and texture-consistency of the flow matches is good), the flow approach is validated.

  PYOPENGL_PLATFORM=egl python mv_flow_poc.py --frames-dir frames --a 145 --b 146 [--az 0 --res 1024]
Deps: pyrender trimesh pillow numpy opencv-python-headless  (NO SAM-3D — light setup).
"""
import os, sys, argparse, math, json
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
import cv2
from PIL import Image


def load_mesh(obj, atlas):
    m = trimesh.load(obj, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    try:
        uv = getattr(m.visual, "uv", None)
        if uv is not None and atlas and os.path.exists(atlas):
            m.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.open(atlas).convert("RGB"))
    except Exception as e:
        print(f"[flow] texture apply failed ({e.__class__.__name__}); shaded gray", flush=True)
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
    return np.ascontiguousarray(color[:, :, :3]), depth   # depth float, 0 = no geometry


def unproject(depth, cam_pose, yfov, W, H):
    """Per-pixel world XYZ from pyrender depth (OpenGL cam: looks −Z, +Y up, image y-down)."""
    fy = (H / 2.0) / math.tan(yfov / 2.0); fx = fy; cx = W / 2.0; cy = H / 2.0
    ys, xs = np.mgrid[0:H, 0:W]
    d = depth
    xcam = (xs - cx) * d / fx
    ycam = -(ys - cy) * d / fy
    zcam = -d
    cam = np.stack([xcam, ycam, zcam, np.ones_like(d)], axis=-1)   # (H,W,4)
    world = cam @ cam_pose.T
    return world[:, :, :3], depth > 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--elev", type=float, default=10.0)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--gate", type=float, default=40.0)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--out", default="flow-out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp = lambda i: (os.path.join(args.frames_dir, f"mesh-f{i:05d}.obj"), os.path.join(args.frames_dir, f"atlas-f{i:05d}.png"))

    W = int(args.res * 0.75); H = args.res; yfov = math.radians(50.0)
    r = pyrender.OffscreenRenderer(W, H)
    mA = load_mesh(*fp(args.a)); mB = load_mesh(*fp(args.b))
    ctr = mA.bounds.mean(axis=0); height = float(mA.bounds[1][1] - mA.bounds[0][1])
    dist = (height / 2.0) / math.tan(yfov / 2.0) * 1.25
    az = math.radians(args.az); el = math.radians(args.elev)
    eye = ctr + np.array([dist * math.cos(el) * math.sin(az), dist * math.sin(el), dist * math.cos(el) * math.cos(az)])
    cam_pose = look_at(eye, ctr)

    rgbA, depthA = render(r, mA, cam_pose, yfov, W, H)
    rgbB, depthB = render(r, mB, cam_pose, yfov, W, H)
    xyzA, maskA = unproject(depthA, cam_pose, yfov, W, H)
    xyzB, maskB = unproject(depthB, cam_pose, yfov, W, H)
    Image.fromarray(rgbA).save(os.path.join(args.out, f"f{args.a:05d}_a.png"))   # for visual inspection
    Image.fromarray(rgbB).save(os.path.join(args.out, f"f{args.b:05d}_b.png"))

    grayA = cv2.cvtColor(rgbA, cv2.COLOR_RGB2GRAY); grayB = cv2.cvtColor(rgbB, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(grayA, grayB, None, 0.5, 5, 31, 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    ys, xs = np.nonzero(maskA)
    if len(ys) == 0:
        print("[flow] no surface pixels in A — check render", flush=True); return
    sel = np.random.RandomState(0).choice(len(ys), size=min(args.samples, len(ys)), replace=False)
    ys, xs = ys[sel], xs[sel]
    p_src = xyzA[ys, xs]                                        # (N,3) source surface points

    # geometric nearest-point target on B's surface (what ARAP would snap to)
    nn_pts, nn_dist, _ = mB.nearest.on_surface(p_src)

    # texture-flow target: follow the flow vector, sample B's world XYZ there
    dx = flow[ys, xs, 0]; dy = flow[ys, xs, 1]
    tx = np.clip(np.round(xs + dx).astype(int), 0, W - 1); ty = np.clip(np.round(ys + dy).astype(int), 0, H - 1)
    onB = maskB[ty, tx]                                          # flow landed on B's surface?
    p_flow = xyzB[ty, tx]
    tex_err = np.linalg.norm(rgbA[ys, xs].astype(float) - rgbB[ty, tx].astype(float), axis=1)

    v = onB                                                     # valid = flow lands on the surface
    scene_flow = np.linalg.norm(p_flow[v] - p_src[v], axis=1)   # texture-tracked motion magnitude
    snap_gap = np.linalg.norm(p_flow[v] - nn_pts[v], axis=1)    # KEY: flow-target vs nearest-point-target
    over = snap_gap > args.gate

    print(f"[flow] f{args.a}->f{args.b} az{args.az:.0f} | surfaceA={len(ys)} sampled, flow-on-B {v.mean()*100:.0f}%", flush=True)
    print(f"[flow] scene-flow (texture motion): mean {scene_flow.mean():.1f}mm  p95 {np.percentile(scene_flow,95):.1f}  max {scene_flow.max():.1f}", flush=True)
    print(f"[flow] nearest-point dist: mean {nn_dist[v].mean():.1f}mm  p95 {np.percentile(nn_dist[v],95):.1f}", flush=True)
    print(f"[flow] *** SNAP GAP (flow vs nearest-point): mean {snap_gap.mean():.1f}mm  p95 {np.percentile(snap_gap,95):.1f}  max {snap_gap.max():.1f} ***", flush=True)
    print(f"[flow] fraction where nearest-point is >{args.gate:.0f}mm off the flow target: {over.mean()*100:.1f}%  (these are the wrong-surface snaps flow avoids)", flush=True)
    print(f"[flow] flow-match texture error (0-441): mean {tex_err[v].mean():.1f}  p95 {np.percentile(tex_err[v],95):.1f}  (low = flow matches are texture-consistent)", flush=True)
    json.dump({"a": args.a, "b": args.b, "az": args.az, "n": int(v.sum()),
               "scene_flow_mean": float(scene_flow.mean()), "snap_gap_mean": float(snap_gap.mean()),
               "snap_gap_p95": float(np.percentile(snap_gap, 95)), "snap_over_gate_pct": float(over.mean() * 100),
               "tex_err_mean": float(tex_err[v].mean())}, open(os.path.join(args.out, "flow_stats.json"), "w"))
    r.delete()
    print("[flow] DONE", flush=True)


if __name__ == "__main__":
    main()
