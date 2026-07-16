#!/usr/bin/env python3
"""
Multiview SAM-3D-Body capture (pod-side) — the motion-robust registration fix.

WHY: single-view SAM-3D-Body is depth-ambiguous — it nails the pose in the
image plane but GUESSES depth + focal, and re-rolls that guess every frame → 31mm/frame joint jitter and a
noisy metric scale that broke the naive skeleton prior. But we already HAVE the 3D, so we can render the
SAME frame from many KNOWN cameras and triangulate. Multiview pins the depth DOF (view A's ray ambiguity is
view B's in-plane certainty), gives metric scale for free, and — crucially — fuses SPATIALLY (same instant,
many views) so it removes jitter with ZERO motion blur, unlike temporal smoothing.

KEY TRICK: the render cameras are defined in the MESH's own coordinate space (ARES millimetres). We save
each view's 3x4 world→pixel projection P = K[R|t]. Triangulating the 2D keypoints with those P matrices
yields joints DIRECTLY in ARES space — no SAM-3D→ARES calibration, no scale guess. Calibration + metric
scale both fall out of the geometry.

We use ONLY each view's `pred_keypoints_2d` (image pixels) + our known P. The model's own 3D / cam_t /
focal (its monocular guess) are discarded. Fusion (DLT triangulation, robust) runs LOCALLY afterward, so
the fusion method can be tuned without re-billing the pod.

Deps on the pod (per the 2026-07-13 validation report, all resolved without downgrading the ML stack):
  pip install "setuptools<81" PyOpenGL==3.1.7 pyrender trimesh pillow numpy
  (SAM-3D-Body repo + weights already set up as in tools/sam3d-body-colab.ipynb; MoGe NOT needed here.)

Usage on pod:
  PYOPENGL_PLATFORM=egl python mv_render_infer.py --manifest jobs.json --repo <REPO_DIR> --hf <HF_REPO_ID> \
      --frames-dir frames --out out --views 8 --elev 10 --res 1024

`jobs.json` = {"frames":[121,140,141,...]}; frames-dir holds mesh-fNNNNN.obj + atlas-fNNNNN.png.
Output: out/fNNNNN_mv.json = {frame, views:[{az,el,P(3x4),bbox,keypoints_2d,detected}], meshCentroid, meshHeight}.
"""
import os, sys, json, argparse, math
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np


# ----------------------------- mesh + camera -----------------------------
def load_mesh(obj_path, atlas_path):
    import trimesh
    from PIL import Image
    m = trimesh.load(obj_path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    # texture if UVs are present, else a neutral shaded material (detection works on shape/shading too)
    try:
        uv = getattr(m.visual, "uv", None)
        if uv is not None and atlas_path and os.path.exists(atlas_path):
            img = Image.open(atlas_path).convert("RGB")
            m.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)
    except Exception as e:
        print(f"[mv] texture apply failed ({e.__class__.__name__}); rendering shaded gray", flush=True)
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """OpenGL camera-to-world pose (pyrender convention: camera looks down -Z, +Y up)."""
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= (np.linalg.norm(f) + 1e-12)          # forward (toward target)
    s = np.cross(f, up); s /= (np.linalg.norm(s) + 1e-12)       # right
    u = np.cross(s, f)
    M = np.eye(4)
    M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye   # columns: right, up, +Z(=-forward)
    return M


def projection_P(cam_pose, yfov, W, H):
    """3x4 world→pixel projection matching the pyrender render, CV/image convention (x right, y DOWN).
       SAM-3D keypoints_2d are in this image frame, so triangulating with this P is consistent."""
    fy = (H / 2.0) / math.tan(yfov / 2.0)
    fx = fy                                   # square pixels; pyrender uses yfov + aspect=W/H
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    w2c_gl = np.linalg.inv(cam_pose)          # world -> camera (OpenGL: +Y up, -Z forward)
    gl2cv = np.diag([1.0, -1.0, -1.0])        # OpenGL cam -> CV cam (+Y down, +Z forward)
    R = gl2cv @ w2c_gl[:3, :3]
    t = gl2cv @ w2c_gl[:3, 3]
    return K @ np.hstack([R, t.reshape(3, 1)])


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo", required=True, help="SAM-3D-Body REPO_DIR (has notebook/utils.py)")
    ap.add_argument("--hf", required=True, help="HF_REPO_ID for setup_sam_3d_body")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--elev", type=float, default=10.0, help="camera elevation degrees")
    ap.add_argument("--res", type=int, default=1024, help="render height px")
    ap.add_argument("--margin", type=float, default=1.25, help="frame-fill margin (>1 = zoom out)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import pyrender, trimesh
    sys.path.insert(0, args.repo)
    from notebook.utils import setup_sam_3d_body
    from PIL import Image

    print("[mv] building estimator (vitdet, no MoGe)…", flush=True)
    estimator = setup_sam_3d_body(hf_repo_id=args.hf, detector_name="vitdet", fov_name="")

    frames = json.load(open(args.manifest))["frames"]
    yfov = math.radians(50.0)
    W = int(args.res * 0.75); H = args.res                      # portrait frame suits a standing human
    az_list = [i * 360.0 / args.views for i in range(args.views)]
    el = math.radians(args.elev)
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    tmp_png = os.path.join(args.out, "_view.png")

    for fi in frames:
        stub = f"f{fi:05d}"
        obj = os.path.join(args.frames_dir, f"mesh-{stub}.obj")
        atlas = os.path.join(args.frames_dir, f"atlas-{stub}.png")
        if not os.path.exists(obj):
            print(f"[mv] {stub}: missing {obj}, skip", flush=True); continue
        mesh_t = load_mesh(obj, atlas)
        ctr = mesh_t.bounds.mean(axis=0)
        height = float(mesh_t.bounds[1][1] - mesh_t.bounds[0][1])
        dist = (height / 2.0) / math.tan(yfov / 2.0) * args.margin

        mesh = pyrender.Mesh.from_trimesh(mesh_t, smooth=False)
        views = []
        for az_deg in az_list:
            az = math.radians(az_deg)
            eye = ctr + np.array([dist * math.cos(el) * math.sin(az),
                                  dist * math.sin(el),
                                  dist * math.cos(el) * math.cos(az)])
            cam_pose = look_at(eye, ctr)
            scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.35, 0.35, 0.35])
            scene.add(mesh)
            cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W / H)
            scene.add(cam, pose=cam_pose)
            # two-point lighting rigidly attached to the camera so every view is evenly lit
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=cam_pose)
            color, _ = r.render(scene)
            Image.fromarray(color[:, :, :3]).save(tmp_png)

            P = projection_P(cam_pose, yfov, W, H)
            try:
                outs = estimator.process_one_image(tmp_png)
            except Exception as e:
                print(f"[mv] {stub} az{az_deg:.0f}: inference error {e.__class__.__name__}", flush=True); outs = []
            if outs:
                p = max(outs, key=lambda o: float(np.asarray(o.get("bbox", [0, 0, 1, 1]))[2:].prod()))
                kp = np.asarray(p["pred_keypoints_2d"], float).tolist()
                bbox = np.asarray(p.get("bbox", [0, 0, 0, 0]), float).tolist()
                views.append({"az": az_deg, "el": args.elev, "P": P.tolist(), "keypoints_2d": kp, "bbox": bbox, "detected": True})
                print(f"[mv] {stub} az{az_deg:.0f}: {len(kp)} kp", flush=True)
            else:
                views.append({"az": az_deg, "el": args.elev, "P": P.tolist(), "keypoints_2d": None, "bbox": None, "detected": False})
                print(f"[mv] {stub} az{az_deg:.0f}: NO DETECT", flush=True)

        rec = {"frame": fi, "res": [W, H], "yfov": yfov, "meshCentroid": ctr.tolist(), "meshHeight": height, "views": views}
        json.dump(rec, open(os.path.join(args.out, f"{stub}_mv.json"), "w"))
        ndet = sum(1 for v in views if v["detected"])
        print(f"[mv] {stub}: {ndet}/{len(views)} views detected → {stub}_mv.json", flush=True)

    r.delete()
    print("[mv] DONE", flush=True)


if __name__ == "__main__":
    main()
