#!/usr/bin/env python
# ARES volumetric worker: one process per CUDA device, spawned by depth_volume.py for the two GPU
# phases a `volumetric` depth job adds after its mask and depth passes.
#
#   geometry  MoGe-2 (Ruicheng/moge-2-vitl-normal) on the UNMASKED frame: metric depth, camera-space
#             normals and normalized intrinsics per frame. Scene context is what gives the scale,
#             which is why this model never sees the black background the depth pass sees.
#   body      SAM 3D Body (MHR, 18,439 vertices) per frame, from the box of the subject mask: the
#             completion mesh for the side of the subject the camera never saw.
#
# WHY A PROCESS PER GPU. SAM 3D Body's process_one_image moves every batch to the literal device
# "cuda" (sam_3d_body_estimator.py, recursive_to(batch, "cuda")), so one process can drive one card
# only. The job therefore starts one of these per card with CUDA_VISIBLE_DEVICES=k, splits the clip
# into contiguous shards, and every worker writes its frames POSITIONALLY into files the job has
# already preallocated at full size. Two workers never touch the same byte range.
#
# FRAME PARITY IS A LAW. The ffmpeg argv arrives from the job, built by depth.py's _decode_args (the
# chain the depth pass used), at the size this phase needs. A worker reads frames 0..end and
# processes start..end; the last shard reads to the end of the stream and fails when the count
# differs from the run's frame count by even one.
#
# MEASURED ON THIS MACHINE (2026-09-19, 2x RTX 2080 Ti 11 GB, sm_75, torch 2.6.0+cu124):
#   MoGe-2 vitl-normal   1280x720 frames, 20 frames warm after a throwaway forward: fp16 autocast
#                        193 ms/frame, 2978 MiB peak; fp32 514 ms/frame, 2312 MiB peak; metric depth
#                        differs by 0.05 % median (0.31 % p99), valid pixels agree on 99.98 %. An
#                        earlier probe that ran fp16 FIRST paid cuDNN's algorithm search inside it and
#                        read fp16 as the slower one. Default fp16 (upstream's own infer default).
#   SAM 3D Body          1545 ms/frame warm at 1920x1080 with the box and intrinsics supplied (no
#                        detector, no FOV model), 3470 MiB peak, no NaN, FP16_TYPE float16 (bf16 runs
#                        6x slower on Turing, main.py _best_dtype). inference_type "body" runs 403
#                        ms/frame against 1482 for "full" on the same 20 frames, but moves the whole
#                        mesh by up to 204 mm on 8 of them (the hand pass re-prompts the body
#                        decoder), so "full" is the default
#
# PROTOCOL. The job writes one JSON spec on stdin. Every machine-readable line on stdout is
#   ARES-VOLUME {"ev": "<event>", ...}
# with events load, frame, check, done and error. What the libraries print to stdout (sam_3d_body
# prints three lines per frame) is dropped; stderr carries warnings and tracebacks, and the job keeps
# its tail for error reports.
#
#   python volume_worker.py < spec.json        run one shard
#   python volume_worker.py --probe            import moge and sam_3d_body, print one JSON line

import json
import os
import subprocess
import sys
import threading
import time

EMIT = sys.stdout
# Library prints must never interleave with the protocol lines, and sam_3d_body prints three lines
# per frame, which would push a traceback out of the stderr tail the job reports. They are dropped;
# warnings and tracebacks still reach stderr.
sys.stdout = open(os.devnull, "w", encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ARES_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SAM3D_BODY_CODE = os.environ.get("SAM3D_BODY_CODE_DIR") or os.path.join(ARES_ROOT, "tools", "ext", "sam-3d-body")
BODY_VERTICES = 18439
BODY_FACES = 36874
# The body mesh is discarded when any vertex sits closer to the camera plane than this: a mesh
# through or behind the camera is a failed fit, never a pose.
BODY_MIN_Z = 0.05


def emit(ev: str, **fields) -> None:
    EMIT.write("ARES-VOLUME " + json.dumps({"ev": ev, **fields}, separators=(",", ":")) + "\n")
    EMIT.flush()


# ------------------------------------------------------------------ decode ----


def _no_window() -> dict:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags} if flags else {}


def _read_exact(stream, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


class Decoder:
    """The job's ffmpeg argv, one rgb24 frame at a time. stderr drained on its own thread so a
    chatty source cannot deadlock the pipe."""

    def __init__(self, args: list[str], width: int, height: int):
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.frame_bytes * 2, **_no_window())
        self._err: list[bytes] = []
        proc = self.proc
        threading.Thread(target=lambda: self._err.append(proc.stderr.read()), daemon=True).start()

    def read(self):
        import numpy as np
        raw = _read_exact(self.proc.stdout, self.frame_bytes)
        if len(raw) < self.frame_bytes:
            return None
        return np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3)

    def close(self) -> tuple[int | None, str]:
        try:
            self.proc.stdout.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        err = (self._err[0] or b"").decode("utf-8", "replace").strip() if self._err else ""
        return self.proc.returncode, err


def frames_of_shard(spec: dict):
    """Yields (index, rgb uint8 HxWx3) for start..end-1 and enforces parity when the stream ends."""
    dec = spec["decode"]
    start, end, total = int(spec["start"]), int(spec["end"]), int(spec["frames"])
    last = end == total
    d = Decoder(dec["args"], int(dec["width"]), int(dec["height"]))
    n, early = 0, False
    try:
        while True:
            f = d.read()
            if f is None:
                break
            if n >= end and not last:
                early = True                # a truncated chain never gets here; a guard, not a path
                break
            if start <= n < end:
                yield n, f
            n += 1
    finally:
        code, err = d.close()
    tail = f" (ffmpeg exit {code}{': ' + err[-300:] if err else ''})"
    if last and n != total:
        raise RuntimeError(f"frame parity: the decode produced {n} frames and the run has {total}{tail}")
    if not last and n < end:
        raise RuntimeError(f"frame parity: the decode ended at frame {n}, the shard ends at {end}{tail}")
    if not early and code not in (0, None):
        raise RuntimeError(f"ffmpeg exited {code}: {err[-300:] or 'no stderr'}")


# ----------------------------------------------------------------- geometry ----


def _to_map(depth, normal, valid, H: int, W: int):
    """(Hd, Wd) MoGe outputs -> (H, W) map-size tensors. Downscaling averages over VALID pixels only
    (area weights), so the black band MoGe masks off never drags an edge pixel toward zero; a map
    pixel less than half covered by valid source pixels is invalid. Upscaling is nearest, on pixel
    centres. Normals are re-normalized after averaging."""
    import torch
    import torch.nn.functional as F

    Hd, Wd = depth.shape
    w = valid.float()[None, None]
    d = torch.where(valid, depth, torch.zeros_like(depth))[None, None]
    n = (normal * valid[..., None].float()).permute(2, 0, 1)[None]
    if H <= Hd and W <= Wd:
        a = F.interpolate(w, size=(H, W), mode="area")
        dm = F.interpolate(d, size=(H, W), mode="area") / a.clamp_min(1e-6)
        nm = F.interpolate(n, size=(H, W), mode="area")
        ok = a >= 0.5
    else:
        a = F.interpolate(w, size=(H, W), mode="nearest-exact")
        dm = F.interpolate(d, size=(H, W), mode="nearest-exact")
        nm = F.interpolate(n, size=(H, W), mode="nearest-exact")
        ok = a > 0.5
    norm = nm.norm(dim=1, keepdim=True)
    ok_n = ok & (norm > 1e-6)
    depth_out = torch.where(ok, dm, torch.zeros_like(dm))[0, 0]
    normal_out = torch.where(ok_n, nm / norm.clamp_min(1e-6), torch.zeros_like(nm))[0].permute(1, 2, 0)
    n8 = (normal_out * 127.0).round().clamp(-127, 127).to(torch.int8)
    return depth_out, n8, ok


def run_geometry(spec: dict) -> None:
    import numpy as np
    import torch
    from moge.model.v2 import MoGeModel

    run, H, W = spec["run"], int(spec["mapHeight"]), int(spec["mapWidth"])
    g = spec["geometry"]
    fp16 = bool(g.get("fp16", False))
    Wd, Hd = int(spec["decode"]["width"]), int(spec["decode"]["height"])
    t0 = time.time()
    model = MoGeModel.from_pretrained(g["weights"]).to("cuda").eval()
    with torch.inference_mode():
        model.infer(torch.zeros((3, Hd, Wd), dtype=torch.float32, device="cuda"), use_fp16=fp16)
    torch.cuda.synchronize()
    emit("load", ms=round((time.time() - t0) * 1000), device=torch.cuda.get_device_name(0),
         gpu=spec.get("gpu"), decode=[Wd, Hd], fp16=fp16)

    fm = open(os.path.join(run, "metric.f32"), "r+b")
    fn = open(os.path.join(run, "normals.i8"), "r+b")
    fk = open(os.path.join(run, "intrinsics.f32"), "r+b")
    plane_m, plane_n = H * W * 4, H * W * 3
    done, infer_ms, t_run = 0, 0.0, time.time()
    try:
        for i, frame in frames_of_shard(spec):
            torch.cuda.synchronize()
            t = time.time()
            with torch.inference_mode():
                x = torch.from_numpy(frame).to("cuda").permute(2, 0, 1).float().div_(255.0)
                out = model.infer(x, use_fp16=fp16)
                depth, normal, K = out["depth"].float(), out["normal"].float(), out["intrinsics"].float()
                valid = out["mask"].bool() & torch.isfinite(depth) & (depth > 0)
                dmap, nmap, ok = _to_map(depth, normal, valid, H, W)
                k = K.cpu().numpy()
                dm = dmap.cpu().numpy()
                nm = nmap.cpu().numpy()
                cover = float(ok.float().mean())
            torch.cuda.synchronize()
            ms = (time.time() - t) * 1000.0
            fm.seek(i * plane_m)
            fm.write(np.ascontiguousarray(dm, dtype="<f4").tobytes())
            fn.seek(i * plane_n)
            fn.write(np.ascontiguousarray(nm, dtype=np.int8).tobytes())
            row = np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype="<f4")
            if not np.isfinite(row).all() or row[0] <= 0 or row[1] <= 0:
                row[:] = np.nan               # the job's median skips it; a whole clip of NaN fails there
            fk.seek(i * 16)
            fk.write(row.tobytes())
            done += 1
            infer_ms += ms
            emit("frame", i=i, ms=round(ms, 1), valid=round(cover, 4))
    finally:
        fm.close()
        fn.close()
        fk.close()
    emit("done", frames=done, inferMs=round(infer_ms, 1), wallMs=round((time.time() - t_run) * 1000),
         peakMiB=round(torch.cuda.max_memory_allocated() / 2**20),
         reservedMiB=round(torch.cuda.max_memory_reserved() / 2**20))


# --------------------------------------------------------------------- body ----


def _local_torch_hub() -> None:
    """DINOv3's code comes from torch.hub.load("facebookresearch/dinov3", source="github"), which
    asks github.com for the default branch on EVERY load (torch/hub.py _parse_repo_info) before it
    looks at its own cache. With the repo already in the hub cache the same entry point is loaded
    with source="local": no network, and the run works offline."""
    import torch

    local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov3_main")
    if not os.path.isfile(os.path.join(local, "hubconf.py")):
        return
    orig = torch.hub.load

    def load(repo_or_dir, model, *args, source="github", **kwargs):
        if repo_or_dir == "facebookresearch/dinov3" and source == "github":
            kwargs.pop("trust_repo", None)
            kwargs.pop("force_reload", None)
            return orig(local, model, *args, source="local", **kwargs)
        return orig(repo_or_dir, model, *args, source=source, **kwargs)

    torch.hub.load = load


def load_body_model(checkpoint: str, mhr: str):
    """load_sam_3d_body with the config's FP16_TYPE switched to float16. The checkpoint's
    model_config.yaml says bfloat16, which Turing runs 6x slower at the same memory; the override is
    applied to the config object the loader builds, never to a file."""
    if SAM3D_BODY_CODE not in sys.path:
        sys.path.insert(0, SAM3D_BODY_CODE)
    _local_torch_hub()
    from sam_3d_body import build_models as bm
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

    orig = bm.get_config

    def patched(path):
        cfg = orig(path)
        cfg.defrost()
        cfg.TRAIN.FP16_TYPE = "float16"
        cfg.freeze()
        return cfg

    bm.get_config = patched
    try:
        model, cfg = bm.load_sam_3d_body(checkpoint, device="cuda", mhr_path=mhr)
    finally:
        bm.get_config = orig
    return SAM3DBodyEstimator(model, cfg), cfg


def footprint_iou(verts, faces, kmap, mask, box, H: int, W: int) -> float:
    """IoU of the projected vertex footprint against the subject mask inside the padded box, both at
    map size. Projection is the run's convention: u = fx*x/z + cx in map pixels, pixel centres at
    integer + 0.5, so pixel index = floor(u). The footprint is dilated by the median projected edge
    length (1-8 px), which closes the gaps between vertex splats at any subject distance."""
    import cv2
    import numpy as np

    fx, fy, cx, cy = kmap
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    zs = np.maximum(z, 1e-6)
    u = fx * x / zs + cx
    v = fy * y / zs + cy
    ix, iy = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inb = (z > 0) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    fp = np.zeros((H, W), np.uint8)
    fp[iy[inb], ix[inb]] = 1
    e = np.hypot(u[faces[:, 0]] - u[faces[:, 1]], v[faces[:, 0]] - v[faces[:, 1]])
    r = int(np.clip(np.ceil(np.median(e)), 1, 8))
    fp = cv2.dilate(fp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(W, int(np.ceil(box[2]))), min(H, int(np.ceil(box[3])))
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = mask[y0:y1, x0:x1] > 0
    union = int((fp | m).sum())
    return float((fp & m).sum()) / union if union else 0.0


def run_body(spec: dict) -> None:
    import numpy as np
    import torch

    run, H, W, N = spec["run"], int(spec["mapHeight"]), int(spec["mapWidth"]), int(spec["frames"])
    b = spec["body"]
    Wd, Hd = int(spec["decode"]["width"]), int(spec["decode"]["height"])
    V = int(b.get("vertices", BODY_VERTICES))
    inference = b.get("inference", "full")
    fx, fy, cx, cy = (float(t) for t in b["intrinsics"])
    # The model runs with the intrinsics of the image it is fed: normalized K times the decode size.
    # Normalized intrinsics are resolution-independent, so the mesh it returns is already the mesh
    # the MAP intrinsics (normalized K times the map size) project onto the map.
    cam = torch.tensor([[[fx * Wd, 0.0, cx * Wd], [0.0, fy * Hd, cy * Hd], [0.0, 0.0, 1.0]]],
                       dtype=torch.float32)
    kmap = (fx * W, fy * H, cx * W, cy * H)
    sx, sy = Wd / W, Hd / H

    t0 = time.time()
    est, cfg = load_body_model(b["checkpoint"], b["mhr"])
    faces = np.asarray(est.faces).astype(np.int64)
    if faces.shape != (BODY_FACES, 3):
        raise RuntimeError(f"body faces are {list(faces.shape)}, expected [{BODY_FACES}, 3]")
    if b.get("writeFaces"):
        dst = os.path.join(run, "body-faces.u32")
        with open(dst + ".tmp", "wb") as fh:
            fh.write(np.ascontiguousarray(faces, dtype="<u4").tobytes())
        os.replace(dst + ".tmp", dst)
    # Throwaway forward at the decode size: the first call pays cuDNN's algorithm search (6.0 s
    # against 1.6 s warm, measured), which is load cost, not per-frame cost.
    est.process_one_image(np.zeros((Hd, Wd, 3), np.uint8),
                          bboxes=np.array([[Wd * 0.4, Hd * 0.2, Wd * 0.6, Hd * 0.8]], np.float32),
                          cam_int=cam, inference_type=inference)
    torch.cuda.synchronize()
    emit("load", ms=round((time.time() - t0) * 1000), device=torch.cuda.get_device_name(0),
         gpu=spec.get("gpu"), decode=[Wd, Hd], inference=inference,
         fp16=f"{cfg.TRAIN.get('USE_FP16')}:{cfg.TRAIN.get('FP16_TYPE')}")

    boxes = np.fromfile(os.path.join(run, b["boxes"]), dtype="<f4").reshape(N, 4)
    fb = open(os.path.join(run, "body.f32"), "r+b")
    fv = open(os.path.join(run, "body-valid.u8"), "r+b")
    fmask = open(os.path.join(run, "mask.u8"), "rb")
    plane = V * 12
    done, valid_n, infer_ms, t_run = 0, 0, 0.0, time.time()
    iou_sum, iou_n, iou_min, checked = 0.0, 0, 1.0, False
    reasons: dict[str, int] = {}
    try:
        for i, frame in frames_of_shard(spec):
            box = boxes[i]
            ok, why, ms, iou = False, "", 0.0, None
            if not np.isfinite(box).all():
                why = "no-box"
            else:
                bd = np.array([[box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]], np.float32)
                torch.cuda.synchronize()
                t = time.time()
                res = est.process_one_image(frame, bboxes=bd, cam_int=cam, inference_type=inference)
                torch.cuda.synchronize()
                ms = (time.time() - t) * 1000.0
                infer_ms += ms
                if not res:
                    why = "no-output"
                else:
                    verts = np.asarray(res[0]["pred_vertices"], np.float64) + \
                        np.asarray(res[0]["pred_cam_t"], np.float64).reshape(1, 3)
                    if verts.shape != (V, 3):
                        why = "vertex-count"
                    elif not np.isfinite(verts).all():
                        why = "non-finite"
                    elif float(verts[:, 2].min()) < BODY_MIN_Z:
                        why = "behind-camera"
                    else:
                        ok = True
                        fb.seek(i * plane)
                        fb.write(np.ascontiguousarray(verts, dtype="<f4").tobytes())
                        fmask.seek(i * H * W)
                        m = np.frombuffer(fmask.read(H * W), np.uint8).reshape(H, W)
                        iou = footprint_iou(verts, faces, kmap, m, box, H, W)
                        iou_sum, iou_n, iou_min = iou_sum + iou, iou_n + 1, min(iou_min, iou)
                        if not checked:
                            checked = True
                            emit("check", i=i, iou=round(iou, 4))
            if not ok:
                reasons[why] = reasons.get(why, 0) + 1
            fv.seek(i)
            fv.write(b"\x01" if ok else b"\x00")
            done += 1
            valid_n += int(ok)
            # The IoU rides on the frame event too: a worker lost before "done" still counts toward
            # the phase's mean and minimum.
            emit("frame", i=i, ms=round(ms, 1), valid=int(ok), **({"why": why} if why else {}),
                 **({"iou": round(iou, 4)} if iou is not None else {}))
    finally:
        fb.close()
        fv.close()
        fmask.close()
    emit("done", frames=done, valid=valid_n, inferMs=round(infer_ms, 1),
         wallMs=round((time.time() - t_run) * 1000), iouSum=round(iou_sum, 5), iouN=iou_n,
         iouMin=round(iou_min, 4) if iou_n else None, invalid=reasons,
         peakMiB=round(torch.cuda.max_memory_allocated() / 2**20),
         reservedMiB=round(torch.cuda.max_memory_reserved() / 2**20))


# -------------------------------------------------------------------- probe ----


def probe() -> dict:
    """Import both model packages exactly as a shard would. Loads no weights, touches no GPU."""
    out: dict = {}
    try:
        import moge.model.v2  # noqa: F401
        out["moge"] = True
    except Exception as e:
        out["moge"] = False
        out["mogeError"] = f"{type(e).__name__}: {e}"
    try:
        if SAM3D_BODY_CODE not in sys.path:
            sys.path.insert(0, SAM3D_BODY_CODE)
        import sam_3d_body.build_models  # noqa: F401
        import sam_3d_body.sam_3d_body_estimator  # noqa: F401
        out["sam3dBody"] = True
    except Exception as e:
        out["sam3dBody"] = False
        out["sam3dBodyError"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    if "--probe" in sys.argv[1:]:
        emit("probe", **probe())
        return 0
    try:
        spec = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        phase = spec.get("phase")
        if phase == "geometry":
            run_geometry(spec)
        elif phase == "body":
            run_body(spec)
        else:
            raise ValueError(f"unknown phase {phase!r}: geometry or body")
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit("error", message=f"{type(e).__name__}: {e}"[:1000])
        return 1


if __name__ == "__main__":
    sys.exit(main())
