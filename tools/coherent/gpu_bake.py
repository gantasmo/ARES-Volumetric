"""GPU coherent-bake Pass 2 (NVIDIA Warp). Same math as lib.mjs bakeFrame, on the GPU:
per occupied template texel -> deformed 3D point -> nearest point on the SOURCE mesh (Warp BVH) ->
source UV -> bilinear-sample the source atlas -> write. ~100x the CPU per-texel loop, 2048 unchanged.

Inputs mirror one coherent run: template OBJ (fixed topology+UVs), positions.bin (deformed template
positions per frame, frame 0 = template), source frames dir (mesh/atlas per GLOBAL frame). Emits the
same honest [PROGRESS] lines as the Node pipeline.

  python gpu_bake.py --template <obj> --positions <bin> --src <dir> --first <globalFrame> --out <dir>
                     [--size 2048] [--pad 5] [--dilate 3]
"""
import sys, os, json, time, argparse, struct
import numpy as np
import warp as wp
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--template", required=True)
ap.add_argument("--positions", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--first", type=int, required=True)   # global frame # of the run's template (frame 0)
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=0)        # 0 = use source atlas native size
ap.add_argument("--pad", type=int, default=5)
ap.add_argument("--dilate", type=int, default=2)   # matches coherent-clip.mjs DILATE_RADIUS
ap.add_argument("--stage-total", type=int, default=0) # for global progress across runs (optional)
ap.add_argument("--stage-done", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
DEV = "cuda:0"
wp.init()

def parse_obj(path):
    """positions (N,3) f32, uvs (N,2) f32 V-flipped to top-left (matches parseObj), idx (M,) int32.
    Assumes 1:1 v/vt (the Microsoft-volcap + our writeObjText outputs — faces are a/a)."""
    pos, uv, idx = [], [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                _, x, y, z = ln.split()[:4]; pos.append((float(x), float(y), float(z)))
            elif ln.startswith("vt "):
                a = ln.split(); u, v = float(a[1]), float(a[2]); uv.append((u, 1.0 - v))  # flip V -> top-left
            elif ln.startswith("f "):
                for tok in ln.split()[1:4]:
                    idx.append(int(tok.split("/")[0]) - 1)
    return (np.array(pos, np.float32), np.array(uv, np.float32) if uv else None, np.array(idx, np.int32))

def load_positions_bin(path):
    with open(path, "rb") as f:
        nF = struct.unpack("<I", f.read(4))[0]; vc = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(nF * vc * 3 * 4), np.float32).reshape(nF, vc, 3)
    return data

def rasterize(uvs, idx, W, H):
    """triId (H*W,), and per-vertex barycentric weights ba/bb/bc (H*W,) for the covering triangle
    (a,b,c). Numpy: per-triangle bbox + edge-function barycentric fill (interiors; slivers <1px are
    gutter-filled later, same net effect as the CPU raster)."""
    triId = np.full(H * W, -1, np.int32)
    ba = np.zeros(H * W, np.float32); bb = np.zeros(H * W, np.float32); bc = np.zeros(H * W, np.float32)
    P = uvs.copy(); P[:, 0] *= W; P[:, 1] *= H
    ntri = len(idx) // 3
    for t in range(ntri):
        a, b, c = idx[3*t], idx[3*t+1], idx[3*t+2]
        ax, ay = P[a]; bx, by = P[b]; cx, cy = P[c]
        minx = max(0, int(np.floor(min(ax, bx, cx)))); maxx = min(W-1, int(np.ceil(max(ax, bx, cx))))
        miny = max(0, int(np.floor(min(ay, by, cy)))); maxy = min(H-1, int(np.ceil(max(ay, by, cy))))
        if maxx < minx or maxy < miny: continue
        area = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
        if area == 0: continue
        xs = np.arange(minx, maxx+1); ys = np.arange(miny, maxy+1)
        gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
        s = 1.0 if area > 0 else -1.0
        w0 = ((bx-ax)*(gy-ay) - (by-ay)*(gx-ax)) * s   # -> weight c
        w1 = ((cx-bx)*(gy-by) - (cy-by)*(gx-bx)) * s   # -> weight a
        w2 = ((ax-cx)*(gy-cy) - (ay-cy)*(gx-cx)) * s   # -> weight b
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any(): continue
        ssum = (w0 + w1 + w2); ssum[ssum == 0] = 1.0
        wa = (w1/ssum)[inside]; wb = (w2/ssum)[inside]; wc = (w0/ssum)[inside]
        px = gx[inside].astype(np.int32); py = gy[inside].astype(np.int32)
        flat = py * W + px
        triId[flat] = t; ba[flat] = wa; bb[flat] = wb; bc[flat] = wc
    return triId, ba, bb, bc

# ---------------- Warp kernel: query nearest source surface + bilinear-sample its atlas -----------
@wp.kernel
def bake_kernel(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3),
                sidx: wp.array(dtype=int), suv: wp.array(dtype=wp.vec2),
                atlas: wp.array(dtype=float), aw_: int, ah_: int,
                out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    p = pts[i]
    q = wp.mesh_query_point(mesh, p, 1.0e7)
    if not q.result:
        out[i] = wp.vec3(0.0, 0.0, 0.0); return
    f = q.face
    a = sidx[f*3+0]; b = sidx[f*3+1]; c = sidx[f*3+2]
    wa = q.u; wb = q.v; wc = 1.0 - q.u - q.v
    uv = suv[a]*wa + suv[b]*wb + suv[c]*wc
    # bilinear sample (matches lib.mjs bilinearSample: pixel centers, clamp)
    fx = uv[0]*float(aw_) - 0.5; fy = uv[1]*float(ah_) - 0.5
    if fx < 0.0: fx = 0.0
    if fy < 0.0: fy = 0.0
    x0 = wp.min(aw_-1, int(fx)); y0 = wp.min(ah_-1, int(fy))
    x1 = wp.min(aw_-1, x0+1); y1 = wp.min(ah_-1, y0+1)
    tx = fx - float(x0); ty = fy - float(y0)
    for ch in range(3):
        i00 = float(atlas[(y0*aw_+x0)*4+ch]); i10 = float(atlas[(y0*aw_+x1)*4+ch])
        i01 = float(atlas[(y1*aw_+x0)*4+ch]); i11 = float(atlas[(y1*aw_+x1)*4+ch])
        top = i00*(1.0-tx) + i10*tx; bot = i01*(1.0-tx) + i11*tx
        out[i][ch] = top*(1.0-ty) + bot*ty

def gutter_map(occ2d, radius):
    """nearest occupied pixel for each dilated ring pixel (BFS), mirrors buildGutterMap."""
    from collections import deque
    H, W = occ2d.shape
    nearest = np.full(H*W, -1, np.int32); dist = np.full(H*W, 1e9, np.float32)
    dq = deque()
    ys, xs = np.nonzero(occ2d)
    for y, x in zip(ys, xs):
        p = y*W+x; nearest[p] = p; dist[p] = 0; dq.append(p)
    r2 = radius*radius
    while dq:
        p = dq.popleft(); x = p % W; y = p // W; src = nearest[p]; sx = src % W; sy = src // W
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                nx, ny = x+dx, y+dy
                if nx<0 or nx>=W or ny<0 or ny>=H: continue
                d2 = (nx-sx)**2 + (ny-sy)**2
                if d2 > r2: continue
                np_ = ny*W+nx
                if d2 < dist[np_]:
                    dist[np_] = d2; nearest[np_] = src; dq.append(np_)
    return nearest

def write_obj(path, pos, uv, idx):
    lines = []
    for p in pos: lines.append(f"v {p[0]} {p[1]} {p[2]}")
    for u in uv: lines.append(f"vt {u[0]} {1.0-u[1]}")   # un-flip V on write (matches writeObjText)
    for t in range(len(idx)//3):
        a,b,c = idx[3*t]+1, idx[3*t+1]+1, idx[3*t+2]+1
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    open(path, "w").write("\n".join(lines) + "\n")

def emit(stage, done, total, t0):
    el = time.time()-t0; rate = done/el if done>0 else 0; eta = (total-done)/rate if rate>0 else 0
    print(f"[PROGRESS] {json.dumps({'stage':stage,'done':int(done),'total':int(total),'pct':round(100*done/total) if total else 0,'elapsedS':round(el),'etaS':round(eta),'fps':round(rate,3)})}", flush=True)

# ------------------------------------- run -------------------------------------
t_pos, t_uv, t_idx = parse_obj(args.template)
pos = load_positions_bin(args.positions)
nF = pos.shape[0]
pad = args.pad
def fname(n): return os.path.join(args.src, f"mesh-f{n:0{pad}d}.obj")
def aname(n): return os.path.join(args.src, f"atlas-f{n:0{pad}d}.png")

# atlas size: source native unless overridden
probe = Image.open(aname(args.first)); SW, SH = probe.size
W = H = args.size if args.size else SW
print(f"[gpu_bake] template verts={len(t_pos)} tris={len(t_idx)//3} | frames={nF} | atlas {W}x{H} (src {SW}x{SH})", flush=True)

triId, ba, bb, bc = rasterize(t_uv, t_idx, W, H)
occ = triId >= 0
occ_idx = np.nonzero(occ)[0]
K = len(occ_idx)
ta = t_idx[triId[occ_idx]*3+0]; tb = t_idx[triId[occ_idx]*3+1]; tc = t_idx[triId[occ_idx]*3+2]
baK = ba[occ_idx][:,None]; bbK = bb[occ_idx][:,None]; bcK = bc[occ_idx][:,None]
occ2d = occ.reshape(H, W)
# gutter: for each pixel within `dilate` of an occupied pixel, its nearest occupied source; the
# ring (dilated-but-not-occupied) copies that color so atlas seams don't crack. Radius = CPU's
# DILATE_RADIUS so the filled ring matches coherent-clip.mjs exactly.
gut = gutter_map(occ2d, args.dilate)
dil_ring = (gut >= 0) & (~occ)

t0 = time.time()
gtot = args.stage_total if args.stage_total else (nF-1)
gdone = args.stage_done
for f in range(nF):
    g = args.first + f
    if f == 0:
        # template frame: copy source atlas verbatim (+ resize if baking at a different size) + template OBJ
        im = Image.open(aname(g))
        if (W, H) != (SW, SH): im = im.resize((W, H))
        im.save(os.path.join(args.out, f"atlas-f{g:0{pad}d}.png"))
        write_obj(os.path.join(args.out, f"mesh-f{g:0{pad}d}.obj"), pos[0], t_uv, t_idx)
        continue
    s_pos, s_uv, s_idx = parse_obj(fname(g))
    rgba = np.asarray(Image.open(aname(g)).convert("RGBA"), np.float32)   # (SH,SW,4)
    ah, aw = rgba.shape[0], rgba.shape[1]
    deformed = pos[f]
    ptsK = (baK*deformed[ta] + bbK*deformed[tb] + bcK*deformed[tc]).astype(np.float32)  # (K,3)
    mesh = wp.Mesh(points=wp.array(s_pos, dtype=wp.vec3, device=DEV), indices=wp.array(s_idx, dtype=int, device=DEV))
    d_pts = wp.array(ptsK, dtype=wp.vec3, device=DEV)
    d_sidx = wp.array(s_idx, dtype=int, device=DEV)
    d_suv = wp.array(s_uv, dtype=wp.vec2, device=DEV)
    d_atlas = wp.array(rgba.reshape(-1), dtype=float, device=DEV)
    d_out = wp.zeros(K, dtype=wp.vec3, device=DEV)
    wp.launch(bake_kernel, dim=K, inputs=[mesh.id, d_pts, d_sidx, d_suv, d_atlas, aw, ah, d_out], device=DEV)
    wp.synchronize()
    colsK = d_out.numpy()   # (K,3) float RGB
    out_atlas = np.zeros((H*W, 4), np.uint8)
    out_atlas[occ_idx, :3] = np.clip(np.round(colsK), 0, 255).astype(np.uint8)
    out_atlas[occ_idx, 3] = 255
    # gutter bleed: ring pixels copy their nearest occupied pixel's color
    ring = np.nonzero(dil_ring)[0]
    src = gut[ring]
    out_atlas[ring] = out_atlas[src]
    Image.fromarray(out_atlas.reshape(H, W, 4)).save(os.path.join(args.out, f"atlas-f{g:0{pad}d}.png"))  # 4ch -> RGBA inferred
    write_obj(os.path.join(args.out, f"mesh-f{g:0{pad}d}.obj"), deformed, t_uv, t_idx)
    gdone += 1
    emit("bake-gpu", gdone, gtot, t0)
print(f"[gpu_bake] DONE {nF-1} frames in {time.time()-t0:.1f}s ({(time.time()-t0)/max(1,nF-1):.2f}s/frame)", flush=True)
