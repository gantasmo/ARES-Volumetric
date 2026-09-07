/**
 * WebGPU renderer for the mesh profile (spec §10.3, §12.3–§12.4).
 *
 * - Positions are pulled and DEQUANTIZED in the vertex shader from a u16 storage
 *   buffer using the GOP AABB — the CPU never touches float positions (§12.4).
 * - The index buffer is uploaded once per GOP (persistent topology, §12.3); only
 *   the position buffer rotates per frame across THREE slots (triple buffering, §10.2).
 * - The texture atlas is a still imported to a sampled texture; the fragment shader
 *   derives a normal from screen-space derivatives for simple lighting.
 *
 * The VideoDecoder→importExternalTexture path (spec §7.1) is the P1.5 texture-video
 * upgrade; P1 ships the still-atlas path (§7.7).
 */
import type { Aabb } from "./quant.js";
import { invert, multiply } from "./camera.js";
import { buildOriginTripod, type GridParams } from "./overlay.js";
import type { DecodedSplat } from "./splat.js";
import type { SplatCamera, SplatParams } from "./renderer-gl2.js";

// Origin tripod: world-space colored lines. Reads only viewProj from the shared uniform (first 64
// bytes); vertex buffer is interleaved position(vec3)+color(vec3).
const LINES_WGSL = /* wgsl */ `
struct GU { viewProj : mat4x4<f32> };
@group(0) @binding(0) var<uniform> u : GU;
struct VO { @builtin(position) clip : vec4<f32>, @location(0) color : vec3<f32> };
@vertex fn vs(@location(0) pos : vec3<f32>, @location(1) col : vec3<f32>) -> VO {
  var o : VO; o.clip = u.viewProj * vec4<f32>(pos, 1.0); o.color = col; return o;
}
@fragment fn fs(in : VO) -> @location(0) vec4<f32> { return vec4<f32>(in.color, 1.0); }
`;

// INFINITE ground grid — a fullscreen triangle that raymarches the y = floorY plane per pixel.
// No geometry, so no extent to run out of and no fixed cell size to alias: the fragment knows its
// exact ground hit point, so cells are sized to the live zoom (overlay.ts gridLod), each line is
// antialiased against its own screen-space derivative, and the whole thing fades out with distance.
// Writes frag_depth from the hit point so the mesh occludes the floor correctly (depth WRITE is off
// — the grid must never occlude the mesh).
const GRID_WGSL = /* wgsl */ `
struct GU {
  invViewProj : mat4x4<f32>,
  viewProj    : mat4x4<f32>,
  params      : vec4<f32>,   // step0 (fine cell), step1 (coarse cell), fineFade, floorY
  fade        : vec4<f32>,   // centerX, centerZ, fadeRadius, unused
};
@group(0) @binding(0) var<uniform> g : GU;

struct VO { @builtin(position) clip : vec4<f32>, @location(0) ndc : vec2<f32> };
struct FO { @location(0) color : vec4<f32>, @builtin(frag_depth) depth : f32 };

// One oversized triangle covers the viewport with no vertex buffer and no clipped diagonal seam.
@vertex fn vs(@builtin(vertex_index) vi : u32) -> VO {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VO;
  o.clip = vec4<f32>(p[vi], 0.0, 1.0);
  o.ndc = p[vi];
  return o;
}

// NDC + clip-space depth -> world. z = 0 is the near plane, z = 1 the far plane (camera.ts builds
// WebGPU-convention [0,1] projections). Works for ortho as well as perspective: a true inverse of
// viewProj carries whichever one is in force.
fn unproj(ndc : vec2<f32>, z : f32) -> vec3<f32> {
  let p = g.invViewProj * vec4<f32>(ndc.x, ndc.y, z, 1.0);
  return p.xyz / p.w;
}

// Antialiased line coverage for a grid of the given cell size, using screen-space derivatives: the
// distance to the nearest cell edge, measured in PIXELS, so every line lands ~1px wide at any zoom.
fn gridAA(xz : vec2<f32>, cell : f32) -> f32 {
  let c = xz / cell;
  let d = max(fwidth(c), vec2<f32>(1e-8, 1e-8));
  let l = abs(fract(c - vec2<f32>(0.5, 0.5)) - vec2<f32>(0.5, 0.5)) / d;
  return 1.0 - min(min(l.x, l.y), 1.0);
}

@fragment fn fs(in : VO) -> FO {
  var o : FO;
  o.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  o.depth = 1.0;

  let n = unproj(in.ndc, 0.0);
  let f = unproj(in.ndc, 1.0);
  let dir = f - n;
  let t = (g.params.w - n.y) / select(dir.y, 1e-9, abs(dir.y) < 1e-9);
  // The ground is behind the camera or past the far plane for this pixel. Clamp t BEFORE deriving
  // the hit point regardless: fwidth reads the neighbouring lanes in the quad, and a discarded lane
  // holding an infinity would poison a surviving neighbour's derivative along the horizon.
  let hitsPlane = t >= 0.0 && t <= 1.0 && abs(dir.y) >= 1e-9;
  let p = n + clamp(t, 0.0, 1.0) * dir;

  let fine   = gridAA(p.xz, g.params.x) * g.params.z;
  let coarse = gridAA(p.xz, g.params.y);
  // The X (z = 0) and Z (x = 0) center lines, so the origin stays findable at any zoom.
  let d = max(fwidth(p.xz), vec2<f32>(1e-8, 1e-8));
  let centerLine = max(1.0 - min(abs(p.z) / d.y, 1.0), 1.0 - min(abs(p.x) / d.x, 1.0));

  var a = max(max(fine, coarse), centerLine);
  let col = mix(vec3<f32>(0.28, 0.28, 0.31), vec3<f32>(0.46, 0.46, 0.50), centerLine);

  // "Endlessly extend / fade into the distance": radial fade around the point being looked at, in
  // ground-plane distance — camera-azimuth independent, so it doesn't swim while you orbit.
  let radial = 1.0 - smoothstep(g.fade.z * 0.45, g.fade.z, length(p.xz - g.fade.xy));
  // Near-edge-on, a plane's cells compress below a pixel and any line grid turns to moiré. Fading
  // as the ray flattens is what an infinite grid must do instead of shimmering.
  let grazing = smoothstep(0.0, 0.06, abs(normalize(dir).y));
  a = a * radial * grazing * 0.9 * select(0.0, 1.0, hitsPlane);
  if (a < 0.002) { discard; }

  let cp = g.viewProj * vec4<f32>(p, 1.0);
  o.depth = clamp(cp.z / cp.w, 0.0, 1.0);
  o.color = vec4<f32>(col, a);
  return o;
}
`;

/** Per-vertex normal unpack, chosen by superblock normal_encoding (0 = legacy i8×4, 1 = oct16). */
const NRM_DECODE: Record<number, string> = {
  0: /* wgsl */ `o.normal = vec3<f32>(sx8(np), sx8(np >> 8u), sx8(np >> 16u)) / 127.0;`,
  1: /* wgsl */ `
  let onx = f32(i32(np << 16u) >> 16u) / 32767.0;
  let ony = f32(i32(np) >> 16u) / 32767.0;
  var on = vec3<f32>(onx, ony, 1.0 - abs(onx) - abs(ony));
  let ot = clamp(-on.z, 0.0, 1.0);
  on.x = on.x + select(ot, -ot, on.x >= 0.0);
  on.y = on.y + select(ot, -ot, on.y >= 0.0);
  o.normal = normalize(on);`,
};

const WGSL = (normalEncoding: number) => /* wgsl */ `
struct Uniforms {
  viewProj : mat4x4<f32>,
  aabbMin  : vec3<f32>,
  invLevels: f32,
  aabbSize : vec3<f32>,
  litMix   : f32,        // 1 = lit (lambert), 0 = unlit — video-textured captures carry baked lighting
  cropMin  : vec3<f32>,   // world-space crop box (mesh editor preview)
  cropOn   : f32,
  cropMax  : vec3<f32>,
  texMix   : f32,        // 1 = textured (shaded), 0 = untextured clay — the viewport shading mode
  shadeMode : f32,       // 0 shaded, 1 normals, 2 uv checker, 3 depth, 4 points
  pointSize : f32,       // pixels (mode 4)
  depthLo   : f32,       // depth view window (view-space distance)
  depthHi   : f32,
  viewport  : vec2<f32>, // pixels (mode 4 quad expansion)
  pad2      : vec2<f32>,
};
@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read> qpos : array<u32>; // 2 u32 / vertex (u16 x,y,z,pad)
@group(0) @binding(2) var<storage, read> quv  : array<u32>; // 1 u32 / vertex (u16 u,v)
@group(0) @binding(3) var samp : sampler;
@group(0) @binding(4) var tex  : texture_2d<f32>;
@group(0) @binding(5) var<storage, read> qnrm : array<u32>; // 1 u32 / vertex (i8 x,y,z,pad snorm)
@group(0) @binding(6) var<uniform> fx : array<vec4<f32>, 8>;   // playback effects (fx.ts)

// ---- playback effects (fx.ts packFx layout: 8 × vec4) ----
fn fxHash(p : vec3<f32>) -> f32 { return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453); }
fn fxNoise(p : vec3<f32>) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(fxHash(i), fxHash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x), mix(fxHash(i + vec3<f32>(0.0, 1.0, 0.0)), fxHash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y);
  let b = mix(mix(fxHash(i + vec3<f32>(0.0, 0.0, 1.0)), fxHash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x), mix(fxHash(i + vec3<f32>(0.0, 1.0, 1.0)), fxHash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}
fn fxWobble(p : vec3<f32>) -> vec3<f32> {
  let amp = fx[4].w;
  if (amp <= 0.0) { return p; }
  let f = fx[5].x; let t = fx[5].y;
  return p + amp * vec3<f32>(sin(p.y * f + t * 3.0), sin(p.z * f * 1.3 + t * 2.2), sin(p.x * f * 0.7 + t * 2.7));
}
// Colour-side effects; alpha 0 = discard. n is a unit normal (meshes) or zero (no rim).
fn fxApply(world : vec3<f32>, n : vec3<f32>, colIn : vec3<f32>) -> vec4<f32> {
  var col = colIn;
  if (fx[1].x > 0.5 && dot(world, fx[0].xyz) + fx[0].w < 0.0) { return vec4<f32>(col, 0.0); }
  let d = fx[1].y;
  if (d > 0.0) {
    let nz = fxNoise(world * fx[1].z);
    if (nz < d) { return vec4<f32>(col, 0.0); }
    col = mix(fx[4].xyz, col, smoothstep(d, d + 0.08, nz));
  }
  if (fx[1].w > 0.0) { col = col * (1.0 - fx[1].w * 0.5 * (0.5 + 0.5 * sin(world.y * fx[5].z + fx[5].y * 4.0))); }
  if (fx[3].w > 0.0 && dot(n, n) > 0.5) { let v = normalize(fx[6].xyz - world); let fr = pow(1.0 - abs(dot(n, v)), 3.0); col = col + fx[3].xyz * fx[3].w * fr; }
  if (fx[2].w > 0.0) { col = mix(col, col * fx[2].xyz, fx[2].w); }
  return vec4<f32>(col, 1.0);
}

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv         : vec2<f32>,
  @location(1) world      : vec3<f32>,
  @location(2) normal     : vec3<f32>,
  @location(3) depth      : f32,
};

fn sx8(v : u32) -> f32 { return f32(i32(v << 24u) >> 24u); }

fn vertexAt(vi : u32) -> VSOut {
  let a = qpos[vi * 2u];
  let b = qpos[vi * 2u + 1u];
  let q = vec3<f32>(f32(a & 0xffffu), f32((a >> 16u) & 0xffffu), f32(b & 0xffffu));
  let world = fxWobble(u.aabbMin + (q * u.invLevels) * u.aabbSize);   // GPU-side dequant (+ wobble fx)
  let uvp = quv[vi];
  let np = qnrm[vi];
  var o : VSOut;
  o.clip = u.viewProj * vec4<f32>(world, 1.0);
  o.uv = vec2<f32>(f32(uvp & 0xffffu), f32((uvp >> 16u) & 0xffffu)) / 65535.0;
  o.world = world;
  o.depth = o.clip.w;
  ${NRM_DECODE[normalEncoding] ?? NRM_DECODE[0]!}
  return o;
}

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut { return vertexAt(vi); }

// Point-cloud view: one screen-aligned quad per vertex (4 strip vertices × vertexCount instances).
@vertex
fn vs_points(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  var o = vertexAt(ii);
  let corner = vec2<f32>(f32((vi & 1u) * 2u) - 1.0, f32((vi >> 1u) * 2u) - 1.0);
  let px = corner * u.pointSize * 0.5;
  o.clip = vec4<f32>(o.clip.xy + px * 2.0 / u.viewport * o.clip.w, o.clip.zw);
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  // Face normal computed unconditionally (dpdx/dpdy require uniform control flow);
  // prefer the smooth interpolated normal, fall back to the face normal if it degenerates.
  let faceN = cross(dpdx(in.world), dpdy(in.world));
  let vertN = in.normal;
  var n = select(faceN, vertN, dot(vertN, vertN) > 0.01);
  n = normalize(n);
  let L = normalize(vec3<f32>(0.35, 0.75, 0.55));
  let diff = abs(dot(n, L));                 // two-sided (winding-agnostic)
  // texMix 0 = untextured "clay" (neutral paper grey, same lighting) so form reads without the
  // atlas; sampled unconditionally to keep derivatives/control flow uniform.
  let albedo = mix(vec3<f32>(0.72, 0.71, 0.68), textureSample(tex, samp, in.uv).rgb, u.texMix);
  // Crop preview (mesh editor): discard LAST so derivatives/samples above stay uniform.
  if (u.cropOn > 0.5 && (any(in.world < u.cropMin) || any(in.world > u.cropMax))) { discard; }
  let lit = mix(1.0, 0.4 + 0.6 * diff, u.litMix);
  var col = albedo * lit;
  let mode = u32(u.shadeMode + 0.5);
  if (mode == 1u) { col = n * 0.5 + vec3<f32>(0.5, 0.5, 0.5); }
  else if (mode == 2u) {
    let ck = (floor(in.uv.x * 32.0) + floor(in.uv.y * 32.0)) % 2.0;
    col = mix(vec3<f32>(0.22, 0.22, 0.24), vec3<f32>(0.82, 0.80, 0.76), ck) * mix(1.0, lit, 0.5);
  } else if (mode == 3u) {
    let d = clamp((in.depth - u.depthLo) / max(1e-6, u.depthHi - u.depthLo), 0.0, 1.0);
    col = vec3<f32>(1.0 - d) * vec3<f32>(0.92, 0.9, 0.86);
  } else if (mode == 4u) { col = albedo; }   // points: unlit, the texture verbatim
  let fxo = fxApply(in.world, n, col);
  if (fxo.a < 0.5) { discard; }
  return vec4<f32>(fxo.rgb, 1.0);
}
`;

/**
 * Gaussian splat pipeline (spec §6.8): one instanced quad per splat, ordered back-to-front by the
 * CPU sorter (splat-sort.ts) through the `order` indirection. The vertex stage dequantizes the
 * centre exactly like the mesh path, rebuilds the 3D covariance R·S·Sᵀ·Rᵀ from the packed scale +
 * quaternion (splat.ts), projects it to a screen-space 2×2 covariance through the view rotation
 * and the projection Jacobian (EWA splatting, Zwicker 2001 / 3DGS), and stretches the quad along
 * the ellipse's eigenvectors to 3σ. The fragment stage evaluates the Gaussian and composites
 * premultiplied "over". Colour is the base 8-bit colour plus the optional SH bands evaluated in
 * the direction from the camera (model space, so SH survive the model transform).
 */
const SPLAT_WGSL = /* wgsl */ `
struct SU {
  view     : mat4x4<f32>,      // view * model
  proj     : mat4x4<f32>,
  aabbMin  : vec3<f32>, invLevels : f32,
  aabbSize : vec3<f32>, shDegree  : f32,
  viewport : vec2<f32>, focal     : vec2<f32>,   // pixels
  camPos   : vec3<f32>, ortho     : f32,         // camera position in model space
  cropMin  : vec3<f32>, cropOn    : f32,
  cropMax  : vec3<f32>, scaleMul  : f32,
  opacityMul : f32, texMix : f32, pad0 : f32, pad1 : f32,
};
@group(0) @binding(0) var<uniform> u : SU;
@group(0) @binding(1) var<storage, read> qpos  : array<u32>;  // 2 u32 / splat (u16 x,y,z,pad)
@group(0) @binding(2) var<storage, read> attr  : array<u32>;  // 3 u32 / splat (splat.ts)
@group(0) @binding(3) var<storage, read> order : array<u32>;  // back-to-front splat ids
@group(0) @binding(4) var<storage, read> sh    : array<u32>;  // shStride/4 u32 per splat (degree >= 1)
@group(0) @binding(5) var<uniform> fx : array<vec4<f32>, 8>;   // playback effects (fx.ts)

struct VO {
  @builtin(position) clip : vec4<f32>,
  @location(0) local : vec2<f32>,
  @location(1) color : vec4<f32>,
  @location(2) world : vec3<f32>,
};

// ---- playback effects (fx.ts packFx layout: 8 × vec4) ----
fn fxHash(p : vec3<f32>) -> f32 { return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453); }
fn fxNoise(p : vec3<f32>) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(fxHash(i), fxHash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x), mix(fxHash(i + vec3<f32>(0.0, 1.0, 0.0)), fxHash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y);
  let b = mix(mix(fxHash(i + vec3<f32>(0.0, 0.0, 1.0)), fxHash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x), mix(fxHash(i + vec3<f32>(0.0, 1.0, 1.0)), fxHash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}
fn fxWobble(p : vec3<f32>) -> vec3<f32> {
  let amp = fx[4].w;
  if (amp <= 0.0) { return p; }
  let f = fx[5].x; let t = fx[5].y;
  return p + amp * vec3<f32>(sin(p.y * f + t * 3.0), sin(p.z * f * 1.3 + t * 2.2), sin(p.x * f * 0.7 + t * 2.7));
}
// Colour-side effects; alpha 0 = discard. n is a unit normal (meshes) or zero (no rim).
fn fxApply(world : vec3<f32>, n : vec3<f32>, colIn : vec3<f32>) -> vec4<f32> {
  var col = colIn;
  if (fx[1].x > 0.5 && dot(world, fx[0].xyz) + fx[0].w < 0.0) { return vec4<f32>(col, 0.0); }
  let d = fx[1].y;
  if (d > 0.0) {
    let nz = fxNoise(world * fx[1].z);
    if (nz < d) { return vec4<f32>(col, 0.0); }
    col = mix(fx[4].xyz, col, smoothstep(d, d + 0.08, nz));
  }
  if (fx[1].w > 0.0) { col = col * (1.0 - fx[1].w * 0.5 * (0.5 + 0.5 * sin(world.y * fx[5].z + fx[5].y * 4.0))); }
  if (fx[3].w > 0.0 && dot(n, n) > 0.5) { let v = normalize(fx[6].xyz - world); let fr = pow(1.0 - abs(dot(n, v)), 3.0); col = col + fx[3].xyz * fx[3].w * fr; }
  if (fx[2].w > 0.0) { col = mix(col, col * fx[2].xyz, fx[2].w); }
  return vec4<f32>(col, 1.0);
}

const SH_C1 : f32 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396);
const SH_C3 = array<f32, 7>(-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435);

fn shByte(base : u32, idx : u32) -> f32 {
  let w = sh[base + (idx >> 2u)];
  return (f32((w >> ((idx & 3u) * 8u)) & 0xffu) - 128.0) / 128.0;
}
fn shCoef(base : u32, k : u32) -> vec3<f32> {
  return vec3<f32>(shByte(base, k * 3u), shByte(base, k * 3u + 1u), shByte(base, k * 3u + 2u));
}
// Higher-order SH (3DGS ordering); stride = u32 words per splat.
fn evalSH(id : u32, stride : u32, d : vec3<f32>, degree : u32) -> vec3<f32> {
  let base = id * stride;
  let x = d.x; let y = d.y; let z = d.z;
  var c = -SH_C1 * y * shCoef(base, 0u) + SH_C1 * z * shCoef(base, 1u) - SH_C1 * x * shCoef(base, 2u);
  if (degree >= 2u) {
    let xx = x * x; let yy = y * y; let zz = z * z; let xy = x * y; let yz = y * z; let xz = x * z;
    c = c + SH_C2[0] * xy * shCoef(base, 3u) + SH_C2[1] * yz * shCoef(base, 4u)
          + SH_C2[2] * (2.0 * zz - xx - yy) * shCoef(base, 5u)
          + SH_C2[3] * xz * shCoef(base, 6u) + SH_C2[4] * (xx - yy) * shCoef(base, 7u);
    if (degree >= 3u) {
      c = c + SH_C3[0] * y * (3.0 * xx - yy) * shCoef(base, 8u)
            + SH_C3[1] * xy * z * shCoef(base, 9u)
            + SH_C3[2] * y * (4.0 * zz - xx - yy) * shCoef(base, 10u)
            + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shCoef(base, 11u)
            + SH_C3[4] * x * (4.0 * zz - xx - yy) * shCoef(base, 12u)
            + SH_C3[5] * z * (xx - yy) * shCoef(base, 13u)
            + SH_C3[6] * x * (xx - 3.0 * yy) * shCoef(base, 14u);
    }
  }
  return c;
}

// SPZ v3 "smallest three" (splat.ts packQuaternion) -> unit quaternion (x, y, z, w).
fn unpackQuat(c : u32) -> vec4<f32> {
  let iL = c >> 30u;
  let m0 = f32((c >> 20u) & 0x1ffu) / 511.0 * 0.70710678118;
  let m1 = f32((c >> 10u) & 0x1ffu) / 511.0 * 0.70710678118;
  let m2 = f32(c & 0x1ffu) / 511.0 * 0.70710678118;
  let v0 = select(m0, -m0, ((c >> 29u) & 1u) == 1u);
  let v1 = select(m1, -m1, ((c >> 19u) & 1u) == 1u);
  let v2 = select(m2, -m2, ((c >> 9u) & 1u) == 1u);
  let big = sqrt(max(0.0, 1.0 - v0 * v0 - v1 * v1 - v2 * v2));
  if (iL == 0u) { return vec4<f32>(big, v0, v1, v2); }
  if (iL == 1u) { return vec4<f32>(v0, big, v1, v2); }
  if (iL == 2u) { return vec4<f32>(v0, v1, big, v2); }
  return vec4<f32>(v0, v1, v2, big);
}
fn quatToMat(q : vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z),       2.0 * (x * z - w * y),
    2.0 * (x * y - w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y),       2.0 * (y * z - w * x),       1.0 - 2.0 * (x * x + y * y));
}

@vertex fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VO {
  var o : VO;
  o.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);   // default: outside the clip volume
  o.local = vec2<f32>(0.0, 0.0);
  o.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  o.world = vec3<f32>(0.0, 0.0, 0.0);
  let id = order[ii];
  let a = qpos[id * 2u];
  let b = qpos[id * 2u + 1u];
  let q = vec3<f32>(f32(a & 0xffffu), f32((a >> 16u) & 0xffffu), f32(b & 0xffffu));
  var pos = fxWobble(u.aabbMin + (q * u.invLevels) * u.aabbSize);      // model space (+ wobble fx)
  if (fx[5].w > 0.0) {                                                   // per-splat jitter (holo-glitch)
    let hid = f32(id) * 0.001; let t = floor(fx[5].y * 12.0);
    pos = pos + (vec3<f32>(fxHash(vec3<f32>(hid, 1.0, t)), fxHash(vec3<f32>(hid, 2.0, t)), fxHash(vec3<f32>(hid, 3.0, t))) - vec3<f32>(0.5, 0.5, 0.5)) * 2.0 * fx[5].w;
  }
  o.world = pos;
  if (u.cropOn > 0.5 && (any(pos < u.cropMin) || any(pos > u.cropMax))) { return o; }
  if (fx[1].x > 0.5 && dot(pos, fx[0].xyz) + fx[0].w < 0.0) { return o; }
  var edgeMix = 0.0;
  if (fx[1].y > 0.0) { let nz = fxNoise(pos * fx[1].z); if (nz < fx[1].y) { return o; } edgeMix = 1.0 - smoothstep(fx[1].y, fx[1].y + 0.08, nz); }
  let cam = u.view * vec4<f32>(pos, 1.0);
  let clip = u.proj * cam;
  if (u.ortho < 0.5 && clip.w <= 1e-6) { return o; }
  let ndc = clip.xyz / clip.w;
  if (any(abs(ndc.xy) > vec2<f32>(1.5, 1.5)) || ndc.z < 0.0 || ndc.z > 1.0) { return o; }

  let w0 = attr[id * 3u];
  let w1 = attr[id * 3u + 1u];
  let w2 = attr[id * 3u + 2u];
  let s = exp(vec3<f32>(f32(w0 & 0xffu), f32((w0 >> 8u) & 0xffu), f32((w0 >> 16u) & 0xffu)) / 16.0 - 10.0) * u.scaleMul * fx[6].w;
  let alpha = f32(w0 >> 24u) / 255.0 * u.opacityMul * fx[7].x;
  let R = quatToMat(unpackQuat(w1));
  let M = R * mat3x3<f32>(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  let cov3 = M * transpose(M);
  let W = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
  var J : mat3x3<f32>;
  if (u.ortho > 0.5) {
    J = mat3x3<f32>(u.focal.x, 0.0, 0.0,  0.0, u.focal.y, 0.0,  0.0, 0.0, 0.0);
  } else {
    let z = cam.z;
    let lim = 1.3 * 0.5 * u.viewport / u.focal;
    let tx = clamp(cam.x / z, -lim.x, lim.x) * z;
    let ty = clamp(cam.y / z, -lim.y, lim.y) * z;
    J = mat3x3<f32>(u.focal.x / z, 0.0, 0.0,  0.0, u.focal.y / z, 0.0,  -u.focal.x * tx / (z * z), -u.focal.y * ty / (z * z), 0.0);
  }
  let T = J * W;
  let cov2 = T * cov3 * transpose(T);
  let cxx = cov2[0][0] + 0.3;
  let cyy = cov2[1][1] + 0.3;
  let cxy = cov2[0][1];
  let mid = 0.5 * (cxx + cyy);
  let rad = sqrt(max(0.0, mid * mid - (cxx * cyy - cxy * cxy)));
  let l1 = mid + rad;
  let l2 = max(mid - rad, 0.1);
  let r1 = 3.0 * sqrt(l1);
  let r2 = 3.0 * sqrt(l2);
  var v1 : vec2<f32>;
  if (abs(cxy) < 1e-6) { v1 = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), cxx >= cyy); }
  else { v1 = normalize(vec2<f32>(cxy, l1 - cxx)); }
  let v2 = vec2<f32>(-v1.y, v1.x);
  let corner = vec2<f32>(f32((vi & 1u) * 2u) - 1.0, f32((vi >> 1u) * 2u) - 1.0);
  let offsetPx = corner.x * r1 * v1 + corner.y * r2 * v2;
  let ndcOff = offsetPx * 2.0 / u.viewport;
  o.clip = vec4<f32>((ndc.xy + ndcOff) * clip.w, ndc.z * clip.w, clip.w);
  o.local = corner * 3.0;
  var col = vec3<f32>(f32(w2 & 0xffu), f32((w2 >> 8u) & 0xffu), f32((w2 >> 16u) & 0xffu)) / 255.0;
  let degree = u32(u.shDegree + 0.5);
  if (degree > 0u) {
    let stride = select(select(12u, 6u, degree == 2u), 3u, degree == 1u);
    col = col + evalSH(id, stride, normalize(pos - u.camPos), degree);
  }
  col = clamp(col, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
  col = mix(vec3<f32>(0.72, 0.71, 0.68), col, u.texMix);
  col = mix(col, fx[4].xyz, edgeMix);                                   // dissolve rim
  if (fx[2].w > 0.0) { col = mix(col, col * fx[2].xyz, fx[2].w); }     // tint
  o.color = vec4<f32>(col, alpha);
  return o;
}

@fragment fn fs(in : VO) -> @location(0) vec4<f32> {
  let d2 = dot(in.local, in.local);
  if (d2 > 9.0) { discard; }
  let a = in.color.a * exp(-0.5 * d2);
  if (a < 1.0 / 255.0) { discard; }
  var col = in.color.rgb;
  if (fx[1].w > 0.0) { col = col * (1.0 - fx[1].w * 0.5 * (0.5 + 0.5 * sin(in.world.y * fx[5].z + fx[5].y * 4.0))); }
  return vec4<f32>(col * a, a);
}
`;

const SLOTS = 3;
/** Storage/index buffer binding sizes must be a multiple of 4 (WebGPU); round up with headroom. */
const roundUp16 = (n: number) => (n + 15) & ~15;

export class WebGPURenderer {
  private ctx: GPUCanvasContext;
  private format: GPUTextureFormat;
  private pipeline!: GPURenderPipeline;
  private uniform: GPUBuffer;
  private sampler: GPUSampler;
  private posBufs: (GPUBuffer | null)[] = [null, null, null];
  private posCap = 0;
  private uvBuf: GPUBuffer | null = null;
  private uvCap = 0;
  private nrmBuf: GPUBuffer | null = null;
  private nrmCap = 0;
  private idxBuf: GPUBuffer | null = null;
  private idxCap = 0;
  private texture: GPUTexture | null = null;
  private texView: GPUTextureView | null = null;
  private texW = 0;
  private texH = 0;
  private depth: GPUTexture | null = null;
  private slot = 0;
  private uni = new Float32Array(40);
  private normalEncoding = 0;
  private shadeMode = 0;
  private pointSize = 2;
  private depthRange: [number, number] = [0, 1];
  private vertexCount = 0;
  private pointsPipeline: GPURenderPipeline | null = null;
  private crop: { min: [number, number, number]; max: [number, number, number] } | null = null;
  // Wireframe (mesh editor): triangles expanded to a line-list (3 edges/tri) + a line-topology pipeline.
  private wireframe = false;
  private texMix = 1;          // viewport shading: 1 = textured, 0 = untextured clay
  private litMix = 1;          // 1 = lit (lambert), 0 = unlit (albedo verbatim)
  private linePipeline: GPURenderPipeline | null = null;
  private lineIdxBuf: GPUBuffer | null = null;
  private lineIdxCap = 0;
  private lineIndexCount = 0;
  private lastIndices: Uint32Array | null = null;
  // Ground grid (analytic, infinite) + origin tripod (line geometry). Default ON — the floor is a
  // reference, not a tool: without it there is no way to tell a capture that sits on the ground from
  // one that floats, and nothing to read scale against.
  private gridOn = true;
  private gridPipeline: GPURenderPipeline | null = null;
  private gridUni: GPUBuffer | null = null;
  private gridBind: GPUBindGroup | null = null;
  private gridData = new Float32Array(40);   // invViewProj(16) + viewProj(16) + params(4) + fade(4)
  private gridParams: GridParams = { step0: 100, step1: 1000, fineFade: 1, floorY: 0, centerX: 0, centerZ: 0, fadeRadius: 1e4, axisLen: 300 };
  // Model transform (import-time orientation fix). Folded into the MESH's viewProj only — the grid
  // and tripod are world reference and must not move with the model.
  private model: Float32Array | null = null;
  private tripodPipeline: GPURenderPipeline | null = null;
  private tripodVbuf: GPUBuffer | null = null;
  private tripodCount = 0;
  private tripodBind: GPUBindGroup | null = null;
  // Gaussian splat profile (spec §6.8): separate storage buffers + pipeline, see SPLAT_WGSL.
  private splatPipeline: GPURenderPipeline | null = null;
  private splatUni: GPUBuffer | null = null;
  private splatUniData = new Float32Array(64);
  private sposBuf: GPUBuffer | null = null; private sposCap = 0;
  private sattrBuf: GPUBuffer | null = null; private sattrCap = 0;
  private sshBuf: GPUBuffer | null = null; private sshCap = 0;
  private sorderBuf: GPUBuffer | null = null; private sorderCap = 0;
  private splatCount = 0;
  private splatShDegree = 0;
  private splatParams: SplatParams = { scaleMul: 1, opacityMul: 1 };
  // Playback effects (fx.ts): one 128-byte uniform block shared by the mesh and splat pipelines.
  private fxUni: GPUBuffer | null = null;
  private fxData = new Float32Array(32);

  private constructor(public readonly device: GPUDevice, canvas: HTMLCanvasElement | OffscreenCanvas) {
    this.ctx = canvas.getContext("webgpu") as unknown as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: "opaque" });
    this.uniform = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" });
    this.buildPipeline();
    // 1x1 white placeholder so the first render works even before the atlas loads.
    this.setTextureFromPixels(new Uint8Array([255, 255, 255, 255]), 1, 1);
    // Default (zero) normals so clips without per-vertex normals still render (shader falls
    // back to a face normal). Real normals overwrite this via uploadNormals().
    [this.nrmBuf, this.nrmCap] = this.growStorage(null, 0, 256);
  }

  static async create(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<WebGPURenderer> {
    if (!navigator.gpu) throw new Error("WebGPU unavailable (no navigator.gpu) — the player selects the WebGL2 fallback (§10.4) before reaching this");
    // powerPreference is silently ignored by Chrome's WebGPU on Windows and logs a warning on every
    // adapter request (crbug.com/369219127); omit it there (no functional effect) but keep the hint
    // on platforms that honor it.
    const onWindows = typeof navigator !== "undefined" && /Windows/i.test(navigator.userAgent ?? "");
    const adapter = await navigator.gpu.requestAdapter(onWindows ? {} : { powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    return new WebGPURenderer(device, canvas);
  }

  private buildPipeline(): void {
    const module = this.device.createShaderModule({ code: WGSL(this.normalEncoding) });
    this.pipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    this.linePipeline = null; // rebuilt lazily against the new module/encoding
    this.pointsPipeline = null;
  }

  private buildPointsPipeline(): GPURenderPipeline {
    const module = this.device.createShaderModule({ code: WGSL(this.normalEncoding) });
    return this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_points" },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-strip", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
  }

  private buildLinePipeline(): GPURenderPipeline {
    const module = this.device.createShaderModule({ code: WGSL(this.normalEncoding) });
    return this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "line-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
  }

  /** Wireframe mode (mesh editor): draw the mesh as its triangle edges. */
  setWireframe(on: boolean): void {
    this.wireframe = on;
    if (on && this.lastIndices) this.uploadLineIndices(this.lastIndices);
  }

  /** Viewport shading: textured (default) vs untextured clay. Judge FORM without the atlas. */
  setTextured(on: boolean): void { this.texMix = on ? 1 : 0; }
  setShadeMode(mode: number): void { this.shadeMode = mode | 0; }
  setPointSize(px: number): void { this.pointSize = Math.max(1, px); }
  setDepthRange(lo: number, hi: number): void { this.depthRange = [lo, hi]; }
  /** Lit (lambert) vs unlit (albedo verbatim) — unlit is the honest display for video-textured
   *  captures whose footage already carries the scene's real lighting. */
  setLit(on: boolean): void { this.litMix = on ? 1 : 0; }

  /** Infinite ground grid + origin tripod overlay; builds the tripod's vertex buffer lazily. */
  setGrid(on: boolean): void {
    this.gridOn = on;
    if (on && !this.tripodVbuf) {
      const { data, count } = buildOriginTripod({ floorY: this.gridParams.floorY, axisLen: this.gridParams.axisLen });
      this.tripodVbuf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.tripodVbuf, 0, data);
      this.tripodCount = count;
    }
  }

  /** Live model transform (column-major mat4), or null for identity. Preview only — the bake applies
   *  the same transform to the geometry itself (encoder --up-axis/--center/...). */
  setModelMatrix(m: Float32Array | null): void { this.model = m; }

  /** Cell sizes, ground height and fade radius for the infinite grid — the player recomputes these
   *  from the live camera every frame (that is what keeps the grid readable at any zoom). */
  setGridParams(p: GridParams): void { this.gridParams = p; }

  private uploadLineIndices(indices: Uint32Array): void {
    // tri (a,b,c) → edges a-b, b-c, c-a. Shared edges draw twice — visually fine, no dedup cost.
    const lines = new Uint32Array(indices.length * 2);
    for (let t = 0, w = 0; t < indices.length; t += 3) {
      const a = indices[t]!, b = indices[t + 1]!, c = indices[t + 2]!;
      lines[w++] = a; lines[w++] = b;
      lines[w++] = b; lines[w++] = c;
      lines[w++] = c; lines[w++] = a;
    }
    const need = lines.byteLength;
    if (!this.lineIdxBuf || this.lineIdxCap < need) {
      this.lineIdxBuf?.destroy();
      this.lineIdxCap = roundUp16(Math.max(need, Math.ceil(this.lineIdxCap * 1.5), 256));
      this.lineIdxBuf = this.device.createBuffer({ size: this.lineIdxCap, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.lineIdxBuf, 0, lines);
    this.lineIndexCount = lines.length;
  }

  /** World-space crop box for the mesh-editor preview (fragment discard); null disables. */
  setCrop(crop: { min: [number, number, number]; max: [number, number, number] } | null): void {
    this.crop = crop;
  }

  /** Select the per-vertex normal unpack from the file's superblock normal_encoding (0 = i8×4, 1 = oct16). */
  setNormalEncoding(encoding: number): void {
    if (encoding === this.normalEncoding) return;
    this.normalEncoding = encoding;
    this.buildPipeline();
  }

  resize(width: number, height: number): void {
    const w = Math.max(1, width | 0), h = Math.max(1, height | 0);
    this.depth?.destroy();
    this.depth = this.device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  private growStorage(cur: GPUBuffer | null, cap: number, need: number): [GPUBuffer, number] {
    if (cur && cap >= need) return [cur, cap];
    cur?.destroy();
    const size = roundUp16(Math.max(need, Math.ceil(cap * 1.5), 256));
    return [this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), size];
  }

  /** Upload the index buffer (per I-frame: every frame in intra mode, once per GOP with temporal coding). */
  uploadTopology(indices: Uint32Array): void {
    const need = indices.byteLength;
    if (!this.idxBuf || this.idxCap < need) {
      this.idxBuf?.destroy();
      this.idxCap = roundUp16(Math.max(need, Math.ceil(this.idxCap * 1.5), 256));
      this.idxBuf = this.device.createBuffer({ size: this.idxCap, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.idxBuf, 0, indices);
    this.lastIndices = indices;                                // kept so wireframe can (re)build lazily
    if (this.wireframe) this.uploadLineIndices(indices);
  }

  uploadUVs(uvsQ: Uint16Array): void {
    const need = uvsQ.byteLength;
    [this.uvBuf, this.uvCap] = this.growStorage(this.uvBuf, this.uvCap, need);
    this.device.queue.writeBuffer(this.uvBuf, 0, uvsQ.buffer as ArrayBuffer, uvsQ.byteOffset, uvsQ.byteLength);
  }

  uploadNormals(normalsQ: Int8Array): void {
    const need = normalsQ.byteLength;
    [this.nrmBuf, this.nrmCap] = this.growStorage(this.nrmBuf, this.nrmCap, need);
    this.device.queue.writeBuffer(this.nrmBuf, 0, normalsQ.buffer as ArrayBuffer, normalsQ.byteOffset, normalsQ.byteLength);
  }

  /** Write the next frame's quantized positions into the next triple-buffer slot. */
  writePositions(positionsQ: Uint16Array): void {
    this.slot = (this.slot + 1) % SLOTS;
    const need = positionsQ.byteLength;
    let buf = this.posBufs[this.slot];
    if (!buf || this.posCap < need) {
      // grow all slots together so the cap stays uniform
      for (let i = 0; i < SLOTS; i++) this.posBufs[i]?.destroy();
      this.posCap = roundUp16(Math.max(need, Math.ceil(this.posCap * 1.5), 256));
      for (let i = 0; i < SLOTS; i++) this.posBufs[i] = this.device.createBuffer({ size: this.posCap, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      buf = this.posBufs[this.slot]!;
    }
    this.device.queue.writeBuffer(buf, 0, positionsQ.buffer as ArrayBuffer, positionsQ.byteOffset, positionsQ.byteLength);
    this.vertexCount = positionsQ.length >> 2;
  }

  setTextureFromBitmap(bitmap: ImageBitmap): void {
    this.allocTexture(bitmap.width, bitmap.height);
    this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: this.texture! }, [bitmap.width, bitmap.height]);
  }

  /** Import a decoded video frame (WebCodecs VideoFrame) into the sampled texture — GPU-side copy, no CPU pixels (spec §10.3). */
  setTextureFromVideoFrame(frame: VideoFrame): void {
    const w = frame.displayWidth, h = frame.displayHeight;
    if (!this.texture || this.texW !== w || this.texH !== h) this.allocTexture(w, h);
    this.device.queue.copyExternalImageToTexture({ source: frame }, { texture: this.texture! }, [w, h]);
  }

  private setTextureFromPixels(rgba: Uint8Array, w: number, h: number): void {
    this.allocTexture(w, h);
    this.device.queue.writeTexture({ texture: this.texture! }, rgba, { bytesPerRow: w * 4 }, [w, h]);
  }

  private allocTexture(w: number, h: number): void {
    this.texture?.destroy();
    this.texture = this.device.createTexture({
      size: [w, h], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.texView = this.texture.createView();
    this.texW = w; this.texH = h;
  }

  render(viewProj: Float32Array, aabb: Aabb, invLevels: number, indexCount: number): void {
    if (!this.idxBuf || !this.uvBuf || !this.nrmBuf || !this.posBufs[this.slot] || !this.depth || !this.texView) return;
    // The mesh draws through viewProj*model; everything below that reads `viewProj` (grid, tripod)
    // deliberately keeps the world one.
    this.uni.set(this.model ? multiply(viewProj, this.model) : viewProj, 0);
    this.uni[16] = aabb.min[0]; this.uni[17] = aabb.min[1]; this.uni[18] = aabb.min[2];
    this.uni[19] = invLevels;
    this.uni[20] = aabb.max[0] - aabb.min[0]; this.uni[21] = aabb.max[1] - aabb.min[1]; this.uni[22] = aabb.max[2] - aabb.min[2];
    this.uni[23] = this.litMix;
    const c = this.crop;
    this.uni[24] = c ? c.min[0] : 0; this.uni[25] = c ? c.min[1] : 0; this.uni[26] = c ? c.min[2] : 0;
    this.uni[27] = c ? 1 : 0;
    this.uni[28] = c ? c.max[0] : 0; this.uni[29] = c ? c.max[1] : 0; this.uni[30] = c ? c.max[2] : 0;
    this.uni[31] = this.texMix;
    this.uni[32] = this.shadeMode; this.uni[33] = this.pointSize; this.uni[34] = this.depthRange[0]; this.uni[35] = this.depthRange[1];
    this.uni[36] = this.depth ? this.depth.width : 1; this.uni[37] = this.depth ? this.depth.height : 1; this.uni[38] = 0; this.uni[39] = 0;
    this.device.queue.writeBuffer(this.uniform, 0, this.uni);

    // Pick the pipeline FIRST — with layout:"auto" the bind group must come from the active pipeline.
    const points = this.shadeMode === 4 && this.vertexCount > 0;
    const wire = !points && this.wireframe && this.lineIdxBuf && this.lineIndexCount > 0;
    const activePipeline = points ? (this.pointsPipeline ??= this.buildPointsPipeline()) : wire ? (this.linePipeline ??= this.buildLinePipeline()) : this.pipeline;
    const bindGroup = this.device.createBindGroup({
      layout: activePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: this.posBufs[this.slot]! } },
        { binding: 2, resource: { buffer: this.uvBuf } },
        { binding: 3, resource: this.sampler },
        { binding: 4, resource: this.texView },
        { binding: 5, resource: { buffer: this.nrmBuf } },
        { binding: 6, resource: { buffer: this.fxBuffer() } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.055, b: 0.07, a: 1 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depth.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });
    pass.setPipeline(activePipeline);
    pass.setBindGroup(0, bindGroup);
    if (points) {
      pass.draw(4, this.vertexCount);
    } else {
      pass.setIndexBuffer(wire ? this.lineIdxBuf! : this.idxBuf, "uint32");
      pass.drawIndexed(wire ? this.lineIndexCount : indexCount);
    }
    // Overlay (after the mesh; depth write off so it never occludes, but the mesh occludes it).
    this.drawOverlay(pass, viewProj);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // ---- Gaussian splat profile ---------------------------------------------------------------

  /** Upload one decoded splat frame (positions, packed attributes, optional SH) — every frame. */
  uploadSplats(f: DecodedSplat): void {
    [this.sposBuf, this.sposCap] = this.growStorage(this.sposBuf, this.sposCap, Math.max(16, f.positionsQ.byteLength));
    this.device.queue.writeBuffer(this.sposBuf, 0, f.positionsQ.buffer as ArrayBuffer, f.positionsQ.byteOffset, f.positionsQ.byteLength);
    [this.sattrBuf, this.sattrCap] = this.growStorage(this.sattrBuf, this.sattrCap, Math.max(16, f.attrs.byteLength));
    this.device.queue.writeBuffer(this.sattrBuf, 0, f.attrs.buffer as ArrayBuffer, f.attrs.byteOffset, f.attrs.byteLength);
    const shBytes = f.sh && f.shDegree > 0 ? f.sh : null;
    [this.sshBuf, this.sshCap] = this.growStorage(this.sshBuf, this.sshCap, Math.max(16, shBytes?.byteLength ?? 0));
    if (shBytes) this.device.queue.writeBuffer(this.sshBuf, 0, shBytes.buffer as ArrayBuffer, shBytes.byteOffset, shBytes.byteLength);
    this.splatCount = f.count;
    this.splatShDegree = shBytes ? f.shDegree : 0;
    if (!this.sorderBuf || this.sorderCap < f.count * 4) {
      // Identity order until the sorter delivers one, so a frame never draws with a stale/short list.
      const id = new Uint32Array(f.count);
      for (let i = 0; i < f.count; i++) id[i] = i;
      this.setSplatOrder(id);
    }
  }

  /** Back-to-front splat ids from the sorter (splat-sort.ts). */
  setSplatOrder(order: Uint32Array): void {
    [this.sorderBuf, this.sorderCap] = this.growStorage(this.sorderBuf, this.sorderCap, Math.max(16, order.byteLength));
    this.device.queue.writeBuffer(this.sorderBuf, 0, order.buffer as ArrayBuffer, order.byteOffset, order.byteLength);
  }

  setSplatParams(p: Partial<SplatParams>): void { this.splatParams = { ...this.splatParams, ...p }; }

  /** Packed effects block (fx.ts packFx). Uploaded before each pass. */
  setFx(data: Float32Array): void { this.fxData.set(data.subarray(0, 32)); }
  private fxBuffer(): GPUBuffer {
    this.fxUni ??= this.device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.fxUni, 0, this.fxData);
    return this.fxUni;
  }

  private buildSplatPipeline(): GPURenderPipeline {
    const module = this.device.createShaderModule({ code: SPLAT_WGSL });
    return this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module, entryPoint: "fs",
        targets: [{
          format: this.format,
          blend: {   // premultiplied "over", drawn back-to-front
            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        }],
      },
      primitive: { topology: "triangle-strip", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
    });
  }

  /**
   * Draw the current splat frame. The overlay (grid, tripod) goes FIRST here: splats write no depth,
   * so drawing them last lets an opaque-looking cloud cover the floor grid instead of the grid
   * being painted over it.
   */
  renderSplats(cam: SplatCamera, aabb: Aabb, invLevels: number, count: number): void {
    if (!this.depth) return;
    const n = Math.min(count, this.splatCount);
    const modelView = this.model ? multiply(cam.view, this.model) : cam.view;
    const viewProjWorld = multiply(cam.proj, cam.view);
    // Camera position in MODEL space: inverse(view*model) · origin.
    const inv = invert(modelView);
    const camPos: [number, number, number] = inv ? [inv[12]!, inv[13]!, inv[14]!] : [0, 0, 0];
    const d = this.splatUniData;
    d.set(modelView, 0); d.set(cam.proj, 16);
    d[32] = aabb.min[0]; d[33] = aabb.min[1]; d[34] = aabb.min[2]; d[35] = invLevels;
    d[36] = aabb.max[0] - aabb.min[0]; d[37] = aabb.max[1] - aabb.min[1]; d[38] = aabb.max[2] - aabb.min[2]; d[39] = this.splatShDegree;
    d[40] = cam.width; d[41] = cam.height;
    d[42] = cam.proj[0]! * cam.width / 2; d[43] = cam.proj[5]! * cam.height / 2;
    d[44] = camPos[0]; d[45] = camPos[1]; d[46] = camPos[2]; d[47] = cam.ortho ? 1 : 0;
    const c = this.crop;
    d[48] = c ? c.min[0] : 0; d[49] = c ? c.min[1] : 0; d[50] = c ? c.min[2] : 0; d[51] = c ? 1 : 0;
    d[52] = c ? c.max[0] : 0; d[53] = c ? c.max[1] : 0; d[54] = c ? c.max[2] : 0; d[55] = this.splatParams.scaleMul;
    d[56] = this.splatParams.opacityMul; d[57] = this.texMix; d[58] = 0; d[59] = 0;
    this.splatUni ??= this.device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.splatUni, 0, d);
    // The overlay's tripod reads viewProj from the mesh uniform's first 64 bytes.
    this.uni.set(viewProjWorld, 0);
    this.device.queue.writeBuffer(this.uniform, 0, this.uni);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), clearValue: { r: 0.05, g: 0.055, b: 0.07, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    this.drawOverlay(pass, viewProjWorld);
    if (n > 0 && this.sposBuf && this.sattrBuf && this.sshBuf && this.sorderBuf) {
      this.splatPipeline ??= this.buildSplatPipeline();
      const bind = this.device.createBindGroup({
        layout: this.splatPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.splatUni } },
          { binding: 1, resource: { buffer: this.sposBuf } },
          { binding: 2, resource: { buffer: this.sattrBuf } },
          { binding: 3, resource: { buffer: this.sorderBuf } },
          { binding: 4, resource: { buffer: this.sshBuf } },
          { binding: 5, resource: { buffer: this.fxBuffer() } },
        ],
      });
      pass.setPipeline(this.splatPipeline);
      pass.setBindGroup(0, bind);
      pass.draw(4, n);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Infinite grid + origin tripod (shared by the mesh and splat passes). Depth write off. */
  private drawOverlay(pass: GPURenderPassEncoder, viewProj: Float32Array): void {
    if (!this.gridOn) return;
    const inv = invert(viewProj);
    if (inv) {
      if (!this.gridPipeline) {
        const gm = this.device.createShaderModule({ code: GRID_WGSL });
        this.gridPipeline = this.device.createRenderPipeline({
          layout: "auto",
          vertex: { module: gm, entryPoint: "vs" },
          fragment: {
            module: gm, entryPoint: "fs",
            targets: [{
              format: this.format,
              blend: {
                color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
              },
            }],
          },
          primitive: { topology: "triangle-list" },
          depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
        });
        this.gridBind = null;
      }
      this.gridUni ??= this.device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const gp = this.gridParams;
      this.gridData.set(inv, 0);
      this.gridData.set(viewProj, 16);
      this.gridData[32] = gp.step0; this.gridData[33] = gp.step1; this.gridData[34] = gp.fineFade; this.gridData[35] = gp.floorY;
      this.gridData[36] = gp.centerX; this.gridData[37] = gp.centerZ; this.gridData[38] = gp.fadeRadius; this.gridData[39] = 0;
      this.device.queue.writeBuffer(this.gridUni, 0, this.gridData);
      this.gridBind ??= this.device.createBindGroup({ layout: this.gridPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.gridUni } }] });
      pass.setPipeline(this.gridPipeline);
      pass.setBindGroup(0, this.gridBind);
      pass.draw(3);
    }
    if (this.tripodVbuf && this.tripodCount > 0) {
      if (!this.tripodPipeline) {
        const tm = this.device.createShaderModule({ code: LINES_WGSL });
        this.tripodPipeline = this.device.createRenderPipeline({
          layout: "auto",
          vertex: { module: tm, entryPoint: "vs", buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }] },
          fragment: { module: tm, entryPoint: "fs", targets: [{ format: this.format }] },
          primitive: { topology: "line-list" },
          depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
        });
        this.tripodBind = null;
      }
      this.tripodBind ??= this.device.createBindGroup({ layout: this.tripodPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }] });
      pass.setPipeline(this.tripodPipeline);
      pass.setBindGroup(0, this.tripodBind);
      pass.setVertexBuffer(0, this.tripodVbuf);
      pass.draw(this.tripodCount);
    }
  }

  dispose(): void {
    this.sposBuf?.destroy(); this.sattrBuf?.destroy(); this.sshBuf?.destroy(); this.sorderBuf?.destroy(); this.splatUni?.destroy(); this.fxUni?.destroy();
    this.posBufs.forEach((b) => b?.destroy());
    this.uvBuf?.destroy();
    this.nrmBuf?.destroy();
    this.idxBuf?.destroy();
    this.lineIdxBuf?.destroy();
    this.tripodVbuf?.destroy();
    this.gridUni?.destroy();
    this.texture?.destroy();
    this.depth?.destroy();
    this.uniform.destroy();
    this.device.destroy();
  }
}
