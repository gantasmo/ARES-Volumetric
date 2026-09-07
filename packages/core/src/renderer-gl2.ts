/**
 * WebGL2 fallback renderer (spec §10.4) — same public surface as WebGPURenderer
 * (renderer.ts) so the player can swap it in when `navigator.gpu` is missing.
 *
 * WebGL2 has no storage buffers, so the vertex-pull layout maps onto plain
 * vertex attributes with the SAME byte layouts the player already uploads:
 *   - positions: 3 × UNSIGNED_SHORT, stride 8 (u16 x,y,z,pad — raw, NOT normalized;
 *     the shader dequantizes exactly like the WGSL: world = min + q·invLevels·size)
 *   - uvs:       2 × UNSIGNED_SHORT normalized, stride 4  (u/65535)
 *   - normals:   stride 4, two program variants mirroring the WGSL NRM_DECODE:
 *       0 legacy: 4 × BYTE normalized (i8 snorm, w = pad)
 *       1 oct16:  2 × SHORT normalized + octahedral decode in GLSL
 * Texture uploads go through texImage2D — ImageBitmap directly; VideoFrame
 * directly where supported (modern Chrome), else via a createImageBitmap copy.
 */
import type { Aabb } from "./quant.js";
import { invert, multiply, type Mat4 } from "./camera.js";
import { buildOriginTripod, type GridParams } from "./overlay.js";
import { shStrideBytes, type DecodedSplat } from "./splat.js";

/** Camera pieces the splat pass needs separately (see renderer.ts SPLAT_WGSL). */
export interface SplatCamera { view: Mat4; proj: Mat4; ortho: boolean; width: number; height: number; }
/** Live splat display multipliers (VFX hooks): 1/1 = as captured. */
export interface SplatParams { scaleMul: number; opacityMul: number; }

/**
 * The renderer surface the player drives — implemented by both WebGPURenderer
 * (structurally) and WebGL2Renderer, so the player can hold either.
 */
export interface AresRenderer {
  resize(width: number, height: number): void;
  setNormalEncoding(encoding: number): void;
  /** World-space crop-box preview (mesh editor); null disables. */
  setCrop(crop: { min: [number, number, number]; max: [number, number, number] } | null): void;
  /** Wireframe render mode (mesh editor). */
  setWireframe(on: boolean): void;
  /** Viewport shading: textured (default) vs untextured clay. */
  setTextured(on: boolean): void;
  setLit(on: boolean): void;
  /** Debug/analysis view: 0 shaded (texMix/litMix apply), 1 normals, 2 UV checker, 3 depth, 4 points. */
  setShadeMode(mode: number): void;
  /** Point size in pixels for mode 4. */
  setPointSize(px: number): void;
  /** View-depth window for the depth view (world units along the view axis). */
  setDepthRange(lo: number, hi: number): void;
  /** Infinite ground grid + origin tripod overlay. */
  setGrid(on: boolean): void;
  /** Live model transform (column-major mat4), null = identity. Preview only; the bake is encoder-side. */
  setModelMatrix(m: Float32Array | null): void;
  /** Per-frame grid sizing (cell sizes, ground height, fade radius) — see overlay.ts gridLod. */
  setGridParams(p: GridParams): void;
  uploadTopology(indices: Uint32Array): void;
  uploadUVs(uvsQ: Uint16Array): void;
  uploadNormals(normalsQ: Int8Array): void;
  writePositions(positionsQ: Uint16Array): void;
  setTextureFromBitmap(bitmap: ImageBitmap): void;
  setTextureFromVideoFrame(frame: VideoFrame): void;
  render(viewProj: Float32Array, aabb: Aabb, invLevels: number, indexCount: number): void;
  // Gaussian splat profile (spec §6.8)
  uploadSplats(f: DecodedSplat): void;
  setSplatOrder(order: Uint32Array): void;
  setSplatParams(p: Partial<SplatParams>): void;
  renderSplats(cam: SplatCamera, aabb: Aabb, invLevels: number, count: number): void;
  /** Packed playback-effects block (fx.ts packFx, 32 floats). */
  setFx(data: Float32Array): void;
  dispose(): void;
}

const A_POS = 0, A_UV = 1, A_NRM = 2;

/** GLSL 300 es vertex shader — mirrors the WGSL dequant + normal unpack exactly. */
const VS = (normalEncoding: number): string => `#version 300 es
precision highp float;
layout(location=${A_POS}) in vec3 aPosQ; // quantized u16 position, raw integer values (not normalized)
layout(location=${A_UV}) in vec2 aUV;    // u16 normalized -> [0,1]
${normalEncoding === 1
    ? `layout(location=${A_NRM}) in vec2 aNrm; // oct16: 2 x i16 snorm`
    : `layout(location=${A_NRM}) in vec4 aNrm; // legacy: i8 x4 snorm (w = pad)`}
uniform mat4 uViewProj;
uniform vec3 uAabbMin;
uniform vec3 uAabbSize;
uniform float uInvLevels;
uniform float uPointSize;
out vec2 vUV;
out vec3 vWorld;
out vec3 vNormal;
out float vDepth;

// ---- playback effects (fx.ts packFx layout: 8 x vec4) ----
uniform vec4 uFx[8];
float fxHash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
float fxNoise(vec3 p) {
  vec3 i = floor(p); vec3 f = fract(p); vec3 u = f * f * (3.0 - 2.0 * f);
  float a = mix(mix(fxHash(i), fxHash(i + vec3(1.0, 0.0, 0.0)), u.x), mix(fxHash(i + vec3(0.0, 1.0, 0.0)), fxHash(i + vec3(1.0, 1.0, 0.0)), u.x), u.y);
  float b = mix(mix(fxHash(i + vec3(0.0, 0.0, 1.0)), fxHash(i + vec3(1.0, 0.0, 1.0)), u.x), mix(fxHash(i + vec3(0.0, 1.0, 1.0)), fxHash(i + vec3(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}
vec3 fxWobble(vec3 p) {
  float amp = uFx[4].w;
  if (amp <= 0.0) return p;
  float f = uFx[5].x, t = uFx[5].y;
  return p + amp * vec3(sin(p.y * f + t * 3.0), sin(p.z * f * 1.3 + t * 2.2), sin(p.x * f * 0.7 + t * 2.7));
}
vec4 fxApply(vec3 world, vec3 n, vec3 colIn) {
  vec3 col = colIn;
  if (uFx[1].x > 0.5 && dot(world, uFx[0].xyz) + uFx[0].w < 0.0) return vec4(col, 0.0);
  float d = uFx[1].y;
  if (d > 0.0) {
    float nz = fxNoise(world * uFx[1].z);
    if (nz < d) return vec4(col, 0.0);
    col = mix(uFx[4].xyz, col, smoothstep(d, d + 0.08, nz));
  }
  if (uFx[1].w > 0.0) col *= 1.0 - uFx[1].w * 0.5 * (0.5 + 0.5 * sin(world.y * uFx[5].z + uFx[5].y * 4.0));
  if (uFx[3].w > 0.0 && dot(n, n) > 0.5) { vec3 v = normalize(uFx[6].xyz - world); float fr = pow(1.0 - abs(dot(n, v)), 3.0); col += uFx[3].xyz * uFx[3].w * fr; }
  if (uFx[2].w > 0.0) col = mix(col, col * uFx[2].xyz, uFx[2].w);
  return vec4(col, 1.0);
}

void main() {
  vec3 world = fxWobble(uAabbMin + (aPosQ * uInvLevels) * uAabbSize); // GPU-side dequant (+ wobble fx)
  gl_Position = uViewProj * vec4(world, 1.0);
  gl_PointSize = uPointSize;
  vDepth = gl_Position.w;
  vUV = aUV;
  vWorld = world;
${normalEncoding === 1
    ? `  vec3 on = vec3(aNrm, 1.0 - abs(aNrm.x) - abs(aNrm.y));
  float ot = clamp(-on.z, 0.0, 1.0);
  on.x += (on.x >= 0.0) ? -ot : ot;
  on.y += (on.y >= 0.0) ? -ot : ot;
  vNormal = normalize(on);`
    : `  vNormal = aNrm.xyz;`}
}
`;

/** Fragment stage: same two-sided lighting as the WGSL — prefer the vertex normal, fall back to a face normal. */
const FS = `#version 300 es
precision highp float;
in vec2 vUV;
in vec3 vWorld;
in vec3 vNormal;
uniform sampler2D uTex;
uniform float uTexMix;                       // 1 = textured (shaded), 0 = untextured clay
uniform float uLitMix;                       // 1 = lit (lambert), 0 = unlit (albedo verbatim)
uniform vec3 uCropMin;                       // world-space crop box (mesh editor preview)
uniform vec3 uCropMax;
uniform float uCropOn;
uniform int uShadeMode;                      // 0 shaded, 1 normals, 2 uv checker, 3 depth, 4 points
uniform vec2 uDepthRange;
in float vDepth;
out vec4 fragColor;

// ---- playback effects (fx.ts packFx layout: 8 x vec4) ----
uniform vec4 uFx[8];
float fxHash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
float fxNoise(vec3 p) {
  vec3 i = floor(p); vec3 f = fract(p); vec3 u = f * f * (3.0 - 2.0 * f);
  float a = mix(mix(fxHash(i), fxHash(i + vec3(1.0, 0.0, 0.0)), u.x), mix(fxHash(i + vec3(0.0, 1.0, 0.0)), fxHash(i + vec3(1.0, 1.0, 0.0)), u.x), u.y);
  float b = mix(mix(fxHash(i + vec3(0.0, 0.0, 1.0)), fxHash(i + vec3(1.0, 0.0, 1.0)), u.x), mix(fxHash(i + vec3(0.0, 1.0, 1.0)), fxHash(i + vec3(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}
vec3 fxWobble(vec3 p) {
  float amp = uFx[4].w;
  if (amp <= 0.0) return p;
  float f = uFx[5].x, t = uFx[5].y;
  return p + amp * vec3(sin(p.y * f + t * 3.0), sin(p.z * f * 1.3 + t * 2.2), sin(p.x * f * 0.7 + t * 2.7));
}
vec4 fxApply(vec3 world, vec3 n, vec3 colIn) {
  vec3 col = colIn;
  if (uFx[1].x > 0.5 && dot(world, uFx[0].xyz) + uFx[0].w < 0.0) return vec4(col, 0.0);
  float d = uFx[1].y;
  if (d > 0.0) {
    float nz = fxNoise(world * uFx[1].z);
    if (nz < d) return vec4(col, 0.0);
    col = mix(uFx[4].xyz, col, smoothstep(d, d + 0.08, nz));
  }
  if (uFx[1].w > 0.0) col *= 1.0 - uFx[1].w * 0.5 * (0.5 + 0.5 * sin(world.y * uFx[5].z + uFx[5].y * 4.0));
  if (uFx[3].w > 0.0 && dot(n, n) > 0.5) { vec3 v = normalize(uFx[6].xyz - world); float fr = pow(1.0 - abs(dot(n, v)), 3.0); col += uFx[3].xyz * uFx[3].w * fr; }
  if (uFx[2].w > 0.0) col = mix(col, col * uFx[2].xyz, uFx[2].w);
  return vec4(col, 1.0);
}

void main() {
  vec3 faceN = cross(dFdx(vWorld), dFdy(vWorld));
  vec3 n = normalize(dot(vNormal, vNormal) > 0.01 ? vNormal : faceN);
  vec3 L = normalize(vec3(0.35, 0.75, 0.55));
  float diff = abs(dot(n, L));               // two-sided (winding-agnostic)
  // Matches the WebGPU path's clay colour so the two renderers agree.
  vec3 albedo = mix(vec3(0.72, 0.71, 0.68), texture(uTex, vUV).rgb, uTexMix);
  // Crop preview: discard LAST so the derivatives above stay uniform (same order as the WGSL).
  if (uCropOn > 0.5 && (any(lessThan(vWorld, uCropMin)) || any(greaterThan(vWorld, uCropMax)))) discard;
  float lit = mix(1.0, 0.4 + 0.6 * diff, uLitMix);
  vec3 col = albedo * lit;
  if (uShadeMode == 1) col = n * 0.5 + 0.5;
  else if (uShadeMode == 2) {
    float ck = mod(floor(vUV.x * 32.0) + floor(vUV.y * 32.0), 2.0);
    col = mix(vec3(0.22, 0.22, 0.24), vec3(0.82, 0.80, 0.76), ck) * mix(1.0, lit, 0.5);
  } else if (uShadeMode == 3) {
    float d = clamp((vDepth - uDepthRange.x) / max(1e-6, uDepthRange.y - uDepthRange.x), 0.0, 1.0);
    col = vec3(1.0 - d) * vec3(0.92, 0.9, 0.86);
  } else if (uShadeMode == 4) col = albedo;   // points: unlit, the texture verbatim
  vec4 fxo = fxApply(vWorld, n, col);
  if (fxo.a < 0.5) discard;
  fragColor = vec4(fxo.rgb, 1.0);
}
`;

/**
 * Splat pass — the GLSL mirror of renderer.ts SPLAT_WGSL. WebGL2 has no storage buffers, so the
 * per-splat records live in an RGBA32UI data texture (2 texels per splat: pos0, pos1, attr0,
 * attr1 | attr2, 0, 0, 0) plus an optional SH texture, and the back-to-front order is a
 * per-instance uint attribute. Same math line for line; kept comparable on purpose.
 */
const SPLAT_VS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
layout(location=0) in uint aOrder;
uniform usampler2D uSplat;
uniform usampler2D uSh;
uniform int uTexW;
uniform int uShTexels;
uniform int uShDegree;
uniform mat4 uView;                         // view * model
uniform mat4 uProj;
uniform vec3 uAabbMin; uniform vec3 uAabbSize; uniform float uInvLevels;
uniform vec2 uViewport; uniform vec2 uFocal;
uniform vec3 uCamPos; uniform float uOrtho;
uniform vec3 uCropMin; uniform vec3 uCropMax; uniform float uCropOn;
uniform float uScaleMul; uniform float uOpacityMul; uniform float uTexMix;
out vec2 vLocal;
out vec4 vColor;
out vec3 vWorld;

// ---- playback effects (fx.ts packFx layout: 8 x vec4) ----
uniform vec4 uFx[8];
float fxHash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
float fxNoise(vec3 p) {
  vec3 i = floor(p); vec3 f = fract(p); vec3 u = f * f * (3.0 - 2.0 * f);
  float a = mix(mix(fxHash(i), fxHash(i + vec3(1.0, 0.0, 0.0)), u.x), mix(fxHash(i + vec3(0.0, 1.0, 0.0)), fxHash(i + vec3(1.0, 1.0, 0.0)), u.x), u.y);
  float b = mix(mix(fxHash(i + vec3(0.0, 0.0, 1.0)), fxHash(i + vec3(1.0, 0.0, 1.0)), u.x), mix(fxHash(i + vec3(0.0, 1.0, 1.0)), fxHash(i + vec3(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}
vec3 fxWobble(vec3 p) {
  float amp = uFx[4].w;
  if (amp <= 0.0) return p;
  float f = uFx[5].x, t = uFx[5].y;
  return p + amp * vec3(sin(p.y * f + t * 3.0), sin(p.z * f * 1.3 + t * 2.2), sin(p.x * f * 0.7 + t * 2.7));
}
vec4 fxApply(vec3 world, vec3 n, vec3 colIn) {
  vec3 col = colIn;
  if (uFx[1].x > 0.5 && dot(world, uFx[0].xyz) + uFx[0].w < 0.0) return vec4(col, 0.0);
  float d = uFx[1].y;
  if (d > 0.0) {
    float nz = fxNoise(world * uFx[1].z);
    if (nz < d) return vec4(col, 0.0);
    col = mix(uFx[4].xyz, col, smoothstep(d, d + 0.08, nz));
  }
  if (uFx[1].w > 0.0) col *= 1.0 - uFx[1].w * 0.5 * (0.5 + 0.5 * sin(world.y * uFx[5].z + uFx[5].y * 4.0));
  if (uFx[3].w > 0.0 && dot(n, n) > 0.5) { vec3 v = normalize(uFx[6].xyz - world); float fr = pow(1.0 - abs(dot(n, v)), 3.0); col += uFx[3].xyz * uFx[3].w * fr; }
  if (uFx[2].w > 0.0) col = mix(col, col * uFx[2].xyz, uFx[2].w);
  return vec4(col, 1.0);
}

const float SH_C1 = 0.4886025119029199;
const float SH_C2[5] = float[5](1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396);
const float SH_C3[7] = float[7](-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435);

uvec4 fetchT(usampler2D t, int i) { return texelFetch(t, ivec2(i % uTexW, i / uTexW), 0); }
float shByte(int id, int idx) {
  int j = idx >> 2;
  uvec4 t = fetchT(uSh, id * uShTexels + (j >> 2));
  int c = j & 3;
  uint w = c == 0 ? t.x : c == 1 ? t.y : c == 2 ? t.z : t.w;
  return (float((w >> uint((idx & 3) * 8)) & 0xffu) - 128.0) / 128.0;
}
vec3 shCoef(int id, int k) { return vec3(shByte(id, k * 3), shByte(id, k * 3 + 1), shByte(id, k * 3 + 2)); }
vec3 evalSH(int id, vec3 d, int degree) {
  float x = d.x, y = d.y, z = d.z;
  vec3 c = -SH_C1 * y * shCoef(id, 0) + SH_C1 * z * shCoef(id, 1) - SH_C1 * x * shCoef(id, 2);
  if (degree >= 2) {
    float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
    c += SH_C2[0] * xy * shCoef(id, 3) + SH_C2[1] * yz * shCoef(id, 4)
       + SH_C2[2] * (2.0 * zz - xx - yy) * shCoef(id, 5)
       + SH_C2[3] * xz * shCoef(id, 6) + SH_C2[4] * (xx - yy) * shCoef(id, 7);
    if (degree >= 3) {
      c += SH_C3[0] * y * (3.0 * xx - yy) * shCoef(id, 8)
         + SH_C3[1] * xy * z * shCoef(id, 9)
         + SH_C3[2] * y * (4.0 * zz - xx - yy) * shCoef(id, 10)
         + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shCoef(id, 11)
         + SH_C3[4] * x * (4.0 * zz - xx - yy) * shCoef(id, 12)
         + SH_C3[5] * z * (xx - yy) * shCoef(id, 13)
         + SH_C3[6] * x * (xx - 3.0 * yy) * shCoef(id, 14);
    }
  }
  return c;
}
vec4 unpackQuat(uint c) {
  uint iL = c >> 30u;
  float m0 = float((c >> 20u) & 0x1ffu) / 511.0 * 0.70710678118;
  float m1 = float((c >> 10u) & 0x1ffu) / 511.0 * 0.70710678118;
  float m2 = float(c & 0x1ffu) / 511.0 * 0.70710678118;
  float v0 = ((c >> 29u) & 1u) == 1u ? -m0 : m0;
  float v1 = ((c >> 19u) & 1u) == 1u ? -m1 : m1;
  float v2 = ((c >> 9u) & 1u) == 1u ? -m2 : m2;
  float big = sqrt(max(0.0, 1.0 - v0 * v0 - v1 * v1 - v2 * v2));
  if (iL == 0u) return vec4(big, v0, v1, v2);
  if (iL == 1u) return vec4(v0, big, v1, v2);
  if (iL == 2u) return vec4(v0, v1, big, v2);
  return vec4(v0, v1, v2, big);
}
mat3 quatToMat(vec4 q) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  return mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z),       2.0 * (x * z - w * y),
    2.0 * (x * y - w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y),       2.0 * (y * z - w * x),       1.0 - 2.0 * (x * x + y * y));
}
void main() {
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  vLocal = vec2(0.0);
  vColor = vec4(0.0);
  vWorld = vec3(0.0);
  int id = int(aOrder);
  uvec4 t0 = fetchT(uSplat, id * 2);
  uint w2 = fetchT(uSplat, id * 2 + 1).x;
  uint a = t0.x, b = t0.y, w0 = t0.z, w1 = t0.w;
  vec3 q = vec3(float(a & 0xffffu), float((a >> 16u) & 0xffffu), float(b & 0xffffu));
  vec3 pos = fxWobble(uAabbMin + (q * uInvLevels) * uAabbSize);
  if (uFx[5].w > 0.0) {
    float hid = float(id) * 0.001, t = floor(uFx[5].y * 12.0);
    pos += (vec3(fxHash(vec3(hid, 1.0, t)), fxHash(vec3(hid, 2.0, t)), fxHash(vec3(hid, 3.0, t))) - 0.5) * 2.0 * uFx[5].w;
  }
  vWorld = pos;
  if (uCropOn > 0.5 && (any(lessThan(pos, uCropMin)) || any(greaterThan(pos, uCropMax)))) return;
  if (uFx[1].x > 0.5 && dot(pos, uFx[0].xyz) + uFx[0].w < 0.0) return;
  float edgeMix = 0.0;
  if (uFx[1].y > 0.0) { float nz = fxNoise(pos * uFx[1].z); if (nz < uFx[1].y) return; edgeMix = 1.0 - smoothstep(uFx[1].y, uFx[1].y + 0.08, nz); }
  vec4 cam = uView * vec4(pos, 1.0);
  vec4 clip = uProj * cam;
  if (uOrtho < 0.5 && clip.w <= 1e-6) return;
  vec3 ndc = clip.xyz / clip.w;
  if (any(greaterThan(abs(ndc.xy), vec2(1.5))) || ndc.z < 0.0 || ndc.z > 1.0) return;
  vec3 s = exp(vec3(float(w0 & 0xffu), float((w0 >> 8u) & 0xffu), float((w0 >> 16u) & 0xffu)) / 16.0 - 10.0) * uScaleMul * uFx[6].w;
  float alpha = float(w0 >> 24u) / 255.0 * uOpacityMul * uFx[7].x;
  mat3 R = quatToMat(unpackQuat(w1));
  mat3 M = R * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 cov3 = M * transpose(M);
  mat3 W = mat3(uView[0].xyz, uView[1].xyz, uView[2].xyz);
  mat3 J;
  if (uOrtho > 0.5) {
    J = mat3(uFocal.x, 0.0, 0.0,  0.0, uFocal.y, 0.0,  0.0, 0.0, 0.0);
  } else {
    float z = cam.z;
    vec2 lim = 1.3 * 0.5 * uViewport / uFocal;
    float tx = clamp(cam.x / z, -lim.x, lim.x) * z;
    float ty = clamp(cam.y / z, -lim.y, lim.y) * z;
    J = mat3(uFocal.x / z, 0.0, 0.0,  0.0, uFocal.y / z, 0.0,  -uFocal.x * tx / (z * z), -uFocal.y * ty / (z * z), 0.0);
  }
  mat3 T = J * W;
  mat3 cov2 = T * cov3 * transpose(T);
  float cxx = cov2[0][0] + 0.3;
  float cyy = cov2[1][1] + 0.3;
  float cxy = cov2[0][1];
  float mid = 0.5 * (cxx + cyy);
  float rad = sqrt(max(0.0, mid * mid - (cxx * cyy - cxy * cxy)));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.1);
  float r1 = 3.0 * sqrt(l1);
  float r2 = 3.0 * sqrt(l2);
  vec2 v1 = abs(cxy) < 1e-6 ? (cxx >= cyy ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(cxy, l1 - cxx));
  vec2 v2 = vec2(-v1.y, v1.x);
  vec2 corner = vec2(float((gl_VertexID & 1) * 2) - 1.0, float((gl_VertexID >> 1) * 2) - 1.0);
  vec2 offsetPx = corner.x * r1 * v1 + corner.y * r2 * v2;
  vec2 ndcOff = offsetPx * 2.0 / uViewport;
  gl_Position = vec4((ndc.xy + ndcOff) * clip.w, ndc.z * clip.w, clip.w);
  vLocal = corner * 3.0;
  vec3 col = vec3(float(w2 & 0xffu), float((w2 >> 8u) & 0xffu), float((w2 >> 16u) & 0xffu)) / 255.0;
  if (uShDegree > 0) col += evalSH(id, normalize(pos - uCamPos), uShDegree);
  col = clamp(col, 0.0, 1.0);
  col = mix(vec3(0.72, 0.71, 0.68), col, uTexMix);
  col = mix(col, uFx[4].xyz, edgeMix);
  if (uFx[2].w > 0.0) col = mix(col, col * uFx[2].xyz, uFx[2].w);
  vColor = vec4(col, alpha);
}
`;

const SPLAT_FS = `#version 300 es
precision highp float;
in vec2 vLocal;
in vec4 vColor;
in vec3 vWorld;
uniform vec4 uFx[8];
out vec4 fragColor;
void main() {
  float d2 = dot(vLocal, vLocal);
  if (d2 > 9.0) discard;
  float a = vColor.a * exp(-0.5 * d2);
  if (a < 1.0 / 255.0) discard;
  vec3 col = vColor.rgb;
  if (uFx[1].w > 0.0) col *= 1.0 - uFx[1].w * 0.5 * (0.5 + 0.5 * sin(vWorld.y * uFx[5].z + uFx[5].y * 4.0));
  fragColor = vec4(col * a, a);
}
`;

interface ProgramInfo {
  prog: WebGLProgram;
  uViewProj: WebGLUniformLocation | null;
  uAabbMin: WebGLUniformLocation | null;
  uAabbSize: WebGLUniformLocation | null;
  uInvLevels: WebGLUniformLocation | null;
  uTexMix: WebGLUniformLocation | null;
  uLitMix: WebGLUniformLocation | null;
  uCropMin: WebGLUniformLocation | null;
  uCropMax: WebGLUniformLocation | null;
  uCropOn: WebGLUniformLocation | null;
  uShadeMode: WebGLUniformLocation | null;
  uDepthRange: WebGLUniformLocation | null;
  uPointSize: WebGLUniformLocation | null;
  uFx: WebGLUniformLocation | null;
}

interface SplatProgram {
  prog: WebGLProgram;
  u: Record<string, WebGLUniformLocation | null>;
}

const SLOTS = 3;
const roundUp16 = (n: number) => (n + 15) & ~15;
const A_ORDER = 0;

export class WebGL2Renderer implements AresRenderer {
  private programs = new Map<number, ProgramInfo>();
  private normalEncoding = 0;
  private texMix = 1;          // viewport shading: 1 = textured, 0 = untextured clay
  private litMix = 1;          // 1 = lit (lambert), 0 = unlit (albedo verbatim)
  private shadeMode = 0;
  private pointSize = 2;
  private depthRange: [number, number] = [0, 1];
  private posCount = 0;
  private posBufs: (WebGLBuffer | null)[] = [null, null, null];
  private posCap = 0;
  private uvBuf: WebGLBuffer | null = null;
  private uvCap = 0;
  private nrmBuf: WebGLBuffer | null = null;
  private nrmCap = 0;
  private hasNormals = false;
  private idxBuf: WebGLBuffer | null = null;
  private idxCap = 0;
  private tex: WebGLTexture;
  private slot = 0;
  private texGen = 0;
  /** texImage2D(VideoFrame) works on modern Chrome; flips off after the first throw. */
  private videoFrameDirect = true;
  private disposed = false;
  // Infinite ground grid (analytic, fullscreen triangle) + origin tripod (interleaved pos+color VBO).
  // Default ON, matching the WebGPU path — the floor is a reference, not a tool.
  private gridOn = true;
  private gridParams: GridParams = { step0: 100, step1: 1000, fineFade: 1, floorY: 0, centerX: 0, centerZ: 0, fadeRadius: 1e4, axisLen: 300 };
  private gridProg: { prog: WebGLProgram; uInvViewProj: WebGLUniformLocation | null; uViewProj: WebGLUniformLocation | null;
    uParams: WebGLUniformLocation | null; uFade: WebGLUniformLocation | null } | null = null;
  // Model transform: folded into the MESH's viewProj only — the grid/tripod are world reference.
  private model: Float32Array | null = null;
  private tripodProg: { prog: WebGLProgram; uViewProj: WebGLUniformLocation | null } | null = null;
  private tripodVbo: WebGLBuffer | null = null;
  private tripodCount = 0;
  // Crop preview + wireframe (mesh editor) — parity with the WebGPU path.
  private crop: { min: [number, number, number]; max: [number, number, number] } | null = null;
  private wireframe = false;
  private lineIdxBuf: WebGLBuffer | null = null;
  private lineIdxCap = 0;
  private lineIndexCount = 0;
  private lastIndices: Uint32Array | null = null;
  // Gaussian splat profile (spec §6.8): data textures + per-instance order attribute.
  private splatProg: SplatProgram | null = null;
  private splatTex: WebGLTexture | null = null;
  private splatTexH = 0;
  private shTex: WebGLTexture | null = null;
  private shTexH = 0;
  private texW = 2048;
  private splatData = new Uint32Array(0);
  private shData = new Uint32Array(0);
  private orderBuf: WebGLBuffer | null = null;
  private orderCap = 0;
  private splatCount = 0;
  private splatShDegree = 0;
  private splatShTexels = 0;
  private splatParams: SplatParams = { scaleMul: 1, opacityMul: 1 };
  private fxData = new Float32Array(32);

  private constructor(private readonly gl: WebGL2RenderingContext) {
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);            // match copyExternalImageToTexture (no flip)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.disable(gl.CULL_FACE);                                 // cull off, like the WebGPU pipeline
    gl.clearColor(0.05, 0.055, 0.07, 1);

    const tex = gl.createTexture();
    if (!tex) throw new Error("createTexture failed");
    this.tex = tex;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    // 1x1 white placeholder so the first render works even before the atlas/video loads.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));

    this.getProgram(0); // compile the default program eagerly so shader errors surface at create()
    this.texW = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE) as number) || 2048;
  }

  static async create(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<WebGL2Renderer> {
    const gl = (canvas as HTMLCanvasElement).getContext("webgl2", {
      alpha: false, antialias: false, depth: true, stencil: false,
      powerPreference: "high-performance", preserveDrawingBuffer: false,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("WebGL2 unavailable");
    return new WebGL2Renderer(gl);
  }

  private compile(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type);
    if (!sh) throw new Error("createShader failed");
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) ?? "";
      gl.deleteShader(sh);
      throw new Error("GLSL compile failed: " + log);
    }
    return sh;
  }

  private getProgram(encoding: number): ProgramInfo {
    const key = encoding === 1 ? 1 : 0;
    const cached = this.programs.get(key);
    if (cached) return cached;
    const gl = this.gl;
    const vs = this.compile(gl.VERTEX_SHADER, VS(key));
    const fs = this.compile(gl.FRAGMENT_SHADER, FS);
    const prog = gl.createProgram();
    if (!prog) throw new Error("createProgram failed");
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog) ?? "";
      gl.deleteProgram(prog);
      throw new Error("GLSL link failed: " + log);
    }
    gl.useProgram(prog);
    gl.uniform1i(gl.getUniformLocation(prog, "uTex"), 0);
    const info: ProgramInfo = {
      prog,
      uViewProj: gl.getUniformLocation(prog, "uViewProj"),
      uAabbMin: gl.getUniformLocation(prog, "uAabbMin"),
      uAabbSize: gl.getUniformLocation(prog, "uAabbSize"),
      uInvLevels: gl.getUniformLocation(prog, "uInvLevels"),
      uTexMix: gl.getUniformLocation(prog, "uTexMix"),
      uLitMix: gl.getUniformLocation(prog, "uLitMix"),
      uCropMin: gl.getUniformLocation(prog, "uCropMin"),
      uCropMax: gl.getUniformLocation(prog, "uCropMax"),
      uCropOn: gl.getUniformLocation(prog, "uCropOn"),
      uShadeMode: gl.getUniformLocation(prog, "uShadeMode"),
      uDepthRange: gl.getUniformLocation(prog, "uDepthRange"),
      uPointSize: gl.getUniformLocation(prog, "uPointSize"),
      uFx: gl.getUniformLocation(prog, "uFx"),
    };
    this.programs.set(key, info);
    return info;
  }

  /** Select the per-vertex normal unpack from the superblock normal_encoding (0 = i8×4, 1 = oct16). */
  setNormalEncoding(encoding: number): void {
    this.normalEncoding = encoding === 1 ? 1 : 0;
    // Program variants are compiled lazily in render(); the attribute pointer switches with them.
  }

  /** World-space crop box for the mesh-editor preview (fragment discard); null disables. */
  setCrop(crop: { min: [number, number, number]; max: [number, number, number] } | null): void { this.crop = crop; }

  /** Wireframe mode (mesh editor): the triangle list expanded to its edges, drawn as GL_LINES. */
  setWireframe(on: boolean): void {
    this.wireframe = on;
    if (on && this.lastIndices) this.uploadLineIndices(this.lastIndices);
  }

  private uploadLineIndices(indices: Uint32Array): void {
    const lines = new Uint32Array(indices.length * 2);
    for (let t = 0, w = 0; t < indices.length; t += 3) {
      const a = indices[t]!, b = indices[t + 1]!, c = indices[t + 2]!;
      lines[w++] = a; lines[w++] = b; lines[w++] = b; lines[w++] = c; lines[w++] = c; lines[w++] = a;
    }
    [this.lineIdxBuf, this.lineIdxCap] = this.upload(this.gl.ELEMENT_ARRAY_BUFFER, this.lineIdxBuf, this.lineIdxCap, lines);
    this.lineIndexCount = lines.length;
  }

  /** Viewport shading: textured (default) vs untextured clay. Parity with the WebGPU path. */
  setTextured(on: boolean): void { this.texMix = on ? 1 : 0; }
  setLit(on: boolean): void { this.litMix = on ? 1 : 0; }
  setShadeMode(mode: number): void { this.shadeMode = mode | 0; }
  setPointSize(px: number): void { this.pointSize = Math.max(1, px); }
  setDepthRange(lo: number, hi: number): void { this.depthRange = [lo, hi]; }

  setGridParams(p: GridParams): void { this.gridParams = p; }

  setModelMatrix(m: Float32Array | null): void { this.model = m; }

  setGrid(on: boolean): void {
    this.gridOn = on;
    if (!on || this.gridProg) return;
    const gl = this.gl;
    const mk = (t: number, s: string): WebGLShader => {
      const sh = gl.createShader(t)!; gl.shaderSource(sh, s); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error("grid shader: " + gl.getShaderInfoLog(sh));
      return sh;
    };
    const link = (vsSrc: string, fsSrc: string): WebGLProgram => {
      const prog = gl.createProgram()!;
      gl.attachShader(prog, mk(gl.VERTEX_SHADER, vsSrc));
      gl.attachShader(prog, mk(gl.FRAGMENT_SHADER, fsSrc));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("grid link: " + gl.getProgramInfoLog(prog));
      return prog;
    };

    // --- Origin tripod: plain colored lines.
    const tProg = link(`#version 300 es
precision highp float;
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aColor;
uniform mat4 uViewProj; out vec3 vColor;
void main(){ vColor = aColor; gl_Position = uViewProj * vec4(aPos, 1.0); }`, `#version 300 es
precision highp float;
in vec3 vColor; out vec4 frag;
void main(){ frag = vec4(vColor, 1.0); }`);
    this.tripodProg = { prog: tProg, uViewProj: gl.getUniformLocation(tProg, "uViewProj") };
    const { data, count } = buildOriginTripod({ floorY: this.gridParams.floorY, axisLen: this.gridParams.axisLen });
    this.tripodVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tripodVbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    this.tripodCount = count;

    // --- Infinite grid: the exact math of the WGSL in renderer.ts, kept line-for-line comparable so
    // the two backends can't drift. ONE deliberate difference, and it is load-bearing: camera.ts
    // builds WebGPU-convention projections (clip z in [0,1]), but GL's window depth is z*0.5+0.5, so
    // gl_FragDepth needs that remap or the grid would depth-test against the mesh in the wrong space.
    const gp = link(`#version 300 es
precision highp float;
out vec2 vNdc;
void main(){
  vec2 p = vec2(-1.0, -1.0);
  if (gl_VertexID == 1) p = vec2(3.0, -1.0);
  else if (gl_VertexID == 2) p = vec2(-1.0, 3.0);
  vNdc = p;
  gl_Position = vec4(p, 0.0, 1.0);
}`, `#version 300 es
precision highp float;
uniform mat4 uInvViewProj;
uniform mat4 uViewProj;
uniform vec4 uParams;   // step0, step1, fineFade, floorY
uniform vec4 uFade;     // centerX, centerZ, fadeRadius, unused
in vec2 vNdc; out vec4 frag;

vec3 unproj(vec2 ndc, float z){ vec4 p = uInvViewProj * vec4(ndc, z, 1.0); return p.xyz / p.w; }

float gridAA(vec2 xz, float cell){
  vec2 c = xz / cell;
  vec2 d = max(fwidth(c), vec2(1e-8));
  vec2 l = abs(fract(c - 0.5) - 0.5) / d;
  return 1.0 - min(min(l.x, l.y), 1.0);
}

void main(){
  vec3 n = unproj(vNdc, 0.0);
  vec3 f = unproj(vNdc, 1.0);
  vec3 dir = f - n;
  float dy = abs(dir.y) < 1e-9 ? 1e-9 : dir.y;
  float t = (uParams.w - n.y) / dy;
  bool hitsPlane = t >= 0.0 && t <= 1.0 && abs(dir.y) >= 1e-9;
  vec3 p = n + clamp(t, 0.0, 1.0) * dir;

  float fine   = gridAA(p.xz, uParams.x) * uParams.z;
  float coarse = gridAA(p.xz, uParams.y);
  vec2 d = max(fwidth(p.xz), vec2(1e-8));
  float centerLine = max(1.0 - min(abs(p.z) / d.y, 1.0), 1.0 - min(abs(p.x) / d.x, 1.0));

  float a = max(max(fine, coarse), centerLine);
  vec3 col = mix(vec3(0.28, 0.28, 0.31), vec3(0.46, 0.46, 0.50), centerLine);

  float radial  = 1.0 - smoothstep(uFade.z * 0.45, uFade.z, length(p.xz - uFade.xy));
  float grazing = smoothstep(0.0, 0.06, abs(normalize(dir).y));
  a *= radial * grazing * 0.9 * (hitsPlane ? 1.0 : 0.0);
  if (a < 0.002) discard;

  vec4 cp = uViewProj * vec4(p, 1.0);
  gl_FragDepth = clamp(cp.z / cp.w, 0.0, 1.0) * 0.5 + 0.5;
  frag = vec4(col, a);
}`);
    this.gridProg = {
      prog: gp,
      uInvViewProj: gl.getUniformLocation(gp, "uInvViewProj"),
      uViewProj: gl.getUniformLocation(gp, "uViewProj"),
      uParams: gl.getUniformLocation(gp, "uParams"),
      uFade: gl.getUniformLocation(gp, "uFade"),
    };
  }

  resize(_width: number, _height: number): void {
    // The GL drawing buffer tracks canvas.width/height (set by the player before this call);
    // the viewport is (re)set from drawingBufferWidth/Height on every render.
  }

  /** Bind (creating/growing as needed) and upload into a dynamic buffer. Returns [buffer, cap]. */
  private upload(target: number, buf: WebGLBuffer | null, cap: number, data: ArrayBufferView): [WebGLBuffer, number] {
    const gl = this.gl;
    const need = data.byteLength;
    let b = buf;
    if (!b) {
      b = gl.createBuffer();
      if (!b) throw new Error("createBuffer failed");
    }
    gl.bindBuffer(target, b);
    if (cap < need) {
      cap = roundUp16(Math.max(need, Math.ceil(cap * 1.5), 256));
      gl.bufferData(target, cap, gl.DYNAMIC_DRAW);
    }
    gl.bufferSubData(target, 0, data);
    return [b, cap];
  }

  /** Upload the index buffer (persistent per GOP; per frame in intra mode). */
  uploadTopology(indices: Uint32Array): void {
    [this.idxBuf, this.idxCap] = this.upload(this.gl.ELEMENT_ARRAY_BUFFER, this.idxBuf, this.idxCap, indices);
    this.lastIndices = indices;
    if (this.wireframe) this.uploadLineIndices(indices);
  }

  uploadUVs(uvsQ: Uint16Array): void {
    [this.uvBuf, this.uvCap] = this.upload(this.gl.ARRAY_BUFFER, this.uvBuf, this.uvCap, uvsQ);
  }

  uploadNormals(normalsQ: Int8Array): void {
    [this.nrmBuf, this.nrmCap] = this.upload(this.gl.ARRAY_BUFFER, this.nrmBuf, this.nrmCap, normalsQ);
    this.hasNormals = true;
  }

  /** Write the next frame's quantized positions into the next triple-buffer slot. */
  writePositions(positionsQ: Uint16Array): void {
    const gl = this.gl;
    this.slot = (this.slot + 1) % SLOTS;
    const need = positionsQ.byteLength;
    if (this.posCap < need || !this.posBufs[this.slot]) {
      // grow all slots together so the cap stays uniform (mirrors the WebGPU renderer)
      for (let i = 0; i < SLOTS; i++) {
        const b = this.posBufs[i];
        if (b) gl.deleteBuffer(b);
        this.posBufs[i] = null;
      }
      this.posCap = roundUp16(Math.max(need, Math.ceil(this.posCap * 1.5), 256));
      for (let i = 0; i < SLOTS; i++) {
        const b = gl.createBuffer();
        if (!b) throw new Error("createBuffer failed");
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, this.posCap, gl.DYNAMIC_DRAW);
        this.posBufs[i] = b;
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBufs[this.slot]!);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, positionsQ);
    this.posCount = positionsQ.length >> 2;
  }

  setTextureFromBitmap(bitmap: ImageBitmap): void {
    this.texGen++;
    this.teximage(bitmap);
  }

  /** Upload a decoded VideoFrame: texImage2D directly where supported, else via ImageBitmap. */
  setTextureFromVideoFrame(frame: VideoFrame): void {
    const gen = ++this.texGen;
    if (this.videoFrameDirect) {
      try {
        this.teximage(frame);
        return;
      } catch {
        this.videoFrameDirect = false;
      }
    }
    // Fallback: snapshot the frame (taken synchronously by createImageBitmap while the
    // frame is still open) and upload only if no newer texture landed meanwhile.
    void createImageBitmap(frame).then((bmp) => {
      if (gen === this.texGen && !this.disposed) this.teximage(bmp);
      bmp.close();
    }).catch(() => { /* frame closed before the snapshot — keep the previous texture */ });
  }

  private teximage(src: TexImageSource): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, src);
  }

  render(viewProj: Float32Array, aabb: Aabb, invLevels: number, indexCount: number): void {
    if (this.disposed) return;
    const gl = this.gl;
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.idxBuf || !this.posBufs[this.slot] || indexCount <= 0) return;

    const p = this.getProgram(this.normalEncoding);
    gl.useProgram(p.prog);
    gl.uniformMatrix4fv(p.uViewProj, false, this.model ? multiply(viewProj, this.model) : viewProj);
    gl.uniform3f(p.uAabbMin, aabb.min[0], aabb.min[1], aabb.min[2]);
    gl.uniform3f(p.uAabbSize, aabb.max[0] - aabb.min[0], aabb.max[1] - aabb.min[1], aabb.max[2] - aabb.min[2]);
    gl.uniform1f(p.uInvLevels, invLevels);
    gl.uniform1f(p.uTexMix, this.texMix);
    gl.uniform1f(p.uLitMix, this.litMix);
    const cr = this.crop;
    gl.uniform3f(p.uCropMin, cr ? cr.min[0] : 0, cr ? cr.min[1] : 0, cr ? cr.min[2] : 0);
    gl.uniform3f(p.uCropMax, cr ? cr.max[0] : 0, cr ? cr.max[1] : 0, cr ? cr.max[2] : 0);
    gl.uniform1f(p.uCropOn, cr ? 1 : 0);
    gl.uniform1i(p.uShadeMode, this.shadeMode);
    gl.uniform2f(p.uDepthRange, this.depthRange[0], this.depthRange[1]);
    gl.uniform1f(p.uPointSize, this.pointSize);
    gl.uniform4fv(p.uFx, this.fxData);

    // Attribute pointers are re-specified per draw (cheap) so buffer growth/encoding
    // switches never leave a stale binding behind.
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBufs[this.slot]!);
    gl.vertexAttribPointer(A_POS, 3, gl.UNSIGNED_SHORT, false, 8, 0); // raw u16 → float q
    gl.enableVertexAttribArray(A_POS);

    if (this.uvBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
      gl.vertexAttribPointer(A_UV, 2, gl.UNSIGNED_SHORT, true, 4, 0); // normalized → u/65535
      gl.enableVertexAttribArray(A_UV);
    } else {
      gl.disableVertexAttribArray(A_UV);
      gl.vertexAttrib2f(A_UV, 0, 0);
    }

    if (this.nrmBuf && this.hasNormals) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.nrmBuf);
      if (this.normalEncoding === 1) gl.vertexAttribPointer(A_NRM, 2, gl.SHORT, true, 4, 0); // oct16 snorm
      else gl.vertexAttribPointer(A_NRM, 4, gl.BYTE, true, 4, 0);                            // legacy i8 snorm
      gl.enableVertexAttribArray(A_NRM);
    } else {
      gl.disableVertexAttribArray(A_NRM);
      gl.vertexAttrib4f(A_NRM, 0, 0, 0, 0); // zero normal → fragment falls back to the face normal
    }

    if (this.shadeMode === 4) {
      gl.drawArrays(gl.POINTS, 0, this.posCount);            // point-cloud view: one sprite per vertex
    } else if (this.wireframe && this.lineIdxBuf && this.lineIndexCount > 0) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineIdxBuf);
      gl.drawElements(gl.LINES, this.lineIndexCount, gl.UNSIGNED_INT, 0);
    } else {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
      gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);
    }

    // Overlay after the mesh: no depth write (never occludes), but the mesh occludes it.
    this.drawOverlay(viewProj);
  }

  /** Infinite grid + origin tripod (shared by the mesh and splat passes). Depth write off. */
  private drawOverlay(viewProj: Float32Array): void {
    const gl = this.gl;
    if (this.gridOn && this.gridProg) {
      const inv = invert(viewProj);
      if (inv) {
        const gp = this.gridParams;
        gl.useProgram(this.gridProg.prog);
        gl.uniformMatrix4fv(this.gridProg.uInvViewProj, false, inv);
        gl.uniformMatrix4fv(this.gridProg.uViewProj, false, viewProj);
        gl.uniform4f(this.gridProg.uParams, gp.step0, gp.step1, gp.fineFade, gp.floorY);
        gl.uniform4f(this.gridProg.uFade, gp.centerX, gp.centerZ, gp.fadeRadius, 0);
        // The grid shader pulls nothing per-vertex (gl_VertexID only), so the mesh's attribute
        // arrays must be OFF: a still-enabled array whose buffer is shorter than 3 vertices is an
        // INVALID_OPERATION on some drivers even when the program never reads it.
        gl.disableVertexAttribArray(A_POS);
        gl.disableVertexAttribArray(A_UV);
        gl.disableVertexAttribArray(A_NRM);
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
      }
    }
    if (this.gridOn && this.tripodProg && this.tripodVbo && this.tripodCount > 0) {
      gl.useProgram(this.tripodProg.prog);
      gl.uniformMatrix4fv(this.tripodProg.uViewProj, false, viewProj);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tripodVbo);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0); gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12); gl.enableVertexAttribArray(1);
      gl.depthMask(false);
      gl.drawArrays(gl.LINES, 0, this.tripodCount);
      gl.depthMask(true);
    }
  }

  // ---- Gaussian splat profile ---------------------------------------------------------------

  private getSplatProgram(): SplatProgram {
    if (this.splatProg) return this.splatProg;
    const gl = this.gl;
    const vs = this.compile(gl.VERTEX_SHADER, SPLAT_VS);
    const fs = this.compile(gl.FRAGMENT_SHADER, SPLAT_FS);
    const prog = gl.createProgram();
    if (!prog) throw new Error("createProgram failed");
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    gl.deleteShader(vs); gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog) ?? ""; gl.deleteProgram(prog);
      throw new Error("GLSL link failed (splat): " + log);
    }
    const names = ["uSplat", "uSh", "uTexW", "uShTexels", "uShDegree", "uView", "uProj", "uAabbMin", "uAabbSize", "uInvLevels",
      "uViewport", "uFocal", "uCamPos", "uOrtho", "uCropMin", "uCropMax", "uCropOn", "uScaleMul", "uOpacityMul", "uTexMix", "uFx"];
    const u: Record<string, WebGLUniformLocation | null> = {};
    for (const n of names) u[n] = gl.getUniformLocation(prog, n);
    gl.useProgram(prog);
    gl.uniform1i(u.uSplat!, 1);
    gl.uniform1i(u.uSh!, 2);
    this.splatProg = { prog, u };
    return this.splatProg;
  }

  /** (Re)upload an RGBA32UI data texture of `texels` texels at width texW; grows only when needed. */
  private uploadDataTexture(tex: WebGLTexture | null, curH: number, data: Uint32Array, texels: number, unit: number): [WebGLTexture, number] {
    const gl = this.gl;
    const W = this.texW;
    const H = Math.max(1, Math.ceil(texels / W));
    let t = tex;
    if (!t) { t = gl.createTexture(); if (!t) throw new Error("createTexture failed"); }
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    if (!tex || H > curH) {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, W, H, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, null);
      curH = H;
    }
    // Upload whole rows: pad the tail row from the (zero-filled) staging array.
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA_INTEGER, gl.UNSIGNED_INT, data, 0);
    return [t, curH];
  }

  uploadSplats(f: DecodedSplat): void {
    const n = f.count;
    const W = this.texW;
    // 2 texels (8 words) per splat: pos0, pos1, attr0, attr1 | attr2, 0, 0, 0
    const texels = n * 2;
    const words = Math.max(1, Math.ceil(texels / W)) * W * 4;
    if (this.splatData.length < words) this.splatData = new Uint32Array(words);
    const d = this.splatData;
    const pq = new Uint32Array(f.positionsQ.buffer, f.positionsQ.byteOffset, f.positionsQ.length >> 1);
    for (let i = 0; i < n; i++) {
      const o = i * 8;
      d[o] = pq[i * 2]!; d[o + 1] = pq[i * 2 + 1]!;
      d[o + 2] = f.attrs[i * 3]!; d[o + 3] = f.attrs[i * 3 + 1]!;
      d[o + 4] = f.attrs[i * 3 + 2]!; d[o + 5] = 0; d[o + 6] = 0; d[o + 7] = 0;
    }
    [this.splatTex, this.splatTexH] = this.uploadDataTexture(this.splatTex, this.splatTexH, d, texels, 1);

    this.splatShDegree = f.sh && f.shDegree > 0 ? f.shDegree : 0;
    if (this.splatShDegree > 0 && f.sh) {
      const stride = shStrideBytes(this.splatShDegree);       // 12 / 24 / 48 bytes
      const shTexels = Math.ceil(stride / 16);                // 1 / 2 / 3 texels per splat
      this.splatShTexels = shTexels;
      const total = n * shTexels;
      const shWords = Math.max(1, Math.ceil(total / W)) * W * 4;
      if (this.shData.length < shWords) this.shData = new Uint32Array(shWords);
      const sd = this.shData;
      const src = new Uint32Array(f.sh.buffer, f.sh.byteOffset, f.sh.byteLength >> 2);
      const wps = stride >> 2;                                // words per splat in the source
      for (let i = 0; i < n; i++) {
        const so = i * wps, to = i * shTexels * 4;
        for (let j = 0; j < shTexels * 4; j++) sd[to + j] = j < wps ? src[so + j]! : 0;
      }
      [this.shTex, this.shTexH] = this.uploadDataTexture(this.shTex, this.shTexH, sd, total, 2);
    } else if (!this.shTex) {
      [this.shTex, this.shTexH] = this.uploadDataTexture(null, 0, new Uint32Array(W * 4), 1, 2);
    }
    this.splatCount = n;
    if (!this.orderBuf || this.orderCap < n * 4) {
      const id = new Uint32Array(n);
      for (let i = 0; i < n; i++) id[i] = i;
      this.setSplatOrder(id);
    }
  }

  setSplatOrder(order: Uint32Array): void {
    [this.orderBuf, this.orderCap] = this.upload(this.gl.ARRAY_BUFFER, this.orderBuf, this.orderCap, order);
  }

  setSplatParams(p: Partial<SplatParams>): void { this.splatParams = { ...this.splatParams, ...p }; }
  setFx(data: Float32Array): void { this.fxData.set(data.subarray(0, 32)); }

  renderSplats(cam: SplatCamera, aabb: Aabb, invLevels: number, count: number): void {
    if (this.disposed) return;
    const gl = this.gl;
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const viewProjWorld = multiply(cam.proj, cam.view);
    // Overlay FIRST (splats write no depth; drawn last they cover the floor grid where opaque).
    gl.disableVertexAttribArray(A_POS); gl.disableVertexAttribArray(A_UV); gl.disableVertexAttribArray(A_NRM);
    this.drawOverlay(viewProjWorld);
    const n = Math.min(count, this.splatCount);
    if (n <= 0 || !this.splatTex || !this.shTex || !this.orderBuf) return;

    const modelView = this.model ? multiply(cam.view, this.model) : cam.view;
    const inv = invert(modelView);
    const camPos: [number, number, number] = inv ? [inv[12]!, inv[13]!, inv[14]!] : [0, 0, 0];
    const p = this.getSplatProgram();
    gl.useProgram(p.prog);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.splatTex);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.shTex);
    gl.uniform1i(p.u.uTexW!, this.texW);
    gl.uniform1i(p.u.uShTexels!, this.splatShTexels);
    gl.uniform1i(p.u.uShDegree!, this.splatShDegree);
    gl.uniformMatrix4fv(p.u.uView!, false, modelView);
    gl.uniformMatrix4fv(p.u.uProj!, false, cam.proj);
    gl.uniform3f(p.u.uAabbMin!, aabb.min[0], aabb.min[1], aabb.min[2]);
    gl.uniform3f(p.u.uAabbSize!, aabb.max[0] - aabb.min[0], aabb.max[1] - aabb.min[1], aabb.max[2] - aabb.min[2]);
    gl.uniform1f(p.u.uInvLevels!, invLevels);
    gl.uniform2f(p.u.uViewport!, cam.width, cam.height);
    gl.uniform2f(p.u.uFocal!, cam.proj[0]! * cam.width / 2, cam.proj[5]! * cam.height / 2);
    gl.uniform3f(p.u.uCamPos!, camPos[0], camPos[1], camPos[2]);
    gl.uniform1f(p.u.uOrtho!, cam.ortho ? 1 : 0);
    const cr = this.crop;
    gl.uniform3f(p.u.uCropMin!, cr ? cr.min[0] : 0, cr ? cr.min[1] : 0, cr ? cr.min[2] : 0);
    gl.uniform3f(p.u.uCropMax!, cr ? cr.max[0] : 0, cr ? cr.max[1] : 0, cr ? cr.max[2] : 0);
    gl.uniform1f(p.u.uCropOn!, cr ? 1 : 0);
    gl.uniform1f(p.u.uScaleMul!, this.splatParams.scaleMul);
    gl.uniform1f(p.u.uOpacityMul!, this.splatParams.opacityMul);
    gl.uniform1f(p.u.uTexMix!, this.texMix);
    gl.uniform4fv(p.u.uFx!, this.fxData);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.orderBuf);
    gl.vertexAttribIPointer(A_ORDER, 1, gl.UNSIGNED_INT, 0, 0);
    gl.enableVertexAttribArray(A_ORDER);
    gl.vertexAttribDivisor(A_ORDER, 1);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.vertexAttribDivisor(A_ORDER, 0);
    gl.disableVertexAttribArray(A_ORDER);
    gl.activeTexture(gl.TEXTURE0);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    if (this.lineIdxBuf) gl.deleteBuffer(this.lineIdxBuf);
    if (this.orderBuf) gl.deleteBuffer(this.orderBuf);
    if (this.splatTex) gl.deleteTexture(this.splatTex);
    if (this.shTex) gl.deleteTexture(this.shTex);
    if (this.splatProg) gl.deleteProgram(this.splatProg.prog);
    for (const b of this.posBufs) if (b) gl.deleteBuffer(b);
    if (this.uvBuf) gl.deleteBuffer(this.uvBuf);
    if (this.nrmBuf) gl.deleteBuffer(this.nrmBuf);
    if (this.idxBuf) gl.deleteBuffer(this.idxBuf);
    if (this.tripodVbo) gl.deleteBuffer(this.tripodVbo);
    if (this.tripodProg) gl.deleteProgram(this.tripodProg.prog);
    if (this.gridProg) gl.deleteProgram(this.gridProg.prog);
    gl.deleteTexture(this.tex);
    for (const [, p] of this.programs) gl.deleteProgram(p.prog);
    this.programs.clear();
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
