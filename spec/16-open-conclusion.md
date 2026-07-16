## 16. Open research questions and risks

These are the load-bearing unknowns. Each is tagged with the phase
([§14](#14-development-roadmap)) that resolves it and a fallback if it fails. Nothing in the shipping
path (mesh profile, Phases 0–3) depends on an unresolved *research* question — the risky ideas are
isolated behind profiles.

### 16.1 Open questions

| # | Question | Resolves in | Fallback if it fails |
|---|---|---|---|
| Q1 | Can persistent-topology correspondence be established robustly on fast/occluded motion (`dance`,`two`)? | P2 | Per-GOP re-keyframing with meshopt intra; revise size targets up (§13.6) |
| Q2 | Does the video-geometry profile (§8.5) beat binary-delta on the Pareto front? | P2 | Drop it; ship binary-delta only |
| Q3 | Is AV1 hardware decode via WebCodecs broad enough on target devices (A2)? | P0 | VP9 primary; AV1 opportunistic; document device matrix |
| Q4 | What GOP length best balances size vs seek across the corpus? | P3 | Per-capture adaptive GOP from tracking error |
| Q5 | Splat vs mesh: which per capture type, and can they share one runtime cleanly? | P4 | Ship mesh first; splat as a second profile |
| Q6 | Temporal 3DGS attribute deltas — stable enough to code as P/B frames? | P4 | Intra splat frames per GOP |
| Q7 | Blendshape/morph delivery mode for generated avatars (§15.2)? | Post-v1 | Bake to standard geometry frames |
| Q8 | Is a Matroska/MP4 mapping worth the interop for tooling (§11.9)? | Post-v1 | Keep bespoke container only |
| Q9 | Live/low-latency profile shape (§9.6)? | Post-v1 | On-demand only in v1 |

### 16.2 Risks and mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Persistent topology proves impractical broadly | Undercuts headline size claims | Medium | Isolated to a profile; re-keyframing fallback keeps ARES viable as "chunked meshopt + video texture," still beating baselines |
| WebCodecs/WebGPU device gaps | Excludes some users | Low–med | WebGL2 + VP9/AVC fallbacks; capability detection; honest device matrix |
| Encoder complexity (tracking, rate control) balloons | Slips schedule | Medium | Phase gates; ship intra-only slice (P1) first; tracking is P2 |
| Video-geometry precision issues underestimated | Wasted effort | Medium | §8.5 analysis already scopes it as experimental; guarded by ablation |
| Scope creep into the avatar pipeline | Distracts from runtime | Medium | §15 firewalled as a separate module, post-v1 |
| "Yet another format" adoption problem | Low uptake | Medium | Great DX (`<Ares/>`), open spec, converters from what people already have, glTF-subgroup alignment |

### 16.3 Assumptions register (single source of truth)

| ID | Assumption | Validated in | Status |
|---|---|---|---|
| A1 | WebGPU available on primary targets (WebGL2 fallback otherwise) | P0 | Confirmed on the reference desktop (2026-07 probe, §13.4.1); wider device matrix pending |
| A2 | WebCodecs exposes HW AV1/VP9 decode on primary targets | P0 | Confirmed on the reference desktop (all four codecs hardware, §13.4.1); wider device matrix pending |
| A3 | Most target captures have 95–99% stable connectivity within a GOP | P2 | Open |
| A4 | Atlas can be held stable across a GOP by the correspondence step | P2 | Open |
| A5 | Sparse per-vertex deltas dominate the size win | P2 | Open |

---

## 17. Conclusion

Every mature volumetric system today is either browser-native but per-frame and CPU-heavy (UVOL,
Draco-GLB, VVglTF), or temporally smart but proprietary and non-browser (Microsoft HoloVideo,
4DViews, Arcturus). **No open format combines browser-native decode, persistent topology, and
temporal geometry compression.** ARES targets exactly that gap.

The design rests on one inversion — *a frame is a compressed set of GPU state changes, not a 3D
model* — and three mechanisms that follow from it: persistent topology with I/P/B geometry frames,
hardware `WebCodecs` decode for texture (and, experimentally, geometry), and GPU-resident
triple-buffered playback. Around that core sits a conventional, proven streaming model (chunked GOPs,
a seek index, an ABR ladder) so the novel parts are contained and the risky ideas are firewalled
behind optional profiles.

This document is deliberately falsifiable. The headline numbers are labeled **[PROJECTED]** and tied
to a benchmark methodology and an honesty clause; the load-bearing unknowns are enumerated with
fallbacks; the assumptions are registered and scheduled for validation in Phase 0 before anything is
built on them. Even in the worst case — persistent topology proving impractical at scale — ARES
degrades to "one chunked container of meshopt-intra geometry plus a hardware-decoded video texture
with real seeking and ABR," which already beats the mesh-per-frame status quo on request count, CPU,
and seek. The upside case — temporal geometry working as well for humans as video prediction works
for pixels — is a genuinely new compression model for browser-native volumetric media, and a concrete,
shipping counterpart to the standardization the Khronos glTF Volumetric subgroup is beginning.

The next step is not more design. It is **Phase 0**: build the harness, benchmark the
representations, and validate A1/A2 on real devices.
