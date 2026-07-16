// SVFFrameExporter — dumps a Microsoft SVF/HoloVideo .mp4 to per-frame OBJ + atlas PNG for the
// ARES encoder ("mesh-fNNNNN.obj" + "atlas-fNNNNN.png" + "manifest.jsonl").
//
// USE: menu  Tools ▸ ARES ▸ SVF Exporter…  → press Play.
//
// ── TWO WAYS TO GET GEOMETRY BACK (readbackMode)
// The plugin exposes no CPU mesh path: it decodes straight into GPU buffers whose native pointers
// Unity hands it (SetUnityBuffers + a render-thread event).
//
//  MeshGraphicsBuffer (Plan B, DEFAULT on 2021.1+) — let the plugin do exactly what it normally
//    does (decode into HoloVideoObject's OWN mesh buffers, which demonstrably works: that's what you
//    see on screen), and read THAT buffer back with Mesh.GetVertexBuffer(0) + AsyncGPUReadback.
//    Requires mesh.vertexBufferTarget |= Raw BEFORE the plugin takes the pointer — setting it after
//    silently orphans the pointer, so we then ReleaseUnityBuffers() to force a re-take.
//    Bonus: the mesh reports its own stride/attribute offsets, so risk R3 disappears entirely.
//
//  ComputeBufferSubstitution (Plan A, the only option < 2021.1) — REPLACE the pointers with our own
//    ComputeBuffers so the plugin decodes into them, then ComputeBuffer.GetData(). Empirically this
//    FAILED here (2026-07-15, Unity 6000.4): the frame-0 gate found no stride putting positions
//    inside the plugin's own reported bounds — i.e. risk R2, the plugin will not write a
//    ComputeBuffer that lacks vertex-buffer bind flags. Kept for reference/fallback only.
//
// ── API NAMES ARE VERIFIED against this project's plugin sources; the compiler does NOT check the
//    strings passed to GetField/GetMethod, so `verify/check-reflection.ps1` asserts them and
//    `verify/compile-svf.ps1` builds this file against the real Unity assemblies. Run both.
//      * SVFUnityPluginInterop.{SetUnityBuffers,IssueUnityRenderModePluginEvent,SeekToFrame(ulong),
//        ReleaseUnityBuffers}
//      * SVFFileInfo texture size is fileWidth/fileHeight — textureWidth/textureHeight are on
//        SVFFrameInfo. Reading the wrong one silently yields 0.
//      * HoloVideoObject.fileInfo is PUBLIC and Unity-SERIALIZED — it holds stale values from the
//        last editor session, so a plausible frameCount never proves "the clip is open".
//
// ── HOW WE LAND ON A SPECIFIC FRAME (the non-obvious part)
// The decoder only advances while HoloVideoObject.isPlaying, and lastFrameInfo only refreshes on
// that same path — so you cannot "pause, then step". The plugin's own preview lands on a frame with
// DisplayFrame(n) = Rewind()+Play()+pause-when-frameId>=n: O(n) per frame ⇒ O(n^2) over 4911 frames.
// We do the same MINUS the rewind:
//     interop.SeekToFrame(i)          — O(1) DLL seek
//     ShouldPauseAfterPlay = true     — private field, set by reflection
//     PauseFrameID = i                — public
//     hvo.Play()                      — the plugin's own PauseOnFrame() stops us on arrival
// then wait for SVFPlaybackState.Paused. NEVER wait-for-Paused after a bare SeekToFrame: it returns
// void and never pauses, so the wait burns its whole timeout on every frame (10s x 4911 ≈ 13h).
//
// ── RISKS (the frame-0 gate proves/disproves these, DATA-ONLY — no visual judgement needed, and on
//    failure it writes gate-report.txt telling you WHICH risk fired instead of just "it failed")
//  R1 ordering: readback may race the plugin's out-of-band GPU writes → previous frame's data.
//     Knob: extraSettleFrames.
//  R2 bind flags: plugin won't write a destination lacking vertex-buffer binds → all zeros.
//     CONFIRMED for Plan A. Plan B is the answer.
//  R3 stride: Plan A must guess the vertex layout (32/20/24). Plan B asks the mesh.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

public class SVFFrameExporter : MonoBehaviour {
  public enum TargetMode {
    Auto,            // HoloVideoObject on this GameObject, else the only one in the scene
    Explicit,        // use the `hvo` field
    BatchAllInScene, // export every HoloVideoObject in turn, into outDir/<GameObject name>/
  }
  public enum Readback {
    Auto,                        // MeshGraphicsBuffer on 2021.1+, else ComputeBufferSubstitution
    MeshGraphicsBuffer,          // Plan B — read HoloVideoObject's own mesh buffer
    ComputeBufferSubstitution,   // Plan A — substitute our buffers (known to fail with R2 here)
  }

  [Header("What to export")]
  [Tooltip("Auto: the HoloVideoObject on this GameObject, else the only one in the scene (errors and "
         + "lists them if there are several).\nExplicit: use the 'hvo' field below.\n"
         + "BatchAllInScene: export every HoloVideoObject one after another into outDir/<name>/.")]
  public TargetMode targetMode = TargetMode.Auto;
  [Tooltip("Only used when targetMode = Explicit (the menu fills this in for you).")]
  public HoloVideoObject hvo;

  [Header("Output")]
  public string outDir = @"D:\SVF-export";
  public int startFrame = 0;
  [Tooltip("0 = ALL frames. A 4911-frame clip is roughly 40-90 min and 5-15 GB. "
         + "Leave at 30 for a smoke test, then set 0 for the real run.")]
  public int maxFrames = 30;
  [Tooltip("Skip atlas PNGs (geometry only). PNG encoding dominates the per-frame cost, so turning "
         + "this off makes each frame much faster — and lets the decoder clock run faster too.")]
  public bool exportTextures = true;
  [Tooltip("Delete existing mesh-f*.obj / atlas-f*.png in the output folder before exporting. Off "
         + "means two runs MIX in one folder and you cannot tell which frames came from which.")]
  public bool cleanOutDir = true;

  [Header("Decoder clock")]
  [Tooltip("Playback speed while exporting. The decoder's clock keeps running while we write files, "
         + "so at 1.0 it sails ~7 frames past whatever we asked for and the export comes out sparse. "
         + "Lower = lands exactly, but each frame takes longer to reach. Auto-tune adjusts it for you.")]
  public float exportClockScale = 0.25f;
  [Tooltip("On an overshoot, halve the clock and restart the clip. Converges on a value that steps "
         + "exactly on THIS machine, instead of trusting a hardcoded guess.")]
  public bool autoTuneClock = true;

  [Header("Logging")]
  [Tooltip("1 = one Console line per frame (with a running ETA). Raise it for long runs.")]
  public int logEveryNFrames = 1;

  [Header("Readback")]
  [Tooltip("Auto picks MeshGraphicsBuffer (Plan B) on Unity 2021.1+. ComputeBufferSubstitution "
         + "(Plan A) is the legacy path and FAILS here with risk R2 — the plugin won't write it.")]
  public Readback readbackMode = Readback.Auto;
  [Tooltip("Plan B: add GraphicsBuffer.Target.Raw to the mesh's vertex/index buffers. This CHANGES "
         + "the bind flags of the very buffer the plugin writes, and recreates it (zero-filled), so "
         + "we leave it alone by default and just read the buffer as-is. Only tick this if the gate "
         + "report says GetVertexBuffer is unavailable.")]
  public bool forceRawVertexTarget = false;

  [Header("Sync knobs (only touch these if the frame-0 gate fails)")]
  [Tooltip("Extra full frames between the plugin's render-thread write and our readback (risk R1).")]
  public int extraSettleFrames = 1;
  [Tooltip("Seconds to wait for one seeked frame to arrive before giving up on it.")]
  public float perFrameTimeoutSec = 5f;
  [Tooltip("Seconds to wait at startup for the clip to open.")]
  public float openTimeoutSec = 20f;
  [Tooltip("Plan A only: 0 = auto-probe 32/20/24 bytes-per-vertex on the first frame (risk R3). "
         + "Plan B reads the real stride from the mesh.")]
  public int strideOverride = 0;
  public int paddedVerts = 65000;
  public int paddedIndices = 195000;

  // Vertex layout, in FLOATS (not bytes). Plan B fills this from the mesh; Plan A probes it.
  struct Layout {
    public int strideF, posF, nrmF, uvF;   // -1 = attribute absent
    public bool HasN => nrmF >= 0;
    public bool HasUv => uvF >= 0;
    public override string ToString() => $"strideF={strideF} pos={posF} nrm={nrmF} uv={uvF}";
  }

  ComputeBuffer vb, ib;                        // Plan A only
  Texture2D tex; RenderTexture rt; Texture2D readTex;
  object interop;
  MethodInfo miSetBuffers, miIssueEvent, miSeek, miRelease;
  FieldInfo fiShouldPause;                     // HoloVideoObject.ShouldPauseAfterPlay — private
  Layout layout;
  StreamWriter manifest;
  bool landed;                                 // LandOnFrame result (coroutines can't return values)

  IEnumerator Start() {
    var targets = ResolveTargets();
    if (targets == null) yield break;

    bool batch = targets.Count > 1;
    for (int t = 0; t < targets.Count; t++) {
      var h = targets[t];
      string dir = batch ? Path.Combine(outDir, Sanitize(h.gameObject.name)) : outDir;
      if (batch) Debug.Log($"[svf-export] ── clip {t + 1}/{targets.Count}: {h.gameObject.name} → {dir}");
      yield return ExportOne(h, dir);
    }
    if (batch) Debug.Log($"[svf-export] ALL {targets.Count} clips done → {outDir}");
  }

  // ── target resolution ─────────────────────────────────────────────────────────────────────────
  List<HoloVideoObject> ResolveTargets() {
    var all = FindAllHvo();
    if (targetMode == TargetMode.BatchAllInScene) {
      if (all.Length == 0) { Debug.LogError("[svf-export] BatchAllInScene: no HoloVideoObject in the scene."); return null; }
      return new List<HoloVideoObject>(all);
    }
    if (targetMode == TargetMode.Explicit) {
      if (hvo == null) { Debug.LogError("[svf-export] targetMode=Explicit but 'hvo' is empty — assign it, or switch to Auto."); return null; }
      return new List<HoloVideoObject> { hvo };
    }
    var mine = GetComponent<HoloVideoObject>();
    if (mine != null) return new List<HoloVideoObject> { mine };
    if (hvo != null) return new List<HoloVideoObject> { hvo };
    if (all.Length == 1) return new List<HoloVideoObject> { all[0] };
    if (all.Length == 0) { Debug.LogError("[svf-export] Auto: no HoloVideoObject in the scene. Open the scene with your volcap, or set targetMode=Explicit."); return null; }
    Debug.LogError($"[svf-export] Auto: {all.Length} HoloVideoObjects in the scene ({string.Join(", ", Names(all))}). "
                 + "Pick one (assign 'hvo' + targetMode=Explicit) or set targetMode=BatchAllInScene to do all of them.");
    return null;
  }

  static HoloVideoObject[] FindAllHvo() {
#if UNITY_2023_1_OR_NEWER
    return UnityEngine.Object.FindObjectsByType<HoloVideoObject>(FindObjectsInactive.Include, FindObjectsSortMode.None);
#else
    return UnityEngine.Object.FindObjectsOfType<HoloVideoObject>(true);
#endif
  }
  static string[] Names(HoloVideoObject[] a) { var n = new string[a.Length]; for (int i = 0; i < a.Length; i++) n[i] = a[i].gameObject.name; return n; }
  static string Sanitize(string s) { foreach (char c in Path.GetInvalidFileNameChars()) s = s.Replace(c, '_'); return s; }

  // ── one clip ──────────────────────────────────────────────────────────────────────────────────
  IEnumerator ExportOne(HoloVideoObject h, string dir) {
    // 1. Open. `pluginInterop` is created by Initialize() ← Open() ← HoloVideoObject.Start(), and
    //    only when ShouldAutoPlay is ticked — plus Start-vs-Start order between scripts is undefined.
    //    So never assume it exists; open the clip ourselves if it doesn't.
    if (GetInterop(h) == null && !string.IsNullOrEmpty(h.Url)) {
      float wait = Time.realtimeSinceStartup + 1f;
      while (GetInterop(h) == null && Time.realtimeSinceStartup < wait) yield return null;
      if (GetInterop(h) == null) {
        Debug.Log($"[svf-export] {h.gameObject.name}: not open (ShouldAutoPlay off?) — opening {h.Url}");
        if (!h.Open(h.Url)) {
          Debug.LogError($"[svf-export] {h.gameObject.name}: Open('{h.Url}') FAILED. The Url must be a path under "
                       + "StreamingAssets, or an absolute path to the .mp4.");
          yield break;
        }
      }
    }
    float deadline = Time.realtimeSinceStartup + openTimeoutSec;
    while (GetInterop(h) == null && Time.realtimeSinceStartup < deadline) yield return null;
    interop = GetInterop(h);
    if (interop == null) {
      Debug.LogError($"[svf-export] {h.gameObject.name}: clip never opened (Url='{h.Url}').");
      yield break;
    }

    // 2. Reflect the plugin entry points.
    miSetBuffers  = interop.GetType().GetMethod("SetUnityBuffers");
    miIssueEvent  = interop.GetType().GetMethod("IssueUnityRenderModePluginEvent");
    miSeek        = interop.GetType().GetMethod("SeekToFrame");
    miRelease     = interop.GetType().GetMethod("ReleaseUnityBuffers");
    fiShouldPause = typeof(HoloVideoObject).GetField("ShouldPauseAfterPlay", BindingFlags.NonPublic | BindingFlags.Instance);
    // miSeek is looked up but deliberately UNUSED — see LandOnFrame for why seeking loses to
    // rewind-once-then-step-forward. It stays here so nobody "rediscovers" it as a fix.
    if (miSetBuffers == null || miIssueEvent == null || miRelease == null) {
      Debug.LogError($"[svf-export] plugin build differs — setBuffers={miSetBuffers != null} issueEvent={miIssueEvent != null} "
                   + $"release={miRelease != null}. Open SVFUnityPluginInterop.cs and match the names.");
      yield break;
    }
    if (fiShouldPause == null)
      Debug.LogWarning("[svf-export] HoloVideoObject.ShouldPauseAfterPlay not found — falling back to DisplayFrame() per frame, "
                     + "which rewinds+replays each time (O(n^2)). Fine for 30 frames, far too slow for the full clip.");

    bool planB = readbackMode == Readback.MeshGraphicsBuffer;
#if UNITY_2021_1_OR_NEWER
    if (readbackMode == Readback.Auto) planB = true;
#else
    if (readbackMode == Readback.MeshGraphicsBuffer) {
      Debug.LogError("[svf-export] MeshGraphicsBuffer needs Unity 2021.1+ (Mesh.GetVertexBuffer). Use ComputeBufferSubstitution.");
      yield break;
    }
#endif

    // 3. Sizes. Texture size is SVFFileInfo.fileWidth/fileHeight — NOT textureWidth (SVFFrameInfo).
    //    An earlier build read the wrong name, got 0, and then CLAMPED to 64x64 and exported garbage
    //    rather than stopping. No clamp: if we can't determine the size, that's a hard failure.
    var fi = GetFileInfo(h);
    if (fi.texW <= 0 || fi.texH <= 0) {
      Debug.LogError($"[svf-export] {h.gameObject.name}: file reports texture {fi.texW}x{fi.texH} — the clip isn't really open. Check the Url.");
      yield break;
    }
    if (fi.frameCount <= 0) { Debug.LogError($"[svf-export] {h.gameObject.name}: frameCount={fi.frameCount} — clip not open."); yield break; }

    Directory.CreateDirectory(dir);
    Cleanup();                                                               // batch: free the previous clip's
    if (!planB) {
      vb = new ComputeBuffer(paddedVerts * 8, 4, ComputeBufferType.Default); // 8 floats @ 32 B worst case
      ib = new ComputeBuffer(paddedIndices, 4, ComputeBufferType.Default);
    }
    if (exportTextures) {
      tex = new Texture2D(fi.texW, fi.texH, TextureFormat.BGRA32, false);
      rt = new RenderTexture(fi.texW, fi.texH, 0);
      readTex = new Texture2D(fi.texW, fi.texH, TextureFormat.RGBA32, false);
    }

    int end = maxFrames > 0 ? Mathf.Min(startFrame + maxFrames, fi.frameCount) : fi.frameCount;
    int count = end - startFrame;
    if (count <= 0) { Debug.LogError($"[svf-export] nothing to do: startFrame={startFrame} ≥ frameCount={fi.frameCount}."); yield break; }
    Debug.Log($"[svf-export] {h.gameObject.name}: {fi.frameCount} frames in file; exporting {startFrame}..{end - 1} "
            + $"({count} frames), tex {fi.texW}x{fi.texH}, readback={(planB ? "MeshGraphicsBuffer (Plan B)" : "ComputeBufferSubstitution (Plan A)")} → {dir}"
            + (maxFrames <= 0 ? "\n[svf-export] maxFrames=0 → FULL CLIP. Expect tens of minutes and several GB." : ""));

#if UNITY_2021_1_OR_NEWER
    if (planB) {
      // Flag the mesh buffers Raw BEFORE the first decode, so the plugin's OWN render event writes
      // an already-readable buffer during play — the exact path that makes the volcap appear on
      // screen, so we know it works.
      //
      // Do NOT decode first and flag afterwards: changing vertexBufferTarget RECREATES the GPU
      // buffer (zero-filled from HoloVideoObject's placeholder CPU arrays), and once we pause,
      // nothing ever rewrites it — Update() and FillUnityBuffers both early-out on !isPlaying, and
      // that render event is the only thing that writes these buffers. That is precisely why the
      // first Plan B attempt read back 100% zeros in BOTH the vertex and index buffers.
      //
      // HoloVideoObject allocates the mesh in PostOpenMainThreadSetup → UpdateUnityBuffers(true),
      // sized to the file's MAX vertex/index counts, so it won't be resized mid-clip.
      var mf0 = h.GetComponent<MeshFilter>();
      if (mf0 == null) { Debug.LogError("[svf-export] HoloVideoObject has no MeshFilter — can't use Plan B."); yield break; }
      float dl0 = Time.realtimeSinceStartup + 5f;
      while ((mf0.mesh == null || mf0.mesh.vertexCount <= 0) && Time.realtimeSinceStartup < dl0) yield return null;
      if (mf0.mesh == null || mf0.mesh.vertexCount <= 0) {
        Debug.Log("[svf-export] mesh not allocated yet — decoding one frame to force it.");
        RewindForClip(h);
        yield return LandOnFrame(h, startFrame);
      }
      if (!EnsureRawTarget(h)) yield break;
      yield return null;
    }
#endif

    // ---- decoder clock -----------------------------------------------------------------------
    // The decoder's clock keeps running while we do our per-frame work, so by the time we ask for
    // the next frame it has already sailed past it. Measured 2026-07-15: asked 2 → got 5, asked 6 →
    // got 13, asked 22 → got 30, i.e. a steady ~7-frame lead ≈ 230 ms ≈ exactly our per-frame cost
    // (PNG encoding dominates). Since `want = actual + 1`, every overshoot SKIPS frames — that's why
    // a 30-frame run produced 5 sparse files.
    // Fix: slow the clock so it cannot outrun a capture. The right value depends on this machine's
    // encode speed, so don't hardcode a guess — measure it and adapt (below).
    float clock = Mathf.Max(0.005f, exportClockScale);
    h.ClockScale = clock;
    Debug.Log($"[svf-export] decoder clock scale {clock:0.###}{(autoTuneClock ? " (auto-tuning on overshoot)" : "")}");

    int logEvery = Mathf.Max(1, logEveryNFrames);
    float t0 = Time.realtimeSinceStartup;
    float[] prevSample = null;
    int written = 0, skipped = 0, staleCount = 0, misseek = 0, dupes = 0, gaps = 0;
    int lastWritten = -1, tunes = 0;
    bool restart = true;

    while (restart) {
    restart = false;
    // Rewind LAST: the >= pause semantics need the decoder BEHIND the first target, and everything
    // above (mesh warm-up, a previous tuning attempt) may have consumed frames.
    RewindForClip(h);
    written = 0; skipped = 0; staleCount = 0; misseek = 0; dupes = 0; gaps = 0; lastWritten = -1;
    prevSample = null; t0 = Time.realtimeSinceStartup;
    // Truncate per attempt — a restarted run must not leave the previous attempt's rows behind.
    if (manifest != null) { try { manifest.Flush(); manifest.Close(); } catch {} }
    manifest = new StreamWriter(Path.Combine(dir, "manifest.jsonl"));
    CleanFrameFiles(dir);

    // `want` is what we ASK for; the decoder may hand us something else. Rather than pretend
    // otherwise, we name every file by the frameId we ACTUALLY got and then ask for the next one
    // after that — so an overshoot can never silently mislabel a frame, and we can't spin forever.
    int want = startFrame;
    while (want < end) {
      float tFrame = Time.realtimeSinceStartup;
      int i = want;

      yield return LandOnFrame(h, i);
      if (!landed) {
        Debug.LogWarning($"[svf-export] frame {i}: never paused within {perFrameTimeoutSec}s (state={h.GetCurrentState()}) — skipped.");
        skipped++; want = i + 1;
        // Bail rather than grind: at 5s each, 4911 doomed frames is ~7 HOURS of warnings.
        if (written == 0 && skipped >= 3) { Debug.LogError($"[svf-export] first {skipped} frames never reached Paused (state={h.GetCurrentState()}) — aborting. The clip is not really playing: check the Url opened, and that nothing else drives this HoloVideoObject."); yield break; }
        continue;
      }

      var info = GetFrameInfo(h);
      int nv = info.vertexCount, ni = info.indexCount;
      if (nv <= 0 || ni <= 0 || (!planB && (nv > paddedVerts || ni > paddedIndices))) {
        Debug.LogWarning($"[svf-export] frame {i}: counts out of range (v={nv} i={ni}; capacity {paddedVerts}/{paddedIndices}) — skipped.");
        skipped++; want = i + 1;
        if (written == 0 && skipped >= 3) { Debug.LogError($"[svf-export] first {skipped} frames all reported unusable counts — aborting. v/i of 0 means the plugin isn't reporting frame info; over capacity means raise paddedVerts/paddedIndices."); yield break; }
        continue;
      }

      int actual = info.frameId;
      if (actual != i) misseek++;
      // Always make progress, even if the decoder hands back something at/behind where we were.
      want = Math.Max(actual + 1, i + 1);

      // An overshoot means the clock outran our capture, so we SKIPPED the frames in between.
      // Halve the clock and start this clip over rather than hand back a sparse export. Only once
      // we're in steady state (written > 0) — the very first landing after a Rewind overshoots for
      // preroll reasons that no clock scale fixes. Bounded, so it cannot loop forever.
      if (autoTuneClock && written > 0 && actual > i && tunes < 6) {
        clock = Mathf.Max(0.005f, clock * 0.5f);
        h.ClockScale = clock;
        tunes++;
        Debug.Log($"[svf-export] overshoot: asked {i}, got {actual} (+{actual - i}) — the decoder is outrunning the capture. "
                + $"Halving clock to {clock:0.###} and restarting this clip (tune {tunes}/6).");
        restart = true;
        break;
      }

      if (actual <= lastWritten) { dupes++; continue; }              // decoder repeated a frame
      if (lastWritten >= 0 && actual > lastWritten + 1) gaps += actual - lastWritten - 1;

      // ---- geometry readback ----
      float[] raw = null; int[] idx = null;
      if (planB) {
#if UNITY_2021_1_OR_NEWER
        yield return ReadMeshBuffers(h, nv, ni, r => raw = r, x => idx = x);
#endif
      } else {
        miSetBuffers.Invoke(interop, new object[] { tex != null ? tex.GetNativeTexturePtr() : IntPtr.Zero,
          tex != null ? tex.width : 0, tex != null ? tex.height : 0,
          vb.GetNativeBufferPtr(), paddedVerts, ib.GetNativeBufferPtr(), paddedIndices });
        miIssueEvent.Invoke(interop, null);
        yield return new WaitForEndOfFrame();
        for (int s = 0; s < extraSettleFrames; s++) yield return null;
        if (written == 0 && strideOverride <= 0) {
          if (!ProbeStrideA(nv, info, dir)) yield break;
        } else if (written == 0) {
          layout = LayoutForStride(strideOverride);
        }
        raw = new float[nv * layout.strideF];
        vb.GetData(raw, 0, 0, raw.Length);
        idx = new int[ni];
        ib.GetData(idx, 0, 0, ni);
      }
      if (raw == null || idx == null) { Debug.LogError($"[svf-export] frame {i}: readback returned nothing — aborting."); yield break; }

      // ---- the gate: positions must sit inside the plugin's OWN reported bounds ----
      if (!BoundsOk(raw, nv, layout, info)) {
        WriteGateReport(dir, h, i, raw, idx, nv, ni, info, planB);
        Debug.LogError($"[svf-export] GATE FAIL frame {i}: positions fall outside the plugin's reported bounds.\n"
                     + $"  Wrote {Path.Combine(dir, "gate-report.txt")} — it says which risk fired and what to do.");
        yield break;
      }
      var sample = SamplePositions(raw, nv, layout);
      bool stale = prevSample != null && Same(sample, prevSample);
      if (stale) staleCount++;
      prevSample = sample;

      WriteObj(Path.Combine(dir, $"mesh-f{actual + 1:d5}.obj"), raw, nv, idx, ni, layout);

      // ---- texture ----
      if (exportTextures) {
        if (!planB) {
          Graphics.Blit(tex, rt);
        } else {
          var live = LiveTexture(h);
          if (live == null) { Debug.LogError($"[svf-export] frame {i}: HoloVideoObject has no material texture to read."); yield break; }
          Graphics.Blit(live, rt);
        }
        var prevActive = RenderTexture.active; RenderTexture.active = rt;
        readTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0); readTex.Apply();
        RenderTexture.active = prevActive;
        File.WriteAllBytes(Path.Combine(dir, $"atlas-f{actual + 1:d5}.png"), readTex.EncodeToPNG());
      }
      written++; lastWritten = actual;

      manifest.WriteLine($"{{\"frame\":{actual},\"requested\":{i},\"verts\":{nv},\"indices\":{ni},"
        + $"\"stale\":{(stale ? "true" : "false")},\"bounds\":[{info.minX},{info.minY},{info.minZ},{info.maxX},{info.maxY},{info.maxZ}]}}");

      if (written % logEvery == 0 || want >= end) {
        manifest.Flush();
        float per = (Time.realtimeSinceStartup - t0) / written;
        string eta = TimeSpan.FromSeconds(per * Math.Max(0, count - written)).ToString(@"mm\:ss");
        Debug.Log($"[svf-export] {written}/{count}  f={actual} v={nv} i={ni}"
                + $"{(stale ? " STALE" : "")}{(actual != i ? $" (asked {i})" : "")}"
                + $"  {Time.realtimeSinceStartup - tFrame:0.00}s  eta {eta}");
      }
    }
    }   // while (restart)

    manifest.Flush(); manifest.Close(); manifest = null;
    float total = Time.realtimeSinceStartup - t0;
    Debug.Log($"[svf-export] DONE {h.gameObject.name} → {dir}\n"
            + $"  {written}/{count} written, {skipped} skipped, {dupes} repeats, {gaps} gaps, {staleCount} stale, "
            + $"{misseek} off-target, clock {clock:0.###} ({tunes} tune(s)), "
            + $"{total:0.0}s ({total / Mathf.Max(1, written):0.00}s/frame)"
            + (written < count ? $"\n  INCOMPLETE: {count - written} frame(s) of the requested range are missing." : "")
            + (staleCount > 0 ? "\n  STALE frames mean readback outran the render thread (R1) — raise extraSettleFrames and re-run." : "")
            + (gaps > 0 ? $"\n  {gaps} frame(s) were never delivered — files are named by the REAL frameId, so the sequence has holes. Check the manifest." : "")
            + (misseek > 0 ? $"\n  {misseek} frame(s) came back as a different frameId than asked. Files are named by what we actually got, never by what we requested." : ""));
    Cleanup();
  }

  // ── landing on a frame ────────────────────────────────────────────────────────────────────────
  // Step forward ONE frame and pause on it.
  //
  // WHY NOT SeekToFrame: the plugin's PauseOnFrame() fires on `frameId >= PauseFrameID`, so it only
  // lands correctly when the decoder is BEHIND the target — which is exactly why the plugin's own
  // DisplayFrame() rewinds first. An earlier build called SeekToFrame(i) and then waited for Paused;
  // SeekToFrame is a void, fire-and-forget DLL call we never waited on, so `frameId >= 0` was
  // satisfied instantly and it paused wherever the decoder already happened to be (observed: asked
  // frame 0, got frame 30 — and every later frame would have landed on 30 too).
  // So: Rewind ONCE per clip (RewindForClip), then step strictly forward. We are paused while we do
  // our per-frame work, so the decoder's clock cannot run away between frames.
  IEnumerator LandOnFrame(HoloVideoObject h, int i) {
    landed = false;
    if (fiShouldPause == null) {
      h.DisplayFrame((uint)i);                      // fallback: rewind + replay every time (O(n^2))
    } else {
      h.PauseFrameID = (uint)i;
      fiShouldPause.SetValue(h, true);
      h.Play();                                     // PauseOnFrame() stops us when frameId >= i
    }
    float fDeadline = Time.realtimeSinceStartup + perFrameTimeoutSec;
    while (h.GetCurrentState() != SVFPlaybackState.Paused && Time.realtimeSinceStartup < fDeadline) yield return null;
    if (h.GetCurrentState() != SVFPlaybackState.Paused) yield break;
    yield return null;
    for (int s = 0; s < extraSettleFrames; s++) yield return null;   // R1 slack

    // The plugin writes its bound GPU buffers ONLY from the render-thread event, and HoloVideoObject
    // issues that from FillUnityBuffers `if (isInitialized && isPlaying)` — we are PAUSED, so it
    // never fires. Nothing else writes the buffer. That is why a freshly (re)created buffer read back
    // 100% zeros: SetUnityBuffers had handed the plugin the new pointer, but no event ever told it to
    // write. Issue it ourselves so the CURRENT frame lands in the bound buffers. Idempotent — the
    // plugin's own coroutine fires this every rendered frame at 60Hz over 30Hz content.
    miIssueEvent.Invoke(interop, null);
    yield return new WaitForEndOfFrame();
    for (int s = 0; s < extraSettleFrames; s++) yield return null;
    landed = true;
  }

  // The >= pause semantics need the decoder BEHIND the first target, so rewind to 0 once. If
  // startFrame > 0 the first Play() simply plays forward to it — O(startFrame) once, not per frame.
  void RewindForClip(HoloVideoObject h) {
    if (!h.Rewind()) Debug.LogWarning("[svf-export] Rewind() returned false — frame 0 may not be the true start.");
  }

  // Two runs into one folder is indistinguishable afterwards: files are named by frameId, so a
  // sparse earlier run leaves orphans that look like part of the current export. Start clean.
  void CleanFrameFiles(string dir) {
    if (!cleanOutDir || !Directory.Exists(dir)) return;
    int n = 0;
    foreach (var pat in new[] { "mesh-f*.obj", "atlas-f*.png" })
      foreach (var f in Directory.GetFiles(dir, pat)) { try { File.Delete(f); n++; } catch {} }
    if (n > 0) Debug.Log($"[svf-export] cleaned {n} file(s) from a previous run in {dir}");
  }

#if UNITY_2021_1_OR_NEWER
  // ── Plan B ────────────────────────────────────────────────────────────────────────────────────
  // Flag the mesh's GPU buffers Raw so we can read them, then make the plugin re-take the pointers.
  // Returns false on a hard, explained failure.
  bool EnsureRawTarget(HoloVideoObject h) {
    var mf = h.GetComponent<MeshFilter>();
    if (mf == null || mf.mesh == null) { Debug.LogError("[svf-export] HoloVideoObject has no MeshFilter/mesh — can't use Plan B."); return false; }
    var mesh = mf.mesh;
    if (mesh.vertexCount <= 0) { Debug.LogError("[svf-export] mesh still has no vertices — the clip never decoded a frame. Check the Url and that it plays."); return false; }

    // Default: DON'T touch vertexBufferTarget. It changes the bind flags of the exact buffer the
    // plugin writes AND recreates it zero-filled — the least-invasive read is no modification at
    // all. Only opt in if GetVertexBuffer turns out to need it.
    if (forceRawVertexTarget) {
      bool changed = false;
      if ((mesh.vertexBufferTarget & GraphicsBuffer.Target.Raw) == 0) { mesh.vertexBufferTarget |= GraphicsBuffer.Target.Raw; changed = true; }
      if ((mesh.indexBufferTarget & GraphicsBuffer.Target.Raw) == 0) { mesh.indexBufferTarget |= GraphicsBuffer.Target.Raw; changed = true; }
      // Changing the target recreates the GPU buffer, so the plugin's pointer is now stale.
      // ReleaseUnityBuffers makes TestUnityBuffersValid() false, so UpdateUnityBuffers re-calls
      // SetUnityBuffers with the fresh pointer on the next PLAYED frame (it early-outs when paused).
      if (changed) { miRelease.Invoke(interop, null); Debug.Log("[svf-export] forceRawVertexTarget: buffers recreated + ReleaseUnityBuffers issued."); }
    }

    layout = LayoutFromMesh(mesh);
    if (layout.strideF <= 0 || layout.posF < 0) {
      Debug.LogError($"[svf-export] unexpected mesh vertex layout ({layout}) — expected float32 Position in stream 0.");
      return false;
    }
    Debug.Log($"[svf-export] Plan B: mesh stride {layout.strideF * 4} B/vertex, {layout}, "
            + $"vertexBufferTarget={mesh.vertexBufferTarget} (forceRaw={forceRawVertexTarget})  (layout read from the mesh — no guessing)");
    return true;
  }

  Layout LayoutFromMesh(Mesh mesh) {
    var l = new Layout { strideF = mesh.GetVertexBufferStride(0) / 4, posF = -1, nrmF = -1, uvF = -1 };
    // Only stream 0 — HoloVideoObject builds pos+normal+uv, which Unity interleaves into one stream.
    if (mesh.HasVertexAttribute(VertexAttribute.Position) && mesh.GetVertexAttributeStream(VertexAttribute.Position) == 0)
      l.posF = mesh.GetVertexAttributeOffset(VertexAttribute.Position) / 4;
    if (mesh.HasVertexAttribute(VertexAttribute.Normal) && mesh.GetVertexAttributeStream(VertexAttribute.Normal) == 0)
      l.nrmF = mesh.GetVertexAttributeOffset(VertexAttribute.Normal) / 4;
    if (mesh.HasVertexAttribute(VertexAttribute.TexCoord0) && mesh.GetVertexAttributeStream(VertexAttribute.TexCoord0) == 0)
      l.uvF = mesh.GetVertexAttributeOffset(VertexAttribute.TexCoord0) / 4;
    return l;
  }

  IEnumerator ReadMeshBuffers(HoloVideoObject h, int nv, int ni, Action<float[]> onVerts, Action<int[]> onIdx) {
    var mf = h.GetComponent<MeshFilter>();
    var mesh = mf != null ? mf.mesh : null;
    if (mesh == null) yield break;
    // GetVertexBuffer/GetIndexBuffer hand back a NEW GraphicsBuffer each call — we own it, so dispose.
    GraphicsBuffer gvb = null, gib = null;
    try { gvb = mesh.GetVertexBuffer(0); gib = mesh.GetIndexBuffer(); }
    catch (Exception e) { Debug.LogError($"[svf-export] GetVertexBuffer/GetIndexBuffer failed: {e.Message}"); yield break; }
    if (gvb == null || gib == null) { Debug.LogError("[svf-export] mesh GPU buffers unavailable (vertexBufferTarget not Raw?)."); if (gvb != null) gvb.Dispose(); if (gib != null) gib.Dispose(); yield break; }

    var rv = AsyncGPUReadback.Request(gvb);
    var ri = AsyncGPUReadback.Request(gib);
    while (!rv.done || !ri.done) yield return null;
    if (!rv.hasError && !ri.hasError) {
      var vdata = rv.GetData<float>();
      var idata = ri.GetData<int>();
      int needV = nv * layout.strideF, needI = ni;
      if (vdata.Length >= needV && idata.Length >= needI) {
        var vArr = new float[needV]; vdata.GetSubArray(0, needV).CopyTo(vArr); onVerts(vArr);
        var iArr = new int[needI]; idata.GetSubArray(0, needI).CopyTo(iArr); onIdx(iArr);
      } else {
        Debug.LogError($"[svf-export] readback too small: verts {vdata.Length}<{needV} or idx {idata.Length}<{needI}.");
      }
    } else {
      Debug.LogError("[svf-export] AsyncGPUReadback reported an error.");
    }
    gvb.Dispose(); gib.Dispose();
  }

  static Texture LiveTexture(HoloVideoObject h) {
    var mr = h.GetComponent<MeshRenderer>();
    return mr != null && mr.material != null ? mr.material.mainTexture : null;
  }
#endif

  // ── Plan A stride probe ───────────────────────────────────────────────────────────────────────
  static Layout LayoutForStride(int strideBytes) {
    switch (strideBytes) {
      case 32: return new Layout { strideF = 8, posF = 0, nrmF = 3, uvF = 6 };   // pos+normal+uv
      case 20: return new Layout { strideF = 5, posF = 0, nrmF = -1, uvF = 3 };  // pos+uv
      case 24: return new Layout { strideF = 6, posF = 0, nrmF = 3, uvF = -1 };  // pos+normal
      default: return new Layout { strideF = 0, posF = -1, nrmF = -1, uvF = -1 };
    }
  }
  bool ProbeStrideA(int nv, FrameInfo info, string dir) {
    foreach (int s in new[] { 32, 20, 24 }) {
      var l = LayoutForStride(s);
      var raw = new float[nv * l.strideF];
      vb.GetData(raw, 0, 0, raw.Length);
      if (BoundsOk(raw, nv, l, info)) { layout = l; Debug.Log($"[svf-export] stride probe: {s} B/vertex ✓"); return true; }
    }
    // Nothing fit — say WHY, don't just fail.
    var probe = new float[nv * 8];
    vb.GetData(probe, 0, 0, probe.Length);
    WriteGateReport(dir, null, 0, probe, null, nv, 0, info, false);
    Debug.LogError("[svf-export] GATE FAIL: no stride (32/20/24) puts positions inside the plugin's reported bounds.\n"
                 + $"  Wrote {Path.Combine(dir, "gate-report.txt")} — it says which risk fired and what to do.");
    return false;
  }

  // ── the gate report: turn a failure into a diagnosis ───────────────────────────────────────────
  void WriteGateReport(string dir, HoloVideoObject h, int frame, float[] raw, int[] idx, int nv, int ni, FrameInfo info, bool planB) {
    var sb = new StringBuilder();
    var ci = CultureInfo.InvariantCulture;
    sb.AppendLine("SVF exporter — GATE REPORT");
    sb.AppendLine("Generated because geometry readback did not match the plugin's own reported bounds.");
    sb.AppendLine("This file is DATA ONLY (counts/coordinates) — nothing here is a rendered image.");
    sb.AppendLine();
    sb.AppendLine($"frame requested : {frame}   (plugin says frameId={info.frameId})");
    sb.AppendLine($"readback path   : {(planB ? "MeshGraphicsBuffer (Plan B)" : "ComputeBufferSubstitution (Plan A)")}");
    sb.AppendLine($"plugin counts   : vertexCount={nv}  indexCount={ni}");
    sb.AppendLine($"plugin bounds   : x[{info.minX.ToString(ci)}, {info.maxX.ToString(ci)}]  y[{info.minY.ToString(ci)}, {info.maxY.ToString(ci)}]  z[{info.minZ.ToString(ci)}, {info.maxZ.ToString(ci)}]");
    sb.AppendLine($"buffer floats   : {(raw != null ? raw.Length : 0)}");
    sb.AppendLine();

    // How much of the buffer is actually zero / NaN? This is what separates the risks.
    int zero = 0, nan = 0, n = raw != null ? raw.Length : 0;
    for (int k = 0; k < n; k++) { if (raw[k] == 0f) zero++; else if (float.IsNaN(raw[k]) || float.IsInfinity(raw[k])) nan++; }
    double zf = n > 0 ? (double)zero / n : 1.0;
    sb.AppendLine($"zero floats     : {zero}/{n}  ({zf * 100:0.0}%)");
    sb.AppendLine($"NaN/Inf floats  : {nan}");
    sb.AppendLine();

    sb.AppendLine("Per-stride hypothesis (position min/max over the first 2048 vertices):");
    sb.AppendLine("  stride  inBounds%   x[min,max]                y[min,max]                z[min,max]");
    foreach (int s in new[] { 32, 20, 24 }) {
      var l = LayoutForStride(s);
      if (raw == null || l.strideF == 0 || nv * l.strideF > raw.Length) { sb.AppendLine($"  {s,5}   (buffer too small to test)"); continue; }
      int cnt = Math.Min(nv, 2048), inb = 0;
      float xmn = float.MaxValue, xmx = float.MinValue, ymn = float.MaxValue, ymx = float.MinValue, zmn = float.MaxValue, zmx = float.MinValue;
      for (int v = 0; v < cnt; v++) {
        float x = raw[v * l.strideF + l.posF], y = raw[v * l.strideF + l.posF + 1], z = raw[v * l.strideF + l.posF + 2];
        xmn = Math.Min(xmn, x); xmx = Math.Max(xmx, x); ymn = Math.Min(ymn, y); ymx = Math.Max(ymx, y); zmn = Math.Min(zmn, z); zmx = Math.Max(zmx, z);
        if (InBounds(x, y, z, info)) inb++;
      }
      sb.AppendLine($"  {s,5}   {100.0 * inb / cnt,7:0.0}   [{xmn,9:0.000},{xmx,9:0.000}]  [{ymn,9:0.000},{ymx,9:0.000}]  [{zmn,9:0.000},{zmx,9:0.000}]");
    }
    sb.AppendLine();

    if (raw != null && raw.Length >= 32) {
      sb.AppendLine("First 4 vertices as raw floats (8 per row = the 32 B hypothesis):");
      for (int v = 0; v < 4 && (v + 1) * 8 <= raw.Length; v++) {
        var row = new StringBuilder("  ");
        for (int k = 0; k < 8; k++) row.Append(raw[v * 8 + k].ToString("0.0000", ci)).Append(' ');
        sb.AppendLine(row.ToString());
      }
      sb.AppendLine();
    }
    if (idx != null && idx.Length > 0) {
      int imn = int.MaxValue, imx = int.MinValue, oob = 0;
      for (int k = 0; k < idx.Length; k++) { imn = Math.Min(imn, idx[k]); imx = Math.Max(imx, idx[k]); if (idx[k] < 0 || idx[k] >= nv) oob++; }
      sb.AppendLine($"index buffer    : min={imn} max={imx} out-of-range={oob}/{idx.Length} (valid range 0..{nv - 1})");
      sb.AppendLine();
    }

    // Auto-diagnosis. This is the whole point of the file.
    sb.AppendLine("DIAGNOSIS");
    if (zf > 0.99) {
      sb.AppendLine("  The destination buffer is ~entirely ZERO: the plugin never wrote it. This is risk R2.");
      if (!planB) {
        sb.AppendLine("  → Plan A (ComputeBufferSubstitution) cannot work: the plugin refuses to write a");
        sb.AppendLine("    destination without vertex-buffer bind flags. Set readbackMode =");
        sb.AppendLine("    MeshGraphicsBuffer (Plan B) — it reads HoloVideoObject's OWN mesh buffer, which the");
        sb.AppendLine("    plugin provably writes (it's what renders on screen). Needs Unity 2021.1+; you're on 6000.");
      } else {
        sb.AppendLine("  → On Plan B this means HoloVideoObject's OWN mesh buffer read back empty. Split it:");
        sb.AppendLine("    (a) Does the volcap RENDER ON SCREEN while this runs? If NO, the clip isn't decoding");
        sb.AppendLine("        at all and no readback path can help — fix the Url/playback first.");
        sb.AppendLine("    (b) If YES, the plugin IS writing that buffer, so we're reading the wrong resource.");
        sb.AppendLine("        Try toggling forceRawVertexTarget (a Raw bind flag is what makes the buffer");
        sb.AppendLine("        compute/readback-visible; without it GetVertexBuffer may hand back a copy — but");
        sb.AppendLine("        WITH it the buffer is recreated and the plugin must re-take the pointer, which");
        sb.AppendLine("        only happens on a PLAYED frame). Flip it and compare this report's zero-fraction.");
        sb.AppendLine("    (c) Note the plugin writes ONLY from its render-thread event, and its own coroutine");
        sb.AppendLine("        issues that only while isPlaying — so a buffer recreated after the last play");
        sb.AppendLine("        stays zero forever. That is what produced the first 100%-zero Plan B report.");
      }
    } else if (nan > 0) {
      sb.AppendLine("  NaN/Inf present: the buffer holds garbage rather than a stale-but-valid frame.");
      sb.AppendLine("  → Most likely the wrong stride (R3) or an unrelated buffer. Compare the per-stride table above.");
    } else {
      sb.AppendLine("  The buffer contains real non-zero data, but no stride puts positions inside the plugin's");
      sb.AppendLine("  reported bounds. Either the layout differs (R3) or these are the wrong bytes.");
      sb.AppendLine("  → Look at the per-stride table: if one row's ranges look like a human-sized object");
      sb.AppendLine("    (roughly matching the plugin bounds), set strideOverride to that stride.");
      sb.AppendLine("  → If every row looks like noise, the pointer we read is not the decode destination.");
      sb.AppendLine("  → If ranges look plausible but shifted, suspect R1 (stale frame): raise extraSettleFrames.");
    }
    sb.AppendLine();

    try {
      Directory.CreateDirectory(dir);
      File.WriteAllText(Path.Combine(dir, "gate-report.txt"), sb.ToString());
    } catch (Exception e) { Debug.LogError($"[svf-export] couldn't write gate-report.txt: {e.Message}"); }
    Debug.Log("[svf-export] gate report:\n" + sb);
  }

  // ── helpers ───────────────────────────────────────────────────────────────────────────────────
  struct FrameInfo { public int vertexCount, indexCount, frameId; public float minX, minY, minZ, maxX, maxY, maxZ; }

  static object GetInterop(HoloVideoObject h) {
    if (h == null) return null;
    var f = typeof(HoloVideoObject).GetField("pluginInterop", BindingFlags.NonPublic | BindingFlags.Instance)
         ?? typeof(HoloVideoObject).GetField("Interop", BindingFlags.NonPublic | BindingFlags.Instance);
    return f == null ? null : f.GetValue(h);
  }

  (int frameCount, int texW, int texH) GetFileInfo(HoloVideoObject h) {
    var fi = typeof(HoloVideoObject).GetField("fileInfo", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(h);
    if (fi == null) return (0, 0, 0);
    int F(string n) { var x = fi.GetType().GetField(n); return x != null ? Convert.ToInt32(x.GetValue(fi)) : 0; }
    return (F("frameCount"), F("fileWidth"), F("fileHeight"));   // VERIFIED names — see header
  }

  FrameInfo GetFrameInfo(HoloVideoObject h) {
    var lf = typeof(HoloVideoObject).GetField("lastFrameInfo", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(h);
    if (lf == null) return new FrameInfo { frameId = -1 };
    float G(string n) { var x = lf.GetType().GetField(n); return x != null ? Convert.ToSingle(x.GetValue(lf)) : 0f; }
    return new FrameInfo {
      vertexCount = (int)G("vertexCount"), indexCount = (int)G("indexCount"), frameId = (int)G("frameId"),
      minX = G("minX"), minY = G("minY"), minZ = G("minZ"), maxX = G("maxX"), maxY = G("maxY"), maxZ = G("maxZ"),
    };
  }

  static bool InBounds(float x, float y, float z, FrameInfo b) {
    float sx = (b.maxX - b.minX) * 0.05f + 1e-4f, sy = (b.maxY - b.minY) * 0.05f + 1e-4f, sz = (b.maxZ - b.minZ) * 0.05f + 1e-4f;
    return !(x < b.minX - sx || x > b.maxX + sx || y < b.minY - sy || y > b.maxY + sy || z < b.minZ - sz || z > b.maxZ + sz);
  }
  static bool BoundsOk(float[] raw, int nv, Layout l, FrameInfo b) {
    if (raw == null || l.strideF <= 0 || l.posF < 0) return false;
    int bad = 0, n = Math.Min(nv, 2048);
    if ((long)n * l.strideF > raw.Length) return false;
    for (int v = 0; v < n; v++)
      if (!InBounds(raw[v * l.strideF + l.posF], raw[v * l.strideF + l.posF + 1], raw[v * l.strideF + l.posF + 2], b)) bad++;
    return bad < n / 20;                                          // ≤5% outliers tolerated
  }

  static float[] SamplePositions(float[] raw, int nv, Layout l) {
    var s = new float[30];
    for (int k = 0; k < 10; k++) {
      int v = (int)((long)k * (nv - 1) / 9);
      s[k * 3] = raw[v * l.strideF + l.posF]; s[k * 3 + 1] = raw[v * l.strideF + l.posF + 1]; s[k * 3 + 2] = raw[v * l.strideF + l.posF + 2];
    }
    return s;
  }
  static bool Same(float[] a, float[] b) { for (int i = 0; i < a.Length; i++) if (Math.Abs(a[i] - b[i]) > 1e-7f) return false; return true; }

  static void WriteObj(string path, float[] raw, int nv, int[] idx, int ni, Layout l) {
    var sb = new StringBuilder(nv * 64);
    var ci = CultureInfo.InvariantCulture;
    for (int v = 0; v < nv; v++) sb.Append("v ").Append(raw[v * l.strideF + l.posF].ToString(ci)).Append(' ').Append(raw[v * l.strideF + l.posF + 1].ToString(ci)).Append(' ').Append(raw[v * l.strideF + l.posF + 2].ToString(ci)).Append('\n');
    if (l.HasUv) for (int v = 0; v < nv; v++) sb.Append("vt ").Append(raw[v * l.strideF + l.uvF].ToString(ci)).Append(' ').Append(raw[v * l.strideF + l.uvF + 1].ToString(ci)).Append('\n');
    if (l.HasN) for (int v = 0; v < nv; v++) sb.Append("vn ").Append(raw[v * l.strideF + l.nrmF].ToString(ci)).Append(' ').Append(raw[v * l.strideF + l.nrmF + 1].ToString(ci)).Append(' ').Append(raw[v * l.strideF + l.nrmF + 2].ToString(ci)).Append('\n');
    for (int t = 0; t + 2 < ni; t += 3) {
      int a = idx[t] + 1, b2 = idx[t + 1] + 1, c = idx[t + 2] + 1;
      sb.Append("f ").Append(Face(a, l.HasUv, l.HasN)).Append(' ').Append(Face(b2, l.HasUv, l.HasN)).Append(' ').Append(Face(c, l.HasUv, l.HasN)).Append('\n');
    }
    File.WriteAllText(path, sb.ToString());
  }
  static string Face(int i, bool uv, bool n) => uv && n ? $"{i}/{i}/{i}" : uv ? $"{i}/{i}" : n ? $"{i}//{i}" : i.ToString();

  // Stopping Play mid-run used to leak vb+ib ("Leak Detected : Persistent allocates 2 individual
  // allocations") because the coroutine just died. Clean up however we exit, and between batch clips.
  void Cleanup() {
    if (vb != null) { vb.Dispose(); vb = null; }
    if (ib != null) { ib.Dispose(); ib = null; }
    if (rt != null) { rt.Release(); Destroy(rt); rt = null; }
    if (tex != null) { Destroy(tex); tex = null; }
    if (readTex != null) { Destroy(readTex); readTex = null; }
  }
  void OnDestroy() {
    Cleanup();
    if (manifest != null) { try { manifest.Flush(); manifest.Close(); } catch {} manifest = null; }
  }
}
