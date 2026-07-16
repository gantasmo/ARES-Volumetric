// Tools ▸ ARES ▸ SVF Exporter… — the dropdown entry point for SVFFrameExporter.
//
// INSTALL: this file MUST live in an `Editor/` folder (e.g. Assets/SVFExport/Editor/) — Unity only
// compiles UnityEditor references there, and putting it elsewhere breaks player builds.
// SVFFrameExporter.cs itself goes one level up, in Assets/SVFExport/.
//
// Why a menu item and not an EditorWindow that does the work: the export is Play-Mode-only by
// nature — GL.IssuePluginEvent + WaitForEndOfFrame are render-thread/Play-Mode semantics, and
// batchmode/EditMode do NOT drive them (runbook Plan C). So this menu only SETS UP and validates;
// you still press Play to run.

using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class SVFExporterMenu {
  const string Title = "SVF Exporter";

  static HoloVideoObject[] FindAll() {
#if UNITY_2023_1_OR_NEWER
    return Object.FindObjectsByType<HoloVideoObject>(FindObjectsInactive.Include, FindObjectsSortMode.None);
#else
    return Object.FindObjectsOfType<HoloVideoObject>(true);
#endif
  }
  static SVFFrameExporter FindExporter() {
#if UNITY_2023_1_OR_NEWER
    return Object.FindFirstObjectByType<SVFFrameExporter>(FindObjectsInactive.Include);
#else
    return Object.FindObjectOfType<SVFFrameExporter>();
#endif
  }

  [MenuItem("Tools/ARES/SVF Exporter…", false, 10)]
  public static void SetUp() {
    var all = FindAll();
    if (all.Length == 0) {
      EditorUtility.DisplayDialog(Title,
        "No HoloVideoObject in the open scene.\n\nOpen the scene that plays the volcap (the " +
        "HoloVideoExamplePrefab works), point its Url at the .mp4 you want, then run this again.", "OK");
      return;
    }

    // Decide what we're exporting BEFORE touching the scene.
    var mode = SVFFrameExporter.TargetMode.Explicit;
    HoloVideoObject pick = all[0];

    if (all.Length > 1) {
      // Prefer whatever is selected in the Hierarchy — that's the least surprising reading of
      // "export this one". Only ask when the selection doesn't disambiguate.
      var sel = Selection.activeGameObject != null ? Selection.activeGameObject.GetComponentInParent<HoloVideoObject>() : null;
      if (sel != null) {
        pick = sel;
      } else {
        var names = string.Join("\n  • ", all.Select(h => h.gameObject.name + "  —  " + Short(h.Url)));
        int c = EditorUtility.DisplayDialogComplex(Title,
          all.Length + " volcaps in this scene:\n  • " + names +
          "\n\nBatch exports every one, each into its own subfolder of outDir.\n" +
          "To export just one instead: Cancel, select it in the Hierarchy, and run this again.",
          "Batch all " + all.Length, "Cancel", "Just '" + all[0].gameObject.name + "'");
        if (c == 1) return;
        if (c == 0) mode = SVFFrameExporter.TargetMode.BatchAllInScene;
      }
    }

    var ex = FindExporter();
    bool fresh = ex == null;
    if (fresh) {
      // Live on the volcap's own GameObject when there's one obvious target, so the exporter's
      // Auto mode keeps working even if someone later clears the fields.
      var host = mode == SVFFrameExporter.TargetMode.BatchAllInScene
        ? new GameObject("ARES SVF Exporter")
        : pick.gameObject;
      ex = host.AddComponent<SVFFrameExporter>();
      Undo.RegisterCreatedObjectUndo(mode == SVFFrameExporter.TargetMode.BatchAllInScene ? host : (Object)ex, "Add SVF Exporter");
    }
    ex.targetMode = mode;
    ex.hvo = mode == SVFFrameExporter.TargetMode.BatchAllInScene ? null : pick;
    EditorUtility.SetDirty(ex);
    Selection.activeObject = ex.gameObject;
    EditorGUIUtility.PingObject(ex.gameObject);

    // Warn about the things that actually bite, rather than restating the Inspector.
    var warn = new List<string>();
    foreach (var h in (mode == SVFFrameExporter.TargetMode.BatchAllInScene ? all : new[] { pick })) {
      if (string.IsNullOrEmpty(h.Url)) warn.Add("• '" + h.gameObject.name + "' has an empty Url — it has nothing to export.");
    }
    if (ex.maxFrames == 0)
      warn.Add("• maxFrames = 0 → the FULL clip (thousands of frames, tens of minutes, several GB). Set 30 for a smoke test.");

    string what = mode == SVFFrameExporter.TargetMode.BatchAllInScene
      ? "Batching all " + all.Length + " volcaps (each into its own subfolder)."
      : "Exporting '" + pick.gameObject.name + "'.";

    EditorUtility.DisplayDialog(Title,
      what + "\n\nThe exporter is on '" + ex.gameObject.name + "' and selected in the Hierarchy — its " +
      "fields are in the Inspector now.\n\n" +
      "1. Set  outDir  to a fast disk with room (~5-15 GB for a full clip).\n" +
      "2. maxFrames = " + ex.maxFrames + "   (30 = smoke test, 0 = every frame).\n" +
      "3. Press Play. The export runs while the scene plays.\n\n" +
      (warn.Count > 0 ? "Heads up:\n" + string.Join("\n", warn) + "\n\n" : "") +
      "The Console logs one line per frame with a running ETA, then 'DONE'. It does NOT need to be " +
      "opened first — but the clip does need to open, so a wrong Url fails on frame 0 rather than " +
      "silently exporting nothing.", "OK");
  }

  static string Short(string url) => string.IsNullOrEmpty(url) ? "(no Url set)" : System.IO.Path.GetFileName(url);

  [MenuItem("Tools/ARES/Open export folder", false, 11)]
  public static void OpenOut() {
    var ex = FindExporter();
    if (ex == null || string.IsNullOrEmpty(ex.outDir)) {
      EditorUtility.DisplayDialog(Title, "No exporter in the scene yet — run Tools ▸ ARES ▸ SVF Exporter… first.", "OK");
      return;
    }
    if (!System.IO.Directory.Exists(ex.outDir)) {
      EditorUtility.DisplayDialog(Title, "Nothing exported yet:\n" + ex.outDir, "OK");
      return;
    }
    EditorUtility.RevealInFinder(ex.outDir);
  }
}
