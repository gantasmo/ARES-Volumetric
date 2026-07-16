# The compiler never checks a string passed to GetField/GetMethod — a typo there fails at RUNTIME,
# which is exactly the class of bug that produced the NullReferenceException. Assert every reflected
# name actually exists as a DECLARATION in the plugin sources.
# Set SVF_PLUGIN_DIR to your SVFUnityPlugin\Scripts directory.
$plug = $env:SVF_PLUGIN_DIR
if (-not $plug) { throw "Set SVF_PLUGIN_DIR to your SVFUnityPlugin\Scripts directory." }
$hvo  = Get-Content "$plug\HoloVideoObject.cs" -Raw
$intp = Get-Content "$plug\SVFUnityPluginInterop.cs" -Raw

# name -> regex matching its declaration, and which source it must live in
$checks = @(
  @{ n="pluginInterop";                    src=$hvo;  rx='\bSVFUnityPluginInterop\s+pluginInterop\b';        where="HoloVideoObject field" }
  @{ n="fileInfo";                         src=$hvo;  rx='\bSVFFileInfo\s+fileInfo\b';                       where="HoloVideoObject field" }
  @{ n="lastFrameInfo";                    src=$hvo;  rx='\bSVFFrameInfo\s+lastFrameInfo\b';                 where="HoloVideoObject field" }
  @{ n="ShouldPauseAfterPlay";             src=$hvo;  rx='\bbool\s+ShouldPauseAfterPlay\b';                  where="HoloVideoObject field (private)" }
  @{ n="fileWidth";                        src=$intp; rx='\buint\s+fileWidth\s*;';                           where="SVFFileInfo field" }
  @{ n="fileHeight";                       src=$intp; rx='\buint\s+fileHeight\s*;';                          where="SVFFileInfo field" }
  @{ n="frameCount";                       src=$intp; rx='\buint\s+frameCount\s*;';                          where="SVFFileInfo field" }
  @{ n="vertexCount";                      src=$intp; rx='\buint\s+vertexCount\s*;';                         where="SVFFrameInfo field" }
  @{ n="indexCount";                       src=$intp; rx='\buint\s+indexCount\s*;';                          where="SVFFrameInfo field" }
  @{ n="frameId";                          src=$intp; rx='\buint\s+frameId\s*;';                             where="SVFFrameInfo field" }
  @{ n="SetUnityBuffers";                  src=$intp; rx='public\s+void\s+SetUnityBuffers\s*\(';             where="interop method" }
  @{ n="IssueUnityRenderModePluginEvent";  src=$intp; rx='public\s+void\s+IssueUnityRenderModePluginEvent\s*\('; where="interop method" }
  @{ n="SeekToFrame";                      src=$intp; rx='public\s+void\s+SeekToFrame\s*\(\s*ulong';         where="interop method" }
  @{ n="ReleaseUnityBuffers";              src=$intp; rx='public\s+bool\s+ReleaseUnityBuffers\s*\(';         where="interop method" }
  @{ n="minX";                             src=$intp; rx='\bdouble\s+minX\s*;';                             where="SVFFrameInfo field" }
  @{ n="minY";                             src=$intp; rx='\bdouble\s+minY\s*;';                             where="SVFFrameInfo field" }
  @{ n="minZ";                             src=$intp; rx='\bdouble\s+minZ\s*;';                             where="SVFFrameInfo field" }
  @{ n="maxX";                             src=$intp; rx='\bdouble\s+maxX\s*;';                             where="SVFFrameInfo field" }
  @{ n="maxY";                             src=$intp; rx='\bdouble\s+maxY\s*;';                             where="SVFFrameInfo field" }
  @{ n="maxZ";                             src=$intp; rx='\bdouble\s+maxZ\s*;';                             where="SVFFrameInfo field" }
)

# Every GetField("x")/GetMethod("x") literal actually present in my exporter.
$mine = Get-Content (Join-Path (Split-Path -Parent $PSScriptRoot) "SVFFrameExporter.cs") -Raw
$used = [regex]::Matches($mine, 'Get(?:Field|Method)\("([^"]+)"\)') | ForEach-Object { $_.Groups[1].Value }
$used += [regex]::Matches($mine, '\bF\("([^"]+)"\)|\bG\("([^"]+)"\)') | ForEach-Object { if ($_.Groups[1].Value) { $_.Groups[1].Value } else { $_.Groups[2].Value } }
$used = $used | Sort-Object -Unique

$fail = 0
foreach ($c in $checks) {
  $ok = [regex]::IsMatch($c.src, $c.rx)
  if (-not $ok) { $fail++ }
  "{0,-32} {1,-6} {2}" -f $c.n, $(if ($ok) { "OK" } else { "FAIL" }), $c.where
}
""
"--- reflection literals used in SVFFrameExporter.cs ---"
$known = $checks | ForEach-Object { $_.n }
foreach ($u in $used) {
  # "Interop" is a documented fallback name for the pluginInterop field on other plugin builds.
  $status = if ($known -contains $u) { "checked" } elseif ($u -eq "Interop") { "fallback-only" } else { "UNVERIFIED" }
  if ($status -eq "UNVERIFIED") { $fail++ }
  "  {0,-32} {1}" -f $u, $status
}
""
if ($fail -gt 0) { "RESULT: $fail PROBLEM(S)"; exit 1 } else { "RESULT: all reflected names verified against plugin sources"; exit 0 }
