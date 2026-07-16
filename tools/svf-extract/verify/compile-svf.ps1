# Compile-check the SVF exporter against the REAL Unity 6000.4.11f1 assemblies + the actual plugin
# sources. This exists because a previous pass shipped this file with two invented method names and
# it would not compile — "it looks right" is not a check.
#
# Machine-specific paths come from the environment so this file stays portable:
#   UNITY_EDITOR_DATA  Unity's Editor\Data dir (tested on 6000.4.11f1)
#   SVF_PLUGIN_DIR     the SVFUnityPlugin\Scripts dir from your volcap Unity project
$ErrorActionPreference = "Stop"
$E    = if ($env:UNITY_EDITOR_DATA) { $env:UNITY_EDITOR_DATA } else { "C:\Program Files\Unity\Hub\Editor\6000.4.11f1\Editor\Data" }
$out  = Join-Path $env:TEMP "ares-svf-compile"
$plug = $env:SVF_PLUGIN_DIR
$tool = Split-Path -Parent $PSScriptRoot

if (-not $plug) { throw "Set SVF_PLUGIN_DIR to your SVFUnityPlugin\Scripts directory." }
if (-not (Test-Path $E)) { throw "Unity Editor Data not found at '$E'. Set UNITY_EDITOR_DATA." }
New-Item -ItemType Directory -Force -Path $out | Out-Null

# Managed\UnityEngine already contains the MODULAR UnityEditor.CoreModule.dll. Also referencing the
# legacy facade Managed\UnityEditor.dll makes every editor type resolve twice (CS0433) — that's a
# harness artifact, not a code error, so reference the modules only, like Unity itself does.
$refs = @()
$refs += (Get-ChildItem "$E\Managed\UnityEngine" -Filter *.dll -File).FullName
$refs += "$E\NetStandard\ref\2.1.0\netstandard.dll"
$rargs = $refs | ForEach-Object { "-r:$_" }

$srcs = @(
  "$tool\SVFFrameExporter.cs",
  "$tool\Editor\SVFExporterMenu.cs",
  "$plug\HoloVideoObject.cs",
  "$plug\SVFUnityPluginInterop.cs",
  "$plug\HVConductor.cs",
  "$plug\HoloVideoPreview.cs"
)
foreach ($s in $srcs) { if (-not (Test-Path $s)) { Write-Host "MISSING SOURCE: $s"; exit 2 } }

# Unity 6 defines these for an Editor-platform compile of an Editor-folder assembly.
$defs = "-define:UNITY_EDITOR;UNITY_EDITOR_WIN;UNITY_2023_1_OR_NEWER;UNITY_2022_1_OR_NEWER;UNITY_2021_1_OR_NEWER;UNITY_6000_0_OR_NEWER;UNITY_64;UNITY_STANDALONE;UNITY_STANDALONE_WIN"

$argv = @("-noconfig","-nostdlib+","-target:library","-langversion:9","-nowarn:0169,0414,0649,CS0618",$defs) + $rargs + $srcs + @("-out:$out\svf-check.dll")
& dotnet "$E\DotNetSdkRoslyn\csc.dll" $argv
Write-Host "csc exit=$LASTEXITCODE"
