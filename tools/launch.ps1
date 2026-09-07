# ARES Windows bootstrap — the only job here is getting Node.js, then handing over to
# ARES.mjs at the repo root, which is the launcher for every platform. Run via "ARES.vbs"
# (windowless) or ARES-console.cmd (visible console). Windows PowerShell 5.1 compatible.
#
#   powershell -File tools\launch.ps1 [app|probe|bench|sam] [launcher options]
#
# Everything else — dependencies, build, demo clip, dev server, browser — lives in ARES.mjs
# and is logged to tools\launch.log. Failures raise a message box, because the windowless path
# has nowhere else to report.
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $LaunchArgs)

$ErrorActionPreference = "Continue"
$ToolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AresDir  = Split-Path -Parent $ToolsDir
$LogFile  = Join-Path $ToolsDir "launch.log"
$LaunchJs = Join-Path $AresDir "ARES.mjs"

Add-Type -AssemblyName System.Windows.Forms | Out-Null
function Fail($msg) {
    Add-Content -Path $LogFile -Value "$(Get-Date -Format o) FATAL: $msg" -ErrorAction SilentlyContinue
    [System.Windows.Forms.MessageBox]::Show(
        "$msg`n`nLog: $LogFile", "ARES",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}

# --- Node.js (>= 22.15 per package.json engines; tested on 24 LTS) ------------
$node = $null
try { $node = (Get-Command node -ErrorAction Stop).Source } catch {}
if (-not $node) {
    foreach ($c in @("$env:ProgramFiles\nodejs\node.exe", "$env:LOCALAPPDATA\Programs\nodejs\node.exe")) {
        if (Test-Path $c) { $node = $c; break }
    }
}
if (-not $node) {
    $winget = $null
    try { $winget = (Get-Command winget -ErrorAction Stop).Source } catch {}
    if ($winget) {
        & $winget install -e --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
        foreach ($c in @("$env:ProgramFiles\nodejs\node.exe", "$env:LOCALAPPDATA\Programs\nodejs\node.exe")) {
            if (Test-Path $c) { $node = $c; break }
        }
    }
}
if (-not $node) {
    Start-Process "https://nodejs.org/en/download"
    Fail "Node.js is required and could not be installed automatically. The download page has been opened - install the LTS build, then start ARES again."
}
if (-not (Test-Path $LaunchJs)) { Fail "Cannot find $LaunchJs" }

# --- Hand over ----------------------------------------------------------------
# --detach: the launcher leaves the dev server running and returns, so this window can close.
$argv = @($LaunchJs, "--detach")
if ($LaunchArgs) { $argv += $LaunchArgs }
Push-Location $AresDir
& $node $argv
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { Fail "ARES could not start (exit $code)." }
exit 0
