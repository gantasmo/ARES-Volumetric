# ARES 1-click launcher worker. Run via "Launch ARES Probe.vbs" (hidden, no console)
# or tools\launch-debug.cmd (visible console). Windows PowerShell 5.1 compatible.
#
# Does, in order: find Node (installs LTS via winget if missing) -> npm install if
# needed -> tsc build (non-fatal) -> start tools\serve.mjs hidden (COOP/COEP server)
# unless one is already running -> open the probe in the default browser.
# Everything is logged to tools\launch.log.

$ErrorActionPreference = "Continue"
$ToolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AresDir  = Split-Path -Parent $ToolsDir
$LogFile  = Join-Path $ToolsDir "launch.log"
$ServeJs  = Join-Path $ToolsDir "serve.mjs"
$ProbeUrl = "/apps/phase0-probe/"
$Ports    = 8137..8147

try { Start-Transcript -Path $LogFile -Force | Out-Null } catch {}
Add-Type -AssemblyName System.Windows.Forms | Out-Null

function Done { try { Stop-Transcript | Out-Null } catch {} }
function Fail($msg) {
    Write-Output "FATAL: $msg"
    Done
    [System.Windows.Forms.MessageBox]::Show(
        "$msg`n`nLog: $LogFile", "ARES Launcher",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}
function Find-AresServer {
    foreach ($p in $Ports) {
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$p/__ares" -TimeoutSec 1
            if ($r.server -eq "ares-dev") { return $p }
        } catch {}
    }
    return $null
}

# --- 1. Locate Node.js (>=18 needed; repo is tested on 24 LTS) ---------------
Write-Output "[1/4] Locating Node.js..."
$node = $null
try { $node = (Get-Command node -ErrorAction Stop).Source } catch {}
if (-not $node) {
    foreach ($c in @("$env:ProgramFiles\nodejs\node.exe", "$env:LOCALAPPDATA\Programs\nodejs\node.exe")) {
        if (Test-Path $c) { $node = $c; break }
    }
}
if (-not $node) {
    Write-Output "Node.js not found - attempting silent install via winget (a UAC prompt may appear)..."
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
    Fail "Node.js is required and could not be installed automatically. The download page has been opened - install the LTS build, then double-click the launcher again."
}
$nodeDir = Split-Path -Parent $node
$npm = Join-Path $nodeDir "npm.cmd"
if (-not (Test-Path $npm)) { $npm = "npm" }
Write-Output "node: $node ($(& $node --version))"

# --- 2. Dependencies ----------------------------------------------------------
Write-Output "[2/4] Checking dependencies..."
$stamp = Join-Path $AresDir "node_modules\.package-lock.json"
$manifest = Join-Path $AresDir "package.json"
$needInstall = -not (Test-Path $stamp)
if (-not $needInstall) {
    if ((Get-Item $manifest).LastWriteTime -gt (Get-Item $stamp).LastWriteTime) { $needInstall = $true }
}
if ($needInstall) {
    Write-Output "Running npm install..."
    Push-Location $AresDir
    & $npm install --no-fund --no-audit
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Fail "npm install failed (exit $code). See the log for output." }
} else {
    Write-Output "node_modules up to date."
}

# --- 3. Build (incremental; probe works even if this fails) -------------------
Write-Output "[3/4] Building TypeScript packages (tsc -b)..."
Push-Location $AresDir
& $npm run build
if ($LASTEXITCODE -ne 0) { Write-Output "WARNING: build failed (exit $LASTEXITCODE) - probe still works; see log." }
Pop-Location

# --- 4. Serve + open ----------------------------------------------------------
Write-Output "[4/4] Starting COOP/COEP dev server..."
$port = Find-AresServer
if ($port) {
    Write-Output "Reusing running server on port $port."
} else {
    Start-Process -FilePath $node -ArgumentList "`"$ServeJs`"" -WorkingDirectory $AresDir -WindowStyle Hidden
    for ($i = 0; $i -lt 40 -and -not $port; $i++) {
        Start-Sleep -Milliseconds 250
        $port = Find-AresServer
    }
    if (-not $port) { Fail "The dev server did not come up on ports $($Ports[0])-$($Ports[-1])." }
    Write-Output "Server up on port $port."
}
Start-Process "http://127.0.0.1:$port$ProbeUrl"
Write-Output "Opened http://127.0.0.1:$port$ProbeUrl"
Done
exit 0
