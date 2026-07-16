# ARES 1-click bench worker. Run via "Run ARES Bench.vbs" (hidden) — builds if
# needed, runs the Phase 0 intra bench (~1-2 min), then opens the report page.
# Log: tools\bench.log. Windows PowerShell 5.1 compatible.

$ErrorActionPreference = "Continue"
$ToolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AresDir  = Split-Path -Parent $ToolsDir
$LogFile  = Join-Path $ToolsDir "bench.log"
$ServeJs  = Join-Path $ToolsDir "serve.mjs"
$Ports    = 8137..8147

try { Start-Transcript -Path $LogFile -Force | Out-Null } catch {}
Add-Type -AssemblyName System.Windows.Forms | Out-Null

function Done { try { Stop-Transcript | Out-Null } catch {} }
function Fail($msg) {
    Write-Output "FATAL: $msg"
    Done
    [System.Windows.Forms.MessageBox]::Show(
        "$msg`n`nLog: $LogFile", "ARES Bench",
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

$node = $null
try { $node = (Get-Command node -ErrorAction Stop).Source } catch {}
if (-not $node) {
    foreach ($c in @("$env:ProgramFiles\nodejs\node.exe", "$env:LOCALAPPDATA\Programs\nodejs\node.exe")) {
        if (Test-Path $c) { $node = $c; break }
    }
}
if (-not $node) { Fail "Node.js not found. Double-click 'Launch ARES Probe.vbs' once first - it installs everything." }
$npm = Join-Path (Split-Path -Parent $node) "npm.cmd"
if (-not (Test-Path $npm)) { $npm = "npm" }

if (-not (Test-Path (Join-Path $AresDir "node_modules\.package-lock.json"))) {
    Write-Output "Installing dependencies..."
    Push-Location $AresDir
    & $npm install --no-fund --no-audit
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Fail "npm install failed (exit $code)." }
}

Write-Output "Building..."
Push-Location $AresDir
& $npm run build
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "Build failed (exit $LASTEXITCODE). See the log." }

Write-Output "Running Phase 0 intra bench (this takes a minute or two)..."
& $node (Join-Path $AresDir "bench\dist\run.js")
$benchCode = $LASTEXITCODE
Pop-Location
if ($benchCode -ne 0) { Fail "Bench run failed (exit $benchCode). See the log." }

$port = Find-AresServer
if (-not $port) {
    Start-Process -FilePath $node -ArgumentList "`"$ServeJs`"" -WorkingDirectory $AresDir -WindowStyle Hidden
    for ($i = 0; $i -lt 40 -and -not $port; $i++) {
        Start-Sleep -Milliseconds 250
        $port = Find-AresServer
    }
}
if (-not $port) { Fail "The dev server did not come up." }
Start-Process "http://127.0.0.1:$port/bench/report/"
Write-Output "Opened http://127.0.0.1:$port/bench/report/"
Done
exit 0
