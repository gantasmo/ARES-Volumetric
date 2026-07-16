# ARES 1-click demo worker. Run via a "Play ARES … Demo.vbs" (hidden) — builds if
# needed, generates the demo clip if missing, starts the COOP/COEP dev server, and
# opens the volumetric player. Log: tools\demo.log. Windows PowerShell 5.1 compatible.
#   -Src <file.ares>  open that clip (e.g. daniel-p2.ares); default = auto (daniel.ares if present).
param([string]$Src = "")

$ErrorActionPreference = "Continue"
$ToolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AresDir  = Split-Path -Parent $ToolsDir
$LogFile  = Join-Path $ToolsDir "demo.log"
$ServeJs  = Join-Path $ToolsDir "serve.mjs"
$DemoAres = Join-Path $AresDir "apps\demo\demo.ares"
$CliJs    = Join-Path $AresDir "packages\encoder\dist\cli.js"
$Ports    = 8137..8147

try { Start-Transcript -Path $LogFile -Force | Out-Null } catch {}
Add-Type -AssemblyName System.Windows.Forms | Out-Null

function Done { try { Stop-Transcript | Out-Null } catch {} }
function Fail($msg) {
    Write-Output "FATAL: $msg"
    Done
    [System.Windows.Forms.MessageBox]::Show("$msg`n`nLog: $LogFile", "ARES Demo",
        [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}
function Find-AresServer {
    foreach ($p in $Ports) {
        try { $r = Invoke-RestMethod -Uri "http://127.0.0.1:$p/__ares" -TimeoutSec 1; if ($r.server -eq "ares-dev") { return $p } } catch {}
    }
    return $null
}

# Node
$node = $null
try { $node = (Get-Command node -ErrorAction Stop).Source } catch {}
if (-not $node) { foreach ($c in @("$env:ProgramFiles\nodejs\node.exe", "$env:LOCALAPPDATA\Programs\nodejs\node.exe")) { if (Test-Path $c) { $node = $c; break } } }
if (-not $node) { Fail "Node.js not found. Double-click 'Launch ARES Probe.vbs' once first - it installs everything." }
$npm = Join-Path (Split-Path -Parent $node) "npm.cmd"
if (-not (Test-Path $npm)) { $npm = "npm" }

# Dependencies
if (-not (Test-Path (Join-Path $AresDir "node_modules\.package-lock.json"))) {
    Write-Output "Installing dependencies..."
    Push-Location $AresDir; & $npm install --no-fund --no-audit; $code = $LASTEXITCODE; Pop-Location
    if ($code -ne 0) { Fail "npm install failed (exit $code)." }
}

# Build
Write-Output "Building..."
Push-Location $AresDir; & $npm run build; $bc = $LASTEXITCODE; Pop-Location
if ($bc -ne 0) { Fail "Build failed (exit $bc). See the log." }

# Generate the demo clip if missing
if (-not (Test-Path $DemoAres)) {
    Write-Output "Generating demo.ares..."
    Push-Location $AresDir; & $node $CliJs synth -o "apps/demo/demo.ares" --shape object --frames 60 --fps 30; $sc = $LASTEXITCODE; Pop-Location
    if ($sc -ne 0) { Fail "Demo generation failed (exit $sc)." }
}

# Serve + open
$port = Find-AresServer
if (-not $port) {
    Start-Process -FilePath $node -ArgumentList "`"$ServeJs`"" -WorkingDirectory $AresDir -WindowStyle Hidden
    for ($i = 0; $i -lt 40 -and -not $port; $i++) { Start-Sleep -Milliseconds 250; $port = Find-AresServer }
}
if (-not $port) { Fail "The dev server did not come up." }
# Choose the clip: explicit -Src, else the keeper recipe (smooth 0), else the old capture, else synth.
$query = ""
if ($Src) { $query = "?src=$Src" }
elseif (Test-Path (Join-Path $AresDir "apps\demo\daniel-s0.ares")) { $query = "?src=daniel-s0.ares" }
elseif (Test-Path (Join-Path $AresDir "apps\demo\daniel.ares")) { $query = "?src=daniel.ares" }
Start-Process "http://127.0.0.1:$port/apps/demo/$query"
Write-Output "Opened http://127.0.0.1:$port/apps/demo/$query"
Done
exit 0
