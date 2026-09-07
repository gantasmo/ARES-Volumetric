# ARES SAM service worker. Started from the app (serve.mjs samEnsure / the editor's SAM
# row) or via `ARES.vbs sam` (hidden) — runs the local segmentation API
# (FastAPI/uvicorn, http://127.0.0.1:7263). Backend: SAM 3 (transformers, bf16) with a
# SAM v1 ViT-H fallback — see main.py. Log: tools\sam-service\sam-service.log.
# Windows PowerShell 5.1 compatible.
#   -Port <n>  listen port (default 7263 — the dev server's /sam/* proxy default).

param([int]$Port = 7263)

$ErrorActionPreference = "Continue"
$SvcDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $SvcDir "sam-service.log"
# Dedicated service env first (torch cu124 + transformers 5, created 2026-07-10); the WanGP
# venv remains a legacy fallback that can only serve the vit_h backend.
$PyCandidates = @(
    (Join-Path $SvcDir "env\Scripts\python.exe"),
    "D:\Dev\pinokio\api\wan.git\app\env\Scripts\python.exe",
    "D:\Dev\pinokio\api\wan.git\env\Scripts\python.exe"
)

try { Start-Transcript -Path $LogFile -Force | Out-Null } catch {}

function Done { try { Stop-Transcript | Out-Null } catch {} }
function Fail($msg) {
    # This script is always launched HIDDEN (serve.mjs and the launcher both pass -WindowStyle Hidden).
    # A MessageBox here is therefore invisible AND blocks the process forever — which strands serve.mjs's
    # samChild at exitCode=null so samEnsure never re-spawns (the "SAM keeps failing to start" deadlock,
    # diagnosed + fixed 2026-07-13). Log loudly to the transcript and exit so the parent detects the exit
    # and the next Start SAM cleanly re-launches.
    Write-Output "FATAL: $msg"
    Done
    exit 1
}

function SamHealth {
    try { return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 } catch { return $null }
}

# Already running? The service binds its port immediately and reports loading:true while the
# ViT-H load (~40 s) runs, so any health answer means an instance owns the port — exit instead
# of racing it (a double start used to lose the bind with WinError 10048).
$h = SamHealth
if ($h) {
    if ($h.ok) { Write-Output "SAM service already running on port $Port (device=$($h.device))." }
    else       { Write-Output "SAM service already starting on port $Port (model loading)." }
    Done
    exit 0
}

# Port bound but /health silent? An older build bound only after the model load — wait for it
# rather than starting a competitor.
$tcp = New-Object System.Net.Sockets.TcpClient
$busy = $false
try { $busy = $tcp.ConnectAsync("127.0.0.1", $Port).Wait(1500) -and $tcp.Connected } catch {} finally { $tcp.Close() }
if ($busy) {
    Write-Output "Port $Port is bound but /health is not answering yet - waiting up to 120 s for the existing instance..."
    $deadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        $h = SamHealth
        if ($h) { Write-Output "Existing SAM instance is up (ok=$($h.ok))."; Done; exit 0 }
    }
    Fail "Port $Port is occupied by a process that never answered /health. Free it (Get-NetTCPConnection -LocalPort $Port) and retry."
}

$Py = $null
foreach ($c in $PyCandidates) { if (Test-Path $c) { $Py = $c; break } }
if (-not $Py) { Fail "WanGP python not found. Looked for:`n$($PyCandidates -join "`n")`nInstall/repair the Pinokio 'wan' app, or edit `$PyCandidates in run-sam-service.ps1." }
Write-Output "Using python: $Py"

# Weights availability is a warning, not a gate: main.py tries SAM 3 first (repo-root
# sam3\ snapshot), then the ViT-H fallback, and reports failures via /health.
$Sam3Dir = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $SvcDir))) "sam3"
$Ckpt = "D:\Dev\pinokio\api\wan.git\app\ckpts\mask\sam_vit_h_4b8939_fp16.safetensors"
if (-not (Test-Path $Sam3Dir)) { Write-Output "note: SAM3 weights dir missing ($Sam3Dir) - vit_h fallback will be used." }
if (-not (Test-Path $Ckpt))    { Write-Output "note: ViT-H checkpoint missing ($Ckpt) - sam3 must load or the service reports the error via /health." }

# Run uvicorn in the foreground of this (hidden) process so the transcript captures its
# output. First start loads the model (~40 s for ViT-H, ~30-60 s for SAM 3 on the 3060);
# the port binds immediately and /health reports loading until ready.
Write-Output "Starting SAM service on http://127.0.0.1:$Port ..."
Push-Location $SvcDir
& $Py -m uvicorn main:app --host 127.0.0.1 --port $Port
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { Fail "SAM service exited with code $code. See the log." }
Done
exit 0
