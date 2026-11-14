#Requires -Version 5.1
<#
.SYNOPSIS
    Start the torch ROCm inference server (embeddings + rerank only).

.DESCRIPTION
    NOT IN USE. The project uses the Vulkan llama-server on 8094
    (start-llama-server.ps1) for all three models. Kept as a working reference.

    This server loads the two 0.6B GGUFs in deploy/local-ai/gguf with torch + ROCm
    from D:\AI\ComfyUI\venv, and serves /v1/embeddings and /v1/rerank in about
    0.19 s and 0.07 s per call. That part works.

    The judge model is NOT served here. Under ROCm/transformers the 4B runs at
    8.8 tok/s, so eidolon's full 8-question prompt takes about 21 s with the card
    to itself (Vulkan: 2.6 s). That is inside JUDGE_TIMEOUT=30 but with little
    margin, and the JSON comes back flat where eidolon needs it nested, which
    would need constrained decoding and much longer. So the judge stays on Vulkan.

    If you do run this, .env wants:
      SILICONFLOW_BASE_URL=http://127.0.0.1:8095/v1    <- this server
      LOCAL_JUDGE_BASE_URL=http://127.0.0.1:8094/v1    <- llama-server

    NOTE: this file must stay ASCII-only. PowerShell 5.1 decodes a .ps1 with no
    UTF-8 BOM as ANSI, which mangles non-ASCII text and breaks the quotes.

.PARAMETER Port
    Listen port, default 8095.

.PARAMETER Stop
    Stop whatever listens on that port and exit.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\local-ai\rocm\start-rocm-server.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\local-ai\rocm\start-rocm-server.ps1 -Stop
#>
[CmdletBinding()]
param(
    [int] $Port = 8095,
    [int] $TimeoutSeconds = 300,
    [switch] $Stop
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$server = Join-Path $root "server.py"
$python = "D:\AI\ComfyUI\venv\Scripts\python.exe"
$logDir = Join-Path $env:TEMP "eidolon-rocm"

function Get-ListeningPids {
    param([int] $LocalPort)
    $pids = @()
    try {
        $pids = @(Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique)
    }
    catch {
        $pids = @(netstat -ano |
            Select-String -Pattern "LISTENING\s+(\d+)$" |
            Where-Object { $_.Line -match "[:.]$LocalPort\s" } |
            ForEach-Object { [int]$_.Matches[0].Groups[1].Value } |
            Select-Object -Unique)
    }
    return @($pids | Where-Object { $_ -gt 0 })
}

if ($Stop) {
    $pids = Get-ListeningPids -LocalPort $Port
    if (-not $pids) {
        Write-Host "Port $Port is not listening."
        exit 0
    }
    foreach ($processId in $pids) {
        Write-Host "Stopping ROCm server on port $Port (pid $processId)"
        try { Stop-Process -Id $processId -Force -ErrorAction Stop }
        catch { Write-Warning "Could not stop pid ${processId}: $_" }
    }
    exit 0
}

foreach ($path in @($server, $python)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Not found: $path" }
}

$takenPids = Get-ListeningPids -LocalPort $Port
if ($takenPids) {
    throw "Port $Port is already used by pid $($takenPids -join ','). Run with -Stop first."
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outLog = Join-Path $logDir "rocm-$stamp.stdout.log"
$errLog = Join-Path $logDir "rocm-$stamp.stderr.log"

$serverArgs = @(
    "-u", $server,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--only", "embed", "rerank",
    "--preload"
)
Write-Host "Starting ROCm server: $python"
Write-Host "  args: $($serverArgs -join ' ')"
Write-Host "  logs: $outLog"

$proc = Start-Process -FilePath $python -ArgumentList $serverArgs -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
    -WorkingDirectory $root -WindowStyle Hidden

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -eq 200) { $ready = $true; break }
    }
    catch { Start-Sleep -Milliseconds 800 }
}

if (-not $ready) {
    Write-Warning "ROCm server not ready within $TimeoutSeconds s (pid $($proc.Id)). Last log lines:"
    if (Test-Path -LiteralPath $errLog) { Get-Content -LiteralPath $errLog -Tail 20 }
    if (Test-Path -LiteralPath $outLog) { Get-Content -LiteralPath $outLog -Tail 20 }
    exit 1
}

Write-Host "ROCm server is up on port $Port (pid $($proc.Id))."
Write-Host ""
Write-Host "Check .env:"
Write-Host "  SILICONFLOW_BASE_URL=http://127.0.0.1:$Port/v1"
Write-Host "  LOCAL_JUDGE_BASE_URL=http://127.0.0.1:8094/v1   (judge stays on llama-server)"
Write-Host ""
Write-Host "Also keep the judge running:"
Write-Host "  powershell -ExecutionPolicy Bypass -File deploy\local-ai\start-llama-server.ps1"
Write-Host ""
Write-Host "Stop: powershell -ExecutionPolicy Bypass -File <this script> -Stop -Port $Port"
