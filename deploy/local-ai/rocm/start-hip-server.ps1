#Requires -Version 5.1
<#
.SYNOPSIS
    Start llama.cpp's ROCm/HIP build as a router server for the three local models.

.DESCRIPTION
    DOES NOT WORK ON THIS MACHINE. Kept as a reproducible record of the attempt;
    use start-llama-server.ps1 (Vulkan, port 8094) instead. See RESULTS.md.

    Why it fails: the RX 6750 XT is gfx1031, which AMD's Windows ROCm support
    table marks unsupported. This server starts and even lists the card, but
    reports 0 MiB free, and every spawned model instance dies with an access
    violation (0xC0000005) in load_model. That matches llama.cpp issues #26963
    (prebuilt ROCm Windows build crashes, Vulkan works) and #24836
    (hipMemGetInfo() is unreliable on Windows).

    Laying it out was still worth it, so it is left runnable for whoever has a
    supported card. It starts one router that lazily loads all three models from
    deploy/local-ai/models.ini:

      Qwen/Qwen3-Embedding-0.6B   embeddings   -> SILICONFLOW_BASE_URL
      Qwen/Qwen3-Reranker-0.6B    reranking    -> SILICONFLOW_BASE_URL
      local-judge                  chat         -> LOCAL_JUDGE_BASE_URL

    Binary and its runtime: corpus/llama-hip/ (git-ignored, downloaded by
    fetch_llama_hip.py) plus the ROCm runtime inside D:\AI\ComfyUI\venv.

    Do not trust the memory figures this build prints: it once reported
    12272 MiB free and a benchmark reported 99.4 tok/s, and neither has been
    reproducible since. RESULTS.md records that.

    NOTE: ASCII only. PowerShell 5.1 decodes a .ps1 without a UTF-8 BOM as ANSI,
    which mangles non-ASCII text and breaks quoting.

.PARAMETER Port
    Listen port, default 8097.

.PARAMETER Stop
    Stop whatever listens on that port and exit.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\local-ai\rocm\start-hip-server.ps1
#>
[CmdletBinding()]
param(
    [int] $Port = 8097,
    [int] $TimeoutSeconds = 300,
    [string] $Preset = "",
    [switch] $Stop
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $here "..\..\..")
$exe = Join-Path $repo "corpus\llama-hip\llama-server.exe"
$presetPath = if ($Preset) { $Preset } else { Join-Path $repo "deploy\local-ai\models.ini" }
$workDir = Join-Path $repo "deploy\local-ai"   # models.ini uses gguf/... relative paths
$venv = "D:\AI\ComfyUI\venv\Lib\site-packages"
$rocmCore = Join-Path $venv "_rocm_sdk_core"
$rocmLibs = Join-Path $venv "_rocm_sdk_libraries_gfx103X_dgpu"
$logDir = Join-Path $env:TEMP "eidolon-hip"

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
        Write-Host "Stopping HIP server on port $Port (pid $processId)"
        # Router mode spawns one child server per loaded model. Collect the children
        # before the parent dies, then stop those too; matching on the process name
        # would also kill the unrelated Vulkan server on 8094.
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$processId" -ErrorAction SilentlyContinue |
            Where-Object { $_.ExecutablePath -like "*corpus\llama-hip\*" } |
            Select-Object -ExpandProperty ProcessId)
        try { Stop-Process -Id $processId -Force -ErrorAction Stop }
        catch { Write-Warning "Could not stop pid ${processId}: $_" }
        foreach ($child in $children) {
            Write-Host "Stopping child pid $child"
            try { Stop-Process -Id $child -Force -ErrorAction Stop } catch { }
        }
    }
    exit 0
}

foreach ($path in @($exe, $presetPath, $rocmCore)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Not found: $path" }
}

$takenPids = Get-ListeningPids -LocalPort $Port
if ($takenPids) {
    throw "Port $Port is already used by pid $($takenPids -join ','). Run with -Stop first."
}

# Warn loudly about anything else holding the card: the 4B needs the VRAM to itself.
$others = @()
foreach ($p in @(8094, 8095, 8096)) {
    $pids = Get-ListeningPids -LocalPort $p
    if ($pids) { $others += "port $p (pid $($pids -join ','))" }
}
if ($others) {
    Write-Warning "Another local model server is running: $($others -join ', ')."
    Write-Warning "They will share the 12 GB card and the judge will fall to ~9 tok/s,"
    Write-Warning "overrunning JUDGE_TIMEOUT=30. Stop them first."
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outLog = Join-Path $logDir "hip-$stamp.stdout.log"
$errLog = Join-Path $logDir "hip-$stamp.stderr.log"

$serverArgs = @(
    "--models-preset", $presetPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--models-max", "4"
)

# ggml-hip.dll needs the ROCm runtime; the venv ships one, so point PATH/HIP_PATH at it.
$env:HIP_PATH = $rocmCore
$env:PATH = "$($rocmCore)\bin;$($rocmLibs)\bin;$(Split-Path -Parent $exe);$env:PATH"

Write-Host "Starting llama.cpp HIP server"
Write-Host "  exe:    $exe"
Write-Host "  preset: $presetPath"
Write-Host "  cwd:    $workDir"
Write-Host "  logs:   $outLog"

$proc = Start-Process -FilePath $exe -ArgumentList $serverArgs -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
    -WorkingDirectory $workDir -WindowStyle Hidden

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
    Write-Warning "HIP server not ready within $TimeoutSeconds s (pid $($proc.Id)). Last log lines:"
    foreach ($f in @($errLog, $outLog)) {
        if (Test-Path -LiteralPath $f) { Get-Content -LiteralPath $f -Tail 25 }
    }
    exit 1
}

Write-Host "HIP server is up on port $Port (pid $($proc.Id))."
Write-Host ""
Write-Host "WARNING: this build does not work on this machine (gfx1031 is not supported"
Write-Host "by AMD's Windows ROCm). Loading a model will crash. Use start-llama-server.ps1."
Write-Host ""
Write-Host "If it does work on your card, point .env at it:"
Write-Host "  SILICONFLOW_BASE_URL=http://127.0.0.1:$Port/v1"
Write-Host "  LOCAL_JUDGE_BASE_URL=http://127.0.0.1:$Port/v1"
Write-Host ""
Write-Host "Models load on first use (embedding, rerank, judge)."
Write-Host "Stop: powershell -ExecutionPolicy Bypass -File <this script> -Stop -Port $Port"
