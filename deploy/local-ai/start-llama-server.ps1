#Requires -Version 5.1
<#
.SYNOPSIS
    起仓库里 deploy/local-ai 这套本地服务：llama-server（embedding + rerank + 判断模型，
    三模型同进程、都走 /v1 兼容接口）。

.DESCRIPTION
    eidolon 的向量化、重排、判断模型都指向同一个本机地址：
      - 向量化/重排共用 SILICONFLOW_BASE_URL（embedding.py 拼 /embeddings，rerank.py 拼 /rerank）
      - 判断模型用 JUDGE_PROVIDER=local，LOCAL_JUDGE_BASE_URL 指向同一个地址（judge.py 拼 /chat/completions）
    所以只要一个 llama-server 端口（默认 8094）就够，不再需要 judge_shim.py。

    模型 preset 在 models.ini；里面的 [段名] 必须与 eidolon 的 EMBEDDING_MODEL /
    RERANK_MODEL / LOCAL_JUDGE_MODEL 一模一样（eidolon 把模型名原样放进请求体做路由）。
    models.ini 里的 model = gguf/... 是相对路径，按进程工作目录解析 —— 这个脚本用
    -WorkingDirectory 把子进程的 CWD 固定为脚本所在目录，别改成从别的目录启动。

.PARAMETER Port
    llama-server 监听端口，默认 8094。

.PARAMETER Stop
    停掉这个端口上跑着的进程，不启动任何东西。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\local-ai\start-llama-server.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\local-ai\start-llama-server.ps1 -Stop
#>
[CmdletBinding()]
param(
    [int] $Port = 8094,
    [int] $TimeoutSeconds = 180,
    [switch] $Stop
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $root "bin\llama-server\llama-server.exe"
$preset = Join-Path $root "models.ini"
$logs = Join-Path $root "logs"

function Get-ListeningPids {
    param([int] $LocalPort)
    $pids = @()
    try {
        $pids = @(Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique)
    }
    catch {
        # 受限环境下 CIM 可能被拒，退回 netstat
        $pids = @(netstat -ano |
            Select-String -Pattern "LISTENING\s+(\d+)$" |
            Where-Object { $_.Line -match "[:.]$LocalPort\s" } |
            ForEach-Object { [int]$_.Matches[0].Groups[1].Value } |
            Select-Object -Unique)
    }
    return @($pids | Where-Object { $_ -gt 0 })
}

function Stop-OnPort {
    param([int] $LocalPort, [string] $Who)
    $pids = Get-ListeningPids -LocalPort $LocalPort
    if (-not $pids) {
        Write-Host "端口 $LocalPort 本来就没在跑"
        return
    }
    foreach ($processId in $pids) {
        Write-Host "停 $Who（端口 $LocalPort，pid $processId）"
        try { Stop-Process -Id $processId -Force -ErrorAction Stop } catch { Write-Warning "停 pid $processId 失败：$_" }
    }
}

function Wait-Healthy {
    param([string] $Url, [int] $Seconds, [string] $Who)
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host "  $Who 就绪（$Url）"
                return $true
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Warning "$Who 在 $Seconds 秒内没就绪，看日志：$logs"
    return $false
}

if ($Stop) {
    Stop-OnPort -LocalPort $Port -Who "llama-server"
    exit 0
}

foreach ($path in @($exe, $preset)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "找不到 $path" }
}

$takenPids = Get-ListeningPids -LocalPort $Port
if ($takenPids) {
    throw "端口 $Port 已被 pid $($takenPids -join ',') 占用。先跑：powershell -File `"$PSCommandPath`" -Stop"
}

New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

# ── llama-server ────────────────────────────────────────────────────────
$serverLog = Join-Path $logs "llama-server-$stamp.log"
$serverOut = Join-Path $logs "llama-server-$stamp.stdout.log"
$serverArgs = @(
    "--models-preset", $preset,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--models-max", "4"
)
Write-Host "起 llama-server：$exe"
Write-Host "  参数：$($serverArgs -join ' ')"
Write-Host "  工作目录：$root（models.ini 的 gguf/... 相对路径按这里解析）"
Write-Host "  日志：$serverLog"
# models.ini 里的相对路径按 CWD 解析，必须把子进程 CWD 固定到脚本目录
$server = Start-Process -FilePath $exe -ArgumentList $serverArgs -PassThru `
    -RedirectStandardOutput $serverOut -RedirectStandardError $serverLog `
    -WorkingDirectory $root -WindowStyle Hidden

if (-not (Wait-Healthy -Url "http://127.0.0.1:$Port/health" -Seconds $TimeoutSeconds -Who "llama-server")) {
    Write-Warning "llama-server 没起来，pid $($server.Id)。最后 20 行日志："
    if (Test-Path -LiteralPath $serverLog) { Get-Content -LiteralPath $serverLog -Tail 20 }
    exit 1
}

# 三个模型是按需加载的：第一个打进来的请求要等模型加载完才返回。eidolon 的一次检索
# 是「向量化 → 取候选 → 重排」，而重排只有 3 秒预算（retry.py 的 TURN_TIMEOUT），
# 冷加载 + 首次推理会超。所以这里先把三个模型各唤一次，把它们加载进显存。
Write-Host ""
Write-Host "预热三个模型（首次加载要一会儿）…"
$warmups = @(
    @{ Name = "向量化"; Url = "/v1/embeddings"; Body = @{ model = "Qwen/Qwen3-Embedding-0.6B"; input = @("预热") } },
    @{ Name = "重排";   Url = "/v1/rerank";    Body = @{ model = "Qwen/Qwen3-Reranker-0.6B"; query = "预热"; documents = @("预热") } },
    @{ Name = "判断";   Url = "/v1/chat/completions"; Body = @{ model = "local-judge"; messages = @(@{ role = "user"; content = "ok" }); max_tokens = 4 } }
)
foreach ($w in $warmups) {
    $json = $w.Body | ConvertTo-Json -Depth 6 -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port$($w.Url)" -Method Post `
            -ContentType "application/json; charset=utf-8" `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec 300
        $sw.Stop()
        Write-Host ("  {0} 就绪（{1:N1} 秒）" -f $w.Name, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    } catch {
        $sw.Stop()
        Write-Warning ("  {0} 预热失败（{1:N1} 秒）：{2}" -f $w.Name, $sw.Elapsed.TotalSeconds, $_.Exception.Message)
    }
}

Write-Host ""
Write-Host "llama-server 已起（pid $($server.Id)）。eidolon 的 .env 里这几行覆盖成：" 
Write-Host "  SILICONFLOW_BASE_URL=http://127.0.0.1:$Port/v1"
Write-Host "  SILICONFLOW_API_KEY=fake"
Write-Host "  EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B"
Write-Host "  RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B"
Write-Host "  JUDGE_PROVIDER=local"
Write-Host "  LOCAL_JUDGE_BASE_URL=http://127.0.0.1:$Port/v1"
Write-Host "  LOCAL_JUDGE_MODEL=local-judge"
Write-Host "  JUDGE_TIMEOUT=30"
Write-Host ""
Write-Host "改完 .env 重启 eidolon，然后 uv run eidolon doctor 看 embedding / rerank / judge 三项。"

Write-Host ""
Write-Host "停服务：powershell -File `"$PSCommandPath`" -Stop -Port $Port"
Write-Host "验证：curl.exe -s http://127.0.0.1:$Port/v1/models   （三条，status 为 unloaded 属正常，按需加载）"
Write-Host "说明：judge_shim.py 保留作零改码（方案 A）回退，走 JUDGE_PROVIDER=jev + JEV_BASE_URL；"
Write-Host "      本分支默认用方案 B（JUDGE_PROVIDER=local），不需要 shim，也不占 8097。"
