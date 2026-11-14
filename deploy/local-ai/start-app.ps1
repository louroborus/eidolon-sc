#Requires -Version 5.1
<#
.SYNOPSIS
    Start Eidolon-SC with no console windows: services run hidden, only the app shows.

.DESCRIPTION
    Double-clicking this (or the desktop shortcut that points at it) brings the stack
    up and opens it as an app window, the way a native program looks: no PowerShell
    windows, no browser address bar.

    What it starts, in order:
      1. llama-server on 8094 (three models, one process) - hidden
      2. stand assets on 8765 (Spine stands and the outfit picker) - hidden
      3. the backend on 8000                               - hidden
      4. the frontend on 3000, with EIDOLON_STAND_ASSETS_URL so stands load - hidden
      5. the app window: Chrome/Edge in --app mode on http://localhost:3000

    Each service runs detached with no window. Output goes to logs/desktop/, so
    nothing is lost by hiding it:
      logs/desktop/llama.log, stands.log, backend.log, frontend.log (+ .err.log)
    If a service fails to come up, this script shows what it said (it is the only
    window that appears, and only when something went wrong).

    Stop the four services with -Stop. One desktop entry does both: pressing it
    while running stops the stack, pressing it while stopped starts it.

    NOTE: ASCII only. PowerShell 5.1 decodes a .ps1 without a UTF-8 BOM as ANSI.

.PARAMETER Port
    Backend port, default 8000.

.PARAMETER FrontendPort
    Frontend port, default 3000.

.PARAMETER Stop
    Stop the three services and exit.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\local-ai\start-app.ps1
#>
[CmdletBinding()]
param(
    [int] $Port = 8000,
    [int] $FrontendPort = 3000,
    [switch] $Stop,
    [switch] $NoToggle
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = $here
$repo = Split-Path -Parent (Split-Path -Parent $here)
$logDir = Join-Path $repo 'logs\desktop'
$python = Join-Path $repo 'backend\.venv\Scripts\eidolon.exe'

function Say($text, $color = 'Gray') { Write-Host $text -ForegroundColor $color }

function Test-Port($p) {
    $c = @(netstat -ano | Select-String 'LISTENING' | Where-Object { $_.Line -match "[:.]$p\s" })
    return $c.Count -gt 0
}

function Stop-Port($p, $who) {
    $owners = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $owners) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Say "  stopped $who (pid $procId, port $p)" 'Green'
        } catch {
            Say "  could not stop $who (pid $procId): $($_.Exception.Message)" 'Red'
        }
    }
}

# One desktop entry does both: running means this press stops it, stopped means it
# starts. -Stop always stops, and -NoToggle always starts, for scripting.
# All four must be up to count as "running": a half-started stack would otherwise
# be stopped instead of repaired.
if (-not $Stop -and -not $NoToggle) {
    $running = (Test-Port $FrontendPort) -and (Test-Port $Port) -and
               (Test-Port 8094) -and (Test-Port 8765)
    if ($running) {
        Say 'Eidolon-SC is running - stopping it' 'Yellow'
        $Stop = [switch]$true
    }
}

if ($Stop) {
    Say 'Stopping Eidolon-SC' 'Cyan'
    Stop-Port $FrontendPort 'frontend'
    Stop-Port $Port 'backend'
    Stop-Port 8094 'llama-server'
    Stop-Port 8765 'stands'
    Say 'All stopped.' 'Green'
    exit 0
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# --- 1. preconditions -----------------------------------------------------
$envPath = Join-Path $repo '.env'
if (-not (Test-Path -LiteralPath $envPath)) {
    Say "No .env at $envPath" 'Red'
    Say 'Copy .env.example to .env and fill CHAT_LLM_API_KEY first.' 'Yellow'
    Read-Host 'Press Enter to close'
    exit 1
}
if (-not (Test-Path -LiteralPath $python)) {
    Say "No backend venv: $python" 'Red'
    Say 'Create it in backend/ first (uv sync).' 'Yellow'
    Read-Host 'Press Enter to close'
    exit 1
}

$standUrl = (Get-Content -LiteralPath $envPath |
    Where-Object { $_ -match '^EIDOLON_STAND_ASSETS_URL=' } |
    Select-Object -First 1) -replace '^EIDOLON_STAND_ASSETS_URL=', ''

# `eidolon serve` reads its settings from the process environment only, so load .env
# here the way `uv run --env-file=../.env` would. Existing values win, matching uv.
function Import-DotEnv($path) {
    foreach ($line in Get-Content -LiteralPath $path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) { continue }
        $name, $value = $trimmed.Split('=', 2)
        $name = $name.Trim()
        if ($name -and -not [Environment]::GetEnvironmentVariable($name)) {
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
}
Import-DotEnv $envPath

# --- 2. llama-server (8094) ----------------------------------------------
if (Test-Port 8094) {
    Say 'models: already running' 'DarkGray'
} else {
    Say 'models: starting (first load takes a while)' 'Cyan'
    $llamaPs1 = Join-Path $here 'start-llama-server.ps1'
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $llamaPs1, '-Port', '8094') `
        -RedirectStandardOutput (Join-Path $logDir 'llama.log') `
        -RedirectStandardError  (Join-Path $logDir 'llama.err.log') `
        -WorkingDirectory $here -WindowStyle Hidden | Out-Null

    $deadline = (Get-Date).AddSeconds(300)
    while ((Get-Date) -lt $deadline) {
        try { $null = Invoke-WebRequest 'http://127.0.0.1:8094/health' -UseBasicParsing -TimeoutSec 3; break }
        catch { Start-Sleep -Milliseconds 800 }
    }
    if (-not (Test-Port 8094)) {
        Say 'models did not come up. Last lines of logs/desktop/llama.err.log:' 'Red'
        Get-Content (Join-Path $logDir 'llama.err.log') -Tail 15 -ErrorAction SilentlyContinue
        Read-Host 'Press Enter to close'
        exit 1
    }
    Say 'models: ready' 'Green'
}

# --- 3. stands (8765) ----------------------------------------------------
# The call layer loads Spine stands and the outfit picker from here; without it
# calls fall back to the background image.
if (Test-Port 8765) {
    Say 'stands: already running' 'DarkGray'
} elseif (Test-Path (Join-Path $here 'rocm\serve-stands.cjs')) {
    Say 'stands: starting on 8765' 'Cyan'
    $node = 'C:\Program Files\nodejs\node.exe'
    if (-not (Test-Path $node)) { $node = 'node.exe' }
    Start-Process -FilePath $node `
        -ArgumentList @((Join-Path $here 'rocm\serve-stands.cjs'), '8765') `
        -RedirectStandardOutput (Join-Path $logDir 'stands.log') `
        -RedirectStandardError  (Join-Path $logDir 'stands.err.log') `
        -WorkingDirectory $repo -WindowStyle Hidden | Out-Null
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port 8765) { break }
        Start-Sleep -Milliseconds 400
    }
    if (Test-Port 8765) { Say 'stands: ready' 'Green' }
    else { Say 'stands: did not start; calls will show the background image' 'Yellow' }
}

# --- 4. backend (8000) ---------------------------------------------------
if (Test-Port $Port) {
    Say "backend: already running on $Port" 'DarkGray'
} else {
    Say "backend: starting on $Port" 'Cyan'
    # Set these for the child process; PowerShell 5.1 has no -Environment on Start-Process.
    $env:PORT = "$Port"
    Start-Process -FilePath $python `
        -ArgumentList @('serve') `
        -RedirectStandardOutput (Join-Path $logDir 'backend.log') `
        -RedirectStandardError  (Join-Path $logDir 'backend.err.log') `
        -WorkingDirectory (Join-Path $repo 'backend') -WindowStyle Hidden | Out-Null

    $deadline = (Get-Date).AddSeconds(180)
    $up = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-WebRequest "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 3
            $up = $true; break
        } catch { Start-Sleep -Milliseconds 800 }
    }
    if (-not $up) {
        Say 'backend did not come up. Last lines of logs/desktop/backend.err.log:' 'Red'
        Get-Content (Join-Path $logDir 'backend.err.log') -Tail 20 -ErrorAction SilentlyContinue
        Get-Content (Join-Path $logDir 'backend.log') -Tail 10 -ErrorAction SilentlyContinue
        Read-Host 'Press Enter to close'
        exit 1
    }
    Say 'backend: ready' 'Green'
}

# --- 4. frontend (3000) --------------------------------------------------
if (Test-Port $FrontendPort) {
    Say "frontend: already running on $FrontendPort" 'DarkGray'
} else {
    Say "frontend: starting on $FrontendPort" 'Cyan'
    $npm = 'C:\Program Files\nodejs\npm.cmd'
    if (-not (Test-Path $npm)) { $npm = 'npm.cmd' }
    # The app reads this at request time for stands and the bar's music (CSP too).
    $env:BACKEND_URL = "http://127.0.0.1:$Port"
    if ($standUrl) { $env:EIDOLON_STAND_ASSETS_URL = $standUrl }
    Start-Process -FilePath $npm `
        -ArgumentList @('run', 'start', '--', '--port', "$FrontendPort") `
        -RedirectStandardOutput (Join-Path $logDir 'frontend.log') `
        -RedirectStandardError  (Join-Path $logDir 'frontend.err.log') `
        -WorkingDirectory (Join-Path $repo 'frontend') -WindowStyle Hidden | Out-Null

    $deadline = (Get-Date).AddSeconds(120)
    $up = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-WebRequest "http://127.0.0.1:$FrontendPort" -UseBasicParsing -TimeoutSec 3
            $up = $true; break
        } catch { Start-Sleep -Milliseconds 800 }
    }
    if (-not $up) {
        Say 'frontend did not come up. Last lines of logs/desktop/frontend.err.log:' 'Red'
        Get-Content (Join-Path $logDir 'frontend.err.log') -Tail 20 -ErrorAction SilentlyContinue
        Read-Host 'Press Enter to close'
        exit 1
    }
    Say 'frontend: ready' 'Green'
}

# --- 5. the app window ---------------------------------------------------
$site = "http://localhost:$FrontendPort"
$chrome = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

# A dedicated profile keeps it out of the everyday browser and gives it its own icon.
$appProfile = Join-Path $env:LOCALAPPDATA 'Eidolon-SC\app-window'
New-Item -ItemType Directory -Force -Path $appProfile | Out-Null

$appArgs = @(
    "--app=$site",
    "--user-data-dir=$appProfile",
    "--window-size=1280,860",
    "--no-first-run",
    "--no-default-browser-check"
)

if (Test-Path $chrome) {
    Start-Process -FilePath $chrome -ArgumentList $appArgs | Out-Null
    Say "opened the app window (Chrome)" 'Green'
} elseif (Test-Path $edge) {
    Start-Process -FilePath $edge -ArgumentList $appArgs | Out-Null
    Say "opened the app window (Edge)" 'Green'
} else {
    Start-Process $site
    Say 'no Chrome or Edge found, opened in the default browser' 'Yellow'
}

Say ''
Say "Running: $site" 'Green'
Say 'Logs: logs\desktop\' 'DarkGray'
Say 'Stop: run with -Stop (or the Stop shortcut)' 'DarkGray'
