# Freight Disruption Digital Twin - one-click public link (Windows).
# Started by "Start Freight Twin.bat". Starts the backend and a Cloudflare Quick Tunnel in their own
# windows (or reuses them if already running), then shows the public link, copies it and opens it.
$ErrorActionPreference = "Stop"
$Backend = Join-Path $HOME "Desktop\backend"
$Python  = Join-Path $Backend ".venv\Scripts\python.exe"
$Health  = "http://127.0.0.1:8000/api/health"
$Metrics = "127.0.0.1:20241"

function Test-Backend {
    try { Invoke-RestMethod $Health -TimeoutSec 3 | Out-Null; return $true } catch { return $false }
}
function Get-TunnelLink {
    try {
        $info = Invoke-RestMethod "http://$Metrics/quicktunnel" -TimeoutSec 3
        if ($info.hostname) { return "https://" + $info.hostname }
    } catch { }
    return $null
}
function Start-Window([string]$Title, [string]$Command) {
    # -EncodedCommand avoids every quoting problem with spaces in paths.
    $script  = "`$Host.UI.RawUI.WindowTitle = '$Title'; Write-Host 'Keep this window open while people use the app.' -ForegroundColor Yellow; $Command"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    Start-Process powershell -ArgumentList "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded | Out-Null
}
function Find-Cloudflared {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $found = Get-ChildItem "C:\Program Files\cloudflared", "C:\Program Files (x86)\cloudflared", "$env:LOCALAPPDATA\Microsoft\WinGet" -Filter cloudflared.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}

Write-Host ""
Write-Host "  Freight Disruption Digital Twin - public link" -ForegroundColor Cyan
Write-Host ""

# ---- 1. backend --------------------------------------------------------------
if (Test-Backend) {
    Write-Host "[1/3] Backend already running." -ForegroundColor Green
} else {
    if (-not (Test-Path $Python)) { throw "Python environment not found at $Python. Is the backend folder on the Desktop?" }
    $busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        Write-Host "[1/3] An old backend is stuck on port 8000 - stopping it." -ForegroundColor Yellow
        $busy | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 2
    }
    Write-Host "[1/3] Starting the backend (window 'Freight Twin - backend') ..."
    Start-Window "Freight Twin - backend" "Set-Location '$Backend'; & '$Python' -m uvicorn app.main:app"
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) { Start-Sleep -Seconds 2; if (Test-Backend) { $ok = $true; break } }
    if (-not $ok) { throw "The backend did not start within 2 minutes. Look at the 'Freight Twin - backend' window for the error." }
    Write-Host "      Backend is running." -ForegroundColor Green
}

# ---- 2. public tunnel --------------------------------------------------------
$link = Get-TunnelLink
if ($link) {
    Write-Host "[2/3] Public tunnel already running." -ForegroundColor Green
} else {
    $cf = Find-Cloudflared
    if (-not $cf) { throw "cloudflared is not installed. Run once: winget install --id Cloudflare.cloudflared" }
    Write-Host "[2/3] Starting the public tunnel (window 'Freight Twin - public link') ..."
    Start-Window "Freight Twin - public link" "& '$cf' tunnel --url http://localhost:8000 --metrics $Metrics"
    for ($i = 0; $i -lt 45; $i++) { Start-Sleep -Seconds 2; $link = Get-TunnelLink; if ($link) { break } }
    if (-not $link) { throw "No public link after 90 seconds. Look at the 'Freight Twin - public link' window for the error." }
}

# ---- 3. show the link --------------------------------------------------------
Set-Clipboard -Value $link
Write-Host "[3/3] Your public link (copied to the clipboard):" -ForegroundColor Green
Write-Host ""
Write-Host "      $link" -ForegroundColor White -BackgroundColor DarkGreen
Write-Host ""
Write-Host "  - Share it: testers click 'Create account' and use their own username and password."
Write-Host "  - The first visit can take up to a minute. If it shows an error, wait and press F5."
Write-Host "  - Keep both 'Freight Twin' windows open and the laptop plugged in and awake."
Write-Host "  - To stop sharing: double-click 'Stop Freight Twin' (or close the two windows)."
Start-Sleep -Seconds 5
Start-Process $link
