# Freight Disruption Digital Twin - stop the public link and the backend (Windows).
# Started by "Stop Freight Twin.bat".
$stopped = 0
Get-Process cloudflared -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force; $stopped++ }
if ($stopped) { Write-Host "Public link stopped." -ForegroundColor Green } else { Write-Host "Public link was not running." }

$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listen) {
    $listen | ForEach-Object {
        $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -like "python*") { Stop-Process -Id $p.Id -Force }
    }
    Write-Host "Backend stopped." -ForegroundColor Green
} else {
    Write-Host "Backend was not running."
}
Write-Host "You can close the 'Freight Twin' windows now."
