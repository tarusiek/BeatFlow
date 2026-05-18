# BeatFlow dev startup — run both backend and frontend
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"

Write-Host "Starting BeatFlow..." -ForegroundColor Cyan

# Backend
$backendJob = Start-Job -ScriptBlock {
    Set-Location "D:\APKI\BeatFlow"
    & ".venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 2>&1
}
Write-Host "Backend starting on http://localhost:8000" -ForegroundColor Green

Start-Sleep 2

# Frontend
$frontendJob = Start-Job -ScriptBlock {
    Set-Location "D:\APKI\BeatFlow\frontend"
    npm run dev 2>&1
}
Write-Host "Frontend starting on http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers." -ForegroundColor Yellow
Write-Host ""

try {
    while ($true) {
        Receive-Job $backendJob | Where-Object { $_ } | ForEach-Object { Write-Host "[backend] $_" -ForegroundColor DarkGray }
        Receive-Job $frontendJob | Where-Object { $_ } | ForEach-Object { Write-Host "[frontend] $_" -ForegroundColor DarkBlue }
        Start-Sleep 1
    }
} finally {
    Stop-Job $backendJob, $frontendJob
    Remove-Job $backendJob, $frontendJob
}
