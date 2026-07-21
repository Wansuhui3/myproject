$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RadarDir = Join-Path $ProjectDir 'radar_wave_analyzer'
$PythonExe = Join-Path $env:USERPROFILE '.workbuddy\binaries\python\versions\3.14.3\python.exe'
$Url = 'http://127.0.0.1:8050'

Write-Host "Project: $RadarDir" -ForegroundColor Cyan

# Check Python
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Python not found: $PythonExe" -ForegroundColor Red
    Write-Host "Please install workbuddy Python 3.14 first."
    return
}

# Check & install dependencies
Write-Host "Checking dependencies..." -ForegroundColor Cyan
$null = & $PythonExe -c "import dash, plotly, pandas, numpy, flask_caching, scipy, yaml, pywebview" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
    & $PythonExe -m pip install -r (Join-Path $RadarDir 'requirements.txt')
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Dependency install failed" -ForegroundColor Red
        return
    }
}
Write-Host "Dependencies OK" -ForegroundColor Green

# Go to project dir
Set-Location $RadarDir

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Server Started" -ForegroundColor Green
Write-Host "  URL: $Url" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""

# Open browser
Start-Process $Url

# Start server
& $PythonExe 'app.py'
