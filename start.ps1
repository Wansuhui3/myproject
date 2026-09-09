$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RadarDir = Join-Path $ProjectDir 'radar_wave_analyzer'
$Url = 'http://127.0.0.1:8050'

Write-Host "Project: $RadarDir" -ForegroundColor Cyan

# Check Python：按候选顺序查找本机可用的 Python
$PythonExe = $null
$PythonCandidates = @(
    (Join-Path $env:USERPROFILE '.workbuddy\binaries\python\versions\3.14.3\python.exe'),
    (Join-Path $env:ProgramFiles 'Python313\python.exe'),
    'python',
    'python3'
)
foreach ($candidate in $PythonCandidates) {
    $resolved = if (Test-Path $candidate) { $candidate }
                elseif ($candidate -notmatch '[\\/]') {
                    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
                    if ($cmd) { $cmd.Source }
                } else { $null }
    if ($resolved -and (Test-Path $resolved)) {
        $PythonExe = $resolved
        break
    }
}
if (-not $PythonExe) {
    Write-Host "[ERROR] Python not found. Please install Python 3.13+ first." -ForegroundColor Red
    return
}
Write-Host "Python: $PythonExe" -ForegroundColor Cyan

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

# Go to project root (app.py 使用包内相对导入，必须以模块方式启动)
Set-Location $ProjectDir

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
