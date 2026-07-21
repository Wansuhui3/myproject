# ============================================================
# Radar Wave Analyzer - PyInstaller build script
# Output: dist/RadarWaveAnalyzer/RadarWaveAnalyzer.exe
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File build_package.ps1
# ============================================================

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RadarDir = Join-Path $ProjectRoot 'radar_wave_analyzer'
$DistDir = Join-Path $ProjectRoot 'dist'
$BuildDir = Join-Path $ProjectRoot 'build'
$SpecFile = Join-Path $ProjectRoot 'RadarWaveAnalyzer.spec'

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Radar Wave Analyzer - PyInstaller Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Find Python
$PythonCandidates = @(
    (Join-Path $env:USERPROFILE '.workbuddy\binaries\python\versions\3.14.3\python.exe'),
    'python',
    'python3'
)
$PythonExe = $null
foreach ($c in $PythonCandidates) {
    $result = Get-Command $c -ErrorAction SilentlyContinue
    if ($result) {
        $PythonExe = $result.Source
        break
    }
}
if (-not $PythonExe) {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    exit 1
}
Write-Host "Python: $PythonExe" -ForegroundColor Green

# 2. Install build dependencies
Write-Host ""
Write-Host "[1/4] Installing build deps (pyinstaller + pywebview)..." -ForegroundColor Cyan
& $PythonExe -m pip install pyinstaller pywebview --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install build dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "  Done" -ForegroundColor Green

# 3. Ensure app dependencies (matplotlib for image export, no kaleido needed)
Write-Host ""
Write-Host "[2/4] Installing app dependencies..." -ForegroundColor Cyan
& $PythonExe -m pip install -r (Join-Path $RadarDir 'requirements.txt') --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install app dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "  Done" -ForegroundColor Green

# 4. Clean old builds (skip if files are locked)
Write-Host ""
Write-Host "[3/4] Cleaning old builds..." -ForegroundColor Cyan
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue }
if (Test-Path $SpecFile) { Remove-Item -Force $SpecFile -ErrorAction SilentlyContinue }
if (Test-Path $DistDir) {
    $distFolder = Join-Path $DistDir 'RadarWaveAnalyzer'
    if (Test-Path $distFolder) { Remove-Item -Recurse -Force $distFolder -ErrorAction SilentlyContinue }
}
Write-Host "  Done" -ForegroundColor Green

# 5. PyInstaller build
Write-Host ""
Write-Host "[4/4] PyInstaller building (may take a few minutes)..." -ForegroundColor Cyan
Write-Host "  Target: $ProjectRoot\launcher.py" -ForegroundColor DarkGray

& $PythonExe -m PyInstaller `
    --onedir `
    --windowed `
    --noconfirm `
    --name "RadarWaveAnalyzer" `
    --distpath $DistDir `
    --workpath $BuildDir `
    --specpath $ProjectRoot `
    --paths "$ProjectRoot" `
    --add-data "$RadarDir\config.yaml;." `
    --add-data "$RadarDir\assets;assets" `
    --hidden-import dash `
    --hidden-import dash.html `
    --hidden-import dash.dcc `
    --hidden-import dash_bootstrap_components `
    --hidden-import plotly `
    --hidden-import plotly.express `
    --hidden-import flask_caching `
    --hidden-import flask_caching.backends `
    --hidden-import yaml `
    --hidden-import numpy `
    --hidden-import pandas `
    --hidden-import matplotlib `
    --hidden-import matplotlib.backends.backend_agg `
    --hidden-import webview `
    --hidden-import webview.platforms.winforms `
    --collect-submodules radar_wave_analyzer `
    --collect-all dash `
    --collect-all plotly `
    --collect-all dash_bootstrap_components `
    --collect-all webview `
    (Join-Path $ProjectRoot 'launcher.py')

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] PyInstaller build failed, exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Check the output above for details." -ForegroundColor Red
    exit 1
}

# 6. Verify output
$OutputExe = Join-Path (Join-Path $DistDir 'RadarWaveAnalyzer') 'RadarWaveAnalyzer.exe'
if (Test-Path $OutputExe) {
    $Size = [math]::Round((Get-Item $OutputExe).Length / 1MB, 1)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  BUILD SUCCESS" -ForegroundColor Green
    Write-Host "  Output: $OutputExe" -ForegroundColor Green
    Write-Host "  Size: $Size MB (including runtime)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To distribute, copy the entire 'RadarWaveAnalyzer' folder." -ForegroundColor Yellow
} else {
    Write-Host "[ERROR] Build output not found" -ForegroundColor Red
    exit 1
}
