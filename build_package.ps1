# 雷达目标轨迹波动分析系统：可交付发布包构建脚本
#
# 直接运行：双击“打包发布版.cmd”；或执行
# powershell -ExecutionPolicy Bypass -File .\build_package.ps1
#
# 输出：release\RadarWaveAnalyzer_yyyyMMdd_HHmmss\package\RadarWaveAnalyzer\
#       release\RadarWaveAnalyzer_yyyyMMdd_HHmmss.zip
#
# 不会清理 dist、build 或历史 release，避免误删既有软件包。

[CmdletBinding()]
param([switch]$SkipDependencyInstall)

$ErrorActionPreference = 'Stop'

# ── 隔离编辑器注入的 sitecustomize 钩子 ──
# 系统 PYTHONPATH 指向 VSCode 扩展的 shim 目录，其中 hook 了 subprocess 与
# 文件删除，会导致 PyInstaller 依赖分析子进程超时（Timed out while waiting
# for the child process to exit）与目录清理失败。打包全程清除该注入。
$env:PYTHONPATH = $null
$env:PYTHONNOUSERSITE = '1'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$AppDir = Join-Path $ProjectRoot 'radar_wave_analyzer'
$EntryScript = Join-Path $ProjectRoot 'launcher.py'
$Requirements = Join-Path $AppDir 'requirements.txt'
$VenvDir = Join-Path $ProjectRoot '.build-venv'
$ReleaseRoot = Join-Path $ProjectRoot 'release'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$ReleaseDir = Join-Path $ReleaseRoot "RadarWaveAnalyzer_$Stamp"
$DistDir = Join-Path $ReleaseDir 'package'
$WorkDir = Join-Path $ReleaseDir 'work'
$SpecDir = Join-Path $ReleaseDir 'spec'
$ZipFile = Join-Path $ReleaseRoot "RadarWaveAnalyzer_$Stamp.zip"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Find-Python {
    # 优先 Python Launcher，避免误用 Microsoft Store 的 python 占位程序。
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        try {
            & $py.Source -3 -c "import sys; print(sys.executable)" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { return @($py.Source, '-3') }
        } catch { }
    }
    foreach ($name in @('python.exe', 'python3.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        try {
            & $command.Source -c "import sys; print(sys.executable)" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { return @($command.Source) }
        } catch { }
    }
    throw '未找到可用 Python。请安装 Python 3.10–3.13，并勾选 Add Python to PATH。'
}

function Invoke-BasePython([string[]]$Arguments) {
    & $script:PythonExe @script:PythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python 命令失败：$($Arguments -join ' ')" }
}

if (-not (Test-Path -LiteralPath $EntryScript)) { throw "找不到启动文件：$EntryScript" }
if (-not (Test-Path -LiteralPath $Requirements)) { throw "找不到依赖文件：$Requirements" }

Write-Host '雷达目标轨迹波动分析系统 - 发布包构建器' -ForegroundColor Green
Write-Host "项目目录：$ProjectRoot"
$PythonCommand = Find-Python
$PythonExe = $PythonCommand[0]
$PythonPrefix = @($PythonCommand | Select-Object -Skip 1)
Write-Host "构建 Python：$PythonExe $($PythonPrefix -join ' ')" -ForegroundColor DarkGray

Write-Step '准备独立构建环境'
if (-not (Test-Path -LiteralPath (Join-Path $VenvDir 'Scripts\python.exe'))) {
    Invoke-BasePython @('-m', 'venv', $VenvDir)
}
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) { throw '虚拟环境创建失败。' }

if (-not $SkipDependencyInstall) {
    Write-Step '安装构建与运行依赖（首次可能需要几分钟）'
    & $VenvPython -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) { throw 'pip 更新失败。' }
    & $VenvPython -m pip install -r $Requirements pyinstaller --quiet
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败，请检查网络或 requirements.txt。' }
    & $VenvPython -c "import openpyxl; print('openpyxl', openpyxl.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'openpyxl 安装校验失败，无法生成支持 Excel 导出的发布包。' }
}

Write-Step '执行 PyInstaller 打包'
New-Item -ItemType Directory -Force -Path $ReleaseRoot, $DistDir, $WorkDir, $SpecDir | Out-Null

# 相对路径 + 显式资源，不依赖含机器绝对路径的 RadarWaveAnalyzer.spec。
$PyInstallerArgs = @(
    '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
    '--name', 'RadarWaveAnalyzer', '--distpath', $DistDir,
    '--workpath', $WorkDir, '--specpath', $SpecDir, '--paths', $ProjectRoot,
    '--add-data', "$AppDir\config.yaml;.", '--add-data', "$AppDir\assets;assets",
    '--collect-submodules', 'radar_wave_analyzer', '--collect-all', 'dash',
    '--collect-all', 'plotly', '--collect-all', 'dash_bootstrap_components',
    '--collect-all', 'webview', '--collect-all', 'openpyxl', '--hidden-import', 'flask_caching.backends',
    $EntryScript
)
& $VenvPython @PyInstallerArgs
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller 打包失败，请查看上方错误信息。' }

$AppPackage = Join-Path $DistDir 'RadarWaveAnalyzer'
$Exe = Join-Path $AppPackage 'RadarWaveAnalyzer.exe'
if (-not (Test-Path -LiteralPath $Exe)) { throw "未找到打包产物：$Exe" }

Write-Step '生成可交付 ZIP'
Compress-Archive -Path $AppPackage -DestinationPath $ZipFile -CompressionLevel Optimal -Force
$PackageSize = [math]::Round(((Get-ChildItem -LiteralPath $AppPackage -Recurse -File |
    Measure-Object -Property Length -Sum).Sum / 1MB), 1)
$ZipSize = [math]::Round(((Get-Item -LiteralPath $ZipFile).Length / 1MB), 1)

Write-Host "`n打包成功。" -ForegroundColor Green
Write-Host "程序目录：$AppPackage" -ForegroundColor Green
Write-Host "交付 ZIP：$ZipFile（$ZipSize MB）" -ForegroundColor Green
Write-Host "程序展开后大小：$PackageSize MB" -ForegroundColor Green
Write-Host '交付时发送 ZIP；使用者解压后双击 RadarWaveAnalyzer.exe 即可。' -ForegroundColor Yellow
