# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('D:\\个人\\雷达波动软件\\radar_wave_analyzer\\config.yaml', '.'), ('D:\\个人\\雷达波动软件\\radar_wave_analyzer\\assets', 'assets')]
binaries = []
hiddenimports = ['dash', 'dash.html', 'dash.dcc', 'dash_bootstrap_components', 'plotly', 'plotly.express', 'flask_caching', 'flask_caching.backends', 'yaml', 'numpy', 'pandas', 'matplotlib', 'matplotlib.backends.backend_agg', 'webview', 'webview.platforms.winforms']
hiddenimports += collect_submodules('radar_wave_analyzer')
tmp_ret = collect_all('dash')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('plotly')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('dash_bootstrap_components')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('webview')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['D:\\个人\\雷达波动软件\\launcher.py'],
    pathex=['D:\\个人\\雷达波动软件'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RadarWaveAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RadarWaveAnalyzer',
)
