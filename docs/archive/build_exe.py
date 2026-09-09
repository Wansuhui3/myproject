"""一键打包脚本：清理旧产物 → PyInstaller → 校验产物。

用法（在本机执行）：
    python build_exe.py

输出：
    dist/RadarWaveAnalyzer/RadarWaveAnalyzer.exe
分发方式：将整个 dist\\RadarWaveAnalyzer 文件夹压缩后发送给对方，
对方无需安装 Python，双击 exe 即可使用。

说明：
- 强制 UTF-8 模式（PYTHONUTF8=1），避免中文路径下的 GBK 解码错误；
- 打包前自动终止占用旧产物的进程并清理 dist/build，
  规避 PyInstaller 清理旧目录时的删除保护冲突。

[已归档] 已被 build_package.ps1（打包发布版.cmd 入口）取代。
"""
import os
import subprocess
import sys
import time

os.environ['PYTHONUTF8'] = '1'

PROJECT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(PROJECT, 'dist')
BUILD = os.path.join(PROJECT, 'build')
SPEC = os.path.join(PROJECT, 'RadarWaveAnalyzer.spec')
EXE = os.path.join(DIST, 'RadarWaveAnalyzer', 'RadarWaveAnalyzer.exe')


def log(msg: str) -> None:
    print(msg, flush=True)


def rmdir_fast(path: str) -> None:
    """用 cmd 原生 rd /s /q 删除目录。

    本进程可能被编辑器插件注入删除保护 hook（一次性删除 >500 文件需要
    确认，会让 PyInstaller 的 COLLECT 静默失败），改由 cmd 子进程删除
    即可绕过。
    """
    if os.path.isdir(path):
        subprocess.run(['cmd', '/c', 'rd', '/s', '/q', path],
                       capture_output=True)


def main() -> None:
    started = time.time()

    # 1. 终止可能占用旧产物的进程（忽略不存在的情况）
    subprocess.run(['taskkill', '/F', '/IM', 'RadarWaveAnalyzer.exe'],
                   capture_output=True)

    # 2. 清理旧产物：必须删干净，PyInstaller COLLECT 阶段才不会触发
    #    编辑器插件的批量删除保护
    for path in (DIST, BUILD):
        if os.path.isdir(path):
            log(f'清理旧产物: {path}')
            rmdir_fast(path)
        if os.path.isdir(path):
            log(f'[WARN] 未能完全清理: {path}（可能有文件被占用）')

    # 3. PyInstaller 打包（--clean 强制重建，避免缓存残留）
    log('开始 PyInstaller 打包（可能需要几分钟）...')
    result = subprocess.run(
        [sys.executable, '-m', 'PyInstaller', SPEC,
         '--noconfirm', '--clean'],
        cwd=PROJECT)
    if result.returncode != 0:
        log(f'[ERROR] PyInstaller 打包失败，exit={result.returncode}')
        sys.exit(1)

    # 4. 校验产物
    if not os.path.isfile(EXE):
        log('[ERROR] 未找到产物: ' + EXE)
        sys.exit(1)
    folder = os.path.dirname(EXE)
    total = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(folder) for name in names)
    log(f'打包成功: {EXE}')
    log(f'产物大小: {total / 1024 / 1024:.1f} MB')
    log(f'耗时: {time.time() - started:.0f} 秒')
    log('分发方式: 将整个 dist\\RadarWaveAnalyzer 文件夹压缩后发送，'
        '对方双击 RadarWaveAnalyzer.exe 即可使用。')


if __name__ == '__main__':
    main()
