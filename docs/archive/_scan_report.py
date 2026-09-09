"""代码库扫描：死导入、僵尸函数、依赖来源、体积来源。"""
import ast
import os
import subprocess
import sys

ROOT = 'radar_wave_analyzer'
VENV_PY = os.path.join('.build-venv', 'Scripts', 'python.exe')

print('=== 1. venv 依赖来源 ===')
for pkg in ('shapely', 'openpyxl', 'psutil'):
    r = subprocess.run(
        [VENV_PY, '-m', 'pip', 'show', pkg],
        capture_output=True, text=True, errors='ignore')
    if r.returncode != 0:
        print(f'  {pkg}: 未安装')
        continue
    required_by = [line for line in r.stdout.splitlines()
                   if line.startswith('Required-by')]
    size = None
    pkg_dir = os.path.join('.build-venv', 'Lib', 'site-packages', pkg)
    if os.path.isdir(pkg_dir):
        size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(pkg_dir) for f in files)
    print(f'  {pkg}: Required-by {required_by[0] if required_by else "?"}'
          f' | venv内体积 {((size or 0) / 1048576):.1f} MB')

print()
print('=== 2. exporter.py 顶层导入使用检查 ===')
src = open(os.path.join(ROOT, 'comparison', 'exporter.py'),
           encoding='utf-8').read()
tree = ast.parse(src)
imported = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported += [a.asname or a.name.split('.')[0] for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        imported += [a.asname or a.name for a in node.names]
body_src = src
for name in set(imported):
    uses = body_src.count(name)
    # 出现次数 1 = 仅 import 行本身
    if uses <= 1:
        print(f'  未使用导入: {name}')
print(f'  （共检查 {len(set(imported))} 个顶层导入）')

print()
print('=== 3. callbacks.py 顶层导入使用检查 ===')
src = open(os.path.join(ROOT, 'callbacks.py'), encoding='utf-8').read()
tree = ast.parse(src)
imported = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported += [a.asname or a.name.split('.')[0] for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        imported += [a.asname or a.name for a in node.names]
for name in sorted(set(imported)):
    if src.count(name) <= 1:
        print(f'  未使用导入: {name}')
print(f'  （共检查 {len(set(imported))} 个顶层导入）')

print()
print('=== 4. 疑似僵尸函数（定义后全项目零引用） ===')
candidates = [
    ('comparison/exporter.py', 'export_summary_json'),
    ('comparison/exporter.py', 'export_comparison_workbook'),
    ('comparison/exporter.py', '_perf_frame_columns'),
    ('core/exporter.py', 'export_trajectory_csv'),
    ('components/performance_panel.py', 'render_performance_distance_summary'),
    ('components/stats_panel.py', 'render_cmp_error_stats'),
]
search_roots = [ROOT, 'callbacks.py']
for rel, func in candidates:
    path = os.path.join(ROOT, rel)
    count = 0
    for root, _, files in os.walk(ROOT):
        if '__pycache__' in root:
            continue
        for f in files:
            if not f.endswith('.py'):
                continue
            fp = os.path.join(root, f)
            try:
                content = open(fp, encoding='utf-8').read()
            except OSError:
                continue
            count += content.count(func)
    print(f'  {rel}:{func} → 全项目出现 {count} 次'
          f'（1 = 仅定义，属僵尸/仅测试引用）')

print()
print('=== 5. 根目录调试/临时文件 ===')
debug_files = [
    '_debug_cmp.py', '_debug_perf.py', '_verify5.py', '_verify_exe.py',
    '_run_build.py', '_diag_shim.py', '_scan_report.py',
    'segment_validation.py', 'segment_validation_v2.py',
    'check_deps.py' if os.path.exists(
        os.path.join(ROOT, 'check_deps.py')) else None,
    'svc_check.txt', 'svc_err.txt', 'build_log.txt', 'build_err.txt',
]
for f in debug_files:
    if f and os.path.exists(f):
        size = os.path.getsize(f)
        print(f'  {f} ({size / 1024:.1f} KB)')
dbg_dir = os.path.join(ROOT, '.dbg')
if os.path.isdir(dbg_dir):
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(dbg_dir) for f in fs)
    print(f'  {dbg_dir}/ ({total / 1024:.1f} KB)')
