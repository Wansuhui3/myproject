"""依赖自检：按 requirements.txt 校验运行环境（供 start.ps1 调用，零硬编码清单）。

用法: python check_deps.py [requirements.txt 路径]
退出码: 0 = 依赖齐全；1 = 有缺失/版本不符（stdout 输出明细）
"""
import re
import sys
from importlib import metadata

try:
    from packaging.requirements import Requirement
    _HAS_PACKAGING = True
except ImportError:
    # 无 packaging 时退化为"是否安装"检查（pip 环境通常自带 packaging）
    _HAS_PACKAGING = False


def parse_requirement(line: str) -> tuple[str, object | None]:
    """解析单条依赖为 (包名, 版本约束)；无约束时第二项为 None。"""
    if _HAS_PACKAGING:
        req = Requirement(line)
        return req.name, req.specifier
    name = re.split(r'[><=!~;[ ]', line, maxsplit=1)[0]
    return name, None


def check(requirements_path: str) -> list[str]:
    problems: list[str] = []
    with open(requirements_path, encoding='utf-8') as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            name, spec = parse_requirement(line)
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                problems.append(f'MISSING {name}')
                continue
            if spec is not None and not spec.contains(installed, prereleases=True):
                problems.append(f'OUTDATED {name}=={installed} (需要 {spec})')
    return problems


if __name__ == '__main__':
    requirements_file = sys.argv[1] if len(sys.argv) > 1 else 'requirements.txt'
    issues = check(requirements_file)
    if issues:
        print('\n'.join(issues))
        sys.exit(1)
    sys.exit(0)
