"""导出格式化辅助：值类型转换、时间戳格式化、限值 Excel 公式与评判标准描述。

本模块不依赖任何业务模块（零内部依赖），供 exporter 工作簿组装层引用。
公式语义与 performance_common._resolve_limit 保持一致。
"""
import re

import numpy as np
import pandas as pd


def _py(value):
    """numpy 标量转 Python 原生类型，避免 openpyxl 拒写 np.int64 等。"""
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    try:
        if isinstance(value, (int,)):
            return int(value)
        fv = float(value)
    except (TypeError, ValueError):
        return str(value)
    return fv if np.isfinite(fv) else None


def _fmt_epoch_ts(value) -> str:
    """epoch 秒 → 无时区偏移字符串（与图表悬浮口径一致，1970 基准）。"""
    try:
        sec = float(value)
    except (TypeError, ValueError):
        return ''
    if not np.isfinite(sec):
        return ''
    return pd.to_datetime(sec, unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def _limit_excel(limit_rule: dict, basis_col: str, row: int):
    """限值规则 → Excel 活公式/常量（语义与 performance._resolve_limit 一致）。

    percent 类公式直接引用基准列（真值距离 F 或真值数据 D），双击单元格
    即可看到阈值的计算过程；无法解析的规则返回 None（回退静态值）。
    """
    if not isinstance(limit_rule, dict):
        return None
    mode = limit_rule.get('mode')
    absolute = limit_rule.get('absolute')
    percent = limit_rule.get('percent')
    scale = limit_rule.get('scale')
    basis = f'ABS({basis_col}{row})'
    try:
        if mode == 'absolute':
            return float(absolute)
        if mode == 'percent':
            return f'={float(percent)}*{basis}'
        if mode == 'max_absolute_percent':
            return f'=MAX({float(absolute)},{float(percent)}*{basis})'
        if mode == 'scaled_max_absolute_percent':
            return (f'=ROUND({float(scale)}*MAX({float(absolute)},'
                    f'{float(percent)}*{basis}),6)')
    except (TypeError, ValueError):
        return None
    return None


def _describe_limit(limit_rule: dict) -> str:
    """限值规则的人类可读描述（用于评判标准块）。"""
    if not isinstance(limit_rule, dict):
        return '未配置'
    mode = limit_rule.get('mode')
    absolute = limit_rule.get('absolute')
    percent = limit_rule.get('percent')
    scale = limit_rule.get('scale')
    try:
        if mode == 'absolute':
            return f'固定 {float(absolute)}'
        if mode == 'percent':
            return f'{float(percent) * 100:g}% × 基准'
        if mode == 'max_absolute_percent':
            return f'max({float(absolute)}, {float(percent) * 100:g}% × 基准)'
        if mode == 'scaled_max_absolute_percent':
            return (f'{float(scale):g} × max({float(absolute)}, '
                    f'{float(percent) * 100:g}% × 基准)')
    except (TypeError, ValueError):
        pass
    return '未配置'


def _criteria_lines(label: str, unit: str, rule: dict, perf_cfg: dict) -> list:
    """页首评判标准块：判定公式语义 + 逐距离段阈值配置。"""
    accuracy_req = float(perf_cfg.get('accuracy_requirement', 0.9545))
    three_req = float(perf_cfg.get('three_frame_requirement', 1.0))
    min_samples = int(perf_cfg.get('min_samples_for_conclusion', 3))
    valid_range = perf_cfg.get('valid_distance_range') or [0, 150]
    by_distance = str(rule.get('percent_basis', 'truth')) == 'distance'
    basis_text = ('RTK 纵向距离（F 列，d=|rtk_center_x|）' if by_distance
                  else '本物理量真值绝对值（D 列）')
    lines = [
        f'{label}（单位 {unit}）逐帧验收明细与判定',
        f'百分比基准：{basis_text}',
        '单帧合格：|雷达数据−真值数据| < 单帧阈值（G 列绝对误差、I 列判定均为 Excel 公式）',
        f'准确率：段内合格帧数 ÷ 段样本数（O 列公式），须 > {accuracy_req * 100:.2f}%',
        '三帧归一化（K 列）= |绝对误差| ÷ 三帧阈值，即单帧误差的超限倍数',
        ('三帧违规（L 列）= 连续三帧归一化最小值 MIN(K前, K本, K后) ≥ 1，'
         '即连续三帧全部超限才记违规，结果以中间帧归属输出一次；'
         '数据首帧、中断后首帧、连续段末帧无完整三帧窗口，不输出'),
        (f'结论：准确率 > {accuracy_req * 100:.2f}% 且 连续三帧指标 < {three_req:g}'
         f' → 通过；样本 < {min_samples} → 样本不足'),
        (f'有效帧：雷达-真值匹配成功、数值有限且真值距离在 '
         f'{valid_range[0]}–{valid_range[1]}m 内；本表仅含有效帧'),
    ]
    for i, entry in enumerate(rule.get('bins') or []):
        span = entry.get('range') or []
        lo = span[0] if len(span) > 0 else '?'
        hi = span[1] if len(span) > 1 else '?'
        normal = _describe_limit(entry.get('normal_limit'))
        three = _describe_limit(entry.get('three_frame_limit'))
        lines.append(f'  距离段 [{lo}, {hi}]: 单帧阈值 = {normal}；三帧阈值 = {three}')
    if not (rule.get('bins') or []):
        lines.append(f'  单帧阈值 = {_describe_limit(rule.get("normal_limit"))}（完整曲线单限值）')
    return lines


def _sheet_safe_name(name: str, used: set) -> str:
    """sheet 名安全化：替换非法字符、限长 31、去重。"""
    clean = re.sub(r'[\[\]:*?/\\]', '_', str(name)).strip()[:31] or '物理量'
    base, k = clean, 2
    while clean in used:
        suffix = f'_{k}'
        clean = base[:31 - len(suffix)] + suffix
        k += 1
    used.add(clean)
    return clean
