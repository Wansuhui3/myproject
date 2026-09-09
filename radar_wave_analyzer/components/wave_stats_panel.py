"""波动分析统计面板（精简版）。
全段统计：各距离量首帧距离 + 各物理量最大跳变
选中区域统计：各物理量框选区最大跳变
波动摘要：可插入快照（与真值对比分距离摘要同款交互）
"""
from typing import Optional

import numpy as np
from dash import html

from .panel_common import _fmt_val, _render_empty, _stat_row


def _calc_max_jump(stats: Optional[dict]) -> Optional[float]:
    """从 stats 字典计算最大跳变绝对值。"""
    if not stats:
        return None
    mp = stats.get('max_positive')
    mn = stats.get('max_negative')
    vals = [v for v in [mp, mn] if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    return max(abs(v) for v in vals)


def _render_quantity_rows(
    selected_quantities: list,
    quantities_config: dict,
    stats_source: dict,
) -> tuple[list, bool]:
    """为每个勾选的物理量渲染统计行，返回 (rows, has_data)。"""
    rows = []
    has_data = False
    for qty in selected_quantities:
        qty_info = quantities_config.get(qty, {})
        label = qty_info.get('label', qty)
        unit = qty_info.get('unit', '')
        max_jump = _calc_max_jump(stats_source.get(qty))
        if max_jump is not None:
            has_data = True
        rows.append(_stat_row(f'{label} 最大跳变', _fmt_val(max_jump), unit))
    return rows, has_data


# ===================== 全段统计 =====================

def render_multi_full_stats(
    selected_quantities: list,
    quantities_config: dict,
    stats_per_qty: dict,
    first_frame_dists: Optional[dict],
) -> html.Div:
    """全段统计：各距离量首帧距离 + 各物理量最大跳变。

    Args:
        selected_quantities: 当前勾选的物理量列表。
        quantities_config: config.yaml quantities 字典。
        stats_per_qty: {qty: compute_segment_stats result}。
        first_frame_dists: 各距离量首帧距离（来自 compute_fluctuation_stats），
                           键为 Dx/Dy/Rx_front/Rx_rear/Ry。
    """
    rows = []

    # 首帧距离 = 目标起批（段首帧）时的距离，始终显示（来自全段首帧原始值）
    dists = first_frame_dists or {}
    for col in ('Dx', 'Dy', 'Rx_front', 'Rx_rear', 'Ry'):
        rows.append(_stat_row(f'{col} 首帧距离', _fmt_val(dists.get(col)), 'm'))

    # 各物理量最大跳变
    qty_rows, has_data = _render_quantity_rows(selected_quantities, quantities_config, stats_per_qty)
    rows.extend(qty_rows)

    has_dist = any(v is not None for v in dists.values())
    if not has_data and not has_dist:
        return _render_empty('全段统计', '暂无有效数据')

    return html.Div([
        html.Div('全段统计', className='stats-card-title'),
        html.Div(rows, className='stats-compact-list'),
    ], className='stats-card')


def render_multi_full_stats_placeholder() -> html.Div:
    """全段统计占位卡片（未选轨迹时）。"""
    return _render_empty('全段统计', '请选择目标ID和轨迹段')


# ===================== 选中区域统计 =====================

def render_multi_box_stats(
    selected_quantities: list,
    quantities_config: dict,
    box_stats_per_qty: dict,
) -> html.Div:
    """选中区域统计：各物理量框选区最大跳变。

    Args:
        selected_quantities: 当前勾选的物理量列表。
        quantities_config: config.yaml quantities 字典。
        box_stats_per_qty: {qty: compute_segment_stats(mask=mask) result}。
    """
    rows, has_data = _render_quantity_rows(selected_quantities, quantities_config, box_stats_per_qty)

    if not has_data:
        return _render_empty('选中区域统计', '暂无有效数据')

    return html.Div([
        html.Div('选中区域统计', className='stats-card-title'),
        html.Div(rows, className='stats-compact-list'),
    ], className='stats-card')


def render_box_stats_empty() -> html.Div:
    """选中区域统计占位卡片。"""
    return _render_empty('选中区域统计', '框选曲线区间后显示')


# ===================== 波动摘要（可插入快照） =====================

def render_wave_snapshot_summary(
    snapshots: list,
    quantities_config: dict,
) -> html.Div:
    """渲染多次插入的波动摘要快照（与真值对比分距离摘要同款交互）。

    Args:
        snapshots: [{'label': 轨迹段标识,
                     'values': {qty: {'value': float|None, 'unit': str}}}]
        quantities_config: config.yaml quantities 字典（取物理量显示名）。

    每个物理量一行：``Dx最大波动：0.123 / 0.456 m``（多快照以 " / " 连接，
    无值显示 —）。
    """
    if not snapshots:
        return html.Div('点击“插入当前”保留波动摘要', className='stats-empty')

    metadata = [
        html.Div(f'[{index}] {s.get("label") or "未命名"}',
                 className='perf-summary-snapshot-meta')
        for index, s in enumerate(snapshots, start=1)
    ]

    qty_order: list = []
    for s in snapshots:
        for qty in (s.get('values') or {}):
            if qty not in qty_order:
                qty_order.append(qty)

    rows = []
    for qty in qty_order:
        qty_info = quantities_config.get(qty, {})
        label = qty_info.get('label', qty)
        vals = []
        for s in snapshots:
            entry = (s.get('values') or {}).get(qty)
            if entry is None or entry.get('value') is None:
                vals.append('—')
                continue
            unit = str(entry.get('unit') or '')
            vals.append(f'{float(entry["value"]):.3f} {unit}'.rstrip())
        rows.append(html.Div(
            f'{label}最大波动：{" / ".join(vals)}', className='perf-summary-row'))

    return html.Div([
        html.Div(metadata, className='perf-summary-snapshot-list'),
        *rows,
    ], className='perf-distance-summary')
