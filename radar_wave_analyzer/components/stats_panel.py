"""
统计面板渲染组件（精简版）。
全段统计：Dx最远检出距离 + 各物理量最大跳变
选中区域统计：各物理量框选区最大跳变
"""
from typing import Optional

import numpy as np
from dash import html


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


def _fmt_val(val: Optional[float], decimals: int = 2) -> str:
    """格式化数值。"""
    if val is None:
        return '—'
    try:
        return f'{float(val):.{decimals}f}'
    except (ValueError, TypeError):
        return '—'


def _stat_row(label: str, value: str, unit: str = '') -> html.Div:
    """渲染单行统计。"""
    full = f'{value} {unit}'.strip()
    return html.Div([
        html.Span(label, className='stat-row-label'),
        html.Span(full, className='stat-row-value'),
    ], className='stat-row')


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


def _render_empty(title: str, hint: str) -> html.Div:
    return html.Div([
        html.Div(title, className='stats-card-title'),
        html.Div(hint, className='stats-empty'),
    ], className='stats-card')


# ===================== 全段统计 =====================

def render_multi_full_stats(
    selected_quantities: list,
    quantities_config: dict,
    stats_per_qty: dict,
    dx_max_dist: Optional[float],
) -> html.Div:
    """全段统计：Dx 最远检出距离 + 各物理量最大跳变。

    Args:
        selected_quantities: 当前勾选的物理量列表。
        quantities_config: config.yaml quantities 字典。
        stats_per_qty: {qty: compute_segment_stats result}。
        dx_max_dist: Dx 最远检出距离（来自 compute_fluctuation_stats）。
    """
    rows = []

    # Dx 最远检出距离（始终显示，来自全段 Dx 原始值）
    rows.append(_stat_row('Dx 最远检出距离', _fmt_val(dx_max_dist), 'm'))

    # 各物理量最大跳变
    qty_rows, has_data = _render_quantity_rows(selected_quantities, quantities_config, stats_per_qty)
    rows.extend(qty_rows)

    if not has_data and dx_max_dist is None:
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


# ===================== 真值对比统计 =====================

def render_cmp_error_stats(summary: dict, match_summary: dict) -> html.Div:
    """渲染误差统计卡片（复用 _stat_row）。

    Args:
        summary: align_trajectories 返回的 summary。
        match_summary: 匹配概况。
    """
    rows = []

    # 位置误差
    pos_x = summary.get('pos_error_x', {})
    pos_y = summary.get('pos_error_y', {})
    pos_abs = summary.get('pos_error_abs', {})
    rows.append(_stat_row('ΔDx RMSE', _fmt_val(pos_x.get('rmse')), 'm'))
    rows.append(_stat_row('ΔDy RMSE', _fmt_val(pos_y.get('rmse')), 'm'))
    rows.append(_stat_row('ΔDist RMSE', _fmt_val(pos_abs.get('rmse')), 'm'))
    rows.append(html.Hr(className='stats-separator'))
    rows.append(_stat_row('ΔDx Mean', _fmt_val(pos_x.get('mean')), 'm'))
    rows.append(_stat_row('ΔDy Mean', _fmt_val(pos_y.get('mean')), 'm'))

    # 速度误差
    rows.append(html.Hr(className='stats-separator'))
    vel_x = summary.get('vel_error_x', {})
    vel_y = summary.get('vel_error_y', {})
    vel_abs = summary.get('vel_error_abs', {})
    rows.append(_stat_row('ΔVx RMSE', _fmt_val(vel_x.get('rmse')), 'm/s'))
    rows.append(_stat_row('ΔVy RMSE', _fmt_val(vel_y.get('rmse')), 'm/s'))
    rows.append(_stat_row('ΔV RMSE', _fmt_val(vel_abs.get('rmse')), 'm/s'))

    # 匹配概况
    rows.append(html.Hr(className='stats-separator'))
    rows.append(_stat_row('匹配率',
                          f'{match_summary.get("matched_frames", 0)}/{match_summary.get("total_frames", 0)}'))
    rows.append(_stat_row('延迟', _fmt_val(match_summary.get('delay_ms', 0), 0), 'ms'))

    return html.Div([
        html.Div('误差统计', className='stats-card-title'),
        html.Div(rows, className='stats-compact-list'),
    ], className='stats-card')


def render_cmp_distance_bins(bin_stats: list) -> html.Div:
    """渲染分距离区间统计表（复用 traj-table 样式）。

    Args:
        bin_stats: compute_distance_bin_stats 的返回结果。
    """
    if not bin_stats:
        return _render_empty('分距离区间统计', '执行对齐后显示')

    rows = []
    for b in bin_stats:
        # 空桶（无该距离区间样本）保留整行，数值列填 null 而非跳过
        rmse = b.get('rmse')
        mean = b.get('mean')
        maxv = b.get('max')
        rows.append(html.Tr([
            html.Td(b['bin']),
            html.Td(str(b['frames'])),
            html.Td('null' if rmse is None else _fmt_val(rmse, 3)),
            html.Td('null' if mean is None else _fmt_val(mean, 3)),
            html.Td('null' if maxv is None else _fmt_val(maxv, 3)),
        ], className='bin-row-empty' if b['frames'] == 0 else None))

    return html.Div([
        html.Div('分距离区间统计', className='stats-card-title'),
        html.Table([
            html.Thead(html.Tr([
                html.Th('区间'), html.Th('帧数'),
                html.Th('RMSE(m)'), html.Th('Mean(m)'), html.Th('Max(m)'),
            ])),
            html.Tbody(rows),
        ], className='traj-table'),
    ], className='stats-card')


def render_cmp_error_stats_empty() -> html.Div:
    """误差统计占位卡片。"""
    return _render_empty('误差统计', '执行对齐后显示')


def render_cmp_bins_empty() -> html.Div:
    """距离区间统计占位卡片。"""
    return _render_empty('分距离区间统计', '执行对齐后显示')
