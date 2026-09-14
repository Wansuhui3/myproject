"""真值对比统计面板。
误差统计仅覆盖当前曲线中可比较（单位兼容）的物理量映射。
"""
from dash import html

from .panel_common import _fmt_val, _render_empty, _stat_row


def render_cmp_error_stats(
    mapping_results: list[dict],
    match_summary: dict | None = None,
) -> html.Div:
    """仅渲染当前曲线中可比较物理量的 RMSE 与匹配率。

    曲线通道由用户映射决定，因此不能继续展示固定的 Dx/Dy/速度合量或延迟。
    单位不兼容的叠加曲线只用于观察，不生成误差数值。
    """
    rows = []
    for mapping in mapping_results:
        if not mapping.get('stats_enabled'):
            continue
        radar_col = str(mapping.get('radar_col') or mapping.get('label') or '物理量')
        metrics = mapping.get('metrics') or {}
        unit = str(mapping.get('radar_unit') or mapping.get('unit') or '')
        rows.append(_stat_row(
            f'{radar_col} RMSE',
            _fmt_val(metrics.get('rmse')),
            unit,
        ))

    if match_summary:
        rows.append(_stat_row(
            '匹配率',
            f'{match_summary.get("matched_frames", 0)}/{match_summary.get("total_frames", 0)} '
            f'({float(match_summary.get("match_rate", 0.0)):.1%})',
            '',
        ))

    if not rows:
        return _render_empty('误差统计', '当前曲线没有可计算误差的同单位物理量')

    return html.Div([
        html.Div('误差统计', className='stats-card-title'),
        html.Div(rows, className='stats-compact-list'),
    ], className='stats-card')


def render_cmp_error_stats_empty() -> html.Div:
    """误差统计占位卡片。"""
    return _render_empty('误差统计', '执行对齐后显示')


def render_cmp_bins_empty() -> html.Div:
    """距离区间统计占位卡片。"""
    return html.Div('执行对齐后显示', className='stats-empty')
