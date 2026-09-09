"""统计面板共享原语：数值格式化与卡片骨架。

被 wave_stats_panel 与 comparison_stats_panel 共同依赖，
避免两域面板模块互相导入私有函数或复制实现。
"""
from typing import Optional

from dash import html


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


def _render_empty(title: str, hint: str) -> html.Div:
    return html.Div([
        html.Div(title, className='stats-card-title'),
        html.Div(hint, className='stats-empty'),
    ], className='stats-card')
