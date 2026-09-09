"""真值对比误差/阈值散点子图构建器。

overlay（双线叠加）主图与误差不合格帧高亮见 comparison_overlay；
三种 chart_type 的编排逻辑见 comparison_charts。
"""
import numpy as np
import plotly.graph_objects as go

from .chart_common import COLORS

_SCATTER_COLOR = '#dc2626'     # 误差散点红色


def _build_error_subplot(
    fig: go.Figure,
    row: int,
    timestamps,
    errors,
    y_label: str,
    y_unit: str,
    color_idx: int,
    zero_line: bool = True,
    time_labels=None,
):
    """构建一个误差散点子图。

    Args:
        zero_line: 是否添加零线。
    """
    color = COLORS[color_idx % len(COLORS)]

    # 零线
    if zero_line:
        fig.add_hline(y=0, line_dash='dash', line_color='#94a3b8',
                      line_width=1, row=row, col=1)

    # 误差散点
    labels = np.asarray(time_labels) if time_labels is not None else np.asarray(['N/A'] * len(errors))
    error_text = [
        f'{label}<br>{y_label}: {value:.4f}{y_unit}'
        for label, value in zip(labels, errors)
    ]
    fig.add_trace(go.Scatter(
        x=timestamps, y=errors,
        mode='markers',
        name=f'{y_label} 误差',
        marker=dict(color=_SCATTER_COLOR, size=3, opacity=0.5),
        text=error_text,
        hoverinfo='text',
        showlegend=False,
    ), row=row, col=1)

    y_title = f'{y_label}({y_unit})' if y_unit else y_label
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=10, color=color),
        title_standoff=0,
        tickfont=dict(size=9, color=color),
        row=row, col=1,
    )

    yref = 'y domain' if row == 1 else f'y{row} domain'
    fig.add_annotation(
        text=f'<b>{y_label} ({y_unit})</b>',
        xref='x domain', yref=yref,
        x=0.99, y=0.93,
        xanchor='right', yanchor='middle',
        showarrow=False,
        bgcolor='rgba(255,255,255,0.78)',
        font=dict(size=11, color=color),
    )


def _build_scatter_subplot(
    fig: go.Figure,
    row: int,
    timestamps,
    values,
    qty_label: str,
    qty_unit: str,
    color_idx: int,
    threshold=None,
    time_labels=None,
):
    """构建一个阈值散点子图（散点 + 可配置阈值线）。

    Args:
        values: 散点 y 值（字段缺失时传 []，与旧内联分支行为一致）。
        threshold: 水平阈值线位置；None 表示不画阈值线。
        time_labels: 与 values 对齐的可读时间标签（zip 以较短者为准）。
    """
    color = COLORS[color_idx % len(COLORS)]

    if threshold is not None:
        fig.add_hline(y=threshold, line_dash='dash', line_color='#f59e0b',
                      line_width=1.5, row=row, col=1,
                      annotation_text=f'阈值={threshold}')

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='markers',
        name=qty_label,
        marker=dict(color=_SCATTER_COLOR, size=3, opacity=0.5),
        text=[
            f'{label}<br>{qty_label}: {value:.4f}{qty_unit}'
            for label, value in zip(time_labels or [], values)
        ],
        hoverinfo='text',
        showlegend=False,
    ), row=row, col=1)

    y_title = f'{qty_label}({qty_unit})' if qty_unit else qty_label
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=10, color=color),
        title_standoff=0,
        tickfont=dict(size=9),
        row=row, col=1,
    )
