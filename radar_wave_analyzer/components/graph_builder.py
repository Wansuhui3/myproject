"""
Plotly 图表构建模块（波动分析域）。
多物理量子图、框选高亮、最大跳变标记；降采样/子图布局等共享工具见 chart_common。
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import get
from .chart_common import (
    COLORS,
    _compute_subplot_y_domains,
    _select_display_indices,
    _wrap_with_resampler,
)


def _get_scatter_cls(n_panels: int = 1):
    """按配置与面板数返回主曲线渲染器：webgl=Scattergl(GPU) / svg=Scatter(CPU)。

    已从 plotly.js 源码证实：WebGL 上下文按子图独立创建、无共享池。
    N 个面板 = N 个 GL 上下文，缩放时多画布异步重绘导致闪烁抽搐，
    因此 webgl 模式在面板数超过 WEBGL_MAX_SUBPLOTS 时自动回退 svg。

    框选统计只依赖 selectedData.range（框选矩形坐标边界），与渲染器无关。
    """
    if str(get('GRAPH_RENDERER', 'svg')).lower() != 'webgl':
        return go.Scatter
    if n_panels > int(get('WEBGL_MAX_SUBPLOTS', 4)):
        return go.Scatter
    return go.Scattergl


def build_highlight_shapes(
    seg_df: pd.DataFrame,
    quantities_list: list,
    highlight_range: Optional[tuple[int, int]] = None,
    vertical_spacing: float = 0.03,
    highlight_time_range: Optional[tuple[object, object]] = None,
) -> list:
    """为每个子图构建高亮矩形 shape（用 paper y 坐标 + 数据 x 坐标）。

    返回 Plotly shape dict 列表，可直接用于 fig.update_layout(shapes=...)
    或 Dash Patch 的 patch['layout']['shapes'] = ...
    """
    n = len(quantities_list)
    if n == 0:
        return []

    if highlight_time_range is not None:
        x0, x1 = highlight_time_range
    elif highlight_range is not None:
        start_idx, end_idx = highlight_range
        if not (0 <= start_idx < end_idx < len(seg_df)):
            return []
        hl = seg_df.iloc[start_idx:end_idx + 1]
        x0 = hl['timestamp_parsed'].iloc[0]
        x1 = hl['timestamp_parsed'].iloc[-1]
    else:
        return []

    try:
        x0, x1 = pd.Timestamp(x0), pd.Timestamp(x1)
    except (TypeError, ValueError):
        return []
    if x0 > x1:
        x0, x1 = x1, x0
    # 转为 ISO 字符串以确保 JSON 可序列化（Patch 需要）
    if hasattr(x0, 'isoformat'):
        x0 = x0.isoformat()
    if hasattr(x1, 'isoformat'):
        x1 = x1.isoformat()

    domains = _compute_subplot_y_domains(n, vertical_spacing)

    shapes = []
    for i in range(n):
        xref = 'x' if i == 0 else f'x{i + 1}'
        y_bottom, y_top = domains[i]
        shapes.append(dict(
            type='rect',
            xref=xref,
            yref='paper',
            x0=x0,
            x1=x1,
            y0=y_bottom,
            y1=y_top,
            fillcolor='rgba(255, 127, 14, 0.12)',
            line=dict(color='rgba(255, 127, 14, 0.6)', width=1),
            layer='below',
        ))
    return shapes


BOX_JUMP_COLOR = '#7c3aed'   # 框选区间最大跳变标记（区别于全段跳变红线 #dc2626）


def build_box_jump_shapes(
    masked_df: pd.DataFrame,
    quantities_list: list,
    diff_cache: Optional[dict] = None,
) -> list:
    """为框选区间内每个物理量的最大跳变构建高亮线段 shape。

    在框选子区间数据上重新定位最大跳变（帧间差分绝对值最大处），以
    紫色加粗线段叠加显示，区别于全段最大跳变的红色线段；由
    callbacks._get_non_highlight_shapes 按颜色识别并在清除框选时移除，
    不影响全段跳变标记。

    Args:
        masked_df: 框选区间内的轨迹数据（至少 2 帧）。
        quantities_list: 物理量字段名列表（与子图顺序一致）。
        diff_cache: 帧间差分缓存。

    Returns:
        Plotly shape dict 列表。
    """
    if masked_df is None or len(masked_df) < 2 or not quantities_list:
        return []

    from ..core.wave_calc import find_max_jump as _find_max_jump

    shapes: list[dict] = []
    for i, qty in enumerate(quantities_list):
        if qty not in masked_df.columns:
            continue
        jump = _find_max_jump(masked_df, qty, diff_cache=diff_cache)
        if not jump or jump['idx'] < 1 or jump['idx'] >= len(masked_df):
            continue
        prev_row = masked_df.iloc[jump['idx'] - 1]
        curr_row = masked_df.iloc[jump['idx']]
        xref = 'x' if i == 0 else f'x{i + 1}'
        yref = 'y' if i == 0 else f'y{i + 1}'
        shapes.append(dict(
            type='line',
            xref=xref, yref=yref,
            x0=prev_row['timestamp_parsed'].isoformat(),
            x1=curr_row['timestamp_parsed'].isoformat(),
            y0=float(prev_row[qty]),
            y1=float(curr_row[qty]),
            line=dict(color=BOX_JUMP_COLOR, width=5),
            layer='above',
        ))
    return shapes


def build_multi_subplot_graph(
    seg_df: pd.DataFrame,
    quantities_list: list[str],
    trajectory_id: str = '',
    highlight_range: Optional[tuple[int, int]] = None,
    highlight_time_range: Optional[tuple[object, object]] = None,
    use_resampler: bool = True,
    diff_cache: Optional[dict] = None,
) -> go.Figure:
    """构建多物理量纵向堆叠子图（共享 X 轴）。

    每个物理量独立子图，各自 Y 轴 + 最大跳变标记。
    框选高亮按需叠加到所有子图。

    Args:
        seg_df: 轨迹段 DataFrame。
        quantities_list: 物理量字段名列表。
        trajectory_id: 轨迹 ID。
        highlight_range: 高亮区间 (start_idx, end_idx)。
        highlight_time_range: 鼠标框选的精确时间边界，优先于 highlight_range。
        use_resampler: 是否启用降采样包装。

    Returns:
        Plotly Figure 对象（含 n 行子图）。
    """
    from ..core.wave_calc import find_max_jump as _find_max_jump

    if not quantities_list:
        fig = go.Figure()
        fig.add_annotation(
            text='当前轨迹中没有可绘制的已选物理量',
            x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14, color='#64748b'),
        )
        fig.update_layout(
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        )
        return fig

    all_quantities = get('quantities', {})
    timestamps = seg_df['timestamp_parsed']
    n = len(quantities_list)
    display_indices = _select_display_indices(seg_df, quantities_list)
    display_timestamps = timestamps.iloc[display_indices]
    # 渲染器按面板数解析：webgl 多面板时自动回退 svg（避免多个独立 GL 上下文闪烁）
    scatter_cls = _get_scatter_cls(n)

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[1] * n,
    )

    # 收集子图标注形状（彩色短线段）
    legend_shapes: list[dict] = []
    # 最大跳变标记改为 shape：较 trace 少一份每帧重绘路径与悬停扫描开销
    jump_marker_shapes: list[dict] = []

    for i, qty in enumerate(quantities_list):
        qty_info = all_quantities.get(qty, {})
        qty_label = qty_info.get('label', qty)
        qty_unit = qty_info.get('unit', '')
        color = COLORS[i % len(COLORS)]
        row = i + 1
        xref = 'x' if row == 1 else f'x{row}'
        yref_axis = 'y' if row == 1 else f'y{row}'
        displayed_values = seg_df[qty].to_numpy()[display_indices]

        # 单 trace 同时负责绘制与悬停。旧实现额外叠加一层 1200 点透明 marker，
        # 横向缩放时每个面板要重绘两份几何对象；线段本身已可命中悬停，无需
        # 复制整套数据。
        fig.add_trace(scatter_cls(
            x=display_timestamps,
            y=displayed_values,
            mode='lines',
            name=qty_label,
            line=dict(color=color, width=2),
            hovertemplate=f'{qty_label}: %{{y:.4f}}{qty_unit}<extra></extra>',
            showlegend=False,
        ), row=row, col=1)

        # 最大跳变标记（shape 形式，绘制在曲线层之上）
        max_jump = _find_max_jump(seg_df, qty, diff_cache=diff_cache)
        if max_jump is not None and 0 < max_jump['idx'] < len(seg_df):
            prev_row_data = seg_df.iloc[max_jump['idx'] - 1]
            curr_row_data = seg_df.iloc[max_jump['idx']]
            jump_marker_shapes.append(dict(
                type='line',
                xref=xref, yref=yref_axis,
                x0=prev_row_data['timestamp_parsed'].isoformat(),
                x1=curr_row_data['timestamp_parsed'].isoformat(),
                y0=float(prev_row_data[qty]),
                y1=float(curr_row_data[qty]),
                line=dict(color='#dc2626', width=3),
            ))

        # 非底部 X 轴完全隐藏：shared_xaxes 仅隐藏刻度标签、仍计算刻度；
        # visible=False 让 plotly 跳过这些轴的刻度计算与绘制（缩放经 matches 同步）
        if row < n:
            fig.update_xaxes(visible=False, row=row, col=1)

        # 紧凑 Y 轴：彩色短标题，紧贴轴线，消除大留白
        short_label = qty_info.get('short_label', qty_label[:4])
        y_title = f'{short_label}({qty_unit})' if qty_unit else short_label
        fig.update_yaxes(
            title_text=y_title,
            title_font=dict(size=10, color=color),
            title_standoff=0,
            tickfont=dict(size=9, color=color),
            row=row, col=1,
        )

        # 子图右上角标注：文字标签 + 紧挨短线段（白底无边框，替代全局图例）
        yref = 'y domain' if row == 1 else f'y{row} domain'
        leg_y = 0.93

        # 文字标签（白底无边框，右对齐）
        fig.add_annotation(
            text=f'<b>{qty_label} ({qty_unit})</b>',
            xref='x domain', yref=yref,
            x=0.99, y=leg_y,
            xanchor='right', yanchor='middle',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.78)',
            font=dict(size=11, color=color),
        )
        # 短线段紧跟在文字右侧（作为图形标注累积到 shapes 中，避免覆盖高亮 shape）
        legend_shapes.append(dict(
            type='line',
            x0=0.99, y0=leg_y, x1=1.0, y1=leg_y,
            xref='x domain', yref=yref,
            line=dict(color=color, width=2.5),
        ))

    # 仅最底部显示共享时间轴。显式设置刻度属性，避免多子图构建时继承上方
    # 隐藏轴的 showticklabels=False。
    fig.update_xaxes(
        title_text='时间',
        visible=True,
        showticklabels=True,
        tickfont=dict(size=10, color='#475569'),
        row=n,
        col=1,
    )

    # 布局：不设固定 height，由 CSS 容器 + responsive 撑满
    # 标题由外部 graph-title-bar 元素统一管理，Plotly 内置标题不再显示

    hl_shapes = []
    if highlight_time_range is not None or highlight_range is not None:
        hl_shapes = build_highlight_shapes(
            seg_df, quantities_list, highlight_range, vertical_spacing=0.01,
            highlight_time_range=highlight_time_range,
        )

    fig.update_layout(
        title='',
        template='plotly_white',
        # x unified：悬停时显示竖直参考线，可精确定位当前 X 位置（时间）
        hovermode='x unified',
        margin=dict(l=50, r=10, t=30, b=30),
        dragmode='select',
        # 缩放/平移状态跨图重建保持：勾选物理量、框选高亮等触发的重建不再重置视图；
        # 切换轨迹时 ID 变化 → 视图重置
        uirevision=f'traj-{trajectory_id}' if trajectory_id else 'default',
        shapes=hl_shapes + jump_marker_shapes + legend_shapes,
    )

    if use_resampler:
        return _wrap_with_resampler(fig, len(display_indices))
    return fig
