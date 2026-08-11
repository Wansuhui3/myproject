"""
Plotly 图表构建模块。
集成 plotly-resampler 实现大数据量自动降采样。
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from ..config import get
except ImportError:
    from config import get

# 尝试导入 FigureResampler（可选依赖，缺失时降级为普通 figure）
try:
    from plotly_resampler import FigureResampler  # pyright: ignore[reportMissingImports]
    _RESAMPLER_AVAILABLE = True
except ImportError:
    _RESAMPLER_AVAILABLE = False


def _select_display_indices(
    df: pd.DataFrame,
    columns: list[str],
    max_points: Optional[int] = None,
) -> np.ndarray:
    """为显示选择代表性行，保留每个时间桶内各曲线的最小/最大值。

    统计、框选和导出继续使用原始 DataFrame；这里只缩小发送给 Plotly 的
    JSON 体积。多物理量共用同一组索引，保证共享 X 轴严格对齐。
    """
    row_count = len(df)
    if row_count == 0:
        return np.array([], dtype=np.int64)
    if max_points is None:
        max_points = int(get('DISPLAY_MAX_POINTS', 3000))
    if (not get('DISPLAY_DOWNSAMPLING_ENABLED', True)
            or row_count <= max_points or max_points < 4):
        return np.arange(row_count, dtype=np.int64)

    numeric_columns = [column for column in columns if column in df.columns]
    if not numeric_columns:
        return np.unique(np.linspace(0, row_count - 1, max_points, dtype=np.int64))

    bucket_count = max(1, (max_points - 2) // (2 * len(numeric_columns)))
    edges = np.linspace(0, row_count, bucket_count + 1, dtype=np.int64)
    selected: set[int] = {0, row_count - 1}
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
        for start, end in zip(edges[:-1], edges[1:]):
            if end <= start:
                continue
            block = values[start:end]
            finite_positions = np.flatnonzero(np.isfinite(block))
            if len(finite_positions) == 0:
                continue
            finite_values = block[finite_positions]
            selected.add(int(start + finite_positions[int(np.argmin(finite_values))]))
            selected.add(int(start + finite_positions[int(np.argmax(finite_values))]))

    result = np.array(sorted(selected), dtype=np.int64)
    if len(result) > max_points:
        keep = np.linspace(0, len(result) - 1, max_points, dtype=np.int64)
        result = result[keep]
    return result


def _wrap_with_resampler(fig: go.Figure, n_points: int) -> go.Figure:
    """数据点超过阈值时用 FigureResampler 包装，实现交互式降采样。

    小数据集（<= RESAMPLER_MAX_POINTS）原样返回，避免不必要的回调注册。
    注意：FigureResampler 需要 Dash app 上下文（用于注册重采样回调），
    在无 app 上下文时（如单元测试）会自动降级为普通 figure。
    """
    if not _RESAMPLER_AVAILABLE or not get('RESAMPLER_ENABLED', True):
        return fig
    max_pts = get('RESAMPLER_MAX_POINTS', 5000)
    if n_points <= max_pts:
        return fig
    n_samples = get('RESAMPLER_DEFAULT_N_SAMPLES', 2000)
    try:
        # 兼容不同版本参数名：default_n_samples (>=0.9) / max_n_samples (旧版)
        try:
            return FigureResampler(fig, default_n_samples=n_samples)
        except TypeError:
            return FigureResampler(fig, max_n_samples=n_samples)
    except Exception:  # noqa: BLE001
        # 无 app 上下文或构造失败时降级为普通 figure，保证可用性
        return fig


def _compute_subplot_y_domains(n: int, vertical_spacing: float = 0.03) -> list:
    """计算每个子图在 paper 坐标中的 y 范围 [y_bottom, y_top]。

    返回列表，索引 0 对应第 1 行（顶部），索引 n-1 对应第 n 行（底部）。
    """
    if n <= 0:
        return []
    total_spacing = vertical_spacing * max(0, n - 1)
    subplot_height = (1.0 - total_spacing) / n
    domains = []
    for i in range(n):
        y_top = 1.0 - i * (subplot_height + vertical_spacing)
        y_bottom = y_top - subplot_height
        domains.append((y_bottom, y_top))
    return domains


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
    try:
        from ..core.wave_calc import find_max_jump as _find_max_jump
    except ImportError:
        from core.wave_calc import find_max_jump as _find_max_jump

    all_quantities = get('quantities', {})
    timestamps = seg_df['timestamp_parsed']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    n = len(quantities_list)
    display_indices = _select_display_indices(seg_df, quantities_list)
    display_timestamps = timestamps.iloc[display_indices]

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[1] * n,
    )

    # 收集子图标注形状（彩色短线段）
    legend_shapes: list[dict] = []

    for i, qty in enumerate(quantities_list):
        qty_info = all_quantities.get(qty, {})
        qty_label = qty_info.get('label', qty)
        qty_unit = qty_info.get('unit', '')
        color = colors[i % len(colors)]
        row = i + 1

        # 主曲线（SVG 渲染，保证 selectedData 框选事件兼容 Plotly 6.x）
        fig.add_trace(go.Scatter(
            x=display_timestamps,
            y=seg_df[qty].to_numpy()[display_indices],
            mode='lines',
            name=qty_label,
            line=dict(color=color, width=1.5),
            hovertemplate=f'{qty_label}: %{{y:.4f}}{qty_unit}<extra></extra>',
            showlegend=False,
        ), row=row, col=1)

        # 最大跳变标记
        max_jump = _find_max_jump(seg_df, qty, diff_cache=diff_cache)
        if max_jump is not None and 0 < max_jump['idx'] < len(seg_df):
            prev_row_data = seg_df.iloc[max_jump['idx'] - 1]
            curr_row_data = seg_df.iloc[max_jump['idx']]
            fig.add_trace(go.Scatter(
                x=[prev_row_data['timestamp_parsed'], curr_row_data['timestamp_parsed']],
                y=[prev_row_data[qty], curr_row_data[qty]],
                mode='lines',
                name=f'{qty_label} 跳变',
                line=dict(color='#dc2626', width=3),
                showlegend=False,
                hovertemplate=(
                    f'<b>最大跳变</b><br>{qty_label}: %{{y:.4f}}{qty_unit}<extra></extra>'
                ),
            ), row=row, col=1)

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

    # X 轴标题（仅最底行）
    fig.update_xaxes(title_text='时间', row=n, col=1)

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
        hovermode='x unified',
        margin=dict(l=50, r=10, t=30, b=30),
        dragmode='select',
        shapes=hl_shapes + legend_shapes,
    )

    if use_resampler:
        return _wrap_with_resampler(fig, len(display_indices))
    return fig
