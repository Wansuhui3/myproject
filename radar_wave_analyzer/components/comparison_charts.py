"""
真值对比图表构建模块。
构建雷达 vs RTK 的对比子图（双线叠加/误差散点）。
复用 graph_builder 的 _compute_subplot_y_domains、_wrap_with_resampler、配色方案。
"""
from datetime import UTC, datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _fmt_ts(epoch_sec):
    """将 epoch 秒数格式化为统一的 ``YYYY-MM-DD HH:mm:ss.SSS``。"""
    if epoch_sec is None or (isinstance(epoch_sec, float) and np.isnan(epoch_sec)):
        return 'N/A'
    # comparison.parser 用无时区 datetime 相对 epoch 计算秒数；这里必须使用
    # UTC 解释该数值，不能受运行机器本地时区影响而平移 8 小时。
    dt = datetime.fromtimestamp(float(epoch_sec), UTC)
    ms = dt.microsecond // 1000
    return f'{dt.year:04d}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:03d}'


def _fmt_clock(timestamp_label) -> str:
    """从统一时间标签中取 ``HH:mm:ss.SSS``，用于紧凑的图表悬浮框。"""
    return str(timestamp_label).strip().rsplit(' ', 1)[-1]


# 向量化版本，用于处理 numpy 数组
_fmt_ts_vec = np.vectorize(_fmt_ts)


def _insert_radar_gap_breaks(timestamps, values, hover_texts, gap_factor: float = 2.5):
    """在雷达采样中断处插入 None 断点，并返回边界标记。

    阈值按该轨迹的中位采样间隔自适应计算，因此既适合 20Hz 数据，也适合
    其他采样率。RTK 曲线不经过这里，保持其原始连续采样。
    """
    x = np.asarray(timestamps, dtype=float)
    y = np.asarray(values, dtype=float)
    texts = list(hover_texts)
    if len(x) < 2:
        return x.tolist(), y.tolist(), texts, [], [], []

    positive_diffs = np.diff(x)
    positive_diffs = positive_diffs[positive_diffs > 0]
    if len(positive_diffs) == 0:
        return x.tolist(), y.tolist(), texts, [], [], []
    # 普通中位数会被长中断本身抬高；用较低分位数近似正常采样周期，
    # 才能在“少量正常帧 + 一次明显中断”的单文件数据中识别断点。
    nominal_period = float(np.percentile(positive_diffs, 25))
    gap_threshold = nominal_period * gap_factor

    plot_x, plot_y, plot_text = [], [], []
    marker_x, marker_y, marker_text = [], [], []
    for index, (x_value, y_value, text) in enumerate(zip(x, y, texts)):
        plot_x.append(float(x_value))
        plot_y.append(float(y_value))
        plot_text.append(text)
        if index < len(x) - 1 and x[index + 1] - x_value > gap_threshold:
            midpoint = float((x_value + x[index + 1]) / 2.0)
            plot_x.append(midpoint)
            plot_y.append(None)
            plot_text.append(None)
            gap_seconds = float(x[index + 1] - x_value)
            marker_x.extend([float(x_value), float(x[index + 1])])
            marker_y.extend([float(y[index]), float(y[index + 1])])
            marker_text.extend([
                f'雷达数据中断开始<br>间隔: {gap_seconds:.3f}s',
                f'雷达数据恢复<br>间隔: {gap_seconds:.3f}s',
            ])
    return plot_x, plot_y, plot_text, marker_x, marker_y, marker_text

try:
    from .graph_builder import _compute_subplot_y_domains, _wrap_with_resampler
except ImportError:
    from graph_builder import _compute_subplot_y_domains, _wrap_with_resampler  # type: ignore

try:
    from ..config import get
except ImportError:
    from config import get  # type: ignore

# 雷达配色（与 graph_builder 一致）
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# RTK 对比色系：与雷达 _COLORS 一一对应，形成强烈视觉对比
# 蓝↔珊瑚红  橙↔青绿  绿↔紫  红↔深蓝  紫↔琥珀  棕↔翠绿  粉↔深红  灰↔金
_RTK_COLORS = ['#e74c3c', '#16a085', '#8e44ad', '#3498db',
               '#e67e22', '#27ae60', '#c0392b', '#f39c12']

_SCATTER_COLOR = '#dc2626'     # 误差散点红色


def _build_overlay_subplot(
    fig: go.Figure,
    row: int,
    timestamps,
    radar_y,
    rtk_y,
    y_label: str,
    y_unit: str,
    color_idx: int,
    legend_shapes: list,
    radar_time_rel,
    rtk_time_rel,
    radar_name: str = 'radar',
    rtk_name: str = 'RTK',
    rtk_source: str = 'RTK',
    rtk_curve_timestamps=None,
    rtk_curve_y=None,
    rtk_curve_time_labels=None,
    gap_factor: float = 2.5,
):
    """构建一个双线叠加子图（雷达实线 + 真值平滑连续曲线），颜色对比鲜明。

    单一组合悬浮框（雷达 trace 承载全部信息，真值 trace 静默）：
    - 时间戳 + Dx (radar) + center_x (RTK) + |差值| 紧凑排列
    - 帧号行显示可读格式 "Jul 1, 2026, 17:15:29.576"。

    Args:
        fig: Plotly figure 对象。
        row: 行号(1-based)。
        timestamps: x轴时间序列（相对秒）。
        radar_y: 雷达数据。
        rtk_y: RTK插值数据。
        y_label: Y轴短标签。
        y_unit: 单位。
        color_idx: 配色索引。
        legend_shapes: 累积图例线段 shapes 列表（会原地追加）。
        radar_time_rel: 雷达帧的可读时间字符串数组（如 'Jul 1, 2026, 17:15:29.576'）。
        rtk_time_rel: 最近RTK采样的可读时间字符串数组。
        radar_name: 雷达系列图例前缀（如 'radar'）。
        rtk_name: 真值系列图例前缀（如 'RTK'）。
        rtk_source: 真值数据源名（如 'center_x'），用于图例后缀与悬停。
    """
    color = _COLORS[color_idx % len(_COLORS)]
    rtk_color = _RTK_COLORS[color_idx % len(_RTK_COLORS)]

    # 取两者都有效的公共点，确保悬停框能展示完整对比信息
    radar_series = pd.Series(radar_y, dtype=float)
    rtk_series = pd.Series(rtk_y, dtype=float)
    valid = radar_series.notna() & rtk_series.notna()
    x_valid = timestamps[valid]
    r_valid = radar_series[valid].values
    t_valid = rtk_series[valid].values
    radar_time_valid = np.asarray(radar_time_rel)[valid]
    rtk_time_valid = np.asarray(rtk_time_rel)[valid]
    n = len(x_valid)

    if n == 0:
        return

    # 差值：雷达实测值与真值插值结果的绝对误差
    diff_arr = r_valid - t_valid
    abs_diff_arr = np.abs(diff_arr)

    # ── 雷达实线（trace1）：承载唯一悬浮框，紧凑展示时间+雷达值+真值+差值 ──
    # hoverinfo="text" 屏蔽 Plotly 默认追加的 x 坐标数值，仅渲染 text 属性内容
    # hoverlabel showarrow=False 关闭悬浮框到曲线的蓝色连接引线
    unit_suffix = f' {y_unit}' if y_unit else ''
    radar_label = y_label.ljust(8)
    rtk_label = rtk_source.ljust(8)
    hover_texts = [
        f'{radar_label}: {rv:.3f}{unit_suffix}  {_fmt_clock(rt)}<br>'
        f'{rtk_label}: {tv:.3f}{unit_suffix}  {_fmt_clock(rkt)}<br>'
        f'<b>绝对误差 {ad:.3f}{unit_suffix}</b>'
        for rt, rv, rkt, tv, ad in zip(
            radar_time_valid, r_valid, rtk_time_valid, t_valid, abs_diff_arr
        )
    ]
    plot_x, plot_y, plot_text, gap_x, gap_y, gap_text = _insert_radar_gap_breaks(
        x_valid, r_valid, hover_texts, gap_factor,
    )
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y,
        mode='lines',
        name='',
        line=dict(color=color, width=1.8),
        text=plot_text,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='#ffffff', bordercolor='#000000',
            font=dict(family='Consolas, Microsoft YaHei, monospace', size=12, color='#000000'),
            showarrow=False,
        ),
        connectgaps=False,
        showlegend=False,
    ), row=row, col=1)

    if gap_x:
        fig.add_trace(go.Scatter(
            x=gap_x, y=gap_y,
            mode='markers',
            marker=dict(color='#dc2626', symbol='x', size=8),
            text=gap_text,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='#ffffff', bordercolor='#000000',
                font=dict(family='Microsoft YaHei, sans-serif', size=12, color='#000000'),
                showarrow=False,
            ),
            showlegend=False,
        ), row=row, col=1)

    # ── 真值平滑曲线：优先使用原始 RTK 连续采样，不随雷达缺帧而中断 ──
    if rtk_curve_timestamps is not None and rtk_curve_y is not None:
        curve_values = pd.Series(rtk_curve_y, dtype=float)
        curve_valid = curve_values.notna()
        rtk_x = np.asarray(rtk_curve_timestamps)[curve_valid]
        rtk_values = curve_values[curve_valid].to_numpy()
        rtk_labels = np.asarray(rtk_curve_time_labels)[curve_valid]
    else:
        rtk_x = x_valid
        rtk_values = t_valid
        rtk_labels = rtk_time_valid

    fig.add_trace(go.Scatter(
        x=rtk_x, y=rtk_values,
        mode='lines',
        name=f'{rtk_source} (RTK)',
        line=dict(color=rtk_color, width=2.0, shape='spline', smoothing=1.3),
        # 对比信息已经由雷达曲线承载，禁用 RTK 独立悬浮框避免重复。
        hoverinfo='skip',
        connectgaps=False,
        showlegend=False,
    ), row=row, col=1)

    # Y轴（使用雷达颜色，与波动分析一致）
    y_title = f'{y_label}({y_unit})' if y_unit else y_label
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=10, color=color),
        title_standoff=0,
        tickfont=dict(size=9, color=color),
        row=row, col=1,
    )

    # 图例标注：与雷达波动分析样式一致（白底 + 粗体 + 彩色短线段），雷达(上) / 真值(下)
    yref = 'y domain' if row == 1 else f'y{row} domain'
    leg_y_radar = 0.93
    leg_y_rtk = 0.82

    # 雷达
    fig.add_annotation(
        text=f'<b>{radar_name} {y_label} ({y_unit})</b>',
        xref='x domain', yref=yref,
        x=0.99, y=leg_y_radar,
        xanchor='right', yanchor='middle',
        showarrow=False,
        bgcolor='rgba(255,255,255,0.78)',
        font=dict(size=11, color=color),
    )
    legend_shapes.append(dict(
        type='line',
        x0=0.99, y0=leg_y_radar, x1=1.0, y1=leg_y_radar,
        xref='x domain', yref=yref,
        line=dict(color=color, width=2.5),
    ))

    # 真值
    fig.add_annotation(
        text=f'<b>{rtk_name} {rtk_source} ({y_unit})</b>',
        xref='x domain', yref=yref,
        x=0.99, y=leg_y_rtk,
        xanchor='right', yanchor='middle',
        showarrow=False,
        bgcolor='rgba(255,255,255,0.78)',
        font=dict(size=11, color=rtk_color),
    )
    legend_shapes.append(dict(
        type='line',
        x0=0.99, y0=leg_y_rtk, x1=1.0, y1=leg_y_rtk,
        xref='x domain', yref=yref,
        line=dict(color=rtk_color, width=2.5),
    ))


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
    color = _COLORS[color_idx % len(_COLORS)]

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


def build_comparison_subplots(
    aligned_df: pd.DataFrame,
    selected_quantities: list,
    quantities_config: dict,
    trajectory_label: str = '',
    rtk_curve_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """构建对比多子图（纵向堆叠，共享X轴）。数据驱动，根据 config 中 chart_type 决定子图类型。

    三种 chart_type：
    - overlay:  雷达实线 + 真值平滑曲线(对比色) 双线叠加
    - error:    误差散点(红) + 零线
    - scatter:  散点(红) + 可配置阈值线

    Args:
        aligned_df: 对齐结果 DataFrame，列含 radar_Dx/radar_Dy/radar_Vx/radar_Vy
                    及 rtk_center_x/rtk_center_y/rtk_Vx/rtk_Vy / pos_error_abs / match_dist。
        selected_quantities: 用户选中的对比指标列表。
        quantities_config: comparison.quantities 配置。
        trajectory_label: 轨迹标签（如 'ID=36'）。

    Returns:
        Plotly Figure。
    """
    n = len(selected_quantities)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title='请选择对比指标', template='plotly_white')
        return fig

    # 转换为相对时间（秒），避免 epoch 秒数在坐标轴上显示为科学计数法
    t0 = aligned_df['timestamp_parsed'].iloc[0]
    timestamps = (aligned_df['timestamp_parsed'] - t0).to_numpy(dtype=float)

    # 雷达/真值 CSV 真实时间戳 → 可读格式 "Jul 1, 2026, 17:15:29.576"
    if 'radar_ts_parsed' in aligned_df.columns:
        radar_time_labels = _fmt_ts_vec(aligned_df['radar_ts_parsed'].values)
    else:
        radar_time_labels = _fmt_ts_vec(aligned_df['timestamp_parsed'].values)

    if 'rtk_nearest_ts_parsed' in aligned_df.columns:
        rtk_time_labels = _fmt_ts_vec(aligned_df['rtk_nearest_ts_parsed'].values)
    else:
        rtk_time_labels = _fmt_ts_vec(aligned_df['timestamp_parsed'].values)

    rtk_curve_timestamps = None
    rtk_curve_time_labels = None
    if rtk_curve_df is not None and not rtk_curve_df.empty and 'timestamp_parsed' in rtk_curve_df.columns:
        rtk_curve_df = rtk_curve_df.sort_values('timestamp_parsed').reset_index(drop=True)
        rtk_curve_timestamps = (rtk_curve_df['timestamp_parsed'] - t0).to_numpy(dtype=float)
        rtk_curve_time_labels = _fmt_ts_vec(rtk_curve_df['timestamp_parsed'].values)

    gap_factor = get('comparison', {}).get('radar_gap_break_factor', 2.5)

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[1] * n,
    )

    # 累积图例短线段（与 graph_builder 一致，避免覆盖高亮 shape）
    legend_shapes: list[dict] = []

    for i, qty in enumerate(selected_quantities):
        qty_info = quantities_config.get(qty, {})
        qty_label = qty_info.get('label', qty)
        qty_unit = qty_info.get('unit', '')
        chart_type = qty_info.get('chart_type', 'overlay')
        row = i + 1

        if chart_type == 'overlay':
            radar_col = qty_info.get('radar_col', '')
            rtk_col = qty_info.get('rtk_col', '')
            rtk_source_col = rtk_col.replace('rtk_', '', 1)
            continuous_rtk_y = (
                rtk_curve_df[rtk_source_col]
                if rtk_curve_df is not None and rtk_source_col in rtk_curve_df.columns else None
            )
            _build_overlay_subplot(
                fig, row, timestamps,
                aligned_df[radar_col] if radar_col in aligned_df.columns else [],
                aligned_df[rtk_col] if rtk_col in aligned_df.columns else [],
                qty_label, qty_unit, i, legend_shapes,
                radar_time_labels,
                rtk_time_labels,
                radar_name='radar',
                rtk_name='RTK',
                rtk_source=rtk_source_col if rtk_col else 'RTK',
                rtk_curve_timestamps=rtk_curve_timestamps,
                rtk_curve_y=continuous_rtk_y,
                rtk_curve_time_labels=rtk_curve_time_labels,
                gap_factor=gap_factor,
            )

        elif chart_type == 'error':
            field = qty_info.get('field', '')
            _build_error_subplot(
                fig, row, timestamps,
                aligned_df[field] if field in aligned_df.columns else [],
                qty_label, qty_unit, i, zero_line=True, time_labels=radar_time_labels,
            )

        elif chart_type == 'scatter':
            field = qty_info.get('field', '')
            threshold = qty_info.get('threshold')
            color = _COLORS[i % len(_COLORS)]
            if threshold is not None:
                fig.add_hline(y=threshold, line_dash='dash', line_color='#f59e0b',
                              line_width=1.5, row=row, col=1,
                              annotation_text=f'阈值={threshold}')
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=aligned_df[field] if field in aligned_df.columns else [],
                mode='markers',
                name=qty_label,
                marker=dict(color=_SCATTER_COLOR, size=3, opacity=0.5),
                text=[
                    f'{label}<br>{qty_label}: {value:.4f}{qty_unit}'
                    for label, value in zip(radar_time_labels, aligned_df[field])
                ] if field in aligned_df.columns else [],
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

    # X轴标题（仅最底行）+ spike 跨图同步竖线
    fig.update_xaxes(
        tickformat='.0f',
        showspikes=True,
        spikemode='across',
        spikethickness=1,
        spikecolor='#94a3b8',
        spikedash='dot',
    )
    fig.update_xaxes(title_text='时间 (s)', row=n, col=1)

    fig.update_layout(
        title='',
        template='plotly_white',
        hovermode='x',
        margin=dict(l=50, r=10, t=30, b=30),
        dragmode='pan',
        shapes=legend_shapes,
    )

    return _wrap_with_resampler(fig, len(aligned_df))


def build_delay_curve_chart(delay_results: dict) -> go.Figure:
    """构建延迟扫描曲线。

    Args:
        delay_results: scan_delay 返回的结果。

    Returns:
        Plotly Figure。
    """
    curve = delay_results.get('delay_curve', [])
    if not curve:
        fig = go.Figure()
        fig.update_layout(title='无延迟数据', template='plotly_white')
        return fig

    delays = [d for d, _ in curve]
    rmses = [r for _, r in curve]
    best_ms = delay_results.get('optimal_delay_ms', 0)
    best_rmse = delay_results.get('min_rmse', 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=delays, y=rmses,
        mode='lines+markers',
        name='RMSE',
        line=dict(color='#2563eb', width=2),
        marker=dict(size=5),
        hovertemplate='延迟: %{x}ms<br>RMSE: %{y:.4f}m<extra></extra>',
    ))

    # 无候选满足覆盖率门槛时不绘制伪“最优”延迟线。
    if best_rmse is not None:
        fig.add_vline(x=best_ms, line_dash='dash', line_color='#dc2626',
                      line_width=1.5,
                      annotation_text=f'{best_ms}ms<br>RMSE={best_rmse}m',
                      annotation_font=dict(size=10, color='#dc2626'))

    fig.update_layout(
        title='RMSE vs 时间延迟',
        xaxis_title='延迟 (ms)',
        yaxis_title='RMSE (m)',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig
