"""
真值对比图表构建模块。
构建雷达 vs RTK 的对比子图（双线叠加/误差散点）。
降采样/子图布局等共享工具与统一色板见 chart_common。
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import get
from .chart_common import COLORS, _select_display_indices, _wrap_with_resampler


def _fmt_ts(epoch_sec):
    """按本地时区格式化 epoch 秒数，与 Plotly 日期横轴保持一致。"""
    if epoch_sec is None or (isinstance(epoch_sec, float) and np.isnan(epoch_sec)):
        return 'N/A'
    # comparison.parser 将 CSV 墙钟时间换算为真实 epoch 秒。Plotly 日期轴在
    # WebView/浏览器中按本地时区显示，悬浮框必须使用同一口径，避免相差 8 小时。
    # Windows CRT 的 localtime 不支持 1970-01-01 00:00 UTC 之前的时间戳
    # （负值抛 OSError [Errno 22]），先做 UTC 纯算术再取本地时区；测试用的
    # 合成时间戳可能落在 epoch 附近，此时以当前本地偏移兜底（真实数据为
    # 近期日期，不走此分支）。
    dt_utc = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(epoch_sec))
    try:
        dt = dt_utc.astimezone()
    except OSError:
        dt = dt_utc + datetime.now().astimezone().utcoffset()
    ms = dt.microsecond // 1000
    return f'{dt.year:04d}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:03d}'


def _fmt_clock(timestamp_label) -> str:
    """从统一时间标签中取 ``HH:mm:ss.SSS``，用于紧凑的图表悬浮框。"""
    return str(timestamp_label).strip().rsplit(' ', 1)[-1]


# 向量化版本，用于处理 numpy 数组
_fmt_ts_vec = np.vectorize(_fmt_ts)


def _find_radar_gap_indices(timestamps, gap_factor: float = 2.5) -> np.ndarray:
    """在完整雷达时间序列中查找真实中断的前帧索引。

    必须在显示降采样前调用。若在降采样后的点上判断，相邻保留点之间被
    跳过的正常帧会被误判为中断。
    """
    x = np.asarray(timestamps, dtype=float)
    if len(x) < 2:
        return np.array([], dtype=np.int64)
    diffs = np.diff(x)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(positive_diffs) == 0:
        return np.array([], dtype=np.int64)
    nominal_period = float(np.percentile(positive_diffs, 25))
    if nominal_period <= 0:
        return np.array([], dtype=np.int64)
    return np.flatnonzero(diffs > nominal_period * gap_factor).astype(np.int64)


def _insert_radar_gap_breaks(
    timestamps,
    values,
    hover_texts,
    gap_factor: float = 2.5,
    true_gap_start_times=None,
):
    """在真实雷达采样中断处插入 None 断点，并返回边界标记。

    ``true_gap_start_times`` 由完整时间序列预先计算。仅当该参数未提供时
    才回退到当前显示点间隔判断，保持旧调用的兼容性。
    """
    x = np.asarray(timestamps, dtype=float)
    y = np.asarray(values, dtype=float)
    texts = list(hover_texts)
    if len(x) < 2:
        return x.tolist(), y.tolist(), texts, [], [], []

    if true_gap_start_times is None:
        gap_indices = _find_radar_gap_indices(x, gap_factor)
        true_starts = x[gap_indices]
    else:
        true_starts = np.sort(np.asarray(true_gap_start_times, dtype=float))
    if len(true_starts) == 0:
        return x.tolist(), y.tolist(), texts, [], [], []

    plot_x, plot_y, plot_text = [], [], []
    marker_x, marker_y, marker_text = [], [], []
    for index, (x_value, y_value, text) in enumerate(zip(x, y, texts)):
        plot_x.append(float(x_value))
        plot_y.append(float(y_value))
        plot_text.append(text)
        if index < len(x) - 1:
            # 降采样后两个显示点之间可跨过很多正常帧；只有完整序列已确认
            # 的中断起点落在这个显示区间内，才允许断线。
            has_true_gap = np.any(
                (true_starts >= x_value - 1e-9)
                & (true_starts < x[index + 1] - 1e-9)
            )
            if not has_true_gap:
                continue
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

# RTK 对比色系：与雷达 COLORS 一一对应，形成强烈视觉对比
# 蓝↔珊瑚红  橙↔青绿  绿↔紫  红↔深蓝  紫↔琥珀  棕↔翠绿  粉↔深红  灰↔金
_RTK_COLORS = ['#e74c3c', '#16a085', '#8e44ad', '#3498db',
               '#e67e22', '#27ae60', '#c0392b', '#f39c12']

_SCATTER_COLOR = '#dc2626'     # 误差散点红色
_PERF_FAIL_COLOR = '#b91c1c'   # 误差不合格帧标记


def _add_fail_highlights(
    fig: go.Figure,
    row: int,
    timestamps: np.ndarray,
    frames: pd.DataFrame,
) -> None:
    """在物理量子图上高亮普通精度不合格帧。

    只添加失败帧标记点，不新增任何曲线，保持雷达/真值双曲线的原始
    阅读形态；标记 y 值取自雷达曲线本身，不改变子图坐标尺度。
    标记点随密度自适应缩小，海量重叠时降低透明度保持可读。
    悬停不弹出提示框（无实际意义）。
    """
    n = len(frames)
    if n == 0 or len(timestamps) != n:
        return
    valid = frames['valid'].to_numpy(dtype=bool)
    if not valid.any():
        return

    abs_err = frames['abs_error'].to_numpy(dtype=float)
    t_limit = frames['normal_limit'].to_numpy(dtype=float)
    normal_pass = frames['normal_pass'].to_numpy(dtype=bool)
    radar_val = frames['radar_value'].to_numpy(dtype=float)

    fail_mask = valid & np.isfinite(abs_err) & ~normal_pass
    if not fail_mask.any():
        return

    count = int(fail_mask.sum())
    if count > 500:
        size, opacity = 4, 0.65
    elif count > 100:
        size, opacity = 5, 0.75
    else:
        size, opacity = 6, 0.95

    fig.add_trace(go.Scatter(
        x=timestamps[fail_mask], y=radar_val[fail_mask],
        mode='markers', name='误差不合格帧',
        marker=dict(color=_PERF_FAIL_COLOR, size=size, symbol='circle',
                    opacity=opacity),
        # 悬浮提示框无实际意义，直接跳过，鼠标悬停不弹出任何方框
        hoverinfo='skip',
        showlegend=False,
    ), row=row, col=1)


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
    true_gap_start_times=None,
    show_difference: bool = True,
    radar_unit: str | None = None,
    rtk_unit: str | None = None,
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
    color = COLORS[color_idx % len(COLORS)]
    rtk_color = _RTK_COLORS[color_idx % len(_RTK_COLORS)]

    # 取两者都有效的公共点，确保悬停框能展示完整对比信息
    radar_series = pd.Series(radar_y, dtype=float)
    rtk_series = pd.Series(rtk_y, dtype=float)
    # “雷达”或“真值”可单独选择。单线模式不应落入双线公共有效点
    # 的判断，否则另一侧全为空时会被误认为没有可绘制的数据。
    radar_present = radar_series.notna().any()
    rtk_present = rtk_series.notna().any()
    radar_display_unit = y_unit if radar_unit is None else radar_unit
    rtk_display_unit = y_unit if rtk_unit is None else rtk_unit
    y_title = f'{y_label}({y_unit})' if y_unit else y_label
    yref = 'y domain' if row == 1 else f'y{row} domain'

    if radar_present and not rtk_present:
        valid = radar_series.notna().to_numpy()
        x_values = np.asarray(timestamps)[valid]
        values = radar_series[valid].to_numpy()
        labels = np.asarray(radar_time_rel)[valid]
        unit_suffix = f' {radar_display_unit}' if radar_display_unit else ''
        hover_texts = [
            f'{y_label}: {value:.3f}{unit_suffix}  {_fmt_clock(label)}'
            for label, value in zip(labels, values)
        ]
        plot_x, plot_y, plot_text, gap_x, gap_y, gap_text = _insert_radar_gap_breaks(
            x_values, values, hover_texts, gap_factor, true_gap_start_times,
        )
        fig.add_trace(go.Scatter(
            x=plot_x, y=plot_y, mode='lines', name='',
            line=dict(color=color, width=1.8), text=plot_text, hoverinfo='text',
            hoverlabel=dict(bgcolor='#ffffff', bordercolor='#000000',
                            font=dict(family='Consolas, Microsoft YaHei, monospace', size=12, color='#000000'),
                            showarrow=False),
            connectgaps=False, showlegend=False,
        ), row=row, col=1)
        if gap_x:
            fig.add_trace(go.Scatter(
                x=gap_x, y=gap_y, mode='markers',
                marker=dict(color='#dc2626', symbol='x', size=8), text=gap_text,
                hoverinfo='text', showlegend=False,
            ), row=row, col=1)
        fig.update_yaxes(title_text=y_title, title_font=dict(size=10, color=color),
                         title_standoff=0, tickfont=dict(size=9, color=color), row=row, col=1)
        fig.add_annotation(
            text=f'<b>{radar_name} {y_label}{f" ({radar_display_unit})" if radar_display_unit else ""}</b>',
            xref='x domain', yref=yref, x=0.99, y=0.93, xanchor='right', yanchor='middle',
            showarrow=False, bgcolor='rgba(255,255,255,0.78)',
            font=dict(size=11, color=color),
        )
        legend_shapes.append(dict(type='line', x0=0.99, y0=0.93, x1=1.0, y1=0.93,
                                  xref='x domain', yref=yref, line=dict(color=color, width=2.5)))
        return

    if rtk_present and not radar_present:
        if rtk_curve_timestamps is not None and rtk_curve_y is not None:
            curve_values = pd.Series(rtk_curve_y, dtype=float)
            valid = curve_values.notna().to_numpy()
            x_values = np.asarray(rtk_curve_timestamps)[valid]
            values = curve_values[valid].to_numpy()
            labels = np.asarray(rtk_curve_time_labels)[valid]
        else:
            valid = rtk_series.notna().to_numpy()
            x_values = np.asarray(timestamps)[valid]
            values = rtk_series[valid].to_numpy()
            labels = np.asarray(rtk_time_rel)[valid]
        unit_suffix = f' {rtk_display_unit}' if rtk_display_unit else ''
        fig.add_trace(go.Scatter(
            x=x_values, y=values, mode='lines', name='',
            line=dict(color=rtk_color, width=2.0, shape='spline', smoothing=1.3),
            text=[f'{rtk_source}: {value:.3f}{unit_suffix}  {_fmt_clock(label)}'
                  for label, value in zip(labels, values)],
            hoverinfo='text', connectgaps=False, showlegend=False,
        ), row=row, col=1)
        fig.update_yaxes(title_text=y_title, title_font=dict(size=10, color=rtk_color),
                         title_standoff=0, tickfont=dict(size=9, color=rtk_color), row=row, col=1)
        fig.add_annotation(
            text=f'<b>{rtk_name} {rtk_source}{f" ({rtk_display_unit})" if rtk_display_unit else ""}</b>',
            xref='x domain', yref=yref, x=0.99, y=0.93, xanchor='right', yanchor='middle',
            showarrow=False, bgcolor='rgba(255,255,255,0.78)',
            font=dict(size=11, color=rtk_color),
        )
        legend_shapes.append(dict(type='line', x0=0.99, y0=0.93, x1=1.0, y1=0.93,
                                  xref='x domain', yref=yref, line=dict(color=rtk_color, width=2.5)))
        return

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
    radar_unit_suffix = f' {radar_display_unit}' if radar_display_unit else ''
    rtk_unit_suffix = f' {rtk_display_unit}' if rtk_display_unit else ''
    error_unit_suffix = radar_unit_suffix if show_difference else ''
    radar_label = y_label.ljust(8)
    rtk_label = rtk_source.ljust(8)
    hover_texts = []
    for rt, rv, rkt, tv, ad in zip(
        radar_time_valid, r_valid, rtk_time_valid, t_valid, abs_diff_arr
    ):
        comparison_line = (
            f'<b>绝对误差 {ad:.3f}{error_unit_suffix}</b>'
            if show_difference else '<b>仅叠加显示（单位未确认或不兼容）</b>'
        )
        hover_texts.append(
            f'{radar_label}: {rv:.3f}{radar_unit_suffix}  {_fmt_clock(rt)}<br>'
            f'{rtk_label}: {tv:.3f}{rtk_unit_suffix}  {_fmt_clock(rkt)}<br>'
            f'{comparison_line}'
        )
    plot_x, plot_y, plot_text, gap_x, gap_y, gap_text = _insert_radar_gap_breaks(
        x_valid, r_valid, hover_texts, gap_factor, true_gap_start_times,
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
        text=f'<b>{radar_name} {y_label}{f" ({radar_display_unit})" if radar_display_unit else ""}</b>',
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
        text=f'<b>{rtk_name} {rtk_source}{f" ({rtk_display_unit})" if rtk_display_unit else ""}</b>',
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


def build_comparison_subplots(
    aligned_df: pd.DataFrame,
    selected_quantities: list,
    quantities_config: dict,
    trajectory_label: str = '',
    rtk_curve_df: Optional[pd.DataFrame] = None,
    perf_results: Optional[dict] = None,
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
        perf_results: 性能指标评估结果 {metric: evaluate_metric 返回}；
            overlay 子图会按物理量标签匹配并叠加动态阈值与违规标记（文档 10.4）。

    Returns:
        Plotly Figure。
    """
    n = len(selected_quantities)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title='请选择对比指标', template='plotly_white')
        return fig

    # 只缩小浏览器显示数据，误差统计仍由调用方基于完整 aligned_df 计算。
    display_columns: list[str] = []
    for quantity in selected_quantities:
        info = quantities_config.get(quantity, {})
        display_columns.extend([
            info.get('radar_col', ''), info.get('rtk_col', ''), info.get('field', ''),
        ])
    # 中断先在完整雷达序列计算，再将中断两侧点强制保留到显示索引中。
    # 这样曲线仍可按规模降采样，但不会把抽样跳点误显示成数据中断。
    full_timestamps = aligned_df['timestamp_parsed'].to_numpy(dtype=float)
    true_gap_indices = _find_radar_gap_indices(full_timestamps)
    display_indices = _select_display_indices(aligned_df, display_columns)
    if len(true_gap_indices):
        gap_boundary_indices = np.concatenate([
            true_gap_indices,
            np.minimum(true_gap_indices + 1, len(aligned_df) - 1),
        ])
        display_indices = np.unique(np.concatenate([display_indices, gap_boundary_indices]))
    aligned_df = aligned_df.iloc[display_indices].reset_index(drop=True)

    # 使用真实 epoch 毫秒并声明 date 坐标轴。不能使用相对秒，否则横轴会和
    # CSV 时间戳脱节；毫秒数同时保持中断检测函数所需的数值比较语义。
    timestamps = aligned_df['timestamp_parsed'].to_numpy(dtype=float) * 1000.0
    true_gap_start_times = full_timestamps[true_gap_indices] * 1000.0

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
        rtk_columns = [
            quantities_config.get(quantity, {}).get(
                'rtk_curve_col',
                quantities_config.get(quantity, {}).get('rtk_col', '').replace('rtk_', '', 1),
            )
            for quantity in selected_quantities
        ]
        if not rtk_curve_df.empty:
            rtk_display_indices = _select_display_indices(rtk_curve_df, rtk_columns)
            rtk_curve_df = rtk_curve_df.iloc[rtk_display_indices].reset_index(drop=True)
            rtk_curve_timestamps = rtk_curve_df['timestamp_parsed'].to_numpy(dtype=float) * 1000.0
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
            rtk_source_col = qty_info.get('rtk_curve_col', rtk_col.replace('rtk_', '', 1))
            continuous_rtk_y = (
                rtk_curve_df[rtk_source_col]
                if rtk_curve_df is not None and rtk_source_col in rtk_curve_df.columns else None
            )
            _build_overlay_subplot(
                fig, row, timestamps,
                # 缺少一侧列表示用户明确选择单线显示；传入等长 NaN，
                # 让子图构建器区分“该侧未选”与长度不一致的数据错误。
                aligned_df[radar_col] if radar_col in aligned_df.columns else np.full(len(aligned_df), np.nan),
                aligned_df[rtk_col] if rtk_col in aligned_df.columns else np.full(len(aligned_df), np.nan),
                qty_label, qty_unit, i, legend_shapes,
                radar_time_labels,
                rtk_time_labels,
                radar_name=qty_info.get('radar_source_label', '雷达'),
                rtk_name=qty_info.get('rtk_source_label', '真值'),
                rtk_source=qty_info.get('rtk_label', rtk_source_col if rtk_col else 'RTK'),
                rtk_curve_timestamps=rtk_curve_timestamps,
                rtk_curve_y=continuous_rtk_y,
                rtk_curve_time_labels=rtk_curve_time_labels,
                gap_factor=gap_factor,
                true_gap_start_times=true_gap_start_times,
                show_difference=qty_info.get('show_difference', True),
                radar_unit=qty_info.get('radar_unit'),
                rtk_unit=qty_info.get('rtk_unit'),
            )
            # 误差不合格帧高亮：仅标记点，不新增曲线，保持双曲线原始形态
            perf_result = (perf_results or {}).get(qty_label)
            if isinstance(perf_result, dict) and perf_result.get('frames') is not None:
                _add_fail_highlights(
                    fig, row, timestamps, perf_result['frames'],
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
            color = COLORS[i % len(COLORS)]
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
        type='date',
        tickformat='%H:%M:%S',
        hoverformat='%Y-%m-%d %H:%M:%S.%L',
        showspikes=True,
        spikemode='across',
        spikethickness=1,
        spikecolor='#94a3b8',
        spikedash='dot',
    )
    fig.update_xaxes(title_text='时间戳', row=n, col=1)

    fig.update_layout(
        title='',
        template='plotly_white',
        hovermode='x',
        margin=dict(l=50, r=10, t=30, b=30),
        dragmode='pan',
        shapes=legend_shapes,
    )

    curve_points = len(rtk_curve_df) if rtk_curve_df is not None else 0
    return _wrap_with_resampler(fig, max(len(aligned_df), curve_points))
