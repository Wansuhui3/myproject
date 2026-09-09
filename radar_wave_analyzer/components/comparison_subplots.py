"""真值对比子图构建器。

overlay（双线叠加）/ error（误差散点）/ scatter（阈值散点）三种 chart_type
的绘制实现，以及误差不合格帧高亮。编排逻辑见 comparison_charts。
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .chart_common import COLORS
from .comparison_time import _fmt_clock
from .radar_gap import _insert_radar_gap_breaks

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

    # 取两者都有效的公共点，确保悬浮框能展示完整对比信息
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
