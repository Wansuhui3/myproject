"""
真值对比图表构建模块（编排器）。
构建雷达 vs RTK 的对比子图（双线叠加/误差散点）。
子图绘制实现见 comparison_overlay（叠加主图）/ comparison_subplots
（误差、阈值散点），时间格式化见 comparison_time，
雷达中断处理见 radar_gap，共享工具与统一色板见 chart_common。
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import get
from .chart_common import _select_display_indices, _wrap_with_resampler
from .comparison_overlay import _add_fail_highlights, _build_overlay_subplot
from .comparison_subplots import _build_error_subplot, _build_scatter_subplot
from .comparison_time import _fmt_ts_vec
from .radar_gap import _find_radar_gap_indices


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
            _build_scatter_subplot(
                fig, row, timestamps,
                aligned_df[field] if field in aligned_df.columns else [],
                qty_label, qty_unit, i,
                threshold=qty_info.get('threshold'),
                time_labels=radar_time_labels,
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
        # 深度缩放后点间距可超默认 hoverdistance=20px 导致悬停概率性失效，
        # -1 = 不限制吸附距离（与波动页保持一致）
        hoverdistance=-1,
        spikedistance=-1,
        margin=dict(l=50, r=10, t=30, b=30),
        dragmode='pan',
        shapes=legend_shapes,
    )

    curve_points = len(rtk_curve_df) if rtk_curve_df is not None else 0
    return _wrap_with_resampler(fig, max(len(aligned_df), curve_points))
