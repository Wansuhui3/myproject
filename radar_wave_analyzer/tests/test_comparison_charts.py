"""真值对比图的时间提示和雷达中断渲染测试。"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from radar_wave_analyzer.components.comparison_charts import (  # noqa: E402
    build_comparison_subplots,
)
from radar_wave_analyzer.components.comparison_time import _fmt_ts  # noqa: E402


def test_tooltip_epoch_uses_the_same_local_timezone_as_the_date_axis():
    """悬浮时间必须与浏览器日期横轴的本地时区口径一致。"""
    # 经 UTC 感知路径取本地时间：Windows + Python 3.13 下 naive astimezone()
    # 对 epoch 早期时间戳会抛 OSError。
    local = datetime.fromtimestamp(0.0, tz=timezone.utc).astimezone()
    assert _fmt_ts(0.0) == local.strftime('%Y-%m-%d %H:%M:%S.') + '000'


def test_perf_fail_frames_highlighted_without_extra_lines():
    """传入 perf_results 时仅在误差不合格帧位置添加标记点。

    保持雷达/真值双曲线原始形态：不新增阈值曲线、误差曲线与背景高亮。
    """
    n = 12
    # 构造误差：大部分 0.1（合格），第 3-5 帧 0.6（超 T=0.3）
    errors = np.full(n, 0.1)
    errors[3:6] = 0.6
    aligned = pd.DataFrame({
        'timestamp_parsed': np.arange(n, dtype=float) * 0.05,
        'radar_ts_parsed': np.arange(n, dtype=float) * 0.05,
        'rtk_nearest_ts_parsed': np.arange(n, dtype=float) * 0.05,
        'time_diff_ms': [0.0] * n,
        'radar_Dx': 5.0 + errors,
        'rtk_center_x': 5.0,
    })
    # frames 由 evaluate_metric 生成，这里直接构造等价结构
    frames = pd.DataFrame({
        'radar_frame': np.arange(1, n + 1),
        'timestamp': [f't{i}' for i in range(n)],
        'distance_bin': np.zeros(n, dtype=int),
        'truth_distance': np.full(n, 5.0),
        'radar_value': 5.0 + errors,
        'truth_value': np.full(n, 5.0),
        'signed_error': errors,
        'abs_error': errors,
        'normal_limit': np.full(n, 0.3),
        'normal_pass': errors < 0.3,
        'three_frame_limit': np.full(n, 0.5),
        'normalized_severe_error': np.full(n, np.nan),
        'three_frame_violation': np.zeros(n, dtype=bool),
        'valid': np.ones(n, dtype=bool),
    })
    quantities = {
        'cmp_dx': {
            'label': 'Dx', 'unit': 'm', 'radar_col': 'radar_Dx',
            'rtk_col': 'rtk_center_x', 'chart_type': 'overlay',
        },
    }
    perf_results = {'Dx': {
        'available': True, 'mode': 'binned',
        'bins': [], 'overall': {}, 'coverage': {},
        'frames': frames,
    }}

    fig = build_comparison_subplots(
        aligned, ['cmp_dx'], quantities, perf_results=perf_results)

    # 雷达 + 真值（无 gap 不生成中断标记）+ 不合格帧标记 = 3，无多余曲线
    assert len(fig.data) == 3
    fail_trace = fig.data[2]
    assert fail_trace.mode == 'markers'
    assert fail_trace.marker.color == '#b91c1c'
    # 仅 3 个不合格帧被标记
    assert len(fail_trace.x) == 3
    # 标记 y 值取自雷达曲线本身，不引入新的 y 尺度
    assert list(fail_trace.y) == pytest.approx([5.6, 5.6, 5.6])
    # 悬浮提示框已删除：悬停不合格标记不弹出任何方框
    assert fail_trace.hoverinfo == 'skip'
    # 无违规背景高亮形状（图例线段除外）
    violation_fills = [
        s for s in fig.layout.shapes
        if getattr(s, 'fillcolor', None)
    ]
    assert len(violation_fills) == 0


def test_perf_overlays_absent_without_perf_results():
    """未传 perf_results 时图表保持原有 trace 结构，不受影响。"""
    n = 3
    aligned = pd.DataFrame({
        'timestamp_parsed': np.arange(n, dtype=float) * 0.05,
        'radar_ts_parsed': np.arange(n, dtype=float) * 0.05,
        'rtk_nearest_ts_parsed': np.arange(n, dtype=float) * 0.05,
        'time_diff_ms': [0.0] * n,
        'radar_Dx': [0.0, 0.5, 5.0],
        'rtk_center_x': [0.0, 0.5, 5.0],
    })
    quantities = {
        'cmp_dx': {
            'label': 'Dx', 'unit': 'm', 'radar_col': 'radar_Dx',
            'rtk_col': 'rtk_center_x', 'chart_type': 'overlay',
        },
    }
    fig = build_comparison_subplots(aligned, ['cmp_dx'], quantities)
    assert len(fig.data) == 2


def test_radar_only_mapping_renders_a_single_line():
    """真值字段留空时，雷达字段仍应单独绘制。"""
    aligned = pd.DataFrame({
        'timestamp_parsed': [0.0, 0.05, 0.10],
        'radar_ts_parsed': [0.0, 0.05, 0.10],
        'rtk_nearest_ts_parsed': [0.0, 0.05, 0.10],
        'radar[Rx_front]': [1.0, 2.0, 3.0],
    })
    quantities = {
        'radar_rx': {
            'label': 'Rx_front', 'unit': 'm',
            'radar_col': 'radar[Rx_front]', 'rtk_col': '', 'chart_type': 'overlay',
            'radar_source_label': '雷达', 'rtk_source_label': '真值',
        },
    }

    fig = build_comparison_subplots(aligned, ['radar_rx'], quantities)

    assert len(fig.data) == 1
    assert list(fig.data[0].y) == pytest.approx([1.0, 2.0, 3.0])


def test_truth_only_mapping_renders_a_single_line():
    """雷达字段留空时，真值字段仍应单独绘制。"""
    aligned = pd.DataFrame({
        'timestamp_parsed': [0.0, 0.05, 0.10],
        'radar_ts_parsed': [0.0, 0.05, 0.10],
        'rtk_nearest_ts_parsed': [0.0, 0.05, 0.10],
        'rtk[center_x]': [4.0, 5.0, 6.0],
    })
    quantities = {
        'rtk_x': {
            'label': 'center_x', 'unit': 'm',
            'radar_col': '', 'rtk_col': 'rtk[center_x]', 'chart_type': 'overlay',
            'radar_source_label': '雷达', 'rtk_source_label': '真值',
        },
    }

    fig = build_comparison_subplots(aligned, ['rtk_x'], quantities)

    assert len(fig.data) == 1
    assert list(fig.data[0].y) == pytest.approx([4.0, 5.0, 6.0])


def test_rtk_curve_stays_continuous_and_radar_gap_is_marked():
    """雷达缺帧时真值仍连续，雷达曲线插入断点和中断标记。"""
    aligned = pd.DataFrame({
        'timestamp_parsed': [0.0, 0.05, 0.50],
        'radar_ts_parsed': [0.0, 0.05, 0.50],
        'rtk_nearest_ts_parsed': [0.0, 0.05, 0.50],
        'time_diff_ms': [0.0, 0.0, 0.0],
        'radar_Dx': [0.0, 0.5, 5.0],
        'rtk_center_x': [0.0, 0.5, 5.0],
    })
    rtk_curve = pd.DataFrame({
        'timestamp_parsed': [-1.0, 0.0, 0.05, 0.10, 0.20, 0.50, 1.0],
        'center_x': [-10.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    })
    quantities = {
        'cmp_dx': {
            'label': 'Dx', 'unit': 'm', 'radar_col': 'radar_Dx',
            'rtk_col': 'rtk_center_x', 'chart_type': 'overlay',
        },
    }

    fig = build_comparison_subplots(aligned, ['cmp_dx'], quantities, rtk_curve_df=rtk_curve)

    radar_trace = fig.data[0]
    gap_trace = fig.data[1]
    rtk_trace = fig.data[2]
    assert any(value is None for value in radar_trace.y)
    assert gap_trace.mode == 'markers'
    assert '雷达数据中断开始' in gap_trace.text[0]
    assert len(rtk_trace.x) == len(rtk_curve)
    assert all(value is not None for value in rtk_trace.y)
    # 悬浮框时间与 _fmt_ts 同口径（本地时区），不能硬编码 UTC 的 00:00。
    zero_clock = datetime.fromtimestamp(0.0).strftime('%H:%M:%S') + '.000'
    assert str(radar_trace.text[0]).startswith(f'Dx      : 0.000 m  {zero_clock}<br>')
    assert f'center_x: 0.000 m  {zero_clock}' in radar_trace.text[0]
    assert '绝对误差 0.000 m' in radar_trace.text[0]
    assert '差值' not in radar_trace.text[0]
    assert radar_trace.hoverlabel.bordercolor == '#000000'
    assert radar_trace.hoverlabel.bgcolor == '#ffffff'
    assert rtk_trace.hoverinfo == 'skip'
    # x 轴为 Plotly 日期轴（type='date'），数值须为 epoch 毫秒而非秒。
    assert min(rtk_trace.x) == -1000.0
    assert max(rtk_trace.x) == 1000.0


def test_display_downsampling_does_not_create_false_radar_gaps():
    """降采样跳过的正常帧不能被画成雷达中断。"""
    n = 10000  # 超过 DISPLAY_MAX_POINTS=5000，必然进入显示降采样
    timestamps = np.arange(n, dtype=float) * 0.05
    values = np.sin(timestamps)
    aligned = pd.DataFrame({
        'timestamp_parsed': timestamps,
        'radar_ts_parsed': timestamps,
        'rtk_nearest_ts_parsed': timestamps,
        'radar_Dx': values,
        'rtk_center_x': values,
    })
    quantities = {
        'cmp_dx': {
            'label': 'Dx', 'unit': 'm', 'radar_col': 'radar_Dx',
            'rtk_col': 'rtk_center_x', 'chart_type': 'overlay',
        },
    }

    fig = build_comparison_subplots(aligned, ['cmp_dx'], quantities)

    # 雷达 + 真值，不能因显示抽样间隔而增加“中断”红叉 trace。
    assert len(fig.data) == 2
