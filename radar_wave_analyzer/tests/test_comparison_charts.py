"""真值对比图的时间提示和雷达中断渲染测试。"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.comparison_charts import _fmt_ts, build_comparison_subplots  # noqa: E402


def test_tooltip_epoch_is_formatted_without_local_timezone_shift():
    """原始 CSV 时间基准应保持 1970-01-01 00:00:00，不受本机时区影响。"""
    assert _fmt_ts(0.0) == '1970-01-01 00:00:00.000'


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
        'timestamp_parsed': [0.0, 0.05, 0.10, 0.20, 0.50],
        'center_x': [0.0, 0.5, 1.0, 2.0, 5.0],
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
    assert str(radar_trace.text[0]).startswith('Dx      : 0.000 m  00:00:00.000<br>')
    assert 'center_x: 0.000 m  00:00:00.000' in radar_trace.text[0]
    assert '绝对误差 0.000 m' in radar_trace.text[0]
    assert '差值' not in radar_trace.text[0]
    assert radar_trace.hoverlabel.bordercolor == '#000000'
    assert radar_trace.hoverlabel.bgcolor == '#ffffff'
    assert rtk_trace.hoverinfo == 'skip'
