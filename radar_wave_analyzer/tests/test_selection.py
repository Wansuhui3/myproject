"""框选时间范围与高亮区域的回归测试。"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.graph_builder import build_highlight_shapes  # noqa: E402
from core.selection import extract_x_selection  # noqa: E402


def test_range_is_preferred_over_discrete_selected_points():
    """高亮必须使用鼠标矩形边界，而不是首尾命中采样点。"""
    selected_data = {
        'range': {'x2': ['2026-04-20T10:00:00.250', '2026-04-20T10:00:00.750']},
        'points': [
            {'x': '2026-04-20T10:00:00.000'},
            {'x': '2026-04-20T10:00:01.000'},
        ],
    }

    start, end = extract_x_selection(selected_data)

    assert start == pd.Timestamp('2026-04-20 10:00:00.250')
    assert end == pd.Timestamp('2026-04-20 10:00:00.750')


def test_highlight_shape_preserves_exact_mouse_time_bounds():
    """shape 的 x0/x1 应等于鼠标框选边界，不可吸附到最近数据点。"""
    df = pd.DataFrame({
        'timestamp_parsed': pd.to_datetime([
            '2026-04-20 10:00:00.000',
            '2026-04-20 10:00:01.000',
        ]),
        'Dx': [0.0, 1.0],
    })

    shapes = build_highlight_shapes(
        df, ['Dx'], highlight_range=(0, 1),
        highlight_time_range=(
            pd.Timestamp('2026-04-20 10:00:00.250'),
            pd.Timestamp('2026-04-20 10:00:00.750'),
        ),
    )

    assert shapes[0]['x0'] == '2026-04-20T10:00:00.250000'
    assert shapes[0]['x1'] == '2026-04-20T10:00:00.750000'
