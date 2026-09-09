"""框选时间范围与高亮区域的回归测试。"""
import pandas as pd
import pytest

from radar_wave_analyzer.components.graph_builder import (
    BOX_JUMP_COLOR,
    build_box_jump_shapes,
    build_highlight_shapes,
)
from radar_wave_analyzer.core.selection import extract_x_selection


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


def test_box_jump_shape_marks_max_diff_position():
    """框选跳变高亮：紫色加粗线段落在差分绝对值最大的帧对上。"""
    base = pd.Timestamp('2026-04-20 10:00:00.000')
    df = pd.DataFrame({
        # 50ms 采样间隔（默认标称周期），差分全部有效
        'timestamp_parsed': [
            base + pd.Timedelta(milliseconds=50 * i) for i in range(4)],
        'Dx': [0.0, 0.1, 5.0, 5.1],   # 最大跳变在帧 2（0.1 → 5.0）
    })

    shapes = build_box_jump_shapes(df, ['Dx'])

    assert len(shapes) == 1
    shape = shapes[0]
    assert shape['line']['color'] == BOX_JUMP_COLOR
    assert shape['line']['width'] == 5
    assert shape['xref'] == 'x' and shape['yref'] == 'y'
    assert shape['x0'] == '2026-04-20T10:00:00.050000'
    assert shape['x1'] == '2026-04-20T10:00:00.100000'
    assert shape['y0'] == pytest.approx(0.1)
    assert shape['y1'] == pytest.approx(5.0)


def test_box_jump_shape_skips_when_no_jump():
    """无有效差分、物理量缺失或帧数不足时不生成跳变 shape。"""
    base = pd.Timestamp('2026-04-20 10:00:00.000')
    # 帧间隔 1 秒 > 2×标称周期（50ms）→ 差分全部无效 → 无 shape
    gappy = pd.DataFrame({
        'timestamp_parsed': [base, base + pd.Timedelta(seconds=1)],
        'Dx': [1.0, 2.0],
    })
    assert build_box_jump_shapes(gappy, ['Dx']) == []
    # 物理量不存在
    normal = pd.DataFrame({
        'timestamp_parsed': [base, base + pd.Timedelta(milliseconds=50)],
        'Dx': [1.0, 2.0],
    })
    assert build_box_jump_shapes(normal, ['Missing']) == []
    # 帧数不足
    assert build_box_jump_shapes(normal.head(1), ['Dx']) == []
