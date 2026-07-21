"""图表框选范围的纯数据处理辅助函数。"""
from typing import Optional

import pandas as pd


def extract_x_selection(selected_data: object) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    """从 Plotly selectedData 提取鼠标框选的精确 X 轴范围。

    ``range`` 是用户矩形在坐标系中的真实边界，必须优先于 ``points``；后者
    只包含落在曲线上的离散采样点，会使高亮区域收缩到首尾采样点而偏离鼠标框。
    """
    if not isinstance(selected_data, dict):
        return None

    range_data = selected_data.get('range')
    if isinstance(range_data, dict):
        for key in sorted(range_data):
            value = range_data[key]
            if key.startswith('x') and isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    start, end = pd.Timestamp(value[0]), pd.Timestamp(value[1])
                except (TypeError, ValueError):
                    continue
                return (start, end) if start <= end else (end, start)

    points = selected_data.get('points')
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        x_values = [pd.Timestamp(point['x']) for point in points if 'x' in point]
    except (TypeError, ValueError):
        return None
    if len(x_values) < 2:
        return None
    return min(x_values), max(x_values)
