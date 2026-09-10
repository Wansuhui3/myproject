"""波动分析纯数据辅助：物理量发现、有效性过滤与统计计算。

被 wave_callbacks（交互域）与 wave_upload_callbacks（数据加载域）共享；
不得导入任何回调模块。
"""
import numpy as np
import pandas as pd

from ..components.graph_builder import BOX_JUMP_COLOR
from ..core.wave_calc import compute_fluctuation_stats, compute_segment_stats

_QUANTITY_METADATA_COLUMNS = frozenset({
    'timestamp', 'timestamp_parsed', 'ID', 'Track_Age', 'file_index',
    'csv_row', 'radar_frame', 'rtk_frame', 'time_diff_ms',
    'radar_source_key', 'radar_source_label', 'source_filename',
    'MotionStatus', 'MeasurementState', 'ExistProb',
})


def _discover_quantity_columns(df: pd.DataFrame) -> list[str]:
    """从当前 CSV 实际字段发现可绘制物理量。

    配置文件只提供单位/显示名；是否出现在波动页由当前数据源中是否存在
    且含有限数值决定。所有内部列、时间列、ID 和状态/来源字段均排除。
    """
    if df is None or df.empty:
        return []

    columns: list[str] = []
    for raw_column in df.columns:
        column = str(raw_column)
        if (column in _QUANTITY_METADATA_COLUMNS
                or column.startswith('__')
                or column.startswith('wave_')
                or column.startswith('radar_source_')):
            continue
        values = pd.to_numeric(df[raw_column], errors='coerce').to_numpy(dtype=float)
        if np.isfinite(values).any():
            columns.append(column)
    return columns


def _ensure_valid_quantities(seg_df, selected_qties) -> list:
    """返回当前轨迹真正可绘制的已选字段，绝不静默改画 Dx。"""
    available = set(_discover_quantity_columns(seg_df))
    requested = [str(q) for q in (selected_qties or [])]
    return [qty for qty in requested if qty in available]


def _unavailable_quantities(seg_df, selected_qties) -> list[str]:
    """返回用户已选但当前轨迹不存在或全为空的字段。"""
    available = set(_discover_quantity_columns(seg_df))
    return [str(q) for q in (selected_qties or []) if str(q) not in available]


def _get_non_highlight_shapes(figure: dict | None) -> list:
    """保留图表自身线形（图例与全段最大跳变红线），移除上一次框选产生的
    矩形与框选区间最大跳变紫色高亮线。"""
    if not isinstance(figure, dict):
        return []
    shapes = figure.get('layout', {}).get('shapes', [])
    return [
        shape for shape in shapes
        if isinstance(shape, dict)
        and shape.get('type') != 'rect'
        and (shape.get('line') or {}).get('color') != BOX_JUMP_COLOR
    ]


def _compute_quantities_stats(seg_df, selected_qties, mask=None, diff_cache=None):
    """为所有勾选物理量计算 segment_stats。

    Returns:
        (stats_per_qty, first_frame_dists)
    """
    stats_per_qty = {}
    for qty in selected_qties:
        if qty in seg_df.columns:
            stats_per_qty[qty] = compute_segment_stats(seg_df, qty, mask=mask, diff_cache=diff_cache)
        else:
            stats_per_qty[qty] = None

    fluct = compute_fluctuation_stats(seg_df, mask=mask, diff_cache=diff_cache)
    first_frame_dists = fluct.get('first_frame_dists') if fluct else None

    return stats_per_qty, first_frame_dists
