"""波动页物理量必须由当前 CSV 字段驱动的回归测试。"""
import numpy as np
import pandas as pd

from radar_wave_analyzer.callbacks import (
    _cmp_mapping_options,
    _cmp_resolve_mappings,
    _discover_quantity_columns,
    _ensure_valid_quantities,
)
from radar_wave_analyzer.components.graph_builder import build_multi_subplot_graph


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        'timestamp': ['t0', 't1', 't2'],
        'timestamp_parsed': pd.date_range('2026-01-01', periods=3, freq='50ms'),
        'ID': [7, 7, 7],
        'Track_Age': [1, 2, 3],
        'MotionStatus': [1, 1, 1],
        'Dx': [1.0, 1.1, 1.2],
        'Rx_front': [10.0, 10.5, 11.0],
        'Rx_rear': [np.nan, np.nan, np.nan],
        'CustomSignal': [3.0, 4.0, 5.0],
        '__source_row_index__': [0, 1, 2],
    })


def test_quantity_discovery_uses_csv_signals_not_configured_whitelist():
    fields = _discover_quantity_columns(_dataframe())

    assert fields == ['Dx', 'Rx_front', 'CustomSignal']
    assert 'Track_Age' not in fields
    assert 'MotionStatus' not in fields
    assert 'Rx_rear' not in fields


def test_invalid_requested_signal_is_not_silently_replaced_by_dx():
    df = _dataframe()

    assert _ensure_valid_quantities(df, ['Rx_front']) == ['Rx_front']
    assert _ensure_valid_quantities(df, ['Rx_rear']) == []
    assert _ensure_valid_quantities(df, ['MissingField']) == []


def test_graph_reports_empty_selection_instead_of_creating_dx_subplot():
    fig = build_multi_subplot_graph(_dataframe(), [])

    assert len(fig.data) == 0
    assert fig.layout.annotations[0].text == '当前轨迹中没有可绘制的已选物理量'


def test_comparison_mapping_fields_can_be_left_unselected():
    """映射任一侧留空时生成单线通道；两侧均无有效字段才被忽略。"""
    fields = {'Dx': {'name': 'Dx', 'unit': 'm'}}
    assert _cmp_mapping_options(fields)[0] == {'label': '（不选择）', 'value': None}

    state = {
        'radar_meta': {'physical_fields': [{'name': 'Dx', 'unit': 'm'}]},
        'rtk_meta': {'physical_fields': [{'name': 'center_x', 'unit': 'm'}]},
    }
    mappings = {'items': [
        {'uid': 'radar-empty', 'radar_col': None, 'rtk_col': 'center_x'},
        {'uid': 'rtk-empty', 'radar_col': 'Dx', 'rtk_col': None},
        {'uid': 'both-empty', 'radar_col': None, 'rtk_col': None},
        {'uid': 'field-missing', 'radar_col': 'Gone', 'rtk_col': 'center_x'},
    ]}

    resolved = _cmp_resolve_mappings(mappings, state)
    # 单侧留空 / 字段已不存在的通道保留为单线显示（unit_status=single，
    # 不参与误差统计）；仅双侧均无有效字段时丢弃。
    assert [m['uid'] for m in resolved] == ['radar-empty', 'rtk-empty', 'field-missing']
    assert all(m['unit_status'] == 'single' for m in resolved)
    assert all(m['stats_enabled'] is False for m in resolved)
