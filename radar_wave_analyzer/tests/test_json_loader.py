"""json_loader 与 JSON 数据源适配的单元测试。"""
import json
from datetime import datetime

import pandas as pd
import pytest

from radar_wave_analyzer.comparison.parser import load_data_file  # noqa: E402
from radar_wave_analyzer.comparison.service import prepare_comparison_upload  # noqa: E402
from radar_wave_analyzer.core.data_loader import identify_radar_source, parse_epoch_series  # noqa: E402
from radar_wave_analyzer.core.json_loader import (  # noqa: E402
    build_column_normalizer,
    is_json_filename,
    load_json_from_bytes,
)

# 与实车数据同量级的纳秒 epoch（2026-08，间隔约 50ms）
_NS_BASE = 1_787_018_922_641_932_400
_NS_STEP = 50_000_000


def _radar_json_payload() -> dict:
    """构造与 data.json 同构的测试数据：FLR 空 + RLR 两个目标。"""
    return {
        '/C798/High/FLR_OBJ_object': None,
        '/C798/High/RLR_OBJ_object': {
            '15': {
                'timestamp': [_NS_BASE, _NS_BASE + _NS_STEP, _NS_BASE + 2 * _NS_STEP],
                'ID': [15.0, 15.0, 15.0],
                'Dx': [-4.0625, -4.125, -4.25],
                'Dy': [-7.875, -7.75, -7.5],
                'Vx': [-0.125, -0.125, -0.5],
                'Vy': [1.0625, 1.0625, 1.0],
                'Track_Age': [10.0, 10.0, 11.0],
                'Rx_Front': [-0.0875, -0.0875, -0.025],
                'Rx_Rear': [2.7125, 2.7125, 2.775],
            },
            '23': {
                'timestamp': [_NS_BASE + 10_000_000],
                'Dx': [1.0],
                'Dy': [2.0],
                'Vx': [0.0],
                'Vy': [0.0],
                'Rx_Front': [0.5],
                'Rx_Rear': [1.5],
                'Track_Age': [3.0],
            },
        },
    }


def _dump(payload: dict) -> bytes:
    return json.dumps(payload).encode('utf-8')


class TestIsJsonFilename:
    def test_extension_detection(self):
        assert is_json_filename('data.json') is True
        assert is_json_filename('DATA.JSON') is True
        assert is_json_filename('track.csv') is False
        assert is_json_filename(None) is False
        assert is_json_filename('') is False


class TestParseEpochSeries:
    """数值 epoch 单位自动判定。"""

    def test_nanoseconds(self):
        result = parse_epoch_series(pd.Series([_NS_BASE, _NS_BASE + _NS_STEP]))
        assert result.iloc[0] == pd.Timestamp(_NS_BASE, unit='ns')

    def test_microseconds(self):
        # 整除换算到 us 后 ns 尾数自然丢失，期望值以 us 源值为准
        result = parse_epoch_series(pd.Series([_NS_BASE // 1000]))
        assert result.iloc[0] == pd.Timestamp(_NS_BASE // 1000, unit='us')

    def test_milliseconds(self):
        result = parse_epoch_series(pd.Series([_NS_BASE // 1_000_000]))
        assert result.iloc[0] == pd.Timestamp(_NS_BASE // 1_000_000, unit='ms')

    def test_seconds(self):
        result = parse_epoch_series(pd.Series([1_787_018_922.6419324]))
        assert result.iloc[0] == pd.Timestamp(1_787_018_922.6419324, unit='s')

    def test_invalid_values_become_nat(self):
        result = parse_epoch_series(pd.Series(['abc', None]))
        assert result.isna().all()


class TestColumnNormalizer:
    def test_rx_columns_normalized_to_config_names(self):
        normalizer = build_column_normalizer()
        rename = {'Rx_Front': normalizer.get('rx_front'),
                  'Rx_Rear': normalizer.get('rx_rear')}
        assert rename == {'Rx_Front': 'Rx_front', 'Rx_Rear': 'Rx_rear'}

    def test_standard_names_unchanged(self):
        normalizer = build_column_normalizer()
        assert normalizer.get('dx') == 'Dx'
        assert normalizer.get('headingangle') == 'HeadingAngle'
        assert 'type' not in normalizer


class TestLoadJsonFromBytes:
    def test_flatten_multiple_objects(self):
        result = load_json_from_bytes(_dump(_radar_json_payload()), 'data.json')

        assert len(result['topics']) == 1
        topic = result['topics'][0]
        df = topic['df']

        # 两目标合并：3 + 1 = 4 行，且包含全部必需列
        assert len(df) == 4
        assert {'timestamp', 'ID', 'Track_Age', 'timestamp_parsed'}.issubset(df.columns)
        assert set(df['ID'].unique()) == {15, 23}
        # ID 回填外层键并数值化为整数
        assert df['ID'].dtype == 'int64'
        # 纳秒时间戳解析正确且按时间排序
        assert df['timestamp_parsed'].iloc[0] == pd.Timestamp(_NS_BASE, unit='ns')
        assert df['timestamp_parsed'].is_monotonic_increasing
        # Track_Age 完成 0~255 整数校验转换
        assert df['Track_Age'].dtype == 'int64'

    def test_rx_columns_renamed(self):
        result = load_json_from_bytes(_dump(_radar_json_payload()), 'data.json')
        df = result['topics'][0]['df']

        assert 'Rx_front' in df.columns and 'Rx_rear' in df.columns
        assert 'Rx_Front' not in df.columns and 'Rx_Rear' not in df.columns
        assert df['Rx_front'].iloc[0] == pytest.approx(-0.0875)

    def test_empty_topic_skipped_with_warning(self):
        result = load_json_from_bytes(_dump(_radar_json_payload()), 'data.json')

        assert len(result['topics']) == 1
        assert any('FLR' in warning for warning in result['warnings'])
        assert result['errors'] == []

    def test_id_from_object_key_when_missing(self):
        payload = {'/x/RLR_OBJ_object': {'7': {
            'timestamp': [_NS_BASE], 'Dx': [1.0], 'Dy': [1.0], 'Track_Age': [1],
        }}}
        result = load_json_from_bytes(_dump(payload), 'data.json')

        assert result['topics'][0]['df']['ID'].tolist() == [7]

    def test_object_missing_required_column_rows_dropped(self):
        """同话题内个别对象缺 Track_Age：不报错，该对象行在清洗时剔除。"""
        payload = {
            '/x/RLR_OBJ_object': {
                '15': {
                    'timestamp': [_NS_BASE], 'ID': [15],
                    'Dx': [1.0], 'Dy': [1.0], 'Track_Age': [5],
                },
                '16': {'timestamp': [_NS_BASE + _NS_STEP], 'ID': [16], 'Dx': [1.0], 'Dy': [1.0]},
            },
        }
        result = load_json_from_bytes(_dump(payload), 'data.json')

        assert len(result['topics']) == 1
        assert result['errors'] == []
        assert result['topics'][0]['df']['ID'].tolist() == [15]

    def test_topic_missing_required_column_reports_error(self):
        """整个话题无任何对象提供 Track_Age 时按缺失必需列报错。"""
        payload = {'/x/RLR_OBJ_object': {'16': {
            'timestamp': [_NS_BASE], 'ID': [16], 'Dx': [1.0], 'Dy': [1.0],
        }}}
        result = load_json_from_bytes(_dump(payload), 'data.json')

        assert result['topics'] == []
        assert len(result['errors']) == 1
        assert 'Track_Age' in result['errors'][0]

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match='JSON'):
            load_json_from_bytes(b'not a json', 'data.json')

    def test_non_dict_root_raises_value_error(self):
        with pytest.raises(ValueError, match='根节点'):
            load_json_from_bytes(b'[1, 2, 3]', 'data.json')

    def test_topic_label_contains_filename_and_topic(self):
        result = load_json_from_bytes(_dump(_radar_json_payload()), 'data.json')

        assert result['topics'][0]['label'] == 'data.json·RLR_OBJ_object'
        assert result['topics'][0]['topic'] == '/C798/High/RLR_OBJ_object'


class TestRadarSourceFromTopic:
    """话题名可复用既有文件名规则识别雷达来源。"""

    def test_rlr_topic_recognized(self):
        source = identify_radar_source('/C798/High/RLR_OBJ_object')
        assert source['key'] == 'rlr' and source['recognized'] is True

    def test_flr_topic_recognized(self):
        source = identify_radar_source('/C798/High/FLR_OBJ_object')
        assert source['key'] == 'flr' and source['recognized'] is True


class TestComparisonJson:
    """对比模块的 JSON 适配。"""

    def test_load_data_file_dispatches_json(self):
        results = load_data_file(_dump(_radar_json_payload()), 'data.json')

        assert len(results) == 1
        info = results[0]
        assert info['role'] == 'radar'
        assert info['errors'] == []
        assert 'Rx_front' in info['df'].columns
        # 对比模块时间口径为 epoch 秒
        assert info['df']['timestamp_parsed'].iloc[0] == pytest.approx(_NS_BASE / 1e9)
        assert info['filename'] == 'data.json·RLR_OBJ_object'

    def test_load_data_file_keeps_csv_single_entry(self):
        csv_bytes = (
            'timestamp,ID,Track_Age,Dx,Dy,Vx,Vy\n'
            '2026_04_20_11_12_27_690,1,10,1.0,2.0,0.0,0.0\n'
        ).encode('utf-8')
        results = load_data_file(csv_bytes, 'track.csv')

        assert len(results) == 1
        assert results[0]['role'] == 'radar'
        # CSV 墙钟按系统时区折算为真实 UTC epoch，与 JSON 纳秒 epoch 同口径
        expected = pd.Timestamp('2026-04-20 11:12:27.690').tz_localize(
            datetime.now().astimezone().tzinfo,
        ).timestamp()
        assert results[0]['df']['timestamp_parsed'].iloc[0] == pytest.approx(expected)

    def test_prepare_comparison_upload_accepts_json_radar(self):
        upload = prepare_comparison_upload(
            [(_dump(_radar_json_payload()), 'data.json')], 'radar',
        )

        assert upload['errors'] == []
        assert upload['file_count'] == 1
        assert upload['info']['df']['ID'].nunique() == 2

    def test_prepare_comparison_upload_routes_json_into_rtk_zone(self):
        """雷达 JSON 拖入 RTK 区域时自动归类到雷达区域（自包含文件支持）。"""
        upload = prepare_comparison_upload(
            [(_dump(_radar_json_payload()), 'data.json')], 'rtk',
        )

        assert upload['file_count'] == 0
        assert upload['errors'] == []
        routed = upload['routed']
        assert routed['role'] == 'radar'
        assert routed['file_count'] == 1
        assert routed['info']['df']['ID'].nunique() == 2

    def test_rtk_role_json_detected(self):
        payload = {'/truth/rtk': {'2': {
            'timestamp': [_NS_BASE], 'center_x': [1.0], 'center_y': [2.0],
            'Vx': [0.1], 'Vy': [0.1],
        }}}
        results = load_data_file(_dump(payload), 'truth.json')

        assert len(results) == 1
        assert results[0]['role'] == 'rtk'
        assert results[0]['errors'] == []
