"""
data_loader 模块单元测试。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from core.data_loader import identify_radar_source, parse_timestamp, parse_timestamp_series, load_csv


def test_identify_front_and_rear_radar_from_filename():
    front = identify_radar_source('CD701_flr_track_2026_07_28_10_16_53.csv')
    rear = identify_radar_source('CD701_rlr_track_2026_07_28_10_16_53.csv')
    unknown = identify_radar_source('CD701_track_2026_07_28.csv')

    assert front['key'] == 'flr' and front['recognized'] is True
    assert rear['key'] == 'rlr' and rear['recognized'] is True
    assert unknown['key'] == 'unknown' and unknown['recognized'] is False


class TestParseTimestamp:
    """时间戳解析测试"""

    def test_csv_format(self):
        """CSV 格式时间戳解析正确: YYYY_MM_DD_HH_MM_SS_mmm（保留毫秒）"""
        result = parse_timestamp('2026_04_20_11_12_27_690')
        # 必须保留毫秒精度，否则多帧时间戳相同导致 plotly 画成垂直线
        assert result == pd.Timestamp('2026-04-20 11:12:27.690')
        assert result.microsecond == 690_000

    def test_user_input_format(self):
        """用户输入格式时间戳解析正确: YYYY-MM-DD HH:MM:SS"""
        result = parse_timestamp('2026-07-03 11:56:16')
        assert result == pd.Timestamp('2026-07-03 11:56:16')

    def test_user_input_with_ms(self):
        """含毫秒的用户输入格式"""
        result = parse_timestamp('2026-07-03 11:56:16.500')
        assert result == pd.Timestamp('2026-07-03 11:56:16.500')

    def test_invalid_format(self):
        """无法解析的格式应抛出 ValueError"""
        with pytest.raises(ValueError):
            parse_timestamp('not_a_timestamp')

    def test_series_parser_keeps_csv_ms_and_user_input_formats(self):
        """批量解析应保留设备毫秒精度，并兼容用户输入格式。"""
        result = parse_timestamp_series(pd.Series([
            '2026_04_20_11_12_27_690',
            '2026_04_20_11_12_28',
            '2026-04-20 11:12:29.500',
        ]))

        assert result.tolist() == [
            pd.Timestamp('2026-04-20 11:12:27.690'),
            pd.Timestamp('2026-04-20 11:12:28'),
            pd.Timestamp('2026-04-20 11:12:29.500'),
        ]


class TestLoadCSV:
    """CSV 加载测试"""

    def test_missing_required_columns(self, tmp_path):
        """缺少必要字段应抛出 ValueError"""
        csv_path = tmp_path / 'test.csv'
        pd.DataFrame({'ID': [1], 'Track_Age': [10]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match='缺少必要字段'):
            load_csv(str(csv_path))

    def test_file_not_found(self):
        """文件不存在应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_csv('/nonexistent/path.csv')

    def test_chunked_loading_matches_normal_loading(self, tmp_path):
        """分块读取必须保持与普通读取一致的全局时间排序与清洗结果。"""
        csv_path = tmp_path / 'chunked.csv'
        pd.DataFrame({
            'timestamp': [
                '2026_04_20_11_12_27_100',
                '2026_04_20_11_12_27_000',
                '2026_04_20_11_12_27_050',
            ],
            'ID': [1, 1, 1],
            'Track_Age': [3, 1, 2],
        }).to_csv(csv_path, index=False)

        normal = load_csv(str(csv_path))
        chunked = load_csv(str(csv_path), chunk_size=1)

        pd.testing.assert_frame_equal(normal, chunked)

    def test_fractional_track_age_is_rejected_not_truncated(self, tmp_path):
        """Track_Age=12.5 必须被剔除，不能静默转换为整数 12。"""
        csv_path = tmp_path / 'fractional_age.csv'
        pd.DataFrame({
            'timestamp': ['2026_04_20_11_12_27_000', '2026_04_20_11_12_27_050'],
            'ID': [1, 1],
            'Track_Age': [12.5, 13],
        }).to_csv(csv_path, index=False)

        result = load_csv(str(csv_path))

        assert result['Track_Age'].tolist() == [13]
        assert str(result['Track_Age'].dtype).startswith('int')
