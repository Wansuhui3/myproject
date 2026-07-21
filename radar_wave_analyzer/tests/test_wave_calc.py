"""
wave_calc 模块单元测试。
"""
import sys
import os
# 将 radar_wave_analyzer 目录加入 sys.path，使得 core 和 config 可以被直接导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from core.wave_calc import calc_frame_diff, calc_wave_stats, compute_segment_stats


def _make_ts_series(timestamps: list[str]) -> pd.DatetimeIndex:
    """构建测试用时间戳序列。"""
    return pd.DatetimeIndex(pd.to_datetime(timestamps))


class TestCalcFrameDiff:
    """帧间差分计算测试"""

    def test_first_frame_no_wave(self):
        """首帧无波动值"""
        values = np.array([10.0, 10.5, 10.3])
        ts = _make_ts_series([
            '2026-04-20 11:12:27.000',
            '2026-04-20 11:12:27.050',
            '2026-04-20 11:12:27.100',
        ])
        diff = calc_frame_diff(values, ts, sampling_period_ms=50)
        assert np.isnan(diff[0]), '首帧应为 NaN'
        assert not np.isnan(diff[1])
        assert not np.isnan(diff[2])

    def test_normal_diff(self):
        """正常帧间差分"""
        values = np.array([10.0, 10.5, 10.3, 9.8])
        ts = _make_ts_series([
            '2026-04-20 11:12:27.000',
            '2026-04-20 11:12:27.050',
            '2026-04-20 11:12:27.100',
            '2026-04-20 11:12:27.150',
        ])
        diff = calc_frame_diff(values, ts, sampling_period_ms=50)
        assert diff[1] == pytest.approx(0.5)
        assert diff[2] == pytest.approx(-0.2)
        assert diff[3] == pytest.approx(-0.5)

    def test_skip_large_gap(self):
        """段内间隔 > 2x 采样周期处跳过差分"""
        values = np.array([10.0, 10.5, 20.0, 20.1])
        ts = _make_ts_series([
            '2026-04-20 11:12:27.000',
            '2026-04-20 11:12:27.050',
            '2026-04-20 11:12:27.200',  # 150ms 间隔 > 2*50ms
            '2026-04-20 11:12:27.250',
        ])
        diff = calc_frame_diff(values, ts, sampling_period_ms=50)
        assert diff[1] == pytest.approx(0.5)
        assert np.isnan(diff[2]), '帧间隔 > 100ms 应跳过差分'
        assert not np.isnan(diff[3])


class TestCalcWaveStats:
    """波动统计指标测试"""

    def test_empty(self):
        """空数据返回 NaN"""
        stats = calc_wave_stats(np.array([]))
        assert stats['valid_count'] == 0
        assert np.isnan(stats['mean_abs'])

    def test_single_value(self):
        """单个有效值"""
        stats = calc_wave_stats(np.array([np.nan, 1.5, np.nan]))
        assert stats['valid_count'] == 1
        assert stats['mean_abs'] == 1.5
        assert np.isnan(stats['std_dev'])  # 单样本 ddof=1 时 std 为 NaN

    def test_peak_to_peak_definition(self):
        """峰峰值 = max + abs(min)，不是 max - min"""
        diff = np.array([-3.0, -1.0, 2.0, 5.0, -2.0])
        stats = calc_wave_stats(diff)
        assert stats['peak_to_peak'] == pytest.approx(5.0 + 3.0)  # max=5, abs(min)=3
        assert stats['max_positive'] == 5.0
        assert stats['max_negative'] == -3.0

    def test_rms(self):
        """RMS = sqrt(mean(diff²))"""
        diff = np.array([3.0, 4.0])
        stats = calc_wave_stats(diff)
        expected_rms = np.sqrt((9.0 + 16.0) / 2.0)
        assert stats['rms'] == pytest.approx(expected_rms)

    def test_std_ddof1(self):
        """标准差使用 ddof=1（样本标准差）"""
        diff = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = calc_wave_stats(diff)
        expected_std = np.std(diff, ddof=1)
        assert stats['std_dev'] == pytest.approx(expected_std)


class TestComputeSegmentStats:
    """段级统计测试"""

    def test_with_mask(self):
        """框选区间统计"""
        df = pd.DataFrame({
            'timestamp': [
                '2026-04-20 11:12:27.000',
                '2026-04-20 11:12:27.050',
                '2026-04-20 11:12:27.100',
                '2026-04-20 11:12:27.150',
                '2026-04-20 11:12:27.200',
            ],
            'Dx': [10.0, 10.5, 10.3, 9.8, 10.0],
            'timestamp_parsed': pd.to_datetime([
                '2026-04-20 11:12:27.000',
                '2026-04-20 11:12:27.050',
                '2026-04-20 11:12:27.100',
                '2026-04-20 11:12:27.150',
                '2026-04-20 11:12:27.200',
            ]),
        })
        mask = pd.Series([False, True, True, True, False])
        stats = compute_segment_stats(df, 'Dx', mask=mask, sampling_period_ms=50)
        assert stats['valid_count'] == 2  # mask 内3帧，首帧无波动，剩余2个diff

    def test_insufficient_data(self):
        """数据不足时返回 NaN"""
        df = pd.DataFrame({
            'timestamp': ['2026-04-20 11:12:27.000'],
            'Dx': [10.0],
            'timestamp_parsed': pd.to_datetime(['2026-04-20 11:12:27.000']),
        })
        stats = compute_segment_stats(df, 'Dx', sampling_period_ms=50)
        assert stats['valid_count'] == 0
