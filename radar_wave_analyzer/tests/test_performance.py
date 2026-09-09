"""
性能指标评估模块单元测试。

覆盖设计文档 14.1 列出的 12 项测试要求：
 1. 六个距离边界归属（10、20、50、70、100、150m）
 2. 固定/百分比/Max/倍率Max 四种阈值模式
 3. 正负速度、正负加速度及真值为零
 4. 误差与阈值相等时按失败处理
 5. 准确率与 95.45% 相等时按失败处理
 6. 三帧全部达到限值时 R3 >= 1
 7. 三帧中任意一帧低于限值时不构成违规
 8. 动态限值三帧的归一化计算
 9. 数据中断、未匹配帧打断连续计数；跨距离段不打断
10. 无样本、1帧、2帧时的不可判定/样本不足状态
11. RMSE 使用有符号误差平方且不参与结论
12. 不同物理量的无效值不相互污染样本数
"""
import numpy as np
import pandas as pd
import pytest

from radar_wave_analyzer.comparison.performance import (
    STATUS_FAIL,
    STATUS_INSUFFICIENT,
    STATUS_PASS,
    STATUS_UNDECIDABLE,
    _assign_bins,
    _compute_continuity_breaks,
    _resolve_limit,
    _rolling_window_min,
    evaluate_all_metrics,
    evaluate_metric,
)

BINS = [0, 10, 20, 50, 70, 100, 150]


def _make_aligned(rows):
    """构造 aligned_df：字典列表 → DataFrame，补齐必需列。"""
    df = pd.DataFrame(rows)
    n = len(df)
    if 'radar_frame' not in df.columns:
        df['radar_frame'] = np.arange(1, n + 1)
    if 'timestamp_parsed' not in df.columns:
        # 默认 20Hz 等间隔，保证连续
        df['timestamp_parsed'] = np.arange(n, dtype=float) * 0.05
    if 'is_matched' not in df.columns:
        df['is_matched'] = True
    return df


def _simple_config(**overrides):
    """构造全局参数，默认与 config.yaml 一致。"""
    cfg = {
        'distance_bins': list(BINS),
        'include_last_upper': True,
        'valid_distance_range': [0, 150],
        'accuracy_requirement': 0.9545,
        'three_frame_requirement': 1.0,
        'min_samples_for_conclusion': 3,
        'gap_break_factor': 2.5,
        'gap_percentile': 25,
    }
    cfg.update(overrides)
    return cfg


def _fixed_limit_rules(limit):
    """生成六个距离段均使用同一固定阈值的 Dx 规则，便于精确构造边界用例。"""
    return {
        'label': 'Dx', 'unit': 'm', 'mode': 'binned',
        'radar_col': 'radar_Dx', 'truth_col': 'rtk_center_x',
        'percent_basis': 'distance',
        'bins': [
            {'range': [lo, hi],
             'normal_limit': {'mode': 'absolute', 'absolute': limit},
             'three_frame_limit': {'mode': 'absolute', 'absolute': limit}}
            for lo, hi in zip(BINS[:-1], BINS[1:])
        ],
    }


# Dx 规则：0-10m → T=max(0.3, 0.01d)，L=0.5（固定）
DX_RULES = {
    'label': 'Dx',
    'unit': 'm',
    'mode': 'binned',
    'radar_col': 'radar_Dx',
    'truth_col': 'rtk_center_x',
    'percent_basis': 'distance',
    'bins': [
        {'range': [0, 10],
         'normal_limit': {'mode': 'max_absolute_percent', 'absolute': 0.3, 'percent': 0.01},
         'three_frame_limit': {'mode': 'absolute', 'absolute': 0.5}},
        {'range': [10, 20],
         'normal_limit': {'mode': 'max_absolute_percent', 'absolute': 0.3, 'percent': 0.01},
         'three_frame_limit': {'mode': 'absolute', 'absolute': 0.5}},
        {'range': [20, 50],
         'normal_limit': {'mode': 'percent', 'percent': 0.02},
         'three_frame_limit': {'mode': 'absolute', 'absolute': 1.5}},
        {'range': [50, 70],
         'normal_limit': {'mode': 'percent', 'percent': 0.02},
         'three_frame_limit': {'mode': 'percent', 'percent': 0.05}},
        {'range': [70, 100],
         'normal_limit': {'mode': 'percent', 'percent': 0.02},
         'three_frame_limit': {'mode': 'percent', 'percent': 0.07}},
        {'range': [100, 150],
         'normal_limit': {'mode': 'percent', 'percent': 0.025},
         'three_frame_limit': {'mode': 'percent', 'percent': 0.10}},
    ],
}


# ============================================================
# 测试 1：距离边界归属
# ============================================================

class TestDistanceBinning:

    @pytest.mark.parametrize('distance,expected', [
        (0.0, 0), (9.99, 0),
        (10.0, 1), (19.99, 1),
        (20.0, 2), (49.99, 2),
        (50.0, 3), (69.99, 3),
        (70.0, 4), (99.99, 4),
        (100.0, 5), (149.99, 5),
        (150.0, 5),          # 最后一段包含 150m
    ])
    def test_bin_boundaries(self, distance, expected):
        """1. 六个距离边界的归属，特别是 10/20/50/70/100/150m。"""
        result = _assign_bins(np.array([distance]), BINS, include_last_upper=True)
        assert result[0] == expected, f'{distance}m 应归入第 {expected} 段'

    def test_beyond_range_is_excluded(self):
        """超出 [0,150] 的距离不属于任何区间。"""
        result = _assign_bins(np.array([-1.0, 150.1, np.nan]), BINS, True)
        assert list(result) == [-1, -1, -1]

    def test_uses_truth_distance_not_radar(self):
        """分桶必须使用 RTK 真值距离，绝不使用雷达 Dx。"""
        rows = [
            # 雷达 Dx=45（20-50段），真值距离=55（50-70段）→ 应归入第 3 段
            {'radar_Dx': 45.0, 'rtk_center_x': 55.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        # 第 3 段（[50,70)）应有 1 个样本，第 2 段（[20,50)）应为 0
        assert out['bins'][3]['sample_count'] == 1
        assert out['bins'][2]['sample_count'] == 0


# ============================================================
# 测试 2：四种阈值模式
# ============================================================

class TestLimitModes:

    def test_absolute(self):
        basis = np.array([0.0, 10.0, 100.0])
        out = _resolve_limit({'mode': 'absolute', 'absolute': 0.5}, basis)
        assert np.allclose(out, [0.5, 0.5, 0.5])

    def test_percent(self):
        basis = np.array([0.0, 10.0, 100.0])
        out = _resolve_limit({'mode': 'percent', 'percent': 0.02}, basis)
        assert np.allclose(out, [0.0, 0.2, 2.0])

    def test_max_absolute_percent(self):
        """2. max(固定值, 真值百分比)：真值小时取固定下限。"""
        basis = np.array([0.0, 10.0, 100.0])
        out = _resolve_limit(
            {'mode': 'max_absolute_percent', 'absolute': 0.3, 'percent': 0.01}, basis)
        assert np.allclose(out, [0.3, 0.3, 1.0])

    def test_scaled_max_absolute_percent(self):
        """2. 系数 × max(固定值, 真值百分比)。"""
        basis = np.array([0.0, 10.0, 100.0])
        out = _resolve_limit(
            {'mode': 'scaled_max_absolute_percent', 'scale': 1.4,
             'absolute': 1.5, 'percent': 0.05}, basis)
        assert np.allclose(out, [1.4 * 1.5, 1.4 * 1.5, 1.4 * 5.0])

    def test_invalid_mode_returns_nan(self):
        basis = np.array([1.0, 2.0])
        assert np.all(np.isnan(_resolve_limit({'mode': 'bogus'}, basis)))

    def test_zero_truth_keeps_absolute_floor(self):
        """3. 真值为零时百分比项为零，但 max 中的固定下限仍有效。"""
        basis = np.array([0.0])
        out = _resolve_limit(
            {'mode': 'max_absolute_percent', 'absolute': 0.3, 'percent': 0.03}, basis)
        assert out[0] == 0.3


# ============================================================
# 测试 3：正负值与真值为零
# ============================================================

class TestSignedValues:

    def test_negative_truth_uses_abs_for_percent(self):
        """3. 真值为负时百分比基于绝对值计算。"""
        basis = np.array([-100.0])
        out = _resolve_limit({'mode': 'percent', 'percent': 0.03}, np.abs(basis))
        assert np.allclose(out, [3.0])

    def test_negative_velocity_error_is_absolute(self):
        """Vx 真值为负时误差仍按绝对值计算。"""
        rules = {
            'label': 'Vx', 'unit': 'm/s', 'mode': 'binned',
            'radar_col': 'radar_Vx', 'truth_col': 'rtk_Vx',
            'percent_basis': 'truth',
            'bins': [
                {'range': [0, 10],
                 'normal_limit': {'mode': 'absolute', 'absolute': 0.3},
                 'three_frame_limit': {'mode': 'absolute', 'absolute': 1.0}},
                {'range': [10, 20],
                 'normal_limit': {'mode': 'absolute', 'absolute': 0.3},
                 'three_frame_limit': {'mode': 'absolute', 'absolute': 1.0}},
                {'range': [20, 50],
                 'normal_limit': {'mode': 'absolute', 'absolute': 1.0},
                 'three_frame_limit': {'mode': 'absolute', 'absolute': 1.5}},
                {'range': [50, 70],
                 'normal_limit': {'mode': 'absolute', 'absolute': 1.0},
                 'three_frame_limit': {'mode': 'absolute', 'absolute': 1.5}},
                {'range': [70, 100],
                 'normal_limit': {'mode': 'absolute', 'absolute': 1.0},
                 'three_frame_limit': {'mode': 'absolute', 'absolute': 1.5}},
                {'range': [100, 150],
                 'normal_limit': {'mode': 'absolute', 'absolute': 1.0},
                 'three_frame_limit': {'mode': 'absolute', 'absolute': 1.5}},
            ],
        }
        rows = [
            {'radar_Vx': -5.0, 'rtk_Vx': -5.0, 'rtk_center_x': 30.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Vx', rules, _simple_config())
        assert out['frames']['abs_error'].iloc[0] == 0.0
        assert out['frames']['normal_pass'].iloc[0]


# ============================================================
# 测试 4/5：严格不等号
# ============================================================

class TestStrictInequality:

    def test_error_equal_to_limit_fails(self):
        """4. 误差恰好等于普通阈值时该帧不合格（严格小于）。

        阈值取 0.5、真值取 5.0、雷达取 5.5 —— 均可被二进制浮点精确表示，
        确保误差严格等于 0.5，避免浮点误差干扰边界判定。
        """
        rules = _fixed_limit_rules(0.5)
        rows = [{'radar_Dx': 5.5, 'rtk_center_x': 5.0}]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', rules, _simple_config())
        assert out['frames']['abs_error'].iloc[0] == 0.5
        assert not out['frames']['normal_pass'].iloc[0]

    def test_error_below_limit_passes(self):
        """误差严格小于阈值时合格。"""
        rules = _fixed_limit_rules(0.5)
        rows = [{'radar_Dx': 5.25, 'rtk_center_x': 5.0}]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', rules, _simple_config())
        assert out['frames']['abs_error'].iloc[0] == 0.25
        assert out['frames']['normal_pass'].iloc[0]

    def test_accuracy_equal_to_requirement_fails(self):
        """5. 准确率恰好等于 95.45% 时区间不通过（严格大于）。

        构造 1000 帧：954 帧合格 → 准确率恰好 95.4%；
        为精确命中 95.45%，使用 2000 帧中 1909 帧合格 = 95.45%。
        """
        n = 2000
        pass_n = 1909          # 1909 / 2000 = 0.9545
        rows = []
        for i in range(n):
            # 0-10m 段：T=max(0.3, 0.01d)，d=5 → T=0.3
            ok = i < pass_n
            err = 0.1 if ok else 0.5
            rows.append({'radar_Dx': 5.0 + err, 'rtk_center_x': 5.0})
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        bin0 = out['bins'][0]
        assert bin0['accuracy'] == pytest.approx(0.9545)
        assert bin0['accuracy_pass'] is False
        assert bin0['status'] == STATUS_FAIL

    def test_accuracy_just_above_requirement_passes(self):
        n = 2000
        pass_n = 1910          # 1910 / 2000 = 0.955 > 0.9545
        rows = []
        for i in range(n):
            err = 0.1 if i < pass_n else 0.5
            rows.append({'radar_Dx': 5.0 + err, 'rtk_center_x': 5.0})
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['bins'][0]['accuracy_pass'] is True


# ============================================================
# 测试 6/7/8：连续三帧指标
# ============================================================

class TestThreeFrame:

    def test_rolling_window_min(self):
        """mins 与输入等长，前 window-1 项为 NaN，mins[j] = 窗口 [j-2, j] 的最小值。"""
        values = np.array([1.02, 1.24, 1.10, 0.20])
        out = _rolling_window_min(values, 3)
        assert len(out) == len(values)
        assert np.all(np.isnan(out[:2]))
        # 窗口 [0,2] = min(1.02, 1.24, 1.10) = 1.02
        # 窗口 [1,3] = min(1.24, 1.10, 0.20) = 0.20
        assert np.allclose(out[2:], [1.02, 0.20])

    def test_all_three_at_limit_fails(self):
        """6. 三帧全部达到限值 → R3 >= 1 → 不通过。

        设计文档 14.2 示例：0.51、0.62、0.55m，L=0.5
        → min(1.02, 1.24, 1.10) = 1.02 → 不通过
        """
        rows = [
            {'radar_Dx': 5.0 + 0.51, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.62, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.55, 'rtk_center_x': 5.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        # 样本不足 3 帧才能出结论，这里恰好 3 帧
        assert out['bins'][0]['three_frame_index'] == pytest.approx(1.02, abs=1e-2)
        assert out['bins'][0]['three_frame_pass'] is False

    def test_one_below_limit_passes(self):
        """7. 三帧中任意一帧低于限值时不构成违规。

        示例：0.51、0.48、0.60m → min(1.02, 0.96, 1.20) = 0.96 → 通过
        """
        rows = [
            {'radar_Dx': 5.0 + 0.51, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.48, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.60, 'rtk_center_x': 5.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['bins'][0]['three_frame_index'] == pytest.approx(0.96, abs=1e-2)
        assert out['bins'][0]['three_frame_pass'] is True

    def test_one_very_small_error_passes(self):
        """示例：0.80、0.80、0.10m → min(1.60,1.60,0.20) = 0.20 → 通过"""
        rows = [
            {'radar_Dx': 5.0 + 0.80, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.80, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.10, 'rtk_center_x': 5.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['bins'][0]['three_frame_index'] == pytest.approx(0.20, abs=1e-2)
        assert out['bins'][0]['three_frame_pass'] is True

    def test_dynamic_limit_normalization(self):
        """8. 动态限值三帧的归一化计算（50-70m：L = 0.05×d）。"""
        rows = [
            {'radar_Dx': 60.0 + 3.1, 'rtk_center_x': 60.0},   # r = 3.1/3.0 = 1.033
            {'radar_Dx': 60.0 + 3.3, 'rtk_center_x': 60.0},   # r = 3.3/3.0 = 1.100
            {'radar_Dx': 60.0 + 3.2, 'rtk_center_x': 60.0},   # r = 3.2/3.0 = 1.067
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        r3 = out['bins'][3]['three_frame_index']
        assert r3 == pytest.approx(1.0333, abs=1e-3)
        assert out['bins'][3]['three_frame_pass'] is False

    def test_r3_is_max_over_windows(self):
        """R3 = 所有窗口中 min(r_i, r_i+1, r_i+2) 的最大值。"""
        rows = [
            {'radar_Dx': 5.0 + 0.10, 'rtk_center_x': 5.0},   # 窗口1 min 小
            {'radar_Dx': 5.0 + 0.10, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.10, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.60, 'rtk_center_x': 5.0},   # 窗口2 min = 1.2
            {'radar_Dx': 5.0 + 0.60, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 + 0.60, 'rtk_center_x': 5.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        # 窗口2（帧 3,4,5）min = 0.6/0.5 = 1.2
        assert out['bins'][0]['three_frame_index'] == pytest.approx(1.2, abs=1e-2)


# ============================================================
# 测试 9：连续性中断与跨段
# ============================================================

class TestContinuity:

    def test_unmatched_frame_breaks_continuity(self):
        """9. 中间一帧未匹配 → 连续计数中断。"""
        matched = np.array([True, True, False, True, True, True])
        ts = np.arange(6, dtype=float) * 0.05
        breaks = _compute_continuity_breaks(ts, 2.5, 25, matched)
        # 索引 2 未匹配 → 2 和 3 都断开
        assert breaks[2] and breaks[3]

    def test_time_gap_breaks_continuity(self):
        """9. 数据中断打断连续计数。"""
        matched = np.ones(5, dtype=bool)
        ts = np.array([0.0, 0.05, 5.0, 5.05, 5.10])   # 中间存在 5 秒中断
        breaks = _compute_continuity_breaks(ts, 2.5, 25, matched)
        assert breaks[2]

    def test_crossing_bin_boundary_keeps_continuity(self):
        """9/16.5 跨距离段不打断连续三帧关系。

        49.9、50.1、50.3m 仍是连续三帧：
        第一帧用 20-50m 规则（L=1.5），后两帧用 50-70m 规则（L=0.05×d）。
        窗口归入中间帧所在的 50-70m 区间。
        """
        rows = [
            {'radar_Dx': 49.9 + 1.6, 'rtk_center_x': 49.9},   # r = 1.6/1.5 = 1.067
            {'radar_Dx': 50.1 + 3.0, 'rtk_center_x': 50.1},   # r = 3.0/2.505 = 1.198
            {'radar_Dx': 50.3 + 3.0, 'rtk_center_x': 50.3},   # r = 3.0/2.515 = 1.193
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        # 窗口 min = 1.067，归入中间帧（50.1m）所在的 [50,70) 段
        assert out['bins'][3]['three_frame_index'] is not None
        assert out['bins'][3]['three_frame_index'] == pytest.approx(1.067, abs=1e-2)
        # 20-50m 段只有 1 帧（第 0 帧），无完整窗口，不参与统计
        assert out['bins'][2]['three_frame_index'] is None

    def test_first_and_last_frames_have_no_window_value(self):
        """首帧无前帧、末帧无后帧：均不输出三帧窗口值（窗口只归中间帧）。

        5m 距离属 [0,10) 段，三帧阈值 0.5：r = 1.6/0.5 = 3.2，0.2/0.5 = 0.4。
        仅帧 1（窗口 (0,1,2)）与帧 2（窗口 (1,2,3)）有窗口值。
        """
        rows = [
            {'radar_Dx': 5.0 + 1.6, 'rtk_center_x': 5.0},   # r = 1.6/0.5 = 3.2
            {'radar_Dx': 5.0 + 0.2, 'rtk_center_x': 5.0},   # r = 0.2/0.5 = 0.4
            {'radar_Dx': 5.0 + 0.2, 'rtk_center_x': 5.0},   # r = 0.4
            {'radar_Dx': 5.0 + 0.2, 'rtk_center_x': 5.0},   # r = 0.4
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        frames = out['frames']
        # 首、末帧无完整三帧窗口 → 无值；帧 1/帧 2 分别为窗口 (0,1,2)/(1,2,3)
        assert np.isnan(frames['normalized_severe_error'].iloc[0])
        assert frames['normalized_severe_error'].iloc[1] == pytest.approx(0.4)
        assert frames['normalized_severe_error'].iloc[2] == pytest.approx(0.4)
        assert np.isnan(frames['normalized_severe_error'].iloc[3])
        # 全部窗口最小值 < 1 → 无违规
        assert not frames['three_frame_violation'].any()
        # 段违规窗口数为 0
        assert out['bins'][0]['violation_window_count'] == 0


# ============================================================
# 测试 10：样本不足与不可判定
# ============================================================

class TestInsufficientSamples:

    def test_no_samples_is_undecidable(self):
        """10. 无样本区间显示不可判定，不得显示通过。"""
        rows = [{'radar_Dx': 5.0, 'rtk_center_x': 5.0}]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['bins'][1]['sample_count'] == 0
        assert out['bins'][1]['status'] == STATUS_UNDECIDABLE
        assert out['bins'][1]['status'] != STATUS_PASS

    def test_one_sample_is_insufficient(self):
        """10/16.3 区间仅 1 帧时不出具结论。"""
        rows = [{'radar_Dx': 5.0, 'rtk_center_x': 5.0}]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['bins'][0]['sample_count'] == 1
        assert out['bins'][0]['status'] == STATUS_INSUFFICIENT

    def test_two_samples_is_insufficient(self):
        rows = [
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.1, 'rtk_center_x': 5.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['bins'][0]['sample_count'] == 2
        assert out['bins'][0]['status'] == STATUS_INSUFFICIENT

    def test_insufficient_bin_blocks_overall_conclusion(self):
        """16.3 完整曲线涉及样本不足区间时，整体也不出具结论。"""
        rows = [
            # 0-10m 段：5 帧合格（可判定）
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            # 20-50m 段：仅 1 帧 → 样本不足
            {'radar_Dx': 30.0, 'rtk_center_x': 30.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['bins'][0]['status'] == STATUS_PASS
        assert out['bins'][2]['status'] == STATUS_INSUFFICIENT
        assert out['overall']['status'] == STATUS_INSUFFICIENT


# ============================================================
# 测试 11：RMSE 口径
# ============================================================

class TestRmse:

    def test_rmse_uses_signed_error_squared(self):
        """11. RMSE = sqrt(mean(s_i²))，使用有符号误差平方。"""
        rows = [
            {'radar_Dx': 5.0 + 0.2, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0 - 0.2, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
        ]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        expected = np.sqrt(np.mean([0.2 ** 2, (-0.2) ** 2, 0.0 ** 2]))
        assert out['bins'][0]['rmse'] == pytest.approx(expected)

    def test_rmse_does_not_affect_conclusion(self):
        """11/3.2 RMSE 不参与通过判定：RMSE 很大但准确率达标仍通过。"""
        rows = []
        # 20 帧，全部合格（误差 < T=0.3），但 RMSE 接近 0.29（较大）
        for _ in range(20):
            rows.append({'radar_Dx': 5.29, 'rtk_center_x': 5.0})
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        bin0 = out['bins'][0]
        assert bin0['rmse'] == pytest.approx(0.29, abs=1e-2)
        assert bin0['accuracy'] == 1.0
        assert bin0['status'] == STATUS_PASS


# ============================================================
# 测试 12：无效值不相互污染
# ============================================================

class TestInvalidIsolation:

    def test_nan_in_one_metric_not_affect_other(self):
        """12. 不同物理量的无效值不会相互污染样本数。"""
        rows = [
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
            {'radar_Dx': 5.0, 'rtk_center_x': 5.0},
        ]
        df = _make_aligned(rows)
        # Dy 全为 NaN
        df['radar_Dy'] = np.nan
        df['rtk_center_y'] = np.nan

        dy_rules = {
            'label': 'Dy', 'unit': 'm', 'mode': 'binned',
            'radar_col': 'radar_Dy', 'truth_col': 'rtk_center_y',
            'percent_basis': 'distance',
            'bins': DX_RULES['bins'],
        }
        out_dx = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        out_dy = evaluate_metric(df, 'Dy', dy_rules, _simple_config())

        assert out_dx['bins'][0]['sample_count'] == 4
        assert out_dy['bins'][0]['sample_count'] == 0
        assert out_dy['bins'][0]['status'] == STATUS_UNDECIDABLE

    def test_missing_column_marks_unavailable(self):
        """缺少必需字段时该指标不可用，不影响其他指标。"""
        rows = [{'radar_Dx': 5.0, 'rtk_center_x': 5.0}]
        df = _make_aligned(rows)
        ax_rules = {
            'label': 'Ax', 'unit': 'm/s²', 'mode': 'binned',
            'radar_col': 'radar[Ax]', 'truth_col': 'rtk[Ax]',
            'percent_basis': 'truth', 'bins': DX_RULES['bins'],
        }
        out = evaluate_metric(df, 'Ax', ax_rules, _simple_config())
        assert out['available'] is False
        assert out['reason']


# ============================================================
# Ay 单项限值（设计文档 8.6 / 16.1）
# ============================================================

class TestSingleLimit:

    AY_RULES = {
        'label': 'Ay', 'unit': 'm/s²', 'mode': 'single_limit',
        'abs_limit': 1.0, 'operator': '<=',
        'radar_col': 'radar[Ay]', 'truth_col': 'rtk[Ay]',
    }

    def _df(self, errors):
        rows = [{'radar[Ay]': 0.0 + e, 'rtk[Ay]': 0.0, 'rtk_center_x': 30.0}
                for e in errors]
        return _make_aligned(rows)

    def test_all_within_limit_passes(self):
        out = evaluate_metric(self._df([0.1, 0.5, 1.0]), 'Ay',
                              self.AY_RULES, _simple_config())
        assert out['available']
        assert out['bins'][0]['status'] == STATUS_PASS
        assert out['overall']['status'] == STATUS_PASS

    def test_exceeding_limit_fails(self):
        out = evaluate_metric(self._df([0.1, 1.01]), 'Ay',
                              self.AY_RULES, _simple_config())
        assert out['bins'][0]['status'] == STATUS_FAIL
        assert '超过限值' in out['bins'][0]['failure_reasons'][0]

    def test_limit_is_inclusive(self):
        """Ay 使用 <= 1m/s²，恰好等于 1.0 时合格。"""
        out = evaluate_metric(self._df([1.0, 1.0, 1.0]), 'Ay',
                              self.AY_RULES, _simple_config())
        assert out['bins'][0]['status'] == STATUS_PASS

    def test_no_samples_is_undecidable(self):
        df = _make_aligned([{'radar[Ay]': np.nan, 'rtk[Ay]': np.nan,
                             'rtk_center_x': 30.0}])
        out = evaluate_metric(df, 'Ay', self.AY_RULES, _simple_config())
        assert out['bins'][0]['status'] == STATUS_UNDECIDABLE

    def test_no_accuracy_or_three_frame(self):
        """16.1 Ay 不使用准确率与连续三帧规则。"""
        out = evaluate_metric(self._df([0.1, 0.2, 0.3]), 'Ay',
                              self.AY_RULES, _simple_config())
        bin0 = out['bins'][0]
        assert bin0['accuracy'] is None
        assert bin0['three_frame_index'] is None
        assert bin0['accuracy_pass'] is None


# ============================================================
# 总体结论与批量入口
# ============================================================

class TestOverall:

    def test_any_bin_fails_overall_fails(self):
        """16.6 任一可评估区间失败 → 整体失败。"""
        rows = []
        # 0-10m：20 帧全合格
        for _ in range(20):
            rows.append({'radar_Dx': 5.0, 'rtk_center_x': 5.0})
        # 20-50m：20 帧全不合格（误差 2.0 > T=0.02×30=0.6）
        for _ in range(20):
            rows.append({'radar_Dx': 32.0, 'rtk_center_x': 30.0})
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['bins'][0]['status'] == STATUS_PASS
        assert out['bins'][2]['status'] == STATUS_FAIL
        assert out['overall']['status'] == STATUS_FAIL
        assert out['overall']['failed_bins'] == 1

    def test_all_pass_overall_passes(self):
        rows = [{'radar_Dx': 5.0, 'rtk_center_x': 5.0} for _ in range(20)]
        df = _make_aligned(rows)
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['overall']['status'] == STATUS_PASS

    def test_no_data_overall_undecidable(self):
        df = _make_aligned([{'radar_Dx': 5.0, 'rtk_center_x': 5.0}])
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        # 仅 1 帧 → 样本不足，非无数据
        assert out['overall']['status'] == STATUS_INSUFFICIENT

    def test_coverage_reported(self):
        """5.3 匹配覆盖率单独统计，不参与通过判定。"""
        rows = [{'radar_Dx': 5.0, 'rtk_center_x': 5.0} for _ in range(10)]
        df = _make_aligned(rows)
        df.loc[9, 'is_matched'] = False
        out = evaluate_metric(df, 'Dx', DX_RULES, _simple_config())
        assert out['coverage']['total_frames'] == 10
        assert out['coverage']['matched_frames'] == 9
        assert out['coverage']['match_rate'] == pytest.approx(0.9)
        # 未匹配帧不进入样本统计
        assert out['bins'][0]['sample_count'] == 9

    def test_evaluate_all_metrics(self):
        rows = [{'radar_Dx': 5.0, 'rtk_center_x': 5.0} for _ in range(5)]
        df = _make_aligned(rows)
        df['radar_Dy'] = 2.0
        df['rtk_center_y'] = 2.0
        config = _simple_config(metrics={
            'Dx': DX_RULES,
            'Dy': {'label': 'Dy', 'unit': 'm', 'mode': 'binned',
                   'radar_col': 'radar_Dy', 'truth_col': 'rtk_center_y',
                   'percent_basis': 'distance', 'bins': DX_RULES['bins']},
        })
        results = evaluate_all_metrics(df, config=config)
        assert set(results.keys()) == {'Dx', 'Dy'}
        assert results['Dx']['available']
        assert results['Dy']['available']
