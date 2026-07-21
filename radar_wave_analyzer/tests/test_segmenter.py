"""
segmenter 模块单元测试。
覆盖开发规则中列出的全部必须测试用例。
"""
import sys
import os
# 将 radar_wave_analyzer 目录加入 sys.path，使得 core 和 config 可以被直接导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import core.segmenter as seg_mod
from core.segmenter import (
    segment_trajectories, get_segment_ids_by_time, _detect_breakpoints,
    _detect_lifecycle_breaks, _segment_max_speed,
)
from core.data_loader import parse_timestamp


def _make_test_df(records: list[dict]) -> pd.DataFrame:
    """构建测试用 DataFrame。"""
    df = pd.DataFrame(records)
    df['timestamp_parsed'] = df['timestamp'].apply(parse_timestamp)
    return df


class TestSegmenter:
    """分段算法单元测试"""

    # ---- 规则 A: ID 首现即起点 ----

    def test_rule_a_first_frame_age_not_one(self):
        """ID 首帧 Track_Age=14，应正常识别为段起点（不要求 Track_Age==1）"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 55, 'Track_Age': 14, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 55, 'Track_Age': 15, 'Dx': 10.1, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 55, 'Track_Age': 16, 'Dx': 10.2, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1
        assert meta.iloc[0]['first_track_age'] == 14
        assert meta.iloc[0]['trajectory_id'] == '55_seg1'

    # ---- 规则 B: uint8 回绕 = 延续 ----

    def test_rule_b_wrap_255_to_0(self):
        """255→0 回绕，应延续不切分"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 55, 'Track_Age': 254, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 55, 'Track_Age': 255, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 55, 'Track_Age': 0, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 55, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 55, 'Track_Age': 2, 'Dx': 10.0, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1, f'回绕不应切分，预期1段，实际{len(meta)}段'
        assert meta.iloc[0]['num_wraps'] >= 1

    # ---- 规则 C: 非回绕下降 = ID 复用 ----

    def test_rule_c_non_wrap_drop_83_to_14(self):
        """83→14 非回绕下降，应切分（ID 复用）"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 64, 'Track_Age': 82, 'Dx': 50.0, 'Dy': 10.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 64, 'Track_Age': 83, 'Dx': 50.1, 'Dy': 10.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 64, 'Track_Age': 14, 'Dx': 100.0, 'Dy': 60.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 64, 'Track_Age': 15, 'Dx': 100.1, 'Dy': 60.1},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 2, f'非回绕下降应切分，预期2段，实际{len(meta)}段'
        assert '64_seg1' in segs
        assert '64_seg2' in segs
        assert segs['64_seg2'].index.tolist() == [0, 1]

    # ---- 规则 D: 时间间隔超阈值 ----

    def test_rule_d_time_gap(self):
        """时间间隔超阈值，无条件切分"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 33, 'Track_Age': 10, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 33, 'Track_Age': 11, 'Dx': 10.1, 'Dy': 2.0},
            # 间隔大于 500ms
            {'timestamp': '2026-04-20 11:12:28', 'ID': 33, 'Track_Age': 12, 'Dx': 10.2, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:28', 'ID': 33, 'Track_Age': 13, 'Dx': 10.3, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df, gap_threshold=500)
        # 前2帧与后2帧之间有约1秒间隔
        assert len(meta) == 2, f'时间间隔超阈值应切分，预期2段，实际{len(meta)}段'

    # ---- 重复帧 ----

    def test_duplicate_frames_no_split(self):
        """重复帧（diff=0），不触发切分"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 5, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 5, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 6, 'Dx': 10.1, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1, f'重复帧不应切分，预期1段，实际{len(meta)}段'

    # ---- 单 ID 无断点 ----

    def test_single_id_no_breakpoints(self):
        """单 ID 无断点 → 1 段"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 2, 'Dx': 10.1, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 3, 'Dx': 10.2, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 4, 'Dx': 10.3, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1
        assert meta.iloc[0]['trajectory_id'] == '1_seg1'

    # ---- ID=64 → 3段 ----

    def test_id_64_three_segments(self):
        """ID=64 → 3 段（3 个不同物理目标）"""
        df = _make_test_df([
            # 段1: 目标A
            {'timestamp': '2026-06-23 10:31:42', 'ID': 64, 'Track_Age': 80, 'Dx': 50.0, 'Dy': 10.0},
            {'timestamp': '2026-06-23 10:31:42', 'ID': 64, 'Track_Age': 81, 'Dx': 50.1, 'Dy': 10.0},
            {'timestamp': '2026-06-23 10:31:42', 'ID': 64, 'Track_Age': 82, 'Dx': 50.2, 'Dy': 10.0},
            # 段2: ID复用, 目标B (Track_Age下降)
            {'timestamp': '2026-06-23 10:31:43', 'ID': 64, 'Track_Age': 10, 'Dx': 100.0, 'Dy': 72.0},
            {'timestamp': '2026-06-23 10:31:43', 'ID': 64, 'Track_Age': 11, 'Dx': 100.1, 'Dy': 72.1},
            # 段3: ID再次复用, 目标C
            {'timestamp': '2026-06-23 10:31:44', 'ID': 64, 'Track_Age': 5, 'Dx': 30.0, 'Dy': 15.0},
            {'timestamp': '2026-06-23 10:31:44', 'ID': 64, 'Track_Age': 6, 'Dx': 30.1, 'Dy': 15.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 3, f'ID=64 应有3段，实际{len(meta)}段'
        assert '64_seg1' in segs
        assert '64_seg2' in segs
        assert '64_seg3' in segs

    # ---- 异常检测 ----

    def test_abnormal_track_age_out_of_range(self):
        """Track_Age 超出 0~255 应被预处理过滤，不参与分段"""
        # data_loader 已过滤超出范围的行，此处验证正常范围内正常分段
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 200, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 201, 'Dx': 10.1, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1
        assert not meta.iloc[0]['is_abnormal']

    # ---- 轨迹 ID 命名 ----

    def test_trajectory_id_naming(self):
        """轨迹 ID 命名: {原始ID}_seg{序号}，序号从1递增"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 10, 'Track_Age': 50, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 10, 'Track_Age': 51, 'Dx': 10.1, 'Dy': 2.0},
            # ID 复用
            {'timestamp': '2026-04-20 11:12:28', 'ID': 10, 'Track_Age': 5, 'Dx': 100.0, 'Dy': 50.0},
            {'timestamp': '2026-04-20 11:12:28', 'ID': 10, 'Track_Age': 6, 'Dx': 100.1, 'Dy': 50.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 2
        assert meta.iloc[0]['trajectory_id'] == '10_seg1'
        assert meta.iloc[1]['trajectory_id'] == '10_seg2'

    # ---- 规则 E: 跨文件同 ID 按来源文件硬切分 ----

    def test_rule_e_file_boundary_split_same_id(self):
        """多选文件: 同一 ID 出现在不同时刻/不同文件, 即使 Track_Age 连续且间隔很小,
        也必须被切分为独立段(独立曲线), 不得错误合并。"""
        df = _make_test_df([
            # 文件 0: ID=7 在 10:00:00 附近
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 2, 'Dx': 10.1, 'Dy': 2.0},
            # 文件 1: 同一 ID=7 在 11:30:00 附近(不同时刻), Track_Age 仍连续(3,4)
            {'timestamp': '2026-04-20 11:30:00', 'ID': 7, 'Track_Age': 3, 'Dx': 10.2, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:30:00', 'ID': 7, 'Track_Age': 4, 'Dx': 10.3, 'Dy': 2.0},
        ])
        df['file_index'] = [0, 0, 1, 1]
        meta, segs = segment_trajectories(df)
        assert len(meta) == 2, f'跨文件同ID应切分为2段，实际{len(meta)}段'
        assert '7_seg1' in segs
        assert '7_seg2' in segs
        # 两段时间区间应分别对应两个文件, 不重叠
        assert segs['7_seg1']['timestamp_parsed'].max() < segs['7_seg2']['timestamp_parsed'].min()

    def test_rule_e_no_file_index_keeps_legacy_behavior(self):
        """无 file_index 列时(单文件/旧数据), 保持原有分段行为, 不引入额外切分。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 2, 'Dx': 10.1, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 3, 'Dx': 10.2, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1

    # ---- 规则 F: 位置不连续强制切分（同 ID 复用但 Track_Age 不降） ----

    @pytest.fixture
    def _enable_spatial(self):
        """临时开启空间切分相关配置，供规则 F / spatial_anomaly 测试。"""
        orig = seg_mod.get

        def _fake_get(key, default=None):
            overrides = {
                'SPATIAL_SPLIT_ENABLED': True,
                'MAX_TRACK_SPEED': 50.0,
                'POS_JUMP_THRESHOLD': 5.0,
                'POSITION_COLUMNS': ['Dx', 'Dy'],
            }
            if key in overrides:
                return overrides[key]
            return orig(key, default)

        seg_mod.get = _fake_get
        yield
        seg_mod.get = orig

    def test_rule_f_same_timestamp_reuse_split(self, _enable_spatial):
        """同 ID + 同时间戳 + Track_Age 连续(不降), 但位置发生跳变 → 规则 F 强制切分。
        命中用户描述的'同一时刻同 ID 复用'盲区。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 2, 'Dx': 10.1, 'Dy': 2.0},
            # 同时间戳复用, Track_Age 仍连续(3,4), 但位置从 (10,2) 跳到 (100,50)
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 3, 'Dx': 100.0, 'Dy': 50.0},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 4, 'Dx': 100.1, 'Dy': 50.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 2, f'同ID同时间戳位置跳变应切分为2段，实际{len(meta)}段'
        assert '7_seg1' in segs and '7_seg2' in segs

    def test_rule_f_disabled_keeps_legacy_bug_behavior(self):
        """规则 F 默认关闭时, 同 ID + 同时间戳 + Track_Age 连续 + 位置跳变 → 不切分(保持旧行为)。
        验证默认配置零回归。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 2, 'Dx': 10.1, 'Dy': 2.0},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 3, 'Dx': 100.0, 'Dy': 50.0},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 4, 'Dx': 100.1, 'Dy': 50.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1, '默认关闭规则F时应保持旧行为(1段)'

    def test_rule_f_detect_breakpoints_unit(self):
        """直接验证 _detect_breakpoints 规则 F: dt>0 速度超阈值 → 切分。"""
        ages = np.array([1, 2, 3])
        # 时间戳间隔 100ms（< 500ms 间隔阈值, 避免规则 D 先触发）
        ts = pd.to_datetime([
            '2026-04-20 10:00:00.000', '2026-04-20 10:00:00.100', '2026-04-20 10:00:00.200',
        ])
        # 第2→3帧位置跳变 90m / 1s = 90 m/s > 50
        positions = np.array([[10.0, 2.0], [10.1, 2.0], [100.0, 50.0]])
        _, _, _, spatial_pts, all_breaks = _detect_breakpoints(
            ages, ts, 250, 5, 500,
            spatial_split_enabled=True, positions=positions,
            max_track_speed=50.0, pos_jump_threshold=5.0,
        )
        assert 2 in spatial_pts, '位置跳变帧应被记为空间断点'
        assert 2 in all_breaks

    # ---- 规则 G: 生命周期/状态字段跳变强制切分 ----

    def test_rule_g_lifecycle_break(self):
        """生命周期字段: 前帧'end' + 后帧'new' → 强制切分。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 10:00:00', 'ID': 9, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0, 'Track_Status': 'valid'},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 9, 'Track_Age': 2, 'Dx': 10.1, 'Dy': 2.0, 'Track_Status': 'end'},
            # 同 ID 复用, 状态从 end 跳到 new, 即使 Track_Age 连续也应切分
            {'timestamp': '2026-04-20 10:00:00', 'ID': 9, 'Track_Age': 3, 'Dx': 100.0, 'Dy': 50.0, 'Track_Status': 'new'},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 9, 'Track_Age': 4, 'Dx': 100.1, 'Dy': 50.0, 'Track_Status': 'valid'},
        ])
        end_tokens = {'end', 'dead', 'lost', 'invalid', '0', 'false'}
        start_tokens = {'new', 'begin', 'valid', 'alive', '1', 'true'}
        breaks = _detect_lifecycle_breaks(df, ['Track_Status'], end_tokens, start_tokens)
        assert 2 in breaks, '生命周期 end→new 应触发切分断点'

    def test_rule_g_no_lifecycle_cols_noop(self):
        """未配置生命周期字段 → 返回空, 不影响现有行为。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 10:00:00', 'ID': 1, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 10:00:00', 'ID': 1, 'Track_Age': 2, 'Dx': 10.1, 'Dy': 2.0},
        ])
        end_tokens = {'end'}
        start_tokens = {'new'}
        breaks = _detect_lifecycle_breaks(df, [], end_tokens, start_tokens)
        assert breaks == []

    # ---- 改动 4: 空间跳变标记（spatial_anomaly, 只读） ----

    def test_spatial_anomaly_marker(self):
        """高速机动段应被标记为 spatial_anomaly=True（只读标记, 不改变分段数）。
        使用默认配置(规则 F 关闭), 标记独立于切分逻辑。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 10:00:00.000', 'ID': 5, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            # 间隔 50ms, 位移 10m → 200 m/s > 50
            {'timestamp': '2026-04-20 10:00:00.050', 'ID': 5, 'Track_Age': 2, 'Dx': 20.0, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1
        assert bool(meta.iloc[0]['spatial_anomaly']) is True

    def test_spatial_anomaly_marker_absent_for_normal(self):
        """正常低速段不应标记 spatial_anomaly。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 10:00:00.000', 'ID': 5, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 10:00:00.050', 'ID': 5, 'Track_Age': 2, 'Dx': 10.1, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert bool(meta.iloc[0]['spatial_anomaly']) is False

    def test_segment_max_speed_unit(self):
        """_segment_max_speed 直接验证帧间速度计算。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 10:00:00', 'ID': 5, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 10:00:01', 'ID': 5, 'Track_Age': 2, 'Dx': 100.0, 'Dy': 50.0},
        ])
        speed = _segment_max_speed(df, ['Dx', 'Dy'])
        # dist = sqrt(90^2 + 48^2) ≈ 102.0, dt=1s → ≈102 m/s
        assert speed is not None and speed > 100

    # ---- 改动 2: 跨文件同 (ID, timestamp) 不再误删 ----

    def test_file_index_dedup_keeps_cross_file_same_ts(self):
        """多文件合并后, 跨文件同 (ID, timestamp) 的不同目标应保留(由规则 E 独立分段),
        不被 drop_duplicates 误删。验证去重子集含 file_index。"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
            # 文件1 同 ID 同时间戳, 不同位置(不同物理目标)
            {'timestamp': '2026-04-20 10:00:00', 'ID': 7, 'Track_Age': 1, 'Dx': 200.0, 'Dy': 80.0},
        ])
        df['file_index'] = [0, 1]
        # 模拟 callbacks 中 drop_duplicates(subset=['file_index','ID','timestamp_parsed'])
        deduped = df.drop_duplicates(subset=['file_index', 'ID', 'timestamp_parsed'], keep='first')
        assert len(deduped) == 2, '跨文件同(ID,timestamp)应保留2行而非误删为1行'

    # ---- 规则优先级 ----

    def test_rule_priority_B_before_C(self):
        """规则 B（回绕）优先级高于规则 C（非回绕下降）"""
        df = _make_test_df([
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 253, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 254, 'Dx': 10.0, 'Dy': 2.0},
            # diff < 0, 但 age[i-1]=254 >= 250, age[i]=0 <= 5 → 规则B命中
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 0, 'Dx': 10.0, 'Dy': 2.0},
            {'timestamp': '2026-04-20 11:12:27', 'ID': 1, 'Track_Age': 1, 'Dx': 10.0, 'Dy': 2.0},
        ])
        meta, segs = segment_trajectories(df)
        assert len(meta) == 1, f'回绕不应被规则C误判切分，预期1段，实际{len(meta)}段'


def test_get_segment_ids_by_time_uses_interval_overlap():
    """时间筛选保留与窗口有任意交集的轨迹段。"""
    meta_df = pd.DataFrame({
        'trajectory_id': ['1_seg1', '2_seg1', '3_seg1'],
        'start_time': [
            '2026_04_20_10_00_00_000',
            '2026_04_20_10_01_00_000',
            '2026_04_20_10_02_00_000',
        ],
        'end_time': [
            '2026_04_20_10_00_30_000',
            '2026_04_20_10_01_30_000',
            '2026_04_20_10_02_30_000',
        ],
    })

    result = get_segment_ids_by_time(
        meta_df,
        pd.Timestamp('2026-04-20 10:00:20'),
        pd.Timestamp('2026-04-20 10:01:10'),
    )

    assert result == ['1_seg1', '2_seg1']
