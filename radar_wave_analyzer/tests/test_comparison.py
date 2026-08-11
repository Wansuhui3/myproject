"""雷达与 RTK 真值对齐、延迟扫描的回归测试。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.alignment import (  # noqa: E402
    _nearest_indices,
    align_trajectories,
    compute_distance_bin_stats,
)
from comparison.delay_detect import scan_delay  # noqa: E402
from comparison.parser import load_csv_file  # noqa: E402
from comparison.matching import (  # noqa: E402
    extract_radar_trajectory,
    filter_and_match_ids,
    filter_moving_radar_targets,
)
from comparison.service import (  # noqa: E402
    analyse_selected_track,
    execute_alignment,
    prepare_comparison_upload,
    resolve_track_selection,
)


def _radar_df(timestamps, dx_values):
    return pd.DataFrame({
        'timestamp': [f't{i}' for i in range(len(timestamps))],
        'timestamp_parsed': timestamps,
        'ID': [7] * len(timestamps),
        'Dx': dx_values,
        'Dy': [0.0] * len(timestamps),
        'Vx': [0.0] * len(timestamps),
        'Vy': [0.0] * len(timestamps),
    })


def _rtk_df(timestamps, x_values):
    return pd.DataFrame({
        'timestamp': [f'r{i}' for i in range(len(timestamps))],
        'timestamp_parsed': timestamps,
        'ID': [1] * len(timestamps),
        'center_x': x_values,
        'center_y': [0.0] * len(timestamps),
        'Vx': [0.0] * len(timestamps),
        'Vy': [0.0] * len(timestamps),
    })


def test_nearest_rtk_index_compares_both_neighbors():
    """searchsorted 右侧点不一定最近，必须能够选中左侧相邻点。"""
    indices = _nearest_indices(np.array([0.0, 0.1]), np.array([0.04, 0.08]))
    assert indices.tolist() == [0, 1]


def test_alignment_summary_excludes_unmatched_frames():
    """未匹配离群帧不能污染有效匹配帧的 RMSE 与距离分桶。"""
    radar = _radar_df(np.array([0.0, 0.05, 0.1]), [0.0, 1.0, 100.0])
    rtk = _rtk_df(np.array([0.0, 0.1]), [0.0, 2.0])

    result = align_trajectories(
        radar, rtk, 7, match_threshold_m=5.0, time_gate_ms=60.0,
    )

    assert result['match_summary']['matched_frames'] == 2
    assert result['match_summary']['spatial_rejected_frames'] == 1
    assert result['summary']['pos_error_abs']['rmse'] == 0.0

    bins = compute_distance_bin_stats(result['aligned_df'], [0, 10, 200])
    assert bins[0]['frames'] == 2
    assert bins[1]['frames'] == 0


def test_alignment_rejects_samples_outside_rtk_time_range():
    """RTK 范围外的帧不可因边界外推而标记为匹配。"""
    radar = _radar_df(np.array([0.2]), [0.0])
    rtk = _rtk_df(np.array([0.0, 0.1]), [0.0, 0.0])

    result = align_trajectories(
        radar, rtk, 7, match_threshold_m=1.0, time_gate_ms=1000.0,
    )

    assert result['match_summary']['matched_frames'] == 0
    assert result['match_summary']['out_of_rtk_range_frames'] == 1
    assert result['summary']['pos_error_abs']['rmse'] is None


def test_delay_scan_requires_sufficient_coverage():
    """仅少量帧匹配时不能输出看似精确的最优延迟。"""
    radar = _radar_df(np.array([0.0, 0.1, 0.2]), [0.0, 100.0, 100.0])
    rtk = _rtk_df(np.array([0.0, 0.1, 0.2]), [0.0, 0.0, 100.0])

    result = scan_delay(
        radar, rtk, 7, delay_range=(0, 0), step_ms=10,
        match_threshold_m=1.0, min_matched_frames=3, min_match_rate=1.0,
    )

    assert result['min_rmse'] is None
    assert result['level'] == 'insufficient_coverage'
    assert result['delay_samples'][0]['matched_frames'] == 2


def test_comparison_parser_drops_non_numeric_measurements():
    """对比输入中的非数值位置/速度行应在上传阶段被剔除并提示。"""
    content = (
        'timestamp,ID,Track_Age,Dx,Dy,Vx,Vy\n'
        '2026_04_20_10_00_00_000,7,1,1.0,2.0,0.1,0.2\n'
        '2026_04_20_10_00_00_050,7,2,not-a-number,2.0,0.1,0.2\n'
    ).encode('utf-8')

    result = load_csv_file(content, 'radar.csv')

    assert result['role'] == 'radar'
    assert result['errors'] == []
    assert len(result['df']) == 1
    assert any('不是有限数值' in warning for warning in result['warnings'])


def test_comparison_parser_vectorized_timestamps_keep_millisecond_scale():
    """批量时间戳解析在不同 Pandas 时间分辨率下都应保持 50ms 间隔。"""
    content = (
        'timestamp,ID,Track_Age,Dx,Dy,Vx,Vy\n'
        '2026_04_20_10_00_00_000,7,1,1.0,2.0,0.1,0.2\n'
        '2026_04_20_10_00_00_050,7,2,1.1,2.0,0.1,0.2\n'
    ).encode('utf-8')

    result = load_csv_file(content, 'radar.csv')

    assert result['errors'] == []
    timestamps = result['df']['timestamp_parsed'].to_numpy()
    assert timestamps[1] - timestamps[0] == pytest.approx(0.05)


def test_comparison_parser_reports_missing_velocity_columns():
    """缺少对齐所需速度字段时在上传阶段报告字段错误。"""
    content = (
        'timestamp,ID,center_x,center_y\n'
        '2026_04_20_10_00_00_000,1,1.0,2.0\n'
    ).encode('utf-8')

    result = load_csv_file(content, 'rtk.csv')

    assert result['role'] == 'rtk'
    assert '缺少必要列: Vx' in result['errors']
    assert '缺少必要列: Vy' in result['errors']


def test_comparison_service_runs_selection_diagnosis_and_alignment():
    """服务层可在不依赖 Dash 或缓存的情况下编排完整对比计算。"""
    radar = _radar_df(np.array([0.0, 0.05, 0.1]), [0.0, 1.0, 2.0])
    rtk = _rtk_df(np.array([0.0, 0.05, 0.1]), [0.0, 1.0, 2.0])
    config = {
        'match_threshold': 1.0,
        'time_gate_ms': 50.0,
        'coord_bias_threshold': 0.5,
        'delay_scan_range': [0, 0],
        'delay_scan_step': 10,
        'delay_insensitive_ratio': 0.05,
        'delay_min_matched_frames': 3,
        'delay_min_match_rate': 1.0,
    }

    selection = resolve_track_selection(radar, rtk, config, None, None)
    assert selection['track_id'] == 7

    analysis = analyse_selected_track(radar, rtk, config, 7)
    assert analysis['coordinate']['same_system'] is True
    assert analysis['suggested_delay_ms'] == 0

    result = execute_alignment(radar, rtk, config, 7)
    assert result['match_summary']['matched_frames'] == 3


def test_comparison_upload_service_merges_by_parsed_timestamp():
    """多文件上传应保留来源序号，并依据解析后的时间全局排序。"""
    late = (
        'timestamp,ID,Track_Age,Dx,Dy,Vx,Vy\n'
        '2026_04_20_10_00_00_100,7,3,3.0,0.0,0.0,0.0\n'
    ).encode('utf-8')
    early = (
        'timestamp,ID,Track_Age,Dx,Dy,Vx,Vy\n'
        '2026_04_20_10_00_00_000,7,1,1.0,0.0,0.0,0.0\n'
    ).encode('utf-8')

    result = prepare_comparison_upload(
        [(late, 'late.csv'), (early, 'early.csv')], 'radar',
    )

    assert result['errors'] == []
    assert result['file_count'] == 2
    assert result['info']['df']['Dx'].tolist() == [1.0, 3.0]
    assert result['info']['df']['file_index'].tolist() == [1, 0]


def test_comparison_upload_service_rejects_wrong_drop_zone():
    """RTK 文件被拖入雷达区域时应得到可理解的错误而不是错误缓存。"""
    rtk = (
        'timestamp,ID,center_x,center_y,Vx,Vy\n'
        '2026_04_20_10_00_00_000,1,1.0,2.0,0.0,0.0\n'
    ).encode('utf-8')

    result = prepare_comparison_upload([(rtk, 'truth.csv')], 'radar')

    assert result['info'] is None
    assert any('请上传到雷达区域' in error for error in result['errors'])


def test_motion_filter_removes_static_ids_and_matches_remaining_ids_one_to_one():
    """静止雷达 ID 必须被剔除，运动 ID 与真值 ID 应按时空证据一对一关联。"""
    timestamps = np.array([0.0, 0.1, 0.2])
    radar = pd.DataFrame({
        'timestamp': [f't{i}' for i in range(9)],
        'timestamp_parsed': np.tile(timestamps, 3),
        'ID': [10] * 3 + [20] * 3 + [30] * 3,
        'Dx': [0.0, 0.0, 0.0] + [0.0, 1.0, 2.0] + [100.0, 101.0, 102.0],
        'Dy': [0.0] * 9,
        'Vx': [0.0] * 3 + [10.0] * 6,
        'Vy': [0.0] * 9,
    })
    rtk = pd.DataFrame({
        'timestamp': [f'r{i}' for i in range(6)],
        'timestamp_parsed': np.tile(timestamps, 2),
        'ID': [101] * 3 + [102] * 3,
        'center_x': [0.0, 1.0, 2.0] + [100.0, 101.0, 102.0],
        'center_y': [0.0] * 6,
        'Vx': [10.0] * 6,
        'Vy': [0.0] * 6,
    })
    config = {
        'match_threshold': 0.2,
        'id_matching': {
            'min_frames': 3,
            'min_speed_mps': 0.5,
            'min_accel_mps2': 10.0,
            'min_displacement_m': 0.5,
            'min_pair_frames': 3,
            'min_pair_coverage': 1.0,
        },
    }

    result = filter_and_match_ids(radar, rtk, config)

    assert result['filter_stats']['filtered_static_targets'] == 1
    assert result['filter_stats']['matched_target_pairs'] == 2
    assert {(item['track_id'], item['rtk_id']) for item in result['valid_match_ids']} == {
        (20, 101), (30, 102),
    }
    assert {item['track_id'] for item in result['valid_match_ids']} == {20, 30}


def test_multi_file_same_id_is_retained_by_file_time_and_track_age_segments():
    """三份文件复用同一雷达/RTK ID 时，必须保留三条独立的有效关联。"""
    radar_rows = []
    rtk_rows = []
    for file_index, start in enumerate([0.0, 10.0, 20.0]):
        for frame in range(3):
            timestamp = start + frame * 0.1
            position = file_index * 100.0 + frame
            radar_rows.append({
                'timestamp': f'radar-{file_index}-{frame}',
                'timestamp_parsed': timestamp,
                'file_index': file_index,
                'ID': 7,
                'Track_Age': frame + 1,
                'Dx': position,
                'Dy': 0.0,
                'Vx': 10.0,
                'Vy': 0.0,
            })
            rtk_rows.append({
                'timestamp': f'rtk-{file_index}-{frame}',
                'timestamp_parsed': timestamp,
                'file_index': file_index,
                'ID': 100,
                'center_x': position,
                'center_y': 0.0,
                'Vx': 10.0,
                'Vy': 0.0,
            })

    result = filter_and_match_ids(
        pd.DataFrame(radar_rows), pd.DataFrame(rtk_rows),
        {
            'match_threshold': 0.2,
            'id_matching': {
                'min_frames': 3,
                'min_speed_mps': 0.5,
                'min_accel_mps2': 10.0,
                'min_displacement_m': 0.5,
                'min_pair_frames': 3,
                'min_pair_coverage': 1.0,
            },
        },
    )

    assert result['filter_stats']['matched_target_pairs'] == 3
    assert [(item['file_index'], item['rtk_file_index']) for item in result['valid_match_ids']] == [
        (0, 0), (1, 1), (2, 2),
    ]


def test_track_age_reset_creates_independent_radar_lifecycle_segments():
    """同文件同 ID 的非回绕 Track_Age 下降必须产生独立候选段。"""
    radar = pd.DataFrame({
        'timestamp_parsed': [0.0, 0.1, 0.2, 0.3],
        'ID': [8, 8, 8, 8],
        'Track_Age': [10, 11, 1, 2],
        'Dx': [0.0, 1.0, 100.0, 101.0],
        'Dy': [0.0, 0.0, 0.0, 0.0],
        'Vx': [10.0, 10.0, 10.0, 10.0],
        'Vy': [0.0, 0.0, 0.0, 0.0],
    })
    config = {
        'min_frames': 2,
        'min_speed_mps': 0.5,
        'min_accel_mps2': 10.0,
        'min_displacement_m': 0.5,
    }

    filtered = filter_moving_radar_targets(radar, config)
    second_segment = extract_radar_trajectory(radar, 8, None, 2, config)

    assert [item['segment_index'] for item in filtered['target_stats']] == [1, 2]
    assert second_segment['Track_Age'].tolist() == [1, 2]


def test_track_age_single_frame_glitch_and_radar_gap_do_not_split_track():
    """单帧 Track_Age 回跳和雷达缺帧必须保留在同一条轨迹中。"""
    radar = pd.DataFrame({
        'timestamp': ['t0', 't1', 't2', 't3', 't4'],
        'timestamp_parsed': [0.0, 0.1, 0.2, 5.3, 5.4],
        'ID': [8] * 5,
        'Track_Age': [162, 163, 104, 165, 166],
        'Dx': [0.0, 1.0, 1.1, 2.0, 3.0],
        'Dy': [0.0] * 5,
        'Vx': [10.0] * 5,
        'Vy': [0.0] * 5,
    })
    config = {
        'min_frames': 2,
        'min_speed_mps': 0.5,
        'min_accel_mps2': 10.0,
        'min_displacement_m': 0.5,
        'track_age_gap_ms': 500.0,
    }

    filtered = filter_moving_radar_targets(radar, config)
    result = filter_and_match_ids(
        radar,
        pd.DataFrame({
            'timestamp': [f'r{i}' for i in range(5)],
            'timestamp_parsed': [0.0, 0.1, 0.2, 5.3, 5.4],
            'ID': [1] * 5,
            'center_x': [0.0, 1.0, 1.1, 2.0, 3.0],
            'center_y': [0.0] * 5,
            'Vx': [10.0] * 5,
            'Vy': [0.0] * 5,
        }),
        {'match_threshold': 0.2, 'id_matching': config},
    )

    assert len(filtered['target_stats']) == 1
    assert result['valid_match_ids'][0]['total_frames'] == 5
    assert result['valid_match_ids'][0]['gap_count'] == 1
    assert result['valid_match_ids'][0]['gap_duration_ms'] == 5100.0


def test_non_overlapping_lifecycle_segments_can_match_the_same_rtk_id():
    """同一 RTK ID 可对应不重叠时间段的雷达生命周期段。"""
    radar = pd.DataFrame({
        'timestamp': ['t0', 't1', 't2', 't3'],
        'timestamp_parsed': [0.0, 0.1, 10.0, 10.1],
        'ID': [8] * 4,
        'Track_Age': [10, 11, 1, 2],
        'Dx': [0.0, 1.0, 100.0, 101.0],
        'Dy': [0.0] * 4,
        'Vx': [10.0] * 4,
        'Vy': [0.0] * 4,
    })
    rtk = pd.DataFrame({
        'timestamp': ['r0', 'r1', 'r2', 'r3'],
        'timestamp_parsed': [0.0, 0.1, 10.0, 10.1],
        'ID': [1] * 4,
        'center_x': [0.0, 1.0, 100.0, 101.0],
        'center_y': [0.0] * 4,
        'Vx': [10.0] * 4,
        'Vy': [0.0] * 4,
    })
    config = {
        'match_threshold': 0.2,
        'id_matching': {
            'min_frames': 2,
            'min_speed_mps': 0.5,
            'min_accel_mps2': 10.0,
            'min_displacement_m': 0.5,
            'min_pair_frames': 2,
            'min_pair_coverage': 1.0,
        },
    }

    result = filter_and_match_ids(radar, rtk, config)

    assert result['filter_stats']['matched_target_pairs'] == 2
    assert [(item['segment_index'], item['rtk_id']) for item in result['valid_match_ids']] == [
        (1, 1), (2, 1),
    ]
