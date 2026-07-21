"""基于运动状态过滤与时空门控的雷达—RTK 目标 ID 关联。"""
from typing import Any

import numpy as np
import pandas as pd


_DEFAULTS = {
    'enabled': True,
    'min_frames': 3,
    'min_speed_mps': 0.3,
    'min_accel_mps2': 1.0,
    'min_displacement_m': 0.5,
    'min_pair_frames': 3,
    'min_pair_coverage': 0.5,
    'track_age_gap_ms': 500.0,
    # Track_Age 的偶发回跳不能直接切断轨迹。只有低年龄值连续出现，且
    # 位置也发生明显跳变时，才视为同一雷达 ID 被复用。
    'track_age_confirm_frames': 2,
    'track_age_reset_max': 10,
    'lifecycle_position_jump_m': 8.0,
    'wrap_high': 250,
    'wrap_low': 5,
}


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(_DEFAULTS)
    if config:
        result.update(config)
    return result


def _radar_trajectory_groups(
    radar_df: pd.DataFrame,
    config: dict[str, Any],
) -> list[tuple[int, int | None, int, pd.DataFrame]]:
    """按文件、ID 和已确认的生命周期重置切分雷达候选轨迹。

    ``Track_Age`` 中偶发的单帧回跳很常见，不能将其直接视为 ID 重用；
    雷达上报中断也只是曲线断点，不是新的物理目标。只有低年龄值持续出现
    且位置突变时，才切成新的生命周期段。
    """
    has_file_index = 'file_index' in radar_df.columns
    group_keys = ['file_index', 'ID'] if has_file_index else ['ID']
    results: list[tuple[int, int | None, int, pd.DataFrame]] = []

    for group_key, raw_group in radar_df.groupby(group_keys, sort=True):
        if has_file_index:
            file_index, target_id = group_key
            file_index = int(file_index)
        else:
            target_id = group_key[0] if isinstance(group_key, tuple) else group_key
            file_index = None
        group = raw_group.sort_values('timestamp_parsed').reset_index(drop=True)
        # 真实对比上传会要求 Track_Age；对旧调用/合成数据缺失该列时，
        # 安全降级为仅依赖文件和时间间隔，而不是中断整个关联流程。
        if 'Track_Age' in group.columns:
            ages = pd.to_numeric(group['Track_Age'], errors='coerce').to_numpy(dtype=float)
        else:
            ages = np.full(len(group), np.nan)
        breakpoints: list[int] = []
        for index in range(1, len(group)):
            previous_age, current_age = ages[index - 1], ages[index]
            is_wrap = (
                np.isfinite(previous_age) and np.isfinite(current_age)
                and previous_age >= config['wrap_high']
                and current_age <= config['wrap_low']
                and current_age < previous_age
            )
            is_reset_candidate = (
                np.isfinite(previous_age) and np.isfinite(current_age)
                and current_age < previous_age and not is_wrap
                and current_age <= config['track_age_reset_max']
            )
            confirm_end = index + int(config['track_age_confirm_frames'])
            reset_ages = ages[index:confirm_end]
            is_confirmed_reset = (
                is_reset_candidate
                and len(reset_ages) == int(config['track_age_confirm_frames'])
                and np.isfinite(reset_ages).all()
                and np.all(np.diff(reset_ages) >= 0)
            )

            # 仅 Track_Age 复位不一定意味着 ID 已复用：雷达可能在同一目标
            # 短暂丢失后重新建轨。要求位置也明显跳变，避免将连续曲线切碎。
            position_jump = np.hypot(
                float(group['Dx'].iloc[index] - group['Dx'].iloc[index - 1]),
                float(group['Dy'].iloc[index] - group['Dy'].iloc[index - 1]),
            )
            is_reused_id = (
                is_confirmed_reset
                and np.isfinite(position_jump)
                and position_jump >= config['lifecycle_position_jump_m']
            )
            if is_reused_id:
                breakpoints.append(index)

        start = 0
        segment_index = 1
        for end in [*breakpoints, len(group)]:
            segment = group.iloc[start:end].copy().reset_index(drop=True)
            if not segment.empty:
                results.append((int(target_id), file_index, segment_index, segment))
                segment_index += 1
            start = end
    return results


def _gap_intervals(track_df: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    """提取轨迹内的雷达数据中断，供统计和图表显示，不作为生命周期切分条件。"""
    if len(track_df) < 2:
        return []
    ordered = track_df.sort_values('timestamp_parsed').reset_index(drop=True)
    timestamps = ordered['timestamp_parsed'].to_numpy(dtype=float)
    gaps = np.diff(timestamps) * 1000.0
    result: list[dict[str, Any]] = []
    for index in np.flatnonzero(gaps > config['track_age_gap_ms']):
        result.append({
            'start': float(timestamps[index]),
            'end': float(timestamps[index + 1]),
            'duration_ms': round(float(gaps[index]), 3),
            'start_timestamp': str(ordered['timestamp'].iloc[index]).strip(),
            'end_timestamp': str(ordered['timestamp'].iloc[index + 1]).strip(),
        })
    return result


def extract_radar_trajectory(
    radar_df: pd.DataFrame,
    track_id: int,
    file_index: int | None,
    segment_index: int | None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """提取一个确定的雷达 Track_Age 生命周期段，供诊断与最终对齐使用。"""
    cfg = _config(config)
    for candidate_id, candidate_file, candidate_segment, segment in _radar_trajectory_groups(radar_df, cfg):
        if (candidate_id == track_id and candidate_file == file_index
                and (segment_index is None or candidate_segment == segment_index)):
            return segment
    raise ValueError(f'未找到雷达轨迹 ID={track_id}, file={file_index}, segment={segment_index}')


def _rtk_target_groups(rtk_df: pd.DataFrame) -> list[tuple[int, int | None, pd.DataFrame]]:
    """按来源文件和真值 ID 分开 RTK 轨迹，防止跨测试时段被全局合并。"""
    has_file_index = 'file_index' in rtk_df.columns
    group_keys = ['file_index', 'ID'] if has_file_index else ['ID']
    results: list[tuple[int, int | None, pd.DataFrame]] = []
    for group_key, group in rtk_df.groupby(group_keys, sort=True):
        if has_file_index:
            file_index, target_id = group_key
            file_index = int(file_index)
        else:
            target_id = group_key[0] if isinstance(group_key, tuple) else group_key
            file_index = None
        results.append((int(target_id), file_index, group.sort_values('timestamp_parsed').reset_index(drop=True)))
    return results


def _movement_metrics(track_df: pd.DataFrame) -> dict[str, float]:
    """使用 P90 速度/加速度和相对起点位移抑制静止目标的瞬时噪声。"""
    frames = len(track_df)
    timestamps = track_df['timestamp_parsed'].to_numpy(dtype=float)

    vx = pd.to_numeric(track_df['Vx'], errors='coerce').to_numpy(dtype=float)
    vy = pd.to_numeric(track_df['Vy'], errors='coerce').to_numpy(dtype=float)
    speed = np.hypot(vx, vy)
    valid_speed = speed[np.isfinite(speed)]
    speed_p90 = float(np.percentile(valid_speed, 90)) if len(valid_speed) else 0.0

    accel_p90 = 0.0
    if frames >= 2:
        dt = np.diff(timestamps)
        dvx = np.diff(vx)
        dvy = np.diff(vy)
        valid_dt = np.isfinite(dt) & (dt > 0) & np.isfinite(dvx) & np.isfinite(dvy)
        if valid_dt.any():
            acceleration = np.hypot(dvx[valid_dt] / dt[valid_dt], dvy[valid_dt] / dt[valid_dt])
            accel_p90 = float(np.percentile(acceleration, 90)) if len(acceleration) else 0.0

    x = pd.to_numeric(track_df['Dx'], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(track_df['Dy'], errors='coerce').to_numpy(dtype=float)
    valid_pos = np.isfinite(x) & np.isfinite(y)
    displacement = 0.0
    if valid_pos.any():
        coords = np.column_stack([x[valid_pos], y[valid_pos]])
        displacement = float(np.max(np.hypot(coords[:, 0] - coords[0, 0], coords[:, 1] - coords[0, 1])))

    return {
        'frames': frames,
        'speed_p90_mps': speed_p90,
        'accel_p90_mps2': accel_p90,
        'displacement_m': displacement,
    }


def filter_moving_radar_targets(
    radar_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据速度、加速度与位移证据过滤静止或样本不足的雷达轨迹。"""
    cfg = _config(config)
    active_targets: list[dict[str, Any]] = []
    target_stats: list[dict[str, Any]] = []

    for target_id, file_index, segment_index, track_df in _radar_trajectory_groups(radar_df, cfg):
        metrics = _movement_metrics(track_df)
        has_motion = (
            metrics['speed_p90_mps'] >= cfg['min_speed_mps']
            or metrics['accel_p90_mps2'] >= cfg['min_accel_mps2']
            or metrics['displacement_m'] >= cfg['min_displacement_m']
        )
        if metrics['frames'] < cfg['min_frames']:
            status = 'filtered_short_track'
        elif has_motion:
            status = 'active'
        else:
            status = 'filtered_static'

        entry = {
            'track_id': target_id,
            'file_index': file_index,
            'segment_index': segment_index,
            'status': status,
            **metrics,
        }
        target_stats.append(entry)
        if status == 'active':
            active_targets.append({**entry, 'df': track_df})

    return {
        'active_targets': active_targets,
        'target_stats': target_stats,
        'filter_stats': {
            'total_radar_targets': len(target_stats),
            'active_radar_targets': len(active_targets),
            'filtered_static_targets': sum(s['status'] == 'filtered_static' for s in target_stats),
            'filtered_short_tracks': sum(s['status'] == 'filtered_short_track' for s in target_stats),
        },
    }


def _evaluate_pair(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    match_threshold_m: float,
) -> dict[str, Any]:
    """在真实时间重叠区间内插值 RTK，评估一对雷达/真值轨迹。"""
    radar_ts = radar_df['timestamp_parsed'].to_numpy(dtype=float)
    rtk_ts = rtk_df['timestamp_parsed'].to_numpy(dtype=float)
    in_time_range = (radar_ts >= rtk_ts[0]) & (radar_ts <= rtk_ts[-1])
    if not in_time_range.any():
        return {'matched_frames': 0, 'coverage': 0.0, 'rmse_m': float('inf')}

    query_ts = radar_ts[in_time_range]
    rtk_x = np.interp(query_ts, rtk_ts, rtk_df['center_x'].to_numpy(dtype=float))
    rtk_y = np.interp(query_ts, rtk_ts, rtk_df['center_y'].to_numpy(dtype=float))
    radar_x = radar_df.loc[in_time_range, 'Dx'].to_numpy(dtype=float)
    radar_y = radar_df.loc[in_time_range, 'Dy'].to_numpy(dtype=float)
    distances = np.hypot(radar_x - rtk_x, radar_y - rtk_y)
    matched_mask = np.isfinite(distances) & (distances < match_threshold_m)
    matched_frames = int(matched_mask.sum())
    if not matched_frames:
        return {'matched_frames': 0, 'coverage': 0.0, 'rmse_m': float('inf')}
    return {
        'matched_frames': matched_frames,
        'coverage': matched_frames / len(radar_df),
        'rmse_m': float(np.sqrt(np.mean(distances[matched_mask] ** 2))),
    }


def filter_and_match_ids(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    comparison_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """过滤无效雷达 ID，并与 RTK ID 执行时间一致的受限多对一关联。

    输出 ``valid_match_ids`` 可直接用于后续对齐；每项包含 radar ID、来源文件、
    RTK ID、覆盖率和 RMSE。未通过运动过滤或未能可靠关联的目标均保留在统计中。
    """
    comparison_config = comparison_config or {}
    cfg = _config(comparison_config.get('id_matching'))
    filtered = filter_moving_radar_targets(radar_df, cfg)
    active_targets = filtered['active_targets']
    rtk_targets = _rtk_target_groups(rtk_df)

    stats = dict(filtered['filter_stats'])
    stats['total_rtk_targets'] = len(rtk_targets)
    if not active_targets or not rtk_targets:
        stats.update({'matched_target_pairs': 0, 'unmatched_active_targets': len(active_targets)})
        return {**filtered, 'valid_match_ids': [], 'unmatched_active_ids': active_targets, 'filter_stats': stats}

    pair_candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    match_threshold = comparison_config.get('match_threshold', 5.0)
    for radar_index, radar_target in enumerate(active_targets):
        for rtk_index, (rtk_id, rtk_file_index, rtk_track) in enumerate(rtk_targets):
            metrics = _evaluate_pair(radar_target['df'], rtk_track, match_threshold)
            if (metrics['matched_frames'] >= cfg['min_pair_frames']
                    and metrics['coverage'] >= cfg['min_pair_coverage']):
                # 覆盖率作为小惩罚项，使 RMSE 相近时优先保留样本更完整的关联。
                cost = metrics['rmse_m'] + (1.0 - metrics['coverage']) * match_threshold
                pair_candidates.append((cost, radar_index, rtk_index, metrics))

    # 同一 RTK 可在不重叠的时段对应多个雷达生命周期段；但同一时间窗内
    # 仍只允许一个雷达候选占用它，避免把并行目标误配到同一真值目标。
    pair_candidates.sort(key=lambda item: (item[0], -item[3]['coverage'], -item[3]['matched_frames']))
    valid_match_ids: list[dict[str, Any]] = []
    matched_radar_indices: set[int] = set()
    assigned_intervals: dict[tuple[int, int | None], list[tuple[float, float]]] = {}
    for _cost, radar_index, rtk_index, metrics in pair_candidates:
        if radar_index in matched_radar_indices:
            continue
        radar_target = active_targets[radar_index]
        rtk_id, rtk_file_index, _ = rtk_targets[rtk_index]
        track_df = radar_target['df']
        start_ts = float(track_df['timestamp_parsed'].iloc[0])
        end_ts = float(track_df['timestamp_parsed'].iloc[-1])
        rtk_key = (rtk_id, rtk_file_index)
        overlaps_existing = any(
            start_ts <= assigned_end and end_ts >= assigned_start
            for assigned_start, assigned_end in assigned_intervals.get(rtk_key, [])
        )
        if overlaps_existing:
            continue

        start_row = track_df.iloc[0]
        end_row = track_df.iloc[-1]
        gap_intervals = _gap_intervals(track_df, cfg)
        valid_match_ids.append({
            'track_id': radar_target['track_id'],
            'file_index': radar_target['file_index'],
            'segment_index': radar_target['segment_index'],
            'rtk_id': rtk_id,
            'rtk_file_index': rtk_file_index,
            'total_frames': radar_target['frames'],
            'matched_frames': metrics['matched_frames'],
            'overlap_rate': round(metrics['coverage'], 4),
            'mean_distance': round(metrics['rmse_m'], 4),
            'time_start': start_ts,
            'time_end': end_ts,
            'time_start_str': str(start_row.get('timestamp', '')).strip() or None,
            'time_end_str': str(end_row.get('timestamp', '')).strip() or None,
            'gap_count': len(gap_intervals),
            'gap_duration_ms': round(sum(item['duration_ms'] for item in gap_intervals), 3),
            'gap_intervals': gap_intervals,
            'motion_metrics': {
                'speed_p90_mps': round(radar_target['speed_p90_mps'], 4),
                'accel_p90_mps2': round(radar_target['accel_p90_mps2'], 4),
                'displacement_m': round(radar_target['displacement_m'], 4),
            },
        })
        matched_radar_indices.add(radar_index)
        assigned_intervals.setdefault(rtk_key, []).append((start_ts, end_ts))

    unmatched_active = [target for index, target in enumerate(active_targets) if index not in matched_radar_indices]
    valid_match_ids.sort(key=lambda item: (-item['overlap_rate'], item['mean_distance']))
    stats.update({
        'matched_target_pairs': len(valid_match_ids),
        'unmatched_active_targets': len(unmatched_active),
    })
    return {
        **filtered,
        'valid_match_ids': valid_match_ids,
        'unmatched_active_ids': unmatched_active,
        'filter_stats': stats,
    }
