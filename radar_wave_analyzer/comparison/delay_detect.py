"""
时间延迟检测模块。
扫描法：遍历不同假设延迟，找到使RMSE最小的延迟值。
"""
import numpy as np
import pandas as pd


def _prepare_delay_arrays(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    track_id: int,
    file_index: int = None,
) -> dict:
    """一次性筛选、排序并提取延迟扫描需要的 NumPy 数组。"""
    if file_index is not None:
        id_mask = (radar_df['ID'] == track_id) & (radar_df['file_index'] == file_index)
    else:
        id_mask = radar_df['ID'] == track_id
    id_df = radar_df.loc[id_mask].sort_values('timestamp_parsed')
    ordered_rtk = rtk_df.sort_values('timestamp_parsed')
    return {
        'radar_ts': id_df['timestamp_parsed'].to_numpy(dtype=float),
        'radar_x': id_df['Dx'].to_numpy(dtype=float),
        'radar_y': id_df['Dy'].to_numpy(dtype=float),
        'rtk_ts': ordered_rtk['timestamp_parsed'].to_numpy(dtype=float),
        'rtk_x': ordered_rtk['center_x'].to_numpy(dtype=float),
        'rtk_y': ordered_rtk['center_y'].to_numpy(dtype=float),
    }


def _compute_delay_metrics_arrays(
    arrays: dict,
    delay_sec: float,
    match_threshold_m: float,
) -> dict:
    """使用已准备的数组计算一个延迟候选，避免循环内重复复制 DataFrame。"""
    radar_ts = arrays['radar_ts']
    rtk_ts = arrays['rtk_ts']
    total_frames = len(radar_ts)
    if total_frames == 0 or len(rtk_ts) == 0:
        return {
            'rmse': float('inf'), 'matched_frames': 0,
            'total_frames': total_frames, 'match_rate': 0.0,
        }

    compensated = radar_ts + delay_sec
    cx_interp = np.interp(
        compensated, rtk_ts, arrays['rtk_x'],
        left=arrays['rtk_x'][0], right=arrays['rtk_x'][-1],
    )
    cy_interp = np.interp(
        compensated, rtk_ts, arrays['rtk_y'],
        left=arrays['rtk_y'][0], right=arrays['rtk_y'][-1],
    )
    dist_err = np.hypot(arrays['radar_x'] - cx_interp, arrays['radar_y'] - cy_interp)
    in_rtk_range = (compensated >= rtk_ts[0]) & (compensated <= rtk_ts[-1])
    gated_mask = (dist_err < match_threshold_m) & in_rtk_range
    matched_frames = int(gated_mask.sum())
    if matched_frames == 0:
        return {
            'rmse': float('inf'), 'matched_frames': 0,
            'total_frames': total_frames, 'match_rate': 0.0,
        }
    return {
        'rmse': float(np.sqrt(np.mean(dist_err[gated_mask] ** 2))),
        'matched_frames': matched_frames,
        'total_frames': total_frames,
        'match_rate': matched_frames / total_frames,
    }


def scan_delay(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    track_id: int,
    delay_range: tuple = (-200, 200),
    step_ms: int = 10,
    insensitive_ratio: float = 0.05,
    match_threshold_m: float = 5.0,
    file_index: int = None,
    min_matched_frames: int = 3,
    min_match_rate: float = 0.5,
) -> dict:
    """扫描法检测最优时间延迟。

    算法（来自文档 Step⑥）:
        对于 d ∈ [d_min, d_max], 步长 step_ms (毫秒):
        1. 雷达时间戳向后偏移 d/1000 秒（d>0）；d<0 时向前偏移
        2. 对RTK做线性插值
        3. 计算 RMSE(d)
        最优延迟 d_opt = argmin(RMSE)

    Args:
        radar_df: 雷达全量数据。
        rtk_df: RTK全量数据。
        track_id: 目标ID。
        delay_range: 扫描范围 (min_ms, max_ms)。
        step_ms: 步长(毫秒)。
        insensitive_ratio: 延迟不敏感阈值，RMSE变化低于此比例视为延迟不敏感。
        file_index: 可选，来源文件序号。多文件场景下用于过滤同ID不同时间段的数据。
        min_matched_frames: 参与最优延迟选择的最小有效匹配帧数。
        min_match_rate: 参与最优延迟选择的最小有效匹配覆盖率（0~1）。

    Returns:
        dict:
            optimal_delay_ms: 最优延迟(ms)
            min_rmse: 最优RMSE
            delay_curve: [(delay_ms, rmse | None), ...]
            delay_samples: 每个候选延迟的 RMSE、匹配帧数和覆盖率。
            delay_insensitive: 延迟是否不敏感
            recommendation: 建议描述
            baseline/optimal: 0ms 与最优候选的 RMSE、匹配帧数及匹配率
    """
    d_min, d_max = delay_range
    delays_ms = list(range(d_min, d_max + 1, step_ms))
    curve = []
    eligible_metrics = []
    delay_samples = []

    arrays = _prepare_delay_arrays(radar_df, rtk_df, track_id, file_index)
    # 共同时间样本：扫描范围内所有候选延迟都能落在 RTK 覆盖区间内。
    # 延迟之间使用同一批样本计算 fixed_rmse，避免 argmin 通过减少样本数获胜。
    radar_ts = arrays['radar_ts']
    rtk_ts = arrays['rtk_ts']
    common_mask = np.zeros(len(radar_ts), dtype=bool)
    if len(rtk_ts):
        common_mask = (
            (radar_ts + d_min / 1000.0 >= rtk_ts[0])
            & (radar_ts + d_max / 1000.0 <= rtk_ts[-1])
        )
    for d_ms in delays_ms:
        metrics = _compute_delay_metrics_arrays(
            arrays, d_ms / 1000.0, match_threshold_m,
        )
        eligible = (
            np.isfinite(metrics['rmse'])
            and metrics['matched_frames'] >= min_matched_frames
            and metrics['match_rate'] >= min_match_rate
        )
        rounded_rmse = round(metrics['rmse'], 5) if np.isfinite(metrics['rmse']) else None
        fixed_rmse = None
        fixed_frames = 0
        if common_mask.any():
            compensated = radar_ts[common_mask] + d_ms / 1000.0
            fixed_x = np.interp(compensated, rtk_ts, arrays['rtk_x'])
            fixed_y = np.interp(compensated, rtk_ts, arrays['rtk_y'])
            fixed_dist = np.hypot(
                arrays['radar_x'][common_mask] - fixed_x,
                arrays['radar_y'][common_mask] - fixed_y,
            )
            finite = np.isfinite(fixed_dist)
            fixed_frames = int(finite.sum())
            if fixed_frames:
                fixed_rmse = round(float(np.sqrt(np.mean(fixed_dist[finite] ** 2))), 5)
        # 主曲线改为固定样本 RMSE；旧的门控曲线保留供诊断。
        curve.append((d_ms, fixed_rmse))
        delay_samples.append({
            'delay_ms': d_ms,
            'rmse': rounded_rmse,
            'gated_rmse': rounded_rmse,
            'fixed_rmse': fixed_rmse,
            'fixed_frames': fixed_frames,
            'matched_frames': metrics['matched_frames'],
            'total_frames': metrics['total_frames'],
            'match_rate': round(metrics['match_rate'], 4),
            'eligible': eligible,
        })
        if eligible and fixed_rmse is not None:
            eligible_metrics.append((d_ms, fixed_rmse))

    if not eligible_metrics:
        return {
            'optimal_delay_ms': 0,
            'min_rmse': None,
            'delay_curve': curve,
            'gated_delay_curve': [
                (item['delay_ms'], item['gated_rmse']) for item in delay_samples
            ],
            'delay_samples': delay_samples,
            'delay_insensitive': True,
            'recommendation': (
                f'有效匹配帧不足，无法可靠检测延迟（至少需要 '
                f'{min_matched_frames} 帧且覆盖率不低于 {min_match_rate:.0%}）'
            ),
            'level': 'insufficient_coverage',
            'baseline': next((item for item in delay_samples
                              if item['delay_ms'] == 0), None),
            'optimal': None,
        }

    eligible_delays = [delay for delay, _ in eligible_metrics]
    rmse_arr = np.array([rmse for _, rmse in eligible_metrics])
    best_idx = int(np.argmin(rmse_arr))
    optimal_delay_ms = eligible_delays[best_idx]
    min_rmse = round(float(rmse_arr[best_idx]), 5)

    # 判断延迟是否不敏感（RMSE变化 < 5%）
    rmse_range = float(np.max(rmse_arr) - np.min(rmse_arr))
    avg_rmse = float(np.mean(rmse_arr))
    variation_ratio = rmse_range / avg_rmse if avg_rmse > 0 else 0
    delay_insensitive = variation_ratio < insensitive_ratio

    if delay_insensitive:
        recommendation = f'RMSE变化 < {insensitive_ratio*100:.0f}%，延迟不敏感。建议 d=0ms'
        level = 'insensitive'
    elif abs(optimal_delay_ms) < 10:
        recommendation = f'最优延迟 {optimal_delay_ms}ms < 10ms，无需补偿'
        level = 'no_compensation'
    else:
        recommendation = f'建议补偿延迟 {optimal_delay_ms}ms（RMSE={min_rmse:.2f}m）'
        level = 'need_compensation'

    baseline = next((item for item in delay_samples if item['delay_ms'] == 0), None)
    optimal = next((item for item in delay_samples
                    if item['delay_ms'] == optimal_delay_ms), None)

    return {
        'optimal_delay_ms': optimal_delay_ms,
        'min_rmse': min_rmse,
        'delay_curve': curve,
        'gated_delay_curve': [
            (item['delay_ms'], item['gated_rmse']) for item in delay_samples
        ],
        'delay_samples': delay_samples,
        'delay_insensitive': delay_insensitive,
        'recommendation': recommendation,
        'level': level,
        'baseline': baseline,
        'optimal': optimal,
    }
