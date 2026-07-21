"""
时间延迟检测模块。
扫描法：遍历不同假设延迟，找到使RMSE最小的延迟值。
"""
import numpy as np
import pandas as pd


def _compute_rmse_for_delay(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    track_id: int,
    delay_sec: float,
    match_threshold_m: float = 5.0,
    file_index: int = None,
) -> float:
    """对给定延迟值，计算雷达与RTK插值后的位置RMSE（仅含空间门控匹配的帧）。

    Args:
        delay_sec: 延迟值(秒)，正=RTK比雷达晚。
        match_threshold_m: 空间匹配阈值(米)，用于门控后再算RMSE。
        file_index: 可选，来源文件序号。多文件场景下用于过滤同ID不同时间段的数据。
    """
    return _compute_delay_metrics(
        radar_df, rtk_df, track_id, delay_sec,
        match_threshold_m=match_threshold_m,
        file_index=file_index,
    )['rmse']


def _compute_delay_metrics(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    track_id: int,
    delay_sec: float,
    match_threshold_m: float = 5.0,
    file_index: int = None,
) -> dict:
    """计算一个延迟候选的 RMSE 和有效样本覆盖率。"""
    if file_index is not None:
        id_mask = (radar_df['ID'] == track_id) & (radar_df['file_index'] == file_index)
    else:
        id_mask = radar_df['ID'] == track_id
    id_df = radar_df[id_mask].copy().sort_values('timestamp_parsed')
    if len(id_df) == 0:
        return {'rmse': float('inf'), 'matched_frames': 0, 'total_frames': 0, 'match_rate': 0.0}
    if len(rtk_df) == 0:
        return {
            'rmse': float('inf'), 'matched_frames': 0,
            'total_frames': len(id_df), 'match_rate': 0.0,
        }

    rtk_df = rtk_df.sort_values('timestamp_parsed')

    # 延迟补偿后的雷达时间戳
    t_compensated = id_df['timestamp_parsed'].values + delay_sec

    # RTK 时间戳
    rtk_ts = rtk_df['timestamp_parsed'].values

    # 对 center_x, center_y 做线性插值
    cx_interp = np.interp(t_compensated, rtk_ts, rtk_df['center_x'].values,
                          left=rtk_df['center_x'].iloc[0],
                          right=rtk_df['center_x'].iloc[-1])
    cy_interp = np.interp(t_compensated, rtk_ts, rtk_df['center_y'].values,
                          left=rtk_df['center_y'].iloc[0],
                          right=rtk_df['center_y'].iloc[-1])

    dx_err = id_df['Dx'].values - cx_interp
    dy_err = id_df['Dy'].values - cy_interp
    dist_err = np.sqrt(dx_err ** 2 + dy_err ** 2)

    # 边界外推会掩盖没有真值的事实，延迟估计不接受 RTK 范围外的帧。
    in_rtk_range = (t_compensated >= rtk_ts[0]) & (t_compensated <= rtk_ts[-1])
    gated_mask = (dist_err < match_threshold_m) & in_rtk_range
    total_frames = len(id_df)
    matched_frames = int(np.sum(gated_mask))
    if not np.any(gated_mask):
        return {
            'rmse': float('inf'), 'matched_frames': 0,
            'total_frames': total_frames, 'match_rate': 0.0,
        }

    return {
        'rmse': float(np.sqrt(np.mean(dist_err[gated_mask] ** 2))),
        'matched_frames': matched_frames,
        'total_frames': total_frames,
        'match_rate': matched_frames / total_frames if total_frames > 0 else 0.0,
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
        1. 雷达时间戳偏移 d/1000 秒
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
    """
    d_min, d_max = delay_range
    delays_ms = list(range(d_min, d_max + 1, step_ms))
    curve = []
    eligible_metrics = []
    delay_samples = []

    for d_ms in delays_ms:
        metrics = _compute_delay_metrics(
            radar_df, rtk_df, track_id, d_ms / 1000.0,
            match_threshold_m=match_threshold_m,
            file_index=file_index,
        )
        eligible = (
            np.isfinite(metrics['rmse'])
            and metrics['matched_frames'] >= min_matched_frames
            and metrics['match_rate'] >= min_match_rate
        )
        rounded_rmse = round(metrics['rmse'], 5) if np.isfinite(metrics['rmse']) else None
        curve.append((d_ms, rounded_rmse))
        delay_samples.append({
            'delay_ms': d_ms,
            'rmse': rounded_rmse,
            'matched_frames': metrics['matched_frames'],
            'total_frames': metrics['total_frames'],
            'match_rate': round(metrics['match_rate'], 4),
            'eligible': eligible,
        })
        if eligible:
            eligible_metrics.append((d_ms, metrics['rmse']))

    if not eligible_metrics:
        return {
            'optimal_delay_ms': 0,
            'min_rmse': None,
            'delay_curve': curve,
            'delay_samples': delay_samples,
            'delay_insensitive': True,
            'recommendation': (
                f'有效匹配帧不足，无法可靠检测延迟（至少需要 '
                f'{min_matched_frames} 帧且覆盖率不低于 {min_match_rate:.0%}）'
            ),
            'level': 'insufficient_coverage',
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

    return {
        'optimal_delay_ms': optimal_delay_ms,
        'min_rmse': min_rmse,
        'delay_curve': curve,
        'delay_samples': delay_samples,
        'delay_insensitive': delay_insensitive,
        'recommendation': recommendation,
        'level': level,
    }
