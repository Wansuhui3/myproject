"""
核心对齐算法模块。
一次调用完成: 时间戳解析+延迟补偿 → RTK线性插值 → 逐帧空间匹配 → 误差计算。
"""
import numpy as np
import pandas as pd


def _compute_error_metrics(errors: np.ndarray) -> dict:
    """计算单个误差序列的统计量。

    Returns:
        {mean, std, rmse, max, p50, p95}
    """
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return {'mean': None, 'std': None, 'rmse': None, 'max': None, 'p50': None, 'p95': None}
    abs_valid = np.abs(valid)
    return {
        'mean': round(float(np.mean(valid)), 4),
        'std': round(float(np.std(valid)), 4),
        'rmse': round(float(np.sqrt(np.mean(valid ** 2))), 4),
        'max': round(float(np.max(abs_valid)), 4),
        'p50': round(float(np.percentile(abs_valid, 50)), 4),
        'p95': round(float(np.percentile(abs_valid, 95)), 4),
    }


def _nearest_indices(reference_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """返回每个查询时刻在已排序参考序列中的真正最近索引。

    ``searchsorted`` 返回的是右侧插入点，不等同于最近采样点。帧号显示和
    时间门控必须基于真正最近的 RTK 原始帧，避免把左侧更近的样本误判为超时。
    """
    right = np.searchsorted(reference_ts, query_ts, side='left')
    right = np.clip(right, 0, len(reference_ts) - 1)
    left = np.clip(right - 1, 0, len(reference_ts) - 1)

    choose_left = np.abs(query_ts - reference_ts[left]) <= np.abs(reference_ts[right] - query_ts)
    return np.where(choose_left, left, right)


def align_trajectories(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    track_id: int,
    delay_ms: float = 0.0,
    match_threshold_m: float = 5.0,
    time_gate_ms: float = 50.0,
    file_index: int = None,
    custom_mappings: list[dict] | None = None,
) -> dict:
    """核心对齐算法 — 一次调用完成全部4步。

    Step1: 时间戳解析 + 延迟补偿
        t_compensated = t_radar + delay_ms / 1000

    Step2: RTK线性插值
        对 center_x, center_y, Vx, Vy 逐字段插值到补偿后的雷达时刻
        使用 numpy.interp, 边界用首/末值外推

    Step3: 逐帧空间匹配
        dist = sqrt((Dx - cx_interp)^2 + (Dy - cy_interp)^2)
        dist < match_threshold_m → 匹配成功
        同时检查时间门控 time_gate_ms

    Step4: 误差计算
        pos_error_x  = Dx - center_x_interp
        pos_error_y  = Dy - center_y_interp
        pos_error_abs = sqrt(pos_error_x^2 + pos_error_y^2)
        vel_error_x  = Vx - Vx_interp
        vel_error_y  = Vy - Vy_interp
        vel_error_abs = sqrt(vel_error_x^2 + vel_error_y^2)

    Args:
        radar_df: 雷达全量数据（含 timestamp_parsed, ID, Dx, Dy, Vx, Vy）。
        rtk_df: RTK全量数据（含 timestamp_parsed, center_x, center_y, Vx, Vy）。
        track_id: 选定的雷达目标ID。
        delay_ms: 应用于雷达时间戳的时间补偿（毫秒）；正值表示将雷达时刻向后移动。
        match_threshold_m: 空间匹配阈值(米)。
        time_gate_ms: 时间门控(毫秒)，匹配帧的RTK插值时间差超过此值标记告警。
        file_index: 可选，来源文件序号。多文件场景下用于过滤同ID不同时间段的数据。

    Returns:
        dict:
            aligned_df: pd.DataFrame
                逐帧对齐结果，包含:
                timestamp, radar_Dx, radar_Dy, radar_Vx, radar_Vy,
                rtk_center_x, rtk_center_y, rtk_Vx, rtk_Vy,
                pos_error_x, pos_error_y, pos_error_abs,
                vel_error_x, vel_error_y, vel_error_abs,
                match_dist, is_matched
            summary: dict
                各误差指标的 {mean, std, rmse, max, p50, p95}
            match_summary: dict
                {total_frames, matched_frames, match_rate, track_id, delay_ms}
    """
    # ── 裁取选定ID的雷达数据 ──
    if file_index is not None:
        id_mask = (radar_df['ID'] == track_id) & (radar_df['file_index'] == file_index)
    else:
        id_mask = radar_df['ID'] == track_id
    seg_df = radar_df[id_mask].copy().sort_values('timestamp_parsed').reset_index(drop=True)
    rtk_df = rtk_df.sort_values('timestamp_parsed').reset_index(drop=True)
    total_frames = len(seg_df)

    if total_frames == 0 or len(rtk_df) == 0:
        empty = pd.DataFrame()
        return {
            'aligned_df': empty,
            'summary': {},
            'match_summary': {
                'total_frames': total_frames,
                'matched_frames': 0,
                'match_rate': 0.0,
                'track_id': track_id,
                'delay_ms': delay_ms,
            },
        }

    # ── Step1: 时间戳解析 + 延迟补偿 ──
    delay_sec = delay_ms / 1000.0
    t_compensated = seg_df['timestamp_parsed'].values + delay_sec
    rtk_ts = rtk_df['timestamp_parsed'].values

    # 帧号 & CSV真实时间戳 ────────────────────────────────────────────
    # rtk_idx: 每个雷达对齐时刻真正最近的 RTK 样本索引。
    rtk_idx = _nearest_indices(rtk_ts, t_compensated)

    # 雷达帧号（1-based）：对应原始CSV行号（parser加载时写入csv_row），fallback到轨迹内序号
    if 'csv_row' in seg_df.columns:
        radar_frame = seg_df['csv_row'].values
    else:
        radar_frame = np.arange(len(seg_df)) + 1

    # 真值帧号：最近RTK样本的CSV行号
    if 'csv_row' in rtk_df.columns:
        rtk_frame = rtk_df['csv_row'].values[rtk_idx]
    else:
        rtk_frame = rtk_idx + 1

    # CSV真实时间戳（epoch秒）：供hover显示原始帧时间
    radar_ts_parsed = seg_df['timestamp_parsed'].values
    rtk_nearest_ts_parsed = rtk_ts[rtk_idx]

    # ── Step2: RTK线性插值 ──
    interp_fields = ['center_x', 'center_y', 'Vx', 'Vy']
    interp_values = {}
    for field in interp_fields:
        rtk_vals = rtk_df[field].values
        # numpy.interp 支持向量化插值，边界外推取首/末值
        interp_values[field] = np.interp(
            t_compensated, rtk_ts, rtk_vals,
            left=rtk_vals[0], right=rtk_vals[-1],
        )

    cx_interp = interp_values['center_x']
    cy_interp = interp_values['center_y']
    vx_interp = interp_values['Vx']
    vy_interp = interp_values['Vy']

    # RTK 覆盖范围外不做首尾常数外推。保留 NaN，避免把伪造的真值
    # 带入图表或任何后续统计；这些帧由 within_rtk_range 单独计数。
    within_rtk_range = (
        (t_compensated >= rtk_ts[0]) & (t_compensated <= rtk_ts[-1])
    )
    for values in (cx_interp, cy_interp, vx_interp, vy_interp):
        values[~within_rtk_range] = np.nan

    # ── Step3: 逐帧空间匹配 ──
    dx_err = seg_df['Dx'].values - cx_interp
    dy_err = seg_df['Dy'].values - cy_interp
    match_dist = np.sqrt(dx_err ** 2 + dy_err ** 2)
    spatial_matched = match_dist < match_threshold_m

    # ── Time-gate: 最近 RTK 样本的时间差超过阈值或超出 RTK 采样范围则剔除 ──
    time_diff_ms = np.abs(t_compensated - rtk_ts[rtk_idx]) * 1000.0
    time_gated = (
        (time_diff_ms <= time_gate_ms)
        & within_rtk_range
        & np.isfinite(cx_interp)
        & np.isfinite(cy_interp)
    )
    is_matched = spatial_matched & time_gated

    # ── Step4: 误差计算 ──
    vel_error_x = seg_df['Vx'].values - vx_interp
    vel_error_y = seg_df['Vy'].values - vy_interp
    vel_error_abs = np.sqrt(vel_error_x ** 2 + vel_error_y ** 2)

    # ── 构建结果 DataFrame ──
    aligned_df = pd.DataFrame({
        'timestamp': seg_df['timestamp'].values,
        'timestamp_parsed': seg_df['timestamp_parsed'].values,
        'radar_Dx': seg_df['Dx'].values,
        'radar_Dy': seg_df['Dy'].values,
        'radar_Vx': seg_df['Vx'].values,
        'radar_Vy': seg_df['Vy'].values,
        'rtk_center_x': np.round(cx_interp, 4),
        'rtk_center_y': np.round(cy_interp, 4),
        'rtk_Vx': np.round(vx_interp, 4),
        'rtk_Vy': np.round(vy_interp, 4),
        'pos_error_x': np.round(dx_err, 4),
        'pos_error_y': np.round(dy_err, 4),
        'pos_error_abs': np.round(match_dist, 4),
        'vel_error_x': np.round(vel_error_x, 4),
        'vel_error_y': np.round(vel_error_y, 4),
        'vel_error_abs': np.round(vel_error_abs, 4),
        'match_dist': np.round(match_dist, 4),
        'radar_frame': radar_frame,
        'rtk_frame': rtk_frame,
        'radar_ts_parsed': radar_ts_parsed,
        'rtk_nearest_ts_parsed': rtk_nearest_ts_parsed,
        'time_diff_ms': np.round(time_diff_ms, 2),
        'spatial_matched': spatial_matched,
        'within_rtk_range': within_rtk_range,
        'within_time_gate': time_gated,
        'is_matched': is_matched,
    })

    # 自定义显示映射：列名保持 CSV 原始表头。目标关联仍使用标准 Dx/Dy，
    # 自定义通道只负责显示与兼容单位下的误差统计，不改变匹配结论。
    mapping_results = []
    for mapping in custom_mappings or []:
        radar_col = mapping.get('radar_col')
        rtk_col = mapping.get('rtk_col')
        has_radar = bool(radar_col) and radar_col in seg_df.columns
        has_rtk = bool(rtk_col) and rtk_col in rtk_df.columns
        if not has_radar and not has_rtk:
            continue
        radar_values = (
            pd.to_numeric(seg_df[radar_col], errors='coerce').to_numpy(dtype=float)
            if has_radar else np.full(len(t_compensated), np.nan, dtype=float)
        )
        rtk_values = np.full(len(t_compensated), np.nan, dtype=float)
        if has_rtk:
            raw_rtk_values = pd.to_numeric(rtk_df[rtk_col], errors='coerce').to_numpy(dtype=float)
            valid_rtk = np.isfinite(rtk_ts) & np.isfinite(raw_rtk_values)
            if not valid_rtk.any():
                # 真值单线没有有效数值时不能绘制；雷达单线仍可正常显示。
                if not has_radar:
                    continue
            else:
                valid_rtk_ts = rtk_ts[valid_rtk]
                valid_rtk_values = raw_rtk_values[valid_rtk]
                if len(valid_rtk_values) == 1:
                    rtk_values = np.full(len(t_compensated), valid_rtk_values[0], dtype=float)
                else:
                    rtk_values = np.interp(
                        t_compensated, valid_rtk_ts, valid_rtk_values,
                        left=valid_rtk_values[0], right=valid_rtk_values[-1],
                    )
                rtk_values[~within_rtk_range] = np.nan
        error_values = radar_values - rtk_values
        uid = str(mapping.get('uid', f'mapping-{len(mapping_results) + 1}'))
        radar_output = f'radar[{radar_col}]' if has_radar else ''
        rtk_output = f'rtk[{rtk_col}]' if has_rtk else ''
        error_output = f'error[{radar_col}-{rtk_col}]' if has_radar and has_rtk else ''
        if radar_output:
            aligned_df[radar_output] = np.round(radar_values, 6)
        if rtk_output:
            aligned_df[rtk_output] = np.round(rtk_values, 6)
        if error_output:
            aligned_df[error_output] = np.round(error_values, 6)

        valid_stats = is_matched & np.isfinite(error_values)
        metrics = None
        if mapping.get('stats_enabled') and valid_stats.any():
            metrics = _compute_error_metrics(error_values[valid_stats])
        mapping_results.append({
            **mapping,
            'uid': uid,
            'radar_output_col': radar_output,
            'rtk_output_col': rtk_output,
            'error_output_col': error_output,
            'rtk_curve_col': rtk_col if has_rtk else '',
            'metrics': metrics,
            'valid_frames': int(valid_stats.sum()),
        })

    matched_frames = int(is_matched.sum())

    # ── 汇总统计 ──
    # 质量指标只使用通过空间阈值和时间门控的有效匹配帧。未匹配帧只反映在
    # 匹配率及拒绝原因中，不能混入 RMSE/P95 等结果而误导算法质量判断。
    matched_df = aligned_df.loc[aligned_df['is_matched']]
    # 时间有效样本保留空间门控拒绝帧，用于暴露整体误差；它仍然只包含
    # 有 RTK 真值且满足时间门控的配对帧，不会把无真值帧硬算进 RMSE。
    time_valid = aligned_df['within_time_gate'] & aligned_df['within_rtk_range']
    time_valid_df = aligned_df.loc[time_valid]
    summary = {
        'pos_error_x': _compute_error_metrics(matched_df['pos_error_x'].values),
        'pos_error_y': _compute_error_metrics(matched_df['pos_error_y'].values),
        'pos_error_abs': _compute_error_metrics(matched_df['pos_error_abs'].values),
        'vel_error_x': _compute_error_metrics(matched_df['vel_error_x'].values),
        'vel_error_y': _compute_error_metrics(matched_df['vel_error_y'].values),
        'vel_error_abs': _compute_error_metrics(matched_df['vel_error_abs'].values),
        'pos_error_abs_time_valid': _compute_error_metrics(
            time_valid_df['pos_error_abs'].values,
        ),
        'pos_error_x_time_valid': _compute_error_metrics(
            time_valid_df['pos_error_x'].values,
        ),
        'pos_error_y_time_valid': _compute_error_metrics(
            time_valid_df['pos_error_y'].values,
        ),
    }

    match_summary = {
        'total_frames': total_frames,
        'matched_frames': matched_frames,
        'match_rate': round(matched_frames / total_frames, 4) if total_frames > 0 else 0.0,
        'track_id': track_id,
        'delay_ms': delay_ms,
        'time_valid_frames': int(time_valid.sum()),
        'spatial_rejected_frames': int(
            (within_rtk_range & time_gated & ~spatial_matched).sum()
        ),
        'time_rejected_frames': int((within_rtk_range & ~time_gated).sum()),
        'out_of_rtk_range_frames': int((~within_rtk_range).sum()),
    }

    # 图表渲染不能只复用雷达时间点上的插值值：当雷达目标中断时，这会让
    # RTK 真值曲线同步消失。保留选中 RTK 轨迹的原始连续采样用于独立绘制。
    rtk_curve_columns = [
        column for column in ['timestamp', 'timestamp_parsed', 'center_x', 'center_y', 'Vx', 'Vy']
        if column in rtk_df.columns
    ]
    rtk_curve_df = rtk_df[rtk_curve_columns].copy()

    return {
        'aligned_df': aligned_df,
        'summary': summary,
        'match_summary': match_summary,
        'rtk_curve_df': rtk_curve_df,
        'mapping_results': mapping_results,
    }
