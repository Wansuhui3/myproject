"""性能指标单物理量评估：距离分桶汇总、逐帧判定与总体结论。

阈值/距离段配置由调用方传入（evaluate_all_metrics 编排层负责读取配置），
本模块只做纯计算，不依赖 Dash。口径见设计文档 5.2 / 7.4 / 7.5 / 7.6。
"""
from typing import Any

import numpy as np
import pandas as pd

from .performance_common import (
    STATUS_FAIL,
    STATUS_INSUFFICIENT,
    STATUS_PASS,
    STATUS_UNDECIDABLE,
    _TRUTH_DISTANCE_COL,
    _assign_bins,
    _compute_continuity_breaks,
    _resolve_column,
    _resolve_limit,
    _rolling_window_min,
)


def _build_empty_bins(rules: dict, bins: list, include_last_upper: bool,
                      accuracy_requirement: float) -> list[dict]:
    """构造无数据时的区间行：binned 模式六个距离段，single_limit 模式单行。"""
    if rules.get('mode') == 'single_limit':
        return [{
            'distance_bin': '完整曲线',
            'sample_count': 0,
            'rmse': None,
            'normal_pass_count': 0,
            'accuracy': None,
            'accuracy_requirement': None,
            'accuracy_pass': None,
            'three_frame_index': None,
            'three_frame_pass': None,
            'violation_window_count': 0,
            'longest_violation_run': 0,
            'worst_window_start': None,
            'worst_window_end': None,
            'max_error_in_worst_window': None,
            'status': STATUS_UNDECIDABLE,
            'failure_reasons': [],
        }]

    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        suffix = ']' if (include_last_upper and i == len(bins) - 2) else ')'
        rows.append({
            'distance_bin': f'[{lo}, {hi}{suffix}',
            'sample_count': 0,
            'rmse': None,
            'normal_pass_count': 0,
            'accuracy': None,
            'accuracy_requirement': accuracy_requirement,
            'accuracy_pass': None,
            'three_frame_index': None,
            'three_frame_pass': None,
            'violation_window_count': 0,
            'longest_violation_run': 0,
            'worst_window_start': None,
            'worst_window_end': None,
            'max_error_in_worst_window': None,
            'status': STATUS_UNDECIDABLE,
            'failure_reasons': [],
        })
    return rows


def evaluate_metric(
    aligned_df: pd.DataFrame,
    metric: str,
    rules: dict,
    global_config: dict,
    *,
    track_id: Any = None,
    segment_id: Any = None,
) -> dict:
    """评估单个物理量的性能指标。

    Args:
        aligned_df: 对齐结果 DataFrame（完整原始帧，不可降采样）。
        metric: 物理量名称，如 'Dx'。
        rules: 该物理量的规则 dict（来自 performance_metrics.metrics）。
        global_config: performance_metrics 的全局参数。
        track_id: 目标标识，仅用于逐帧结果标注。
        segment_id: 轨迹段标识，仅用于逐帧结果标注。

    Returns:
        {
          'metric': str,
          'available': bool,
          'reason': str | None,        # 不可用原因
          'mode': 'binned' | 'single_limit',
          'frames': DataFrame | None,  # 逐帧明细
          'bins': list[dict],          # 区间汇总
          'overall': dict,             # 总体结论
          'coverage': dict,            # 匹配覆盖率
        }
    """
    result: dict[str, Any] = {
        'metric': metric,
        'available': False,
        'reason': None,
        'mode': rules.get('mode', 'binned'),
        'frames': None,
        'bins': [],
        'overall': {},
        'coverage': {},
    }

    if aligned_df is None or len(aligned_df) == 0:
        result['reason'] = '无对齐数据'
        return result

    # ── 列解析：固定列优先，映射生成的 Ax/Ay 走候选回退 ──
    radar_col = _resolve_column(
        aligned_df, rules.get('radar_col'), rules.get('radar_col_fallbacks'))
    truth_col = _resolve_column(
        aligned_df, rules.get('truth_col'), rules.get('truth_col_fallbacks'))
    if radar_col is None or truth_col is None:
        result['reason'] = f'缺少必需字段（雷达列={rules.get("radar_col")}，真值列={rules.get("truth_col")}）'
        return result

    if _TRUTH_DISTANCE_COL not in aligned_df.columns:
        result['reason'] = f'缺少 RTK 真实纵向距离列 {_TRUTH_DISTANCE_COL}'
        return result

    bins = list(global_config.get('distance_bins') or [])
    if len(bins) < 2:
        result['reason'] = '距离分段配置无效'
        return result

    include_last_upper = bool(global_config.get('include_last_upper', True))
    valid_range = global_config.get('valid_distance_range') or [bins[0], bins[-1]]
    accuracy_requirement = float(global_config.get('accuracy_requirement', 0.9545))
    three_req = float(global_config.get('three_frame_requirement', 1.0))
    min_samples = int(global_config.get('min_samples_for_conclusion', 3))
    gap_factor = float(global_config.get('gap_break_factor', 2.5))
    gap_percentile = float(global_config.get('gap_percentile', 25))

    # ── 匹配覆盖率：基于全部原始帧，不参与通过判定 ──
    total_frames = len(aligned_df)
    if 'is_matched' in aligned_df.columns:
        matched_all = aligned_df['is_matched'].to_numpy(dtype=bool)
    else:
        matched_all = np.ones(total_frames, dtype=bool)
    matched_count = int(matched_all.sum())
    result['coverage'] = {
        'total_frames': total_frames,
        'matched_frames': matched_count,
        'match_rate': (matched_count / total_frames) if total_frames else None,
    }

    # ── 有效样本筛选（设计文档 5.2）──
    radar_val = pd.to_numeric(aligned_df[radar_col], errors='coerce').to_numpy(dtype=float)
    truth_val = pd.to_numeric(aligned_df[truth_col], errors='coerce').to_numpy(dtype=float)
    truth_dist = pd.to_numeric(
        aligned_df[_TRUTH_DISTANCE_COL], errors='coerce').to_numpy(dtype=float)
    truth_dist = np.abs(truth_dist)

    finite = (np.isfinite(radar_val) & np.isfinite(truth_val)
              & np.isfinite(truth_dist) & matched_all)
    in_range = ((truth_dist >= valid_range[0]) & (truth_dist <= valid_range[1]))
    valid = finite & in_range

    if not valid.any():
        # 无有效样本时仍输出完整区间行，状态为“不可判定”，
        # 保证 UI 始终显示六个距离段且不会误显示“通过”（设计文档 13）。
        result['available'] = True
        result['reason'] = '无有效样本'
        result['bins'] = _build_empty_bins(rules, bins, include_last_upper,
                                           accuracy_requirement)
        result['overall'] = {
            'status': STATUS_UNDECIDABLE,
            'total_samples': 0,
            'passed_bins': 0,
            'decidable_bins': 0,
            'failed_bins': 0,
            'violation_windows': 0,
            'reasons': ['无有效样本'],
        }
        return result

    signed_error = radar_val - truth_val
    abs_error = np.abs(signed_error)
    bin_index = _assign_bins(truth_dist, bins, include_last_upper)
    # 区间外的样本不参与任何统计
    valid &= (bin_index >= 0)

    # 时间戳（用于连续帧与中断判定）
    if 'timestamp_parsed' in aligned_df.columns:
        timestamps = pd.to_numeric(
            aligned_df['timestamp_parsed'], errors='coerce').to_numpy(dtype=float)
    else:
        timestamps = np.arange(len(aligned_df), dtype=float)

    # ── Ay：单项限值，不分段（设计文档 8.6）──
    if rules.get('mode') == 'single_limit':
        return _evaluate_single_limit(
            result, aligned_df, metric, rules, radar_col, truth_col,
            radar_val, truth_val, signed_error, abs_error, valid,
            track_id, segment_id,
        )

    # ── 阈值向量化 ──
    basis_mode = rules.get('percent_basis', 'truth')
    if basis_mode == 'distance':
        basis = truth_dist
    else:
        basis = np.abs(truth_val)

    normal_limit = np.full(len(aligned_df), np.nan, dtype=float)
    three_limit = np.full(len(aligned_df), np.nan, dtype=float)
    rule_bins = rules.get('bins') or []

    for i, entry in enumerate(rule_bins):
        span = entry.get('range') or []
        if len(span) != 2:
            continue
        lo, hi = span
        if i == len(rule_bins) - 1 and include_last_upper:
            in_bin = (truth_dist >= lo) & (truth_dist <= hi)
        else:
            in_bin = (truth_dist >= lo) & (truth_dist < hi)
        if not in_bin.any():
            continue
        # 百分比基准仅在当前区间内有效，避免区间外样本污染
        local_basis = np.where(in_bin, basis, np.nan)
        normal_limit = np.where(
            in_bin, _resolve_limit(entry.get('normal_limit'), local_basis), normal_limit)
        three_limit = np.where(
            in_bin, _resolve_limit(entry.get('three_frame_limit'), local_basis), three_limit)

    # ── 逐帧判定 ──
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.where(three_limit > 0, abs_error / three_limit, np.nan)

    normal_pass = valid & np.isfinite(normal_limit) & (abs_error < normal_limit)

    # ── 连续三帧窗口（设计文档 7.4 / 7.5）──
    breaks = _compute_continuity_breaks(timestamps, gap_factor, gap_percentile, valid)
    # 连续段：breaks[i] 为 True 表示与前一帧不连续 → 新段起点
    segment_id_arr = np.cumsum(breaks) - 1

    window_min_all = np.full(len(aligned_df), np.nan, dtype=float)
    window_valid = np.zeros(len(aligned_df), dtype=bool)
    # 窗口归属：中间帧索引 = 起点 + 1（设计文档 7.5：跨段窗口归入中间帧所在段）
    window_owner = np.full(len(aligned_df), -1, dtype=int)
    # 中间帧索引 → (起点, 终点)，供最严重窗口定位，避免线性查找
    window_span_by_middle: dict[int, tuple[int, int]] = {}

    for seg_no in np.unique(segment_id_arr):
        positions = np.flatnonzero(segment_id_arr == seg_no)
        if len(positions) < 3:
            continue
        seg_norm = normalized[positions]
        seg_valid = valid[positions] & np.isfinite(seg_norm)
        # 窗口内存在无效帧时结果为 NaN，该窗口不参与统计
        mins = _rolling_window_min(
            np.where(seg_valid, seg_norm, np.nan), window=3)
        # mins 与 positions 等长，mins[j] 对应窗口 [j-2, j]，故从 j=2 起有效。
        # 窗口只输出在中间帧：连续段首帧、末帧无完整三帧窗口，不输出
        # （窗口 (1,2,3) 已归属帧 2，首帧不再借用）。
        for j in range(2, len(positions)):
            value = mins[j]
            if not np.isfinite(value):
                continue
            start, middle, end = (
                int(positions[j - 2]), int(positions[j - 1]), int(positions[j]))
            window_min_all[middle] = float(value)
            window_valid[middle] = True
            window_owner[middle] = bin_index[middle]
            window_span_by_middle[middle] = (start, end)

    frames = pd.DataFrame({
        'track_id': track_id,
        'segment_id': segment_id,
        'radar_frame': (aligned_df['radar_frame'].values
                        if 'radar_frame' in aligned_df.columns
                        else np.arange(len(aligned_df))),
        'timestamp': (aligned_df['timestamp'].values
                      if 'timestamp' in aligned_df.columns else None),
        'distance_bin': bin_index,
        'truth_distance': truth_dist,
        'radar_value': radar_val,
        'truth_value': truth_val,
        'signed_error': signed_error,
        'abs_error': abs_error,
        'normal_limit': normal_limit,
        'normal_pass': normal_pass,
        'three_frame_limit': three_limit,
        'normalized_severe_error': window_min_all,
        'three_frame_violation': window_valid & (window_min_all >= three_req),
        # 与前一有效帧的连续性：True 表示中断（时间间隔超阈值），导出层
        # 用它让 Excel 公式正确处理“中断处窗口重新起算/不输出”
        'continuity_break': breaks,
        'valid': valid,
    })
    result['frames'] = frames

    # ── 区间汇总 ──
    bin_results = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        suffix = ']' if (include_last_upper and i == len(bins) - 2) else ')'
        label = f'[{lo}, {hi}{suffix}'
        member = valid & (bin_index == i)
        count = int(member.sum())

        if count == 0:
            bin_results.append({
                'distance_bin': label,
                'sample_count': 0,
                'rmse': None,
                'normal_pass_count': 0,
                'accuracy': None,
                'accuracy_requirement': accuracy_requirement,
                'accuracy_pass': None,
                'three_frame_index': None,
                'three_frame_pass': None,
                'violation_window_count': 0,
                'longest_violation_run': 0,
                'worst_window_start': None,
                'worst_window_end': None,
                'max_error_in_worst_window': None,
                'status': STATUS_UNDECIDABLE,
                'failure_reasons': [],
            })
            continue

        seg_errors = signed_error[member]
        rmse = float(np.sqrt(np.mean(seg_errors ** 2)))
        pass_count = int(normal_pass[member].sum())
        accuracy = pass_count / count

        # 该区间的连续三帧窗口
        owner = window_valid & (window_owner == i)
        violation_flags = owner & (window_min_all >= three_req)
        violation_count = int(violation_flags.sum())
        r3 = float(np.max(window_min_all[owner])) if owner.any() else None

        # 最长连续违规窗口数
        longest_run = 0
        current_run = 0
        for idx in np.flatnonzero(owner):
            if window_min_all[idx] >= three_req:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0

        # 最严重窗口（R3 最大）
        worst_start = worst_end = None
        worst_max_error = None
        if owner.any():
            owner_pos = np.flatnonzero(owner)
            worst_middle = int(owner_pos[int(np.argmax(window_min_all[owner_pos]))])
            span = window_span_by_middle.get(worst_middle)
            if span is not None:
                worst_start, worst_end = span[0], span[1]
                worst_max_error = float(np.max(abs_error[span[0]:span[1] + 1]))

        accuracy_pass = accuracy > accuracy_requirement
        three_pass = (r3 is not None and r3 < three_req)

        if count < min_samples:
            status = STATUS_INSUFFICIENT
            reasons: list[str] = []
        elif r3 is None:
            status = STATUS_UNDECIDABLE
            reasons = []
        else:
            reasons = []
            if not accuracy_pass:
                reasons.append(f'准确率未达到{accuracy_requirement * 100:.2f}%')
            if not three_pass:
                if longest_run >= 3:
                    reasons.append(f'存在连续{longest_run + 2}帧严重超限')
                else:
                    reasons.append('连续三帧指标超限值')
            status = STATUS_PASS if (accuracy_pass and three_pass) else STATUS_FAIL

        bin_results.append({
            'distance_bin': label,
            'sample_count': count,
            'rmse': rmse,
            'normal_pass_count': pass_count,
            'accuracy': accuracy,
            'accuracy_requirement': accuracy_requirement,
            'accuracy_pass': accuracy_pass if count >= min_samples else None,
            'three_frame_index': r3,
            'three_frame_pass': three_pass if count >= min_samples else None,
            'violation_window_count': violation_count,
            'longest_violation_run': longest_run,
            'worst_window_start': worst_start,
            'worst_window_end': worst_end,
            'max_error_in_worst_window': worst_max_error,
            'status': status,
            'failure_reasons': reasons,
        })

    result['bins'] = bin_results
    result['available'] = True

    # ── 总体结论（设计文档 7.6 / 16.6）──
    result['overall'] = _summarize_overall(
        bin_results, min_samples, accuracy_requirement)

    return result


def _evaluate_single_limit(
    result: dict,
    aligned_df: pd.DataFrame,
    metric: str,
    rules: dict,
    radar_col: str,
    truth_col: str,
    radar_val: np.ndarray,
    truth_val: np.ndarray,
    signed_error: np.ndarray,
    abs_error: np.ndarray,
    valid: np.ndarray,
    track_id: Any,
    segment_id: Any,
) -> dict:
    """Ay 单项限值评估：完整曲线内所有有效样本绝对误差 <= 限值才通过。"""
    abs_limit = rules.get('abs_limit')
    operator = rules.get('operator', '<=')

    if not isinstance(abs_limit, (int, float)):
        result['reason'] = 'Ay 限值配置无效'
        return result

    limit_arr = np.full(len(aligned_df), float(abs_limit), dtype=float)
    if operator == '<':
        hit = valid & (abs_error < float(abs_limit))
    else:
        hit = valid & (abs_error <= float(abs_limit))

    # 供展示层显示限值列
    result['abs_limit'] = float(abs_limit)
    result['operator'] = operator

    frames = pd.DataFrame({
        'track_id': track_id,
        'segment_id': segment_id,
        'radar_frame': (aligned_df['radar_frame'].values
                        if 'radar_frame' in aligned_df.columns
                        else np.arange(len(aligned_df))),
        'timestamp': (aligned_df['timestamp'].values
                      if 'timestamp' in aligned_df.columns else None),
        'distance_bin': -1,
        'truth_distance': np.abs(pd.to_numeric(
            aligned_df[_TRUTH_DISTANCE_COL], errors='coerce').to_numpy(dtype=float)),
        'radar_value': radar_val,
        'truth_value': truth_val,
        'signed_error': signed_error,
        'abs_error': abs_error,
        'normal_limit': limit_arr,
        'normal_pass': hit,
        'three_frame_limit': np.full(len(aligned_df), np.nan),
        'normalized_severe_error': np.full(len(aligned_df), np.nan),
        'continuity_break': np.zeros(len(aligned_df), dtype=bool),
        'three_frame_violation': np.zeros(len(aligned_df), dtype=bool),
        'valid': valid,
    })
    result['frames'] = frames

    count = int(valid.sum())
    if count == 0:
        result['available'] = True
        result['bins'] = [{
            'distance_bin': '完整曲线',
            'sample_count': 0,
            'rmse': None,
            'normal_pass_count': 0,
            'accuracy': None,
            'accuracy_requirement': None,
            'accuracy_pass': None,
            'three_frame_index': None,
            'three_frame_pass': None,
            'violation_window_count': 0,
            'longest_violation_run': 0,
            'worst_window_start': None,
            'worst_window_end': None,
            'max_error_in_worst_window': None,
            'status': STATUS_UNDECIDABLE,
            'failure_reasons': [],
        }]
        result['overall'] = {
            'status': STATUS_UNDECIDABLE,
            'total_samples': 0,
            'passed_bins': 0,
            'decidable_bins': 0,
            'failed_bins': 0,
            'violation_windows': 0,
            'reasons': [],
        }
        return result

    seg_errors = signed_error[valid]
    rmse = float(np.sqrt(np.mean(seg_errors ** 2)))
    max_abs_error = float(np.max(abs_error[valid]))
    pass_count = int(hit.sum())
    passed = pass_count == count

    result['available'] = True
    result['bins'] = [{
        'distance_bin': '完整曲线',
        'sample_count': count,
        'rmse': rmse,
        'normal_pass_count': pass_count,
        'accuracy': None,
        'accuracy_requirement': None,
        'accuracy_pass': None,
        'three_frame_index': None,
        'three_frame_pass': None,
        'violation_window_count': 0,
        'longest_violation_run': 0,
        'worst_window_start': None,
        'worst_window_end': None,
        'max_error_in_worst_window': max_abs_error,
        'status': STATUS_PASS if passed else STATUS_FAIL,
        'failure_reasons': [] if passed else [
            f'最大绝对误差 {max_abs_error:.3f} 超过限值 {abs_limit}'],
    }]
    result['overall'] = {
        'status': STATUS_PASS if passed else STATUS_FAIL,
        'total_samples': count,
        'passed_bins': 1 if passed else 0,
        'decidable_bins': 1,
        'failed_bins': 0 if passed else 1,
        'violation_windows': 0,
        'reasons': [] if passed else [f'最大绝对误差 {max_abs_error:.3f} 超过限值 {abs_limit}'],
    }
    return result


def _summarize_overall(bin_results: list[dict], min_samples: int,
                       accuracy_requirement: float) -> dict:
    """汇总总体结论。

    规则（设计文档 7.6 / 16.3 / 16.6）：
      - 任一可评估区间不通过 → 整体不通过；
      - 完整曲线涉及的任一区间样本数 < min_samples → 整体不出具结论；
      - 其余情况全部区间通过 → 整体通过；
      - 无数据区间不参与。
    """
    total_samples = sum(b['sample_count'] for b in bin_results)
    violation_windows = sum(b['violation_window_count'] for b in bin_results)

    # “有数据”= 样本数 > 0
    with_data = [b for b in bin_results if b['sample_count'] > 0]
    if not with_data:
        return {
            'status': STATUS_UNDECIDABLE,
            'total_samples': 0,
            'passed_bins': 0,
            'decidable_bins': 0,
            'failed_bins': 0,
            'violation_windows': 0,
            'reasons': ['无有效样本'],
        }

    # 有数据但样本不足的区间会让整体不出具结论
    insufficient = [b for b in with_data if b['status'] == STATUS_INSUFFICIENT]
    if insufficient:
        return {
            'status': STATUS_INSUFFICIENT,
            'total_samples': total_samples,
            'passed_bins': sum(1 for b in with_data if b['status'] == STATUS_PASS),
            'decidable_bins': sum(1 for b in with_data if b['status'] in
                                  (STATUS_PASS, STATUS_FAIL)),
            'failed_bins': sum(1 for b in with_data if b['status'] == STATUS_FAIL),
            'violation_windows': violation_windows,
            'reasons': [f'距离段 {b["distance_bin"]} 有效样本少于 {min_samples} 帧'
                        for b in insufficient],
        }

    decidable = [b for b in with_data if b['status'] in (STATUS_PASS, STATUS_FAIL)]
    failed = [b for b in decidable if b['status'] == STATUS_FAIL]

    reasons: list[str] = []
    for b in failed:
        for reason in b['failure_reasons']:
            reasons.append(f'{b["distance_bin"]}: {reason}')

    if failed:
        status = STATUS_FAIL
    elif not decidable:
        status = STATUS_UNDECIDABLE
        reasons = ['无可判定区间']
    else:
        status = STATUS_PASS

    return {
        'status': status,
        'total_samples': total_samples,
        'passed_bins': sum(1 for b in decidable if b['status'] == STATUS_PASS),
        'decidable_bins': len(decidable),
        'failed_bins': len(failed),
        'violation_windows': violation_windows,
        'reasons': reasons,
    }
