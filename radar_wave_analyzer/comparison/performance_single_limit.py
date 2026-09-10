"""性能指标单限值模式评估（Ay 类）：完整曲线内所有有效样本不超限。

阈值/距离段配置由调用方传入（evaluate_all_metrics 编排层负责读取配置），
本模块只做纯计算，不依赖 Dash。口径见设计文档 7.5。
"""
from typing import Any

import numpy as np
import pandas as pd

from .performance_common import (
    _TRUTH_DISTANCE_COL,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_UNDECIDABLE,
)


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
