"""
帧间差分波动计算与统计指标模块。
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    from ..config import get
except ImportError:
    from config import get  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def calc_frame_diff(
    values: np.ndarray,
    ts_parsed: pd.DatetimeIndex,
    sampling_period_ms: Optional[float] = None,
) -> np.ndarray:
    """
    计算帧间差分（波动值）。

    规则：
      - wave[i] = value[i] - value[i-1]
      - 首帧无波动值（NaN）
      - 帧间隔 > 2倍标称采样周期时跳过差分（NaN），避免帧丢失导致的突变误计
      - 不同轨迹段之间不计算波动（由调用方保证传入同一段的数据）

    Args:
        values: 物理量数值数组。
        ts_parsed: 解析后的时间戳。
        sampling_period_ms: 标称采样周期（ms），默认从配置读取。

    Returns:
        波动值数组，首帧为 NaN。
    """
    if sampling_period_ms is None:
        sampling_period_ms = get('SAMPLING_PERIOD_MS', 50)

    n = len(values)
    if n < 2:
        return np.full(n, np.nan)

    # 向量化差分：raw_diff[i] = values[i] - values[i-1]，首帧为 NaN
    raw_diff = np.empty(n)
    raw_diff[0] = np.nan
    raw_diff[1:] = values[1:] - values[:-1]

    # 向量化时间间隔掩码：超阈值或 NaN → 置 NaN
    time_diffs_ms = pd.Series(ts_parsed).diff().dt.total_seconds() * 1000
    skip_threshold = 2.0 * sampling_period_ms
    td_arr = np.asarray(time_diffs_ms.values, dtype=float)
    invalid = (td_arr > skip_threshold) | np.isnan(td_arr)
    invalid[0] = True  # 首帧必为 NaN

    return np.where(invalid, np.nan, raw_diff)


def _get_or_compute_diff(
    seg_df: pd.DataFrame,
    quantity: str,
    sampling_period_ms: Optional[float] = None,
    diff_cache: Optional[dict] = None,
) -> np.ndarray:
    """获取或计算指定物理量的帧间差分（带缓存）。

    缓存键为 (id(seg_df), quantity)，确保同一 DataFrame 对象 + 同一物理量
    在单次回调内只计算一次。
    """
    if diff_cache is not None:
        key = (id(seg_df), quantity)
        cached = diff_cache.get(key)
        if cached is not None:
            return cached

    values = seg_df[quantity].values.astype(float)
    ts_parsed = pd.DatetimeIndex(seg_df['timestamp_parsed'])
    diff = calc_frame_diff(values, ts_parsed, sampling_period_ms)

    if diff_cache is not None:
        diff_cache[(id(seg_df), quantity)] = diff
    return diff


def calc_wave_stats(diff_values: np.ndarray) -> dict:
    """
    对波动值序列计算全部统计指标。

    指标集：
      - valid_count: 有效样本数
      - mean_abs: 波动绝对值平均值 = mean(|diff|)
      - std_dev: 波动标准差（ddof=1）
      - rms: 波动均方根 = sqrt(mean(diff²))
      - peak_to_peak: 峰峰值 = max(diff) + abs(min(diff))（非 max-min）
      - max_positive: 最大正向跳变 = max(diff)
      - max_negative: 最大负向跳变 = min(diff)

    Args:
        diff_values: 波动值数组（含 NaN）。

    Returns:
        统计指标字典。
    """
    valid = diff_values[~np.isnan(diff_values)]

    if len(valid) == 0:
        return {
            'valid_count': 0,
            'mean_abs': np.nan,
            'std_dev': np.nan,
            'rms': np.nan,
            'peak_to_peak': np.nan,
            'max_positive': np.nan,
            'max_negative': np.nan,
        }

    # 样本标准差要求至少两个有效差分；显式返回 NaN 而非调用 numpy 并产生
    # "Degrees of freedom" 运行时警告，调用方可将其正常呈现为数据不足。
    std_dev = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan

    return {
        'valid_count': len(valid),
        'mean_abs': float(np.mean(np.abs(valid))),
        'std_dev': std_dev,
        'rms': float(np.sqrt(np.mean(valid ** 2))),
        'peak_to_peak': float(np.max(valid) + abs(np.min(valid))),
        'max_positive': float(np.max(valid)),
        'max_negative': float(np.min(valid)),
    }


def compute_segment_wave(
    seg_df: pd.DataFrame,
    quantity: str,
    sampling_period_ms: Optional[float] = None,
) -> pd.DataFrame:
    """
    对单段轨迹计算指定物理量的波动值，并返回带波动列的 DataFrame。

    Args:
        seg_df: 单段轨迹的 DataFrame。
        quantity: 物理量字段名（如 'Dx', 'Vy'）。
        sampling_period_ms: 标称采样周期（ms）。

    Returns:
        添加了 'wave_{quantity}' 列的 DataFrame。
    """
    if quantity not in seg_df.columns:
        raise ValueError(f'物理量 {quantity} 不在数据字段中')

    values = seg_df[quantity].values.astype(float)
    ts_parsed = pd.DatetimeIndex(seg_df['timestamp_parsed'])

    wave_col = f'wave_{quantity}'
    seg_df = seg_df.copy()
    seg_df[wave_col] = calc_frame_diff(values, ts_parsed, sampling_period_ms)

    return seg_df


def compute_segment_stats(
    seg_df: pd.DataFrame,
    quantity: str,
    mask: Optional[pd.Series] = None,
    sampling_period_ms: Optional[float] = None,
    diff_cache: Optional[dict] = None,
) -> dict:
    """
    计算单段轨迹的波动统计指标。

    Args:
        seg_df: 单段轨迹的 DataFrame。
        quantity: 物理量字段名。
        mask: 可选的布尔掩码，用于框选区间统计。
        sampling_period_ms: 标称采样周期（ms）。
        diff_cache: 可选的差分缓存字典，避免重复计算。

    Returns:
        统计指标字典。
    """
    if mask is not None:
        seg_df = seg_df[mask].copy()

    if len(seg_df) < 2:
        return {
            'valid_count': 0,
            'mean_abs': np.nan,
            'std_dev': np.nan,
            'rms': np.nan,
            'peak_to_peak': np.nan,
            'max_positive': np.nan,
            'max_negative': np.nan,
        }

    diff = _get_or_compute_diff(seg_df, quantity, sampling_period_ms, diff_cache)
    return calc_wave_stats(diff)


def compute_fluctuation_stats(
    seg_df: pd.DataFrame,
    mask: Optional[pd.Series] = None,
    sampling_period_ms: Optional[float] = None,
    diff_cache: Optional[dict] = None,
) -> dict:
    """计算跨维度波动指标（不受当前选中物理量影响）。

    返回指标：
      - dx_max_dist: 最远检出距离 = max(|Dx|)，始终使用全段数据
      - dx_wave:     Dx 帧间差分标准差
      - dy_wave:     Dy 帧间差分标准差
      - vx_wave:     Vx 帧间差分标准差
      - vy_wave:     Vy 帧间差分标准差
      - yaw_wave:    HeadingAngle 原始值标准差（非帧间差分）

    Args:
        seg_df: 单段轨迹的 DataFrame。
        mask: 可选的布尔掩码，用于框选区间。
        sampling_period_ms: 标称采样周期（ms）。

    Returns:
        波动指标字典，所有值可含 None（数据不足时）。
    """
    if mask is not None:
        df = seg_df[mask].copy()
    else:
        df = seg_df

    result: dict = {}

    # 最远检出距离（始终用全段 Dx，不受框选影响）
    if 'Dx' in seg_df.columns and len(seg_df) > 0:
        result['dx_max_dist'] = float(np.max(np.abs(seg_df['Dx'].values.astype(float))))
    else:
        result['dx_max_dist'] = None

    # 各维度波动 = 帧间差分标准差
    dims = [
        ('Dx', 'dx_wave'),
        ('Dy', 'dy_wave'),
        ('Vx', 'vx_wave'),
        ('Vy', 'vy_wave'),
    ]
    for col, key in dims:
        if col in df.columns and len(df) >= 2:
            diff = _get_or_compute_diff(df, col, sampling_period_ms, diff_cache)
            valid = diff[~np.isnan(diff)]
            if len(valid) > 1:
                result[key] = float(np.std(valid, ddof=1))
            else:
                result[key] = None
        else:
            result[key] = None

    # YAW 波动 = HeadingAngle 原始值标准差（与 HeadingAngle 数值一致）
    if 'HeadingAngle' in df.columns and len(df) >= 2:
        ha_values = df['HeadingAngle'].values.astype(float)
        valid = ha_values[~np.isnan(ha_values)]
        if len(valid) > 1:
            result['yaw_wave'] = float(np.std(valid, ddof=1))
        else:
            result['yaw_wave'] = None
    else:
        result['yaw_wave'] = None

    return result


def find_max_jump(
    seg_df: pd.DataFrame,
    quantity: str,
    sampling_period_ms: Optional[float] = None,
    diff_cache: Optional[dict] = None,
) -> Optional[dict]:
    """定位帧间差分绝对值最大的位置（最大跳变处）。

    Args:
        seg_df: 单段轨迹的 DataFrame。
        quantity: 物理量字段名。
        sampling_period_ms: 标称采样周期（ms）。

    Returns:
        包含跳变信息的字典，数据不足时返回 None:
          - idx: 跳变发生的帧索引（diff[i] 对应的 i）
          - diff_value: 该帧的差分值（有正负）
          - timestamp: 该帧的时间戳
          - value_before: 跳变前的值
          - value_after: 跳变后的值
    """
    if quantity not in seg_df.columns or len(seg_df) < 2:
        return None

    diff = _get_or_compute_diff(seg_df, quantity, sampling_period_ms, diff_cache)

    valid_mask = ~np.isnan(diff)
    if not valid_mask.any():
        return None

    # 在有效差分中找绝对值最大的索引
    max_idx = int(np.nanargmax(np.abs(diff)))

    values = seg_df[quantity].values.astype(float)
    ts_parsed = pd.DatetimeIndex(seg_df['timestamp_parsed'])

    return {
        'idx': max_idx,
        'diff_value': float(diff[max_idx]),
        'timestamp': ts_parsed[max_idx],
        'value_before': float(values[max_idx - 1]) if max_idx > 0 else None,
        'value_after': float(values[max_idx]),
    }
