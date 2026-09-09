"""性能指标纯数值辅助：状态常量、列解析、阈值向量化、距离分桶与连续性判定。

本模块不依赖任何业务模块（零内部依赖），供 performance_metric 与
performance 编排层共同引用。设计口径见设计文档 4 / 5.2 / 7.4 / 7.5。
"""
from typing import Optional

import numpy as np
import pandas as pd

# 距离段状态枚举（设计文档 7.6）
STATUS_PASS = '通过'
STATUS_FAIL = '不通过'
STATUS_UNDECIDABLE = '不可判定'
STATUS_INSUFFICIENT = '样本不足'

# 真值纵向距离列名（RTK，绝不使用雷达 Dx）
_TRUTH_DISTANCE_COL = 'rtk_center_x'
_RADAR_COL_CANDIDATES = 'radar_col'
_TRUTH_COL_CANDIDATES = 'truth_col'


def _resolve_column(df: pd.DataFrame, primary: Optional[str],
                    fallbacks: Optional[list]) -> Optional[str]:
    """返回 DataFrame 中实际存在的列名，主列名缺失时按候选列表回退。"""
    if primary and primary in df.columns:
        return primary
    for candidate in (fallbacks or []):
        if candidate in df.columns:
            return candidate
    return None


def _resolve_limit(limit_rule: dict, basis: np.ndarray) -> np.ndarray:
    """按阈值模式向量化计算逐帧阈值。

    Args:
        limit_rule: 阈值规则 dict，含 mode / absolute / percent / scale。
        basis: 百分比基准数组（d 或 abs(真值)），用于 percent 类模式。

    Returns:
        与 basis 等长的阈值数组；规则非法时返回全 NaN。
    """
    if not isinstance(limit_rule, dict):
        return np.full(basis.shape, np.nan, dtype=float)

    mode = limit_rule.get('mode')
    absolute = limit_rule.get('absolute')
    percent = limit_rule.get('percent')

    if mode == 'absolute':
        value = float(absolute) if isinstance(absolute, (int, float)) else np.nan
        return np.full(basis.shape, value, dtype=float)

    if mode == 'percent':
        if not isinstance(percent, (int, float)):
            return np.full(basis.shape, np.nan, dtype=float)
        return float(percent) * basis

    if mode == 'max_absolute_percent':
        if not isinstance(absolute, (int, float)) or not isinstance(percent, (int, float)):
            return np.full(basis.shape, np.nan, dtype=float)
        return np.maximum(float(absolute), float(percent) * basis)

    if mode == 'scaled_max_absolute_percent':
        scale = limit_rule.get('scale')
        if (not isinstance(absolute, (int, float))
                or not isinstance(percent, (int, float))
                or not isinstance(scale, (int, float)) or scale <= 0):
            return np.full(basis.shape, np.nan, dtype=float)
        return float(scale) * np.maximum(float(absolute), float(percent) * basis)

    return np.full(basis.shape, np.nan, dtype=float)


def _assign_bins(distances: np.ndarray, bins: list,
                 include_last_upper: bool) -> np.ndarray:
    """将纵向距离映射到距离段索引，越界返回 -1。

    区间为 [lo, hi)，最后一段在 include_last_upper 时包含上界。
    """
    index = np.full(distances.shape, -1, dtype=int)
    finite = np.isfinite(distances)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if include_last_upper and i == len(bins) - 2:
            hit = (distances >= lo) & (distances <= hi)
        else:
            hit = (distances >= lo) & (distances < hi)
        index[finite & hit] = i
    return index


def _compute_continuity_breaks(
    timestamps: np.ndarray,
    gap_factor: float,
    gap_percentile: float,
    matched: np.ndarray,
) -> np.ndarray:
    """标记连续关系中断位置。

    满足下列任一条件即视为不连续（设计文档 7.5）：
      - 当前帧或前一帧未匹配；
      - 相邻时间间隔超过“正常间隔 × gap_factor”。

    Args:
        timestamps: 逐帧时间戳（秒）。
        gap_factor: 中断倍率，复用对比图 radar_gap_break_factor 口径。
        gap_percentile: 正常间隔分位数，避免长中断本身抬高间隔中位数。
        matched: 布尔数组，标记该帧是否为有效匹配帧。

    Returns:
        布尔数组 breaks[i] 为 True 表示第 i 帧与其前一帧不连续。
    """
    n = len(timestamps)
    breaks = np.zeros(n, dtype=bool)
    if n == 0:
        return breaks

    breaks[0] = True
    if n < 2:
        return breaks

    # 未匹配帧前后均打断连续关系
    breaks[1:] |= ~matched[1:]
    breaks[1:] |= ~matched[:-1]

    diffs = np.diff(timestamps)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive.size == 0:
        return breaks

    nominal = float(np.percentile(positive, gap_percentile))
    if nominal <= 0:
        return breaks

    long_gap = np.zeros(n, dtype=bool)
    long_gap[1:] = (diffs > nominal * gap_factor) | ~np.isfinite(diffs)
    breaks |= long_gap
    return breaks


def _rolling_window_min(values: np.ndarray, window: int = 3) -> np.ndarray:
    """长度为 window 的滑动窗口最小值（设计文档 15 推荐的向量化实现）。

    返回数组**与输入等长**：索引 j < window-1 的项为 NaN，
    索引 j >= window-1 的项对应窗口 [j - window + 1, j] 的最小值。

    窗口内存在 NaN（无效帧或阈值缺失）时结果为 NaN，该窗口不参与统计，
    这与 pandas rolling 默认的 min_periods=window 语义一致。
    """
    if len(values) < window:
        return np.array([], dtype=float)
    return pd.Series(values).rolling(window).min().to_numpy(dtype=float)
