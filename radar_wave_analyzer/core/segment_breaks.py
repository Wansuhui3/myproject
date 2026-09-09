"""轨迹分段断点检测域（纯数值，零内部依赖）。

实现分段规则 B/C/D/F 的断点判定、生命周期规则 G 检测，
以及 Track_Age 的 uint8 展开与单调性校验。
规则口径见设计文档与本包 segmenter 模块 docstring。
"""
from typing import Optional

import numpy as np
import pandas as pd


def _detect_breakpoints(
    ages: np.ndarray,
    ts_parsed: pd.DatetimeIndex,
    wrap_high: int,
    wrap_low: int,
    gap_threshold_ms: float,
    spatial_split_enabled: bool = False,
    positions: Optional[np.ndarray] = None,
    max_track_speed: float = 50.0,
    pos_jump_threshold: float = 5.0,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """
    检测单 ID 分组内的所有断点。

    五层判断按优先级 A→B→C→D→F：
      A: ID 首现 → 由调用方处理（自动作为段起点）
      B: uint8 回绕 → 延续，不切分
      C: 非回绕下降 → ID 复用，切分
      D: 时间间隔超阈值 → 无条件切分
      F: 位置不连续 → 同 ID 内不同物理目标（ID 复用但 Track_Age 不降），
         仅当 spatial_split_enabled=True 时生效；位置列缺失/NaN 时降级跳过。

    Args:
        ages: Track_Age 原始值数组。
        ts_parsed: 解析后的时间戳数组。
        wrap_high: 回绕判定上限。
        wrap_low: 回绕判定下限。
        gap_threshold_ms: 时间间隔阈值（毫秒）。
        spatial_split_enabled: 是否启用规则 F。
        positions: 位置坐标二维数组 (n, k)，用于计算空间位移；None 时跳过规则 F。
        max_track_speed: 帧间最大合理速度（m/s），超过视为不同物理目标。
        pos_jump_threshold: 同时间戳(Δt=0)位置跳变阈值（m）。

    Returns:
        (wrap_points, reuse_points, gap_points, spatial_points, all_breaks)
        - wrap_points: 回绕点索引列表（不切分）
        - reuse_points: ID 复用断点索引列表
        - gap_points: 时间间隔断点索引列表
        - spatial_points: 位置跳变断点索引列表
        - all_breaks: 所有切分断点（reuse + gap + spatial，已排序去重）
    """
    wrap_points: list[int] = []
    reuse_points: list[int] = []
    gap_points: list[int] = []
    spatial_points: list[int] = []

    # 计算帧间时间差（毫秒）
    # DatetimeIndex.diff() 直接返回 TimedeltaIndex，再转 Series 取 total_seconds
    time_diffs_ms = pd.Series(ts_parsed).diff().dt.total_seconds() * 1000

    for i in range(1, len(ages)):
        diff = int(ages[i]) - int(ages[i - 1])

        # 规则 B: uint8 回绕 → 延续，不切分
        # 条件: age[i-1] >= wrap_high 且 age[i] <= wrap_low 且 diff < 0
        if diff < 0 and int(ages[i - 1]) >= wrap_high and int(ages[i]) <= wrap_low:
            wrap_points.append(i)
            continue

        # 规则 C: 非回绕下降 → ID 复用，切分
        # 条件: diff < 0 且不满足规则 B
        if diff < 0:
            reuse_points.append(i)
            continue

        # 规则 D: 时间间隔过大 → 无条件切分
        # 条件: time_diff > gap_threshold，不检查位置跳变
        time_diff = time_diffs_ms.iloc[i]
        if not pd.isna(time_diff) and time_diff > gap_threshold_ms:
            gap_points.append(i)
            continue

        # 规则 F: 位置不连续 → 同一 ID 内不同物理目标（ID 复用但 Track_Age 不降）
        # 仅启用时生效；位置缺失/NaN 自动降级跳过。
        if spatial_split_enabled and positions is not None and positions.shape[1] >= 1:
            prev = positions[i - 1]
            cur = positions[i]
            if not (np.any(np.isnan(prev)) or np.any(np.isnan(cur))):
                dist = float(np.sqrt(np.sum((cur - prev) ** 2)))
                if not pd.isna(time_diff) and time_diff > 0:
                    speed = dist / (time_diff / 1000.0)
                    if speed > max_track_speed:
                        spatial_points.append(i)
                elif dist > pos_jump_threshold:
                    # 同时间戳(Δt=0)直接用距离阈值判定，避免除零
                    # 命中用户描述的"同一时刻同 ID 复用"盲区
                    spatial_points.append(i)

    # 合并所有切分断点（wrap 不切分）
    all_breaks = sorted(set(reuse_points + gap_points + spatial_points))

    return wrap_points, reuse_points, gap_points, spatial_points, all_breaks


def _detect_lifecycle_breaks(
    sub: pd.DataFrame,
    lifecycle_cols: list[str],
    end_tokens: set[str],
    start_tokens: set[str],
) -> list[int]:
    """规则 G: 根据生命周期/状态字段检测 ID 复用断点。

    若某帧(i-1)状态属于"结束/失效"集合、且下一帧(i)状态属于"开始/有效"集合，
    则在第 i 帧处强制切分（不同物理目标）。字段不存在或配置为空时返回空列表。
    """
    breaks: list[int] = []
    if not lifecycle_cols:
        return breaks
    for col in lifecycle_cols:
        if col not in sub.columns:
            continue
        vals = sub[col].astype(str).str.strip().str.lower()
        for i in range(1, len(sub)):
            prev_v = vals.iloc[i - 1]
            cur_v = vals.iloc[i]
            if prev_v in end_tokens and cur_v in start_tokens:
                breaks.append(i)
    return sorted(set(breaks))


def _unwrap_track_age(
    ages: np.ndarray,
    wrap_points: list[int],
) -> np.ndarray:
    """
    对 Track_Age 序列执行 uint8 展开处理。
    回绕点处 offset += 256，使序列单调非递减。

    Args:
        ages: 原始 Track_Age 数组。
        wrap_points: 回绕点索引列表。

    Returns:
        展开后的 Track_Age 数组。
    """
    unwrapped = ages.copy().astype(int)
    offset = 0

    for i in range(len(ages)):
        if i > 0 and i in wrap_points:
            offset += 256
        unwrapped[i] = int(ages[i]) + offset

    return unwrapped


def _check_unwrapped_monotonicity(unwrapped: np.ndarray) -> bool:
    """
    检查展开后 Track_Age 是否单调非递减（排除 diff=0 的重复帧）。

    Returns:
        True 表示正常，False 表示存在非单调异常。
    """
    diffs = np.diff(unwrapped)
    non_zero_diffs = diffs[diffs != 0]
    if len(non_zero_diffs) == 0:
        return True
    return np.all(non_zero_diffs > 0)
