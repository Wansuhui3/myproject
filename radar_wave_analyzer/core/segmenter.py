"""
轨迹分段核心算法模块。
实现分段规则 A→B→C→D→E→F→G + Track_Age 展开处理。

  A: ID 首现 → 段起点（调用方隐式处理）
  B: uint8 回绕 → 延续，不切分
  C: 非回绕下降 → ID 复用，切分
  D: 时间间隔超阈值 → 无条件切分
  E: 文件边界硬切分（跨文件同 ID 必切分，避免多选文件曲线合并）
  F: 位置不连续 → 同 ID 内不同物理目标（ID 复用但 Track_Age 不降），仅 SPATIAL_SPLIT_ENABLED=True 生效
  G: 生命周期/状态字段跳变 → 结束→开始，强制切分（可选增强，LIFECYCLE_COLUMNS 配置时生效）
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    from ..config import get
except ImportError:
    from config import get  # type: ignore[no-redef]

try:
    from .data_loader import parse_timestamp
except ImportError:
    from core.data_loader import parse_timestamp  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


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


def _segment_max_speed(seg_df: pd.DataFrame, pos_cols: list[str]) -> Optional[float]:
    """计算段内最大帧间速度（m/s）。位置列缺失或不足两帧返回 None。"""
    if not pos_cols or len(seg_df) < 2:
        return None
    if not all(c in seg_df.columns for c in pos_cols):
        return None
    pos = seg_df[pos_cols].to_numpy(dtype=float)
    ts_col = 'timestamp_parsed' if 'timestamp_parsed' in seg_df.columns else 'timestamp'
    ts = pd.to_datetime(seg_df[ts_col])
    ts_ms = pd.Series(ts).diff().dt.total_seconds() * 1000
    max_s = 0.0
    for i in range(1, len(pos)):
        if np.any(np.isnan(pos[i])) or np.any(np.isnan(pos[i - 1])):
            continue
        dt = ts_ms.iloc[i]
        if pd.isna(dt) or dt <= 0:
            continue
        dist = float(np.sqrt(np.sum((pos[i] - pos[i - 1]) ** 2)))
        speed = dist / (dt / 1000.0)
        if speed > max_s:
            max_s = speed
    return max_s if max_s > 0 else 0.0


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


def _make_segment_dict(
    traj_id: str,
    id_val: int,
    sub_df: pd.DataFrame,
    ages: np.ndarray,
    wraps: list[int],
    unwrapped: np.ndarray,
    is_abnormal: bool,
    spatial_anomaly: bool = False,
) -> dict:
    """构造单个轨迹段的元信息字典。消除 3 处分段构造处的重复。"""
    n = len(sub_df)
    n_ages = len(ages)
    result = {
        'trajectory_id': traj_id,
        'original_id': id_val,
        'start_time': sub_df.iloc[0]['timestamp'] if n > 0 else None,
        'end_time': sub_df.iloc[-1]['timestamp'] if n > 0 else None,
        'total_frames': n,
        'first_track_age': int(ages[0]) if n_ages > 0 else None,
        'max_raw_age': int(ages.max()) if n_ages > 0 else None,
        'max_unwrapped_age': int(unwrapped.max()) if len(unwrapped) > 0 else None,
        'num_wraps': len(wraps),
        'is_abnormal': is_abnormal,
        'spatial_anomaly': spatial_anomaly,
    }
    for column in (
        'radar_source_key', 'radar_source_label', 'radar_source_short_label',
        'radar_source_recognized', 'radar_source_group',
    ):
        if column in sub_df.columns and n > 0:
            result[column] = sub_df.iloc[0][column]
    return result


def segment_trajectories(
    df: pd.DataFrame,
    wrap_high: Optional[int] = None,
    wrap_low: Optional[int] = None,
    gap_threshold: Optional[float] = None,
    min_traj_frames: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    对预处理后的 DataFrame 执行完整分段算法。

    规则 A→B→C→D→E→F→G 依次判定，命中即处理，不回溯：
      A: ID 首现 → 开启新段，无论 Track_Age 值
      B: uint8 回绕 (age[i-1]>=250 且 age[i]<=5 且 diff<0) → 同段延续
      C: 非回绕下降 (diff<0 且不满足B) → 切分新段
      D: 时间间隔超阈值 → 无条件切分新段
      E: 文件边界 (file_index 变化) → 强制切分新段（跨文件同 ID 独立绘制）
      F: 位置不连续 (speed>MAX_TRACK_SPEED 或 同时间戳 dist>POS_JUMP_THRESHOLD) → 切分，
         仅 SPATIAL_SPLIT_ENABLED=True 生效
      G: 生命周期字段跳变 (结束→开始) → 强制切分，仅 LIFECYCLE_COLUMNS 配置时生效

    Args:
        df: 预处理后的 DataFrame（已按时间排序）。
        wrap_high: 回绕判定上限，默认从配置读取。
        wrap_low: 回绕判定下限，默认从配置读取。
        gap_threshold: 时间间隔阈值（ms），默认从配置读取。
        min_traj_frames: 有效轨迹最小帧数，默认从配置读取。

    Returns:
        (meta_df, segments_dict)
        - meta_df: 轨迹元信息表，每段一行
        - segments_dict: {trajectory_id: 对应段的 DataFrame 子集}
    """
    if wrap_high is None:
        wrap_high = get('WRAP_HIGH', 250)
    if wrap_low is None:
        wrap_low = get('WRAP_LOW', 5)
    if gap_threshold is None:
        gap_threshold = get('GAP_THRESHOLD', 500)
    if min_traj_frames is None:
        min_traj_frames = get('MIN_TRAJ_FRAMES', 20)

    # 空间/生命周期增强判定配置（默认关闭，保证零回归）
    spatial_enabled = get('SPATIAL_SPLIT_ENABLED', False)
    max_track_speed = get('MAX_TRACK_SPEED', 50.0)
    pos_jump_threshold = get('POS_JUMP_THRESHOLD', 5.0)
    pos_cols = get('POSITION_COLUMNS', ['Dx', 'Dy'])
    available_pos_cols = [c for c in pos_cols if c in df.columns]
    lifecycle_cols = get('LIFECYCLE_COLUMNS', [])
    end_tokens = {str(t).lower() for t in get('LIFECYCLE_END_TOKENS',
                                               ['end', 'dead', 'lost', 'invalid', '0', 'false'])}
    start_tokens = {str(t).lower() for t in get('LIFECYCLE_START_TOKENS',
                                                 ['new', 'begin', 'valid', 'alive', '1', 'true'])}

    all_segments: list[dict] = []
    segments_dict: dict[str, pd.DataFrame] = {}
    has_source = 'radar_source_key' in df.columns
    group_columns = ['ID']
    if has_source:
        # 已识别的同一雷达可跨连续分卷统一处理；无法识别的文件由上传层
        # 赋予独立 radar_source_group，避免不同雷达再次因同号 ID 交织。
        source_group_column = (
            'radar_source_group' if 'radar_source_group' in df.columns
            else 'radar_source_key'
        )
        group_columns = [source_group_column]
        if 'file_index' in df.columns:
            group_columns.append('file_index')
        group_columns.append('ID')

    # 全局时间排序由数据加载层保证。groupby 避免为每个 ID 重复扫描完整
    # DataFrame，在大量同时目标时将 O(ID 数 × 总行数) 降为单次分组遍历。
    for group_key, sub in df.groupby(group_columns, sort=True):
        if has_source:
            id_val = group_key[-1]
        else:
            id_val = group_key[0] if isinstance(group_key, tuple) else group_key
        id_val = int(id_val)
        sub = sub.reset_index(drop=True)
        ages = sub['Track_Age'].values.astype(int)
        ts_parsed = sub['timestamp_parsed']
        sub_positions = sub[available_pos_cols].to_numpy(dtype=float) if available_pos_cols else None

        # 规则 A: ID 首现 → 段起点（由后续切分逻辑隐式处理）

        wraps, reuses, gaps, spatial_pts, seg_breaks = _detect_breakpoints(
            ages, ts_parsed, wrap_high, wrap_low, gap_threshold,
            spatial_split_enabled=spatial_enabled,
            positions=sub_positions,
            max_track_speed=max_track_speed,
            pos_jump_threshold=pos_jump_threshold,
        )

        # 规则 E: 文件边界硬切分（修复跨文件同 ID 被错误合并为一条曲线）
        # 同一 ID 内只要来源文件(file_index)发生变化，无论 Track_Age 是否连续、
        # 时间间隔是否超阈值，都必须切分为独立段，从而按时间独立绘制各曲线。
        if 'file_index' in sub.columns:
            file_idx = sub['file_index'].values.astype(int)
            for i in range(1, len(file_idx)):
                if int(file_idx[i]) != int(file_idx[i - 1]):
                    seg_breaks.append(i)
            seg_breaks = sorted(set(seg_breaks))

        # 规则 G: 生命周期/状态字段跳变（可选增强，配置 LIFECYCLE_COLUMNS 时生效）
        if lifecycle_cols:
            lc_breaks = _detect_lifecycle_breaks(sub, lifecycle_cols, end_tokens, start_tokens)
            if lc_breaks:
                seg_breaks = sorted(set(seg_breaks) | set(lc_breaks))

        seg_counter = 0

        source_key = str(sub.iloc[0].get('radar_source_key', '')).strip()
        source_file_index = (
            int(sub.iloc[0]['file_index']) if 'file_index' in sub.columns else None
        )

        def _trajectory_id(number: int) -> str:
            if source_key:
                safe_source = ''.join(
                    char if char.isalnum() or char in ('-', '_') else '_'
                    for char in source_key
                )
                group_suffix = f'_f{source_file_index}' if source_file_index is not None else ''
                source_group = str(sub.iloc[0].get('radar_source_group', source_key))
                if source_group != source_key:
                    group_suffix += '_' + ''.join(
                        char if char.isalnum() or char in ('-', '_') else '_'
                        for char in source_group
                    )
                return f'{safe_source}{group_suffix}__{id_val}_seg{number}'
            return f'{id_val}_seg{number}'

        def _seg_spatial_anomaly(seg_df: pd.DataFrame) -> bool:
            if not available_pos_cols:
                return False
            ms = _segment_max_speed(seg_df, available_pos_cols)
            return ms is not None and ms > max_track_speed

        if len(seg_breaks) == 0:
            # 无断点 → 单条轨迹
            seg_counter += 1
            traj_id = _trajectory_id(seg_counter)

            unwrapped = _unwrap_track_age(ages, wraps)
            is_abnormal = not _check_unwrapped_monotonicity(unwrapped)
            spatial_anomaly = _seg_spatial_anomaly(sub)

            segment = _make_segment_dict(traj_id, id_val, sub, ages, wraps, unwrapped, is_abnormal, spatial_anomaly)
            all_segments.append(segment)
            segments_dict[traj_id] = sub.copy()
        else:
            # 有断点 → 按断点切分
            start_idx = 0
            for break_idx in seg_breaks:
                seg_counter += 1
                traj_id = _trajectory_id(seg_counter)

                # 每个轨迹段必须拥有从 0 开始的本地索引。框选、高亮和局部统计
                # 均以段内位置工作；保留父分组索引会导致第二段及后续段失效。
                seg_sub = sub.iloc[start_idx:break_idx].copy().reset_index(drop=True)
                seg_ages = ages[start_idx:break_idx]
                seg_wraps = [w - start_idx for w in wraps if start_idx <= w < break_idx]

                unwrapped = _unwrap_track_age(seg_ages, seg_wraps)
                is_abnormal = not _check_unwrapped_monotonicity(unwrapped)
                spatial_anomaly = _seg_spatial_anomaly(seg_sub)

                segment = _make_segment_dict(traj_id, id_val, seg_sub, seg_ages, seg_wraps, unwrapped, is_abnormal, spatial_anomaly)
                all_segments.append(segment)
                segments_dict[traj_id] = seg_sub

                start_idx = break_idx

            # 最后一段
            seg_counter += 1
            traj_id = _trajectory_id(seg_counter)

            seg_sub = sub.iloc[start_idx:].copy().reset_index(drop=True)
            seg_ages = ages[start_idx:]
            seg_wraps = [w - start_idx for w in wraps if start_idx <= w]

            unwrapped = _unwrap_track_age(seg_ages, seg_wraps)
            is_abnormal = not _check_unwrapped_monotonicity(unwrapped)
            spatial_anomaly = _seg_spatial_anomaly(seg_sub)

            segment = _make_segment_dict(traj_id, id_val, seg_sub, seg_ages, seg_wraps, unwrapped, is_abnormal, spatial_anomaly)
            all_segments.append(segment)
            segments_dict[traj_id] = seg_sub

    meta_df = pd.DataFrame(all_segments)

    # 生成显示标签：起止时间的 HH/MM/SS/mmm 格式
    if len(meta_df) > 0:
        def _fmt_label(row):
            def _time_part(ts_str):
                if not ts_str:
                    return '??'
                parts = str(ts_str).split('_')
                if len(parts) >= 7:
                    return f'{parts[3]}/{parts[4]}/{parts[5]}/{parts[6]}'
                return str(ts_str)
            st = _time_part(row['start_time'])
            et = _time_part(row['end_time'])
            return f'{st} - {et}'
        meta_df['display_label'] = meta_df.apply(_fmt_label, axis=1)
        meta_df['insufficient_samples'] = meta_df['total_frames'] < min_traj_frames

    logger.info(
        f'分段完成: {len(meta_df)} 段, '
        f'异常段 {meta_df["is_abnormal"].sum()}, '
        f'空间跳变段 {meta_df["spatial_anomaly"].sum()}, '
        f'样本不足 {meta_df["insufficient_samples"].sum()}'
    )
    return meta_df, segments_dict


def get_segment_ids_by_time(
    meta_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[str]:
    """
    筛选与指定时间段有交集的轨迹段 ID。

    Args:
        meta_df: 轨迹元信息表。
        start_ts: 起始时间戳。
        end_ts: 结束时间戳。

    Returns:
        符合条件的 trajectory_id 列表。
    """
    if meta_df.empty:
        return []

    # CSV 原始时间是下划线格式，Pandas 不能直接批量推断。常见的完整格式
    # 走指定格式的向量化路径；其余项目允许的格式才回退到唯一解析器。
    def _parse_meta_times(values: pd.Series) -> pd.Series:
        raw = values.astype(str)
        full_mask = raw.str.fullmatch(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{1,6}')
        parsed = pd.Series(pd.NaT, index=values.index, dtype='datetime64[ns]')
        if full_mask.any():
            parsed.loc[full_mask] = pd.to_datetime(
                raw.loc[full_mask], format='%Y_%m_%d_%H_%M_%S_%f', errors='coerce',
            )
        other_mask = ~full_mask
        if other_mask.any():
            parsed.loc[other_mask] = raw.loc[other_mask].map(parse_timestamp)
        return parsed

    starts = _parse_meta_times(meta_df['start_time'])
    ends = _parse_meta_times(meta_df['end_time'])
    overlap_mask = (starts <= end_ts) & (ends >= start_ts)
    return meta_df.loc[overlap_mask, 'trajectory_id'].tolist()
