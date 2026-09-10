"""
轨迹分段核心算法模块（分段构建与查询层）。
实现分段规则 A→B→C→D→E→F→G + Track_Age 展开处理。

  A: ID 首现 → 段起点（调用方隐式处理）
  B: uint8 回绕 → 延续，不切分
  C: 非回绕下降 → ID 复用，切分
  D: 时间间隔超阈值 → 无条件切分
  E: 文件边界硬切分（跨文件同 ID 必切分，避免多选文件曲线合并）
  F: 位置不连续 → 同 ID 内不同物理目标（ID 复用但 Track_Age 不降），仅 SPATIAL_SPLIT_ENABLED=True 生效
  G: 生命周期/状态字段跳变 → 结束→开始，强制切分（可选增强，LIFECYCLE_COLUMNS 配置时生效）

模块职责拆分（阶段 6）：
  - segment_breaks  断点检测域（纯数值）：规则 B/C/D/F 断点判定、
                    生命周期规则 G、Track_Age 展开与单调性校验；
  - segmenter（本模块）段构建（孤立帧剔除、trajectory_id 生成）、
    segment_trajectories 编排与时间段查询。
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import get
from .data_loader import parse_timestamp

# ── 兼容 re-export：test_segmenter.py 直接从本模块导入 _detect_* ──
from .segment_breaks import (  # noqa: F401
    _check_unwrapped_monotonicity,
    _detect_breakpoints,
    _detect_lifecycle_breaks,
    _unwrap_track_age,
)

logger = logging.getLogger(__name__)


def _segment_spatial_extremes(
    seg_df: pd.DataFrame,
    pos_cols: list[str],
) -> Optional[tuple[float, float]]:
    """扫描段内帧间位移与速度的最大值，返回 ``(max_dist_m, max_speed_mps)``。

    位移与速度分开统计：位移不依赖 Δt，因此 Δt=0（同时间戳复用）的瞬移同样
    能被捕获；速度仅在 Δt>0 时计算，避免除零。

    位置列缺失或不足两帧返回 None。
    """
    if not pos_cols or len(seg_df) < 2:
        return None
    if not all(c in seg_df.columns for c in pos_cols):
        return None
    pos = seg_df[pos_cols].to_numpy(dtype=float)
    ts_col = 'timestamp_parsed' if 'timestamp_parsed' in seg_df.columns else 'timestamp'
    ts = pd.to_datetime(seg_df[ts_col])
    ts_ms = pd.Series(ts).diff().dt.total_seconds() * 1000
    max_dist = 0.0
    max_speed = 0.0
    for i in range(1, len(pos)):
        if np.any(np.isnan(pos[i])) or np.any(np.isnan(pos[i - 1])):
            continue
        # 位移与 Δt 无关：Δt=0 的瞬移同样计入
        dist = float(np.sqrt(np.sum((pos[i] - pos[i - 1]) ** 2)))
        if dist > max_dist:
            max_dist = dist
        dt = ts_ms.iloc[i]
        if pd.isna(dt) or dt <= 0:
            continue
        speed = dist / (dt / 1000.0)
        if speed > max_speed:
            max_speed = speed
    return max_dist, max_speed


def _make_segment_dict(
    traj_id: str,
    id_val: int,
    sub_df: pd.DataFrame,
    ages: np.ndarray,
    wraps: list[int],
    unwrapped: np.ndarray,
    is_abnormal: bool,
    spatial_anomaly: bool = False,
    timestamp_position_conflict: bool = False,
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
        'timestamp_position_conflict': timestamp_position_conflict,
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

        def _seg_spatial_flags(seg_df: pd.DataFrame) -> tuple[bool, bool]:
            """分别判定时序空间跳变与同时间戳位置冲突。

            Δt>0 时，位移或速度超阈值记为普通空间跳变；Δt=0 时不计算速度，
            位移超阈值单独记为“同时间戳位置冲突”，不再混入 spatial_anomaly。
            """
            if not available_pos_cols or len(seg_df) < 2:
                return False, False
            pos = seg_df[available_pos_cols].to_numpy(dtype=float)
            ts = pd.to_datetime(seg_df['timestamp_parsed'])
            time_diffs = pd.Series(ts).diff().dt.total_seconds().to_numpy(dtype=float)
            spatial_anomaly = False
            timestamp_conflict = False
            for i in range(1, len(pos)):
                if np.any(np.isnan(pos[i])) or np.any(np.isnan(pos[i - 1])):
                    continue
                dist = float(np.sqrt(np.sum((pos[i] - pos[i - 1]) ** 2)))
                dt = time_diffs[i]
                if np.isnan(dt) or dt < 0:
                    continue
                if dt == 0:
                    timestamp_conflict |= dist > pos_jump_threshold
                else:
                    speed = dist / dt
                    spatial_anomaly |= (
                        dist > pos_jump_threshold or speed > max_track_speed
                    )
            return spatial_anomaly, timestamp_conflict

        # 组内段先本地收集，切分完成后统一做孤立帧剔除，再并入全局结果。
        local_segments: list[tuple[dict, pd.DataFrame]] = []

        if len(seg_breaks) == 0:
            # 无断点 → 单条轨迹
            seg_counter += 1
            traj_id = _trajectory_id(seg_counter)

            unwrapped = _unwrap_track_age(ages, wraps)
            is_abnormal = not _check_unwrapped_monotonicity(unwrapped)
            spatial_anomaly, timestamp_conflict = _seg_spatial_flags(sub)

            segment = _make_segment_dict(
                traj_id, id_val, sub, ages, wraps, unwrapped, is_abnormal,
                spatial_anomaly, timestamp_conflict,
            )
            local_segments.append((segment, sub.copy()))
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
                spatial_anomaly, timestamp_conflict = _seg_spatial_flags(seg_sub)

                segment = _make_segment_dict(
                    traj_id, id_val, seg_sub, seg_ages, seg_wraps, unwrapped,
                    is_abnormal, spatial_anomaly, timestamp_conflict,
                )
                local_segments.append((segment, seg_sub))

                start_idx = break_idx

            # 最后一段
            seg_counter += 1
            traj_id = _trajectory_id(seg_counter)

            seg_sub = sub.iloc[start_idx:].copy().reset_index(drop=True)
            seg_ages = ages[start_idx:]
            seg_wraps = [w - start_idx for w in wraps if start_idx <= w]

            unwrapped = _unwrap_track_age(seg_ages, seg_wraps)
            is_abnormal = not _check_unwrapped_monotonicity(unwrapped)
            spatial_anomaly, timestamp_conflict = _seg_spatial_flags(seg_sub)

            segment = _make_segment_dict(
                traj_id, id_val, seg_sub, seg_ages, seg_wraps, unwrapped,
                is_abnormal, spatial_anomaly, timestamp_conflict,
            )
            local_segments.append((segment, seg_sub))

        # ---- 孤立单帧段剔除 ----
        # 切分规则（C/D/F/G/E）命中数据毛刺帧时会切出仅含 1 帧的碎片段：
        # 典型如采集启动残留的异常首帧（Track_Age 倒挂 + 时间孤立）。
        # 单帧段无法参与波动分析，组内存在其他段时视为切分毛刺予以剔除；
        # 整组仅 1 帧时保留（走 insufficient_samples 标记路径）。
        if (get('REMOVE_ISOLATED_SINGLE_FRAMES', True)
                and len(local_segments) > 1):
            kept = [(seg_meta, seg_df) for seg_meta, seg_df in local_segments
                    if seg_meta['total_frames'] > 1]
            removed = len(local_segments) - len(kept)
            if removed:
                logger.warning(
                    '剔除 %d 个孤立单帧段（ID=%s，疑似 Track_Age/时间戳异常帧）',
                    removed, id_val,
                )
                local_segments = kept
                # 重编号保持段序号连续，避免 UI 出现 seg1 缺号
                renumbered: list[tuple[dict, pd.DataFrame]] = []
                for number, (seg_meta, seg_df) in enumerate(local_segments, start=1):
                    seg_meta['trajectory_id'] = _trajectory_id(number)
                    renumbered.append((seg_meta, seg_df))
                local_segments = renumbered

        for seg_meta, seg_df in local_segments:
            all_segments.append(seg_meta)
            segments_dict[seg_meta['trajectory_id']] = seg_df

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

    # 所有段被过滤/剔除时（例如数据全为孤立帧）meta_df 为空，
    # 此时 is_abnormal 等派生列尚未创建，直接统计会抛 KeyError。
    if len(meta_df) == 0:
        logger.warning('分段完成: 0 段（全部样本被过滤或作为孤立帧剔除）')
        return meta_df, segments_dict

    logger.info(
        '分段完成: %s 段, 异常段 %s, 空间跳变段 %s, '
        '同时间戳位置冲突段 %s, 样本不足 %s',
        len(meta_df), meta_df['is_abnormal'].sum(),
        meta_df['spatial_anomaly'].sum(),
        meta_df['timestamp_position_conflict'].sum(),
        meta_df['insufficient_samples'].sum(),
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
