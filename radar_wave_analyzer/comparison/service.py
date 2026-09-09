"""真值对比业务编排服务。

本模块只组合核心算法，不依赖 Dash、HTML 或缓存实现。界面回调负责读取缓存和
渲染；CLI、批处理任务和单元测试可直接复用这里的选择、诊断与对齐流程。
"""
import os
from typing import Any

import pandas as pd

from .alignment import align_trajectories
from .delay_detect import scan_delay
from .matching import extract_radar_trajectory, filter_and_match_ids
from .parser import load_data_file


def _summarize_sample_rates(file_infos: list[dict]) -> dict[str, Any]:
    """汇总各文件独立帧率，避免多文件结果依赖浏览器上传顺序。"""
    file_values = []
    valid_rates = []
    for info in file_infos:
        rate = float(info.get('sample_rate_hz') or 0.0)
        filename = str(info.get('filename') or '')
        file_values.append({'filename': filename, 'sample_rate_hz': rate})
        if pd.notna(rate) and rate > 0:
            valid_rates.append(rate)

    invalid_count = len(file_infos) - len(valid_rates)
    notes = []
    if not valid_rates:
        representative = 0.0
        label = '无法估算'
        notes.append('所有文件均无法根据唯一时间戳估算时间帧率')
    else:
        representative = round(float(pd.Series(valid_rates).median()), 1)
        min_rate = min(valid_rates)
        max_rate = max(valid_rates)
        tolerance = max(0.5, representative * 0.05)
        if max_rate - min_rate > tolerance:
            label = f'{min_rate:.1f}–{max_rate:.1f}Hz'
            notes.append(f'各文件时间帧率不一致（{label}）')
        else:
            label = f'{representative:.1f}Hz'

    if invalid_count:
        notes.append(f'{invalid_count}/{len(file_infos)}个文件无法估算时间帧率，已从汇总中排除')

    return {
        'sample_rate_hz': representative,
        'sample_rate_label': label,
        'sample_rate_note': '；'.join(notes),
        'sample_rate_file_values': file_values,
    }


def _merge_role_infos(infos: list[dict]) -> dict:
    """合并同角色条目：多文件去重排序、采样率与物理量汇总（单条目直接返回）。

    单文件与多文件统一走采样率/物理量汇总，保证回调侧元数据结构一致。
    """
    if len(infos) == 1:
        merged_info = infos[0]
    else:
        merged_df = pd.concat([info['df'] for info in infos], ignore_index=True)
        dedup_subset = ['file_index', 'ID', 'timestamp_parsed']
        if not all(column in merged_df.columns for column in dedup_subset):
            dedup_subset = ['file_index', 'ID', 'timestamp']
        merged_df = merged_df.drop_duplicates(subset=dedup_subset, keep='first')
        merged_df = merged_df.sort_values('timestamp_parsed').reset_index(drop=True)

        merged_info = dict(infos[0])
        merged_info['df'] = merged_df
        merged_info['total_rows'] = len(merged_df)
        merged_info['unique_ids'] = merged_df['ID'].nunique()
        merged_info['filename'] = f'{len(infos)}个文件'
        merged_info['time_range'] = [
            min(info['time_range'][0] for info in infos),
            max(info['time_range'][1] for info in infos),
        ]
        merged_info['warnings'] = [
            warning for info in infos for warning in info.get('warnings', [])
        ]

    rate_summary = _summarize_sample_rates(infos)
    merged_info.update(rate_summary)
    if rate_summary['sample_rate_note']:
        merged_info.setdefault('warnings', []).append(rate_summary['sample_rate_note'])

    # 多文件物理量取原始表头并集，同时记录字段覆盖文件数。界面据此明确
    # 标注“部分文件可用”，不擅自合并大小写不同的原始列名。
    field_summary: dict[str, dict[str, Any]] = {}
    for info in infos:
        for field in info.get('physical_fields', []):
            name = str(field['name'])
            current = field_summary.setdefault(name, {
                **field,
                'files_present': 0,
                'file_count': len(infos),
            })
            current['files_present'] += 1
            current['valid_count'] = max(
                int(current.get('valid_count', 0)), int(field.get('valid_count', 0)),
            )
            current['valid_ratio'] = max(
                float(current.get('valid_ratio', 0)), float(field.get('valid_ratio', 0)),
            )
    merged_info['physical_fields'] = list(field_summary.values())
    merged_info['fields'] = list(dict.fromkeys(
        field for info in infos for field in info.get('fields', [])
    ))
    merged_info['original_to_internal'] = {
        str(original): str(internal)
        for info in infos
        for original, internal in info.get('original_to_internal', {}).items()
    }
    return merged_info


def prepare_comparison_upload(
    files: list[tuple[bytes, str | None]],
    expected_role: str,
) -> dict[str, Any]:
    """解析一次上传的文件，按角色分拣到当前区域与另一区域。

    该函数只处理数据，不依赖 Dash 上传组件。

    角色分拣规则（支持自包含 JSON）：
      - 角色匹配当前区域的条目 → ``info``（原行为不变）；
      - 角色属于另一区域的条目 → ``routed``，由回调在另一区域为空时自动
        填充。录制 JSON 常同时包含雷达话题与真值话题，一次拖入即可完成
        两侧加载，无需把同一文件分别上传两次；
      - 无法识别角色的条目 → 报错（缺少特征列）。

    ``file_index`` 按角色独立编号，因此同一 ID、同一时间戳的跨文件数据
    不会在去重时被错误丢弃。
    """
    other_role = 'rtk' if expected_role == 'radar' else 'radar'
    errors: list[str] = []
    kept_infos: list[dict] = []
    routed_infos: list[dict] = []

    # JSON 文件按话题展开为多条数据（FLR/RLR/真值话题共存时各成一条）
    entries: list[dict] = []
    for content, filename in files:
        safe_name = os.path.basename(str(filename or 'upload.dat')).strip()
        try:
            entries.extend(load_data_file(content, safe_name))
        except Exception as exc:  # 由回调记录详细异常，服务返回可展示的摘要
            errors.append(f'{safe_name}: 解析失败（{exc}）')
            continue

    for info in entries:
        safe_name = str(info.get('filename') or 'upload.dat')
        if info.get('errors'):
            errors.append(f'{safe_name}: {"；".join(info["errors"])}')
            continue
        role = info.get('role')
        if role == expected_role:
            target_infos = kept_infos
        elif role == other_role:
            target_infos = routed_infos
        else:
            errors.append(
                f'{safe_name}: 无法识别数据角色（缺少雷达 Dx/Dy 或真值 center_x/center_y 特征列）'
            )
            continue

        info = dict(info)
        info['df'] = info['df'].copy()
        # file_index 按角色独立从 0 编号（自包含 JSON 的雷达/真值话题互不影响）
        info['df']['file_index'] = len(target_infos)
        info['df']['source_filename'] = safe_name
        target_infos.append(info)

    result: dict[str, Any] = {'info': None, 'errors': errors, 'file_count': 0}
    if kept_infos:
        result['info'] = _merge_role_infos(kept_infos)
        result['file_count'] = len(kept_infos)
    if routed_infos:
        result['routed'] = {
            'role': other_role,
            'info': _merge_role_infos(routed_infos),
            'file_count': len(routed_infos),
        }
    return result


def get_candidate_match_result(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    comparison_config: dict[str, Any],
    cached_ids: list[dict] | None = None,
) -> dict[str, Any]:
    """返回清洗后的候选关联与静止 ID 过滤统计。"""
    if cached_ids:
        return {'candidate_ids': cached_ids, 'filter_stats': None}

    matching_cfg = comparison_config.get('id_matching', {})
    if matching_cfg.get('enabled', True):
        result = filter_and_match_ids(radar_df, rtk_df, comparison_config)
        return {
            'candidate_ids': result['valid_match_ids'],
            'filter_stats': result['filter_stats'],
            'target_stats': result['target_stats'],
        }

    # KDTree/Scipy 只在关闭新版关联器的兼容模式下加载。
    from .id_finder import discover_best_id
    return {
        'candidate_ids': discover_best_id(
            radar_df, rtk_df, comparison_config.get('match_threshold', 5.0),
        ),
        'filter_stats': None,
    }


def get_candidate_ids(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    comparison_config: dict[str, Any],
    cached_ids: list[dict] | None = None,
) -> list[dict]:
    """兼容界面调用：仅返回清洗后的可选雷达目标列表。"""
    return get_candidate_match_result(radar_df, rtk_df, comparison_config, cached_ids)['candidate_ids']


def resolve_track_selection(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    comparison_config: dict[str, Any],
    selected_id: int | None,
    selected_file_index: int | None,
    selected_segment_index: int | None = None,
    cached_ids: list[dict] | None = None,
) -> dict[str, Any]:
    """解析用户选择；未选择时使用重合率最高的候选目标。"""
    candidate_ids = get_candidate_ids(radar_df, rtk_df, comparison_config, cached_ids)
    if selected_id is None:
        if not candidate_ids:
            raise ValueError('未找到可用于对齐的雷达目标')
        selected_id = candidate_ids[0]['track_id']
        selected_file_index = candidate_ids[0].get('file_index')
        selected_segment_index = candidate_ids[0].get('segment_index')

    selected_match = next(
        (
            item for item in candidate_ids
            if (item['track_id'] == selected_id
                and item.get('file_index') == selected_file_index
                and (selected_segment_index is None
                     or item.get('segment_index') == selected_segment_index))
        ),
        None,
    )

    return {
        'track_id': selected_id,
        'file_index': selected_file_index,
        'segment_index': selected_match.get('segment_index') if selected_match else selected_segment_index,
        'rtk_id': selected_match.get('rtk_id') if selected_match else None,
        'rtk_file_index': selected_match.get('rtk_file_index') if selected_match else None,
        'candidate_ids': candidate_ids,
    }


def _select_rtk_target(
    rtk_df: pd.DataFrame,
    rtk_id: int | None,
    rtk_file_index: int | None = None,
) -> pd.DataFrame:
    """关联已确定时只保留对应真值 ID，防止其他真值目标污染插值。"""
    if rtk_id is None or 'ID' not in rtk_df.columns:
        return rtk_df
    selected = rtk_df[rtk_df['ID'] == rtk_id].copy()
    if rtk_file_index is not None and 'file_index' in selected.columns:
        selected = selected[selected['file_index'] == rtk_file_index].copy()
    if selected.empty:
        raise ValueError(f'未找到关联的 RTK ID={rtk_id}')
    return selected


def analyse_selected_track(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    comparison_config: dict[str, Any],
    track_id: int,
    file_index: int | None = None,
    segment_index: int | None = None,
    rtk_id: int | None = None,
    rtk_file_index: int | None = None,
    cached_ids: list[dict] | None = None,
) -> dict[str, Any]:
    """执行坐标系诊断和延迟扫描，返回界面无关的分析结果。"""
    # 坐标诊断依赖 SciPy，延迟到用户实际选择对比目标时再加载。
    from .coord_diag import diagnose_coordinate_system

    selected_radar_df = extract_radar_trajectory(
        radar_df, track_id, file_index, segment_index,
        comparison_config.get('id_matching'),
    )
    selected_rtk_df = _select_rtk_target(rtk_df, rtk_id, rtk_file_index)
    coordinate = diagnose_coordinate_system(
        selected_radar_df,
        selected_rtk_df,
        track_id,
        bias_threshold_m=comparison_config.get('coord_bias_threshold', 0.5),
        file_index=None,
    )
    scan_range = comparison_config.get('delay_scan_range', [-200, 200])
    delay = scan_delay(
        selected_radar_df,
        selected_rtk_df,
        track_id,
        delay_range=tuple(scan_range),
        step_ms=comparison_config.get('delay_scan_step', 10),
        insensitive_ratio=comparison_config.get('delay_insensitive_ratio', 0.05),
        match_threshold_m=comparison_config.get('match_threshold', 5.0),
        file_index=None,
        min_matched_frames=comparison_config.get('delay_min_matched_frames', 3),
        min_match_rate=comparison_config.get('delay_min_match_rate', 0.5),
    )
    suggested_delay_ms = 0 if delay['delay_insensitive'] else delay['optimal_delay_ms']

    return {
        'coordinate': coordinate,
        'delay': delay,
        'suggested_delay_ms': suggested_delay_ms,
        'candidate_ids': get_candidate_ids(radar_df, rtk_df, comparison_config, cached_ids),
    }


def execute_alignment(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    comparison_config: dict[str, Any],
    track_id: int,
    delay_ms: float = 0.0,
    file_index: int | None = None,
    segment_index: int | None = None,
    rtk_id: int | None = None,
    rtk_file_index: int | None = None,
    custom_mappings: list[dict] | None = None,
    merge_segments: bool = False,
    segment_indexes: list[int] | None = None,
) -> dict[str, Any]:
    """执行一次轨迹对齐，统一从配置读取算法阈值。

    ``merge_segments=True`` 时合并该 (ID, 文件) 下全部中断段一起对齐
    （按时间排序，中断区间无雷达帧自然不产生对齐帧；三帧连续性按
    时间间隔自动打断）。
    """
    selected_radar_df = extract_radar_trajectory(
        radar_df, track_id, file_index, segment_index,
        comparison_config.get('id_matching'),
        merge_segments=merge_segments,
        segment_indexes=segment_indexes,
    )
    selected_rtk_df = _select_rtk_target(rtk_df, rtk_id, rtk_file_index)
    result = align_trajectories(
        selected_radar_df,
        selected_rtk_df,
        track_id,
        delay_ms=delay_ms,
        match_threshold_m=comparison_config.get('match_threshold', 5.0),
        time_gate_ms=comparison_config.get('time_gate_ms', 50.0),
        file_index=None,
        custom_mappings=custom_mappings,
    )

    # 绘图层需要当前关联 RTK 文件与 ID 的完整时域曲线。误差统计仍只使用
    # aligned_df 中的雷达对齐帧，完整真值数据不会混入 RMSE 等质量指标。
    if 'timestamp_parsed' in selected_rtk_df.columns:
        rtk_curve_columns = ['timestamp', 'timestamp_parsed', 'center_x', 'center_y', 'Vx', 'Vy']
        rtk_curve_columns.extend(
            str(mapping.get('rtk_col')) for mapping in custom_mappings or []
            if mapping.get('rtk_col')
        )
        rtk_curve_columns = list(dict.fromkeys(
            column for column in rtk_curve_columns if column in selected_rtk_df.columns
        ))
        rtk_curve = selected_rtk_df[rtk_curve_columns].copy()
        result['rtk_curve_df'] = rtk_curve.sort_values('timestamp_parsed').reset_index(drop=True)
    return result


def execute_selected_segments_alignment(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    comparison_config: dict[str, Any],
    selections: list[dict],
    delay_ms: float = 0.0,
    custom_mappings: list[dict] | None = None,
) -> dict[str, Any]:
    """对齐用户勾选的同一目标段，支持这些段来自相邻的多个 CSV 文件。"""
    if not selections:
        raise ValueError('未选择可对齐的轨迹段')

    rtk_ids = {item.get('rtk_id') for item in selections}
    if len(rtk_ids) != 1:
        raise ValueError('勾选段必须关联到同一个真值目标')

    radar_parts = []
    rtk_parts = []
    matching_config = comparison_config.get('id_matching')
    for item in selections:
        radar_parts.append(extract_radar_trajectory(
            radar_df,
            int(item['track_id']),
            item.get('file_index'),
            item.get('segment_index'),
            matching_config,
        ))
        rtk_parts.append(_select_rtk_target(
            rtk_df, item.get('rtk_id'), item.get('rtk_file_index'),
        ))

    selected_radar_df = pd.concat(radar_parts, ignore_index=True).sort_values(
        'timestamp_parsed'
    ).reset_index(drop=True)
    selected_rtk_df = pd.concat(rtk_parts, ignore_index=True).sort_values(
        'timestamp_parsed'
    ).drop_duplicates(subset=['timestamp_parsed'], keep='first').reset_index(drop=True)
    # 合并后的雷达帧可包含不同雷达 ID。align_trajectories 会按 track_id 筛选，
    # 因而在这份仅供本次计算的副本中统一为首段 ID，保留全部已勾选帧。
    track_id = int(selections[0]['track_id'])
    selected_radar_df = selected_radar_df.copy()
    selected_radar_df['ID'] = track_id
    result = align_trajectories(
        selected_radar_df,
        selected_rtk_df,
        track_id,
        delay_ms=delay_ms,
        match_threshold_m=comparison_config.get('match_threshold', 5.0),
        time_gate_ms=comparison_config.get('time_gate_ms', 50.0),
        file_index=None,
        custom_mappings=custom_mappings,
    )
    rtk_curve_columns = ['timestamp', 'timestamp_parsed', 'center_x', 'center_y', 'Vx', 'Vy']
    rtk_curve_columns.extend(
        str(mapping.get('rtk_col')) for mapping in custom_mappings or []
        if mapping.get('rtk_col')
    )
    rtk_curve_columns = list(dict.fromkeys(
        column for column in rtk_curve_columns if column in selected_rtk_df.columns
    ))
    result['rtk_curve_df'] = selected_rtk_df[rtk_curve_columns].copy()
    return result
