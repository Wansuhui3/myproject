"""真值对比业务编排服务。

本模块只组合核心算法，不依赖 Dash、HTML 或缓存实现。界面回调负责读取缓存和
渲染；CLI、批处理任务和单元测试可直接复用这里的选择、诊断与对齐流程。
"""
from typing import Any

import pandas as pd

from .alignment import align_trajectories
from .coord_diag import diagnose_coordinate_system
from .delay_detect import scan_delay
from .id_finder import discover_best_id
from .matching import extract_radar_trajectory, filter_and_match_ids
from .parser import load_csv_file


def prepare_comparison_upload(
    files: list[tuple[bytes, str | None]],
    expected_role: str,
) -> dict[str, Any]:
    """解析并合并一次上传的同类 CSV 文件。

    该函数只处理数据，不依赖 Dash 上传组件。不同来源文件保留 ``file_index``，
    因而同一 ID、同一时间戳的跨文件数据不会在去重时被错误丢弃。
    """
    valid_infos: list[dict] = []
    errors: list[str] = []

    for file_index, (content, filename) in enumerate(files):
        safe_name = filename or f'file_{file_index + 1}.csv'
        try:
            info = load_csv_file(content, safe_name)
        except Exception as exc:  # 由回调记录详细异常，服务返回可展示的摘要
            errors.append(f'{safe_name}: 解析失败（{exc}）')
            continue

        if info.get('errors'):
            errors.append(f'{safe_name}: {"；".join(info["errors"])}')
            continue
        if info.get('role') != expected_role:
            actual_role = info.get('role', 'unknown')
            role_label = {'radar': '雷达', 'rtk': 'RTK真值', 'unknown': '未知格式'}.get(
                actual_role, actual_role,
            )
            expected_label = '雷达' if expected_role == 'radar' else 'RTK真值'
            errors.append(f'{safe_name}: 检测为{role_label}文件，请上传到{expected_label}区域')
            continue

        info = dict(info)
        info['df'] = info['df'].copy()
        info['df']['file_index'] = file_index
        valid_infos.append(info)

    if not valid_infos:
        return {'info': None, 'errors': errors, 'file_count': 0}

    if len(valid_infos) == 1:
        merged_info = valid_infos[0]
    else:
        merged_df = pd.concat([info['df'] for info in valid_infos], ignore_index=True)
        dedup_subset = ['file_index', 'ID', 'timestamp_parsed']
        if not all(column in merged_df.columns for column in dedup_subset):
            dedup_subset = ['file_index', 'ID', 'timestamp']
        merged_df = merged_df.drop_duplicates(subset=dedup_subset, keep='first')
        merged_df = merged_df.sort_values('timestamp_parsed').reset_index(drop=True)

        merged_info = dict(valid_infos[0])
        merged_info['df'] = merged_df
        merged_info['total_rows'] = len(merged_df)
        merged_info['unique_ids'] = merged_df['ID'].nunique()
        merged_info['filename'] = f'{len(valid_infos)}个文件'
        merged_info['time_range'] = [
            min(info['time_range'][0] for info in valid_infos),
            max(info['time_range'][1] for info in valid_infos),
        ]
        merged_info['warnings'] = [
            warning for info in valid_infos for warning in info.get('warnings', [])
        ]

    return {
        'info': merged_info,
        'errors': errors,
        'file_count': len(valid_infos),
    }


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
) -> dict[str, Any]:
    """执行一次轨迹对齐，统一从配置读取算法阈值。"""
    selected_radar_df = extract_radar_trajectory(
        radar_df, track_id, file_index, segment_index,
        comparison_config.get('id_matching'),
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
    )

    # RTK 必须在雷达缺帧区间保持连续，但不应绘制整份文件而把雷达曲线
    # 压缩到图表边缘。使用当前选中轨迹的时间范围，加一个小缓冲窗口。
    if not selected_radar_df.empty and 'timestamp_parsed' in selected_rtk_df.columns:
        padding_sec = float(comparison_config.get('rtk_curve_padding_ms', 100.0)) / 1000.0
        start = float(selected_radar_df['timestamp_parsed'].min()) - padding_sec
        end = float(selected_radar_df['timestamp_parsed'].max()) + padding_sec
        rtk_curve = selected_rtk_df[
            (selected_rtk_df['timestamp_parsed'] >= start)
            & (selected_rtk_df['timestamp_parsed'] <= end)
        ].copy()
        result['rtk_curve_df'] = rtk_curve.sort_values('timestamp_parsed').reset_index(drop=True)
    return result
