"""性能验收域共享 helpers：段键解析、摘要快照标识与导出命名。

本模块不导入任何回调模块（纯函数），供 perf_run / perf_panel /
perf_export 三个回调域共同引用。
"""


def _parse_segment_key(key: str, candidates: list[dict]) -> dict | None:
    """解析段键 'track:::file:::seg' 并在候选列表中匹配完整候选记录。"""
    parts = str(key).split(':::')
    if len(parts) != 3:
        return None
    track_id, file_token, seg_token = parts
    for candidate in candidates:
        cand_seg = candidate.get('segment_index')
        if (str(candidate.get('track_id')) == track_id
                and str(candidate.get('file_index')) == file_token
                and (cand_seg is None or str(cand_seg) == seg_token)):
            return candidate
    return None


def _perf_summary_source_key(state: dict | None) -> str:
    """返回摘要快照所属数据源的稳定标识；不同上传批次不混合追加。"""
    state = state or {}
    def _part(role: str) -> str:
        meta = state.get(f'{role}_meta') or {}
        return '|'.join([
            str(meta.get('filename') or ''),
            str(meta.get('total_rows') or ''),
            str(meta.get('time_range') or ''),
        ])
    return f'{_part("radar")}::{_part("rtk")}'


def _perf_summary_snapshot_label(state: dict) -> str:
    """构建可复制摘要中每次插入结果的来源说明。"""
    radar_name = str((state.get('radar_meta') or {}).get('filename') or '雷达数据')
    rtk_name = str((state.get('rtk_meta') or {}).get('filename') or 'RTK数据')
    target = state.get('selected_id')
    segment = state.get('selected_segment_index')
    merged_count = state.get('merged_segments')
    delay = state.get('delay_ms') or 0
    target_text = f'ID={target}' if target is not None else 'ID=—'
    if merged_count:
        segment_text = f'合并{merged_count}段'
    else:
        segment_text = f'段{segment + 1}' if isinstance(segment, int) else '段—'
    return f'{radar_name} · {rtk_name} · {target_text} · {segment_text} · 补偿 {delay}ms'


def _selected_radar_filename(state: dict) -> str:
    """返回选中轨迹段来源的雷达 CSV 文件名。

    优先从 ID 候选缓存（cached_id_list）按 file_index + track_id +
    segment_index 匹配出实际来源文件名；多文件上传时这是"数据所在的
    雷达 csv 文件名"的精确值。无记录时回退上传批次标签（radar_meta）。
    """
    fallback = str((state.get('radar_meta') or {}).get('filename') or '雷达数据')
    file_index = state.get('selected_file_index')
    track_id = state.get('selected_id')
    seg_idx = state.get('selected_segment_index')
    if file_index is None:
        return fallback
    for candidate in state.get('cached_id_list') or []:
        if candidate.get('file_index') != file_index:
            continue
        if str(candidate.get('track_id')) != str(track_id):
            continue
        cand_seg = candidate.get('segment_index')
        if (seg_idx is not None and cand_seg is not None
                and int(cand_seg) != int(seg_idx)):
            continue
        return str(candidate.get('radar_filename') or fallback)
    return fallback
