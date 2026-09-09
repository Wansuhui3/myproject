"""波动分析纯渲染辅助：轨迹表格、标题栏、目标 ID 列表。

被 wave_callbacks（交互域）与 wave_upload_callbacks（数据加载域）共享；
不得导入任何回调模块。
"""
from dash import html

from ..cache import get_meta_df


def _build_trajectory_table(id_meta):
    """构建轨迹段表格 HTML 并返回 (table, first_traj_id)。

    两个位置（on_timestamp_input / on_id_click）共用此逻辑。
    """
    table_header = html.Thead(html.Tr([
        html.Th('时间区间', className='traj-time-header'),
        html.Th('帧数', className='traj-frames-header'),
    ]))
    rows = []
    first_traj_id = None
    for _, seg in id_meta.iterrows():
        traj_id = seg['trajectory_id']
        if first_traj_id is None:
            first_traj_id = traj_id
        label = seg.get('display_label', traj_id)
        time_content = [
            html.Span(
                str(label),
                className='traj-time-label',
                title=str(label),
            ),
        ]
        if seg.get('spatial_anomaly'):
            time_content.append(
                html.Span('⚠空间跳变', className='spatial-anomaly-badge')
            )
        if seg.get('timestamp_position_conflict'):
            time_content.append(
                html.Span('⚠同时间戳位置冲突', className='spatial-anomaly-badge')
            )
        rows.append(html.Tr(
            id={'type': 'traj-row', 'index': traj_id},
            n_clicks=0,
            **{'data-traj-id': traj_id},
            children=[
                html.Td(time_content, className='traj-time-cell'),
                html.Td(str(seg['total_frames']), className='traj-frames-cell'),
            ],
        ))
    table = html.Table([table_header, html.Tbody(rows)], className='traj-table')
    return table, first_traj_id


def _build_title_bar(
        traj_id: str, selected_qties: list, n_frames: int,
        unavailable_qties: list[str] | None = None,
) -> html.Span:
    """构建图表标题栏：显示目标ID + 轨迹段时间区间 + 帧数。"""
    meta_df = get_meta_df()
    time_range_label = traj_id
    original_id = ''
    spatial_anom = False
    timestamp_conflict = False
    source_key = ''
    source_label = ''
    if meta_df is not None and traj_id is not None:
        row = meta_df[meta_df['trajectory_id'] == traj_id]
        if len(row) > 0:
            original_id = str(row.iloc[0].get('original_id', ''))
            if row.iloc[0].get('display_label'):
                time_range_label = f'{row.iloc[0]["display_label"]}'
            else:
                time_range_label = str(traj_id)
            spatial_anom = bool(row.iloc[0].get('spatial_anomaly', False))
            timestamp_conflict = bool(
                row.iloc[0].get('timestamp_position_conflict', False)
            )
            source_key = str(row.iloc[0].get('radar_source_key', ''))
            source_label = str(row.iloc[0].get('radar_source_short_label', ''))

    parts = []
    if source_label:
        parts.append(html.Span(
            source_label,
            className=f'radar-source-badge radar-source-{source_key}',
        ))
    parts.append(html.Span(f'ID: {original_id}', className='traj-id-badge'))
    parts.append(html.Span(' | ', style={'color': '#94a3b8'}))
    parts.append(html.Span(time_range_label, className='traj-name'))
    parts.append(html.Span(' | ', style={'color': '#94a3b8'}))
    parts.append(html.Span(f'{n_frames} frames', className='frame-count'))
    if spatial_anom:
        parts.append(html.Span(' | ', style={'color': '#94a3b8'}))
        parts.append(html.Span('⚠空间跳变', className='spatial-anomaly-badge'))
    if timestamp_conflict:
        parts.append(html.Span(' | ', style={'color': '#94a3b8'}))
        parts.append(html.Span(
            '⚠同时间戳位置冲突', className='spatial-anomaly-badge'
        ))
    if unavailable_qties:
        parts.append(html.Span(' | ', style={'color': '#94a3b8'}))
        parts.append(html.Span(
            f'未绘制：{", ".join(unavailable_qties)}（当前轨迹无有效数据）',
            className='feedback-muted',
        ))
    return html.Span(parts)


def _target_key(source_key: str, id_val: int):
    return f'{source_key}::{int(id_val)}' if source_key else int(id_val)


def _target_parts(target) -> tuple[str, int]:
    text = str(target)
    if '::' in text:
        source_key, raw_id = text.split('::', 1)
        return source_key, int(raw_id)
    return '', int(text)


def _target_meta(meta_df, target):
    source_key, id_val = _target_parts(target)
    mask = meta_df['original_id'] == id_val
    if source_key and 'radar_source_key' in meta_df.columns:
        mask &= meta_df['radar_source_key'].astype(str) == source_key
    return meta_df[mask]


def _target_badge(source_key: str, source_label: str):
    if not source_key:
        return None
    return html.Span(
        source_label or source_key.upper(),
        className=f'radar-source-badge radar-source-{source_key}',
    )


def _build_id_list_html(df, meta_df, center_ts=None, selected_target=None):
    """构建目标ID列表HTML。

    Args:
        df: 合并后的数据DataFrame
        meta_df: 分段元信息DataFrame
        center_ts: 可选，参考时间戳；提供时按时间距离排序并高亮最近ID
        selected_target: 当前选中的目标键（``source_key::id`` 或 int）

    Returns:
        (list_children, nearest_id_or_None)
    """
    if 'radar_source_key' in meta_df.columns:
        target_rows = (
            meta_df[['radar_source_key', 'radar_source_short_label', 'original_id']]
            .drop_duplicates()
            .sort_values(['radar_source_key', 'original_id'])
        )
        targets = [
            (str(row['radar_source_key']), str(row['radar_source_short_label']), int(row['original_id']))
            for _, row in target_rows.iterrows()
        ]
    else:
        targets = [('', '', int(id_val)) for id_val in sorted(df['ID'].unique())]

    if center_ts is None:
        # 无时间戳 → 仅显示ID和段数，按ID排序
        list_children = []
        for source_key, source_label, id_val in targets:
            target = _target_key(source_key, id_val)
            id_meta = _target_meta(meta_df, target)
            label_children = []
            badge = _target_badge(source_key, source_label)
            if badge is not None:
                label_children.append(badge)
            label_children.append(html.Span(f'ID: {id_val}'))
            cls = 'id-list-item selected' if target == selected_target else 'id-list-item'
            list_children.append(html.Div([
                html.Span(label_children, className='id-text radar-target-label'),
                html.Span(f'{len(id_meta)}段', className='id-meta'),
            ], id={'type': 'id-list-item', 'index': target},
               n_clicks=0, className=cls,
               **{'data-id': str(target)}))
        return list_children, None

    # 有时间戳 → 计算每个ID距离输入时间戳最近点的时间差
    id_items = []
    for source_key, source_label, id_val in targets:
        target = _target_key(source_key, id_val)
        id_meta = _target_meta(meta_df, target)
        id_mask = df['ID'] == id_val
        if source_key and 'radar_source_key' in df.columns:
            id_mask &= df['radar_source_key'].astype(str) == source_key
        id_df = df[id_mask]
        if len(id_df) == 0:
            continue
        time_diffs = (id_df['timestamp_parsed'] - center_ts).dt.total_seconds()
        min_diff_idx = time_diffs.abs().idxmin()
        diff_sec = time_diffs[min_diff_idx]
        sign = '+' if diff_sec >= 0 else ''
        id_items.append({
            'id': target,
            'source_key': source_key,
            'source_label': source_label,
            'seg_count': len(id_meta),
            'diff_sec': diff_sec,
            'display': f'ID: {int(id_val)}',
            'meta': f'{sign}{diff_sec:.1f}s, {len(id_meta)}段',
            'abs_diff': abs(diff_sec),
        })

    if not id_items:
        return [html.Div('无有效目标', className='id-list-item',
                         style={'cursor': 'default', 'color': '#94a3b8'})], None

    id_items.sort(key=lambda x: x['abs_diff'])
    nearest_id = id_items[0]['id']

    list_children = []
    for item in id_items:
        active_target = nearest_id if selected_target is None else selected_target
        is_selected = (item['id'] == active_target)
        cls = 'id-list-item selected' if is_selected else 'id-list-item'
        label_children = []
        badge = _target_badge(item['source_key'], item['source_label'])
        if badge is not None:
            label_children.append(badge)
        label_children.append(html.Span(item['display']))
        list_children.append(html.Div([
            html.Span(label_children, className='id-text radar-target-label'),
            html.Span(item['meta'], className='id-meta'),
        ], id={'type': 'id-list-item', 'index': item['id']},
           n_clicks=0, className=cls,
           **{'data-id': str(item['id'])}))

    return list_children, nearest_id
