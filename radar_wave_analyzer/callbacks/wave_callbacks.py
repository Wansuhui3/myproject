"""波动分析回调：上传解析 → 时间筛选 → ID/轨迹选择 → 绘图 → 统计。

回调清单：
  on_radar_change_label / on_radar_change_clear → 雷达切换 → 更新位置标签 / 清空页面
  on_upload_csv            → 拖拽上传 → 解析CSV/JSON → 分段 → 缓存
  on_timestamp_input       → 筛选时间窗口 → 提取ID列表
  on_id_click              → 自定义 ListGroup 点击选中
  on_trajectory_select     → 轨迹段选中 → 绘制多子图 + 全段统计
  on_quantity_change       → 物理量多选变更 → 重建子图 + 重算统计
  on_box_select            → 框选 → 全子图高亮 + 统计
  on_clear_box_select      → 清除框选
  refresh_quantity_options → 数据加载后刷新物理量多选项
  update_wave_summary_snapshots / render_wave_summary_content
                           → 波动摘要快照插入/移除/渲染
  on_wave_clear            → 清空波动分析页面
"""
import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, Patch, callback, no_update, html, ALL
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..config import get

from ..cache import (
    set_data_cache,
    get_df, get_meta_df, get_radar_position, get_file_path,
    get_segment, has_data_loaded,
    clear_data_cache,
    switch_radar,
)

from ..core.data_loader import (
    identify_radar_source, load_csv_from_bytes, parse_timestamp, get_time_range,
)
from ..core.json_loader import is_json_filename, load_json_from_bytes

from ..core.segmenter import segment_trajectories

from ..core.selection import extract_x_selection

from ..core.wave_calc import (
    compute_segment_stats, compute_fluctuation_stats, find_max_jump,
)

from ..components.wave_stats_panel import (
    render_multi_full_stats, render_multi_full_stats_placeholder,
    render_multi_box_stats, render_box_stats_empty,
    render_wave_snapshot_summary,
)

from ..components.graph_builder import (
    BOX_JUMP_COLOR, build_box_jump_shapes,
    build_highlight_shapes, build_multi_subplot_graph,
)

from .helpers import _decode_upload_contents, _summary_plain_text

logger = logging.getLogger(__name__)


# ===================== 辅助函数 =====================

_QUANTITY_METADATA_COLUMNS = frozenset({
    'timestamp', 'timestamp_parsed', 'ID', 'Track_Age', 'file_index',
    'csv_row', 'radar_frame', 'rtk_frame', 'time_diff_ms',
    'radar_source_key', 'radar_source_label', 'source_filename',
    'MotionStatus', 'MeasurementState', 'ExistProb',
})


def _discover_quantity_columns(df: pd.DataFrame) -> list[str]:
    """从当前 CSV 实际字段发现可绘制物理量。

    配置文件只提供单位/显示名；是否出现在波动页由当前数据源中是否存在
    且含有限数值决定。所有内部列、时间列、ID 和状态/来源字段均排除。
    """
    if df is None or df.empty:
        return []

    columns: list[str] = []
    for raw_column in df.columns:
        column = str(raw_column)
        if (column in _QUANTITY_METADATA_COLUMNS
                or column.startswith('__')
                or column.startswith('wave_')
                or column.startswith('radar_source_')):
            continue
        values = pd.to_numeric(df[raw_column], errors='coerce').to_numpy(dtype=float)
        if np.isfinite(values).any():
            columns.append(column)
    return columns


def _ensure_valid_quantities(seg_df, selected_qties) -> list:
    """返回当前轨迹真正可绘制的已选字段，绝不静默改画 Dx。"""
    available = set(_discover_quantity_columns(seg_df))
    requested = [str(q) for q in (selected_qties or [])]
    return [qty for qty in requested if qty in available]


def _unavailable_quantities(seg_df, selected_qties) -> list[str]:
    """返回用户已选但当前轨迹不存在或全为空的字段。"""
    available = set(_discover_quantity_columns(seg_df))
    return [str(q) for q in (selected_qties or []) if str(q) not in available]


def _get_non_highlight_shapes(figure: dict | None) -> list:
    """保留图表自身线形（图例与全段最大跳变红线），移除上一次框选产生的
    矩形与框选区间最大跳变紫色高亮线。"""
    if not isinstance(figure, dict):
        return []
    shapes = figure.get('layout', {}).get('shapes', [])
    return [
        shape for shape in shapes
        if isinstance(shape, dict)
        and shape.get('type') != 'rect'
        and (shape.get('line') or {}).get('color') != BOX_JUMP_COLOR
    ]


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


def _compute_quantities_stats(seg_df, selected_qties, mask=None, diff_cache=None):
    """为所有勾选物理量计算 segment_stats。

    Returns:
        (stats_per_qty, first_frame_dists)
    """
    stats_per_qty = {}
    for qty in selected_qties:
        if qty in seg_df.columns:
            stats_per_qty[qty] = compute_segment_stats(seg_df, qty, mask=mask, diff_cache=diff_cache)
        else:
            stats_per_qty[qty] = None

    fluct = compute_fluctuation_stats(seg_df, mask=mask, diff_cache=diff_cache)
    first_frame_dists = fluct.get('first_frame_dists') if fluct else None

    return stats_per_qty, first_frame_dists


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


# ============================================================
# 回调 [0a]: 雷达位置变更 → 仅更新位置标签（允许首次加载时触发）
# ============================================================
@callback(
    Output('radar-position-label', 'children'),
    Input('radar-selector', 'value'),
)
def on_radar_change_label(radar_key: str):
    if not radar_key:
        raise PreventUpdate
    # 设置当前雷达（首次加载或切换时均需调用，确保 cache 知道当前雷达）
    switch_radar(radar_key)
    radar_sources = get('radar_sources', {})
    radar_info = radar_sources.get(radar_key, {})
    radar_label = radar_info.get('label', radar_key)
    return f'[{radar_label}]'


# ============================================================
# 回调 [0b]: 雷达位置变更 → 切换缓存并恢复/重置下游状态
# ============================================================
@callback(
    Output('store-data-loaded', 'data', allow_duplicate=True),
    Output('store-selected-id', 'data', allow_duplicate=True),
    Output('store-selected-trajectory', 'data', allow_duplicate=True),
    Output('store-box-selection', 'data', allow_duplicate=True),
    Output('id-list-container', 'children', allow_duplicate=True),
    Output('id-count-badge', 'children', allow_duplicate=True),
    Output('trajectory-table', 'children', allow_duplicate=True),
    Output('traj-id-badge', 'children', allow_duplicate=True),
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('graph-title-bar', 'children', allow_duplicate=True),
    Output('stats-full-content', 'children', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('box-select-feedback', 'children', allow_duplicate=True),
    Output('current-trajectory-label', 'children', allow_duplicate=True),
    Output('upload-feedback', 'children', allow_duplicate=True),
    Output('timestamp-feedback', 'children', allow_duplicate=True),
    Input('radar-selector', 'value'),
    prevent_initial_call=True,
)
def on_radar_change_clear(radar_key: str):
    if not radar_key:
        raise PreventUpdate

    has_data = switch_radar(radar_key)

    if has_data:
        # 该雷达已有缓存数据 → 恢复可用状态
        fp = get_file_path() or '已缓存'
        df = get_df()
        t_min, t_max = get_time_range(df) if df is not None else (None, None)
        time_info = ''
        if t_min is not None and t_max is not None:
            time_info = (
                f' | {t_min.strftime("%Y-%m-%d %H:%M:%S")} ~ '
                f'{t_max.strftime("%Y-%m-%d %H:%M:%S")}'
            )
        radar_sources = get('radar_sources', {})
        radar_label = radar_sources.get(radar_key, {}).get('label', radar_key)
        feedback = f'[{radar_label}] 已就绪: {fp}{time_info}（输入时间戳查看目标）'
        return (
            True,                                    # store-data-loaded
            None,                                    # store-selected-id
            None,                                    # store-selected-trajectory
            None,                                    # store-box-selection
            [html.Div('请输入时间戳', className='id-list-item',
                      style={'cursor': 'default', 'color': '#94a3b8'})],
            '',                                      # id-count-badge
            None,                                    # trajectory-table
            '',                                      # traj-id-badge
            {'layout': {}},                          # trajectory-graph
            html.Span('请选择轨迹段查看', className='feedback-muted'),
            render_multi_full_stats_placeholder(),   # stats-full-content
            render_box_stats_empty(),                # stats-box-content
            '',                                      # box-select-feedback
            '',                                      # current-trajectory-label
            html.Span(feedback, style={'color': '#15803d', 'fontWeight': '500'}),
            '',                                      # timestamp-feedback
        )
    else:
        # 该雷达无缓存数据 → 清空 UI 提示上传
        return (
            False,                                   # store-data-loaded
            None,                                    # store-selected-id
            None,                                    # store-selected-trajectory
            None,                                    # store-box-selection
            [html.Div('请先上传数据', className='id-list-item',
                      style={'cursor': 'default', 'color': '#94a3b8'})],
            '',                                      # id-count-badge
            None,                                    # trajectory-table
            '',                                      # traj-id-badge
            {'layout': {}},                          # trajectory-graph
            html.Span('请先拖入数据并选择轨迹', className='feedback-muted'),
            render_multi_full_stats_placeholder(),   # stats-full-content
            render_box_stats_empty(),                # stats-box-content
            '',                                      # box-select-feedback
            '',                                      # current-trajectory-label
            '请上传数据',                             # upload-feedback
            '',                                      # timestamp-feedback
        )


# ============================================================
# 回调 [1]: 拖拽上传多CSV → 解析 → 合并 → 分段 → 缓存
# ============================================================
@callback(
    Output('store-data-loaded', 'data'),
    Output('upload-feedback', 'children'),
    Output('id-list-container', 'children', allow_duplicate=True),
    Output('id-count-badge', 'children', allow_duplicate=True),
    Output('trajectory-table', 'children', allow_duplicate=True),
    Output('traj-id-badge', 'children', allow_duplicate=True),
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('stats-full-content', 'children', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('graph-title-bar', 'children', allow_duplicate=True),
    Output('current-trajectory-label', 'children', allow_duplicate=True),
    Output('store-selected-trajectory', 'data', allow_duplicate=True),
    Output('store-selected-id', 'data', allow_duplicate=True),
    Output('store-box-selection', 'data', allow_duplicate=True),
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename'),
    State('radar-selector', 'value'),
    prevent_initial_call=True,
)
def on_upload_csv(contents_list, filenames, radar_key):
    """拖拽上传CSV → 解析合并 → 分段缓存。"""
    def _clear_state(err_msg=None):
        return (
            False,
            err_msg or '已清空',
            [html.Div('请先输入时间戳', className='id-list-item',
                      style={'cursor': 'default', 'color': '#94a3b8'})],
            '',
            None,
            '',
            {'layout': {}},
            render_multi_full_stats_placeholder(),
            render_box_stats_empty(),
            html.Span('请先拖入数据并选择轨迹', className='feedback-muted'),
            '',
            None,
            None,
            None,
        )

    if not contents_list:
        return _clear_state('未收到文件内容，请重新拖入')

    # 从前端获取当前选中的雷达，避免依赖缓存（切换雷达后缓存已被清空）
    if not radar_key:
        radar_sources = get('radar_sources', {})
        radar_key = next(iter(radar_sources.keys()), 'unknown')

    if isinstance(contents_list, str):
        contents_list = [contents_list]
    if isinstance(filenames, str):
        filenames = [filenames]
    elif not filenames:
        filenames = [f'file_{i+1}.csv' for i in range(len(contents_list))]

    all_dfs = []
    errors = []
    source_files: dict[str, dict] = {}
    # file_index 按展开后条目递增：JSON 单文件多话题（FLR/RLR 共存）时，
    # 每个话题独立编号，防止不同雷达同 ID 同时间戳的数据在合并去重时被误删。
    file_seq = 0
    for content, fn in zip(contents_list, filenames):
        try:
            data_bytes = _decode_upload_contents(content)
            if is_json_filename(fn):
                json_result = load_json_from_bytes(data_bytes, fn)
                # 错误消息已含 "文件名·话题名" 前缀，无需重复包装
                errors.extend(json_result['errors'])
                # payload: (DataFrame, 来源识别名, 溯源显示名)
                payloads = [
                    (topic['df'], topic['topic'], topic['label'])
                    for topic in json_result['topics']
                ]
            else:
                df = load_csv_from_bytes(data_bytes, fn)
                payloads = [(df, str(fn), str(fn))]
        except ValueError as e:
            errors.append(f'{fn}: {e}')
            continue
        except Exception as e:
            errors.append(f'{fn}: {e}')
            continue

        for df, source_name, display_name in payloads:
            # 空白文件（0 字节/仅表头/全空行）会解析出 0 行 DataFrame，
            # 必须跳过，否则后续 df['radar_source_group'].iloc[0] 会抛
            # IndexError 导致回调 500、前端一直显示加载中。
            if df is None or len(df) == 0:
                errors.append(f'{display_name}: 文件内容为空或无有效数据行')
                continue
            # JSON 用话题名识别雷达来源（如 RLR_OBJ_object 命中既有 rlr 规则）；
            # CSV 保持按文件名识别，行为与历史版本一致。
            source = identify_radar_source(source_name)
            df['file_index'] = file_seq
            df['source_filename'] = display_name
            df['radar_source_key'] = str(source['key'])
            df['radar_source_label'] = str(source['label'])
            df['radar_source_short_label'] = str(source['short_label'])
            df['radar_source_recognized'] = bool(source['recognized'])
            # 未识别来源互相隔离（按条目粒度），防止未知来源的同号 ID 重新交织。
            df['radar_source_group'] = (
                str(source['key']) if source['recognized'] else f'unknown_file_{file_seq}'
            )
            summary_key = str(df['radar_source_group'].iloc[0])
            source_summary = source_files.setdefault(summary_key, {
                'key': str(source['key']),
                'short_label': str(source['short_label']),
                'recognized': bool(source['recognized']),
                'filenames': [],
            })
            source_summary['filenames'].append(display_name)
            all_dfs.append(df)
            file_seq += 1

    if not all_dfs:
        err_msg = html.Span(f'所有文件解析失败: {"；".join(errors)}', style={'color': '#dc2626'})
        return _clear_state(err_msg)

    if len(all_dfs) == 1:
        merged_df = all_dfs[0]
    else:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        # 跨文件、跨雷达的相同 ID/时间戳均保留；仅文件内部去重。
        merged_df = merged_df.drop_duplicates(subset=['file_index', 'ID', 'timestamp_parsed'], keep='first')
        merged_df = merged_df.sort_values('timestamp_parsed').reset_index(drop=True)

    # 分段缓存只保存源行号，不长期保留每段 DataFrame 的重复副本。
    # 全局分段的结果同时是各来源缓存的唯一来源：不能再为每个来源重复
    # segment_trajectories()，否则上传 N 个来源会额外执行 N 次完整分段扫描。
    source_row_column = '__source_row_index__'
    merged_df[source_row_column] = np.arange(len(merged_df), dtype=np.int64)
    meta_df, segments = segment_trajectories(merged_df)
    # segment_trajectories 已为每段保留独立副本，故可从主表移除内部列；
    # set_data_cache 仍能从 segments 压缩出行号索引。
    merged_df.drop(columns=[source_row_column], inplace=True)

    upload_label = filenames[0] if len(filenames) == 1 else f'{len(filenames)}个文件'
    # 混合上传时不能把 FLR+RLR 合并数据只写入当前一个 radar_key。
    # 否则切换到另一个雷达源时会命中空缓存，主可视化页面被清空。
    # 保留一个 combined 缓存，同时为每个已识别来源建立独立缓存。
    clear_data_cache()
    set_data_cache(upload_label, 'combined', merged_df, meta_df, segments)
    source_keys = {
        str(value).strip()
        for value in merged_df.get('radar_source_key', pd.Series(dtype=str)).dropna().unique()
        if str(value).strip()
    }
    for source_key in sorted(source_keys):
        # merged_df 的索引仍是全局源行号；为来源副本显式保留该映射，
        # 之后再 reset_index 生成其本地 iloc 行号。
        source_df = merged_df[
            merged_df['radar_source_key'].astype(str) == source_key
        ].copy()
        source_df[source_row_column] = source_df.index.to_numpy(dtype=np.int64)
        source_df.reset_index(drop=True, inplace=True)
        if source_df.empty:
            continue

        # 将全局段内的源行号映射为 source_df 的本地 iloc 行号。这样既能
        # 保留 get_segment() 的紧凑索引契约，又不需要为来源副本重新分段。
        source_meta = meta_df[
            meta_df['radar_source_key'].astype(str) == source_key
        ].copy()
        global_to_local = pd.Series(
            np.arange(len(source_df), dtype=np.int64),
            index=source_df[source_row_column].to_numpy(dtype=np.int64),
        )
        source_segments = {}
        for trajectory_id in source_meta.get('trajectory_id', pd.Series(dtype=str)):
            global_segment = segments.get(trajectory_id)
            if global_segment is None:
                continue
            local_rows = global_to_local.reindex(
                global_segment[source_row_column].to_numpy(dtype=np.int64)
            ).to_numpy()
            if np.isnan(local_rows).any():
                logger.warning('来源 %s 的轨迹段 %s 索引映射不完整，已跳过',
                               source_key, trajectory_id)
                continue
            source_segments[trajectory_id] = pd.DataFrame({
                source_row_column: local_rows.astype(np.int64),
            })
        source_df.drop(columns=[source_row_column], inplace=True)
        set_data_cache(upload_label, source_key, source_df, source_meta, source_segments)

    # 恢复上传前选择的来源；若该来源不在本次数据中则使用 combined。
    active_key = radar_key if radar_key in source_keys else 'combined'
    switch_radar(active_key)

    t_min, t_max = get_time_range(merged_df)
    err_suffix = f'（部分失败: {"；".join(errors)}）' if errors else ''
    source_badges = []
    for info in source_files.values():
        source_class = (
            f'radar-source-{info["key"]}' if info['recognized']
            else 'radar-source-unknown'
        )
        source_badges.append(html.Span(
            f'{info["short_label"]} · {len(info["filenames"])}个文件',
            className=f'radar-source-badge {source_class}',
            title='\n'.join(info['filenames']),
        ))
    unknown_warning = ''
    if any(not info['recognized'] for info in source_files.values()):
        unknown_warning = ' | ⚠ 有文件未识别来源，请检查文件名（支持 FLR/RLR）'
    feedback = html.Div([
        html.Div(source_badges, className='radar-source-summary'),
        html.Div(
            f'已加载: {upload_label} | {len(merged_df)}行, {len(meta_df)}段 | '
            f'{t_min.strftime("%Y-%m-%d %H:%M:%S")} ~ '
            f'{t_max.strftime("%Y-%m-%d %H:%M:%S")}{unknown_warning} {err_suffix}',
            className='upload-result-text',
        ),
    ], className='upload-source-result')

    # 上传成功后直接显示全部目标ID
    active_df = get_df()
    if active_df is None:
        active_df = merged_df
    active_meta = get_meta_df()
    if active_meta is None:
        active_meta = meta_df
    list_children, _ = _build_id_list_html(active_df, active_meta)

    return (
        True,                                    # store-data-loaded
        feedback,                                # upload-feedback
        list_children,                           # id-list-container
        str(len(list_children)),                 # id-count-badge
        None,                                    # trajectory-table
        '',                                      # traj-id-badge
        {'layout': {}},                          # trajectory-graph
        render_multi_full_stats_placeholder(),   # stats-full-content
        render_box_stats_empty(),                # stats-box-content
        html.Span('请选择轨迹段查看', className='feedback-muted'),  # graph-title-bar
        '',                                      # current-trajectory-label
        None,                                    # store-selected-trajectory
        None,                                    # store-selected-id
        None,                                    # store-box-selection
        )


# ============================================================
# 回调 [2]: 时间戳输入 → 筛选时间窗口 → 提取ID列表
# ============================================================
@callback(
    Output('id-list-container', 'children'),
    Output('id-count-badge', 'children'),
    Output('timestamp-feedback', 'children'),
    Output('trajectory-table', 'children', allow_duplicate=True),
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('stats-full-content', 'children', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('current-trajectory-label', 'children'),
    Output('graph-title-bar', 'children', allow_duplicate=True),
    Output('store-selected-trajectory', 'data', allow_duplicate=True),
    Output('store-box-selection', 'data', allow_duplicate=True),
    Output('store-selected-id', 'data', allow_duplicate=True),
    Input('timestamp-input', 'value'),
    State('store-data-loaded', 'data'),
    State('store-selected-quantities', 'data'),
    prevent_initial_call=True,
)
def on_timestamp_input(ts_input: str, data_loaded: bool, selected_qties):
    if not data_loaded or get_meta_df() is None:
        logger.warning(
            f'on_timestamp_input: skipped — data_loaded={data_loaded}, '
            f'has_meta={get_meta_df() is not None}, '
            f'radar={get_radar_position()}'
        )
        raise PreventUpdate

    if not ts_input:
        empty = [html.Div('输入时间戳后显示目标', className='id-list-item',
                          style={'cursor': 'default', 'color': '#94a3b8'})]
        return (empty, '', no_update, *([no_update] * 9))

    try:
        center_ts = parse_timestamp(ts_input)
    except ValueError:
        empty = [html.Div('时间格式错误', className='id-list-item',
                          style={'cursor': 'default', 'color': '#dc2626'})]
        return (empty, '', '时间格式不正确，请使用 YYYY-MM-DD HH:MM:SS 格式', *([no_update] * 9))

    df = get_df()
    meta_df = get_meta_df()
    if df is None or meta_df is None:
        logger.error(
            f'on_timestamp_input: df={df is not None}, meta_df={meta_df is not None} — both must exist'
        )
        raise PreventUpdate

    # 始终列出全部目标ID，按时间距离排序，高亮最近目标
    list_children, nearest_id = _build_id_list_html(df, meta_df, center_ts)

    if not list_children:
        empty = [html.Div('无有效目标', className='id-list-item',
                          style={'cursor': 'default', 'color': '#94a3b8'})]
        return (empty, '', '数据集中无有效目标', *([no_update] * 9))

    if nearest_id is None:
        empty = [html.Div('无有效目标', className='id-list-item',
                          style={'cursor': 'default', 'color': '#94a3b8'})]
        return (empty, '', '数据集中无有效目标', *([no_update] * 9))

    nearest_source, nearest_raw_id = _target_parts(nearest_id)
    nearest_source_text = f'{nearest_source.upper()} · ' if nearest_source else ''
    feedback = f'共 {len(list_children)} 个雷达目标 · 最近: {nearest_source_text}ID={nearest_raw_id}'

    # ---- 自动绘制最近目标的图表 ----
    if nearest_id is not None:
        id_meta = _target_meta(meta_df, nearest_id)
        first_row = id_meta.iloc[0] if len(id_meta) > 0 else None

        if first_row is not None:
            first_traj_id = first_row['trajectory_id']
            first_seg_df = get_segment(first_traj_id)

            if first_seg_df is None:
                logger.warning(
                    f'on_timestamp_input: get_segment returned None for '
                    f'traj={first_traj_id}, nearest_id={nearest_id}'
                )

            if first_seg_df is not None:
                valid_qties = _ensure_valid_quantities(first_seg_df, selected_qties)
                unavailable_qties = _unavailable_quantities(first_seg_df, selected_qties)

                diff_cache = {}
                fig = build_multi_subplot_graph(first_seg_df, valid_qties, first_traj_id, diff_cache=diff_cache)
                stats_per_qty, first_frame_dists = _compute_quantities_stats(first_seg_df, valid_qties, diff_cache=diff_cache)
                quantities_config = get('quantities', {})
                title = _build_title_bar(
                    first_traj_id, valid_qties, len(first_seg_df), unavailable_qties)
                table, _ = _build_trajectory_table(id_meta)

                traj_label = f'{nearest_source_text}ID={nearest_raw_id}'
                return (list_children, str(len(list_children)), feedback,
                        table, fig,
                        render_multi_full_stats(valid_qties, quantities_config, stats_per_qty, first_frame_dists),
                        render_box_stats_empty(),
                        traj_label, title,
                        first_traj_id, None, nearest_id)

    # fallback: 有ID列表但无法绘制图表（缓存丢失或段不存在）
    logger.warning(
        f'on_timestamp_input: nearest_id={nearest_id} 段数据缺失, '
        f'radar={get_radar_position()}'
    )
    return (list_children, str(len(list_children)), feedback,
            None, no_update, no_update, no_update, '', no_update, None, None, nearest_id)


# ============================================================
# 回调 [3]: ID 列表项点击 → 高亮选中 + 轨迹段表格 + 自动绘制
# ============================================================
@callback(
    Output('id-list-container', 'children', allow_duplicate=True),
    Output('store-selected-id', 'data'),
    Output('trajectory-table', 'children'),
    Output('traj-id-badge', 'children'),
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('stats-full-content', 'children', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('graph-title-bar', 'children', allow_duplicate=True),
    Output('store-selected-trajectory', 'data', allow_duplicate=True),
    Output('store-box-selection', 'data', allow_duplicate=True),
    Output('box-select-feedback', 'children', allow_duplicate=True),
    Input({'type': 'id-list-item', 'index': ALL}, 'n_clicks'),
    State('store-data-loaded', 'data'),
    State('store-selected-quantities', 'data'),
    prevent_initial_call=True,
)
def on_id_click(n_clicks_list, data_loaded: bool, selected_qties):
    if not data_loaded or not n_clicks_list or get_meta_df() is None:
        raise PreventUpdate

    ctx = dash_ctx
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]
    try:
        triggered_id = json.loads(trigger['prop_id'].split('.')[0])
        selected_id = triggered_id['index']
        selected_source, selected_raw_id = _target_parts(selected_id)
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        raise PreventUpdate

    meta_df = get_meta_df()

    # 按“雷达来源 + ID”重建列表，避免前后雷达同号目标在 UI 中合并。
    df = get_df()
    new_list, _ = _build_id_list_html(df, meta_df, selected_target=selected_id)

    id_meta = _target_meta(meta_df, selected_id)
    source_prefix = f'{selected_source.upper()} · ' if selected_source else ''
    selected_label = f'{source_prefix}ID={selected_raw_id}'
    if len(id_meta) == 0:
        return (new_list, selected_id, no_update, selected_label,
                no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    # 构建轨迹段表格
    table, first_traj_id = _build_trajectory_table(id_meta)

    if first_traj_id is None:
        return (new_list, selected_id, table, selected_label,
                no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    first_seg_df = get_segment(first_traj_id)
    if first_seg_df is None:
        return (new_list, selected_id, table, selected_label,
                no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    valid_qties = _ensure_valid_quantities(first_seg_df, selected_qties)
    unavailable_qties = _unavailable_quantities(first_seg_df, selected_qties)

    diff_cache = {}
    fig = build_multi_subplot_graph(first_seg_df, valid_qties, first_traj_id, diff_cache=diff_cache)
    stats_per_qty, first_frame_dists = _compute_quantities_stats(first_seg_df, valid_qties, diff_cache=diff_cache)
    quantities_config = get('quantities', {})

    title = _build_title_bar(
        first_traj_id, valid_qties, len(first_seg_df), unavailable_qties)

    return (new_list, selected_id, table, selected_label,
            fig,
            render_multi_full_stats(valid_qties, quantities_config, stats_per_qty, first_frame_dists),
            render_box_stats_empty(),
            title, first_traj_id, None, '')


# ============================================================
# 回调 [4]: 轨迹段点击 → 绘制多子图 + 全段统计
# ============================================================
@callback(
    Output('trajectory-graph', 'figure'),
    Output('stats-full-content', 'children'),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('graph-title-bar', 'children'),
    Output('store-selected-trajectory', 'data'),
    Output('store-box-selection', 'data'),
    Input({'type': 'traj-row', 'index': ALL}, 'n_clicks'),
    State('store-data-loaded', 'data'),
    State('store-selected-quantities', 'data'),
    prevent_initial_call=True,
)
def on_trajectory_select(n_clicks_list, data_loaded: bool, selected_qties):
    if not data_loaded or not has_data_loaded():
        raise PreventUpdate

    ctx = dash_ctx
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]
    if 'traj-row' not in trigger['prop_id']:
        raise PreventUpdate

    try:
        traj_id = json.loads(trigger['prop_id'].split('.')[0])['index']
    except (json.JSONDecodeError, KeyError, IndexError):
        raise PreventUpdate

    seg_df = get_segment(traj_id)
    if seg_df is None:
        raise PreventUpdate

    valid_qties = _ensure_valid_quantities(seg_df, selected_qties)
    unavailable_qties = _unavailable_quantities(seg_df, selected_qties)

    diff_cache = {}
    fig = build_multi_subplot_graph(seg_df, valid_qties, traj_id, diff_cache=diff_cache)
    stats_per_qty, first_frame_dists = _compute_quantities_stats(seg_df, valid_qties, diff_cache=diff_cache)
    quantities_config = get('quantities', {})
    title = _build_title_bar(traj_id, valid_qties, len(seg_df), unavailable_qties)

    return (fig,
            render_multi_full_stats(valid_qties, quantities_config, stats_per_qty, first_frame_dists),
            render_box_stats_empty(),
            title, traj_id, None)


# ============================================================
# 回调 [5]: 物理量多选变更 → 重建子图 + 重算统计
# ============================================================
@callback(
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('stats-full-content', 'children', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('graph-title-bar', 'children', allow_duplicate=True),
    Output('store-selected-quantities', 'data'),
    Input('quantity-checklist', 'value'),
    State('store-selected-trajectory', 'data'),
    State('store-box-selection', 'data'),
    prevent_initial_call=True,
)
def on_quantity_change(selected_values, traj_id, box_selection):
    """物理量多选变更 → 重建多子图 + 全段统计。"""
    if not traj_id or not has_data_loaded():
        raise PreventUpdate

    seg_df = get_segment(traj_id)
    if seg_df is None:
        raise PreventUpdate

    valid_qties = _ensure_valid_quantities(seg_df, selected_values)
    unavailable_qties = _unavailable_quantities(seg_df, selected_values)

    # 框选高亮（如有）
    highlight_range = None
    highlight_time_range = None
    if box_selection:
        if box_selection.get('x_start') and box_selection.get('x_end'):
            highlight_time_range = (box_selection['x_start'], box_selection['x_end'])
        else:
            # 兼容旧页面状态：缺少时间边界时才回退到段内索引。
            highlight_range = (box_selection.get('start_idx'), box_selection.get('end_idx'))

    diff_cache = {}
    fig = build_multi_subplot_graph(
        seg_df, valid_qties, traj_id, highlight_range,
        highlight_time_range=highlight_time_range, diff_cache=diff_cache,
    )

    stats_per_qty, first_frame_dists = _compute_quantities_stats(seg_df, valid_qties, diff_cache=diff_cache)
    quantities_config = get('quantities', {})

    # 框选统计
    if box_selection:
        if box_selection.get('x_start') and box_selection.get('x_end'):
            # 与图上高亮使用同一原始鼠标时间边界，避免物理量切换后统计口径漂移。
            x_start = pd.Timestamp(box_selection['x_start'])
            x_end = pd.Timestamp(box_selection['x_end'])
            mask = (seg_df['timestamp_parsed'] >= x_start) & (seg_df['timestamp_parsed'] <= x_end)
        else:
            mask = pd.Series(False, index=seg_df.index)
            mask.iloc[box_selection['start_idx']:box_selection['end_idx'] + 1] = True
        box_stats_per_qty, _ = _compute_quantities_stats(seg_df, valid_qties, mask=mask, diff_cache=diff_cache)
        box_panel = render_multi_box_stats(valid_qties, quantities_config, box_stats_per_qty)
    else:
        box_panel = render_box_stats_empty()

    return (fig,
            render_multi_full_stats(valid_qties, quantities_config, stats_per_qty, first_frame_dists),
            box_panel,
            _build_title_bar(
                traj_id, valid_qties, len(seg_df), unavailable_qties),
            valid_qties)


# ============================================================
# 回调 [6]: 框选 → 重建 figure（含全子图 shapes）+ 更新统计
# ============================================================
@callback(
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('store-box-selection', 'data', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('box-select-feedback', 'children', allow_duplicate=True),
    Input('trajectory-graph', 'selectedData'),
    State('store-selected-trajectory', 'data'),
    State('store-selected-quantities', 'data'),
    State('trajectory-graph', 'figure'),
    prevent_initial_call=True,
)
def on_box_select(selected_data, traj_id, selected_qties, current_figure):
    """框选 → Patch 更新高亮矩形，并基于原始全量数据计算统计。"""
    if selected_data is None:
        raise PreventUpdate
    x_range = extract_x_selection(selected_data)
    if x_range is None:
        raise PreventUpdate

    if not traj_id or not has_data_loaded():
        raise PreventUpdate

    seg_df = get_segment(traj_id)
    if seg_df is None:
        raise PreventUpdate

    valid_qties = _ensure_valid_quantities(seg_df, selected_qties)

    x_min_ts, x_max_ts = x_range

    mask = (seg_df['timestamp_parsed'] >= x_min_ts) & (seg_df['timestamp_parsed'] <= x_max_ts)
    masked_df = seg_df[mask]

    if len(masked_df) < 2:
        raise PreventUpdate

    # 使用布尔掩码的段内位置，而不是 DataFrame index label。后续 ID 复用段
    # 以前保留了父轨迹索引，导致高亮只对同 ID 的首段有效。
    selected_positions = np.flatnonzero(mask.to_numpy())
    start_idx = int(selected_positions[0])
    end_idx = int(selected_positions[-1])

    # 只更新 layout.shapes，避免为一次框选重新序列化全部曲线数据。
    figure_patch = Patch()
    highlight_shapes = build_highlight_shapes(
        seg_df, valid_qties,
        highlight_range=(start_idx, end_idx),
        highlight_time_range=(x_min_ts, x_max_ts),
        vertical_spacing=0.01,
    )

    diff_cache = {}
    # 计算所有物理量框选统计
    box_stats_per_qty, _ = _compute_quantities_stats(seg_df, valid_qties, mask=mask, diff_cache=diff_cache)
    # 框选区间内各物理量最大跳变高亮（紫色加粗，清除框选时移除；
    # 全段最大跳变红色标记保留，不受影响）
    box_jump_shapes = build_box_jump_shapes(masked_df, valid_qties, diff_cache=diff_cache)
    quantities_config = get('quantities', {})
    figure_patch['layout']['shapes'] = (
        highlight_shapes + box_jump_shapes
        + _get_non_highlight_shapes(current_figure)
    )

    feedback = f'已选 {len(masked_df)} 帧 [{start_idx}–{end_idx}]'

    box_selection_data = {
        'start_idx': start_idx,
        'end_idx': end_idx,
        'x_start': x_min_ts.isoformat(),
        'x_end': x_max_ts.isoformat(),
    }
    return (figure_patch,
            box_selection_data,
            render_multi_box_stats(valid_qties, quantities_config, box_stats_per_qty),
            feedback)


# ============================================================
# 回调 [6clear]: 清除按钮 → 重建 figure（无 shapes）+ 清除统计
# ============================================================
@callback(
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('store-box-selection', 'data', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('box-select-feedback', 'children', allow_duplicate=True),
    Input('clear-box-btn', 'n_clicks'),
    State('store-selected-trajectory', 'data'),
    State('store-selected-quantities', 'data'),
    State('store-box-selection', 'data'),
    State('trajectory-graph', 'figure'),
    prevent_initial_call=True,
)
def on_clear_box_select(_n, traj_id, selected_qties, current_box, current_figure):
    """清除框选：仅移除高亮矩形并清空局部统计。"""
    if not _n or _n <= 0:
        raise PreventUpdate
    if not traj_id or not has_data_loaded():
        raise PreventUpdate

    seg_df = get_segment(traj_id)
    if seg_df is None:
        raise PreventUpdate

    valid_qties = _ensure_valid_quantities(seg_df, selected_qties)

    # 仅删除矩形高亮，保留图表自身的线形标注。
    figure_patch = Patch()
    figure_patch['layout']['shapes'] = _get_non_highlight_shapes(current_figure)
    # 清除 plotly 原生持久选框（layout.selections，虚线矩形）。
    # 用户拖拽框选时前端会创建原生选框，仅删自绘 shapes 无法移除它。
    figure_patch['layout']['selections'] = []

    return (figure_patch,
            None,
            render_box_stats_empty(),
            '')


# ============================================================
# 回调 [Qdyn]: 物理量选项按已加载数据动态生成
# ============================================================
@callback(
    Output('quantity-checklist', 'options'),
    Output('quantity-checklist', 'value'),
    Output('store-selected-quantities', 'data', allow_duplicate=True),
    Input('store-data-loaded', 'data'),
    State('quantity-checklist', 'value'),
    prevent_initial_call=True,
)
def refresh_quantity_options(data_loaded, previous_values):
    """按已加载 CSV 的实际数值列动态生成物理量选项。

    配置只用于已知字段的单位与展示名；选项、顺序和可用性均以当前活动
    雷达 CSV 为准。切换前/后角时保留仍存在的用户选择，否则选 Dx 或首列。
    """
    if not data_loaded:
        return [], [], []
    df = get_df()
    if df is None or len(df) == 0:
        return [], [], []

    quantities_config = get('quantities', {}) or {}
    fields = _discover_quantity_columns(df)
    options = [
        {
            'label': str(quantities_config.get(field, {}).get('label', field)),
            'value': field,
        }
        for field in fields
    ]

    default_qty = get('DEFAULT_QUANTITY', 'Dx')
    selected = [str(value) for value in (previous_values or []) if str(value) in fields]
    if selected:
        value = selected
    elif default_qty in fields:
        value = [default_qty]
    elif fields:
        value = [fields[0]]
    else:
        value = []
    return options, value, value


# ============================================================
# 回调 [Wsum]: 波动摘要快照（插入 / 删除 / 清空 + 渲染）
# ============================================================
@callback(
    Output('wave-summary-snapshots', 'data'),
    Output('wave-summary-feedback', 'children'),
    Input('wave-summary-insert-btn', 'n_clicks'),
    Input('wave-summary-remove-btn', 'n_clicks'),
    Input('wave-summary-clear-btn', 'n_clicks'),
    State('store-selected-trajectory', 'data'),
    State('store-selected-quantities', 'data'),
    State('store-box-selection', 'data'),
    State('wave-summary-snapshots', 'data'),
    prevent_initial_call=True,
)
def update_wave_summary_snapshots(_ins, _rm, _clr, traj_id, selected_qties,
                                  box_selection, snapshot_store):
    """仅由用户操作修改已插入波动摘要，绘制/框选不会覆盖快照。

    插入数据源：有框选区间时用框选内数据（标签带区间），否则用全段。
    """
    triggered = dash_ctx.triggered[0]['prop_id'].split('.')[0] if dash_ctx.triggered else ''
    items = list((snapshot_store or {}).get('items') or [])

    if triggered == 'wave-summary-clear-btn':
        return {'items': []}, '已清空波动摘要'
    if triggered == 'wave-summary-remove-btn':
        if not items:
            return no_update, '没有可删除的摘要'
        items.pop()
        return {'items': items}, '已删除最后一次插入的波动摘要'
    if triggered != 'wave-summary-insert-btn':
        raise PreventUpdate

    if not traj_id or not has_data_loaded():
        return no_update, '请先导入数据并选择轨迹段，再插入波动摘要'
    seg_df = get_segment(traj_id)
    if seg_df is None:
        return no_update, '轨迹数据不存在'

    box = box_selection or {}
    if box.get('start_idx') is not None and box.get('end_idx') is not None:
        start_idx, end_idx = int(box['start_idx']), int(box['end_idx'])
        sub_df = seg_df.iloc[start_idx:end_idx + 1]
        label = f'{traj_id} [{start_idx}–{end_idx}]'
        source_text = f'框选区间 [{start_idx}–{end_idx}]'
    else:
        sub_df = seg_df
        label = f'{traj_id}（全段）'
        source_text = '全段'
    if len(sub_df) < 2:
        return no_update, '选中区域数据不足 2 帧，无法计算波动'

    quantities_config = get('quantities', {})
    valid_qties = _ensure_valid_quantities(sub_df, selected_qties)
    diff_cache = {}
    values = {}
    for qty in valid_qties:
        jump = find_max_jump(sub_df, qty, diff_cache=diff_cache)
        values[qty] = ({
            'value': float(jump['diff_value']),
            'unit': str(quantities_config.get(qty, {}).get('unit', '')),
        } if jump else None)

    items.append({'label': label, 'values': values})
    return {'items': items}, f'已插入第 {len(items)} 次波动摘要（{source_text}）'


@callback(
    Output('wave-summary-content', 'children', allow_duplicate=True),
    Output('wave-summary-copy', 'content'),
    Input('wave-summary-snapshots', 'data'),
    prevent_initial_call=True,
)
def render_wave_summary_content(snapshot_store):
    """快照更新时才重绘波动摘要，不影响其他卡片。

    剪贴板纯文本同样只含数据行，不含快照来源元信息。
    """
    items = (snapshot_store or {}).get('items') or []
    quantities_config = get('quantities', {})
    summary_html = render_wave_snapshot_summary(items, quantities_config)
    return summary_html, _summary_plain_text(summary_html)


# ============================================================
# 回调 [Wclear]: 波动分析一键清除
# ============================================================
@callback(
    Output('store-data-loaded', 'data', allow_duplicate=True),
    Output('store-segments-meta', 'data', allow_duplicate=True),
    Output('store-selected-trajectory', 'data', allow_duplicate=True),
    Output('store-selected-quantities', 'data', allow_duplicate=True),
    Output('store-box-selection', 'data', allow_duplicate=True),
    Output('store-selected-id', 'data', allow_duplicate=True),
    Output('upload-csv', 'contents', allow_duplicate=True),
    Output('upload-csv', 'filename', allow_duplicate=True),
    Output('timestamp-input', 'value', allow_duplicate=True),
    Output('upload-feedback', 'children', allow_duplicate=True),
    Output('timestamp-feedback', 'children', allow_duplicate=True),
    Output('id-list-container', 'children', allow_duplicate=True),
    Output('id-count-badge', 'children', allow_duplicate=True),
    Output('trajectory-table', 'children', allow_duplicate=True),
    Output('traj-id-badge', 'children', allow_duplicate=True),
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('graph-title-bar', 'children', allow_duplicate=True),
    Output('stats-full-content', 'children', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
    Output('box-select-feedback', 'children', allow_duplicate=True),
    Output('quantity-checklist', 'value', allow_duplicate=True),
    Input('wave-clear-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def on_wave_clear(n):
    """一键清除波动分析页面所有数据（含Upload组件重置）。

    关键修复：清空 upload-csv.contents/filename 以清除 Upload 的浏览器缓存，
    确保清除后重新选择同名文件时，contents 值从 None→base64 是一个真实变更，
    Dash 能正确检测并触发 on_upload_csv 回调进行完整渲染。
    """
    if not n or n <= 0:
        raise PreventUpdate
    clear_data_cache()
    return (
        False, None, None, [], None, None,  # stores
        None, None,                          # upload-csv contents / filename
        '', '', '',                          # inputs/feedback
        html.Div('请上传数据并输入时间戳', className='id-list-item',
                 style={'cursor': 'default', 'color': '#94a3b8'}),  # id-list-container
        '',                                  # id-count-badge
        '',                                  # trajectory-table
        '',                                  # traj-id-badge
        go.Figure(),                         # trajectory-graph
        html.Span('请重新上传数据并选择轨迹', className='feedback-muted'),  # graph-title-bar
        render_multi_full_stats_placeholder(),          # stats-full-content
        render_box_stats_empty(),                       # stats-box-content
        '',                                  # box-select-feedback
        [],                                  # quantity-checklist value 清空
    )
