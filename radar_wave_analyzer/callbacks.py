"""
回调函数集合。
唯一连接 core/ 和 components/ 的桥梁。

回调链：
  [0]  on_radar_change       → 雷达切换 → 更新位置标签
  [1]  on_upload_csv         → 拖拽上传 → 解析CSV → 分段 → 缓存
  [2]  on_timestamp_input    → 筛选时间窗口 → 提取ID列表
  [3]  on_id_click           → 自定义 ListGroup 点击选中
  [4]  on_trajectory_click   → 轨迹段选中 → 绘制多子图 + 全段统计
  [5]  on_quantity_change    → 物理量多选变更 → 重建子图 + 重算统计
  [6]  on_box_select         → 框选 → 全子图高亮 + 统计
  [6clear] on_clear_box_select → 清除框选
  [7]  on_export_csv         → 导出CSV
  [8]  on_export_img         → 导出图片

  [C1] on_mode_switch       → 模式切换 → 显示/隐藏面板 + 重置对比状态
  [C2] on_cmp_upload         → 统一上传 → 解析+缓存+预览+ID发现（单回调无轮询）
  [C3] on_cmp_id_select     → ID选择 → 坐标诊断 + 延迟检测
  [C4] on_cmp_run           → 执行对齐 → 图表 + 统计
  [C5] on_cmp_quantity_change → 指标切换 → 更新图表
  [C6] on_cmp_export_csv    → 导出CSV
  [C7] on_cmp_export_img    → 导出图表
"""
import base64
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update, html, dcc, ALL
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

try:
    from .extensions import app
except ImportError:
    from extensions import app

try:
    from .config import get
except ImportError:
    from config import get

try:
    from .cache import (
        set_data_cache,
        get_df, get_meta_df, get_radar_position, get_file_path,
        get_segment, has_data_loaded,
        clear_data_cache,
        switch_radar,
        # 对比缓存
        set_comparison_data, get_comparison_data,
        set_alignment_result, get_alignment_result, get_rtk_curve_result,
        clear_comparison_data,
    )
except ImportError:
    from cache import (  # type: ignore[no-redef]
        set_data_cache,
        get_df, get_meta_df, get_radar_position, get_file_path,
        get_segment, has_data_loaded,
        clear_data_cache,
        switch_radar,
        set_comparison_data, get_comparison_data,
        set_alignment_result, get_alignment_result, get_rtk_curve_result,
        clear_comparison_data,
    )

try:
    from .core.data_loader import (
        load_csv_from_bytes, parse_timestamp,
        filter_by_time_window, get_time_range,
    )
except ImportError:
    from core.data_loader import (  # type: ignore[no-redef]
        load_csv_from_bytes, parse_timestamp,
        filter_by_time_window, get_time_range,
    )

try:
    from .core.segmenter import segment_trajectories
except ImportError:
    from core.segmenter import segment_trajectories  # type: ignore[no-redef]

try:
    from .core.selection import extract_x_selection
except ImportError:
    from core.selection import extract_x_selection  # type: ignore[no-redef]

try:
    from .core.wave_calc import compute_segment_stats, compute_fluctuation_stats
except ImportError:
    from core.wave_calc import compute_segment_stats, compute_fluctuation_stats  # type: ignore[no-redef]

try:
    from .core.exporter import export_trajectory_csv, export_graph_image
except ImportError:
    from core.exporter import export_trajectory_csv, export_graph_image  # type: ignore[no-redef]

try:
    from .components.stats_panel import (
        render_multi_full_stats, render_multi_full_stats_placeholder,
        render_multi_box_stats, render_box_stats_empty,
        render_cmp_error_stats, render_cmp_distance_bins,
        render_cmp_error_stats_empty, render_cmp_bins_empty,
    )
except ImportError:
    from components.stats_panel import (  # type: ignore[no-redef]
        render_multi_full_stats, render_multi_full_stats_placeholder,
        render_multi_box_stats, render_box_stats_empty,
        render_cmp_error_stats, render_cmp_distance_bins,
        render_cmp_error_stats_empty, render_cmp_bins_empty,
    )

try:
    from .components.graph_builder import build_multi_subplot_graph
except ImportError:
    from components.graph_builder import build_multi_subplot_graph  # type: ignore[no-redef]

try:
    from .components.comparison_charts import (
        build_comparison_subplots, build_delay_curve_chart,
    )
except ImportError:
    from components.comparison_charts import (  # type: ignore[no-redef]
        build_comparison_subplots, build_delay_curve_chart,
    )

try:
    from .comparison.parser import validate_overlap
except ImportError:
    from comparison.parser import validate_overlap  # type: ignore[no-redef]

try:
    from .comparison.alignment import compute_distance_bin_stats
except ImportError:
    from comparison.alignment import compute_distance_bin_stats  # type: ignore[no-redef]

try:
    from .comparison.service import (
        analyse_selected_track, execute_alignment, get_candidate_ids, get_candidate_match_result,
        prepare_comparison_upload, resolve_track_selection,
    )
except ImportError:
    from comparison.service import (  # type: ignore[no-redef]
        analyse_selected_track, execute_alignment, get_candidate_ids, get_candidate_match_result,
        prepare_comparison_upload, resolve_track_selection,
    )

try:
    from .comparison.exporter import export_aligned_csv
except ImportError:
    from comparison.exporter import export_aligned_csv  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# ===================== 辅助函数 =====================

def _selected_quantities() -> list:
    """从配置获取默认勾选的物理量列表。"""
    default_qty = get('DEFAULT_QUANTITY', 'Dx')
    quantities_config = get('quantities', {})
    if default_qty in quantities_config:
        return [default_qty]
    # fallback: 第一个物理量
    for qty in quantities_config:
        return [qty]
    return []


def _ensure_valid_quantities(seg_df, selected_qties) -> list:
    """确保物理量列表均存在于数据列中，fallback 到配置默认值。"""
    selected_qties = selected_qties or _selected_quantities()
    valid_qties = [q for q in selected_qties if q in seg_df.columns]
    if not valid_qties:
        valid_qties = _selected_quantities()
    return valid_qties


def _build_trajectory_table(id_meta):
    """构建轨迹段表格 HTML 并返回 (table, first_traj_id)。

    两个位置（on_timestamp_input / on_id_click）共用此逻辑。
    """
    table_header = html.Thead(html.Tr([
        html.Th('时间区间'), html.Th('帧数'), html.Th('状态'),
    ]))
    rows = []
    first_traj_id = None
    for _, seg in id_meta.iterrows():
        traj_id = seg['trajectory_id']
        if first_traj_id is None:
            first_traj_id = traj_id
        label = seg.get('display_label', traj_id)
        if seg.get('spatial_anomaly'):
            status_cell = html.Td(html.Span('⚠空间跳变', className='spatial-anomaly-badge'))
        else:
            status_cell = html.Td('')
        rows.append(html.Tr(
            id={'type': 'traj-row', 'index': traj_id},
            n_clicks=0,
            **{'data-traj-id': traj_id},
            children=[
                html.Td(label, style={'color': '#2563eb', 'fontWeight': '500', 'whiteSpace': 'nowrap'}),
                html.Td(str(seg['total_frames'])),
                status_cell,
            ],
        ))
    table = html.Table([table_header, html.Tbody(rows)], className='traj-table')
    return table, first_traj_id


def _compute_quantities_stats(seg_df, selected_qties, mask=None, diff_cache=None):
    """为所有勾选物理量计算 segment_stats。

    Returns:
        (stats_per_qty, dx_max_dist)
    """
    stats_per_qty = {}
    for qty in selected_qties:
        if qty in seg_df.columns:
            stats_per_qty[qty] = compute_segment_stats(seg_df, qty, mask=mask, diff_cache=diff_cache)
        else:
            stats_per_qty[qty] = None

    fluct = compute_fluctuation_stats(seg_df, mask=mask, diff_cache=diff_cache)
    dx_max_dist = fluct.get('dx_max_dist') if fluct else None

    return stats_per_qty, dx_max_dist


def _build_title_bar(traj_id: str, selected_qties: list, n_frames: int) -> html.Span:
    """构建图表标题栏：显示目标ID + 轨迹段时间区间 + 帧数。"""
    meta_df = get_meta_df()
    time_range_label = traj_id
    original_id = ''
    spatial_anom = False
    if meta_df is not None and traj_id is not None:
        row = meta_df[meta_df['trajectory_id'] == traj_id]
        if len(row) > 0:
            original_id = str(row.iloc[0].get('original_id', ''))
            if row.iloc[0].get('display_label'):
                time_range_label = f'{row.iloc[0]["display_label"]}'
            else:
                time_range_label = str(traj_id)
            spatial_anom = bool(row.iloc[0].get('spatial_anomaly', False))

    parts = [html.Span(f'ID: {original_id}', className='traj-id-badge')]
    parts.append(html.Span(' | ', style={'color': '#94a3b8'}))
    parts.append(html.Span(time_range_label, className='traj-name'))
    parts.append(html.Span(' | ', style={'color': '#94a3b8'}))
    parts.append(html.Span(f'{n_frames} frames', className='frame-count'))
    if spatial_anom:
        parts.append(html.Span(' | ', style={'color': '#94a3b8'}))
        parts.append(html.Span('⚠空间跳变', className='spatial-anomaly-badge'))
    return html.Span(parts)


def _build_id_list_html(df, meta_df, center_ts=None):
    """构建目标ID列表HTML。

    Args:
        df: 合并后的数据DataFrame
        meta_df: 分段元信息DataFrame
        center_ts: 可选，参考时间戳；提供时按时间距离排序并高亮最近ID

    Returns:
        (list_children, nearest_id_or_None)
    """
    all_ids = sorted(df['ID'].unique())

    if center_ts is None:
        # 无时间戳 → 仅显示ID和段数，按ID排序
        list_children = []
        for id_val in all_ids:
            id_meta = meta_df[meta_df['original_id'] == id_val]
            list_children.append(html.Div([
                html.Span(f'ID: {int(id_val)}', className='id-text'),
                html.Span(f'{len(id_meta)}段', className='id-meta'),
            ], id={'type': 'id-list-item', 'index': int(id_val)},
               n_clicks=0, className='id-list-item',
               **{'data-id': str(int(id_val))}))
        return list_children, None

    # 有时间戳 → 计算每个ID距离输入时间戳最近点的时间差
    id_items = []
    for id_val in all_ids:
        id_meta = meta_df[meta_df['original_id'] == id_val]
        id_df = df[df['ID'] == id_val]
        if len(id_df) == 0:
            continue
        time_diffs = (id_df['timestamp_parsed'] - center_ts).dt.total_seconds()
        min_diff_idx = time_diffs.abs().idxmin()
        diff_sec = time_diffs[min_diff_idx]
        sign = '+' if diff_sec >= 0 else ''
        id_items.append({
            'id': int(id_val),
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
        is_selected = (item['id'] == nearest_id)
        cls = 'id-list-item selected' if is_selected else 'id-list-item'
        list_children.append(html.Div([
            html.Span(item['display'], className='id-text'),
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
def _decode_upload_contents(contents) -> bytes:
    if not isinstance(contents, str):
        raise ValueError('文件内容为空')
    try:
        _, content_string = contents.split(',', 1)
        return base64.b64decode(content_string)
    except Exception as e:
        raise ValueError(f'文件解码失败: {e}') from e


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
    for i, (content, fn) in enumerate(zip(contents_list, filenames)):
        try:
            data_bytes = _decode_upload_contents(content)
            df = load_csv_from_bytes(data_bytes, fn)
            # 标记来源文件序号：跨文件同 ID 需按时间独立分段，避免曲线被错误合并
            df['file_index'] = i
            all_dfs.append(df)
        except ValueError as e:
            errors.append(f'{fn}: {e}')
        except Exception as e:
            errors.append(f'{fn}: {e}')

    if not all_dfs:
        err_msg = html.Span(f'所有文件解析失败: {"；".join(errors)}', style={'color': '#dc2626'})
        return _clear_state(err_msg)

    if len(all_dfs) == 1:
        merged_df = all_dfs[0]
    else:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        # 去重子集加入 file_index：跨文件同 (ID, timestamp) 的不同目标不再被误删，
        # 由规则 E 按来源文件独立分段；同文件内真正重复帧仍正常去重。
        merged_df = merged_df.drop_duplicates(subset=['file_index', 'ID', 'timestamp_parsed'], keep='first')
        merged_df = merged_df.sort_values('timestamp_parsed').reset_index(drop=True)

    meta_df, segments = segment_trajectories(merged_df)

    upload_label = filenames[0] if len(filenames) == 1 else f'{len(filenames)}个文件'
    set_data_cache(upload_label, radar_key, merged_df, meta_df, segments)

    t_min, t_max = get_time_range(merged_df)
    err_suffix = f'（部分失败: {"；".join(errors)}）' if errors else ''
    feedback = html.Span(
        f'已加载: {upload_label} | {len(merged_df)}行, {len(meta_df)}段 | '
        f'{t_min.strftime("%Y-%m-%d %H:%M:%S")} ~ {t_max.strftime("%Y-%m-%d %H:%M:%S")} {err_suffix}',
        style={'color': '#15803d', 'fontWeight': '500'},
    )

    # 上传成功后直接显示全部目标ID
    list_children, _ = _build_id_list_html(merged_df, meta_df)
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

    feedback = f'共 {len(list_children)} 个目标 · 最近: ID={nearest_id}'

    # ---- 自动绘制最近目标的图表 ----
    if nearest_id is not None:
        id_meta = meta_df[meta_df['original_id'] == nearest_id]
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

                diff_cache = {}
                fig = build_multi_subplot_graph(first_seg_df, valid_qties, first_traj_id, diff_cache=diff_cache)
                stats_per_qty, dx_max_dist = _compute_quantities_stats(first_seg_df, valid_qties, diff_cache=diff_cache)
                quantities_config = get('quantities', {})
                title = _build_title_bar(first_traj_id, valid_qties, len(first_seg_df))
                table, _ = _build_trajectory_table(id_meta)

                traj_label = f'ID={nearest_id}'
                return (list_children, str(len(list_children)), feedback,
                        table, fig,
                        render_multi_full_stats(valid_qties, quantities_config, stats_per_qty, dx_max_dist),
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
        selected_id = int(triggered_id['index'])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        raise PreventUpdate

    meta_df = get_meta_df()

    # 重建 ListGroup
    new_list = []
    if ctx.inputs_list and ctx.inputs_list[0]:
        for input_dict, n_click in zip(ctx.inputs_list[0], n_clicks_list):
            idx = input_dict['id']['index']
            id_val = int(idx)
            is_selected = (id_val == selected_id)
            cls = 'id-list-item selected' if is_selected else 'id-list-item'
            id_meta = meta_df[meta_df['original_id'] == id_val]
            new_list.append(html.Div(
                [html.Span(f'ID: {id_val}', className='id-text'),
                 html.Span(f'{len(id_meta)}段', className='id-meta')],
                id={'type': 'id-list-item', 'index': id_val},
                n_clicks=n_click,
                className=cls,
            ))

    id_meta = meta_df[meta_df['original_id'] == selected_id]
    if len(id_meta) == 0:
        return (new_list, selected_id, no_update, f'ID={selected_id}',
                no_update, no_update, no_update, no_update, no_update, no_update)

    # 构建轨迹段表格
    table, first_traj_id = _build_trajectory_table(id_meta)

    if first_traj_id is None:
        return (new_list, selected_id, table, f'ID={selected_id}',
                no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    first_seg_df = get_segment(first_traj_id)
    if first_seg_df is None:
        return (new_list, selected_id, table, f'ID={selected_id}',
                no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    valid_qties = _ensure_valid_quantities(first_seg_df, selected_qties)

    diff_cache = {}
    fig = build_multi_subplot_graph(first_seg_df, valid_qties, first_traj_id, diff_cache=diff_cache)
    stats_per_qty, dx_max_dist = _compute_quantities_stats(first_seg_df, valid_qties, diff_cache=diff_cache)
    quantities_config = get('quantities', {})

    title = _build_title_bar(first_traj_id, valid_qties, len(first_seg_df))

    return (new_list, selected_id, table, f'ID={selected_id}',
            fig,
            render_multi_full_stats(valid_qties, quantities_config, stats_per_qty, dx_max_dist),
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

    diff_cache = {}
    fig = build_multi_subplot_graph(seg_df, valid_qties, traj_id, diff_cache=diff_cache)
    stats_per_qty, dx_max_dist = _compute_quantities_stats(seg_df, valid_qties, diff_cache=diff_cache)
    quantities_config = get('quantities', {})
    title = _build_title_bar(traj_id, valid_qties, len(seg_df))

    return (fig,
            render_multi_full_stats(valid_qties, quantities_config, stats_per_qty, dx_max_dist),
            render_box_stats_empty(),
            title, traj_id, None)


# ============================================================
# 回调 [5]: 物理量多选变更 → 重建子图 + 重算统计
# ============================================================
@callback(
    Output('trajectory-graph', 'figure', allow_duplicate=True),
    Output('stats-full-content', 'children', allow_duplicate=True),
    Output('stats-box-content', 'children', allow_duplicate=True),
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

    stats_per_qty, dx_max_dist = _compute_quantities_stats(seg_df, valid_qties, diff_cache=diff_cache)
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
            render_multi_full_stats(valid_qties, quantities_config, stats_per_qty, dx_max_dist),
            box_panel,
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
    prevent_initial_call=True,
)
def on_box_select(selected_data, traj_id, selected_qties):
    """框选 → 服务端重建 figure（含全子图 shapes）+ 统计 + 按钮。
    由于使用 build_multi_subplot_graph(highlight_range=...) 直接将 shapes
    硬编码进 figure 的 layout.shapes，不存在客户端外部修改被回滚的问题。
    """
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

    # 重建 figure，shapes 直接写入 layout.shapes（服务端权威状态）
    diff_cache = {}
    fig = build_multi_subplot_graph(
        seg_df, valid_qties, traj_id,
        highlight_range=(start_idx, end_idx),
        highlight_time_range=(x_min_ts, x_max_ts),
        diff_cache=diff_cache,
    )

    # 计算所有物理量框选统计
    box_stats_per_qty, _ = _compute_quantities_stats(seg_df, valid_qties, mask=mask, diff_cache=diff_cache)
    quantities_config = get('quantities', {})

    feedback = f'已选 {len(masked_df)} 帧 [{start_idx}–{end_idx}]'

    box_selection_data = {
        'start_idx': start_idx,
        'end_idx': end_idx,
        'x_start': x_min_ts.isoformat(),
        'x_end': x_max_ts.isoformat(),
    }
    return (fig,
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
    prevent_initial_call=True,
)
def on_clear_box_select(_n, traj_id, selected_qties, current_box):
    """清除框选：重建 figure（无 shapes）+ 清空统计。"""
    if not _n or _n <= 0:
        raise PreventUpdate
    if not traj_id or not has_data_loaded():
        raise PreventUpdate

    seg_df = get_segment(traj_id)
    if seg_df is None:
        raise PreventUpdate

    valid_qties = _ensure_valid_quantities(seg_df, selected_qties)

    # 重建 figure（无 highlight → 无 shapes）
    fig = build_multi_subplot_graph(seg_df, valid_qties, traj_id)

    return (fig,
            None,
            render_box_stats_empty(),
            '')


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
    default_qty = get('DEFAULT_QUANTITY', 'Dx')
    return (
        False, None, None, [default_qty], None, None,  # stores
        None, None,                                      # upload-csv contents / filename
        '', '', '',                                      # inputs/feedback
        html.Div('请上传数据并输入时间戳', className='id-list-item',
                 style={'cursor': 'default', 'color': '#94a3b8'}),  # id-list-container
        '',                                             # id-count-badge
        '',                                             # trajectory-table
        '',                                             # traj-id-badge
        go.Figure(),                                    # trajectory-graph
        html.Span('请重新上传数据并选择轨迹', className='feedback-muted'),  # graph-title-bar
        render_multi_full_stats_placeholder(),          # stats-full-content
        render_box_stats_empty(),                       # stats-box-content
        '',                                             # box-select-feedback
    )


# ============================================================
# 回调 [Cclear]: 真值对比一键清除
# ============================================================
@callback(
    Output('cmp-state', 'data', allow_duplicate=True),
    Output('cmp-upload-radar-container', 'children', allow_duplicate=True),
    Output('cmp-upload-rtk-container', 'children', allow_duplicate=True),
    Output('cmp-upload-radar-feedback', 'children', allow_duplicate=True),
    Output('cmp-upload-rtk-feedback', 'children', allow_duplicate=True),
    Output('cmp-preview-card', 'children', allow_duplicate=True),
    Output('cmp-config-card', 'children', allow_duplicate=True),
    Output('cmp-graph', 'figure', allow_duplicate=True),
    Output('cmp-graph-title', 'children', allow_duplicate=True),
    Output('cmp-stats-content', 'children', allow_duplicate=True),
    Output('cmp-bins-content', 'children', allow_duplicate=True),
    Input('cmp-clear-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def on_cmp_clear(n):
    """一键清除真值对比页面所有数据（含Upload组件状态重置）。

    注：不单独输出 cmp-id-list / cmp-delay-input 等嵌套组件，
    因单文件等待态下这些组件已被 cmp-config-card 替换销毁，
    尝试更新不存在组件会导致 Dash 内部错误，回调整体中止。
    """
    if not n or n <= 0:
        raise PreventUpdate
    clear_comparison_data('default')
    fresh_state = {
        'radar_meta': None,
        'rtk_meta': None,
        'selected_id': None,
        'delay_ms': 0,
        'alignment_done': False,
    }
    # 重建Upload容器以彻底清除已上传文件的视觉残留
    return (
        fresh_state,
        [_make_cmp_radar_upload()],                         # 重置雷达Upload
        [_make_cmp_rtk_upload()],                           # 重置RTK Upload
        '', '',                                              # upload feedbacks
        _cmp_preview_empty(),                                # preview card
        _cmp_config_blank(),                                 # config card（同时销毁嵌套组件）
        go.Figure(),                                         # graph
        html.Span('请上传雷达与RTK数据并执行对齐', className='feedback-muted'),  # graph-title
        _cmp_stats_placeholder(),                            # stats
        _cmp_bins_placeholder(),                             # bins
    )


def _get_export_dir() -> str:
    """获取导出目录：打包版在同级目录，源码版在项目根目录。"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'exports')


# ============================================================
# 回调 [7]: 导出 CSV
# ============================================================
@callback(
    Output('export-feedback', 'children'),
    Input('export-csv-btn', 'n_clicks'),
    State('store-selected-trajectory', 'data'),
    State('store-selected-quantities', 'data'),
    State('store-box-selection', 'data'),
    prevent_initial_call=True,
)
def on_export_csv(n_clicks, traj_id: str, selected_qties, box_selection):
    if not traj_id or not has_data_loaded():
        raise PreventUpdate

    seg_df = get_segment(traj_id)
    if seg_df is None:
        return '导出失败：轨迹数据不存在'

    # 取元信息：当前 ID 号、帧数、起止时间
    meta_df = get_meta_df()
    if meta_df is None or len(meta_df) == 0:
        return '导出失败：元信息不存在'
    meta_row = meta_df[meta_df['trajectory_id'] == traj_id]
    if len(meta_row) == 0:
        return '导出失败：元信息不存在'
    meta_row = meta_row.iloc[0]
    original_id = meta_row['original_id']
    total_frames = meta_row['total_frames']
    start_time = str(meta_row['start_time'])
    end_time = str(meta_row['end_time'])

    selected_qties = selected_qties or _selected_quantities()

    if box_selection:
        seg_df = seg_df.iloc[
            box_selection['start_idx']:box_selection['end_idx'] + 1
        ].copy()

    output_dir = _get_export_dir()
    try:
        filepath = export_trajectory_csv(
            seg_df, traj_id, original_id, total_frames,
            start_time, end_time, output_dir,
        )
        return f'导出成功: {os.path.basename(filepath)}'
    except Exception as e:
        return f'导出失败: {str(e)}'


# ============================================================
# 回调 [8]: 导出图片
# ============================================================
@callback(
    Output('export-feedback', 'children', allow_duplicate=True),
    Input('export-img-btn', 'n_clicks'),
    State('store-selected-trajectory', 'data'),
    State('store-selected-quantities', 'data'),
    State('store-box-selection', 'data'),
    prevent_initial_call=True,
)
def on_export_img(n_clicks, traj_id: str, selected_qties, box_selection):
    if not traj_id or not has_data_loaded():
        raise PreventUpdate

    seg_df = get_segment(traj_id)
    if seg_df is None:
        return '图片导出失败：轨迹数据不存在'

    # 取元信息：当前 ID 号、帧数、起止时间
    meta_df = get_meta_df()
    if meta_df is None or len(meta_df) == 0:
        return '图片导出失败：元信息不存在'
    meta_row = meta_df[meta_df['trajectory_id'] == traj_id]
    if len(meta_row) == 0:
        return '图片导出失败：元信息不存在'
    meta_row = meta_row.iloc[0]
    original_id = meta_row['original_id']
    total_frames = meta_row['total_frames']
    start_time = str(meta_row['start_time'])
    end_time = str(meta_row['end_time'])

    valid_qties = _ensure_valid_quantities(seg_df, selected_qties)

    highlight_range = None
    if box_selection:
        highlight_range = (box_selection.get('start_idx'), box_selection.get('end_idx'))

    output_dir = _get_export_dir()
    try:
        filepath = export_graph_image(
            seg_df, valid_qties, traj_id, original_id, total_frames,
            start_time, end_time, output_dir,
            highlight_range=highlight_range,
        )
        return f'图片导出成功: {os.path.basename(filepath)}'
    except Exception as e:
        return f'图片导出失败: {str(e)}'


# ============================================================
# 客户端回调: ID 列表更新后自动滚动到已选中条目
# ============================================================
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var el = document.querySelector('.id-list-item.selected');
            if (el) {
                el.scrollIntoView({behavior: 'smooth', block: 'nearest'});
            }
        }, 150);
        return '';
    }
    """,
    Output('scroll-anchor', 'children'),
    Input('id-list-container', 'children'),
)


# ============================================================
# 客户端回调: 文件处理完成后隐藏上传遮罩
# store-data-loaded（波动页）与 cmp-state（对比页）任一变化即关闭遮罩
# ============================================================
app.clientside_callback(
    """
    function(loaded, cmpState) {
        if (window.hideUploadOverlay) { window.hideUploadOverlay(); }
        return '';
    }
    """,
    Output('scroll-anchor', 'children', allow_duplicate=True),
    Input('store-data-loaded', 'data'),
    Input('cmp-state', 'data'),
    prevent_initial_call=True,
)





# ============================================================
# 真值对比回调
# ============================================================

def _decode_upload(contents: str) -> bytes:
    """解码 base64 上传内容。"""
    _, encoded = contents.split(',', 1)
    return base64.b64decode(encoded)


# ---- [C1] 模式切换 ----

# Upload 组件模板（每次切换Tab重新生成，修复WebView2事件丢失）
def _make_cmp_radar_upload():
    """生成全新的雷达 Upload 组件（webview2 DOM 刷新）。"""
    return dcc.Upload(
        id='cmp-upload-radar',
        accept='.csv', multiple=True, max_size=500 * 1024 * 1024,  # 500MB
        children=html.Div([
            html.Div('⬆', className='upload-zone-icon'),
            html.Div('拖拽雷达CSV文件到此处（支持多选）', className='upload-zone-text'),
            html.Div('需含 Dx/Dy 列 (.csv)', className='upload-zone-hint'),
        ], className='upload-zone-inner'),
        className='upload-zone',
        style={'minHeight': '80px', 'padding': '10px'},
    )


def _make_cmp_rtk_upload():
    """生成全新的 RTK Upload 组件（webview2 DOM 刷新）。"""
    return dcc.Upload(
        id='cmp-upload-rtk',
        accept='.csv', multiple=True, max_size=500 * 1024 * 1024,  # 500MB
        children=html.Div([
            html.Div('⬆', className='upload-zone-icon'),
            html.Div('拖拽RTK真值CSV到此处（支持多选）', className='upload-zone-text'),
            html.Div('需含 center_x/center_y 列 (.csv)', className='upload-zone-hint'),
        ], className='upload-zone-inner'),
        className='upload-zone',
        style={'minHeight': '80px', 'padding': '10px'},
    )


@callback(
    Output('panel-wave', 'style'),
    Output('panel-compare', 'style'),
    Output('mode-tab-wave', 'className'),
    Output('mode-tab-compare', 'className'),
    Output('cmp-upload-radar-container', 'children'),
    Output('cmp-upload-rtk-container', 'children'),
    Output('cmp-state', 'data'),
    Output('cmp-preview-card', 'children'),
    Output('cmp-config-card', 'children'),
    Output('cmp-upload-radar-feedback', 'children'),
    Output('cmp-upload-rtk-feedback', 'children'),
    Input('mode-tab-wave', 'n_clicks'),
    Input('mode-tab-compare', 'n_clicks'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_mode_switch(_, __, cmp_state):
    """Tab 切换：显示/隐藏对应面板。保持数据持久化。

    已有数据时只切换可见性不重置状态，确保两个 Tab 数据独立持久。
    仅在首次进入对比且无数据时才初始化干净状态 + 刷新 Upload DOM。
    """
    trig = dash_ctx.triggered[0]['prop_id'] if dash_ctx.triggered else ''
    STYLE_HIDDEN = {'position': 'absolute', 'visibility': 'hidden', 'pointer-events': 'none', 'width': '100%', 'top': 0, 'left': 0}
    STYLE_VISIBLE = {'position': 'relative', 'visibility': 'visible', 'pointer-events': 'auto'}

    if 'mode-tab-compare' in trig:
        # 如果已有对比数据，保持状态不变，仅切换面板可见性
        if cmp_state and (cmp_state.get('radar_meta') or cmp_state.get('rtk_meta')):
            logger.info('[MODE] 切换到真值对比 → 保持已有数据')
            return (
                STYLE_HIDDEN, STYLE_VISIBLE,
                'mode-tab', 'mode-tab active',
                no_update, no_update,
                no_update, no_update, no_update,
                no_update, no_update,
            )
        # 首次进入且无数据 → 初始化干净状态 + 刷新 Upload DOM
        logger.info('[MODE] 切换到真值对比 → 首次初始化')
        fresh_state = {
            'radar_meta': None,
            'rtk_meta': None,
            'selected_id': None,
            'delay_ms': 0,
            'alignment_done': False,
        }
        empty_preview = _cmp_preview_empty()
        empty_config = _cmp_config_blank()
        return (
            STYLE_HIDDEN, STYLE_VISIBLE,
            'mode-tab', 'mode-tab active',
            [_make_cmp_radar_upload()],
            [_make_cmp_rtk_upload()],
            fresh_state,
            empty_preview,
            empty_config,
            '',  # radar feedback 清空
            '',  # rtk feedback 清空
        )
    logger.info('[MODE] 切换到波动分析')
    return (
        STYLE_VISIBLE, STYLE_HIDDEN,
        'mode-tab active', 'mode-tab',
        no_update, no_update,
        no_update, no_update, no_update,
        no_update, no_update,
    )


# ---- [C2] 统一上传 → 解析+缓存+预览+ID发现（单回调，无轮询） ----

@callback(
    Output('cmp-state', 'data', allow_duplicate=True),
    Output('cmp-preview-card', 'children', allow_duplicate=True),
    Output('cmp-config-card', 'children', allow_duplicate=True),
    Output('cmp-upload-radar-feedback', 'children', allow_duplicate=True),
    Output('cmp-upload-rtk-feedback', 'children', allow_duplicate=True),
    Output('cmp-graph', 'figure', allow_duplicate=True),
    Output('cmp-graph-title', 'children', allow_duplicate=True),
    Output('cmp-stats-content', 'children', allow_duplicate=True),
    Output('cmp-bins-content', 'children', allow_duplicate=True),
    Input('cmp-upload-radar', 'contents'),
    Input('cmp-upload-radar', 'filename'),
    Input('cmp-upload-rtk', 'contents'),
    Input('cmp-upload-rtk', 'filename'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_upload(radar_c, radar_n, rtk_c, rtk_n, state):
    """统一上传回调：处理雷达/RTK任意文件上传，直接更新预览和配置。

    核心设计改进（v3）：
      - 单个回调通过 dash_ctx.triggered 区分触发源（雷达 or RTK）
      - 一次调用完成：解析→缓存→复合Store更新→预览渲染→ID发现
      - 不再需要 interval 轮询同步多个 Store，消除时序竞态
      - 预览卡始终同时展示双方数据（已加载✓ / 等待○），无论上传顺序

    数据流：Upload事件 → 解析CSV → 缓存DataFrame → 写入cmp-state
            → 读取state中双方meta → 渲染预览卡 + 配置卡
    """
    trig = dash_ctx.triggered[0]['prop_id'].split('.')[0] if dash_ctx.triggered else ''

    # 确定触发角色
    if 'cmp-upload-radar' in trig:
        role = 'radar'
        new_c = radar_c
        new_n = radar_n
    elif 'cmp-upload-rtk' in trig:
        role = 'rtk'
        new_c = rtk_c
        new_n = rtk_n
    else:
        raise PreventUpdate

    if not new_c:
        raise PreventUpdate

    # ── 统一为多文件列表 ──
    if isinstance(new_c, list):
        contents_list = new_c
        filenames = new_n if isinstance(new_n, list) else [new_n]
    else:
        contents_list = [new_c]
        filenames = [new_n]

    # ── 解码并交给纯服务层解析/校验/合并 ──
    logger.info('[CMP-UPLOAD] 开始处理 %s: 共%d个文件', role, len(contents_list))
    parse_errors = []
    file_payloads = []
    for i, (content, fname) in enumerate(zip(contents_list, filenames)):
        try:
            file_payloads.append((_decode_upload(content), fname))
        except Exception as e:
            logger.exception('[CMP-UPLOAD] %s 文件[%d] %s 解码失败', role, i, fname)
            parse_errors.append(f'{fname}: 文件解码失败（{e}）')

    upload_result = prepare_comparison_upload(file_payloads, role)
    parse_errors.extend(upload_result['errors'])
    info = upload_result['info']
    file_count = upload_result['file_count']
    if info is None:
        fb = html.Span(
            f'❌ {role.upper()}: 全部解析失败 - {"；".join(parse_errors)}',
            className='feedback-error',
        )
        graph_noop = (no_update, no_update, no_update, no_update)
        if role == 'radar':
            return (no_update, no_update, no_update, fb, no_update, *graph_noop)
        return (no_update, no_update, no_update, no_update, fb, *graph_noop)

    # 非致命告警 → 仅日志
    if info.get('warnings'):
        for w in info['warnings']:
            logger.warning('[CMP-UPLOAD] %s: %s', role, w)

    # ── 提取元数据 ──
    tr = info['time_range']
    new_meta = {
        'filename': str(info['filename']),
        'total_rows': int(info['total_rows']),
        'unique_ids': int(info['unique_ids']),
        'sample_rate_hz': float(info['sample_rate_hz']),
        'time_range': [float(tr[0]), float(tr[1])] if tr else [0.0, 0.0],
    }
    logger.info('[CMP-UPLOAD] %s meta: rows=%d ids=%d rate=%.1fHz (来源:%d个文件)',
                role, new_meta['total_rows'], new_meta['unique_ids'],
                new_meta['sample_rate_hz'], file_count)

    # ── 缓存 DataFrame ──
    if role == 'radar':
        set_comparison_data('default', info['df'], None)
    else:
        set_comparison_data('default', None, info['df'])

    # ── 更新复合状态 ──
    state = dict(state) if state else {
        'radar_meta': None, 'rtk_meta': None,
        'selected_id': None, 'selected_file_index': None,
        'delay_ms': 0, 'alignment_done': False,
    }
    state[('radar_meta' if role == 'radar' else 'rtk_meta')] = new_meta
    # 记录雷达文件数量：多文件时需要在 ID 发现中按文件区分同 ID 数据
    if role == 'radar':
        state['radar_file_count'] = file_count
    # 重新上传数据时重置对齐状态，避免旧对齐结果残留
    state['alignment_done'] = False
    state['selected_id'] = None
    state['selected_file_index'] = None
    set_alignment_result('default', None, None)  # 清除旧对齐缓存

    # ── 构建反馈 ──
    ok_fb = html.Span(
        f'✓ {role.upper()}: {new_meta["filename"]} '
        f'({new_meta["total_rows"]}行, {new_meta["unique_ids"]}个ID)',
        className='feedback-info',
    )
    radar_fb = ok_fb if role == 'radar' else no_update
    rtk_fb = ok_fb if role == 'rtk' else no_update

    # ── 判断阶段 → 渲染预览卡 ──
    has_radar = state.get('radar_meta') is not None
    has_rtk = state.get('rtk_meta') is not None

    if has_radar and has_rtk:
        # ═══ 双方就绪 → 完整预览 + ID发现配置 ═══
        try:
            preview = _render_cmp_preview(state['radar_meta'], state['rtk_meta'])
        except Exception:
            logger.exception('[CMP-UPLOAD] _render_cmp_preview 异常')
            preview = _cmp_preview_error(
                str(state.get('radar_meta')), str(state.get('rtk_meta'))
            )

        # ID发现
        radar_df, rtk_df = get_comparison_data('default')
        if radar_df is not None and rtk_df is not None:
            try:
                cmp_cfg = get('comparison', {})
                candidate_result = get_candidate_match_result(radar_df, rtk_df, cmp_cfg)
                id_list = candidate_result['candidate_ids']
                logger.info('[CMP-UPLOAD] ID发现: %d个ID', len(id_list))
                state['cached_id_list'] = id_list  # 缓存到复合状态，后续ID切换直接复用
                state['id_filter_stats'] = candidate_result.get('filter_stats')
                config = _build_cmp_config_with_ids(
                    id_list, candidate_result.get('filter_stats'),
                )
            except Exception:
                logger.exception('[CMP-UPLOAD] ID发现失败')
                config = _build_cmp_config_error()
        else:
            logger.warning('[CMP-UPLOAD] cache缺失: radar=%s rtk=%s',
                           radar_df is not None, rtk_df is not None)
            config = _cmp_config_blank()

        return (state, preview, config, radar_fb, rtk_fb,
                go.Figure(), '请选择目标ID并执行对齐', _cmp_stats_placeholder(), _cmp_bins_placeholder())
    else:
        # ═══ 等待态 → 并列展示双方各自状态 ═══
        preview = _render_cmp_waiting_preview(
            state.get('radar_meta'), state.get('rtk_meta')
        )
        config = html.Div([
            html.Div('对齐配置', className='app-card-title'),
            html.Div(f'已加载: {["雷达" if has_radar else "", "RTK" if has_rtk else ""]}  —  请上传另一个CSV文件',
                     className='stats-empty'),
        ], className='app-card')
        logger.info('[CMP-UPLOAD] → 等待态 (radar=%s rtk=%s)', has_radar, has_rtk)
        return (state, preview, config, radar_fb, rtk_fb,
                go.Figure(), '请选择目标ID并执行对齐', _cmp_stats_placeholder(), _cmp_bins_placeholder())


# ---- [C3] ID选择 + 自动诊断 ----

@callback(
    Output('cmp-state', 'data', allow_duplicate=True),
    Output('cmp-delay-input', 'value', allow_duplicate=True),
    Output('cmp-delay-feedback', 'children', allow_duplicate=True),
    Output('cmp-coord-diag', 'children', allow_duplicate=True),
    Output('cmp-run-feedback', 'children', allow_duplicate=True),
    Output('cmp-id-list', 'children', allow_duplicate=True),
    Input({'type': 'cmp-id-item', 'index': ALL}, 'n_clicks'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_id_select(n_clicks_list, state):
    """ID选择 → 坐标诊断 + 延迟自动检测。"""
    if not state or not dash_ctx.triggered:
        raise PreventUpdate

    has_radar = state.get('radar_meta') is not None
    has_rtk = state.get('rtk_meta') is not None
    if not has_radar or not has_rtk:
        raise PreventUpdate

    trig = dash_ctx.triggered[0]
    trig_id = trig['prop_id'].split('.')[0]
    try:
        trig_dict = json.loads(trig_id)
        raw_index = trig_dict['index']
    except (json.JSONDecodeError, KeyError):
        raise PreventUpdate

    # 解析复合索引：多文件时为 "track_id:::file_index"，否则为纯 track_id
    selected_id, selected_fi, selected_seg = _parse_cmp_index(str(raw_index))
    # 构建显示用的标识字符串（用于高亮和状态存储）
    composite_id = raw_index if isinstance(raw_index, str) else str(raw_index)

    radar_df, rtk_df = get_comparison_data('default')
    if radar_df is None or rtk_df is None:
        raise PreventUpdate

    cmp_cfg = get('comparison', {})
    try:
        selection = resolve_track_selection(
            radar_df, rtk_df, cmp_cfg,
            selected_id, selected_fi, selected_seg, state.get('cached_id_list'),
        )
    except ValueError:
        raise PreventUpdate
    analysis = analyse_selected_track(
        radar_df, rtk_df, cmp_cfg, selection['track_id'],
        file_index=selection['file_index'],
        segment_index=selection['segment_index'],
        rtk_id=selection['rtk_id'],
        rtk_file_index=selection['rtk_file_index'],
        cached_ids=state.get('cached_id_list'),
    )
    coord_result = analysis['coordinate']
    coord_html = html.Span(coord_result['diagnosis'],
                           className='feedback-info' if coord_result['same_system'] else 'feedback-error')

    delay_result = analysis['delay']
    optimal_delay = analysis['suggested_delay_ms']
    level = delay_result.get('level', 'insensitive')
    cls = {
        'insensitive': 'feedback-success',
        'no_compensation': 'feedback-info',
        'need_compensation': 'feedback-warn',
    }.get(level, 'feedback-info')
    icon = '✅ ' if level in ('insensitive', 'no_compensation') else '⚠️ '
    delay_html = html.Span(icon + delay_result['recommendation'], className=cls)

    # 更新ID列表高亮 — 复用缓存，避免重跑全量ID扫描
    id_list = analysis['candidate_ids']
    state['cached_id_list'] = id_list
    id_html = _render_cmp_id_list(id_list, composite_id)

    # 更新复合状态中的 selected_id + file_index + delay
    state['selected_id'] = selection['track_id']
    state['selected_file_index'] = selection['file_index']
    state['selected_segment_index'] = selection['segment_index']
    state['selected_rtk_id'] = selection['rtk_id']
    state['delay_ms'] = optimal_delay
    return state, optimal_delay, delay_html, coord_html, '', id_html


# ---- [C4] 执行对齐 ----

@callback(
    Output('cmp-graph', 'figure'),
    Output('cmp-graph-title', 'children'),
    Output('cmp-stats-content', 'children'),
    Output('cmp-bins-content', 'children'),
    Output('cmp-state', 'data', allow_duplicate=True),
    Output('cmp-run-feedback', 'children', allow_duplicate=True),
    Input('cmp-run-btn', 'n_clicks'),
    State('cmp-state', 'data'),
    State('cmp-delay-input', 'value'),
    State('cmp-quantity-checklist', 'value'),
    prevent_initial_call=True,
)
def on_cmp_run(_n, state, delay_ms, selected_qties):
    """一键执行对齐 + 展示全部结果。"""
    if not _n or not state:
        raise PreventUpdate

    has_radar = state.get('radar_meta') is not None
    has_rtk = state.get('rtk_meta') is not None
    if not has_radar or not has_rtk:
        raise PreventUpdate

    radar_df, rtk_df = get_comparison_data('default')
    if radar_df is None or rtk_df is None:
        raise PreventUpdate

    cmp_cfg = get('comparison', {})

    try:
        selection = resolve_track_selection(
            radar_df, rtk_df, cmp_cfg,
            state.get('selected_id'), state.get('selected_file_index'),
            state.get('selected_segment_index'), state.get('cached_id_list'),
        )
    except ValueError:
        raise PreventUpdate
    track_id = selection['track_id']
    file_index = selection['file_index']
    state['cached_id_list'] = selection['candidate_ids']

    if not selected_qties:
        selected_qties = get('comparison', {}).get('default_quantities', ['cmp_dx', 'cmp_dy'])

    try:
        # 执行对齐
        result = execute_alignment(
            radar_df, rtk_df, cmp_cfg, track_id,
            delay_ms=delay_ms or 0,
            file_index=file_index,
            segment_index=selection['segment_index'],
            rtk_id=selection['rtk_id'],
            rtk_file_index=selection['rtk_file_index'],
        )
    except Exception as e:
        empty_fig = go.Figure()
        state['alignment_done'] = False
        state['selected_id'] = track_id
        return (
            empty_fig, '对齐失败',
            html.Div([html.Div('误差统计', className='stats-card-title'),
                      html.Div(f'错误: {e}', className='stats-empty')], className='stats-card'),
            html.Div([html.Div('分距离区间统计', className='stats-card-title'),
                      html.Div(f'错误: {e}', className='stats-empty')], className='stats-card'),
            state,
            f'对齐失败: {e}',
        )

    aligned_df = result['aligned_df']
    summary = result['summary']
    match_summary = result['match_summary']

    if len(aligned_df) == 0:
        empty_fig = go.Figure()
        state['alignment_done'] = False
        state['selected_id'] = track_id
        return (
            empty_fig, '无匹配数据',
            render_cmp_error_stats_empty(), render_cmp_bins_empty(),
            state, '无匹配帧',
        )

    # 缓存结果
    set_alignment_result('default', aligned_df, summary, result.get('rtk_curve_df'))

    # 构建图表
    cmp_qties = cmp_cfg.get('quantities', {})
    fig = build_comparison_subplots(
        aligned_df, selected_qties, cmp_qties,
        trajectory_label=f'ID={track_id}',
        rtk_curve_df=result.get('rtk_curve_df'),
    )

    # 标题
    # 标题（多文件时附加文件序号）
    if file_index is not None:
        title_id = f'ID={track_id} [文件{file_index + 1}]'
    else:
        title_id = f'ID={track_id}'
    title = [
        html.Span(title_id, className='traj-id-badge'),
        html.Span(' | ', style={'color': '#94a3b8'}),
        html.Span(
            f"{match_summary['matched_frames']}/{match_summary['total_frames']}帧匹配",
            className='frame-count',
        ),
        html.Span(' | ', style={'color': '#94a3b8'}),
        html.Span(
            f"ΔDist RMSE={summary['pos_error_abs']['rmse']}m",
            className='frame-count',
        ),
    ]

    # 统计面板
    stats_html = render_cmp_error_stats(summary, match_summary)

    # 距离区间统计
    bins = cmp_cfg.get('distance_bins', [0, 10, 20, 50, 70, 100, 150])
    bin_stats = compute_distance_bin_stats(aligned_df, bins)
    bins_html = render_cmp_distance_bins(bin_stats)

    # 更新状态
    state['alignment_done'] = True
    state['selected_id'] = track_id
    state['selected_file_index'] = file_index
    state['selected_segment_index'] = selection['segment_index']
    state['selected_rtk_id'] = selection['rtk_id']
    state['delay_ms'] = delay_ms or 0

    return (
        fig, title, stats_html, bins_html,
        state,
        f'✓ 对齐完成 — {match_summary["matched_frames"]}/{match_summary["total_frames"]}帧匹配',
    )


# ---- [C5] 对比指标切换 ----

@callback(
    Output('cmp-graph', 'figure', allow_duplicate=True),
    Input('cmp-quantity-checklist', 'value'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_quantity_change(selected_qties, state):
    """切换对比指标 → 重建对比子图。"""
    if not state or not state.get('alignment_done'):
        raise PreventUpdate

    aligned_df, _summary = get_alignment_result('default')
    if aligned_df is None or len(aligned_df) == 0:
        raise PreventUpdate

    if not selected_qties:
        selected_qties = get('comparison', {}).get('default_quantities', ['cmp_dx', 'cmp_dy'])

    cmp_qties = get('comparison', {}).get('quantities', {})
    fig = build_comparison_subplots(
        aligned_df, selected_qties, cmp_qties,
        rtk_curve_df=get_rtk_curve_result('default'),
    )
    return fig


# ---- [C6] 导出 CSV ----

@callback(
    Output('cmp-export-feedback', 'children'),
    Input('cmp-export-csv-btn', 'n_clicks'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_export_csv(_n, state):
    """导出对齐结果 CSV。"""
    if not _n or not state or not state.get('alignment_done'):
        raise PreventUpdate

    aligned_df, _summary = get_alignment_result('default')
    if aligned_df is None or len(aligned_df) == 0:
        return '无数据可导出'

    output_dir = _get_export_dir()
    filepath = os.path.join(output_dir, 'aligned_result.csv')
    msg = export_aligned_csv(aligned_df, filepath)
    return msg


# ---- [C7] 导出图表 ----

@callback(
    Output('cmp-export-feedback', 'children', allow_duplicate=True),
    Input('cmp-export-img-btn', 'n_clicks'),
    State('cmp-state', 'data'),
    State('cmp-graph', 'figure'),
    prevent_initial_call=True,
)
def on_cmp_export_img(_n, state, fig_data):
    """导出对比图表 PNG。"""
    if not _n or not state or not state.get('alignment_done'):
        raise PreventUpdate

    if fig_data is None:
        return '无图表可导出'

    try:
        import plotly.io as pio

        output_dir = _get_export_dir()
        export_cfg = get('comparison', {})
        fmt = export_cfg.get('export_image_format', 'png')
        filename = f'comparison_chart.{fmt}'
        filepath = os.path.join(output_dir, filename)

        pio.write_image(fig_data, filepath,
                        width=1600, height=900, scale=2)
        return f'图表已导出: {filename}'
    except ImportError:
        return '导出图表需要安装 kaleido: pip install kaleido'
    except Exception as e:
        return f'导出失败: {e}'


# ============================================================
# 对比回调辅助函数
# ============================================================

# ---- 空白占位组件 ----

def _cmp_stats_placeholder():
    """误差统计面板空白占位。"""
    return html.Div([
        html.Div('误差统计', className='stats-card-title'),
        html.Div('执行对齐后显示', className='stats-empty'),
    ])


def _cmp_bins_placeholder():
    """距离区间统计面板空白占位。"""
    return html.Div([
        html.Div('分距离区间统计', className='stats-card-title'),
        html.Div('执行对齐后显示', className='stats-empty'),
    ])


def _cmp_config_blank():
    """配置卡空白态（仅返回内部内容，外层app-card+title已在layout模板中）。"""
    return html.Div('请先上传两个CSV文件', className='stats-empty')


def _build_cmp_config_with_ids(id_list, filter_stats=None):
    """根据ID列表构建配置卡内部内容（不含外层app-card+title，因layout模板已提供）。"""
    id_html = _render_cmp_id_list(id_list)
    filter_summary = None
    if filter_stats:
        filter_summary = html.Div(
            (
                f'ID过滤：总计 {filter_stats.get("total_radar_targets", 0)}，'
                f'静止剔除 {filter_stats.get("filtered_static_targets", 0)}，'
                f'短轨迹剔除 {filter_stats.get("filtered_short_tracks", 0)}，'
                f'有效关联 {filter_stats.get("matched_target_pairs", 0)}'
            ),
            className='feedback-muted',
            style={'fontSize': '11px', 'marginBottom': '6px'},
        )
    return html.Div([
        html.Div('目标ID',
                 style={'fontSize': '12px', 'color': '#64748b', 'marginBottom': '4px'}),
        filter_summary,
        html.Div(id='cmp-id-list', className='id-list', children=id_html),
        html.Div('时间延迟',
                 style={'fontSize': '12px', 'color': '#64748b',
                        'marginTop': '8px', 'marginBottom': '4px'}),
        dcc.Input(
            id='cmp-delay-input', type='number', value=0,
            min=-200, max=200, step=1,
            className='form-control form-control-sm',
            style={'width': '100px', 'display': 'inline-block'},
        ),
        html.Span(' ms', style={'fontSize': '12px', 'color': '#94a3b8', 'marginLeft': '4px'}),
        html.Div(id='cmp-delay-feedback', className='feedback-muted mt-1'),
        html.Div(id='cmp-coord-diag', className='feedback-muted mt-1'),
        html.Button('执行对齐', id='cmp-run-btn', n_clicks=0,
                    className='export-btn', style={'marginTop': '12px'}),
        html.Div(id='cmp-run-feedback', className='feedback-muted mt-1'),
    ])


def _cmp_preview_empty():
    """预览卡空白态。"""
    return html.Div([
        html.Div('数据预览', className='app-card-title'),
        html.Div('请先上传两个CSV文件', className='stats-empty'),
    ], className='app-card')


def _cmp_preview_error(radar_str: str, rtk_str: str):
    """预览卡渲染异常时的兜底展示。"""
    return html.Div([
        html.Div('数据预览', className='app-card-title'),
        html.Div([
            html.P('预览渲染异常，请检查上传的文件格式是否正确。',
                   style={'color': '#ef4444', 'marginBottom': '8px'}),
            html.Details([
                html.Summary('调试信息', style={'cursor': 'pointer', 'fontSize': '11px',
                                                'color': '#94a3b8'}),
                html.Pre(f'雷达meta: {radar_str[:300]}\nRTK meta: {rtk_str[:300]}',
                         style={'fontSize': '10px', 'color': '#64748b',
                                'maxHeight': '200px', 'overflow': 'auto',
                                'background': '#1e293b', 'padding': '8px',
                                'borderRadius': '4px', 'marginTop': '4px'}),
            ]),
        ], className='stats-empty'),
    ], className='app-card')


def _build_cmp_config_error():
    """配置卡ID发现失败时的兜底展示（仅返回内部内容，外层app-card+title已在layout模板中）。"""
    return html.Div('ID发现失败，请检查两份数据的时间范围是否存在交叉。',
                    style={'color': '#f59e0b', 'fontSize': '13px'})


def _cmp_id_list_empty():
    """ID列表空白态。"""
    return [html.Div('上传后自动发现', className='id-list-item',
                     style={'cursor': 'default', 'color': '#94a3b8'})]


def _render_cmp_waiting_preview(radar_meta, rtk_meta):
    """等待态预览：并列展示雷达和RTK各自独立的加载状态。
    无论上传顺序如何，两文件的状态均完整呈现。
    """
    def _row(label, info):
        """单文件状态行：已加载→显示摘要，未加载→显示等待提示。"""
        if info is not None and isinstance(info, dict):
            rows = info.get('total_rows', 0)
            ids = info.get('unique_ids', 0)
            summary = f'{rows}行, {ids}个ID'
            icon = '✓'
            color = '#22c55e'
            tip = summary
        else:
            icon = '○'
            color = '#64748b'
            tip = '等待上传'
        return html.Tr([
            html.Td(label, style={'fontWeight': '600', 'color': '#e2e8f0',
                                  'whiteSpace': 'nowrap', 'paddingRight': '12px'}),
            html.Td(icon, style={'color': color, 'fontSize': '14px',
                                 'textAlign': 'center', 'width': '28px'}),
            html.Td(tip, style={'color': '#94a3b8' if info is None else '#e2e8f0',
                                'fontSize': '13px'}),
        ])

    return html.Div([
        html.Div('数据预览', className='app-card-title'),
        html.Table([
            html.Tbody([
                _row('雷达', radar_meta),
                _row('RTK',  rtk_meta),
            ]),
        ], style={'width': '100%', 'borderCollapse': 'collapse'}),
    ], className='app-card')


def _render_cmp_preview(radar_info, rtk_info):
    """渲染数据预览卡（双方就绪时使用）。"""
    def _fmt_ts(epoch):
        if epoch == 0:
            return '—'
        import datetime as _dt
        return _dt.datetime.fromtimestamp(epoch).strftime('%H:%M:%S')

    return html.Div([
        html.Div('数据预览', className='app-card-title'),
        html.Table([
            html.Thead(html.Tr([
                html.Th(''), html.Th('雷达'), html.Th('RTK'),
            ])),
            html.Tbody([
                html.Tr([html.Td('行数'), html.Td(str(radar_info['total_rows'])),
                         html.Td(str(rtk_info['total_rows']))]),
                html.Tr([html.Td('ID数'), html.Td(str(radar_info['unique_ids'])),
                         html.Td(str(rtk_info['unique_ids']))]),
                html.Tr([html.Td('采样率'), html.Td(f"{radar_info['sample_rate_hz']}Hz"),
                         html.Td(f"{rtk_info['sample_rate_hz']}Hz")]),
                html.Tr([html.Td('起始'), html.Td(_fmt_ts(radar_info['time_range'][0])),
                         html.Td(_fmt_ts(rtk_info['time_range'][0]))]),
                html.Tr([html.Td('结束'), html.Td(_fmt_ts(radar_info['time_range'][1])),
                         html.Td(_fmt_ts(rtk_info['time_range'][1]))]),
            ]),
        ], className='traj-table'),
    ], className='app-card')


def _render_cmp_id_list(id_list, selected_id=None):
    """渲染目标ID列表（两行布局：上行 ID·重合率·帧数，下行 起始时间-结束时间）。
    多文件场景：为每个文件独立显示同 ID 条目，使用复合索 :: : 区分。
    """
    if not id_list:
        return [html.Div('未发现目标ID', className='id-list-item',
                         style={'cursor': 'default', 'color': '#94a3b8'})]

    items = []
    for r in id_list:
        if r['total_frames'] == 0:
            continue
        cls = 'id-list-item cmp-id-item'
        tid = int(r['track_id'])  # 转Python原生int，Dash dict id不接受np.int64
        fi = r.get('file_index')

        # 构建复合索引：同一文件内 Track_Age 重置后也必须区分轨迹段。
        segment_index = r.get('segment_index')
        if segment_index is not None:
            # 单文件同 ID 同样可能有多个已确认生命周期段；必须携带段号，
            # 否则点击列表项会退化为只选中该 ID 的第一段。
            file_token = str(fi) if fi is not None else 'none'
            composite_index = f'{tid}:::{file_token}:::{segment_index}'
            file_label = f'文件{fi + 1}·' if fi is not None else ''
            id_label = f'ID={tid} [{file_label}段{segment_index}]'
        elif fi is not None:
            composite_index = f'{tid}:::{fi}'
            id_label = f'ID={tid} [文件{fi + 1}]'
        else:
            composite_index = str(tid)
            id_label = f'ID={tid}'

        if composite_index == selected_id:
            cls += ' selected'
        rate_pct = int(r['overlap_rate'] * 100)

        # 第一行：ID=36 · 85% · 123/145帧
        row1 = html.Div([
            html.Span(id_label, style={'fontWeight': '600'}),
            *(
                [
                    html.Span(' → ', style={'color': '#94a3b8', 'margin': '0 2px'}),
                    html.Span(f'RTK={r["rtk_id"]}', className='id-meta'),
                ] if r.get('rtk_id') is not None else []
            ),
            html.Span(' · ', style={'color': '#94a3b8', 'margin': '0 2px'}),
            html.Span(f'{rate_pct}%', className='id-meta'),
            html.Span(' · ', style={'color': '#94a3b8', 'margin': '0 2px'}),
            html.Span(f'{r["matched_frames"]}/{r["total_frames"]}帧', className='id-meta'),
            *(
                [
                    html.Span(' · ', style={'color': '#94a3b8', 'margin': '0 2px'}),
                    html.Span(f'{r["gap_count"]}处中断', className='id-meta'),
                ] if r.get('gap_count', 0) else []
            ),
        ], className='cmp-id-row1')

        # 第二行：DD/HH/MM/mmm-DD/HH/MM/mmm 格式时间范围
        row2_text = _format_cmp_time_range(r.get('time_start_str'), r.get('time_end_str'))
        row2 = html.Div(row2_text, className='cmp-id-row2')

        items.append(html.Div(
            id={'type': 'cmp-id-item', 'index': composite_index},
            className=cls,
            n_clicks=0,
            children=[row1, row2],
        ))
    return items


def _parse_cmp_index(index_str: str) -> tuple:
    """解析对比模块的复合索引字符串。

    格式:
      - 单文件: "5"              → (5, None, None)
      - 多文件: "5:::0"           → (5, 0, None)
      - 生命周期段: "5:::0:::2"   → (5, 0, 2)
      - 单文件生命周期段: "5:::none:::2" → (5, None, 2)

    Returns:
        (track_id: int, file_index: int | None, segment_index: int | None)
    """
    if ':::' in index_str:
        parts = index_str.split(':::')
        if len(parts) == 3:
            file_index = None if parts[1] == 'none' else int(parts[1])
            return int(parts[0]), file_index, int(parts[2])
        return int(parts[0]), int(parts[1]), None
    return int(index_str), None, None


def _format_cmp_time_range(start_str, end_str):
    """将原始时间戳字符串 YYYY_MM_DD_HH_MM_SS_mmm 格式化为 DD/HH/MM/mmm。

    Returns:
        str: 如 '04/10/18/507-04/10/18/507' 或 '—'（无数据时）
    """
    if not start_str or not end_str:
        return '—'

    def _fmt_one(ts):
        try:
            parts = ts.replace('-', '_').replace(':', '_').replace(' ', '_').split('_')
            # 期望格式 YYYY_MM_DD_HH_MM_SS_mmm (7段) 或 YYYY_MM_DD_HH_MM_SS (6段)
            if len(parts) >= 5:
                day = parts[2]     # DD
                hour = parts[3]    # HH
                minute = parts[4]  # MM
                ms = parts[6] if len(parts) >= 7 else '000'
                return f'{day}/{hour}/{minute}/{ms}'
        except Exception:
            pass
        return ts

    s = _fmt_one(start_str)
    e = _fmt_one(end_str)
    if s == e:
        return s
    return f'{s}-{e}'
