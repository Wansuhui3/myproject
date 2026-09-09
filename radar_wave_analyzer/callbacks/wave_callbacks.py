"""波动分析交互域回调：ID/轨迹选择 → 绘图 → 框选 → 统计。

数据加载域回调（雷达切换/上传解析/时间筛选/物理量刷新/清空）
见 wave_upload_callbacks。
"""
import json
import logging

import numpy as np
import pandas as pd
from dash import Input, Output, State, Patch, callback, no_update, ALL
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..config import get

from ..cache import (
    get_df, get_meta_df, get_segment, has_data_loaded,
)

from ..core.selection import extract_x_selection

from ..core.wave_calc import find_max_jump

from ..components.wave_stats_panel import (
    render_multi_full_stats,
    render_multi_box_stats, render_box_stats_empty,
    render_wave_snapshot_summary,
)

from ..components.graph_builder import (
    build_box_jump_shapes,
    build_highlight_shapes, build_multi_subplot_graph,
)

from .helpers import _summary_plain_text
from .wave_helpers import (
    _compute_quantities_stats,
    _ensure_valid_quantities,
    _get_non_highlight_shapes,
    _unavailable_quantities,
)
from .wave_views import (
    _build_id_list_html,
    _build_title_bar,
    _build_trajectory_table,
    _target_meta,
    _target_parts,
)

logger = logging.getLogger(__name__)


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


