"""波动分析查询与视图域回调：时间戳筛选 / 物理量选项刷新 / 一键清除。

数据装载域回调（雷达切换 / 上传解析）见 wave_upload_callbacks。
"""
import logging

import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, no_update
from dash.exceptions import PreventUpdate

from ..cache import (
    clear_data_cache,
    get_df,
    get_meta_df,
    get_radar_position,
    get_segment,
)
from ..components.graph_builder import build_multi_subplot_graph
from ..components.wave_stats_panel import (
    render_box_stats_empty,
    render_multi_full_stats,
    render_multi_full_stats_placeholder,
)
from ..config import get
from ..core.data_loader import parse_timestamp
from .wave_helpers import (
    _compute_quantities_stats,
    _discover_quantity_columns,
    _ensure_valid_quantities,
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
            'on_timestamp_input: skipped — data_loaded=%s, has_meta=%s, radar=%s',
            data_loaded, get_meta_df() is not None, get_radar_position(),
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
            'on_timestamp_input: df=%s, meta_df=%s — both must exist',
            df is not None, meta_df is not None,
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
                    'on_timestamp_input: get_segment returned None for '
                    'traj=%s, nearest_id=%s',
                    first_traj_id, nearest_id,
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
        'on_timestamp_input: nearest_id=%s 段数据缺失, radar=%s',
        nearest_id, get_radar_position(),
    )
    return (list_children, str(len(list_children)), feedback,
            None, no_update, no_update, no_update, '', no_update, None, None, nearest_id)


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
