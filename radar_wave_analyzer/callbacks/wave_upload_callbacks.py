"""波动分析数据加载域回调：雷达切换 / 上传解析 / 时间筛选 / 物理量刷新 / 清空。

交互域回调（ID 点击、框选、摘要快照）见 wave_callbacks；
上传解析/合并/分段的纯数据管线见 wave_upload_pipeline。
"""
import logging
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update, html
from dash.exceptions import PreventUpdate

from ..config import get

from ..cache import (
    get_df, get_meta_df, get_radar_position, get_file_path,
    get_segment,
    clear_data_cache,
    switch_radar,
)

from ..core.data_loader import parse_timestamp, get_time_range

from ..components.wave_stats_panel import (
    render_multi_full_stats, render_multi_full_stats_placeholder,
    render_box_stats_empty,
)

from ..components.graph_builder import build_multi_subplot_graph

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
from .wave_upload_pipeline import build_upload_caches, parse_upload_payloads

logger = logging.getLogger(__name__)


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

    all_dfs, errors, source_files = parse_upload_payloads(contents_list, filenames)

    if not all_dfs:
        err_msg = html.Span(f'所有文件解析失败: {"；".join(errors)}', style={'color': '#dc2626'})
        return _clear_state(err_msg)

    upload_label = filenames[0] if len(filenames) == 1 else f'{len(filenames)}个文件'
    merged_df, meta_df, segments, source_keys = build_upload_caches(all_dfs, upload_label)

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
