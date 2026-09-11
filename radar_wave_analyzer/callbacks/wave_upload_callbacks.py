"""波动分析数据装载域回调：雷达切换 / 上传解析。

查询与视图域回调（时间戳筛选 / 物理量刷新 / 一键清除）见 wave_query_callbacks；
交互域回调（ID 点击、框选、摘要快照）见 wave_callbacks；
上传解析/合并/分段的纯数据管线见 wave_upload_pipeline。
"""
import logging
import time

from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from ..cache import (
    get_df,
    get_file_path,
    get_meta_df,
    switch_radar,
)
from ..components.wave_stats_panel import (
    render_box_stats_empty,
    render_multi_full_stats_placeholder,
)
from ..config import get
from ..core.data_loader import get_time_range
from .helpers import is_supported_upload
from .wave_upload_pipeline import build_upload_caches, parse_upload_payloads
from .wave_views import _build_id_list_html

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
    # 每次上传结束（成功或失败）都写入新时间戳，供前端可靠解除加载遮罩
    Output('upload-tick', 'data', allow_duplicate=True),
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename'),
    State('radar-selector', 'value'),
    prevent_initial_call=True,
)
def on_upload_csv(contents_list, filenames, radar_key):
    """拖拽上传CSV → 解析合并 → 分段缓存。

    所有失败路径都会写入新的 upload-tick，前端据此必然解除加载遮罩；
    不支持的文件类型在入口即被拒绝，不会进入解析流程。
    """
    tick = time.time()

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
            tick,
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

    # 类型前置校验：accept 只约束文件对话框，拖拽可绕过，此处统一拦截
    unsupported = [str(fn) for fn in filenames if not is_supported_upload(fn)]
    if unsupported:
        return _clear_state(html.Span(
            f'不支持的文件类型: {"；".join(unsupported)}（仅支持 .csv / .json）',
            style={'color': '#dc2626'}))

    try:
        all_dfs, errors, source_files = parse_upload_payloads(contents_list, filenames)

        if not all_dfs:
            err_msg = html.Span(
                f'所有文件解析失败: {"；".join(errors)}',
                style={'color': '#dc2626'})
            return _clear_state(err_msg)

        upload_label = (filenames[0] if len(filenames) == 1
                        else f'{len(filenames)}个文件')
        merged_df, meta_df, segments, source_keys = build_upload_caches(
            all_dfs, upload_label)
    except Exception as e:
        logger.exception('上传文件处理失败')
        return _clear_state(html.Span(
            f'文件处理失败: {type(e).__name__}: {e}',
            style={'color': '#dc2626'}))

    # 恢复上传前选择的来源；若该来源不在本次数据中则使用 combined。
    active_key = radar_key if radar_key in source_keys else 'combined'
    switch_radar(active_key)

    t_min, t_max = get_time_range(merged_df)
    err_suffix = f'（部分失败: {"；".join(errors)}）' if errors else ''
    unknown_warning = ''
    if any(not info['recognized'] for info in source_files.values()):
        unknown_warning = ' | ⚠ 有文件未识别来源，请检查文件名（支持 FLR/RLR）'
    feedback = html.Div(
        f'已加载: {upload_label} | {len(merged_df)}行, {len(meta_df)}段 | '
        f'{t_min.strftime("%Y-%m-%d %H:%M:%S")} ~ '
        f'{t_max.strftime("%Y-%m-%d %H:%M:%S")}{unknown_warning} {err_suffix}',
        className='upload-result-text',
    )

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
        tick,                                    # upload-tick
        )
