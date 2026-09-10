"""真值对比回调（C3/C5）：ID 选择 → 坐标诊断 + 延迟检测；映射变更提示。

模式切换 / 上传 / 映射配置分别见 cmp_mode_callbacks / cmp_upload_callbacks /
cmp_mapping_callbacks；纯渲染 helpers 见 cmp_preview_render。
"""
import json
import logging

from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    html,
)
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..cache import get_comparison_data
from ..comparison.service import analyse_selected_track, resolve_track_selection
from ..config import get
from .cmp_preview_render import _parse_cmp_index, _render_cmp_id_list

logger = logging.getLogger(__name__)


# ---- [C1] 模式切换 ----


# ---- [C2] 统一上传 → 解析+缓存+预览+ID发现（单回调，无轮询） ----


# ---- [C2a] 原始 CSV 物理量映射配置 ----


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

    # 延迟检测仅用于诊断提示：建议值不写入输入框，默认补偿保持 0ms。
    delay_result = analysis['delay']
    level = delay_result.get('level', 'insensitive')
    cls = {
        'insensitive': 'feedback-success',
        'no_compensation': 'feedback-info',
        'need_compensation': 'feedback-warn',
    }.get(level, 'feedback-info')
    icon = '✅ ' if level in ('insensitive', 'no_compensation') else '⚠️ '
    delay_html = html.Span(
        f'{icon}{delay_result["recommendation"]}',
        className=cls,
    )

    # 更新ID列表高亮 — 复用缓存，避免重跑全量ID扫描
    id_list = analysis['candidate_ids']
    state['cached_id_list'] = id_list
    id_html = _render_cmp_id_list(id_list, composite_id)

    # 更新复合状态中的 selected_id + file_index + delay
    # 自动检测的建议值仅在诊断提示中展示，不写入输入框与状态；用户可以
    # 在输入框中手动采用建议值，再执行对齐。
    state['selected_id'] = selection['track_id']
    state['selected_file_index'] = selection['file_index']
    state['selected_segment_index'] = selection['segment_index']
    state['selected_rtk_id'] = selection['rtk_id']
    state['delay_ms'] = 0
    return state, 0, delay_html, coord_html, '', id_html


# ---- [C5] 映射变更提示 ----

@callback(
    Output('cmp-run-feedback', 'children', allow_duplicate=True),
    Input('cmp-mappings', 'data'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_mapping_change(_mapping_store, state):
    """映射变化后提醒重新执行，避免展示旧通道结果。"""
    if not state or not state.get('alignment_done'):
        raise PreventUpdate
    return '物理量映射已改变，请重新点击“执行对齐”刷新曲线与统计'


