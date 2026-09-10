import logging

import plotly.graph_objects as go
from dash import (
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
    no_update,
)
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..cache import clear_comparison_data
from .cmp_preview_render import (
    _cmp_bins_placeholder,
    _cmp_config_blank,
    _cmp_preview_empty,
    _cmp_stats_placeholder,
)
from .helpers import _perf_placeholder

logger = logging.getLogger(__name__)
"""[C0] 对比页清空 + [C1] 模式切换（面板显隐与上传区重置）。"""



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
    Output('perf-panel-container', 'children', allow_duplicate=True),
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
    perf_panel = _perf_placeholder()
    # 重建Upload容器以彻底清除已上传文件的视觉残留
    return (
        fresh_state,
        [_make_cmp_upload('cmp-upload-radar',
                          '拖拽雷达CSV/JSON文件到此处（支持多选）',
                          '需含 Dx/Dy 列 (.csv/.json)')],   # 重置雷达Upload
        [_make_cmp_upload('cmp-upload-rtk',
                          '拖拽RTK真值CSV/JSON到此处（支持多选）',
                          '需含 center_x/center_y 列 (.csv/.json)')],  # 重置RTK Upload
        '', '',                                              # upload feedbacks
        _cmp_preview_empty(),                                # preview card
        _cmp_config_blank(),                                 # config card（同时销毁嵌套组件）
        go.Figure(),                                         # graph
        html.Span('请上传雷达与RTK数据并执行对齐', className='feedback-muted'),  # graph-title
        _cmp_stats_placeholder(),                            # stats
        _cmp_bins_placeholder(),                             # bins
        perf_panel,                                          # 性能验收表
    )
# Upload 组件模板（每次切换Tab重新生成，修复WebView2事件丢失）
def _make_cmp_upload(component_id: str, title: str, hint: str):
    """生成全新的对比页 Upload 组件（webview2 DOM 刷新）。"""
    return dcc.Upload(
        id=component_id,
        accept='.csv,.json', multiple=True, max_size=500 * 1024 * 1024,  # 500MB
        children=html.Div([
            html.Div('⬆', className='upload-zone-icon'),
            html.Div(title, className='upload-zone-text'),
            html.Div(hint, className='upload-zone-hint'),
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
            [_make_cmp_upload('cmp-upload-radar',
                              '拖拽雷达CSV/JSON文件到此处（支持多选）',
                              '需含 Dx/Dy 列 (.csv/.json)')],
            [_make_cmp_upload('cmp-upload-rtk',
                              '拖拽RTK真值CSV/JSON到此处（支持多选）',
                              '需含 center_x/center_y 列 (.csv/.json)')],
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
