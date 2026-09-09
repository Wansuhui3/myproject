"""布局入口：全局状态存储 + 模式切换 Tab + 波动/对比两页面板装配。

各页面板的内部结构见 layout_wave.py 与 layout_comparison.py。
"""
from dash import dcc, html
import dash_bootstrap_components as dbc

from ..config import get
from .layout_comparison import (
    build_cmp_center_panel,
    build_cmp_left_panel,
    build_cmp_right_panel,
)
from .layout_wave import build_center_panel, build_left_panel, build_right_panel


# ===================== 顶部栏 =====================

def _build_radar_selector_controls() -> html.Div:
    """模式栏中的紧凑雷达切换控件；无多雷达配置时保持隐藏。"""
    radar_sources = get('radar_sources', {})
    radar_options = [
        {'label': cfg['label'], 'value': key}
        for key, cfg in radar_sources.items()
    ]

    default_radar = radar_options[0]['value'] if radar_options else 'default'
    selector_options = radar_options or [{'label': '', 'value': 'default'}]

    return html.Div([
        dbc.RadioItems(
            id='radar-selector',
            options=selector_options,
            value=default_radar,
            inline=True,
            className='radar-btn-group',
        ),
        html.Span(id='radar-position-label', className='mode-radar-tag'),
    ], className='mode-radar-controls',
       style={} if radar_options else {'display': 'none'})


def _build_mode_tabs() -> html.Div:
    """模式切换 Tab 栏。"""
    return html.Div([
        html.Div([
            html.Button('波动分析', id='mode-tab-wave', className='mode-tab active'),
            html.Button('真值对比', id='mode-tab-compare', className='mode-tab'),
        ], className='mode-tab-buttons'),
        html.Div('导入数据  ·  选择目标  ·  分析诊断  ·  导出结果', className='workflow-hint'),
        _build_radar_selector_controls(),
    ], className='mode-tabs')


# ===================== 总布局 =====================

def build_layout() -> html.Div:
    """构建完整布局。"""
    return html.Div([
        # 状态存储
        dcc.Store(id='store-data-loaded', data=False),
        dcc.Store(id='store-segments-meta', data=None),
        dcc.Store(id='store-selected-trajectory', data=None),
        dcc.Store(id='store-selected-quantities', data=[]),
        dcc.Store(id='store-box-selection', data=None),
        dcc.Store(id='store-selected-id', data=None),
        html.Div(id='scroll-anchor', style={'display': 'none'}),
        # 对比功能复合状态（统一管理雷达/RTK元数据、ID选择、对齐状态）
        dcc.Store(id='cmp-state', data={
            'radar_meta': None,
            'rtk_meta': None,
            'selected_id': None,
            'delay_ms': 0,
            'alignment_done': False,
        }),
        dcc.Store(id='cmp-mappings', data={'signature': '', 'items': []}),
        # 勾选的中断段（供「对齐所选段」合并对齐）。
        dcc.Store(id='cmp-selected-segments', data={'segments': []}),
        # 手动插入的分距离性能摘要快照；仅在点击“插入当前”时更新。
        dcc.Store(id='perf-summary-snapshots', data={'source_key': '', 'items': []}),
        # 手动插入的波动分析摘要快照；仅在点击“插入当前”时更新。
        dcc.Store(id='wave-summary-snapshots', data={'items': []}),

        # 模式切换 Tab
        _build_mode_tabs(),

        html.Div([
            # 隐藏面板保持 DOM 活跃，修复 WebView2 中 Upload 事件丢失。
            html.Div([
                dbc.Row([build_left_panel(), build_center_panel(), build_right_panel()], className='g-0'),
            ], id='panel-wave', className='analysis-panel',
               style={'position': 'relative', 'visibility': 'visible', 'pointer-events': 'auto'}),
            html.Div([
                dbc.Row([build_cmp_left_panel(), build_cmp_center_panel(), build_cmp_right_panel()], className='g-0'),
            ], id='panel-compare', className='analysis-panel', style={
                'position': 'absolute', 'visibility': 'hidden', 'pointer-events': 'none',
                'width': '100%', 'top': 0, 'left': 0,
            }),
        ], className='workspace-stack'),

        # 文件上传即时加载遮罩（由 assets/upload_overlay.js 控制显隐）
        html.Div(id='upload-overlay', className='upload-overlay-mask', style={'display': 'none'}, children=[
            html.Div(className='upload-overlay-spinner'),
            html.Div('正在读取并解析文件，请稍候…', className='upload-overlay-text'),
        ]),
    ], className='app-shell')
