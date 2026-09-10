"""真值对比页布局：左侧上传/预览/映射/对齐配置 | 中间对比曲线 | 右侧统计与导出。"""
import dash_bootstrap_components as dbc
from dash import dcc, html

# ===================== 真值对比 — 左侧面板 =====================

def build_cmp_left_panel() -> dbc.Col:
    """左侧面板：上传卡 + 预览卡 + 对齐配置卡。"""
    return dbc.Col([
        # 卡片1：文件上传（双Upload方案，雷达/RTK各一个独立上传区）
        html.Div([
            html.Div('数据文件', className='app-card-title'),
            # 雷达文件上传（包装容器，模式切换时由回调强制刷新以修复WebView2事件丢失）
            html.Div(id='cmp-upload-radar-container', children=[
                dcc.Upload(
                    id='cmp-upload-radar',
                    accept='.csv,.json', multiple=True, max_size=500 * 1024 * 1024,  # 500MB
                    children=html.Div([
                        html.Div('⬆', className='upload-zone-icon'),
                        html.Div('拖拽雷达CSV/JSON文件到此处（支持多选）', className='upload-zone-text'),
                        html.Div('需含 Dx/Dy 列 (.csv/.json)', className='upload-zone-hint'),
                    ], className='upload-zone-inner'),
                    className='upload-zone',
                    style={'minHeight': '80px', 'padding': '10px'},
                ),
            ]),
            html.Div(id='cmp-upload-radar-feedback', className='feedback-muted mt-1'),
            # RTK文件上传（包装容器，模式切换时由回调强制刷新以修复WebView2事件丢失）
            html.Div(id='cmp-upload-rtk-container', children=[
                dcc.Upload(
                    id='cmp-upload-rtk',
                    accept='.csv,.json', multiple=True, max_size=500 * 1024 * 1024,  # 500MB
                    children=html.Div([
                        html.Div('⬆', className='upload-zone-icon'),
                        html.Div('拖拽RTK真值CSV/JSON到此处（支持多选）', className='upload-zone-text'),
                        html.Div('需含 center_x/center_y 列 (.csv/.json)', className='upload-zone-hint'),
                    ], className='upload-zone-inner'),
                    className='upload-zone',
                    style={'minHeight': '80px', 'padding': '10px'},
                ),
            ]),
            html.Div(id='cmp-upload-rtk-feedback', className='feedback-muted mt-1'),
            html.Button('一键清除', id='cmp-clear-btn', n_clicks=0,
                        className='danger-text-btn'),
        ], className='app-card'),

        # 卡片2：数据预览（默认收起，摘要由 cmp-state 实时更新）
        html.Details([
            html.Summary([
                html.Div([
                    html.Span('▦', className='mapping-summary-icon preview-summary-icon'),
                    html.Div([
                        html.Div('数据预览', className='app-card-title'),
                        html.Div(
                            id='cmp-preview-summary',
                            className='mapping-summary-text',
                            children='等待加载雷达与真值数据',
                        ),
                    ]),
                ], className='mapping-summary-main'),
                html.Span('展开', className='preview-summary-action'),
            ], className='mapping-collapse-summary'),
            html.Div(
                id='cmp-preview-card',
                className='preview-collapse-body',
                children=html.Div('请先上传两个CSV文件', className='stats-empty'),
            ),
        ], className='app-card cmp-preview-panel'),

        # 卡片3：物理量映射（左侧单列折叠配置）
        html.Details([
            html.Summary([
                html.Div([
                    html.Span('⇄', className='mapping-summary-icon'),
                    html.Div([
                        html.Div('物理量映射', className='app-card-title'),
                        html.Div(
                            id='cmp-mapping-summary',
                            className='mapping-summary-text',
                            children='等待加载雷达与真值数据',
                        ),
                    ]),
                ], className='mapping-summary-main'),
                html.Span('展开', className='mapping-summary-action'),
            ], className='mapping-collapse-summary'),
            html.Div([
                html.Div(
                    '物理量名称与 CSV 表头一致',
                    className='mapping-panel-hint',
                ),
                html.Div([
                    html.Span('通道'),
                    html.Span('雷达物理量'),
                    html.Span('真值物理量'),
                    html.Span('单位'),
                    html.Span('操作'),
                ], className='mapping-table-head'),
                html.Div(id='cmp-mapping-list', className='mapping-list', children=[
                    html.Div('上传雷达与 RTK 文件后生成推荐映射', className='stats-empty'),
                ]),
                html.Button('+ 添加通道', id='cmp-add-mapping-btn', n_clicks=0,
                            className='mapping-add-btn mapping-add-btn-block'),
                html.Div(id='cmp-mapping-feedback', className='mapping-feedback'),
            ], className='mapping-collapse-body'),
        ], className='app-card cmp-mapping-panel'),

        # 卡片4：对齐配置（Loading包裹，上传/ID切换时显示加载动画）
        dcc.Loading(
            id='cmp-loading-config', type='circle', color='#3b82f6',
            children=html.Div(id='cmp-config-card', className='app-card', children=[
                html.Div('对齐配置', className='app-card-title'),
                html.Div([
                    html.Div('目标ID', style={'fontSize': '12px', 'color': '#64748b', 'marginBottom': '4px'}),
                    html.Div(id='cmp-id-list', className='id-list', children=[
                        html.Div('上传后自动发现', className='id-list-item',
                                 style={'cursor': 'default', 'color': '#94a3b8'}),
                    ]),
                    html.Div('时间补偿（+ 表示雷达时刻向后移）', style={
                        'fontSize': '12px', 'color': '#64748b',
                        'marginTop': '8px', 'marginBottom': '4px',
                    }),
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
                ]),
            ]),
        ),
    ], width=2, className='side-panel left-panel')


# ===================== 真值对比 — 中间面板 =====================

def build_cmp_center_panel() -> dbc.Col:
    """中间面板：专注展示真值对比曲线。"""
    return dbc.Col([
        html.Div([
            dcc.Loading(
                id='cmp-loading-graph', type='circle', color='#3b82f6',
                parent_className='loading-graph-inner',
                children=dcc.Graph(
                    id='cmp-graph',
                    config={
                        'displayModeBar': True, 'displaylogo': False,
                        'modeBarButtons': [
                            ['pan2d'],
                            ['zoomIn2d', 'zoomOut2d', 'autoScale2d'],
                        ],
                        'responsive': True, 'scrollZoom': True,
                        'doubleClick': 'reset+autosize',
                    },
                    style={'width': '100%', 'height': '100%'},
                ),
            ),
        ], className='center-graph-wrapper'),
        # 性能验收表：可折叠，折叠后图表自动获得全部高度
        html.Div(
            id='perf-panel-container',
            children=html.Div('执行对齐后显示', className='stats-empty'),
            className='perf-panel-container',
        ),
    ], width=7, className='center-panel')


# ===================== 真值对比 — 右侧面板 =====================

def build_cmp_right_panel() -> dbc.Col:
    """右侧面板：误差统计 + 分距离性能摘要 + 导出。"""
    return dbc.Col([
        dcc.Loading(
            id='cmp-loading-stats', type='circle', color='#3b82f6',
            children=html.Div([
                html.Div('误差统计', className='stats-card-title'),
                html.Div('执行对齐后显示', className='stats-empty'),
            ], id='cmp-stats-content', className='stats-card'),
        ),
        dcc.Loading(
            id='cmp-loading-bins', type='circle', color='#3b82f6',
            children=html.Div([
                html.Div([
                    html.Div('分距离性能摘要', className='stats-card-title'),
                    html.Div([
                        html.Button('插入当前', id='perf-summary-insert-btn', n_clicks=0,
                                    className='perf-summary-action-btn'),
                        html.Button('删除最后', id='perf-summary-remove-btn', n_clicks=0,
                                    className='perf-summary-action-btn'),
                        html.Button('清空', id='perf-summary-clear-btn', n_clicks=0,
                                    className='perf-summary-action-btn perf-summary-clear-btn'),
                        dcc.Clipboard(
                            id='perf-summary-copy',
                            title='复制全部摘要（不含来源说明行）',
                            style={'display': 'inline-block', 'verticalAlign': 'middle'},
                        ),
                    ], className='perf-summary-actions'),
                ], className='perf-summary-header'),
                html.Div(
                    html.Div('执行对齐后显示', className='stats-empty'),
                    id='cmp-bins-content', className='perf-summary-output',
                ),
                html.Div(id='perf-summary-feedback', className='feedback-muted'),
            ], id='cmp-bins-card', className='stats-card'),
        ),
        html.Div([
            html.Div('导出', className='app-card-title'),
            html.Button('导出CSV', id='cmp-export-csv-btn', n_clicks=0,
                        className='export-btn'),
            dcc.Download(id='cmp-export-download'),
            html.Div(id='cmp-export-feedback', className='feedback-muted mt-2'),
        ], className='export-card'),
    ], width=3, className='side-panel right-panel')
