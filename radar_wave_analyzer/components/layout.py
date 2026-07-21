"""
布局构建模块。
结构：顶部栏 + 左侧控制面板 | 中间图表区（物理量多选+子图） | 右侧统计面板

目标界面：
- 顶部栏：雷达切换按钮组 + 状态标签
- 左侧（3/12）：时间筛选卡 / 目标ID列表 / 轨迹段卡
- 中间（6/12）：
    - 左窄栏（2/12）：物理量多选 Checklist
    - 右宽栏（10/12）：纵向堆叠子图（共享X轴）+ 提示
- 右侧（3/12）：全段统计卡 + 选中区域统计卡 + 导出卡
"""
from dash import dcc, html
import dash_bootstrap_components as dbc

try:
    from ..config import get
except ImportError:
    from config import get


# ===================== 顶部栏 =====================

def _build_top_bar() -> html.Div:
    """顶部栏：雷达切换 + 雷达状态标签。无雷达配置时隐藏。"""
    radar_sources = get('radar_sources', {})
    radar_options = [
        {'label': cfg['label'], 'value': key}
        for key, cfg in radar_sources.items()
    ]

    if not radar_options:
        # 无雷达选项：隐藏选择器，使用默认值使回调正常工作
        return html.Div([
            dbc.RadioItems(
                id='radar-selector',
                options=[{'label': '', 'value': 'default'}],
                value='default',
                style={'display': 'none'},
            ),
            html.Span(id='radar-position-label', style={'display': 'none'}),
        ])

    default_radar = radar_options[0]['value']

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.RadioItems(
                    id='radar-selector',
                    options=radar_options,
                    value=default_radar,
                    inline=True,
                    className='radar-btn-group',
                ),
            ], width='auto'),
            dbc.Col([
                html.Span(
                    id='radar-position-label',
                    className='top-bar-tag',
                ),
            ], width=True, className='text-end d-flex align-items-center justify-content-end'),
        ], align='center', className='g-2'),
    ], className='top-bar')


# ===================== 左侧面板 =====================

def _build_time_filter_card() -> html.Div:
    """时间筛选卡。"""
    return html.Div([
        html.Div('时间筛选', className='app-card-title'),
        html.Div([
            dcc.Input(
                id='timestamp-input',
                type='text',
                placeholder='YYYY-MM-DD HH:MM:SS',
                className='form-control form-control-sm',
                debounce=True,
            ),
        ]),
        dcc.Loading(
            id='loading-upload',
            type='circle',
            color='#3b82f6',
            children=html.Div([
                dcc.Upload(
                    id='upload-csv',
                    accept='.csv',
                    multiple=True,
                    max_size=500 * 1024 * 1024,  # 500MB
                    children=html.Div(
                        [
                            html.Div('⬆', className='upload-zone-icon'),
                            html.Div('拖拽CSV文件到此处 或 点击选择', className='upload-zone-text'),
                            html.Div('支持多文件同时拖入 (.csv)', className='upload-zone-hint'),
                        ],
                        className='upload-zone-inner',
                    ),
                    className='upload-zone',
                ),
                html.Div(id='upload-feedback', className='feedback-muted mt-1'),
            ]),
        ),
        html.Button('一键清除', id='wave-clear-btn', n_clicks=0,
                    className='export-btn', style={'marginTop': '10px', 'width': '100%',
                                                  'backgroundColor': '#ef4444', 'borderColor': '#ef4444',
                                                  'color': '#ffffff', 'fontWeight': 'bold'}),
        html.Div(id='timestamp-feedback', className='feedback-muted mt-2'),
    ], className='app-card')


def _build_id_list_card() -> html.Div:
    """目标 ID 列表卡。"""
    return html.Div([
        html.Div([
            html.Span('目标 ID', className='fw-bold'),
            html.Span(id='id-count-badge', className='badge', children=''),
        ], className='app-card-title'),
        html.Div(id='id-list-container', className='id-list', children=[
            html.Div('输入时间戳后显示目标', className='id-list-item',
                     style={'cursor': 'default', 'color': '#94a3b8'}),
        ]),
    ], className='app-card')


def _build_trajectory_card() -> html.Div:
    """轨迹段卡。"""
    return html.Div([
        html.Div([
            html.Span('轨迹段', className='fw-bold'),
            html.Span(id='traj-id-badge', className='badge', children=''),
        ], className='app-card-title'),
        html.Div(id='trajectory-table'),
    ], className='app-card')


def _build_left_panel() -> dbc.Col:
    """左侧控制面板。"""
    return dbc.Col([
        _build_time_filter_card(),
        _build_id_list_card(),
        _build_trajectory_card(),
    ], width=2, className='p-2', style={'overflowY': 'auto', 'maxHeight': 'calc(100vh - 56px)'})


# ===================== 中间面板 =====================

def _build_quantity_checklist() -> html.Div:
    """物理量多选列表。"""
    quantities = get('quantities', {})
    default_qty = get('DEFAULT_QUANTITY', 'Dx')

    options = [
        {'label': info['label'], 'value': qty}
        for qty, info in quantities.items()
    ]

    return html.Div([
        html.Div('物理量', className='app-card-title'),
        dcc.Checklist(
            id='quantity-checklist',
            options=options,
            value=[default_qty],
            className='qty-checklist',
            labelClassName='qty-checklist-label',
        ),
    ], className='app-card qty-panel')


def _build_center_panel() -> dbc.Col:
    """中间图表区：左窄（物理量多选）+ 右宽（堆叠子图）。"""
    return dbc.Col([
        dbc.Row([
            # 左窄：物理量多选
            dbc.Col([
                _build_quantity_checklist(),
            ], width=2, style={'paddingRight': '8px'}),

            # 右宽：图表
            dbc.Col([
                html.Div([
                    html.Div(
                        id='graph-title-bar',
                        className='graph-title-bar',
                        children=[
                            html.Span('请先拖入数据并选择轨迹', className='feedback-muted'),
                        ],
                    ),
                    dcc.Loading(
                        id='loading-graph',
                        type='circle',
                        color='#3b82f6',
                        parent_className='loading-graph-inner',
                        children=dcc.Graph(
                            id='trajectory-graph',
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtons': [
                                    ['select2d', 'pan2d'],
                                    ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                                ],
                                'responsive': False,
                                'scrollZoom': True,
                                'doubleClick': 'reset+autosize',
                            },
                            style={'width': '100%', 'height': '100%'},
                        ),
                    ),
                    # 清除框选按钮 + 框选反馈 —— 绝对定位置于图表右上角
                    html.Button('✕ 清除框选', id='clear-box-btn', n_clicks=0,
                                className='clear-box-btn'),
                    html.Span(id='box-select-feedback', className='box-select-feedback'),
                ], className='center-graph-wrapper'),
            ], width=10, style={'paddingLeft': '8px'}),
        ], className='g-0', style={'flex': '1', 'minHeight': '0'}),

        html.Div(id='current-trajectory-label', style={'display': 'none'}),
        html.Div(id='graph-resize-trigger', style={'display': 'none'}),
    ], width=7, className='center-panel')


# ===================== 右侧面板 =====================

def _build_right_panel() -> dbc.Col:
    """右侧统计面板。"""
    return dbc.Col([
        # 全段统计
        dcc.Loading(
            id='loading-stats-full',
            type='circle',
            color='#3b82f6',
            children=html.Div([
                html.Div('全段统计', className='stats-card-title'),
                html.Div('请导入数据并选择轨迹段', className='stats-empty'),
            ], id='stats-full-content', className='stats-card'),
        ),

        # 选中区域统计
        dcc.Loading(
            id='loading-stats-box',
            type='circle',
            color='#3b82f6',
            children=html.Div([
                html.Div('选中区域统计', className='stats-card-title'),
                html.Div('框选曲线区间后显示', className='stats-empty'),
            ], id='stats-box-content', className='stats-card'),
        ),

        # 导出卡
        html.Div([
            html.Div('导出', className='app-card-title'),
            html.Button('导出 CSV', id='export-csv-btn', n_clicks=0, className='export-btn'),
            html.Button('导出图片', id='export-img-btn', n_clicks=0, className='export-btn'),
            html.Div(id='export-feedback', className='feedback-muted mt-2'),
        ], className='export-card'),
    ], width=3, className='p-3', style={'overflowY': 'auto', 'maxHeight': 'calc(100vh - 56px)'})


# ===================== 模式切换 Tab =====================

def _build_mode_tabs() -> html.Div:
    """模式切换 Tab 栏。"""
    return html.Div([
        html.Button('雷达波动分析', id='mode-tab-wave',
                    className='mode-tab active'),
        html.Button('真值对比', id='mode-tab-compare',
                    className='mode-tab'),
    ], className='mode-tabs')


# ===================== 真值对比 — 左侧面板 =====================

def _build_cmp_left_panel() -> dbc.Col:
    """左侧面板：上传卡 + 预览卡 + 对齐配置卡。"""
    return dbc.Col([
        # 卡片1：文件上传（双Upload方案，雷达/RTK各一个独立上传区）
        html.Div([
            html.Div('数据文件', className='app-card-title'),
            # 雷达文件上传（包装容器，模式切换时由回调强制刷新以修复WebView2事件丢失）
            html.Div('雷达CSV', style={'fontSize': '12px', 'color': '#64748b', 'marginBottom': '4px', 'marginTop': '8px'}),
            html.Div(id='cmp-upload-radar-container', children=[
                dcc.Upload(
                    id='cmp-upload-radar',
                    accept='.csv', multiple=True, max_size=500 * 1024 * 1024,  # 500MB
                    children=html.Div([
                        html.Div('⬆', className='upload-zone-icon'),
                        html.Div('拖拽雷达CSV文件到此处（支持多选）', className='upload-zone-text'),
                        html.Div('需含 Dx/Dy 列 (.csv)', className='upload-zone-hint'),
                    ], className='upload-zone-inner'),
                    className='upload-zone',
                    style={'minHeight': '80px', 'padding': '10px'},
                ),
            ]),
            html.Div(id='cmp-upload-radar-feedback', className='feedback-muted mt-1'),
            # RTK文件上传（包装容器，模式切换时由回调强制刷新以修复WebView2事件丢失）
            html.Div('RTK真值CSV', style={'fontSize': '12px', 'color': '#64748b', 'marginBottom': '4px', 'marginTop': '12px'}),
            html.Div(id='cmp-upload-rtk-container', children=[
                dcc.Upload(
                    id='cmp-upload-rtk',
                    accept='.csv', multiple=True, max_size=500 * 1024 * 1024,  # 500MB
                    children=html.Div([
                        html.Div('⬆', className='upload-zone-icon'),
                        html.Div('拖拽RTK真值CSV到此处（支持多选）', className='upload-zone-text'),
                        html.Div('需含 center_x/center_y 列 (.csv)', className='upload-zone-hint'),
                    ], className='upload-zone-inner'),
                    className='upload-zone',
                    style={'minHeight': '80px', 'padding': '10px'},
                ),
            ]),
            html.Div(id='cmp-upload-rtk-feedback', className='feedback-muted mt-1'),
            html.Button('一键清除', id='cmp-clear-btn', n_clicks=0,
                        className='export-btn', style={'marginTop': '10px', 'width': '100%',
                                                      'backgroundColor': '#ef4444', 'borderColor': '#ef4444',
                                                      'color': '#ffffff', 'fontWeight': 'bold'}),
        ], className='app-card'),

        # 卡片2：数据预览
        html.Div(id='cmp-preview-card', className='app-card', children=[
            html.Div('数据预览', className='app-card-title'),
            html.Div('请先上传两个CSV文件', className='stats-empty'),
        ]),

        # 卡片3：对齐配置（Loading包裹，上传/ID切换时显示加载动画）
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
                    html.Div('时间延迟', style={
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
    ], width=2, className='p-2', style={'overflowY': 'auto', 'maxHeight': 'calc(100vh - 56px)'})


# ===================== 真值对比 — 中间面板 =====================

def _build_cmp_center_panel() -> dbc.Col:
    """中间面板：对比指标多选 + 图表。"""
    cmp_config = get('comparison', {})
    cmp_qties = cmp_config.get('quantities', {})
    default_qties = cmp_config.get('default_quantities', ['pos_error_abs'])

    options = [
        {'label': info['label'], 'value': qty}
        for qty, info in cmp_qties.items()
    ]

    return dbc.Col([
        dbc.Row([
            # 左窄：对比指标多选 — 复用 qty-panel / qty-checklist 样式
            dbc.Col([
                html.Div([
                    html.Div('对比指标', className='app-card-title'),
                    dcc.Checklist(
                        id='cmp-quantity-checklist',
                        options=options,
                        value=default_qties,
                        className='qty-checklist',
                        labelClassName='qty-checklist-label',
                    ),
                ], className='app-card qty-panel'),
            ], width=2, style={'paddingRight': '8px'}),

            # 右宽：对比图表
            dbc.Col([
                html.Div([
                    html.Div(
                        id='cmp-graph-title',
                        className='graph-title-bar',
                        children=[html.Span('请上传雷达与RTK数据并执行对齐',
                                            className='feedback-muted')],
                    ),
                    dcc.Loading(
                        id='cmp-loading-graph', type='circle', color='#3b82f6',
                        parent_className='loading-graph-inner',
                        children=dcc.Graph(
                            id='cmp-graph',
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtons': [
                                    ['pan2d'],
                                    ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                                ],
                                'responsive': False,
                                'scrollZoom': True,
                                'doubleClick': 'reset+autosize',
                            },
                            style={'width': '100%', 'height': '100%'},
                        ),
                    ),
                ], className='center-graph-wrapper'),
            ], width=10, style={'paddingLeft': '8px'}),
        ], className='g-0', style={'flex': '1', 'minHeight': '0'}),
    ], width=7, className='center-panel')


# ===================== 真值对比 — 右侧面板 =====================

def _build_cmp_right_panel() -> dbc.Col:
    """右侧面板：误差统计 + 距离区间表 + 导出。"""
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
                html.Div('分距离区间统计', className='stats-card-title'),
                html.Div('执行对齐后显示', className='stats-empty'),
            ], id='cmp-bins-content', className='stats-card'),
        ),
        html.Div([
            html.Div('导出', className='app-card-title'),
            html.Button('导出 CSV', id='cmp-export-csv-btn', n_clicks=0,
                        className='export-btn'),
            html.Button('导出图表', id='cmp-export-img-btn', n_clicks=0,
                        className='export-btn'),
            html.Div(id='cmp-export-feedback', className='feedback-muted mt-2'),
        ], className='export-card'),
    ], width=3, className='p-3', style={'overflowY': 'auto', 'maxHeight': 'calc(100vh - 56px)'})


# ===================== 总布局 =====================

def build_layout() -> html.Div:
    """构建完整布局。"""
    default_qty = get('DEFAULT_QUANTITY', 'Dx')

    return html.Div([
        # 状态存储
        dcc.Store(id='store-data-loaded', data=False),
        dcc.Store(id='store-segments-meta', data=None),
        dcc.Store(id='store-selected-trajectory', data=None),
        dcc.Store(id='store-selected-quantities', data=[default_qty]),
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

        # 顶部栏
        _build_top_bar(),

        # 模式切换 Tab
        _build_mode_tabs(),

        # 波动分析面板（已有，默认可见）
        html.Div([
            dbc.Row([
                _build_left_panel(),
                _build_center_panel(),
                _build_right_panel(),
            ], className='g-0'),
        ], id='panel-wave', style={'position': 'relative', 'visibility': 'visible', 'pointer-events': 'auto'}),

        # 真值对比面板（新增，默认隐藏但保持DOM活跃，避免WebView2中Upload事件丢失）
        html.Div([
            dbc.Row([
                _build_cmp_left_panel(),
                _build_cmp_center_panel(),
                _build_cmp_right_panel(),
            ], className='g-0'),
        ], id='panel-compare', style={
            'position': 'absolute',
            'visibility': 'hidden',
            'pointer-events': 'none',
            'width': '100%',
            'top': 0,
            'left': 0,
        }),

        # 文件上传即时加载遮罩（由 assets/upload_overlay.js 控制显隐）
        html.Div(id='upload-overlay', className='upload-overlay-mask', style={'display': 'none'}, children=[
            html.Div(className='upload-overlay-spinner'),
            html.Div('正在读取并解析文件，请稍候…', className='upload-overlay-text'),
        ]),
    ], className='app-shell')
