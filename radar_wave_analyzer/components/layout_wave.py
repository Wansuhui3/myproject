"""波动分析页布局：左侧控制面板 | 中间图表区 | 右侧统计面板。

结构：
- 左侧（2/12）：时间筛选卡 / 目标ID列表 / 轨迹段卡
- 中间（7/12）：物理量多选 Checklist + 纵向堆叠子图（共享X轴）
- 右侧（3/12）：全段统计卡 + 选中区域统计卡 + 波动摘要卡
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


# ===================== 左侧面板 =====================

def build_time_filter_card() -> html.Div:
    """时间筛选卡。"""
    return html.Div([
        html.Div('时间筛选', className='app-card-title'),
        dcc.Loading(
            id='loading-upload',
            type='circle',
            color='#3b82f6',
            children=html.Div([
                dcc.Upload(
                    id='upload-csv',
                    accept='.csv,.json',
                    multiple=True,
                    max_size=500 * 1024 * 1024,  # 500MB
                    children=html.Div(
                        [
                            html.Div('⬆', className='upload-zone-icon'),
                            html.Div('拖拽CSV/JSON文件到此处 或 点击选择', className='upload-zone-text'),
                            html.Div('支持多文件同时拖入 (.csv/.json)', className='upload-zone-hint'),
                        ],
                        className='upload-zone-inner',
                    ),
                    className='upload-zone',
                ),
                html.Div(id='upload-feedback', className='feedback-muted mt-1'),
            ]),
        ),
        html.Label('定位时间', className='field-label'),
        dcc.Input(
            id='timestamp-input',
            type='text',
            placeholder='YYYY-MM-DD HH:MM:SS',
            className='form-control form-control-sm',
            debounce=True,
        ),
        html.Button('一键清除', id='wave-clear-btn', n_clicks=0,
                    className='danger-text-btn'),
        html.Div(id='timestamp-feedback', className='feedback-muted mt-2'),
    ], className='app-card')


def build_id_list_card() -> html.Div:
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


def build_trajectory_card() -> html.Div:
    """轨迹段卡。"""
    return html.Div([
        html.Div([
            html.Span('轨迹段', className='fw-bold'),
            html.Span(id='traj-id-badge', className='badge', children=''),
        ], className='app-card-title'),
        html.Div(id='trajectory-table'),
    ], className='app-card')


def build_left_panel() -> dbc.Col:
    """左侧控制面板。"""
    return dbc.Col([
        build_time_filter_card(),
        build_id_list_card(),
        build_trajectory_card(),
    ], width=2, className='side-panel left-panel')


# ===================== 中间面板 =====================

def build_quantity_checklist() -> html.Div:
    """物理量多选列表；选项在 CSV 加载后按实际字段动态填充。"""

    return html.Div([
        html.Div('物理量', className='app-card-title'),
        dcc.Checklist(
            id='quantity-checklist',
            options=[],
            value=[],
            className='qty-checklist',
            labelClassName='qty-checklist-label',
        ),
    ], className='app-card qty-panel')


def build_center_panel() -> dbc.Col:
    """中间图表区：顶部指标工具栏 + 主图表。"""
    return dbc.Col([
        build_quantity_checklist(),
        html.Div([
            html.Div(
                id='graph-title-bar',
                className='graph-title-bar',
                children=[html.Span('请先拖入数据并选择轨迹', className='feedback-muted')],
            ),
            dcc.Loading(
                id='loading-graph', type='circle', color='#3b82f6',
                parent_className='loading-graph-inner',
                children=dcc.Graph(
                    id='trajectory-graph',
                    config={
                        'displayModeBar': True, 'displaylogo': False,
                        'modeBarButtons': [
                            ['select2d', 'pan2d'],
                            ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                        ],
                        'responsive': True, 'scrollZoom': True,
                        'doubleClick': 'reset+autosize',
                    },
                    style={'width': '100%', 'height': '100%'},
                ),
            ),
            html.Button('✕ 清除框选', id='clear-box-btn', n_clicks=0, className='clear-box-btn'),
            html.Span(id='box-select-feedback', className='box-select-feedback'),
        ], className='center-graph-wrapper'),

        html.Div(id='current-trajectory-label', style={'display': 'none'}),
        html.Div(id='graph-resize-trigger', style={'display': 'none'}),
    ], width=7, className='center-panel')


# ===================== 右侧面板 =====================

def build_right_panel() -> dbc.Col:
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

        # 波动摘要（可插入快照、可复制，交互与真值对比分距离摘要一致）
        html.Div([
            html.Div([
                html.Div('波动摘要', className='stats-card-title'),
                html.Div([
                    html.Button('插入当前', id='wave-summary-insert-btn', n_clicks=0,
                                className='perf-summary-action-btn'),
                    html.Button('删除最后', id='wave-summary-remove-btn', n_clicks=0,
                                className='perf-summary-action-btn'),
                    html.Button('清空', id='wave-summary-clear-btn', n_clicks=0,
                                className='perf-summary-action-btn perf-summary-clear-btn'),
                    dcc.Clipboard(
                        id='wave-summary-copy',
                        title='复制全部摘要（不含来源说明行）',
                        style={'display': 'inline-block', 'verticalAlign': 'middle'},
                    ),
                ], className='perf-summary-actions'),
            ], className='perf-summary-header'),
            html.Div(
                html.Div('点击“插入当前”保留波动摘要', className='stats-empty'),
                id='wave-summary-content', className='perf-summary-output',
            ),
            html.Div(id='wave-summary-feedback', className='feedback-muted'),
        ], className='stats-card'),
    ], width=3, className='side-panel right-panel')
