"""真值对比回调：模式切换 → 上传解析 → 映射配置 → ID选择 → 提示。

回调清单：
  on_cmp_clear             → 清空对比页面与对比缓存
  [C1] on_mode_switch      → 模式切换 → 显示/隐藏面板 + 重置对比状态
  [C2] on_cmp_upload       → 统一上传 → 解析+缓存+预览+ID发现（单回调无轮询）
  [C2a] render_cmp_preview_summary / on_cmp_mapping_sources /
       render_cmp_mappings / on_cmp_mapping_edit
                           → 原始CSV物理量映射配置（预览/增删改/排序）
  [C3] on_cmp_id_select    → ID选择 → 坐标诊断 + 延迟检测
  [C5] on_cmp_mapping_change → 映射变更提示
"""
import json
import logging

import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update, html, dcc, ALL
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..config import get

from ..cache import (
    set_comparison_data, get_comparison_data,
    set_alignment_result,
    clear_comparison_data,
)

from ..core.data_loader import identify_radar_source

from ..comparison.parser import validate_overlap

from ..comparison.service import (
    analyse_selected_track, get_candidate_match_result,
    prepare_comparison_upload, resolve_track_selection,
)

from ..comparison.file_identity import compact_filename_label, natural_filename_key

from .helpers import _decode_upload_contents, _perf_placeholder

logger = logging.getLogger(__name__)


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


# ---- [C1] 模式切换 ----

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


# ---- [C2] 统一上传 → 解析+缓存+预览+ID发现（单回调，无轮询） ----

@callback(
    Output('cmp-state', 'data', allow_duplicate=True),
    Output('cmp-preview-card', 'children', allow_duplicate=True),
    Output('cmp-config-card', 'children', allow_duplicate=True),
    Output('cmp-upload-radar-feedback', 'children', allow_duplicate=True),
    Output('cmp-upload-rtk-feedback', 'children', allow_duplicate=True),
    Output('cmp-graph', 'figure', allow_duplicate=True),
    Output('cmp-graph-title', 'children', allow_duplicate=True),
    Output('cmp-stats-content', 'children', allow_duplicate=True),
    Output('cmp-bins-content', 'children', allow_duplicate=True),
    Output('perf-panel-container', 'children', allow_duplicate=True),
    Input('cmp-upload-radar', 'contents'),
    Input('cmp-upload-radar', 'filename'),
    Input('cmp-upload-rtk', 'contents'),
    Input('cmp-upload-rtk', 'filename'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_upload(radar_c, radar_n, rtk_c, rtk_n, state):
    """统一上传回调：处理雷达/RTK任意文件上传，直接更新预览和配置。

    核心设计改进（v3）：
      - 单个回调通过 dash_ctx.triggered 区分触发源（雷达 or RTK）
      - 一次调用完成：解析→缓存→复合Store更新→预览渲染→ID发现
      - 不再需要 interval 轮询同步多个 Store，消除时序竞态
      - 预览卡始终同时展示双方数据（已加载✓ / 等待○），无论上传顺序

    数据流：Upload事件 → 解析CSV → 缓存DataFrame → 写入cmp-state
            → 读取state中双方meta → 渲染预览卡 + 配置卡
    """
    trig = dash_ctx.triggered[0]['prop_id'].split('.')[0] if dash_ctx.triggered else ''

    # 确定触发角色
    if 'cmp-upload-radar' in trig:
        role = 'radar'
        new_c = radar_c
        new_n = radar_n
    elif 'cmp-upload-rtk' in trig:
        role = 'rtk'
        new_c = rtk_c
        new_n = rtk_n
    else:
        raise PreventUpdate

    if not new_c:
        raise PreventUpdate

    # ── 统一为多文件列表 ──
    if isinstance(new_c, list):
        contents_list = new_c
        filenames = new_n if isinstance(new_n, list) else [new_n]
    else:
        contents_list = [new_c]
        filenames = [new_n]

    # ── 解码并交给纯服务层解析/校验/合并 ──
    logger.info('[CMP-UPLOAD] 开始处理 %s: 共%d个文件', role, len(contents_list))
    parse_errors = []
    file_payloads = []
    for i, (content, fname) in enumerate(zip(contents_list, filenames)):
        try:
            file_payloads.append((_decode_upload_contents(content), fname))
        except Exception as e:
            logger.exception('[CMP-UPLOAD] %s 文件[%d] %s 解码失败', role, i, fname)
            parse_errors.append(f'{fname}: 文件解码失败（{e}）')

    upload_result = prepare_comparison_upload(file_payloads, role)
    parse_errors.extend(upload_result['errors'])
    info = upload_result['info']
    file_count = upload_result['file_count']
    routed = upload_result.get('routed')

    # 自动归类只在另一区域为空时生效：不覆盖用户已显式上传的数据
    if routed is not None:
        other_meta_key = 'rtk_meta' if role == 'radar' else 'radar_meta'
        if (state or {}).get(other_meta_key) is not None:
            logger.info('[CMP-UPLOAD] 跳过自动归类: %s 已有显式上传数据', other_meta_key)
            routed = None

    if info is None and routed is None:
        fb = html.Span(
            f'❌ {role.upper()}: 全部解析失败 - {"；".join(parse_errors)}',
            className='feedback-error',
        )
        graph_noop = (no_update,) * 6   # graph/title/stats/bins/perf面板/perf结论
        if role == 'radar':
            return (no_update, no_update, no_update, fb, no_update, *graph_noop)
        return (no_update, no_update, no_update, no_update, fb, *graph_noop)

    # ── 组装本次要更新的数据侧（本区域 + 自动归类的另一区域）──
    side_updates: list[tuple[str, dict, int]] = []
    if info is not None:
        side_updates.append((role, info, file_count))
    if routed is not None:
        side_updates.append((routed['role'], routed['info'], routed['file_count']))

    # ── 更新复合状态 ──
    state = dict(state) if state else {
        'radar_meta': None, 'rtk_meta': None,
        'selected_id': None, 'selected_file_index': None,
        'delay_ms': 0, 'alignment_done': False,
    }

    side_metas: dict[str, dict] = {}
    for side_role, side_info, side_file_count in side_updates:
        # 非致命告警 → 仅日志
        for w in side_info.get('warnings', []):
            logger.warning('[CMP-UPLOAD] %s: %s', side_role, w)

        tr = side_info['time_range']
        side_meta = {
            'filename': str(side_info['filename']),
            'file_count': int(side_file_count),
            'total_rows': int(side_info['total_rows']),
            'unique_ids': int(side_info['unique_ids']),
            'sample_rate_hz': float(side_info['sample_rate_hz']),
            'sample_rate_label': str(side_info.get('sample_rate_label', '')),
            'sample_rate_note': str(side_info.get('sample_rate_note', '')),
            'time_range': [float(tr[0]), float(tr[1])] if tr else [0.0, 0.0],
            'fields': list(side_info.get('fields', [])),
            'physical_fields': list(side_info.get('physical_fields', [])),
            'original_to_internal': dict(side_info.get('original_to_internal', {})),
        }
        side_metas[side_role] = side_meta
        logger.info('[CMP-UPLOAD] %s meta: rows=%d ids=%d rate=%.1fHz (来源:%d个文件)',
                    side_role, side_meta['total_rows'], side_meta['unique_ids'],
                    side_meta['sample_rate_hz'], side_file_count)

        # ── 缓存 DataFrame ──
        if side_role == 'radar':
            set_comparison_data('default', side_info['df'], None)
            state['radar_meta'] = side_meta
            # 记录雷达文件数量：多文件时需要在 ID 发现中按文件区分同 ID 数据
            state['radar_file_count'] = side_file_count
        else:
            set_comparison_data('default', None, side_info['df'])
            state['rtk_meta'] = side_meta

    # 重新上传数据时重置对齐状态，避免旧对齐结果残留
    state['alignment_done'] = False
    state['selected_id'] = None
    state['selected_file_index'] = None
    set_alignment_result('default', None, None)  # 清除旧对齐缓存

    # ── 构建反馈（本区域与自动归类区域各自提示）──
    radar_fb = no_update
    rtk_fb = no_update
    if info is not None:
        meta = side_metas[role]
        ok_fb = html.Span(
            f'✓ {role.upper()}: {meta["filename"]} '
            f'({meta["total_rows"]}行, {meta["unique_ids"]}个ID)',
            className='feedback-info',
        )
        if role == 'radar':
            radar_fb = ok_fb
        else:
            rtk_fb = ok_fb
    else:
        role_label = '雷达' if role == 'radar' else 'RTK真值'
        miss_fb = html.Span(
            f'⚠ 本次上传未包含{role_label}数据，其余文件已自动归类到另一区域',
            className='feedback-muted',
        )
        if role == 'radar':
            radar_fb = miss_fb
        else:
            rtk_fb = miss_fb
    if routed is not None:
        routed_role = routed['role']
        routed_meta = side_metas[routed_role]
        routed_fb = html.Span(
            f'✓ {routed_role.upper()} 自动归类: {routed_meta["filename"]} '
            f'({routed_meta["total_rows"]}行, {routed_meta["unique_ids"]}个ID)',
            className='feedback-info',
        )
        if routed_role == 'radar':
            radar_fb = routed_fb
        else:
            rtk_fb = routed_fb

    # ── 判断阶段 → 渲染预览卡 ──
    has_radar = state.get('radar_meta') is not None
    has_rtk = state.get('rtk_meta') is not None

    if has_radar and has_rtk:
        # ═══ 双方就绪 → 完整预览 + ID发现配置 ═══
        try:
            preview = _render_cmp_preview(state['radar_meta'], state['rtk_meta'])
        except Exception:
            logger.exception('[CMP-UPLOAD] _render_cmp_preview 异常')
            preview = _cmp_preview_error(
                str(state.get('radar_meta')), str(state.get('rtk_meta'))
            )

        # ID发现
        radar_df, rtk_df = get_comparison_data('default')
        if radar_df is not None and rtk_df is not None:
            try:
                cmp_cfg = get('comparison', {})
                candidate_result = get_candidate_match_result(radar_df, rtk_df, cmp_cfg)
                id_list = candidate_result['candidate_ids']
                logger.info('[CMP-UPLOAD] ID发现: %d个ID', len(id_list))
                state['cached_id_list'] = id_list  # 缓存到复合状态，后续ID切换直接复用
                state['id_filter_stats'] = candidate_result.get('filter_stats')
                config = _build_cmp_config_with_ids(
                    id_list, candidate_result.get('filter_stats'),
                )
                # 时间零重叠是“未发现目标”最常见的根因：给出量化提示而非沉默
                if not id_list:
                    overlap_note = _cmp_no_overlap_note(
                        state['radar_meta'], state['rtk_meta'],
                    )
                    if overlap_note is not None:
                        config = html.Div([overlap_note, config])
            except Exception as exc:
                logger.exception('[CMP-UPLOAD] ID发现失败')
                # 不能把任意算法异常误报成“时间范围无交集”。时间零重叠仅在
                # _cmp_no_overlap_note 已明确检测到时提示；其余情况显示真实异常。
                config = html.Div([
                    html.Div('ID 匹配计算失败，请展开错误详情确认数据字段或格式。',
                             style={'color': '#f59e0b', 'fontSize': '13px'}),
                    html.Details([
                        html.Summary('错误详情', style={
                            'cursor': 'pointer', 'fontSize': '11px',
                            'color': '#94a3b8'}),
                        html.Pre(f'{type(exc).__name__}: {exc}',
                                 style={'fontSize': '10px', 'color': '#ef4444',
                                        'whiteSpace': 'pre-wrap',
                                        'maxHeight': '160px', 'overflow': 'auto',
                                        'background': '#f8fafc', 'padding': '6px',
                                        'borderRadius': '4px'}),
                    ]),
                ])
        else:
            logger.warning('[CMP-UPLOAD] cache缺失: radar=%s rtk=%s',
                           radar_df is not None, rtk_df is not None)
            config = _cmp_config_blank()

        return (state, preview, config, radar_fb, rtk_fb,
                go.Figure(), '请选择目标ID并执行对齐', _cmp_stats_placeholder(), _cmp_bins_placeholder(),
                _perf_placeholder())
    else:
        # ═══ 等待态 → 并列展示双方各自状态 ═══
        preview = _render_cmp_waiting_preview(
            state.get('radar_meta'), state.get('rtk_meta')
        )
        config = html.Div([
            html.Div('对齐配置', className='app-card-title'),
            html.Div(f'已加载: {["雷达" if has_radar else "", "RTK" if has_rtk else ""]}  —  请上传另一个CSV文件',
                     className='stats-empty'),
        ], className='app-card')
        logger.info('[CMP-UPLOAD] → 等待态 (radar=%s rtk=%s)', has_radar, has_rtk)
        return (state, preview, config, radar_fb, rtk_fb,
                go.Figure(), '请选择目标ID并执行对齐', _cmp_stats_placeholder(), _cmp_bins_placeholder(),
                _perf_placeholder())


# ---- [C2a] 原始 CSV 物理量映射配置 ----

@callback(
    Output('cmp-preview-summary', 'children'),
    Input('cmp-state', 'data'),
)
def render_cmp_preview_summary(state):
    """在折叠标题中展示两类数据的文件数与总帧数。"""
    radar_meta = (state or {}).get('radar_meta')
    rtk_meta = (state or {}).get('rtk_meta')
    if not radar_meta and not rtk_meta:
        return '等待加载雷达与真值数据'

    def _summary(label, meta):
        if not meta:
            return f'{label}待上传'
        file_count = int(meta.get('file_count', 1))
        total_rows = int(meta.get('total_rows', 0))
        return f'{label}{file_count}份 / {total_rows:,}帧'

    return f'{_summary("雷达", radar_meta)} · {_summary("RTK", rtk_meta)}'


def _cmp_mapping_signature(state: dict | None) -> str:
    if not state:
        return ''
    parts = []
    for role in ('radar_meta', 'rtk_meta'):
        meta = state.get(role) or {}
        parts.append(str(meta.get('filename', '')))
        parts.extend(
            f'{field.get("name", "")}:{field.get("unit", "")}:{field.get("files_present", 1)}'
            for field in meta.get('physical_fields', [])
        )
    return '|'.join(parts)


def _cmp_field_lookup(state: dict | None, role: str) -> dict[str, dict]:
    meta = (state or {}).get(f'{role}_meta') or {}
    return {
        str(field.get('name')): dict(field)
        for field in meta.get('physical_fields', [])
        if field.get('name')
    }


def _cmp_default_mappings(state: dict) -> list[dict]:
    radar_fields = list(_cmp_field_lookup(state, 'radar').values())
    rtk_fields = list(_cmp_field_lookup(state, 'rtk').values())
    def _match(fields: list[dict], internal: str):
        """按 internal_name 精确匹配；加速度列大小写不定时回退到宽松比较。"""
        exact = next(
            (f for f in fields if f.get('internal_name') == internal), None)
        if exact:
            return exact
        # Ax/Ay 等自定义列在 CSV 中可能是 ax/AX/Accel_X 等写法
        target = internal.lower()
        return next(
            (f for f in fields
             if str(f.get('internal_name') or '').lower() == target
             or str(f.get('name') or '').lower() == target), None)

    defaults = []
    for radar_internal, rtk_internal in (
        ('Dx', 'center_x'), ('Dy', 'center_y'), ('Vx', 'Vx'), ('Vy', 'Vy'),
        # 加速度通道：双方 CSV 均含 Ax/Ay 列时默认纳入映射，
        # 供误差统计与性能验收直接使用，无需手动添加。
        ('Ax', 'Ax'), ('Ay', 'Ay'),
    ):
        radar = _match(radar_fields, radar_internal)
        rtk = _match(rtk_fields, rtk_internal)
        if radar and rtk:
            defaults.append({
                'uid': f'mapping-{len(defaults) + 1}',
                'radar_col': radar['name'],
                'rtk_col': rtk['name'],
            })
    if not defaults and radar_fields and rtk_fields:
        defaults.append({
            'uid': 'mapping-1',
            'radar_col': radar_fields[0]['name'],
            'rtk_col': rtk_fields[0]['name'],
        })
    return defaults


def _cmp_mapping_unit_status(radar_field: dict | None, rtk_field: dict | None) -> tuple[str, str]:
    radar_unit = str((radar_field or {}).get('unit', ''))
    rtk_unit = str((rtk_field or {}).get('unit', ''))
    if radar_unit and rtk_unit and radar_unit == rtk_unit:
        return 'compatible', radar_unit
    if radar_unit and rtk_unit and radar_unit != rtk_unit:
        return 'incompatible', f'{radar_unit} / {rtk_unit}'
    return 'unknown', radar_unit or rtk_unit


def _cmp_mapping_options(fields: dict[str, dict]) -> list[dict]:
    # 两侧都允许不选：未选字段不会进入对齐映射，也不会参与误差统计。
    return [{'label': '（不选择）', 'value': None}] + [
        {'label': name, 'value': name} for name in fields
    ]


def _cmp_resolve_mappings(mapping_store: dict | None, state: dict | None) -> list[dict]:
    radar_fields = _cmp_field_lookup(state, 'radar')
    rtk_fields = _cmp_field_lookup(state, 'rtk')
    resolved = []
    for item in (mapping_store or {}).get('items', []):
        radar_col = item.get('radar_col')
        rtk_col = item.get('rtk_col')
        has_radar = radar_col in radar_fields
        has_rtk = rtk_col in rtk_fields
        # 一侧留空时仍是有效的“单线显示”通道；只有两侧都未选或字段
        # 已不存在时才忽略。此前这里要求两侧同时存在，导致单物理量
        # 在执行对齐时被整个丢弃，图表自然没有可画的数据。
        if not has_radar and not has_rtk:
            continue
        if has_radar and has_rtk:
            status, unit = _cmp_mapping_unit_status(
                radar_fields[radar_col], rtk_fields[rtk_col])
        elif has_radar:
            status, unit = 'single', str(radar_fields[radar_col].get('unit', ''))
        else:
            status, unit = 'single', str(rtk_fields[rtk_col].get('unit', ''))
        resolved.append({
            **item,
            'radar_unit': radar_fields[radar_col].get('unit', '') if has_radar else '',
            'rtk_unit': rtk_fields[rtk_col].get('unit', '') if has_rtk else '',
            'unit': unit,
            'unit_status': status,
            'stats_enabled': status == 'compatible',
        })
    return resolved


def _append_auto_performance_mappings(
    resolved_mappings: list[dict],
    state: dict | None,
) -> list[dict]:
    """自动补建性能评估所需的 Ax/Ay 映射通道（问题：RTK 自带 Ax/Ay 列）。

    用户默认只映射 Dx/Dy/Vx/Vy，导致 aligned_df 缺少 radar[Ax]/rtk[Ax]
    等列、Ax/Ay 无法评估。此处检测双方原始字段是否均含 Ax/Ay，是则自动
    追加映射（标记 auto_injected=True）。该类通道：
      - 参与对齐生成 aligned_df 列，供性能评估使用；
      - 不进入对比图、误差统计与分距离统计（由消费方过滤）。
    已被用户显式映射的同名列不会重复注入。
    """
    radar_fields = _cmp_field_lookup(state, 'radar')
    rtk_fields = _cmp_field_lookup(state, 'rtk')
    used_pairs = {
        (str(m.get('radar_col')), str(m.get('rtk_col')))
        for m in resolved_mappings
    }
    extras: list[dict] = []
    for index, field in enumerate(('Ax', 'Ay')):
        if field not in radar_fields or field not in rtk_fields:
            continue
        if (field, field) in used_pairs:
            continue
        status, unit = _cmp_mapping_unit_status(
            radar_fields[field], rtk_fields[field])
        if status != 'compatible':
            continue
        extras.append({
            'uid': f'auto-perf-{field.lower()}',
            'radar_col': field,
            'rtk_col': field,
            'auto_injected': True,
            'radar_unit': radar_fields[field].get('unit', ''),
            'rtk_unit': rtk_fields[field].get('unit', ''),
            'unit': unit,
            'unit_status': status,
            'stats_enabled': False,
        })
    return resolved_mappings + extras


def _cmp_mapping_chart_config(mapping_results: list[dict]) -> tuple[list[str], dict]:
    selected = []
    config = {}
    for index, mapping in enumerate(mapping_results):
        # 自动注入的性能通道（Ax/Ay）仅用于评估，不绘制对比曲线
        if mapping.get('auto_injected'):
            continue
        key = str(mapping.get('uid', f'mapping-{index + 1}'))
        radar_label = str(mapping.get('radar_col', ''))
        rtk_label = str(mapping.get('rtk_col', ''))
        display_label = radar_label or rtk_label
        selected.append(key)
        config[key] = {
            'label': display_label,
            'unit': mapping.get('unit', ''),
            'chart_type': 'overlay',
            'radar_col': mapping.get('radar_output_col', ''),
            'rtk_col': mapping.get('rtk_output_col', ''),
            'rtk_curve_col': mapping.get('rtk_curve_col', ''),
            'radar_source_label': '雷达',
            'rtk_source_label': '真值',
            'rtk_label': rtk_label,
            'show_difference': bool(mapping.get('stats_enabled')),
            'radar_unit': mapping.get('radar_unit', ''),
            'rtk_unit': mapping.get('rtk_unit', ''),
        }
    return selected, config


@callback(
    Output('cmp-mappings', 'data'),
    Input('cmp-state', 'data'),
    State('cmp-mappings', 'data'),
    prevent_initial_call=True,
)
def on_cmp_mapping_sources(state, current_store):
    signature = _cmp_mapping_signature(state)
    if not signature:
        return {'signature': '', 'items': []}
    if (current_store or {}).get('signature') == signature:
        raise PreventUpdate
    return {'signature': signature, 'items': _cmp_default_mappings(state or {})}


@callback(
    Output('cmp-mapping-list', 'children'),
    Output('cmp-mapping-feedback', 'children'),
    Output('cmp-mapping-summary', 'children'),
    Input('cmp-mappings', 'data'),
    State('cmp-state', 'data'),
)
def render_cmp_mappings(mapping_store, state):
    radar_fields = _cmp_field_lookup(state, 'radar')
    rtk_fields = _cmp_field_lookup(state, 'rtk')
    if not radar_fields or not rtk_fields:
        return (
            html.Div('上传雷达与 RTK 文件后生成推荐映射', className='stats-empty'),
            '',
            '等待加载雷达与真值数据',
        )

    resolved_by_uid = {
        item['uid']: item for item in _cmp_resolve_mappings(mapping_store, state)
    }
    rows = []
    for index, item in enumerate((mapping_store or {}).get('items', [])):
        uid = item['uid']
        resolved = resolved_by_uid.get(uid, {})
        has_radar = bool(item.get('radar_col'))
        has_rtk = bool(item.get('rtk_col'))
        status = resolved.get('unit_status', 'unknown') if (has_radar or has_rtk) else 'unselected'
        status_text = {
            'compatible': f'✓ {resolved.get("unit", "")}',
            'incompatible': f'⚠ {resolved.get("unit", "单位不一致")}',
            'unknown': '？待确认',
            'single': '单线显示',
            'unselected': '— 未选择',
        }[status]
        rows.append(html.Div([
            html.Span(str(index + 1), className='mapping-channel-name'),
            html.Div([
                html.Span('雷达', className='mapping-mobile-label'),
                dcc.Dropdown(
                    id={'type': 'cmp-mapping-radar', 'index': uid},
                    options=_cmp_mapping_options(radar_fields),
                    # 不显示下拉框自带“×”清空按钮：它会遮挡窄栏中的物理量名。
                    # 需要留空时，明确选择列表中的“（不选择）”。
                    value=item.get('radar_col'), clearable=False,
                    className='mapping-dropdown',
                ),
            ], className='mapping-field mapping-radar-field',
               title=str(item.get('radar_col', ''))),
            html.Div([
                html.Span('真值', className='mapping-mobile-label'),
                dcc.Dropdown(
                    id={'type': 'cmp-mapping-rtk', 'index': uid},
                    options=_cmp_mapping_options(rtk_fields),
                    value=item.get('rtk_col'), clearable=False,
                    className='mapping-dropdown',
                ),
            ], className='mapping-field mapping-rtk-field',
               title=str(item.get('rtk_col', ''))),
            html.Span(status_text, className=f'mapping-unit-status {status}',
                      title=status_text),
            html.Div([
                html.Button('↑', id={'type': 'cmp-mapping-up', 'index': uid},
                            n_clicks=0, className='mapping-icon-btn', disabled=index == 0,
                            title='上移通道'),
                html.Button('↓', id={'type': 'cmp-mapping-down', 'index': uid},
                            n_clicks=0, className='mapping-icon-btn',
                            disabled=index == len((mapping_store or {}).get('items', [])) - 1,
                            title='下移通道'),
                html.Button('×', id={'type': 'cmp-mapping-delete', 'index': uid},
                            n_clicks=0, className='mapping-delete-btn', title='删除通道'),
            ], className='mapping-row-actions'),
        ], className='mapping-row'))

    if not rows:
        rows = [html.Div('点击“添加通道”创建物理量对比', className='stats-empty')]
    resolved = list(resolved_by_uid.values())
    compatible = sum(item.get('stats_enabled', False) for item in resolved)
    feedback = f'{len(resolved)} 个显示通道 · {compatible} 个可计算误差统计'
    summary = (
        f'{len(resolved)} 个通道 · {compatible} 个可计算误差'
        if resolved else '尚未配置显示通道'
    )
    return rows, feedback, summary


@callback(
    Output('cmp-mappings', 'data', allow_duplicate=True),
    Input('cmp-add-mapping-btn', 'n_clicks'),
    Input({'type': 'cmp-mapping-radar', 'index': ALL}, 'value'),
    Input({'type': 'cmp-mapping-rtk', 'index': ALL}, 'value'),
    Input({'type': 'cmp-mapping-delete', 'index': ALL}, 'n_clicks'),
    Input({'type': 'cmp-mapping-up', 'index': ALL}, 'n_clicks'),
    Input({'type': 'cmp-mapping-down', 'index': ALL}, 'n_clicks'),
    State('cmp-mappings', 'data'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_mapping_edit(_add, _radar_values, _rtk_values, _delete, _up, _down,
                        mapping_store, state):
    triggered = dash_ctx.triggered_id
    if triggered is None:
        raise PreventUpdate
    store = dict(mapping_store or {'signature': _cmp_mapping_signature(state), 'items': []})
    items = [dict(item) for item in store.get('items', [])]

    if triggered == 'cmp-add-mapping-btn':
        if not _add:
            raise PreventUpdate
        radar_fields = list(_cmp_field_lookup(state, 'radar'))
        rtk_fields = list(_cmp_field_lookup(state, 'rtk'))
        if not radar_fields or not rtk_fields or len(items) >= 8:
            raise PreventUpdate
        numbers = [
            int(item['uid'].rsplit('-', 1)[-1])
            for item in items if str(item.get('uid', '')).rsplit('-', 1)[-1].isdigit()
        ]
        items.append({
            'uid': f'mapping-{max(numbers, default=0) + 1}',
            'radar_col': radar_fields[0],
            'rtk_col': rtk_fields[0],
        })
    elif isinstance(triggered, dict):
        uid = triggered.get('index')
        row_index = next((i for i, item in enumerate(items) if item.get('uid') == uid), None)
        if row_index is None:
            raise PreventUpdate
        control_type = triggered.get('type')
        value = dash_ctx.triggered[0].get('value')
        if control_type == 'cmp-mapping-radar':
            if items[row_index].get('radar_col') == value:
                raise PreventUpdate
            items[row_index]['radar_col'] = value
        elif control_type == 'cmp-mapping-rtk':
            if items[row_index].get('rtk_col') == value:
                raise PreventUpdate
            items[row_index]['rtk_col'] = value
        elif control_type == 'cmp-mapping-delete' and value:
            items.pop(row_index)
        elif control_type == 'cmp-mapping-up' and value and row_index > 0:
            items[row_index - 1], items[row_index] = items[row_index], items[row_index - 1]
        elif control_type == 'cmp-mapping-down' and value and row_index < len(items) - 1:
            items[row_index + 1], items[row_index] = items[row_index], items[row_index + 1]
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

    store['items'] = items
    return store


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


# ============================================================
# 对比回调辅助函数
# ============================================================

# ---- 空白占位组件 ----

def _cmp_stats_placeholder():
    """误差统计面板空白占位。"""
    return html.Div([
        html.Div('误差统计', className='stats-card-title'),
        html.Div('执行对齐后显示', className='stats-empty'),
    ])


def _cmp_bins_placeholder():
    """分距离性能摘要面板空白占位。"""
    return html.Div('执行对齐后显示', className='stats-empty')


def _cmp_config_blank():
    """配置卡空白态（仅返回内部内容，外层app-card+title已在layout模板中）。"""
    return html.Div('请先上传两个CSV文件', className='stats-empty')


def _build_cmp_config_with_ids(id_list, filter_stats=None):
    """根据ID列表构建配置卡内部内容（不含外层app-card+title，因layout模板已提供）。"""
    id_html = _render_cmp_id_list(id_list)
    filter_summary = None
    if filter_stats:
        filter_summary = html.Div(
            (
                f'ID过滤：总计 {filter_stats.get("total_radar_targets", 0)}，'
                f'静止剔除 {filter_stats.get("filtered_static_targets", 0)}，'
                f'静止状态剔除 {filter_stats.get("filtered_motion_status_targets", 0)}，'
                f'短轨迹剔除 {filter_stats.get("filtered_short_tracks", 0)}，'
                f'有效关联 {filter_stats.get("matched_target_pairs", 0)}'
            ),
            className='feedback-muted',
            style={'fontSize': '11px', 'marginBottom': '6px'},
        )
    return html.Div([
        html.Div('目标ID',
                 style={'fontSize': '12px', 'color': '#64748b', 'marginBottom': '4px'}),
        html.Div(id='cmp-id-list', className='id-list', children=id_html),
        html.Div('时间补偿（+ 表示雷达时刻向后移）',
                 style={'fontSize': '12px', 'color': '#64748b',
                        'marginTop': '8px', 'marginBottom': '4px'}),
        dcc.Input(
            id='cmp-delay-input', type='number', value=0,
            min=-200, max=200, step=1,
            className='form-control form-control-sm',
            style={'width': '100px', 'display': 'inline-block'},
        ),
        html.Span(' ms', style={'fontSize': '12px', 'color': '#94a3b8', 'marginLeft': '4px'}),
        html.Div(id='cmp-delay-feedback', className='feedback-muted mt-1'),
        html.Div(id='cmp-coord-diag', className='feedback-muted mt-1'),
        html.Button(
            '执行对齐', id='cmp-run-btn', n_clicks=0,
            className='export-btn', style={'width': '100%', 'marginTop': '12px'},
            title='勾选段时自动对齐所选段；未勾选时对齐当前目标',
        ),
        html.Div(id='cmp-run-feedback', className='feedback-muted mt-1'),
    ])


def _cmp_preview_empty():
    """预览卡空白态。"""
    return html.Div([
        html.Div('数据预览', className='app-card-title'),
        html.Div('请先上传两个CSV文件', className='stats-empty'),
    ], className='app-card')


def _cmp_preview_error(radar_str: str, rtk_str: str):
    """预览卡渲染异常时的兜底展示。"""
    return html.Div([
        html.Div('数据预览', className='app-card-title'),
        html.Div([
            html.P('预览渲染异常，请检查上传的文件格式是否正确。',
                   style={'color': '#ef4444', 'marginBottom': '8px'}),
            html.Details([
                html.Summary('调试信息', style={'cursor': 'pointer', 'fontSize': '11px',
                                                'color': '#94a3b8'}),
                html.Pre(f'雷达meta: {radar_str[:300]}\nRTK meta: {rtk_str[:300]}',
                         style={'fontSize': '10px', 'color': '#64748b',
                                'maxHeight': '200px', 'overflow': 'auto',
                                'background': '#1e293b', 'padding': '8px',
                                'borderRadius': '4px', 'marginTop': '4px'}),
            ]),
        ], className='stats-empty'),
    ], className='app-card')


def _render_cmp_waiting_preview(radar_meta, rtk_meta):
    """等待态预览：并列展示雷达和RTK各自独立的加载状态。
    无论上传顺序如何，两文件的状态均完整呈现。
    """
    def _row(label, info):
        """单文件状态行：已加载→显示摘要，未加载→显示等待提示。"""
        if info is not None and isinstance(info, dict):
            rows = info.get('total_rows', 0)
            ids = info.get('unique_ids', 0)
            summary = f'{rows}行, {ids}个ID'
            icon = '✓'
            color = '#22c55e'
            tip = summary
        else:
            icon = '○'
            color = '#64748b'
            tip = '等待上传'
        return html.Tr([
            html.Td(label, style={'fontWeight': '600', 'color': '#334155',
                                  'whiteSpace': 'nowrap', 'paddingRight': '12px'}),
            html.Td(icon, style={'color': color, 'fontSize': '14px',
                                 'textAlign': 'center', 'width': '28px'}),
            html.Td(tip, style={'color': '#94a3b8' if info is None else '#475569',
                                'fontSize': '13px'}),
        ])

    return html.Div([
        html.Div('数据预览', className='app-card-title'),
        html.Table([
            html.Tbody([
                _row('雷达', radar_meta),
                _row('RTK',  rtk_meta),
            ]),
        ], style={'width': '100%', 'borderCollapse': 'collapse'}),
    ], className='app-card')


def _cmp_no_overlap_note(radar_meta, rtk_meta):
    """雷达与真值时间零重叠时生成醒目提示，解释“未发现目标”的根因。"""
    try:
        overlap = validate_overlap(
            {'time_range': radar_meta['time_range']},
            {'time_range': rtk_meta['time_range']},
        )
    except Exception:
        return None
    if overlap.get('has_overlap'):
        return None

    r_start, r_end = radar_meta['time_range']
    g_start, g_end = rtk_meta['time_range']
    gap = g_start - r_end if r_end <= g_start else r_start - g_end

    import datetime as _dt

    def _fmt_dt(epoch):
        return _dt.datetime.fromtimestamp(float(epoch)).strftime('%m-%d %H:%M:%S')

    gap_sec = int(abs(gap))
    hours, rem = divmod(gap_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        gap_text = f'{hours}小时{minutes}分'
    elif minutes:
        gap_text = f'{minutes}分{seconds}秒'
    else:
        gap_text = f'{seconds}秒'

    return html.Div([
        html.Div('⚠ 时间零重叠：雷达与真值不是同一段录制数据', className='feedback-error'),
        html.Div(
            f'雷达时段 {_fmt_dt(r_start)} ~ {_fmt_dt(r_end)}；'
            f'真值时段 {_fmt_dt(g_start)} ~ {_fmt_dt(g_end)}；'
            f'两段相差约 {gap_text}，无法关联目标。'
            f'请确认两侧文件来自同一次录制（录制 JSON 内含真值话题时可直接单文件上传）。',
            className='feedback-error',
        ),
    ])


def _render_cmp_preview(radar_info, rtk_info):
    """渲染数据预览卡（双方就绪时使用）。"""
    def _fmt_ts(epoch):
        if epoch == 0:
            return '—'
        import datetime as _dt
        return _dt.datetime.fromtimestamp(epoch).strftime('%H:%M:%S')

    def _sample_rate_cell(info):
        rate = float(info.get('sample_rate_hz') or 0.0)
        label = str(info.get('sample_rate_label') or f'{rate:.1f}Hz')
        note = str(info.get('sample_rate_note') or '')
        children = [label]
        if note:
            children.append(html.Span(' ⚠', style={'color': '#b45309'}))
        return html.Span(children, title=note or label)

    return html.Div([
        html.Div('数据预览', className='app-card-title'),
        html.Table([
            html.Thead(html.Tr([
                html.Th(''), html.Th('雷达'), html.Th('RTK'),
            ])),
            html.Tbody([
                html.Tr([html.Td('行数'), html.Td(str(radar_info['total_rows'])),
                         html.Td(str(rtk_info['total_rows']))]),
                html.Tr([html.Td('ID数'), html.Td(str(radar_info['unique_ids'])),
                         html.Td(str(rtk_info['unique_ids']))]),
                html.Tr([html.Td('时间帧率'), html.Td(_sample_rate_cell(radar_info)),
                         html.Td(_sample_rate_cell(rtk_info))]),
                html.Tr([html.Td('物理量'),
                         html.Td(str(len(radar_info.get('physical_fields', [])))),
                         html.Td(str(len(rtk_info.get('physical_fields', []))))]),
                html.Tr([html.Td('起始'), html.Td(_fmt_ts(radar_info['time_range'][0])),
                         html.Td(_fmt_ts(rtk_info['time_range'][0]))]),
                html.Tr([html.Td('结束'), html.Td(_fmt_ts(radar_info['time_range'][1])),
                         html.Td(_fmt_ts(rtk_info['time_range'][1]))]),
            ]),
        ], className='traj-table'),
    ], className='app-card')


def _cmp_candidate_display_key(candidate: dict) -> tuple:
    """候选只在界面中按真实文件名分组，不改变后台质量排序。"""
    filename = str(candidate.get('radar_filename') or '').strip()
    file_index = candidate.get('file_index')
    return (
        0 if filename else 1,
        natural_filename_key(filename),
        int(file_index) if file_index is not None else -1,
        float(candidate.get('time_start') or 0),
        int(candidate.get('track_id') or 0),
        int(candidate.get('segment_index') or 0),
    )


def _render_cmp_id_list(id_list, selected_id=None):
    """按真实来源文件分组渲染候选，点击身份仍使用文件索引与生命周期段。"""
    if not id_list:
        return [html.Div('未发现目标ID', className='id-list-item',
                         style={'cursor': 'default', 'color': '#94a3b8'})]

    display_candidates = sorted(
        (candidate for candidate in id_list if candidate.get('total_frames', 0) > 0),
        key=_cmp_candidate_display_key,
    )
    group_counts = {}
    for candidate in display_candidates:
        group_key = (
            candidate.get('file_index'),
            str(candidate.get('radar_filename') or '').strip(),
        )
        group_counts[group_key] = group_counts.get(group_key, 0) + 1

    items = []
    current_group = None
    for r in display_candidates:
        cls = 'id-list-item cmp-id-item'
        tid = int(r['track_id'])  # 转Python原生int，Dash dict id不接受np.int64
        fi = r.get('file_index')
        radar_filename = str(r.get('radar_filename') or '').strip()
        group_key = (fi, radar_filename)

        if group_key != current_group:
            source = identify_radar_source(radar_filename)
            source_key = str(source.get('key', 'unknown'))
            source_label = str(source.get('short_label', '未识别'))
            fallback_name = f'上传文件 {fi + 1}' if fi is not None else '来源文件未知'
            short_name = compact_filename_label(radar_filename) if radar_filename else fallback_name
            items.append(html.Div([
                html.Span(
                    source_label,
                    className=f'radar-source-badge radar-source-{source_key}',
                ),
                html.Span(short_name, className='cmp-file-short-name'),
                html.Span(
                    f'序号{fi + 1}' if fi is not None else '',
                    className='cmp-file-upload-order',
                ),
                html.Span(
                    f'{group_counts[group_key]}条轨迹',
                    className='cmp-file-track-count',
                ),
            ], className='cmp-file-group-header', title=radar_filename or fallback_name))
            current_group = group_key

        # 构建复合索引：同一文件内 Track_Age 重置后也必须区分轨迹段。
        segment_index = r.get('segment_index')
        if segment_index is not None:
            # 单文件同 ID 同样可能有多个已确认生命周期段；必须携带段号，
            # 否则点击列表项会退化为只选中该 ID 的第一段。
            file_token = str(fi) if fi is not None else 'none'
            composite_index = f'{tid}:::{file_token}:::{segment_index}'
            id_label = f'ID={tid} · 段{segment_index}'
        elif fi is not None:
            composite_index = f'{tid}:::{fi}'
            id_label = f'ID={tid}'
        else:
            composite_index = str(tid)
            id_label = f'ID={tid}'

        if composite_index == selected_id:
            cls += ' selected'
        rate_pct = int(r['overlap_rate'] * 100)

        # 第一行：ID、覆盖率、帧数和中断概况。
        row1 = html.Div([
            html.Span(id_label, style={'fontWeight': '600'}),
            html.Span(' · ', style={'color': '#94a3b8', 'margin': '0 2px'}),
            html.Span(f'{rate_pct}%', className='id-meta'),
            html.Span(' · ', style={'color': '#94a3b8', 'margin': '0 2px'}),
            html.Span(f'{r["matched_frames"]}/{r["total_frames"]}帧', className='id-meta'),
            *(
                [
                    html.Span(' · ', style={'color': '#94a3b8', 'margin': '0 2px'}),
                    html.Span(f'{r["gap_count"]}处中断', className='id-meta'),
                ] if r.get('gap_count', 0) else []
            ),
        ], className='cmp-id-row1')

        # 第二行同时标注雷达时间范围和真实 RTK 来源文件。
        row2_text = _format_cmp_time_range(r.get('time_start_str'), r.get('time_end_str'))
        rtk_filename = str(r.get('rtk_filename') or '').strip()
        rtk_label = compact_filename_label(rtk_filename) if rtk_filename else 'RTK文件未知'
        rtk_text = (
            f'RTK: {rtk_label} / ID={r["rtk_id"]}'
            if r.get('rtk_id') is not None else f'RTK: {rtk_label}'
        )
        row2 = html.Div([
            html.Span(row2_text, className='cmp-id-time-range'),
            html.Span(rtk_text, className='cmp-id-rtk-source', title=rtk_filename or rtk_label),
        ], className='cmp-id-row2')

        items.append(html.Div(
            className=cls,
            children=[
                dcc.Checklist(
                    id={'type': 'cmp-seg-check', 'index': composite_index},
                    options=[{'label': '', 'value': '1'}],
                    value=[],
                    className='cmp-seg-check',
                    inputClassName='cmp-seg-check-input',
                ),
                html.Div(
                    id={'type': 'cmp-id-item', 'index': composite_index},
                    n_clicks=0,
                    className='cmp-id-click-target',
                    children=[row1, row2],
                ),
            ],
        ))
    return items


def _parse_cmp_index(index_str: str) -> tuple:
    """解析对比模块的复合索引字符串。

    格式:
      - 单文件: "5"              → (5, None, None)
      - 多文件: "5:::0"           → (5, 0, None)
      - 生命周期段: "5:::0:::2"   → (5, 0, 2)
      - 单文件生命周期段: "5:::none:::2" → (5, None, 2)

    Returns:
        (track_id: int, file_index: int | None, segment_index: int | None)
    """
    if ':::' in index_str:
        parts = index_str.split(':::')
        if len(parts) == 3:
            file_index = None if parts[1] == 'none' else int(parts[1])
            return int(parts[0]), file_index, int(parts[2])
        return int(parts[0]), int(parts[1]), None
    return int(index_str), None, None


def _format_cmp_time_range(start_str, end_str):
    """将原始时间戳字符串 YYYY_MM_DD_HH_MM_SS_mmm 格式化为 DD/HH/MM/mmm。

    Returns:
        str: 如 '04/10/18/507-04/10/18/507' 或 '—'（无数据时）
    """
    if not start_str or not end_str:
        return '—'

    def _fmt_one(ts):
        try:
            parts = ts.replace('-', '_').replace(':', '_').replace(' ', '_').split('_')
            # 期望格式 YYYY_MM_DD_HH_MM_SS_mmm (7段) 或 YYYY_MM_DD_HH_MM_SS (6段)
            if len(parts) >= 5:
                day = parts[2]     # DD
                hour = parts[3]    # HH
                minute = parts[4]  # MM
                ms = parts[6] if len(parts) >= 7 else '000'
                return f'{day}/{hour}/{minute}/{ms}'
        except Exception:
            pass
        return ts

    s = _fmt_one(start_str)
    e = _fmt_one(end_str)
    if s == e:
        return s
    return f'{s}-{e}'
