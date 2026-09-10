import logging

import pandas as pd
import plotly.graph_objects as go
from dash import (
    Input,
    Output,
    State,
    callback,
    html,
    no_update,
)
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..cache import get_comparison_data, set_alignment_result, set_comparison_data
from ..comparison.service import get_candidate_match_result, prepare_comparison_upload
from ..config import get
from .cmp_preview_render import (
    _build_cmp_config_with_ids,
    _cmp_bins_placeholder,
    _cmp_config_blank,
    _cmp_no_overlap_note,
    _cmp_preview_error,
    _cmp_stats_placeholder,
    _render_cmp_preview,
    _render_cmp_waiting_preview,
)
from .helpers import _decode_upload_contents, _perf_placeholder

logger = logging.getLogger(__name__)
"""[C2] 统一上传：解析 + 缓存 + 预览 + ID 发现（单回调，无轮询）。"""


def _render_upload_feedback(role_label: str, meta: dict, prefix: str = '') -> html.Div:
    """上传成功反馈：与波动页同款结构（结果文本行）。"""
    tr = meta.get('time_range') or [0.0, 0.0]
    time_text = ''
    try:
        t0 = pd.Timestamp(float(tr[0]), unit='s')
        t1 = pd.Timestamp(float(tr[1]), unit='s')
        time_text = f' | {t0:%Y-%m-%d %H:%M:%S} ~ {t1:%Y-%m-%d %H:%M:%S}'
    except (TypeError, ValueError):
        pass
    rate = str(meta.get('sample_rate_label') or '')
    rate_text = f' | {rate}' if rate else ''
    text = (f'已加载: {meta.get("filename", "")} | '
            f'{meta.get("total_rows", 0)}行, {meta.get("unique_ids", 0)}个ID'
            f'{time_text}{rate_text}')
    return html.Div(text, className='upload-result-text')



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
        role_label = '雷达' if role == 'radar' else 'RTK真值'
        ok_fb = _render_upload_feedback(role_label, meta)
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
        routed_label = '雷达' if routed_role == 'radar' else 'RTK真值'
        routed_fb = _render_upload_feedback(routed_label, routed_meta,
                                            prefix='自动归类 · ')
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
