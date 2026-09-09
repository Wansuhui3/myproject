from dash import Input, html, dcc
from ..config import get
from ..core.data_loader import identify_radar_source
from ..comparison.parser import validate_overlap
from ..comparison.file_identity import compact_filename_label, natural_filename_key
"""真值对比纯渲染辅助：占位组件、预览卡片、ID 列表、配置表单。

不得定义回调，不得导入任何回调模块。
"""



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
