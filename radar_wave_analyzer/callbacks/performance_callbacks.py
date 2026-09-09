"""性能验收回调：执行对齐 → 图表+统计+验收表 → 摘要快照 → 导出工作簿。

回调清单：
  [C4] update_selected_segments / on_cmp_run
                           → 段勾选变更；执行对齐 → 图表 + 统计 + 性能验收
  [C4a] update_perf_summary_snapshots / render_perf_summary_snapshots
                           → 分距离性能摘要快照
  [C4b] on_perf_metric_switch → 性能验收指标切换
  [C4c] on_perf_collapse   → 性能验收折叠切换
  [C6] on_cmp_export_workbook → 导出对齐+逐帧性能+验收汇总 xlsx 工作簿
"""
import logging
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update, html, dcc, ALL
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..config import get

from ..cache import (
    get_comparison_data,
    set_alignment_result, get_alignment_result,
    set_performance_result, get_performance_result,
)

from ..components.comparison_stats_panel import (
    render_cmp_error_stats, render_cmp_error_stats_empty,
)

from ..components.comparison_charts import build_comparison_subplots

from ..comparison.performance import evaluate_all_metrics

from ..components.performance_panel import (
    render_performance_table, render_performance_body,
)
from ..components.performance_summary import render_performance_snapshot_summary

from ..comparison.service import (
    execute_alignment, execute_selected_segments_alignment, resolve_track_selection,
)

from ..comparison.exporter import (
    derive_quantity_pairs,
    export_comparison_workbook_bytes,
)

from .helpers import _perf_placeholder, _summary_plain_text
from .cmp_mapping_callbacks import (
    _cmp_mapping_chart_config,
    _cmp_resolve_mappings,
    _append_auto_performance_mappings,
)
from .cmp_preview_render import _cmp_bins_placeholder

logger = logging.getLogger(__name__)


# ---- [C4] 执行对齐 ----


def _build_perf_panel(perf_results: dict):
    """根据评估结果构建验收表面板。

    仅 available 的指标进入单选框（文档 10.2：只有映射成功、单位兼容
    且具有有效真值的物理量可选择）。
    """
    perf_cfg = get('performance_metrics', {}) or {}
    rules = perf_cfg.get('metrics') or {}
    available = [m for m, r in perf_results.items() if r.get('available')]
    options = [
        {'label': str(rules.get(m, {}).get('label', m)), 'value': m}
        for m in available
    ]
    current = available[0] if available else None
    unit = str(rules.get(current, {}).get('unit', '')) if current else ''
    return render_performance_table(
        perf_results.get(current), current or '', unit, options,
        metric_rules=rules.get(current),
    )


def _perf_summary_source_key(state: dict | None) -> str:
    """返回摘要快照所属数据源的稳定标识；不同上传批次不混合追加。"""
    state = state or {}
    def _part(role: str) -> str:
        meta = state.get(f'{role}_meta') or {}
        return '|'.join([
            str(meta.get('filename') or ''),
            str(meta.get('total_rows') or ''),
            str(meta.get('time_range') or ''),
        ])
    return f'{_part("radar")}::{_part("rtk")}'


def _perf_summary_snapshot_label(state: dict) -> str:
    """构建可复制摘要中每次插入结果的来源说明。"""
    radar_name = str((state.get('radar_meta') or {}).get('filename') or '雷达数据')
    rtk_name = str((state.get('rtk_meta') or {}).get('filename') or 'RTK数据')
    target = state.get('selected_id')
    segment = state.get('selected_segment_index')
    merged_count = state.get('merged_segments')
    delay = state.get('delay_ms') or 0
    target_text = f'ID={target}' if target is not None else 'ID=—'
    if merged_count:
        segment_text = f'合并{merged_count}段'
    else:
        segment_text = f'段{segment + 1}' if isinstance(segment, int) else '段—'
    return f'{radar_name} · {rtk_name} · {target_text} · {segment_text} · 补偿 {delay}ms'


@callback(
    Output('cmp-selected-segments', 'data'),
    Input({'type': 'cmp-seg-check', 'index': ALL}, 'value'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def update_selected_segments(check_values, state):
    """收集勾选的中断段，供「对齐所选段」合并对齐。"""
    candidates = (state or {}).get('cached_id_list') or []
    segments: list[dict] = []
    # inputs_list 对单个 ALL 模式 Input 是扁平的 trigger 字典列表；
    # 多个 Input 声明时才按声明分组嵌套。统一拍平后再与 values 一一配对。
    triggers = getattr(dash_ctx, 'inputs_list', []) or []
    checkbox_triggers: list[dict] = []
    for entry in triggers:
        if isinstance(entry, list):
            checkbox_triggers.extend(entry)
        else:
            checkbox_triggers.append(entry)
    for trigger, checked in zip(checkbox_triggers, check_values):
        if not checked:
            continue
        index_key = str((trigger.get('id') or {}).get('index', ''))
        candidate = _parse_segment_key(index_key, candidates)
        if candidate is None:
            continue
        segments.append({
            # 保存完整候选信息，后续合并不再依赖 cmp-state 中可能被刷新过的
            # cached_id_list 重新解析复合键。
            'candidate': dict(candidate),
            'track_id': candidate.get('track_id'),
            'file_index': candidate.get('file_index'),
            'segment_index': candidate.get('segment_index'),
            'key': index_key,
        })
    return {'segments': segments}


@callback(
    Output('cmp-graph', 'figure'),
    Output('cmp-graph-title', 'children'),
    Output('cmp-stats-content', 'children'),
    Output('cmp-bins-content', 'children'),
    Output('perf-panel-container', 'children'),
    Output('cmp-state', 'data', allow_duplicate=True),
    Output('cmp-run-feedback', 'children', allow_duplicate=True),
    Input('cmp-run-btn', 'n_clicks'),
    State('cmp-selected-segments', 'data'),
    State('cmp-state', 'data'),
    State('cmp-delay-input', 'value'),
    State('cmp-mappings', 'data'),
    State('perf-summary-snapshots', 'data'),
    prevent_initial_call=True,
)
def on_cmp_run(_n, selected_segments, state, delay_ms,
               mapping_store, snapshots_store):
    """一键执行对齐 + 展示全部结果。

    未勾选段时对当前单段执行对齐；勾选一个或多个段时自动合并所选段对齐。
    """
    if not _n or not state:
        raise PreventUpdate

    snapshot_items = (snapshots_store or {}).get('items') or []
    keep_snapshot = bool(snapshot_items) and (
        (snapshots_store or {}).get('source_key') == _perf_summary_source_key(state)
    )

    has_radar = state.get('radar_meta') is not None
    has_rtk = state.get('rtk_meta') is not None
    if not has_radar or not has_rtk:
        raise PreventUpdate

    radar_df, rtk_df = get_comparison_data('default')
    if radar_df is None or rtk_df is None:
        raise PreventUpdate

    cmp_cfg = get('comparison', {})

    selected_segment_items = list((selected_segments or {}).get('segments') or [])
    merge_mode = bool(selected_segment_items)

    try:
        selection = resolve_track_selection(
            radar_df, rtk_df, cmp_cfg,
            state.get('selected_id'), state.get('selected_file_index'),
            state.get('selected_segment_index'), state.get('cached_id_list'),
        )
    except ValueError:
        raise PreventUpdate
    track_id = selection['track_id']
    file_index = selection['file_index']
    state['cached_id_list'] = selection['candidate_ids']

    # ── 合并中断段模式：以勾选段为准构造对齐目标 ──
    merged_count = 0
    if merge_mode:
        candidates = state.get('cached_id_list') or []
        matched = []
        for item in selected_segment_items:
            # 新格式由勾选回调直接保存候选快照；旧 Store 仍可退回复合键解析。
            if isinstance(item, dict) and isinstance(item.get('candidate'), dict):
                matched.append(dict(item['candidate']))
                continue
            key = item.get('key') if isinstance(item, dict) else item
            candidate = _parse_segment_key(str(key), candidates)
            if candidate is not None:
                matched.append(candidate)
        if not matched:
            return (
                no_update, no_update, no_update, no_update, no_update,
                no_update,
                '合并失败：勾选的段已失效，请重新勾选后再试',
            )
        track_ids = {str(c.get('track_id')) for c in matched}
        rtk_ids = {c.get('rtk_id') for c in matched}
        if len(rtk_ids) > 1:
            return (
                no_update, no_update, no_update, no_update, no_update,
                no_update,
                '合并失败：勾选的段必须关联到同一个真值目标',
            )
        merged_count = len(matched)
        first = matched[0]
        selection = {
            'track_id': int(first['track_id']),
            'file_index': first.get('file_index'),
            'segment_index': None,          # 合并模式：忽略段号
            'rtk_id': first.get('rtk_id'),
            'rtk_file_index': first.get('rtk_file_index'),
            'candidate_ids': state.get('cached_id_list') or [],
        }
        # 合并模式必须沿用勾选项关联到的同一真值目标，并且仅合并勾选段。
        selection['rtk_id'] = first.get('rtk_id')
        selection['segment_indexes'] = sorted(
            {int(item['segment_index']) for item in matched}
        )
        state['selected_id'] = selection['track_id']
        state['selected_file_index'] = selection['file_index']
        state['selected_segment_index'] = None

    resolved_mappings = _cmp_resolve_mappings(mapping_store, state)
    if not resolved_mappings:
        raise PreventUpdate
    # 自动补建 Ax/Ay 性能通道（RTK 自带该列时），仅参与对齐生成评估列
    alignment_mappings = _append_auto_performance_mappings(resolved_mappings, state)

    try:
        if merge_mode:
            result = execute_selected_segments_alignment(
                radar_df, rtk_df, cmp_cfg, matched,
                delay_ms=delay_ms or 0,
                custom_mappings=alignment_mappings,
            )
        else:
            result = execute_alignment(
                radar_df, rtk_df, cmp_cfg, track_id,
                delay_ms=delay_ms or 0,
                file_index=file_index,
                segment_index=selection['segment_index'],
                rtk_id=selection['rtk_id'],
                rtk_file_index=selection['rtk_file_index'],
                custom_mappings=alignment_mappings,
            )
    except Exception as e:
        empty_fig = go.Figure()
        state['alignment_done'] = False
        state['selected_id'] = track_id
        perf_panel = _perf_placeholder(f'错误: {e}')
        return (
            empty_fig, '对齐失败',
            html.Div([html.Div('误差统计', className='stats-card-title'),
                      html.Div(f'错误: {e}', className='stats-empty')], className='stats-card'),
            no_update if keep_snapshot else html.Div(
                f'错误: {e}', className='stats-empty'),
            perf_panel,
            state,
            f'对齐失败: {e}',
        )

    aligned_df = result['aligned_df']
    summary = result['summary']
    match_summary = result['match_summary']
    mapping_results = result.get('mapping_results', [])

    if len(aligned_df) == 0:
        empty_fig = go.Figure()
        state['alignment_done'] = False
        state['selected_id'] = track_id
        perf_panel = _perf_placeholder('无匹配数据')
        return (
            empty_fig, '无匹配数据',
            render_cmp_error_stats_empty(), no_update if keep_snapshot else _cmp_bins_placeholder(),
            perf_panel,
            state, '无匹配帧',
        )

    if not mapping_results:
        state['alignment_done'] = False
        perf_panel = _perf_placeholder('映射字段不可用')
        return (
            go.Figure(), '映射字段不可用',
            render_cmp_error_stats_empty(), no_update if keep_snapshot else _cmp_bins_placeholder(),
            perf_panel,
            state, '当前目标所在文件不包含所选映射字段，请更换物理量后重试',
        )

    # 缓存结果
    set_alignment_result('default', aligned_df, summary, result.get('rtk_curve_df'))

    # 性能指标评估：一次对齐后基于完整 aligned_df 统一计算全部物理量并缓存
    # （文档 12.3/15），切换指标仅读取缓存，不重新执行对齐。
    try:
        perf_results = evaluate_all_metrics(
            aligned_df,
            track_id=track_id,
            segment_id=selection.get('segment_index'),
        )
        set_performance_result('default', perf_results)
        perf_panel = _build_perf_panel(perf_results)
    except Exception as perf_err:  # noqa: BLE001 评估异常不阻断对齐展示
        import traceback
        perf_trace = traceback.format_exc()
        logger.warning('性能指标评估失败: %s\n%s', perf_err, perf_trace)
        perf_results = None
        perf_panel = _perf_placeholder(f'性能评估失败: {perf_err}')

    # 构建图表：物理量名称使用 CSV 原始表头。
    selected_qties, cmp_qties = _cmp_mapping_chart_config(mapping_results)
    fig = build_comparison_subplots(
        aligned_df, selected_qties, cmp_qties,
        trajectory_label=f'ID={track_id}',
        rtk_curve_df=result.get('rtk_curve_df'),
        perf_results=perf_results,
    )

    # 标题（多文件时附加文件序号；合并模式附加合并段数）
    if merge_mode:
        title_id = f'ID={track_id} [合并{merged_count}段]'
    elif file_index is not None:
        title_id = f'ID={track_id} [文件{file_index + 1}]'
    else:
        title_id = f'ID={track_id}'
    title = [
        html.Span(title_id, className='traj-id-badge'),
        html.Span(' | ', style={'color': '#94a3b8'}),
        html.Span(
            f"{match_summary['matched_frames']}/{match_summary['total_frames']}帧匹配",
            className='frame-count',
        ),
        html.Span(' | ', style={'color': '#94a3b8'}),
        html.Span(
            f"ΔDist RMSE={summary['pos_error_abs']['rmse']}m",
            className='frame-count',
        ),
    ]

    # 统计面板
    stats_html = render_cmp_error_stats(
        mapping_results, summary=summary, match_summary=match_summary,
    )

    # 右栏摘要仅展示用户手动插入的快照；普通执行对齐不得覆盖或预览它。
    bins_html = no_update

    # 更新状态（合并模式记录段数供导出命名；单段模式清除该标记）
    state['alignment_done'] = True
    state['selected_id'] = track_id
    state['selected_file_index'] = file_index
    state['selected_segment_index'] = selection['segment_index']
    if merge_mode:
        state['merged_segments'] = merged_count
    else:
        state.pop('merged_segments', None)
    state['selected_rtk_id'] = selection['rtk_id']
    state['delay_ms'] = delay_ms or 0
    state['resolved_mappings'] = mapping_results
    # 性能评估的轻量汇总（不含 frames 大表）随 state 走客户端 Store，
    # 指标切换直接读 state，彻底避免服务端缓存偶发丢失导致切换无响应。
    state['perf_summaries'] = {
        metric: {k: r.get(k) for k in
                 ('available', 'reason', 'mode', 'bins',
                  'overall', 'coverage', 'abs_limit', 'operator')}
        for metric, r in (perf_results or {}).items() if isinstance(r, dict)
    }

    skipped_mappings = len(alignment_mappings) - len(mapping_results)
    skipped_text = f' · {skipped_mappings}个通道在当前文件不可用' if skipped_mappings else ''
    merge_text = f'已合并 {merged_count} 个所选段 · ' if merge_mode else ''
    diagnostic_text = (
        f' · 时间有效RMSE={summary.get("pos_error_abs_time_valid", {}).get("rmse")}m'
        f' · 空间拒绝={match_summary.get("spatial_rejected_frames", 0)}帧'
        f' · 时间拒绝={match_summary.get("time_rejected_frames", 0)}帧'
        f' · RTK范围外={match_summary.get("out_of_rtk_range_frames", 0)}帧'
    )
    return (
        fig, title, stats_html, bins_html,
        perf_panel,
        state,
        f'✓ {merge_text}对齐完成 — {match_summary["matched_frames"]}/{match_summary["total_frames"]}帧匹配'
        f'{diagnostic_text}{skipped_text}',
    )


# ---- [C4a] 分距离性能摘要快照 ----

@callback(
    Output('perf-summary-snapshots', 'data'),
    Output('perf-summary-feedback', 'children'),
    Input('perf-summary-insert-btn', 'n_clicks'),
    Input('perf-summary-remove-btn', 'n_clicks'),
    Input('perf-summary-clear-btn', 'n_clicks'),
    State('cmp-state', 'data'),
    State('perf-summary-snapshots', 'data'),
    prevent_initial_call=True,
)
def update_perf_summary_snapshots(_insert, _remove, _clear, state, snapshot_store):
    """仅由用户操作修改已插入摘要，普通执行对齐不会覆盖快照。"""
    triggered = dash_ctx.triggered[0]['prop_id'].split('.')[0] if dash_ctx.triggered else ''
    store = dict(snapshot_store or {})
    items = list(store.get('items') or [])

    if triggered == 'perf-summary-clear-btn':
        return {'source_key': '', 'items': []}, '已清空已插入摘要'
    if triggered == 'perf-summary-remove-btn':
        if not items:
            return no_update, '没有可删除的摘要'
        items.pop()
        return {
            'source_key': store.get('source_key') or '',
            'items': items,
        }, '已删除最后一次插入的数据'
    if triggered != 'perf-summary-insert-btn':
        raise PreventUpdate

    summaries = (state or {}).get('perf_summaries') or {}
    if not state or not state.get('alignment_done') or not summaries:
        return no_update, '请先执行对齐，再插入当前数据'

    source_key = _perf_summary_source_key(state)
    if store.get('source_key') != source_key:
        # 重新上传了数据源：新数据从第一个快照开始，不与旧上传批次混合。
        items = []
    items.append({
        'label': _perf_summary_snapshot_label(state),
        'results': summaries,
    })
    return {
        'source_key': source_key,
        'items': items,
    }, f'已插入第 {len(items)} 次摘要'


@callback(
    Output('cmp-bins-content', 'children', allow_duplicate=True),
    Output('perf-summary-copy', 'content'),
    Input('perf-summary-snapshots', 'data'),
    prevent_initial_call=True,
)
def render_perf_summary_snapshots(snapshot_store):
    """快照更新时才重绘已插入摘要，避免执行对齐覆盖历史数据。

    同时生成剪贴板纯文本：只含数据行，不含 [1]/[2] 快照来源元信息。
    """
    items = (snapshot_store or {}).get('items') or []
    rules = (get('performance_metrics', {}) or {}).get('metrics') or {}
    summary_html = render_performance_snapshot_summary(items, rules)
    return summary_html, _summary_plain_text(summary_html)


# ---- [C4b] 性能验收：指标切换 ----

@callback(
    Output('perf-table-body', 'children'),
    Input('perf-metric-selector', 'value'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_perf_metric_switch(metric, state):
    """切换验收物理量：优先读 state 内轻量汇总，不重新对齐（文档 12.3）。

    state 中的汇总随对齐回调一起刷新（客户端 Store，绝不丢失）；
    服务端缓存（含逐帧 frames）仅作为 fallback，供导出等使用。
    """
    if not metric:
        raise PreventUpdate
    rules = (get('performance_metrics', {}) or {}).get('metrics') or {}
    unit = str(rules.get(metric, {}).get('unit', ''))

    summaries = (state or {}).get('perf_summaries') or {}
    result = summaries.get(metric)
    if result is None:
        cached = get_performance_result('default') or {}
        result = cached.get(metric)
    if not result:
        raise PreventUpdate

    return render_performance_body(result, unit, rules.get(metric))


# ---- [C4c] 性能验收：折叠切换 ----

@callback(
    Output('perf-table-body', 'className'),
    Output('perf-panel-container', 'className'),
    Output('perf-collapse-arrow', 'children'),
    Input('perf-collapse-toggle', 'n_clicks'),
    State('perf-table-body', 'className'),
    State('perf-panel-container', 'className'),
    prevent_initial_call=True,
)
def on_perf_collapse(_n, body_cls, container_cls):
    """切换验收表折叠态；容器 class 联动 CSS 恢复图表全高。"""
    collapsed = 'perf-body-collapsed' in (body_cls or '')
    if collapsed:
        # 展开
        return 'perf-table-body', 'perf-panel-container', '▾'
    # 折叠
    return ('perf-table-body perf-body-collapsed',
            'perf-panel-container perf-collapsed', '▸')


# ---- [C6] 导出 CSV（真值对齐 + 性能验收工作簿） ----

@callback(
    Output('cmp-export-download', 'data'),
    Output('cmp-export-feedback', 'children'),
    Input('cmp-export-csv-btn', 'n_clicks'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_export_workbook(_n, state):
    """导出最近一次执行对齐的数据为 xlsx 工作簿（浏览器下载）。

    默认文件名 = 数据来源的雷达 CSV 文件名（去扩展名）+ 对齐 ID，
    如 ``CD701_flr_track_2026_08_11_15_57_24_ID3.xlsx``。下载动作由
    浏览器处理：若浏览器开启“每次下载前询问保存位置”即可自选路径，
    且不占用服务端文件，Excel 打开旧文件也不影响再次导出。
    """
    if not _n or not state or not state.get('alignment_done'):
        raise PreventUpdate

    aligned_df, _summary = get_alignment_result('default')
    if aligned_df is None or len(aligned_df) == 0:
        return no_update, '无数据可导出，请先执行对齐'

    # 从 aligned_df 推导雷达/真值列配对（error[雷达-真值] 列编码了映射
    # 关系，真值字段名与雷达字段名不同时同样能正确配对）
    quantities = {}
    for label, (radar_col, rtk_col) in derive_quantity_pairs(aligned_df).items():
        quantities[label] = {
            'label': label,
            'unit': '',
            'radar_col': radar_col,
            'rtk_col': rtk_col,
        }

    # 默认命名：选中段来源的雷达 CSV 文件名 + 对齐 ID [+ 合并标记] + 导出
    # 时间戳（同 ID 反复导出不互相覆盖）；非法文件名字符替换为下划线。
    radar_name = _selected_radar_filename(state)
    stem = os.path.splitext(os.path.basename(radar_name))[0]
    target_id = state.get('selected_id')
    id_text = f'_ID{target_id}' if target_id is not None else ''
    merged_count = state.get('merged_segments')
    merged_text = f'_merged{merged_count}段' if merged_count else ''
    stamp = pd.Timestamp.now().strftime('%m%d_%H%M')
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, '_')
    default_name = f'{stem}{id_text}{merged_text}_{stamp}.xlsx'

    perf_results = get_performance_result('default') or {}
    perf_cfg = get('performance_metrics', {}) or {}
    try:
        content = export_comparison_workbook_bytes(
            aligned_df, quantities, perf_results, rules=perf_cfg)
    except Exception as e:
        logger.exception('生成工作簿失败')
        return no_update, f'导出失败: {e}'

    data = dcc.send_bytes(
        lambda buf: buf.write(content), filename=default_name)
    return data, f'已生成 {default_name}，浏览器正在下载'


def _selected_radar_filename(state: dict) -> str:
    """返回选中轨迹段来源的雷达 CSV 文件名。

    优先从 ID 候选缓存（cached_id_list）按 file_index + track_id +
    segment_index 匹配出实际来源文件名；多文件上传时这是"数据所在的
    雷达 csv 文件名"的精确值。无记录时回退上传批次标签（radar_meta）。
    """
    fallback = str((state.get('radar_meta') or {}).get('filename') or '雷达数据')
    file_index = state.get('selected_file_index')
    track_id = state.get('selected_id')
    seg_idx = state.get('selected_segment_index')
    if file_index is None:
        return fallback
    for candidate in state.get('cached_id_list') or []:
        if candidate.get('file_index') != file_index:
            continue
        if str(candidate.get('track_id')) != str(track_id):
            continue
        cand_seg = candidate.get('segment_index')
        if (seg_idx is not None and cand_seg is not None
                and int(cand_seg) != int(seg_idx)):
            continue
        return str(candidate.get('radar_filename') or fallback)
    return fallback


def _parse_segment_key(key: str, candidates: list[dict]) -> dict | None:
    """解析段键 'track:::file:::seg' 并在候选列表中匹配完整候选记录。"""
    parts = str(key).split(':::')
    if len(parts) != 3:
        return None
    track_id, file_token, seg_token = parts
    for candidate in candidates:
        cand_seg = candidate.get('segment_index')
        if (str(candidate.get('track_id')) == track_id
                and str(candidate.get('file_index')) == file_token
                and (cand_seg is None or str(cand_seg) == seg_token)):
            return candidate
    return None
