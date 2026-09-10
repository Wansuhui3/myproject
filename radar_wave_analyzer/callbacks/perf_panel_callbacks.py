"""性能验收面板域回调 [C4a/b/c]：摘要快照 → 指标切换 → 折叠切换。

执行对齐见 perf_run_callbacks；导出工作簿见 perf_export_callbacks。
"""
import logging

from dash import Input, Output, State, callback, no_update
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..cache import get_performance_result
from ..components.performance_panel import render_performance_body
from ..components.performance_summary import render_performance_snapshot_summary
from ..config import get
from .helpers import _summary_plain_text
from .perf_shared import _perf_summary_snapshot_label, _perf_summary_source_key

logger = logging.getLogger(__name__)


# ---- [C4a] 分距离性能摘要快照 ----

@callback(
    Output('perf-summary-snapshots', 'data'),
    Output('perf-summary-feedback', 'children'),
    Output('cmp-bins-content', 'children', allow_duplicate=True),
    Output('perf-summary-copy', 'content'),
    Input('perf-summary-insert-btn', 'n_clicks'),
    Input('perf-summary-remove-btn', 'n_clicks'),
    Input('perf-summary-clear-btn', 'n_clicks'),
    State('cmp-state', 'data'),
    State('perf-summary-snapshots', 'data'),
    prevent_initial_call=True,
)
def update_perf_summary_snapshots(_insert, _remove, _clear, state, snapshot_store):
    """用户操作时单次回调输出：快照 store + 已插入摘要 + 剪贴板纯文本。

    原实现拆成「写 store → 监听 store 重渲染」两步链式回调，两次输出都
    落在 cmp-loading-bins 包裹的卡片内，导致插入时 Loading 动画闪两次；
    合并为单次输出后只闪一次。执行对齐不会触发本回调，历史快照安全。
    """
    triggered = dash_ctx.triggered[0]['prop_id'].split('.')[0] if dash_ctx.triggered else ''
    store = dict(snapshot_store or {})
    items = list(store.get('items') or [])
    rules = (get('performance_metrics', {}) or {}).get('metrics') or {}

    def _render(render_items):
        summary_html = render_performance_snapshot_summary(render_items, rules)
        return summary_html, _summary_plain_text(summary_html)

    if triggered == 'perf-summary-clear-btn':
        summary_html, copy_text = _render([])
        return (
            {'source_key': '', 'items': []},
            '已清空已插入摘要',
            summary_html,
            copy_text,
        )
    if triggered == 'perf-summary-remove-btn':
        if not items:
            raise PreventUpdate
        items.pop()
        new_store = {
            'source_key': store.get('source_key') or '',
            'items': items,
        }
        summary_html, copy_text = _render(items)
        return new_store, '已删除最后一次插入的数据', summary_html, copy_text
    if triggered != 'perf-summary-insert-btn':
        raise PreventUpdate

    summaries = (state or {}).get('perf_summaries') or {}
    if not state or not state.get('alignment_done') or not summaries:
        return no_update, '请先执行对齐，再插入当前数据', no_update, no_update

    source_key = _perf_summary_source_key(state)
    if store.get('source_key') != source_key:
        # 重新上传了数据源：新数据从第一个快照开始，不与旧上传批次混合。
        items = []
    items.append({
        'label': _perf_summary_snapshot_label(state),
        'results': summaries,
    })
    summary_html, copy_text = _render(items)
    return (
        {'source_key': source_key, 'items': items},
        f'已插入第 {len(items)} 次摘要',
        summary_html,
        copy_text,
    )


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
