"""
目标性能指标验收表面板组件。

渲染分距离验收表（中栏图表下方，可折叠）与右栏只读分距离性能摘要。

设计依据（技术设计文档）：
  - 10.3 分距离验收表列结构与状态配色；
  - 8.6 Ay 使用完整曲线单行摘要，不使用六段表。
"""
from typing import Optional

from dash import dcc, html

from ..comparison.performance import (
    STATUS_FAIL, STATUS_INSUFFICIENT, STATUS_PASS, STATUS_UNDECIDABLE,
)

# 状态 → CSS 类名（与 10.3 配色规则对应）
_STATUS_CLASS = {
    STATUS_PASS: 'perf-status-pass',
    STATUS_FAIL: 'perf-status-fail',
    STATUS_INSUFFICIENT: 'perf-status-insufficient',
    STATUS_UNDECIDABLE: 'perf-status-undecidable',
}

_ACCURACY_REQUIREMENT_TEXT = '>95.45%'


def _fmt(value, decimals: int = 3) -> str:
    """格式化数值；None 显示占位符。"""
    if value is None:
        return '—'
    try:
        return f'{float(value):.{decimals}f}'
    except (TypeError, ValueError):
        return '—'


def _fmt_percent(value) -> str:
    """格式化百分比。"""
    if value is None:
        return '—'
    try:
        return f'{float(value) * 100:.2f}%'
    except (TypeError, ValueError):
        return '—'


def _format_three_frame_limit(limit: Optional[dict], unit: str, basis: str) -> str:
    """将连续三帧性能指标限值转换为紧凑表达式。

    单位已由当前物理量表头表达，公式内不重复展示；速度/加速度的真值
    基准分别简写为 |V|、|A|，避免动态限值列过宽。
    """
    if not limit:
        return '—'
    mode = str(limit.get('mode') or 'absolute')
    basis_text = (
        '距离' if basis == 'distance'
        else ('|A|' if '²' in unit or '^2' in unit else '|V|')
    )

    def _absolute(value) -> str:
        return f'{float(value):g}'

    def _percent(value) -> str:
        return f'{float(value) * 100:g}%·{basis_text}'

    try:
        if mode == 'absolute':
            return f'<{_absolute(limit["absolute"])}'
        if mode == 'percent':
            return f'<{_percent(limit["percent"])}'
        if mode == 'max_absolute_percent':
            return f'<max({_absolute(limit["absolute"])}, {_percent(limit["percent"])})'
        if mode == 'scaled_max_absolute_percent':
            scale = float(limit.get('scale', 1))
            return (
                f'<{scale:g}×max({_absolute(limit["absolute"])}, '
                f'{_percent(limit["percent"])})'
            )
    except (KeyError, TypeError, ValueError):
        return '—'
    return '—'


def _status_cell(status: str) -> html.Td:
    """状态单元格，按结论着色。"""
    return html.Td(status, className=_STATUS_CLASS.get(status, ''))


def _cell(value: str, failed: bool = False) -> html.Td:
    """普通单元格；failed 为 True 时标红（10.3 标红具体失败单元格）。"""
    return html.Td(value, className='perf-cell-fail' if failed else None)


def _build_binned_table(
    bins: list,
    unit: str,
    metric_rules: Optional[dict] = None,
) -> html.Table:
    """构建六距离段验收表。"""
    unit_suffix = f'({unit})' if unit else ''
    rows = []
    rules_by_bin = (metric_rules or {}).get('bins') or []
    basis = str((metric_rules or {}).get('percent_basis') or 'truth')
    for index, b in enumerate(bins):
        status = b.get('status', '不可判定')
        # 失败单元格标红：准确率不通过 / 连续三帧不通过
        acc_failed = b.get('accuracy_pass') is False
        three_failed = b.get('three_frame_pass') is False

        rows.append(html.Tr([
            html.Td(b.get('distance_bin', '—')),
            html.Td(str(b.get('sample_count', 0))),
            _cell(_fmt(b.get('rmse'))),
            _cell(_fmt_percent(b.get('accuracy')), acc_failed),
            _cell(_ACCURACY_REQUIREMENT_TEXT if b.get('sample_count', 0) > 0 else '—'),
            _cell(_fmt(b.get('max_error_in_worst_window')), three_failed),
            _cell(
                _format_three_frame_limit(
                    (rules_by_bin[index] or {}).get('three_frame_limit')
                    if index < len(rules_by_bin) else None,
                    unit, basis,
                )
            ),
            _status_cell(status),
        ], className='bin-row-empty' if b.get('sample_count', 0) == 0 else None))

    return html.Table([
        html.Thead(html.Tr([
            html.Th('距离段'),
            html.Th('样本数'),
            html.Th(f'RMSE{unit_suffix}'),
            html.Th('准确率'),
            html.Th('要求'),
            html.Th(f'三帧最大误差{unit_suffix}'),
            html.Th('限值'),
            html.Th('结论'),
        ])),
        html.Tbody(rows),
    ], className='traj-table perf-table')


def _build_single_limit_table(bins: list, unit: str, abs_limit) -> html.Table:
    """Ay 单行摘要表：完整曲线的样本数、RMSE、最大绝对误差、限值与结论。"""
    row = bins[0] if bins else {}
    status = row.get('status', '不可判定')
    # 有数据且限值为 None 时说明配置异常，显示占位符
    limit_text = '—' if abs_limit is None else f'≤{abs_limit}'
    max_err = row.get('max_error_in_worst_window')
    failed = status == '不通过'

    body = html.Tr([
        html.Td('完整曲线'),
        html.Td(str(row.get('sample_count', 0))),
        html.Td(_fmt(row.get('rmse'))),
        _cell(_fmt(max_err), failed),
        html.Td(limit_text),
        _status_cell(status),
    ], className='bin-row-empty' if row.get('sample_count', 0) == 0 else None)

    unit_suffix = f'({unit})' if unit else ''
    return html.Table([
        html.Thead(html.Tr([
            html.Th('范围'),
            html.Th('样本数'),
            html.Th(f'RMSE{unit_suffix}'),
            html.Th(f'最大绝对误差{unit_suffix}'),
            html.Th('限值'),
            html.Th('结论'),
        ])),
        html.Tbody([body]),
    ], className='traj-table perf-table')


def render_performance_body(
    result: Optional[dict],
    unit: str = '',
    metric_rules: Optional[dict] = None,
):
    """渲染验收表体（不含折叠头），供首次渲染与指标切换复用。"""
    if not result or not result.get('available'):
        reason = (result or {}).get('reason') or '执行对齐后显示'
        return html.Div(reason, className='stats-empty')
    mode = result.get('mode', 'binned')
    bins = result.get('bins') or []
    if mode == 'single_limit':
        return _build_single_limit_table(bins, unit, result.get('abs_limit'))
    return _build_binned_table(bins, unit, metric_rules)


def _row_class(status: str) -> str:
    """按验收结论着色：仅不通过标红，通过/样本不足/不可判定均保持中性灰。"""
    if status == STATUS_FAIL:
        return 'perf-summary-row perf-summary-row-fail'
    return 'perf-summary-row'


def render_performance_distance_summary(
    results: Optional[dict],
    metric_rules: Optional[dict] = None,
) -> html.Div:
    """将全部可用物理量的分距离结果渲染为右栏只读摘要。

    该摘要用于快速评审，数值沿用性能验收的同一份结果：RMSE 以及最严重
    连续三帧窗口中的最大绝对误差。距离段没有有效样本时明确显示“无”。
    """
    rules = metric_rules or {}

    def _distance_label(value) -> str:
        """将验收内部标签 ``[0, 10)`` 转为评审摘要的 ``0–10m``。"""
        raw = str(value or '—')
        cleaned = raw.strip().lstrip('[').rstrip(')]')
        if ',' not in cleaned:
            return raw
        lo, hi = (part.strip() for part in cleaned.split(',', 1))
        return f'{lo}–{hi}m'

    def _summary_value(value) -> str:
        """摘要中缺失的统计值统一显示“无”，避免被误读为 0。"""
        return '无' if value is None else f'{_fmt(value)}{suffix}'

    def _accuracy_text(value) -> str:
        """准确率复用 _fmt_percent（百分比两位小数），缺失显示“无”。"""
        return '无' if value is None else _fmt_percent(value)

    sections = []
    for metric, result in (results or {}).items():
        if not result or not result.get('available'):
            continue
        rule = rules.get(metric, {})
        label = str(rule.get('label') or metric)
        unit = str(rule.get('unit') or '')
        suffix = f' {unit}' if unit else ''
        rows = []

        if result.get('mode') == 'single_limit':
            row = (result.get('bins') or [{}])[0]
            if int(row.get('sample_count') or 0) <= 0:
                text = '完整曲线：无'
            else:
                text = (
                    f'完整曲线：RMSE {_summary_value(row.get("rmse"))}；'
                    f'准确率 {_accuracy_text(row.get("accuracy"))}；'
                    f'最大误差 {_summary_value(row.get("max_error_in_worst_window"))}'
                )
            rows.append(html.Div(text, className=_row_class(
                row.get('status') or STATUS_UNDECIDABLE)))
        else:
            for bin_result in result.get('bins') or []:
                distance_bin = _distance_label(bin_result.get('distance_bin'))
                if int(bin_result.get('sample_count') or 0) <= 0:
                    text = f'{distance_bin}：无'
                else:
                    text = (
                        f'{distance_bin}：RMSE {_summary_value(bin_result.get("rmse"))}；'
                        f'准确率 {_accuracy_text(bin_result.get("accuracy"))}；'
                        '最大误差(连续三帧) '
                        f'{_summary_value(bin_result.get("max_error_in_worst_window"))}'
                    )
                rows.append(html.Div(text, className=_row_class(
                    bin_result.get('status') or STATUS_UNDECIDABLE)))

        sections.append(html.Div([
            html.Div(label, className='perf-summary-metric'),
            *rows,
        ], className='perf-summary-section'))

    if not sections:
        return html.Div('执行对齐后显示', className='stats-empty')
    return html.Div(sections, className='perf-distance-summary')


def render_performance_snapshot_summary(
    snapshots: list[dict],
    metric_rules: Optional[dict] = None,
) -> html.Div:
    """渲染多个手动插入的性能摘要快照，供复制与横向评审。"""
    rules = metric_rules or {}
    if not snapshots:
        return html.Div('执行对齐后点击“插入当前”保留摘要', className='stats-empty')

    def _distance_label(value) -> str:
        raw = str(value or '—')
        cleaned = raw.strip().lstrip('[').rstrip(')]')
        if ',' not in cleaned:
            return raw
        lo, hi = (part.strip() for part in cleaned.split(',', 1))
        return f'{lo}–{hi}m'

    def _value(value, unit: str) -> str:
        return '无' if value is None else f'{_fmt(value)} {unit}'.rstrip()

    def _accuracy_text(value) -> str:
        """准确率复用 _fmt_percent（百分比两位小数），缺失显示“无”。"""
        return '无' if value is None else _fmt_percent(value)

    def _bin_text(bin_result: Optional[dict], unit: str) -> tuple[str, str, str]:
        if not bin_result or int(bin_result.get('sample_count') or 0) <= 0:
            return '无', '无', '无'
        return (
            _value(bin_result.get('rmse'), unit),
            _accuracy_text(bin_result.get('accuracy')),
            _value(bin_result.get('max_error_in_worst_window'), unit),
        )

    metadata = []
    for index, snapshot in enumerate(snapshots, start=1):
        metadata.append(html.Div(
            f'[{index}] {snapshot.get("label") or "未命名分析"}',
            className='perf-summary-snapshot-meta',
        ))

    metric_order = list(rules)
    for snapshot in snapshots:
        for metric in (snapshot.get('results') or {}):
            if metric not in metric_order:
                metric_order.append(metric)

    sections = [html.Div(metadata, className='perf-summary-snapshot-list')]
    for metric in metric_order:
        rule = rules.get(metric, {})
        unit = str(rule.get('unit') or '')
        label = str(rule.get('label') or metric)
        results = [
            (snapshot.get('results') or {}).get(metric)
            for snapshot in snapshots
        ]
        if not any(result and result.get('available') for result in results):
            continue

        if any((result or {}).get('mode') == 'single_limit' for result in results):
            limit_rows, rmse_values, acc_values, max_values = [], [], [], []
            for result in results:
                row = ((result or {}).get('bins') or [{}])[0]
                limit_rows.append(row)
                rmse, acc, max_error = _bin_text(row, unit)
                rmse_values.append(rmse)
                acc_values.append(acc)
                max_values.append(max_error)
            fail = any(
                (row or {}).get('status') == STATUS_FAIL for row in limit_rows)
            rows = [html.Div(
                f'完整曲线：RMSE {" / ".join(rmse_values)}；'
                f'准确率 {" / ".join(acc_values)}；'
                f'最大误差 {" / ".join(max_values)}',
                className=_row_class(STATUS_FAIL if fail else ''),
            )]
        else:
            bin_count = max(len((result or {}).get('bins') or []) for result in results)
            rows = []
            for bin_index in range(bin_count):
                per_snapshot = [
                    ((result or {}).get('bins') or [])[bin_index]
                    if bin_index < len((result or {}).get('bins') or []) else None
                    for result in results
                ]
                distance_bin = _distance_label(
                    next((item.get('distance_bin') for item in per_snapshot if item), '—')
                )
                values = [_bin_text(item, unit) for item in per_snapshot]
                rmse_values = [value[0] for value in values]
                acc_values = [value[1] for value in values]
                max_values = [value[2] for value in values]
                if all(value == '无' for value in rmse_values):
                    text = f'{distance_bin}：' + ' / '.join(rmse_values)
                else:
                    text = (
                        f'{distance_bin}：RMSE {" / ".join(rmse_values)}；'
                        f'准确率 {" / ".join(acc_values)}；'
                        f'三帧最大误差 {" / ".join(max_values)}'
                    )
                rows.append(html.Div(
                    text, className=_row_class(
                        STATUS_FAIL if any(
                            (item or {}).get('status') == STATUS_FAIL
                            for item in per_snapshot) else '')))

        sections.append(html.Div([
            html.Div(label, className='perf-summary-metric'),
            *rows,
        ], className='perf-summary-section'))

    return html.Div(sections, className='perf-distance-summary')


def render_performance_table(
    result: Optional[dict],
    metric: str,
    unit: str = '',
    metric_options: Optional[list] = None,
    metric_rules: Optional[dict] = None,
    collapsed: bool = True,
) -> html.Div:
    """渲染可折叠的验收表区域。

    Args:
        result: evaluate_metric 的返回结果。
        metric: 当前物理量名称。
        unit: 单位。
        metric_options: 可选物理量选项 [{'label','value'}]，渲染为
            横向单选框（比下拉框少一次点击）。
        collapsed: 是否折叠（折叠时仅显示标题栏），默认初始隐藏。
    """
    collapse_cls = 'perf-body-collapsed' if collapsed else ''
    arrow = '▸' if collapsed else '▾'
    options = metric_options or []

    # 折叠开关必须是独立点击目标。RadioItems 若放在带 n_clicks 的父容器中，
    # 点击物理量会冒泡到父容器，导致刚切换指标表格就被误折叠。
    header = html.Div([
        html.Button([
            html.Span(f'{arrow}', className='perf-collapse-arrow', id='perf-collapse-arrow'),
            html.Span('性能验收', className='app-card-title'),
        ], id='perf-collapse-toggle', className='perf-collapse-toggle',
           n_clicks=0, type='button', title='展开/收起性能验收表'),
        dcc.RadioItems(
            id='perf-metric-selector',
            options=options,
            value=metric or None,
            inline=True,
            className='perf-metric-radios',
        ),
        html.Span(
            '区间依据：RTK 真实纵向距离',
            className='perf-basis-hint',
        ),
    ], id='perf-collapse-header', className='perf-collapse-header')

    body = render_performance_body(result, unit, metric_rules)

    return html.Div([
        header,
        html.Div(body, id='perf-table-body', className=f'perf-table-body {collapse_cls}'),
    ], id='perf-table-card', className='perf-table-card')
