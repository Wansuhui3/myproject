"""分距离性能摘要组件。

右栏只读摘要与手动插入快照的渲染，数值沿用性能验收的同一份结果；
验收表格本身见 performance_panel。
"""
from typing import Optional

from dash import html

from ..comparison.performance import STATUS_FAIL
from .performance_panel import _fmt, _fmt_percent

# 数值级标红：仅不合格的具体数值着色，行容器与指标标签保持中性灰
_FAIL_CLASS = 'perf-summary-value-fail'


def _value_span(text: str, failed: bool = False) -> html.Span:
    """摘要中的单个数值；failed 为 True 时仅该数值标红。"""
    return html.Span(text, className=_FAIL_CLASS if failed else None)


def _join_values(values, sep: str = ' / ') -> list:
    """把 (文本, 是否不合格) 序列展开为 span 列表，分隔符保持中性。"""
    nodes: list = []
    for index, (text, failed) in enumerate(values):
        if index:
            nodes.append(html.Span(sep))
        nodes.append(_value_span(text, failed))
    return nodes


def render_performance_distance_summary(
    results: Optional[dict],
    metric_rules: Optional[dict] = None,
) -> html.Div:
    """将全部可用物理量的分距离结果渲染为右栏只读摘要。

    该摘要用于快速评审，数值沿用性能验收的同一份结果：RMSE 以及最严重
    连续三帧窗口中的最大绝对误差。无有效样本的距离段不占用版面，整个
    物理量都无样本时该物理量也不显示。
    标红粒度与验收表一致：仅准确率/连续三帧不合格的具体数值标红。
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
            if int(row.get('sample_count') or 0) > 0:
                rows.append(html.Div([
                    html.Span('完整曲线：RMSE '),
                    _value_span(_summary_value(row.get('rmse'))),
                    html.Span('；准确率 '),
                    _value_span(_accuracy_text(row.get('accuracy'))),
                    html.Span('；最大误差 '),
                    _value_span(
                        _summary_value(row.get('max_error_in_worst_window')),
                        row.get('status') == STATUS_FAIL),
                ], className='perf-summary-row'))
        else:
            for bin_result in result.get('bins') or []:
                distance_bin = _distance_label(bin_result.get('distance_bin'))
                if int(bin_result.get('sample_count') or 0) <= 0:
                    # 无有效样本的距离段不占用摘要版面
                    continue
                rows.append(html.Div([
                    html.Span(f'{distance_bin}：RMSE '),
                    _value_span(_summary_value(bin_result.get('rmse'))),
                    html.Span('；准确率 '),
                    _value_span(
                        _accuracy_text(bin_result.get('accuracy')),
                        bin_result.get('accuracy_pass') is False),
                    html.Span('；最大误差(连续三帧) '),
                    _value_span(
                        _summary_value(
                            bin_result.get('max_error_in_worst_window')),
                        bin_result.get('three_frame_pass') is False),
                ], className='perf-summary-row'))

        if not rows:
            # 该物理量所有距离段均无有效样本时不显示
            continue
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
    """渲染多个手动插入的性能摘要快照，供复制与横向评审。

    同一行内每个快照的数值单独判定，只有不合格的具体数值标红，
    距离标签与合格数值保持中性灰。所有快照都在某距离段无有效样本
    时该行不显示，整个物理量都无样本时该物理量也不显示。
    """
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
            # 单限值模式无准确率判定，仅按快照结论标红“最大误差”数值
            max_items = [
                (text, (row or {}).get('status') == STATUS_FAIL)
                for text, row in zip(max_values, limit_rows)
            ]
            # 所有快照都无有效样本时不显示该物理量
            rows = [html.Div([
                html.Span('完整曲线：RMSE '),
                *_join_values([(text, False) for text in rmse_values]),
                html.Span('；准确率 '),
                *_join_values([(text, False) for text in acc_values]),
                html.Span('；最大误差 '),
                *_join_values(max_items),
            ], className='perf-summary-row')] if any(
                int((row or {}).get('sample_count') or 0) > 0 for row in limit_rows
            ) else []
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
                # 所有快照在该距离段都没有有效样本时不显示该行
                if all(
                    int((item or {}).get('sample_count') or 0) <= 0
                    for item in per_snapshot
                ):
                    continue
                acc_items = [
                    (text, (item or {}).get('accuracy_pass') is False)
                    for text, item in zip(acc_values, per_snapshot)
                ]
                max_items = [
                    (text, (item or {}).get('three_frame_pass') is False)
                    for text, item in zip(max_values, per_snapshot)
                ]
                rows.append(html.Div([
                    html.Span(f'{distance_bin}：RMSE '),
                    *_join_values([(text, False) for text in rmse_values]),
                    html.Span('；准确率 '),
                    *_join_values(acc_items),
                    html.Span('；三帧最大误差 '),
                    *_join_values(max_items),
                ], className='perf-summary-row'))

        if not rows:
            # 该物理量所有距离段均无有效样本时不显示
            continue
        sections.append(html.Div([
            html.Div(label, className='perf-summary-metric'),
            *rows,
        ], className='perf-summary-section'))

    return html.Div(sections, className='perf-distance-summary')
