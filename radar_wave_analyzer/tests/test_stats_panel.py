"""统计面板测试。"""
from radar_wave_analyzer.components.comparison_stats_panel import (
    render_cmp_error_stats,
)
from radar_wave_analyzer.components.performance_summary import (
    render_performance_distance_summary,
    render_performance_snapshot_summary,
)
from radar_wave_analyzer.components.wave_stats_panel import (
    render_wave_snapshot_summary,
)


def test_summary_plain_text_skips_snapshot_meta():
    """复制文本提取：包含数据行，跳过 [1]/[2] 快照来源元信息。"""
    import dash.html as html

    from radar_wave_analyzer.callbacks import _summary_plain_text

    node = html.Div([
        html.Div([
            html.Div('[1] 3个文件 · ID=36 · 段2 · 补偿 0ms',
                     className='perf-summary-snapshot-meta'),
        ], className='perf-summary-snapshot-list'),
        html.Div([
            html.Div('Dx', className='perf-summary-metric'),
            html.Div('10–20m：RMSE 0.1 m；准确率 95.45%',
                     className='perf-summary-row'),
        ], className='perf-summary-section'),
        html.Div('Vx最大波动：0.050 m/s / —', className='perf-summary-row'),
    ])

    text = _summary_plain_text(node)

    assert 'Dx' in text
    assert '10–20m：RMSE 0.1 m' in text
    assert 'Vx最大波动：0.050 m/s / —' in text
    assert '3个文件' not in text
    assert '[1]' not in text


def test_summary_plain_text_keeps_spans_on_single_line():
    """数值级标红拆成多个 span 后，复制文本仍按整行输出，不拆行。"""
    import dash.html as html

    from radar_wave_analyzer.callbacks import _summary_plain_text

    node = html.Div([
        html.Div([
            html.Div('Dx', className='perf-summary-metric'),
            html.Div([
                html.Span('0–10m：RMSE '),
                html.Span('0.123 m'),
                html.Span('；准确率 '),
                html.Span('90.00%', className='perf-summary-value-fail'),
                html.Span('；最大误差(连续三帧) '),
                html.Span('0.456 m'),
            ], className='perf-summary-row'),
        ], className='perf-summary-section'),
    ])

    text = _summary_plain_text(node)

    assert '0–10m：RMSE 0.123 m；准确率 90.00%；最大误差(连续三帧) 0.456 m' in text


def test_comparison_stats_only_contains_plotted_compatible_quantities():
    """面板仅展示当前曲线通道，不再混入固定位置、速度、匹配率或延迟。"""
    panel = render_cmp_error_stats([
        {
            'radar_col': 'RangeX', 'rtk_col': 'TruthX',
            'radar_unit': 'm', 'stats_enabled': True,
            'metrics': {'rmse': 0.7421, 'mean': 0.5},
        },
        {
            'radar_col': 'SpeedY', 'rtk_col': 'TruthVy',
            'radar_unit': 'm/s', 'stats_enabled': True,
            'metrics': {'rmse': 0.1532, 'mean': -0.1},
        },
        {
            'radar_col': 'Angle', 'rtk_col': 'TruthAngle',
            'radar_unit': 'deg', 'stats_enabled': False,
            'metrics': None,
        },
    ])

    rows = panel.children[1].children
    labels = [row.children[0].children for row in rows]
    values = [row.children[1].children for row in rows]

    assert labels == ['RangeX RMSE', 'SpeedY RMSE']
    assert values == ['0.74 m', '0.15 m/s']
    assert all('匹配率' not in label and '延迟' not in label for label in labels)


def test_performance_distance_summary_lists_metrics_and_skips_empty_bins():
    """右栏摘要按物理量输出；无有效样本的距离段不显示。"""
    from radar_wave_analyzer.callbacks import _summary_plain_text

    results = {
        'Dx': {
            'available': True, 'mode': 'binned',
            'bins': [
                {'distance_bin': '[0, 10)', 'sample_count': 3,
                 'rmse': 0.1234, 'max_error_in_worst_window': 0.4567},
                {'distance_bin': '[10, 20)', 'sample_count': 0,
                 'rmse': None, 'max_error_in_worst_window': None},
            ],
        },
        'Dy': {
            'available': True, 'mode': 'binned',
            'bins': [
                {'distance_bin': '[0, 10)', 'sample_count': 2,
                 'rmse': 0.2, 'accuracy': 0.9545,
                 'max_error_in_worst_window': None},
            ],
        },
    }
    panel = render_performance_distance_summary(
        results, {'Dx': {'unit': 'm'}, 'Dy': {'unit': 'm'}})
    rendered = str(panel.to_plotly_json())
    text = _summary_plain_text(panel)

    assert 'Dx' in rendered and 'Dy' in rendered
    assert '0–10m：RMSE 0.123 m；准确率 无；最大误差(连续三帧) 0.457 m' in text
    assert '0–10m：RMSE 0.200 m；准确率 95.45%；最大误差(连续三帧) 无' in text
    # 无样本的 [10, 20) 段不出现在摘要中
    assert '10–20m' not in text
    # 数据缺少合格标志时不标红
    assert 'perf-summary-value-fail' not in rendered


def test_performance_distance_summary_marks_only_failed_values():
    """只有不合格的具体数值标红，行容器与距离标签保持中性。"""
    import dash.html as html

    results = {
        'Dx': {
            'available': True, 'mode': 'binned',
            'bins': [
                {'distance_bin': '[0, 10)', 'sample_count': 5,
                 'rmse': 0.1, 'accuracy': 0.9, 'accuracy_pass': False,
                 'max_error_in_worst_window': 0.2, 'three_frame_pass': True,
                 'status': '不通过'},
                {'distance_bin': '[10, 20)', 'sample_count': 5,
                 'rmse': 0.1, 'accuracy': 0.99, 'accuracy_pass': True,
                 'max_error_in_worst_window': 1.5, 'three_frame_pass': False,
                 'status': '不通过'},
            ],
        },
    }
    panel = render_performance_distance_summary(results, {'Dx': {'unit': 'm'}})
    rows = panel.children[0].children[1:]

    assert [row.className for row in rows] == ['perf-summary-row', 'perf-summary-row']

    def _failed(row):
        return [child.children for child in row.children
                if isinstance(child, html.Span)
                and getattr(child, 'className', None) == 'perf-summary-value-fail']

    # 第一行仅准确率不合格，第二行仅连续三帧最大误差不合格
    assert _failed(rows[0]) == ['90.00%']
    assert _failed(rows[1]) == ['1.500 m']


def test_performance_snapshot_summary_marks_only_failed_values():
    """快照摘要同一行内，各快照的失败数值单独标红。"""
    import dash.html as html

    from radar_wave_analyzer.callbacks import _summary_plain_text

    snapshot = {
        'label': '1_seg1',
        'results': {
            'Dx': {
                'available': True, 'mode': 'binned',
                'bins': [{
                    'distance_bin': '[0, 10)', 'sample_count': 5,
                    'rmse': 0.1, 'accuracy': 0.9, 'accuracy_pass': False,
                    'max_error_in_worst_window': 0.2, 'three_frame_pass': True,
                    'status': '不通过',
                }],
            },
        },
    }
    panel = render_performance_snapshot_summary(
        [snapshot], {'Dx': {'unit': 'm'}})
    row = panel.children[1].children[1]
    text = _summary_plain_text(panel)

    assert row.className == 'perf-summary-row'
    failed = [child.children for child in row.children
              if isinstance(child, html.Span)
              and getattr(child, 'className', None) == 'perf-summary-value-fail']
    assert failed == ['90.00%']
    assert '0–10m：RMSE 0.100 m；准确率 90.00%；三帧最大误差 0.200 m' in text


def test_wave_snapshot_summary_formats_rows_with_slash_join():
    """波动摘要每物理量一行，多快照以 " / " 连接，缺失显示 —。"""
    snapshots = [
        {'label': '1_seg1',
         'values': {'Dx': {'value': 0.123, 'unit': 'm'},
                    'Vx': {'value': 0.05, 'unit': 'm/s'}}},
        {'label': '1_seg2',
         'values': {'Dx': {'value': 0.456, 'unit': 'm'},
                    'Vx': None}},
    ]
    panel = render_wave_snapshot_summary(
        snapshots, {'Dx': {'label': 'Dx'}, 'Vx': {'label': 'Vx'}})
    rendered = str(panel.to_plotly_json())

    assert 'Dx最大波动：0.123 m / 0.456 m' in rendered
    assert 'Vx最大波动：0.050 m/s / —' in rendered
    assert '[1] 1_seg1' in rendered and '[2] 1_seg2' in rendered


def test_wave_snapshot_summary_empty():
    """无快照时显示空态提示。"""
    panel = render_wave_snapshot_summary([], {})
    assert '插入当前' in str(panel.to_plotly_json())


def test_performance_distance_summary_hides_metric_without_samples():
    """整个物理量都没有有效样本时，摘要不显示该物理量。"""
    results = {
        'Dx': {
            'available': True, 'mode': 'binned',
            'bins': [
                {'distance_bin': '[0, 10)', 'sample_count': 0, 'rmse': None},
            ],
        },
    }
    panel = render_performance_distance_summary(results, {'Dx': {'unit': 'm'}})
    rendered = str(panel.to_plotly_json())

    assert 'Dx' not in rendered
    assert 'stats-empty' in rendered


def test_performance_snapshot_summary_hides_bins_without_samples():
    """快照摘要中所有快照都无样本的距离段不显示。"""
    snapshot = {
        'label': '1_seg1',
        'results': {
            'Dx': {
                'available': True, 'mode': 'binned',
                'bins': [
                    {'distance_bin': '[0, 10)', 'sample_count': 4,
                     'rmse': 0.1, 'accuracy': 0.99, 'accuracy_pass': True,
                     'max_error_in_worst_window': 0.2, 'three_frame_pass': True,
                     'status': '通过'},
                    {'distance_bin': '[10, 20)', 'sample_count': 0,
                     'rmse': None, 'accuracy': None,
                     'max_error_in_worst_window': None, 'status': '不可判定'},
                ],
            },
        },
    }
    panel = render_performance_snapshot_summary(
        [snapshot], {'Dx': {'unit': 'm'}})
    section = panel.children[1]

    # 物理量标题 + 一行有效数据；空距离段不占版面
    assert len(section.children) == 2
    assert '10–20m' not in str(panel.to_plotly_json())
