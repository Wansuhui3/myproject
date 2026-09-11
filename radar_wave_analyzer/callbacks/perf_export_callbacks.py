"""性能验收导出域回调 [C6]：导出对齐 + 逐帧性能 + 验收汇总 xlsx 工作簿。

执行对齐见 perf_run_callbacks；摘要快照 / 指标切换见 perf_panel_callbacks。
"""
import logging
import os

import pandas as pd
from dash import Input, Output, State, callback, dcc, no_update
from dash.exceptions import PreventUpdate

from ..cache import get_alignment_result, get_performance_result
from ..comparison.exporter import (
    derive_quantity_pairs,
    export_comparison_workbook_bytes,
)
from ..config import get
from .perf_shared import _selected_radar_filename

logger = logging.getLogger(__name__)


# ---- [C6] 导出 CSV（真值对齐 + 性能验收工作簿） ----

@callback(
    Output('cmp-export-download', 'data'),
    Output('cmp-export-feedback', 'children'),
    Input('cmp-export-csv-btn', 'n_clicks'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_export_workbook(_n, state):
    """导出最近一次执行对齐的数据为 xlsx 工作簿（本地保存）。

    默认文件名 = 数据来源的雷达 CSV 文件名（去扩展名）+ 对齐 ID，
    如 ``CD701_flr_track_2026_08_11_15_57_24_ID3.xlsx``。桌面端由
    pywebview 弹出系统“另存为”对话框选择保存路径（依赖 launcher 中
    开启的 ALLOW_DOWNLOADS）；浏览器端则走浏览器下载。导出不占用
    服务端文件，Excel 打开旧文件也不影响再次导出。
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
    return data, f'已生成 {default_name}，请选择保存位置'
