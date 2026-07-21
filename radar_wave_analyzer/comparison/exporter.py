"""
对比结果导出模块。
支持导出对齐结果CSV、汇总统计JSON、对比图表PNG。
"""
import json
import logging
import os

import pandas as pd

try:
    from ..config import get
except ImportError:
    from config import get  # type: ignore

logger = logging.getLogger(__name__)

# CSV 公式注入防护：以 =、+、-、@ 开头的单元格前加单引号
_CSV_INJECTION_CHARS = frozenset(('=', '+', '-', '@'))


def _sanitize_csv_cell(value) -> str:
    """防御性转义：为可能被 Excel 解释为公式的单元格添加前缀。"""
    s = str(value)
    if s and s[0] in _CSV_INJECTION_CHARS:
        return "'" + s
    return s


def export_aligned_csv(aligned_df: pd.DataFrame, filepath: str) -> str:
    """导出对齐结果CSV。

    Args:
        aligned_df: 对齐结果DataFrame。
        filepath: 导出文件路径。

    Returns:
        导出成功消息，或错误信息。
    """
    try:
        encoding = get('comparison', {}).get('export_encoding', 'utf-8-sig')
        # 只导出核心列（去掉中间计算列）
        export_cols = [
            'timestamp', 'radar_Dx', 'radar_Dy', 'radar_Vx', 'radar_Vy',
            'rtk_center_x', 'rtk_center_y', 'rtk_Vx', 'rtk_Vy',
            'pos_error_x', 'pos_error_y', 'pos_error_abs',
            'vel_error_x', 'vel_error_y', 'vel_error_abs',
            'match_dist', 'is_matched',
        ]
        export_df = aligned_df[export_cols].copy()
        # CSV 公式注入防护
        for col in export_df.select_dtypes(include=['object']).columns:
            export_df[col] = export_df[col].apply(_sanitize_csv_cell)
        export_df.to_csv(filepath, index=False, encoding=encoding)
        return f'CSV已导出: {os.path.basename(filepath)} ({len(export_df)}行)'
    except Exception as e:
        logger.exception('导出CSV失败')
        return f'导出失败: {e}'


def export_summary_json(summary: dict, match_summary: dict, filepath: str) -> str:
    """导出汇总统计JSON。

    Args:
        summary: 误差统计汇总。
        match_summary: 匹配概况。
        filepath: 导出文件路径。

    Returns:
        导出成功消息。
    """
    try:
        output = {
            'match': match_summary,
            'errors': summary,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return f'JSON已导出: {os.path.basename(filepath)}'
    except Exception as e:
        logger.exception('导出JSON失败')
        return f'导出失败: {e}'
