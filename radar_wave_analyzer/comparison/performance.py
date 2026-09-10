"""
目标性能指标评估模块（编排层）。

按 RTK 真实纵向距离 d = abs(rtk_center_x) 将有效样本划分为六个距离段，
逐帧计算误差，输出准确率、RMSE、连续三帧指标 R3 与区间/整体结论。

模块职责拆分（阶段 6）：
  - performance_common  纯数值辅助：状态常量、列解析、阈值向量化、
                        距离分桶、连续性判定（零内部依赖）；
  - performance_metric  单物理量评估：evaluate_metric 及区间/总体汇总；
  - performance（本模块）批量编排 evaluate_all_metrics，并 re-export
    历史导入路径上的全部公开/兼容名称，外部调用方零改动。

设计原则：
  - 纯计算模块，不依赖 Dash，可直接单元测试；
  - 不修改现有对齐逻辑，也不复用仅含 RMSE/Mean/Max 的通用统计；
  - 分桶严格使用 RTK 真值距离，绝不使用雷达 Dx（见设计文档第 6 节）；
  - 阈值与判定参数全部来自配置，禁止硬编码。

术语（对应设计文档第 4 节）：
  d     RTK 真实纵向距离绝对值
  x_i   雷达测量值
  q_i   RTK 真值
  s_i   有符号误差 x_i - q_i
  e_i   绝对误差 abs(s_i)
  T_i   普通精度阈值
  L_i   连续三帧最大误差阈值
  r_i   归一化误差 e_i / L_i
  R3    区间连续三帧指标 max(min(r_i, r_i+1, r_i+2))
"""
from __future__ import annotations

from typing import Any, Optional

from ..config import get

# ── 兼容 re-export：test_performance.py、performance_panel.py、
#    performance_summary.py、exporter.py 均从本模块路径导入以下名称 ──
from .performance_common import (  # noqa: F401
    _RADAR_COL_CANDIDATES,
    _TRUTH_COL_CANDIDATES,
    _TRUTH_DISTANCE_COL,
    STATUS_FAIL,
    STATUS_INSUFFICIENT,
    STATUS_PASS,
    STATUS_UNDECIDABLE,
    _assign_bins,
    _compute_continuity_breaks,
    _resolve_column,
    _resolve_limit,
    _rolling_window_min,
)
from .performance_metric import (  # noqa: F401
    _build_empty_bins,
    _evaluate_single_limit,
    _summarize_overall,
    evaluate_metric,
)

# ============================================================
# 批量入口
# ============================================================

def evaluate_all_metrics(
    aligned_df,
    *,
    config: Optional[dict] = None,
    track_id: Any = None,
    segment_id: Any = None,
    metrics: Optional[list[str]] = None,
) -> dict[str, dict]:
    """一次性评估全部可用物理量并缓存结果。

    Args:
        aligned_df: 完整对齐结果（不可降采样）。
        config: performance_metrics 配置；None 时自动从全局配置读取。
        track_id: 目标标识。
        segment_id: 轨迹段标识。
        metrics: 限定评估的物理量；None 表示全部。

    Returns:
        {metric: evaluate_metric 的结果}
    """
    if config is None:
        config = get('performance_metrics', {}) or {}
    if not config.get('enabled', True):
        return {}

    all_rules = config.get('metrics') or {}
    targets = metrics if metrics else list(all_rules.keys())

    results: dict[str, dict] = {}
    for metric in targets:
        rules = all_rules.get(metric)
        if not rules:
            continue
        results[metric] = evaluate_metric(
            aligned_df, metric, rules, config,
            track_id=track_id, segment_id=segment_id,
        )
    return results
