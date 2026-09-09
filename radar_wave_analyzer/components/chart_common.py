"""图表共享工具：显示降采样、子图布局、FigureResampler 包装、统一色板。

被 graph_builder（波动图表）与 comparison_charts（对比图表）共同依赖，
保持零业务语义，避免两域图表模块互相导入私有函数。
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import get

# 尝试导入 FigureResampler（可选依赖，缺失时降级为普通 figure）
try:
    from plotly_resampler import FigureResampler  # pyright: ignore[reportMissingImports]
    _RESAMPLER_AVAILABLE = True
except ImportError:
    _RESAMPLER_AVAILABLE = False

# 统一曲线色板（波动页与对比页共用，保证两个页面的雷达曲线配色一致）
COLORS = [
    '#1565C0', '#E65100', '#00796B', '#7B1FA2',
    '#C2185B', '#795548', '#455A64', '#00838F',
]


def _select_display_indices(
    df: pd.DataFrame,
    columns: list[str],
    max_points: Optional[int] = None,
) -> np.ndarray:
    """为显示选择代表性行，保留每个时间桶内各曲线的最小/最大值。

    统计、框选和导出继续使用原始 DataFrame；这里只缩小发送给 Plotly 的
    JSON 体积。多物理量共用同一组索引，保证共享 X 轴严格对齐。
    """
    row_count = len(df)
    if row_count == 0:
        return np.array([], dtype=np.int64)
    if max_points is None:
        max_points = int(get('DISPLAY_MAX_POINTS', 3000))
    if (not get('DISPLAY_DOWNSAMPLING_ENABLED', True)
            or row_count <= max_points or max_points < 4):
        return np.arange(row_count, dtype=np.int64)

    numeric_columns = [column for column in columns if column in df.columns]
    if not numeric_columns:
        return np.unique(np.linspace(0, row_count - 1, max_points, dtype=np.int64))

    bucket_count = max(1, (max_points - 2) // (2 * len(numeric_columns)))
    edges = np.linspace(0, row_count, bucket_count + 1, dtype=np.int64)
    selected: set[int] = {0, row_count - 1}
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
        for start, end in zip(edges[:-1], edges[1:]):
            if end <= start:
                continue
            block = values[start:end]
            finite_positions = np.flatnonzero(np.isfinite(block))
            if len(finite_positions) == 0:
                continue
            finite_values = block[finite_positions]
            selected.add(int(start + finite_positions[int(np.argmin(finite_values))]))
            selected.add(int(start + finite_positions[int(np.argmax(finite_values))]))

    result = np.array(sorted(selected), dtype=np.int64)
    if len(result) > max_points:
        keep = np.linspace(0, len(result) - 1, max_points, dtype=np.int64)
        result = result[keep]
    return result


def _wrap_with_resampler(fig: go.Figure, n_points: int) -> go.Figure:
    """数据点超过阈值时用 FigureResampler 包装，实现交互式降采样。

    小数据集（<= RESAMPLER_MAX_POINTS）原样返回，避免不必要的回调注册。
    注意：FigureResampler 需要 Dash app 上下文（用于注册重采样回调），
    在无 app 上下文时（如单元测试）会自动降级为普通 figure。
    """
    if not _RESAMPLER_AVAILABLE or not get('RESAMPLER_ENABLED', True):
        return fig
    max_pts = get('RESAMPLER_MAX_POINTS', 5000)
    if n_points <= max_pts:
        return fig
    n_samples = get('RESAMPLER_DEFAULT_N_SAMPLES', 2000)
    try:
        # 兼容不同版本参数名：default_n_samples (>=0.9) / max_n_samples (旧版)
        try:
            return FigureResampler(fig, default_n_samples=n_samples)
        except TypeError:
            return FigureResampler(fig, max_n_samples=n_samples)
    except Exception:  # noqa: BLE001
        # 无 app 上下文或构造失败时降级为普通 figure，保证可用性
        return fig


def _compute_subplot_y_domains(n: int, vertical_spacing: float = 0.03) -> list:
    """计算每个子图在 paper 坐标中的 y 范围 [y_bottom, y_top]。

    返回列表，索引 0 对应第 1 行（顶部），索引 n-1 对应第 n 行（底部）。
    """
    if n <= 0:
        return []
    total_spacing = vertical_spacing * max(0, n - 1)
    subplot_height = (1.0 - total_spacing) / n
    domains = []
    for i in range(n):
        y_top = 1.0 - i * (subplot_height + vertical_spacing)
        y_bottom = y_top - subplot_height
        domains.append((y_bottom, y_top))
    return domains
