"""雷达采样中断检测与断线插入。

中断判定必须基于完整时间序列（显示降采样前调用）：若在降采样后的
显示点上判断，相邻保留点之间被跳过的正常帧会被误判为中断。
"""
import numpy as np


def _find_radar_gap_indices(timestamps, gap_factor: float = 2.5) -> np.ndarray:
    """在完整雷达时间序列中查找真实中断的前帧索引。

    必须在显示降采样前调用。若在降采样后的点上判断，相邻保留点之间被
    跳过的正常帧会被误判为中断。
    """
    x = np.asarray(timestamps, dtype=float)
    if len(x) < 2:
        return np.array([], dtype=np.int64)
    diffs = np.diff(x)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(positive_diffs) == 0:
        return np.array([], dtype=np.int64)
    nominal_period = float(np.percentile(positive_diffs, 25))
    if nominal_period <= 0:
        return np.array([], dtype=np.int64)
    return np.flatnonzero(diffs > nominal_period * gap_factor).astype(np.int64)


def _insert_radar_gap_breaks(
    timestamps,
    values,
    hover_texts,
    gap_factor: float = 2.5,
    true_gap_start_times=None,
):
    """在真实雷达采样中断处插入 None 断点，并返回边界标记。

    ``true_gap_start_times`` 由完整时间序列预先计算。仅当该参数未提供时
    才回退到当前显示点间隔判断，保持旧调用的兼容性。
    """
    x = np.asarray(timestamps, dtype=float)
    y = np.asarray(values, dtype=float)
    texts = list(hover_texts)
    if len(x) < 2:
        return x.tolist(), y.tolist(), texts, [], [], []

    if true_gap_start_times is None:
        gap_indices = _find_radar_gap_indices(x, gap_factor)
        true_starts = x[gap_indices]
    else:
        true_starts = np.sort(np.asarray(true_gap_start_times, dtype=float))
    if len(true_starts) == 0:
        return x.tolist(), y.tolist(), texts, [], [], []

    plot_x, plot_y, plot_text = [], [], []
    marker_x, marker_y, marker_text = [], [], []
    for index, (x_value, y_value, text) in enumerate(zip(x, y, texts)):
        plot_x.append(float(x_value))
        plot_y.append(float(y_value))
        plot_text.append(text)
        if index < len(x) - 1:
            # 降采样后两个显示点之间可跨过很多正常帧；只有完整序列已确认
            # 的中断起点落在这个显示区间内，才允许断线。
            has_true_gap = np.any(
                (true_starts >= x_value - 1e-9)
                & (true_starts < x[index + 1] - 1e-9)
            )
            if not has_true_gap:
                continue
            midpoint = float((x_value + x[index + 1]) / 2.0)
            plot_x.append(midpoint)
            plot_y.append(None)
            plot_text.append(None)
            gap_seconds = float(x[index + 1] - x_value)
            marker_x.extend([float(x_value), float(x[index + 1])])
            marker_y.extend([float(y[index]), float(y[index + 1])])
            marker_text.extend([
                f'雷达数据中断开始<br>间隔: {gap_seconds:.3f}s',
                f'雷达数据恢复<br>间隔: {gap_seconds:.3f}s',
            ])
    return plot_x, plot_y, plot_text, marker_x, marker_y, marker_text
