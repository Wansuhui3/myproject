"""
坐标系一致性诊断模块。
使用 KDTree 加速最近邻搜索，验证雷达 Dx/Dy 与 RTK center_x/center_y 是否在同一车辆坐标系下。
"""
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def diagnose_coordinate_system(
    radar_df: pd.DataFrame,
    rtk_df: pd.DataFrame,
    track_id: int,
    n_frames: int = 100,
    bias_threshold_m: float = 0.5,
    file_index: int = None,
) -> dict:
    """诊断雷达与RTK是否在同一坐标系。

    算法（来自文档 Step⑤）:
        取雷达选定ID的前 n_frames 帧，使用 KDTree 批量查询每帧的最近 RTK 点
        Δx_i = Dx_i - nearest_rtk(center_x)_i
        Δy_i = Dy_i - nearest_rtk(center_y)_i
        若 |mean(Δx)| < bias_threshold 且 |mean(Δy)| < bias_threshold → 同系
        否则 → 坐标系不同，需要坐标变换

    Args:
        radar_df: 雷达全量数据。
        rtk_df: RTK全量数据。
        track_id: 选定的雷达目标ID。
        n_frames: 采样帧数。
        bias_threshold_m: 偏差阈值(米)。
        file_index: 可选，来源文件序号。多文件场景下用于过滤同ID不同时间段的数据。

    Returns:
        dict:
            same_system: bool
            mean_dx, mean_dy: 偏差均值
            std_dx, std_dy: 偏差标准差
            diagnosis: 诊断描述
    """
    if file_index is not None:
        id_mask = (radar_df['ID'] == track_id) & (radar_df['file_index'] == file_index)
    else:
        id_mask = radar_df['ID'] == track_id
    id_df = radar_df[id_mask]

    sample_df = id_df.head(n_frames)
    n_sample = len(sample_df)

    if n_sample == 0 or len(rtk_df) == 0:
        return {
            'same_system': False,
            'mean_dx': None,
            'mean_dy': None,
            'std_dx': None,
            'std_dy': None,
            'diagnosis': '数据不足，无法诊断',
        }

    # 使用 KDTree 一次性查询所有样本帧的最近 RTK 点
    rtk_coords = np.column_stack([
        rtk_df['center_x'].values, rtk_df['center_y'].values
    ])
    kdtree = KDTree(rtk_coords)

    radar_coords = np.column_stack([
        sample_df['Dx'].values, sample_df['Dy'].values
    ])
    _, nearest_indices = kdtree.query(radar_coords)

    dx_diffs = sample_df['Dx'].values - rtk_df.iloc[nearest_indices]['center_x'].values
    dy_diffs = sample_df['Dy'].values - rtk_df.iloc[nearest_indices]['center_y'].values

    mean_dx = float(np.mean(dx_diffs))
    mean_dy = float(np.mean(dy_diffs))
    std_dx = float(np.std(dx_diffs))
    std_dy = float(np.std(dy_diffs))

    same_system = abs(mean_dx) < bias_threshold_m and abs(mean_dy) < bias_threshold_m

    if same_system:
        diagnosis = (
            f'同系 ✅  |  Δx bias={mean_dx:+.2f}m  Δy bias={mean_dy:+.2f}m  '
            f'(阈值={bias_threshold_m}m)'
        )
    else:
        diagnosis = (
            f'坐标系不一致 ≠  |  Δx bias={mean_dx:+.2f}m  Δy bias={mean_dy:+.2f}m  '
            f'(超过阈值{bias_threshold_m}m)'
        )

    return {
        'same_system': same_system,
        'mean_dx': round(mean_dx, 3),
        'mean_dy': round(mean_dy, 3),
        'std_dx': round(std_dx, 3),
        'std_dy': round(std_dy, 3),
        'diagnosis': diagnosis,
    }
