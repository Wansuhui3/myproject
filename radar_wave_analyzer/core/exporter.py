"""
CSV / 图片导出模块。
图片导出使用 matplotlib（零额外依赖，兼容所有平台），不再依赖 kaleido。
"""
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    from ..config import get
except ImportError:
    from config import get  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# 中文字体兜底（Windows 常见字体）
_PLATFORM_FONTS = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'sans-serif']

# CSV 公式注入防护：以 =、+、-、@ 开头的单元格前加单引号
_CSV_INJECTION_CHARS = frozenset(('=', '+', '-', '@'))


def _sanitize_csv_cell(value) -> str:
    """防御性转义：为可能被 Excel 解释为公式的单元格添加前缀。"""
    s = str(value)
    if s and s[0] in _CSV_INJECTION_CHARS:
        return "'" + s
    return s


def _init_matplotlib_fonts(plt):
    """设置 matplotlib 中文字体，避免乱码。"""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in _PLATFORM_FONTS:
        if font_name in available:
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def _load_matplotlib():
    """仅在用户导出图片时加载 Matplotlib，缩短应用启动并降低空载内存。"""
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 后端，确保打包环境可用
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _init_matplotlib_fonts(plt)
    return plt, mticker


def export_trajectory_csv(
    seg_df: pd.DataFrame,
    trajectory_id: str,
    original_id,
    total_frames,
    start_time: str,
    end_time: str,
    output_dir: str,
    meta_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    导出单轨迹原始数据与波动数据为 CSV。

    文件名格式: {original_id}_{start_time}_{end_time}_{total_frames}.csv
    编码: UTF-8 with BOM（Excel 兼容）
    """
    os.makedirs(output_dir, exist_ok=True)

    encoding = get('EXPORT_ENCODING', 'utf-8-sig')
    filename = f'{original_id}_{start_time}_{end_time}_{total_frames}.csv'
    filepath = os.path.join(output_dir, filename)

    # 选择导出列
    export_cols = [
        col for col in (
            'timestamp', 'radar_source_key', 'radar_source_label',
            'source_filename', 'ID', 'Track_Age',
        )
        if col in seg_df.columns
    ]
    for col in [
        'Dx', 'Dy', 'Vx', 'Vy', 'Ax', 'Ay', 'HeadingAngle', 'Vabs',
        'Rx_front', 'Rx_rear', 'Ry',
    ]:
        if col in seg_df.columns:
            export_cols.append(col)
    for col in seg_df.columns:
        if col.startswith('wave_') and col not in export_cols:
            export_cols.append(col)

    export_df = seg_df[export_cols].copy()
    # CSV 公式注入防护：对所有对象类型列进行转义
    for col in export_df.select_dtypes(include=['object']).columns:
        export_df[col] = export_df[col].apply(_sanitize_csv_cell)
    export_df.to_csv(filepath, index=False, encoding=encoding)

    logger.info(f'导出轨迹数据: {filepath}')
    return filepath


def export_stats_csv(
    stats_list: list[dict],
    radar_position: str,
    output_dir: str,
) -> str:
    """导出波动统计结果汇总表。"""
    os.makedirs(output_dir, exist_ok=True)

    encoding = get('EXPORT_ENCODING', 'utf-8-sig')
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{radar_position}_wave_stats_{ts_str}.csv'
    filepath = os.path.join(output_dir, filename)

    stats_df = pd.DataFrame(stats_list)
    # CSV 公式注入防护
    for col in stats_df.select_dtypes(include=['object']).columns:
        stats_df[col] = stats_df[col].apply(_sanitize_csv_cell)
    stats_df.to_csv(filepath, index=False, encoding=encoding)

    logger.info(f'导出统计结果: {filepath}')
    return filepath


def export_graph_image(
    seg_df: pd.DataFrame,
    quantities_list: list[str],
    trajectory_id: str,
    original_id,
    total_frames,
    start_time: str,
    end_time: str,
    output_dir: str,
    highlight_range: Optional[tuple[int, int]] = None,
    diff_cache: Optional[dict] = None,
) -> str:
    """
    使用 matplotlib 导出多子图图表为图片。

    文件名格式: {original_id}_{start_time}_{end_time}_{total_frames}.{ext}

    Args:
        seg_df: 轨迹段 DataFrame（含 parsed 时间列）。
        quantities_list: 物理量字段名列表。
        trajectory_id: 轨迹段 ID（用于日志）。
        original_id: 原始 ID 号。
        total_frames: 帧数。
        start_time: 起始时间字符串。
        end_time: 结束时间字符串。
        output_dir: 输出目录。
        highlight_range: 可选高亮区间 (start_idx, end_idx)。
        diff_cache: 可选差分缓存（加速跳变查找）。

    Returns:
        导出的文件路径。
    """
    plt, mticker = _load_matplotlib()
    os.makedirs(output_dir, exist_ok=True)

    fmt = get('EXPORT_IMAGE_FORMAT', 'png')
    filename = f'{original_id}_{start_time}_{end_time}_{total_frames}.{fmt}'
    filepath = os.path.join(output_dir, filename)

    dpi = get('EXPORT_IMAGE_DPI', 150)
    img_width = get('EXPORT_IMAGE_WIDTH', 1600)
    img_height = get('EXPORT_IMAGE_HEIGHT', 900)
    figsize = (img_width / dpi, img_height / dpi)

    all_quantities = get('quantities', {})
    timestamps = seg_df['timestamp_parsed'].values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    n = len(quantities_list)

    if n == 0:
        raise ValueError('quantities_list 不能为空')

    # 导入跳变查找
    try:
        from ..core.wave_calc import find_max_jump as _find_max_jump  # type: ignore[import-not-found]
    except ImportError:
        from core.wave_calc import find_max_jump as _find_max_jump  # type: ignore[no-redef]

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    # 可配置的间距
    fig.subplots_adjust(hspace=0.06, left=0.08, right=0.97, top=0.96, bottom=0.06)

    for i, qty in enumerate(quantities_list):
        ax = axes[i]
        qty_info = all_quantities.get(qty, {})
        qty_label = qty_info.get('label', qty)
        qty_unit = qty_info.get('unit', '')
        color = colors[i % len(colors)]

        # 主曲线
        ax.plot(timestamps, seg_df[qty].values, color=color, linewidth=1.0, label=qty_label)

        # 最大跳变标记（红色线段）
        max_jump = _find_max_jump(seg_df, qty, diff_cache=diff_cache)
        if max_jump is not None and 0 < max_jump['idx'] < len(seg_df):
            prev_idx = max_jump['idx'] - 1
            curr_idx = max_jump['idx']
            ax.plot(
                [timestamps[prev_idx], timestamps[curr_idx]],
                [seg_df[qty].iloc[prev_idx], seg_df[qty].iloc[curr_idx]],
                color='#dc2626', linewidth=2.5, alpha=0.9,
            )

        # 高亮矩形
        if highlight_range is not None:
            hsl, hsr = highlight_range
            if 0 <= hsl < hsr < len(seg_df):
                ax.axvspan(timestamps[hsl], timestamps[hsr],
                           facecolor='#ff7f0e', alpha=0.10, edgecolor='#ff7f0e',
                           linewidth=1.0, linestyle='-', zorder=0)

        # Y 轴：彩色标题 + 紧凑标签
        short_label = qty_info.get('short_label', qty_label[:4])
        y_title = f'{short_label}({qty_unit})' if qty_unit else short_label
        ax.set_ylabel(y_title, color=color, fontsize=8, labelpad=2)
        ax.tick_params(axis='y', labelsize=7, colors=color)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

        # 图例标注（右上角）
        ax.text(0.99, 0.94, f'{qty_label} ({qty_unit})' if qty_unit else qty_label,
                transform=ax.transAxes, fontsize=9, color=color, fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.78, edgecolor='none'))

        # 网格
        ax.grid(True, alpha=0.25, linewidth=0.5)

    # X 轴（仅最底行）
    axes[-1].set_xlabel('时间', fontsize=9)
    axes[-1].tick_params(axis='x', labelsize=7, rotation=0)
    # 自动格式化时间轴标签
    fig.autofmt_xdate(rotation=0, ha='center')

    plt.setp([ax.get_xticklabels() for ax in axes[:-1]], visible=False)

    # 全局标题（可选，默认不显示以保持简洁）
    # fig.suptitle(f'轨迹 {original_id} ({start_time} → {end_time}, {total_frames}帧)',
    #              fontsize=11, fontweight='bold', y=0.99)

    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f'图片导出失败: {e}') from e
    finally:
        plt.close(fig)

    logger.info(f'导出图表图片: {filepath}')
    return filepath
