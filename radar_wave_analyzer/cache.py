"""
flask-caching 服务端缓存模块。

用于缓存大数据（预处理 DataFrame、分段元信息、单轨迹段），
避免在回调中重复加载与计算。

缓存键按 Flask session 与雷达位置双重隔离，避免多个浏览器窗口/用户互相覆盖：
  - 'session:{id}:radar:{radar}:df'          : 预处理后的全量 DataFrame
  - 'session:{id}:radar:{radar}:meta_df'     : 分段元信息表
  - 'session:{id}:radar:{radar}:segments'    : 轨迹段到源行号的紧凑索引 dict
  - 'session:{id}:current_radar'             : 当前选中的雷达位置标识
"""
import logging
import uuid

from flask import has_request_context, session

try:
    from .extensions import cache
except ImportError:
    from extensions import cache

try:
    from .config import get
except ImportError:
    from config import get

logger = logging.getLogger(__name__)

_SESSION_ID_KEY = 'radar_wave_cache_session'
_FALLBACK_SESSION_ID = 'no-request-context'
_SOURCE_ROW_COLUMN = '__source_row_index__'


# ============================================================
# 内部工具
# ============================================================

def _session_prefix() -> str:
    """返回当前浏览器会话的缓存前缀。

    Dash 回调中存在 Flask request context，因此每个浏览器会话都会获得独立
    的随机命名空间。无 request context 的 CLI/单元测试保留稳定回退键。
    """
    if not has_request_context():
        return f'session:{_FALLBACK_SESSION_ID}:'

    session_id = session.get(_SESSION_ID_KEY)
    if not session_id:
        session_id = uuid.uuid4().hex
        session[_SESSION_ID_KEY] = session_id
    return f'session:{session_id}:'


def _radar_prefix(radar_key: str) -> str:
    """生成当前会话中指定雷达的缓存键前缀。"""
    return f'{_session_prefix()}radar:{radar_key}:'


def _current_radar_key() -> str:
    return f'{_session_prefix()}current_radar'


def _radar_keys_key() -> str:
    return f'{_session_prefix()}radar_keys'


def _comparison_key(key: str, suffix: str) -> str:
    return f'{_session_prefix()}cmp:{key}:{suffix}'


def _set_current_radar(radar_key: str) -> None:
    """设置当前雷达。"""
    cache.set(_current_radar_key(), radar_key)


def get_radar_position() -> str:
    """获取当前雷达位置标识。"""
    return cache.get(_current_radar_key())


# ============================================================
# 缓存辅助函数：统一管理大数据的存取
# ============================================================

def set_data_cache(
    file_path: str,
    radar_position: str,
    df,
    meta_df,
    segments: dict,
) -> None:
    """文件加载后，缓存全部预处理与分段结果（per-radar 隔离）。

    segments 作为单个 dict 缓存；上传路径会压缩为源行号数组，避免保存
    全量 DataFrame 与所有分段 DataFrame 两份数据。
    """
    prefix = _radar_prefix(radar_position)
    _set_current_radar(radar_position)
    radar_keys = set(cache.get(_radar_keys_key()) or [])
    radar_keys.add(radar_position)
    cache.set(_radar_keys_key(), sorted(radar_keys))
    cache.set(f'{prefix}file_path', file_path)
    cache.set(f'{prefix}df', df)
    cache.set(f'{prefix}meta_df', meta_df)
    # 分段器为了独立索引会复制每个轨迹段。缓存这些 DataFrame 等价于长期
    # 保存第二份完整数据；上传路径提供源行号时只保留紧凑的整数索引。
    compact_segments = {}
    can_compact = bool(segments) and all(
        hasattr(segment, 'columns') and _SOURCE_ROW_COLUMN in segment.columns
        for segment in segments.values()
    )
    if can_compact:
        compact_segments = {
            trajectory_id: segment[_SOURCE_ROW_COLUMN].to_numpy(dtype='int64')
            for trajectory_id, segment in segments.items()
        }
    else:
        # CLI、旧缓存和外部调用仍可传入原 DataFrame 字典，保持兼容。
        compact_segments = segments
    cache.set(f'{prefix}segments', compact_segments)
    logger.info(
        f'缓存已写入 [{radar_position}]: {len(df)}行, '
        f'{len(meta_df)}段, {len(segments)}个轨迹段'
    )


def clear_current_radar_cache() -> None:
    """清空当前雷达的数据缓存（不删其他雷达的数据）。"""
    radar_key = get_radar_position()
    if not radar_key:
        return
    prefix = _radar_prefix(radar_key)
    cache.delete(f'{prefix}segments')
    cache.delete(f'{prefix}df')
    cache.delete(f'{prefix}meta_df')
    cache.delete(f'{prefix}file_path')


def clear_data_cache() -> None:
    """清空当前会话中所有雷达的数据缓存。"""
    radar_keys = cache.get(_radar_keys_key()) or []
    for radar_key in radar_keys:
        prefix = _radar_prefix(radar_key)
        cache.delete(f'{prefix}segments')
        cache.delete(f'{prefix}df')
        cache.delete(f'{prefix}meta_df')
        cache.delete(f'{prefix}file_path')
    cache.delete(_radar_keys_key())
    cache.delete(_current_radar_key())


def switch_radar(radar_key: str) -> bool:
    """切换到指定雷达，返回该雷达是否有已缓存数据。"""
    _set_current_radar(radar_key)
    return radar_has_data(radar_key)


def radar_has_data(radar_key: str) -> bool:
    """检查指定雷达是否有已缓存的数据。"""
    prefix = _radar_prefix(radar_key)
    return cache.get(f'{prefix}df') is not None


def get_df():
    """获取当前雷达的 DataFrame。"""
    radar_key = get_radar_position()
    if not radar_key:
        return None
    return cache.get(f'{_radar_prefix(radar_key)}df')


def get_meta_df():
    """获取当前雷达的元信息表。"""
    radar_key = get_radar_position()
    if not radar_key:
        return None
    return cache.get(f'{_radar_prefix(radar_key)}meta_df')


def get_file_path() -> str:
    """获取当前雷达加载的文件路径。"""
    radar_key = get_radar_position()
    if not radar_key:
        return None
    return cache.get(f'{_radar_prefix(radar_key)}file_path')


def get_segment(traj_id: str):
    """获取当前雷达的单个轨迹段 DataFrame。"""
    radar_key = get_radar_position()
    if not radar_key:
        return None
    segments = cache.get(f'{_radar_prefix(radar_key)}segments')
    if segments is None:
        return None
    segment = segments.get(traj_id)
    if segment is None:
        return None
    # 新缓存格式是源 DataFrame 的 iloc 行号；旧格式仍直接返回 DataFrame。
    if hasattr(segment, 'dtype') and getattr(segment.dtype, 'kind', '') in ('i', 'u'):
        df = cache.get(f'{_radar_prefix(radar_key)}df')
        if df is None:
            return None
        return df.iloc[segment].copy().reset_index(drop=True)
    return segment


def get_all_segment_ids() -> list:
    """获取当前雷达所有轨迹段 ID 列表。"""
    radar_key = get_radar_position()
    if not radar_key:
        return []
    segments = cache.get(f'{_radar_prefix(radar_key)}segments')
    if segments is None:
        return []
    return list(segments.keys())


def has_data_loaded() -> bool:
    """判断当前雷达是否已加载有效数据。"""
    return get_df() is not None


# ============================================================
# 对比数据缓存（cmp: 前缀，与波动分析数据隔离）
# ============================================================

def set_comparison_data(key: str, radar_df, rtk_df):
    """缓存对比雷达与RTK原始数据。
    支持仅缓存单个（另一个传 None），不会覆盖已有的缓存。
    """
    if radar_df is not None:
        cache.set(_comparison_key(key, 'radar_df'), radar_df)
    if rtk_df is not None:
        cache.set(_comparison_key(key, 'rtk_df'), rtk_df)
    r_len = len(radar_df) if radar_df is not None else 0
    t_len = len(rtk_df) if rtk_df is not None else 0
    logger.info(f'对比数据已缓存 [{key}]: radar={r_len}行, rtk={t_len}行')


def get_comparison_data(key: str):
    """获取缓存的对比原始数据。"""
    return cache.get(_comparison_key(key, 'radar_df')), cache.get(_comparison_key(key, 'rtk_df'))


def set_alignment_result(key: str, aligned_df, summary, rtk_curve_df=None):
    """缓存对齐结果及用于连续渲染的原始 RTK 曲线。"""
    cache.set(_comparison_key(key, 'aligned_df'), aligned_df)
    cache.set(_comparison_key(key, 'summary'), summary)
    if rtk_curve_df is not None:
        cache.set(_comparison_key(key, 'rtk_curve_df'), rtk_curve_df)


def get_alignment_result(key: str):
    """获取对齐结果。"""
    return (
        cache.get(_comparison_key(key, 'aligned_df')),
        cache.get(_comparison_key(key, 'summary')),
    )


def get_rtk_curve_result(key: str):
    """获取当前对齐所关联的完整 RTK 轨迹，用于雷达缺帧时保持真值曲线连续。"""
    return cache.get(_comparison_key(key, 'rtk_curve_df'))


def clear_comparison_data(key: str = 'default'):
    """清除对比数据缓存。"""
    for suffix in ['radar_df', 'rtk_df', 'aligned_df', 'summary', 'rtk_curve_df']:
        cache.delete(_comparison_key(key, suffix))
