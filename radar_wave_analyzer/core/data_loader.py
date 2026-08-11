"""
CSV 数据加载、时间戳解析、预处理模块。
"""
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

try:
    from ..config import get
except ImportError:
    from config import get  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# 必需的 CSV 字段
_REQUIRED_COLUMNS = ['timestamp', 'ID', 'Track_Age']


def identify_radar_source(filename: str) -> dict[str, object]:
    """根据配置的文件名规则识别雷达来源。"""
    basename = os.path.basename(str(filename or '')).strip()
    rules = get('RADAR_FILE_PATTERNS', {}) or {}
    for source_key, source_config in rules.items():
        patterns = source_config.get('patterns', []) if isinstance(source_config, dict) else []
        for pattern in patterns:
            try:
                if re.search(str(pattern), basename, flags=re.IGNORECASE):
                    return {
                        'key': str(source_key),
                        'label': str(source_config.get('label', source_key)),
                        'short_label': str(source_config.get(
                            'short_label', source_config.get('label', source_key),
                        )),
                        'recognized': True,
                        'filename': basename,
                    }
            except re.error as exc:
                logger.warning('忽略无效雷达文件名规则 %r: %s', pattern, exc)

    unknown_label = str(get('UNKNOWN_RADAR_SOURCE_LABEL', '未识别雷达'))
    return {
        'key': 'unknown',
        'label': unknown_label,
        'short_label': '未识别',
        'recognized': False,
        'filename': basename,
    }


def parse_timestamp(ts_str: str) -> pd.Timestamp:
    """
    唯一的时间戳解析函数。
    支持多种输入格式（按尝试顺序）：
      - YYYY_MM_DD_HH_MM_SS_mmm（CSV 原始格式，完整 7 段）
      - YYYY_MM_DD_HH_MM_SS（CSV 原始格式，6 段）
      - YYYY_MM_DD_HH_MM / YYYY_MM_DD_HH / YYYY_MM_DD（部分下划线格式）
      - YYYY-MM-DD HH:MM:SS（用户输入）
      - YYYY-MM-DD HH:MM:SS.%f（含毫秒）
      - 部分输入如 YYYY-MM-DD / HH:MM:SS 等（由 pd.Timestamp 兜底）

    Args:
        ts_str: 时间戳字符串。

    Returns:
        pd.Timestamp 对象。

    Raises:
        ValueError: 所有格式均无法解析时抛出。
    """
    if not isinstance(ts_str, str):
        ts_str = str(ts_str)

    ts_str = ts_str.strip()

    # ---- 下划线格式分支 ----
    if '_' in ts_str:
        try:
            parts = ts_str.split('_')
            if len(parts) >= 7:
                # 完整格式：YYYY_MM_DD_HH_MM_SS_mmm
                ms = parts[6].ljust(3, '0')[:3]
                formatted = f'{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}.{ms}'
                return pd.Timestamp(formatted)
            if len(parts) == 6:
                # 6 段格式：YYYY_MM_DD_HH_MM_SS
                formatted = f'{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}'
                return pd.Timestamp(formatted)
            if 3 <= len(parts) <= 5:
                # 部分下划线格式，按顺序补齐缺失的时间分量
                # parts: [Y, M, D, (H), (M), (S)]
                defaults = ['1970', '01', '01', '00', '00', '00']
                for i in range(min(len(parts), 6)):
                    if i == 0:
                        defaults[i] = parts[i].zfill(4)  # 年份补齐 4 位
                    else:
                        defaults[i] = parts[i].zfill(2)
                formatted = (
                    f'{defaults[0]}-{defaults[1]}-{defaults[2]} '
                    f'{defaults[3]}:{defaults[4]}:{defaults[5]}'
                )
                try:
                    return pd.Timestamp(formatted)
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError):
            pass

    # ---- 标准格式分支（pd.Timestamp 兜底） ----
    try:
        return pd.Timestamp(ts_str)
    except (ValueError, TypeError):
        pass

    raise ValueError(f'无法解析时间戳: {ts_str}')


def parse_timestamp_series(values: pd.Series) -> pd.Series:
    """批量解析时间戳，保留 ``parse_timestamp`` 的全部兼容格式。

    雷达 CSV 的主流格式为下划线分隔的完整时间戳，可用指定格式一次转换；
    只有少数用户输入或历史格式才回退到唯一的单值解析器。
    """
    raw = values.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=values.index, dtype='datetime64[ns]')

    # 完整格式允许毫秒位数为 1~6，兼容设备输出的不同精度。
    full_mask = raw.str.fullmatch(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{1,6}')
    if full_mask.any():
        result.loc[full_mask] = pd.to_datetime(
            raw.loc[full_mask], format='%Y_%m_%d_%H_%M_%S_%f', errors='coerce',
        )

    six_part_mask = raw.str.fullmatch(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}')
    if six_part_mask.any():
        result.loc[six_part_mask] = pd.to_datetime(
            raw.loc[six_part_mask], format='%Y_%m_%d_%H_%M_%S', errors='coerce',
        )

    # 无法由主格式处理的值仍使用项目唯一解析器，避免改变既有格式承诺。
    fallback_mask = result.isna()
    if fallback_mask.any():
        result.loc[fallback_mask] = raw.loc[fallback_mask].map(parse_timestamp)

    return result


def _read_and_preprocess_csv(source, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """读取 CSV；分块模式下逐块清洗，最后仍按全局时间排序。"""
    if not chunk_size:
        return _preprocess_csv(pd.read_csv(source, skipinitialspace=True))

    processed_chunks = []
    for raw_chunk in pd.read_csv(source, skipinitialspace=True, chunksize=chunk_size):
        processed_chunks.append(_preprocess_csv(raw_chunk, sort_by_time=False))

    if not processed_chunks:
        return _preprocess_csv(pd.DataFrame())

    merged = pd.concat(processed_chunks, ignore_index=True)
    return _finalize_preprocessed(merged)


def _configured_chunk_size(content_size_bytes: int) -> Optional[int]:
    """仅在文件达到配置阈值时启用分块读取。"""
    threshold_mb = get('CSV_CHUNK_THRESHOLD_MB', 200)
    if content_size_bytes < threshold_mb * 1024 * 1024:
        return None
    return get('CSV_CHUNK_SIZE_ROWS', 100_000)


def load_csv(file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    加载雷达 CSV 文件并执行预处理。

    预处理顺序：
      加载 → 剔除空值/NaN → Track_Age 类型校验(0~255)
      → 时间戳解析 → 全局按时间排序

    Args:
        file_path: CSV 文件路径。
        chunk_size: 可选分块行数；省略时按文件大小和配置自动决定。

    Returns:
        预处理后的 DataFrame。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺少必要字段。
    """
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'数据文件不存在: {file_path}')

    actual_chunk_size = chunk_size or _configured_chunk_size(os.path.getsize(file_path))
    return _read_and_preprocess_csv(file_path, actual_chunk_size)


def load_csv_from_bytes(
    content_bytes: bytes,
    filename: str = '',
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    从字节流加载雷达 CSV 并执行预处理（用于 dcc.Upload 拖拽上传）。

    Args:
        content_bytes: CSV 文件字节内容。
        filename: 文件名（仅用于日志/错误提示）。
        chunk_size: 可选分块行数；省略时按内容大小和配置自动决定。

    Returns:
        预处理后的 DataFrame。

    Raises:
        ValueError: 缺少必要字段或内容无法解析。
    """
    import io
    actual_chunk_size = chunk_size or _configured_chunk_size(len(content_bytes))
    df = _read_and_preprocess_csv(io.BytesIO(content_bytes), actual_chunk_size)
    logger.info(f'从内存加载: {filename or "<upload>"}')
    return df


def _preprocess_csv(df: pd.DataFrame, sort_by_time: bool = True) -> pd.DataFrame:
    """
    CSV 预处理核心逻辑：校验 → 清洗 → 排序。
    load_csv 和 load_csv_from_bytes 共用此函数。
    """

    # 检查必要字段
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f'数据文件缺少必要字段 {missing_cols}，请检查文件格式')

    # 剔除空值/NaN
    before = len(df)
    df = df.dropna(subset=_REQUIRED_COLUMNS)
    if len(df) < before:
        logger.info(f'剔除 {before - len(df)} 行空值/NaN 数据')

    # Track_Age 必须是 0~255 的整数。不能直接 astype(int)：它会把 12.7
    # 静默截断为 12，使非法帧参与分段并改变轨迹边界。
    numeric_age = pd.to_numeric(df['Track_Age'], errors='coerce')
    finite_age = numeric_age.notna() & np.isfinite(numeric_age)
    integer_age = finite_age & numeric_age.eq(np.floor(numeric_age))
    in_range_age = integer_age & numeric_age.between(0, 255)
    invalid_mask = ~in_range_age

    if invalid_mask.any():
        logger.warning(f'标记 {invalid_mask.sum()} 行 Track_Age 异常数据，将跳过')
        df = df[~invalid_mask].copy()

    # 在完成校验后再转换，保证 Track_Age 的语义始终是 uint8 整数。
    df['Track_Age'] = numeric_age.loc[df.index].astype('int64')

    # 时间戳解析：主流设备格式走向量化路径，少量兼容格式自动回退。
    df['timestamp_parsed'] = parse_timestamp_series(df['timestamp'])

    if sort_by_time:
        return _finalize_preprocessed(df)

    return df


def _finalize_preprocessed(df: pd.DataFrame) -> pd.DataFrame:
    """在所有数据块清洗完成后执行唯一的全局排序与加载日志。"""
    # 全局按时间排序（不能先分组再排序）
    df = df.sort_values('timestamp_parsed').reset_index(drop=True)
    logger.info(f'加载完成: {len(df)} 行有效数据, {df["ID"].nunique()} 个唯一 ID')
    return df


def get_time_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    获取数据的时间范围。

    Returns:
        (最早时间, 最晚时间)。
    """
    return df['timestamp_parsed'].min(), df['timestamp_parsed'].max()


def filter_by_time_window(
    df: pd.DataFrame,
    center_ts: pd.Timestamp,
    window_sec: float = 30.0,
) -> pd.DataFrame:
    """
    以指定时间戳为中心，筛选前后 window_sec 秒窗口内的数据。

    Args:
        df: 预处理后的 DataFrame。
        center_ts: 中心时间戳。
        window_sec: 窗口半径（秒）。

    Returns:
        窗口内的数据子集。
    """
    start = center_ts - pd.Timedelta(seconds=window_sec)
    end = center_ts + pd.Timedelta(seconds=window_sec)
    result = df[(df['timestamp_parsed'] >= start) & (df['timestamp_parsed'] <= end)].copy()
    logger.info(
        f'时间窗口: center={center_ts}, window=±{window_sec}s, '
        f'range=[{start}, {end}], matched={len(result)} rows, '
        f'full_df={len(df)} rows, t_min={df["timestamp_parsed"].min()}, '
        f't_max={df["timestamp_parsed"].max()}'
    )
    return result
