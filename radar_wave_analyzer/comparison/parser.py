"""
文件解析与角色识别模块。
自动识别雷达CSV/JSON和RTK真值CSV，解析时间戳，校验数据完整性。
"""
import io
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.data_loader import parse_epoch_series
from ..core.json_loader import is_json_filename, parse_json_topics

# 时间戳正则：YYYY_MM_DD_HH_MM_SS_mmm 或 YYYY_MM_DD_HH_MM_SS
_RE_TS_7 = re.compile(r'^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})$')
_RE_TS_6 = re.compile(r'^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})$')

# 雷达特征列
_RADAR_COLS = {'Dx', 'Dy'}
# RTK 特征列
_RTK_COLS = {'center_x', 'center_y'}

_REQUIRED_COLUMNS = {
    'radar': ['timestamp', 'ID', 'Track_Age', 'Dx', 'Dy', 'Vx', 'Vy'],
    'rtk': ['timestamp', 'ID', 'center_x', 'center_y', 'Vx', 'Vy'],
    'unknown': ['timestamp'],
}

_NUMERIC_COLUMNS = {
    'radar': ['Dx', 'Dy', 'Vx', 'Vy', 'Rx_front', 'Rx_rear', 'Ry'],
    'rtk': ['center_x', 'center_y', 'Vx', 'Vy'],
    'unknown': [],
}

_SYSTEM_COLUMNS = {'timestamp', 'ID', 'Track_Age'}


def _infer_field_unit(field_name: str, internal_name: str) -> str:
    """根据原始/标准字段名推测单位；无法可靠判断时返回空字符串。"""
    token = internal_name.lower().replace(' ', '').replace('_', '')
    original = field_name.lower().replace(' ', '').replace('_', '')
    if token in {'dx', 'dy', 'centerx', 'centery', 'rxfront', 'rxrear', 'ry'}:
        return 'm'
    if token in {'vx', 'vy', 'vabs'} or 'speed' in original or 'velocity' in original:
        return 'm/s'
    if token in {'ax', 'ay'} or original in {'acceleration', 'accel'}:
        return 'm/s²'
    if 'angle' in token or 'heading' in token or 'yaw' in token:
        return '°'
    return ''


def _build_physical_fields(
    df: pd.DataFrame,
    original_fields: list[str],
    original_to_internal: dict[str, str],
) -> list[dict]:
    """生成保持 CSV 原名的可选数值物理量元数据。"""
    fields = []
    for field in original_fields:
        internal = original_to_internal.get(field, field)
        if internal in _SYSTEM_COLUMNS or field not in df.columns:
            continue
        numeric = pd.to_numeric(df[field], errors='coerce')
        valid_count = int(np.isfinite(numeric).sum())
        if valid_count == 0:
            continue
        fields.append({
            'name': field,
            'internal_name': internal,
            'unit': _infer_field_unit(field, internal),
            'valid_count': valid_count,
            'valid_ratio': round(valid_count / len(df), 4) if len(df) else 0.0,
        })
    return fields


def detect_file_role(df: pd.DataFrame) -> str:
    """自动识别CSV数据角色。

    检测顺序说明：RTK列(center_x/center_y)更特异，优先检查，
    避免包含 Dx/Dy 的 RTK 数据被误判为雷达。

    - 包含 center_x,center_y 列 → 'rtk'（优先）
    - 包含 Dx,Dy 列           → 'radar'
    - 都不满足                 → 'unknown'
    """
    cols = set(df.columns)
    # RTK 特征列优先，避免被雷达列误判（某些 RTK 数据也含有 Dx/Dy）
    if _RTK_COLS.issubset(cols):
        return 'rtk'
    if _RADAR_COLS.issubset(cols):
        return 'radar'
    return 'unknown'


def _wallclock_tz():
    """下划线墙钟时间戳所属时区：系统本地时区。

    设备 CSV 的时间戳是本地墙钟（如北京时间 16:42:13），而录制 JSON 的
    epoch 纳秒是真实 UTC。若把墙钟直接当 UTC 相减，两类文件相差整个时区
    偏移（北京时间即 8 小时），时间零重叠导致目标关联失败。CSV↔CSV 比较
    时双方同步折算，相对时间关系不受影响。
    """
    return datetime.now().astimezone().tzinfo


def _wallclock_epoch(dt: datetime) -> float:
    """naive 墙钟 datetime → 真实 UTC epoch 秒（按系统时区折算）。"""
    return dt.replace(tzinfo=_wallclock_tz()).timestamp()


def _naive_to_epoch_seconds(timestamps: pd.Series) -> pd.Series:
    """naive 墙钟时间序列 → 真实 UTC epoch 秒（向量化路径）。"""
    tz = _wallclock_tz()
    localized = timestamps.dt.tz_localize(
        tz, ambiguous=True, nonexistent='shift_forward',
    )
    return (localized - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta(seconds=1)


def parse_timestamp(ts_str: str) -> float:
    """解析下划线分隔时间戳字符串，返回真实 UTC epoch 秒。

    支持格式:
      YYYY_MM_DD_HH_MM_SS_mmm (7段，含毫秒)
      YYYY_MM_DD_HH_MM_SS     (6段，无毫秒)

    注意: 实测发现时间戳字段末尾有空格，先 strip() 处理。
    """
    s = str(ts_str).strip()
    m = _RE_TS_7.match(s)
    if m:
        parts = [int(x) for x in m.groups()]
        dt = datetime(parts[0], parts[1], parts[2],
                      parts[3], parts[4], parts[5])
        return _wallclock_epoch(dt) + parts[6] / 1000.0
    m = _RE_TS_6.match(s)
    if m:
        parts = [int(x) for x in m.groups()]
        return _wallclock_epoch(datetime(*parts))
    # 兜底：尝试 pandas 解析
    try:
        ts = pd.Timestamp(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize(
                _wallclock_tz(), ambiguous=True, nonexistent='shift_forward',
            )
        return ts.timestamp()
    except Exception:
        raise ValueError(f'无法解析时间戳: {ts_str!r}')


def parse_timestamp_series(values: pd.Series) -> pd.Series:
    """批量解析时间戳并返回真实 UTC epoch 秒。

    设备主流的 6/7 段下划线格式走 Pandas 向量化路径；只有少量历史格式
    回退到单值解析器，避免大文件逐行执行 Python ``apply``。
    """
    raw = values.astype(str).str.strip()
    parsed = pd.Series(np.nan, index=values.index, dtype='float64')

    seven_mask = raw.str.fullmatch(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{3}')
    if seven_mask.any():
        timestamps = pd.to_datetime(
            raw.loc[seven_mask], format='%Y_%m_%d_%H_%M_%S_%f', errors='coerce',
        )
        parsed.loc[timestamps.index] = _naive_to_epoch_seconds(timestamps)

    six_mask = raw.str.fullmatch(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}')
    if six_mask.any():
        timestamps = pd.to_datetime(
            raw.loc[six_mask], format='%Y_%m_%d_%H_%M_%S', errors='coerce',
        )
        parsed.loc[timestamps.index] = _naive_to_epoch_seconds(timestamps)

    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = raw.loc[fallback_mask].map(parse_timestamp)
    return parsed


def _compute_sample_rate(timestamps_sec: np.ndarray) -> float:
    """根据时间戳序列估算采样率(Hz)。"""
    unique_timestamps = np.sort(np.unique(timestamps_sec))
    if len(unique_timestamps) < 2:
        return 0.0
    diffs = np.diff(unique_timestamps)
    median_diff = np.median(diffs)
    if median_diff <= 0:
        return 0.0
    return round(1.0 / median_diff, 1)


def _make_error_result(filename: str, errors: list, warnings: list = None) -> dict:
    """构建错误结果字典。

    errors:  致命错误（如缺少必要列、编码失败），阻止后续处理
    warnings: 非致命提示（如部分行被剔除），不阻止处理流程
    """
    return {
        'role': 'unknown',
        'df': pd.DataFrame(),
        'filename': filename or '',
        'total_rows': 0,
        'unique_timestamps': 0,
        'unique_ids': 0,
        'time_range': (0, 0),
        'sample_rate_hz': 0,
        'fields': [],
        'errors': errors,
        'warnings': warnings or [],
    }


def load_csv_file(file_bytes: bytes, filename: str) -> dict:
    """加载并解析单个CSV文件。

    Args:
        file_bytes: CSV文件原始字节。
        filename: 原始文件名。

    Returns:
        dict:
            role:           'radar' | 'rtk' | 'unknown'
            df:             DataFrame（已解析时间戳，已排序）
            filename:       文件名
            total_rows:     总行数
            unique_timestamps: 唯一时间戳数
            unique_ids:     不同 ID 数
            time_range:     (start_epoch, end_epoch)
            sample_rate_hz: 采样率
            fields:         可用字段列表
            errors:         校验错误列表
    """
    errors = []    # 致命错误：缺少必要列、编码/解析失败
    warnings = []  # 非致命提示：部分行被剔除、时间戳解析降级等
    # 列名映射表（不区分大小写 + 常见变体）
    col_map = {
        'timestamp': 'timestamp',
        'Timestamp': 'timestamp',
        'TIMESTAMP': 'timestamp',
        'ts': 'timestamp',
        'TS': 'timestamp',
        'time': 'timestamp',
        'Time': 'timestamp',
        'ID': 'ID',
        'id': 'ID',
        'Id': 'ID',
        'track_id': 'ID',
        'Track ID': 'ID',
        'Track_Age': 'Track_Age',
        'Track Age': 'Track_Age',
        'track_age': 'Track_Age',
        'track age': 'Track_Age',
        'TRACK_AGE': 'Track_Age',
        'Dx': 'Dx',
        'dx': 'Dx',
        'DX': 'Dx',
        'Dy': 'Dy',
        'dy': 'Dy',
        'DY': 'Dy',
        'Vx': 'Vx',
        'vx': 'Vx',
        'VX': 'Vx',
        'Vy': 'Vy',
        'vy': 'Vy',
        'VY': 'Vy',
        'Rx_front': 'Rx_front',
        'RX_front': 'Rx_front',
        'rx_front': 'Rx_front',
        'Rx_rear': 'Rx_rear',
        'RX_rear': 'Rx_rear',
        'rx_rear': 'Rx_rear',
        'Ry': 'Ry',
        'RY': 'Ry',
        'ry': 'Ry',
        'center_x': 'center_x',
        'Center_X': 'center_x',
        'CENTER_X': 'center_x',
        'centerX': 'center_x',
        'cx': 'center_x',
        'CX': 'center_x',
        'pos_x': 'center_x',
        'Pos_X': 'center_x',
        'center_y': 'center_y',
        'Center_Y': 'center_y',
        'CENTER_Y': 'center_y',
        'centerY': 'center_y',
        'cy': 'center_y',
        'CY': 'center_y',
        'pos_y': 'center_y',
        'Pos_Y': 'center_y',
    }

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
    except (UnicodeDecodeError, pd.errors.ParserError):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='gbk')
        except (UnicodeDecodeError, pd.errors.ParserError):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8-sig')
            except Exception:
                errors.append('无法读取CSV文件，请检查编码格式是否为 UTF-8 或 GBK')
                return _make_error_result(filename, errors)

    # 保留 CSV 原始列名，并额外创建算法所需的标准别名。界面、图例和自定义
    # 映射使用原始列名；目标发现、坐标诊断等既有算法继续使用标准字段。
    df = df.rename(columns=lambda c: str(c).strip())
    original_fields = list(df.columns)
    original_to_internal = {field: col_map.get(field, field) for field in original_fields}
    for original, internal in original_to_internal.items():
        if original != internal and internal not in df.columns:
            df[internal] = df[original]

    try:
        timestamps_sec = parse_timestamp_series(df['timestamp'])
    except Exception as e:
        errors.append(f'时间戳解析失败: {e}')
        timestamps_sec = pd.Series(0.0, index=df.index)

    return _finalize_role_result(
        df, filename, errors, warnings, original_fields, original_to_internal, timestamps_sec,
    )


def _finalize_role_result(
    df: pd.DataFrame,
    filename: str,
    errors: list[str],
    warnings: list[str],
    original_fields: list[str],
    original_to_internal: dict[str, str],
    timestamps_sec: pd.Series,
) -> dict:
    """角色检测、必需列校验、数值清洗、排序与指标组装（CSV/JSON 共用后处理）。

    timestamps_sec 为已换算的 epoch 秒序列（与对比模块时间口径一致）。
    """
    role = detect_file_role(df)
    fields = original_fields

    # 对齐算法会直接使用位置与速度字段，必须在上传阶段给出中文错误，
    # 而不是在后续 NumPy 插值时暴露 KeyError/类型错误。
    for col in _REQUIRED_COLUMNS[role]:
        if col not in df.columns:
            errors.append(f'缺少必要列: {col}')

    if not errors:
        required_cols = _REQUIRED_COLUMNS[role]
        before = len(df)
        df = df.dropna(subset=required_cols)
        if len(df) < before:
            warnings.append(f'已剔除 {before - len(df)} 行（必要列为空）')

        # Rx_front/Rx_rear/Ry 等测距字段是可选列：存在时参与数值清洗，
        # 缺失时不能让一份满足 REQUIRED_COLUMNS 的合法文件上传失败。
        numeric_cols = [column for column in _NUMERIC_COLUMNS[role] if column in df.columns]
        if numeric_cols:
            numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            valid_numeric = np.isfinite(numeric_df).all(axis=1)
            invalid_count = int((~valid_numeric).sum())
            if invalid_count:
                warnings.append(f'已剔除 {invalid_count} 行（位置或速度字段不是有限数值）')
            # 逐列替换允许 Pandas 将 StringDtype 升级为浮点列；对整块
            # ``df.loc[:, cols]`` 赋值会在新版 Pandas 中因 dtype 不兼容失败。
            for col in numeric_cols:
                df[col] = numeric_df[col]
            df = df.loc[valid_numeric].copy()

        if len(df) == 0:
            errors.append('没有可用于对齐的有效数据行')

    if not errors and timestamps_sec.isna().all():
        errors.append('时间戳解析失败: 所有时间戳均为无效数值')
        timestamps_sec = pd.Series(0.0, index=df.index)

    df['timestamp_parsed'] = timestamps_sec.reindex(df.index)

    if not errors and len(df) > 0:
        # 在排序前保留原始行号（1-based），供后续帧号标注使用
        df['csv_row'] = range(1, len(df) + 1)
        df = df.sort_values('timestamp_parsed').reset_index(drop=True)

    physical_fields = _build_physical_fields(df, original_fields, original_to_internal)

    ts_vals = df['timestamp_parsed'].values

    return {
        'role': role,
        'df': df,
        'filename': filename if filename else '',
        'total_rows': len(df),
        'unique_timestamps': len(np.unique(ts_vals)) if len(df) > 0 else 0,
        'unique_ids': df['ID'].nunique() if 'ID' in df.columns and len(df) > 0 else 0,
        'time_range': (ts_vals.min(), ts_vals.max()) if len(df) > 0 else (0, 0),
        'sample_rate_hz': _compute_sample_rate(ts_vals),
        'fields': fields,
        'physical_fields': physical_fields,
        'original_to_internal': original_to_internal,
        'errors': errors,
        'warnings': warnings,
    }


def _epoch_seconds(values: pd.Series) -> pd.Series:
    """数值 epoch（JSON）→ epoch 秒，复用 data_loader 唯一的量级判单位实现。"""
    timestamps = parse_epoch_series(values)
    return (timestamps - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1)


def _load_json_topics(file_bytes: bytes, filename: str) -> list[dict]:
    """加载目标 JSON：每个话题展开为独立结果（角色检测与 CSV 一致）。

    使用 json_loader 的宽松展开层（仅要求 timestamp），Track_Age 等主页面
    专用必需列不强制——RTK 真值 JSON 同样可经此处解析。列名已按 config
    quantities 规范化（如 Rx_Front → Rx_front），恒等映射即可。
    """
    json_result = parse_json_topics(file_bytes, filename)
    results: list[dict] = []
    for error in json_result['errors']:
        results.append(_make_error_result(filename, [error]))

    for topic_index, entry in enumerate(json_result['topics']):
        df = entry['df'].copy()
        original_fields = list(df.columns)
        original_to_internal = {field: field for field in original_fields}
        timestamps_sec = _epoch_seconds(df['timestamp'])
        # 空话题跳过等文件级提示挂到首个有效话题，避免丢失
        topic_warnings = list(json_result['warnings']) if topic_index == 0 else []
        results.append(_finalize_role_result(
            df, entry['label'], [], topic_warnings, original_fields,
            original_to_internal, timestamps_sec,
        ))
    return results


def load_data_file(file_bytes: bytes, filename: str) -> list[dict]:
    """统一数据文件入口：按扩展名分派 CSV / JSON，返回结果列表。

    CSV 恒返回单元素列表；JSON 每话题一个元素（如 FLR/RLR 共存时拆为两条）。
    """
    if is_json_filename(filename):
        return _load_json_topics(file_bytes, os.path.basename(str(filename or 'upload.json')))
    return [load_csv_file(file_bytes, filename)]


def validate_overlap(radar_info: dict, rtk_info: dict) -> dict:
    """校验两个文件的时间重叠情况。

    Returns:
        dict:
            has_overlap: bool
            overlap_start: float (epoch秒)
            overlap_end: float
            overlap_duration_sec: float
        """
    r_start, r_end = radar_info['time_range']
    g_start, g_end = rtk_info['time_range']

    overlap_start = max(r_start, g_start)
    overlap_end = min(r_end, g_end)

    return {
        'has_overlap': overlap_start < overlap_end,
        'overlap_start': overlap_start,
        'overlap_end': overlap_end,
        'overlap_duration_sec': max(0, overlap_end - overlap_start),
    }
