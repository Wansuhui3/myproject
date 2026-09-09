"""对比文件的稳定显示名称与自然排序辅助函数。"""
import os
import re


_TRACK_FILENAME_PATTERN = re.compile(
    r'^(?P<device>.+?)_(?P<source>flr|rlr)_track_'
    r'(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<day>\d{1,2})_'
    r'(?P<hour>\d{1,2})_(?P<minute>\d{1,2})_(?P<second>\d{1,2})'
    r'(?:_(?P<millisecond>\d{1,3}))?$',
    flags=re.IGNORECASE,
)


def natural_filename_key(filename: str | None) -> tuple:
    """生成不混淆数字大小的文件名排序键（file2 在 file10 前）。"""
    basename = os.path.basename(str(filename or '')).strip().casefold()
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r'(\d+)', basename)
        if part
    )


def middle_ellipsis(text: str, max_length: int = 34) -> str:
    """保留文件名前后两端，避免公共长前缀隐藏真正的差异。"""
    if len(text) <= max_length:
        return text
    left_length = max(8, (max_length - 1) // 2)
    right_length = max_length - left_length - 1
    return f'{text[:left_length]}…{text[-right_length:]}'


def compact_filename_label(filename: str | None, max_length: int = 34) -> str:
    """将标准雷达长文件名压缩为“设备 · 月-日 时:分:秒”。"""
    basename = os.path.basename(str(filename or '')).strip()
    if not basename:
        return '来源文件未知'
    stem, _extension = os.path.splitext(basename)
    match = _TRACK_FILENAME_PATTERN.match(stem)
    if not match:
        return middle_ellipsis(basename, max_length)
    values = match.groupdict()
    return (
        f'{values["device"]} · '
        f'{int(values["month"]):02d}-{int(values["day"]):02d} '
        f'{int(values["hour"]):02d}:{int(values["minute"]):02d}:'
        f'{int(values["second"]):02d}'
    )
