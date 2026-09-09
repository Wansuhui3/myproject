"""对比图表时间格式化工具。

将 epoch 秒按本地时区格式化，与 Plotly 日期横轴在 WebView/浏览器中的
显示口径保持一致（悬浮框时间与横轴必须同源，避免相差数小时）。
"""
from datetime import datetime, timedelta, timezone

import numpy as np


def _fmt_ts(epoch_sec):
    """按本地时区格式化 epoch 秒数，与 Plotly 日期横轴保持一致。"""
    if epoch_sec is None or (isinstance(epoch_sec, float) and np.isnan(epoch_sec)):
        return 'N/A'
    # comparison.parser 将 CSV 墙钟时间换算为真实 epoch 秒。Plotly 日期轴在
    # WebView/浏览器中按本地时区显示，悬浮框必须使用同一口径，避免相差 8 小时。
    # Windows CRT 的 localtime 不支持 1970-01-01 00:00 UTC 之前的时间戳
    # （负值抛 OSError [Errno 22]），先做 UTC 纯算术再取本地时区；测试用的
    # 合成时间戳可能落在 epoch 附近，此时以当前本地偏移兜底（真实数据为
    # 近期日期，不走此分支）。
    dt_utc = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(epoch_sec))
    try:
        dt = dt_utc.astimezone()
    except OSError:
        dt = dt_utc + datetime.now().astimezone().utcoffset()
    ms = dt.microsecond // 1000
    return f'{dt.year:04d}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:03d}'


def _fmt_clock(timestamp_label) -> str:
    """从统一时间标签中取 ``HH:mm:ss.SSS``，用于紧凑的图表悬浮框。"""
    return str(timestamp_label).strip().rsplit(' ', 1)[-1]


# 向量化版本，用于处理 numpy 数组
_fmt_ts_vec = np.vectorize(_fmt_ts)
