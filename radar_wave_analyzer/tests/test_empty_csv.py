"""空白 CSV 上传回归测试。

背景 bug：仅表头（或全空行）的 CSV 会解析出“0 行但列齐全”的 DataFrame，
上传回调此前在该 df 上执行 iloc[0] 抛 IndexError，导致回调 500、前端一直
显示加载中。契约：解析不抛异常，返回 0 行 DataFrame，由回调层跳过。
"""
import pandas as pd
import pytest

from radar_wave_analyzer.core.data_loader import load_csv_from_bytes  # noqa: E402

HEADER_ONLY_CSV = b'ID,timestamp,Track_Age,Dx,Dy,Vx,Vy\r\n'
ALL_EMPTY_ROWS_CSV = b'ID,timestamp,Track_Age,Dx,Dy,Vx,Vy\r\n,,,\r\n,,,\r\n'


def test_header_only_csv_returns_zero_row_dataframe():
    """仅表头的 CSV：返回 0 行但列齐全的 DataFrame，不抛异常。"""
    df = load_csv_from_bytes(HEADER_ONLY_CSV, 'empty.csv')
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert 'timestamp_parsed' in df.columns


def test_all_empty_rows_csv_returns_zero_row_dataframe():
    """全部空值行的 CSV：剔除后 0 行，不抛异常。"""
    df = load_csv_from_bytes(ALL_EMPTY_ROWS_CSV, 'blank-rows.csv')
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert 'timestamp_parsed' in df.columns


def test_completely_empty_csv_raises_value_error():
    """0 字节文件：缺少必要字段，抛 ValueError（由上传回调捕获记入错误）。"""
    with pytest.raises(ValueError):
        load_csv_from_bytes(b'', 'zero-byte.csv')
