"""上传文件类型校验与加载遮罩信号（upload-tick）测试。

覆盖“上传错误文件后一直转圈”的回归场景：不支持的扩展名必须在回调入口
被拒绝，并且无论成功/失败都要写入新的 upload-tick，供前端解除遮罩。
"""

from radar_wave_analyzer.callbacks.helpers import is_supported_upload


def test_is_supported_upload_accepts_csv_and_json():
    """仅 .csv / .json 合法，大小写与前后空白不敏感。"""
    assert is_supported_upload('a.csv')
    assert is_supported_upload('A.CSV')
    assert is_supported_upload('data.json')
    assert is_supported_upload('  b.JSON ')


def test_is_supported_upload_rejects_other_types():
    for name in ('notes.txt', 'book.xlsx', 'archive.zip', 'noext', '', None):
        assert not is_supported_upload(name), name


def test_unsupported_file_type_stops_loading_with_clear_error():
    """上传不支持的类型：立即返回明确错误，并写入新的 upload-tick。"""
    from radar_wave_analyzer.callbacks.wave_upload_callbacks import on_upload_csv

    result = on_upload_csv(
        ['data:text/plain;base64,AAAA'], ['notes.txt'], 'default')

    assert result[0] is False
    feedback = str(result[1])
    assert '不支持的文件类型' in feedback
    assert 'notes.txt' in feedback
    # 最后一个输出为 upload-tick（浮点时间戳），前端据此关闭遮罩
    assert isinstance(result[-1], float)


def test_broken_csv_reports_error_without_raising():
    """扩展名合法但内容无法解析时，返回失败提示而非抛出异常。"""
    from radar_wave_analyzer.callbacks.wave_upload_callbacks import on_upload_csv

    result = on_upload_csv(
        ['data:text/csv;base64,AAAA'], ['broken.csv'], 'default')

    assert result[0] is False
    assert '失败' in str(result[1])
    assert isinstance(result[-1], float)


def test_upload_tick_changes_between_calls():
    """每次上传回调都产生不同的 tick，保证遮罩信号必然变化。"""
    from radar_wave_analyzer.callbacks.wave_upload_callbacks import on_upload_csv

    first = on_upload_csv(
        ['data:text/plain;base64,AAAA'], ['a.txt'], 'default')
    second = on_upload_csv(
        ['data:text/plain;base64,AAAA'], ['a.txt'], 'default')

    assert first[-1] != second[-1]
