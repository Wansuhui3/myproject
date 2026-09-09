"""导出文件命名辅助函数测试：选中段来源雷达文件名定位。"""
from radar_wave_analyzer.callbacks import _selected_radar_filename  # noqa: E402


def _state(candidates, file_index=1, track_id=50, seg_idx=0, meta_name='5个文件'):
    return {
        'cached_id_list': candidates,
        'selected_file_index': file_index,
        'selected_id': track_id,
        'selected_segment_index': seg_idx,
        'radar_meta': {'filename': meta_name},
    }


CANDIDATES = [
    {'file_index': 0, 'track_id': 2, 'segment_index': 0,
     'radar_filename': 'CD701_flr_track_2026_08_11_15_57_24.csv'},
    {'file_index': 1, 'track_id': 50, 'segment_index': 0,
     'radar_filename': 'CD701_rlr_track_2026_08_11_16_44_13.csv'},
    {'file_index': 1, 'track_id': 50, 'segment_index': 1,
     'radar_filename': 'CD701_rlr_track_2026_08_11_17_30_05.csv'},
]


def test_multi_file_upload_uses_selected_segment_source_filename():
    """多文件上传：按 file_index+ID+段 定位到选中段实际来源的雷达文件名。"""
    name = _selected_radar_filename(_state(CANDIDATES))
    assert name == 'CD701_rlr_track_2026_08_11_16_44_13.csv'


def test_segment_index_disambiguates_same_id():
    """同文件同 ID 多段：按段号区分。"""
    name = _selected_radar_filename(_state(CANDIDATES, seg_idx=1))
    assert name == 'CD701_rlr_track_2026_08_11_17_30_05.csv'


def test_fallback_to_upload_batch_label_when_no_candidate():
    """候选缓存无匹配记录时回退上传批次标签。"""
    name = _selected_radar_filename(_state([], file_index=None))
    assert name == '5个文件'
    # file_index 有值但候选列表为空 → 同样回退
    name = _selected_radar_filename(_state([], file_index=3))
    assert name == '5个文件'


def test_candidate_without_filename_falls_back():
    """候选缺 radar_filename 字段时回退批次标签。"""
    candidates = [{'file_index': 1, 'track_id': 50, 'segment_index': 0}]
    name = _selected_radar_filename(_state(candidates))
    assert name == '5个文件'
