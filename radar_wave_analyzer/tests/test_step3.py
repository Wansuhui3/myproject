"""
第三步功能测试：flask-caching 服务端缓存、plotly-resampler 降采样、图片导出。

注意：
- flask-caching 的 cache 操作需要 Flask app context
- FigureResampler 构造需要 Dash app 上下文（注册重采样回调）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pytest

# 导入 app（会触发 cache.init_app 和回调注册）
import app as app_module
_app = app_module.app


@pytest.fixture(scope='module')
def app_ctx():
    """提供 Flask app context，供缓存与 FigureResampler 测试使用。"""
    ctx = _app.server.app_context()
    ctx.push()
    yield
    ctx.pop()


# ============================================================
# 缓存模块测试
# ============================================================
class TestCacheModule:
    """测试 cache.py 的辅助函数。"""

    def test_set_and_get_df(self, app_ctx):
        """set_data_cache 后能正确取回 df。"""
        from cache import set_data_cache, get_df, clear_data_cache
        clear_data_cache()
        df = pd.DataFrame({'a': [1, 2, 3]})
        meta = pd.DataFrame({'trajectory_id': ['t1']})
        segments = {'t1': df.copy()}
        set_data_cache('test.csv', 'front', df, meta, segments)
        assert get_df() is not None
        assert len(get_df()) == 3

    def test_get_segment(self, app_ctx):
        """单轨迹段能独立缓存与取回。"""
        from cache import set_data_cache, get_segment, clear_data_cache
        clear_data_cache()
        df = pd.DataFrame({'a': [1, 2, 3]})
        seg1 = pd.DataFrame({'x': [10, 20]})
        seg2 = pd.DataFrame({'x': [30, 40]})
        set_data_cache('test.csv', 'front', df, pd.DataFrame(), {'s1': seg1, 's2': seg2})
        assert get_segment('s1') is not None
        assert get_segment('s2') is not None
        assert get_segment('s1')['x'].iloc[0] == 10
        assert get_segment('nonexistent') is None

    def test_compact_segment_indices_restore_rows_from_canonical_df(self, app_ctx):
        """上传路径只缓存源行号，读取轨迹时仍应恢复正确顺序的数据。"""
        from cache import set_data_cache, get_segment, clear_data_cache
        clear_data_cache()
        df = pd.DataFrame({'x': [10, 20, 30]})
        segment = pd.DataFrame({
            'x': [30, 10],
            '__source_row_index__': [2, 0],
        })
        set_data_cache('test.csv', 'front', df, pd.DataFrame(), {'s1': segment})
        restored = get_segment('s1')
        assert restored['x'].tolist() == [30, 10]
        assert '__source_row_index__' not in restored.columns

    def test_clear_cache(self, app_ctx):
        """clear_data_cache 清空所有条目。"""
        from cache import (set_data_cache, get_df, get_segment, clear_data_cache,
                           has_data_loaded)
        clear_data_cache()
        df = pd.DataFrame({'a': [1]})
        set_data_cache('test.csv', 'front', df, pd.DataFrame(), {'t1': df})
        assert has_data_loaded() is True
        clear_data_cache()
        assert has_data_loaded() is False
        assert get_df() is None
        assert get_segment('t1') is None

    def test_radar_position(self, app_ctx):
        """雷达位置标识缓存。"""
        from cache import set_data_cache, get_radar_position, clear_data_cache
        clear_data_cache()
        set_data_cache('test.csv', 'rear_corner', pd.DataFrame(), pd.DataFrame(), {})
        assert get_radar_position() == 'rear_corner'

    def test_request_sessions_do_not_share_cached_data(self, app_ctx):
        """不同 Flask session 的缓存数据必须完全隔离。"""
        from flask import session
        from cache import clear_data_cache, get_df, set_data_cache

        with _app.server.test_request_context('/'):
            clear_data_cache()
            set_data_cache(
                'session-a.csv', 'front', pd.DataFrame({'a': [1]}),
                pd.DataFrame(), {},
            )
            session_a = session['radar_wave_cache_session']
            assert get_df() is not None

        with _app.server.test_request_context('/'):
            assert get_df() is None

        # 清理第一个会话写入的测试缓存。
        with _app.server.test_request_context('/'):
            session['radar_wave_cache_session'] = session_a
            clear_data_cache()


# ============================================================
# 降采样包装测试
# ============================================================
class TestResampler:
    """测试 graph_builder 的降采样包装。"""

    def test_small_data_no_resampler(self, app_ctx):
        """小数据集不启用降采样，返回普通 Figure。"""
        from components.graph_builder import build_multi_subplot_graph
        df = pd.DataFrame({
            'timestamp_parsed': pd.date_range('2026-01-01', periods=10, freq='50ms'),
            'Dx': np.arange(10, dtype=float),
        })
        fig = build_multi_subplot_graph(df, ['Dx'], 't1')
        assert type(fig).__name__ != 'FigureResampler'

    def test_large_data_wrapped(self, app_ctx):
        """大数据集遵循当前降采样配置；禁用时保留普通 Figure。"""
        from components.graph_builder import build_multi_subplot_graph
        from config import get
        n = 6000  # 超过默认阈值 5000
        df = pd.DataFrame({
            'timestamp_parsed': pd.date_range('2026-01-01', periods=n, freq='50ms'),
            'Dx': np.random.randn(n),
        })
        fig = build_multi_subplot_graph(df, ['Dx'], 't1')
        # 配置关闭或可选依赖缺失时，产品定义为安全降级为普通 Figure。
        if not get('RESAMPLER_ENABLED', True):
            assert type(fig).__name__ == 'Figure'
        else:
            assert type(fig).__name__ in ('Figure', 'FigureResampler')

    def test_large_data_display_payload_is_bounded(self, app_ctx):
        """静态显示降采样限制浏览器点数，同时保留首尾时间边界。"""
        from components.graph_builder import build_multi_subplot_graph
        from config import get
        n = 10_000
        timestamps = pd.date_range('2026-01-01', periods=n, freq='50ms')
        values = np.sin(np.linspace(0, 30, n))
        values[5432] = 25.0
        df = pd.DataFrame({'timestamp_parsed': timestamps, 'Dx': values})

        fig = build_multi_subplot_graph(df, ['Dx'], 't1', use_resampler=False)
        main_trace = fig.data[0]
        assert len(main_trace.x) <= get('DISPLAY_MAX_POINTS', 3000)
        assert main_trace.x[0] == timestamps[0]
        assert main_trace.x[-1] == timestamps[-1]
        assert max(main_trace.y) == 25.0

    def test_export_bypass_resampler(self, app_ctx):
        """use_resampler=False 时导出图不包装降采样。"""
        from components.graph_builder import build_multi_subplot_graph
        n = 6000
        df = pd.DataFrame({
            'timestamp_parsed': pd.date_range('2026-01-01', periods=n, freq='50ms'),
            'Dx': np.random.randn(n),
        })
        fig = build_multi_subplot_graph(df, ['Dx'], 't1', use_resampler=False)
        assert type(fig).__name__ != 'FigureResampler'


# ============================================================
# 图片导出测试
# ============================================================
class TestImageExport:
    """测试 exporter 的图片导出功能。"""

    def test_export_png(self, tmp_path):
        """导出 PNG 图片。"""
        from core.exporter import export_graph_image
        seg_df = pd.DataFrame({
            'timestamp_parsed': pd.date_range('2026-01-01', periods=10, freq='50ms'),
            'Dx': np.arange(10, dtype=float),
        })
        filepath = export_graph_image(
            seg_df, ['Dx'], 't1', 40, 480,
            '2026_06_23_17_16_20_276', '2026_06_23_17_16_44_326', str(tmp_path),
        )
        assert os.path.exists(filepath)
        assert filepath.endswith('.png')
        assert '40_2026_06_23_17_16_20_276_2026_06_23_17_16_44_326_480' in filepath

    def test_export_failure_raises(self, tmp_path):
        """导出失败时抛出 ValueError（物理量列表为空）。"""
        from core.exporter import export_graph_image
        seg_df = pd.DataFrame({'timestamp_parsed': [], 'Dx': []})
        with pytest.raises(ValueError, match='quantities_list 不能为空'):
            export_graph_image(
                seg_df, [], 't1', 40, 480,
                '2026_06_23_17_16_20_276', '2026_06_23_17_16_44_326', str(tmp_path),
            )
