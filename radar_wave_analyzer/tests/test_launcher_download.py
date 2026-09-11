"""启动器配置测试：导出功能依赖 WebView2 的下载许可。

背景：pywebview 的 settings['ALLOW_DOWNLOADS'] 默认 False，WebView2 在下载
开始时（DownloadStarting）会直接取消，导致「导出 CSV/Excel」既不弹出系统
“另存为”对话框也不落盘。launcher 必须在导入时开启该设置。
"""
import pytest


def test_launcher_enables_webview_downloads():
    webview = pytest.importorskip('webview')
    import launcher  # noqa: F401  导入即应用设置

    assert webview.settings['ALLOW_DOWNLOADS'] is True
