"""
桌面启动器 — 用 pywebview 包装 Dash 应用为独立桌面窗口。
优化：先显示窗口（加载动画），后台异步初始化 Flask，减少感知启动时间。

可被两种方式执行：
  - python launcher.py (开发模式，CWD=项目根)
  - PyInstaller 打包后的 exe
"""
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import webview

logger = logging.getLogger(__name__)

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8050
WINDOW_TITLE = '雷达目标轨迹波动分析系统'
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900


def _get_loading_html() -> str:
    """获取加载页 HTML，优先从磁盘读取，fallback 到内联。"""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radar_wave_analyzer', 'assets', 'loading.html'),
        os.path.join(os.path.dirname(sys.executable), 'assets', 'loading.html'),
        os.path.join(sys._MEIPASS, 'assets', 'loading.html') if getattr(sys, 'frozen', False) else '',
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                return f.read()

    # 兜底：内联加载页
    return """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><style>
*{margin:0;padding:0;box-sizing:border-box}body{display:flex;flex-direction:column;
align-items:center;justify-content:center;height:100vh;
background:linear-gradient(135deg,#0f172a,#1e293b);
font-family:'Segoe UI','Microsoft YaHei',sans-serif}
.spinner{width:48px;height:48px;border:4px solid rgba(59,130,246,.2);
border-top-color:#3b82f6;border-radius:50%;animation:spin .8s linear infinite;margin-bottom:24px}
@keyframes spin{to{transform:rotate(360deg)}}
.title{color:#e2e8f0;font-size:20px;font-weight:600;margin-bottom:8px}
.sub{color:#94a3b8;font-size:14px}</style></head><body>
<div class="spinner"></div><div class="title">雷达目标轨迹波动分析系统</div>
<div class="sub">正在初始化，请稍候...</div></body></html>"""


def _find_free_port(start_port: int = DEFAULT_PORT) -> int:
    """查找可用端口，避免端口冲突。"""
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((DEFAULT_HOST, port))
                return port
            except OSError:
                port += 1
    return start_port


def _start_flask(port: int):
    """后台线程：导入并启动 Flask 服务器。"""
    # 延迟导入重量级库（dash/pandas/plotly），此时窗口已显示
    try:
        from radar_wave_analyzer.app import app
    except ImportError:
        from app import app  # type: ignore[no-redef]

    app.run(
        host=DEFAULT_HOST,
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


def _wait_for_server(url: str, timeout_sec: float = 30.0) -> bool:
    """轮询 HTTP 服务，确认 Dash 已真正开始响应后才允许窗口跳转。"""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.2)
    return False


def main():
    port = _find_free_port()
    url = f'http://{DEFAULT_HOST}:{port}'

    # 后台线程启动 Flask（导入重量级库在此线程中，不阻塞 UI）
    server_thread = threading.Thread(
        target=_start_flask,
        args=(port,),
        daemon=True,
    )
    server_thread.start()

    # 立即显示窗口（加载动画页面）
    loading_html = _get_loading_html()
    window = webview.create_window(
        title=WINDOW_TITLE,
        html=loading_html,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(800, 600),
        resizable=True,
        confirm_close=True,
    )

    # 仅在 HTTP 已可访问后导航，避免“模块导入完成但端口尚未监听”的竞态。
    def _navigate_when_ready():
        if _wait_for_server(url):
            try:
                window.load_url(url)
            except Exception:
                pass  # 窗口可能已被关闭
        else:
            logger.error('Dash 服务在 30 秒内未就绪: %s', url)

    threading.Thread(target=_navigate_when_ready, daemon=True).start()

    webview.start()
    logger.info('窗口已关闭，程序退出')
    sys.exit(0)


if __name__ == '__main__':
    log_dir = os.path.dirname(os.path.abspath(__file__))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'app.log'), encoding='utf-8'),
        ],
    )
    main()
