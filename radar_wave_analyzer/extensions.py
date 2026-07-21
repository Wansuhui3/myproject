"""
共享扩展实例模块。
集中管理 cache 和 app 实例，消除 app.py ↔ cache.py ↔ callbacks.py 之间的循环导入。
"""
import os
import secrets
import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

try:
    from .config import get, _get_base_dir
except ImportError:
    from config import get, _get_base_dir  # type: ignore[no-redef]

LOCAL_BOOTSTRAP = '/assets/bootstrap.min.css'


def _get_assets_folder() -> str:
    """跨模式（开发 / PyInstaller 打包）定位 assets 目录。"""
    # 优先从根目录查找
    path = os.path.join(_get_base_dir(), 'assets')
    if os.path.isdir(path):
        return path
    # 兜底
    return os.path.join(os.getcwd(), 'assets')

# 缓存配置（从 config.yaml 读取，提供默认值兜底）
_cache_config = {
    'CACHE_TYPE': get('CACHE_TYPE', 'SimpleCache'),
    'CACHE_DEFAULT_TIMEOUT': get('CACHE_DEFAULT_TIMEOUT', 3600),
    'CACHE_THRESHOLD': get('CACHE_THRESHOLD', 5000),
}

# 全局 Cache 实例
cache = Cache(config=_cache_config)

# 创建 Dash 应用
# assets_folder 必须显式指定：因为 __name__ 不是 __main__ 且 PyInstaller 内模块在 PYZ 归档
app = dash.Dash(
    __name__,
    assets_folder=_get_assets_folder(),
    external_stylesheets=[LOCAL_BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=get('WINDOW_TITLE', '雷达目标轨迹波动分析系统'),
)

# 初始化 flask-caching（绑定到 Dash 底层 Flask server）
cache.init_app(app.server)

# Flask session 用于隔离不同浏览器窗口/用户的服务端缓存键。优先允许部署环境
# 注入固定密钥；桌面单机模式下使用进程级随机密钥，应用重启会自然失效旧会话。
app.server.secret_key = os.environ.get('RADAR_WAVE_SECRET_KEY') or secrets.token_urlsafe(32)

# 暴露 Flask server 对象，供 launcher.py / PyInstaller 使用
server = app.server
