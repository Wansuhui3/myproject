"""
Dash 应用入口 + 布局定义。
app 实例来自 extensions.py（含 assets_folder 配置），此处仅挂载布局和回调。
"""
# 导入已配置好的 Dash 实例（包含 assets_folder、cache 等）
try:
    from .extensions import app
except ImportError:
    from extensions import app  # type: ignore[no-redef]

# 构建布局
try:
    from .components.layout import build_layout
except ImportError:
    from components.layout import build_layout  # type: ignore[no-redef]

app.layout = build_layout()

# 导入回调（必须在 app 创建之后；extensions.py 已初始化 cache）
try:
    from . import callbacks as _callbacks  # noqa: F401
except ImportError:
    import callbacks  # type: ignore[no-redef]  # noqa: E402, F401
