"""
Dash 应用入口 + 布局定义。
app 实例来自 extensions.py（含 assets_folder 配置），此处仅挂载布局和回调。
"""
# 导入已配置好的 Dash 实例（包含 assets_folder、cache 等）
# 构建布局
from .components.layout import build_layout
from .extensions import app

app.layout = build_layout()

# 导入回调（必须在 app 创建之后；extensions.py 已初始化 cache）
from . import callbacks as _callbacks  # noqa: F401

if __name__ == '__main__':
    # 与 launcher.py / start.ps1 保持一致的默认地址
    app.run(host='127.0.0.1', port=8050, debug=False, threaded=True)
