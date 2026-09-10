"""客户端回调注册（需要 app 实例）。

Python 回调由各域模块通过 ``dash.callback`` 装饰器注册；仅这两个
clientside 回调必须显式持有 app 实例，单独成模块便于依赖隔离。
"""
from dash import Input, Output

from ..extensions import app

# ============================================================
# 客户端回调: ID 列表更新后自动滚动到已选中条目
# ============================================================
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var el = document.querySelector('.id-list-item.selected');
            if (el) {
                el.scrollIntoView({behavior: 'smooth', block: 'nearest'});
            }
        }, 150);
        return '';
    }
    """,
    Output('scroll-anchor', 'children'),
    Input('id-list-container', 'children'),
)


# ============================================================
# 客户端回调: 文件处理完成后隐藏上传遮罩
# store-data-loaded（波动页）与 cmp-state（对比页）任一变化即关闭遮罩
# ============================================================
app.clientside_callback(
    """
    function(loaded, cmpState) {
        if (window.hideUploadOverlay) { window.hideUploadOverlay(); }
        return '';
    }
    """,
    Output('scroll-anchor', 'children', allow_duplicate=True),
    Input('store-data-loaded', 'data'),
    Input('cmp-state', 'data'),
    prevent_initial_call=True,
)
