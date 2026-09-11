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
# upload-tick 由每次上传回调无条件写入新时间戳，因此无论成功、失败
# （含解析失败/类型不支持）都会触发本回调关闭遮罩；store-data-loaded
# 与 cmp-state 作为兼容兜底保留。
# ============================================================
app.clientside_callback(
    """
    function(loaded, cmpState, tick) {
        if (window.hideUploadOverlay) { window.hideUploadOverlay(); }
        return '';
    }
    """,
    Output('scroll-anchor', 'children', allow_duplicate=True),
    Input('upload-tick', 'data'),
    Input('store-data-loaded', 'data'),
    Input('cmp-state', 'data'),
    prevent_initial_call=True,
)
