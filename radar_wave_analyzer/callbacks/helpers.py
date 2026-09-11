"""回调模块间共享的无状态辅助函数。

被 wave / comparison / performance 三个回调域共同使用，保持零业务依赖
（不导入 cache、service 等），避免模块间循环导入。
"""
import base64

from dash import html


def _decode_upload_contents(contents) -> bytes:
    if not isinstance(contents, str):
        raise ValueError('文件内容为空')
    try:
        _, content_string = contents.split(',', 1)
        return base64.b64decode(content_string)
    except Exception as e:
        raise ValueError(f'文件解码失败: {e}') from e


# 上传组件允许的扩展名（与 dcc.Upload 的 accept 保持一致）
SUPPORTED_UPLOAD_SUFFIXES = ('.csv', '.json')


def is_supported_upload(filename) -> bool:
    """上传前置校验：仅接受 .csv / .json（大小写不敏感）。

    前端 accept 只约束文件对话框，拖拽仍可绕过；在此统一拦截，
    使不支持的类型在上传回调入口即被明确拒绝，不会进入解析路径。
    """
    return str(filename or '').strip().lower().endswith(SUPPORTED_UPLOAD_SUFFIXES)


def _perf_placeholder(message: str = '执行对齐后显示'):
    """性能验收表面板占位输出（对齐失败/空数据时）。"""
    return html.Div(message, className='stats-empty')


def _summary_plain_text(node) -> str:
    """递归提取摘要组件的纯文本，跳过快照来源元信息行。

    复制摘要时用户只需要数据行（如 ``Dx最大波动：…`` 或分距离行），
    [1]/[2] 等快照来源说明（perf-summary-snapshot-list/meta）不复制。
    摘要行为数值级标红拆成多个 span 后，同一行仍按整行文本输出，不拆行。
    """
    skip_fragments = ('perf-summary-snapshot-list', 'perf-summary-snapshot-meta')
    lines: list[str] = []

    def _collect(item) -> str:
        """收集节点内全部文本，用于把同一行的多个 span 还原为一行。"""
        if item is None:
            return ''
        if isinstance(item, (list, tuple)):
            return ''.join(_collect(child) for child in item)
        if isinstance(item, str):
            return item
        return _collect(getattr(item, 'children', None))

    def _walk(item):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                _walk(child)
            return
        if isinstance(item, str):
            if item.strip():
                lines.append(item.strip())
            return
        cls = str(getattr(item, 'className', '') or '')
        if any(fragment in cls for fragment in skip_fragments):
            return
        if 'perf-summary-row' in cls:
            text = _collect(getattr(item, 'children', None)).strip()
            if text:
                lines.append(text)
            return
        _walk(getattr(item, 'children', None))

    _walk(node)
    return '\n'.join(lines)
