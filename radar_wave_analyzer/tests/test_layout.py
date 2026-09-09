"""Layout regression tests."""
from radar_wave_analyzer.components.layout import build_layout  # noqa: E402


def _walk(component):
    """Yield every Dash component and text node in a component tree."""
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)
        return

    yield component
    children = getattr(component, 'children', None)
    if children is not None:
        yield from _walk(children)


def test_brand_bar_is_removed_but_radar_callback_components_remain():
    layout = build_layout()
    nodes = list(_walk(layout))
    text = ''.join(node for node in nodes if isinstance(node, str))
    ids = {
        getattr(node, 'id', None)
        for node in nodes
        if not isinstance(node, str)
    }
    classes = {
        getattr(node, 'className', None)
        for node in nodes
        if not isinstance(node, str)
    }

    assert 'RADAR LAB' not in text
    assert '本地模式' not in text
    assert 'top-bar' not in classes
    assert {'radar-selector', 'radar-position-label'} <= ids
