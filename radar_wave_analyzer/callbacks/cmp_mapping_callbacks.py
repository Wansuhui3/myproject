from dash import (
    Input,
    Output,
    State,
    callback,
    html,
    dcc,
    ALL,
)
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate
from ..config import get
"""[C2a] 原始 CSV 物理量映射配置：预览、增删改、排序与映射解析。"""



@callback(
    Output('cmp-preview-summary', 'children'),
    Input('cmp-state', 'data'),
)
def render_cmp_preview_summary(state):
    """在折叠标题中展示两类数据的文件数与总帧数。"""
    radar_meta = (state or {}).get('radar_meta')
    rtk_meta = (state or {}).get('rtk_meta')
    if not radar_meta and not rtk_meta:
        return '等待加载雷达与真值数据'

    def _summary(label, meta):
        if not meta:
            return f'{label}待上传'
        file_count = int(meta.get('file_count', 1))
        total_rows = int(meta.get('total_rows', 0))
        return f'{label}{file_count}份 / {total_rows:,}帧'

    return f'{_summary("雷达", radar_meta)} · {_summary("RTK", rtk_meta)}'
def _cmp_mapping_signature(state: dict | None) -> str:
    if not state:
        return ''
    parts = []
    for role in ('radar_meta', 'rtk_meta'):
        meta = state.get(role) or {}
        parts.append(str(meta.get('filename', '')))
        parts.extend(
            f'{field.get("name", "")}:{field.get("unit", "")}:{field.get("files_present", 1)}'
            for field in meta.get('physical_fields', [])
        )
    return '|'.join(parts)
def _cmp_field_lookup(state: dict | None, role: str) -> dict[str, dict]:
    meta = (state or {}).get(f'{role}_meta') or {}
    return {
        str(field.get('name')): dict(field)
        for field in meta.get('physical_fields', [])
        if field.get('name')
    }
def _cmp_default_mappings(state: dict) -> list[dict]:
    radar_fields = list(_cmp_field_lookup(state, 'radar').values())
    rtk_fields = list(_cmp_field_lookup(state, 'rtk').values())
    def _match(fields: list[dict], internal: str):
        """按 internal_name 精确匹配；加速度列大小写不定时回退到宽松比较。"""
        exact = next(
            (f for f in fields if f.get('internal_name') == internal), None)
        if exact:
            return exact
        # Ax/Ay 等自定义列在 CSV 中可能是 ax/AX/Accel_X 等写法
        target = internal.lower()
        return next(
            (f for f in fields
             if str(f.get('internal_name') or '').lower() == target
             or str(f.get('name') or '').lower() == target), None)

    defaults = []
    for radar_internal, rtk_internal in (
        ('Dx', 'center_x'), ('Dy', 'center_y'), ('Vx', 'Vx'), ('Vy', 'Vy'),
        # 加速度通道：双方 CSV 均含 Ax/Ay 列时默认纳入映射，
        # 供误差统计与性能验收直接使用，无需手动添加。
        ('Ax', 'Ax'), ('Ay', 'Ay'),
    ):
        radar = _match(radar_fields, radar_internal)
        rtk = _match(rtk_fields, rtk_internal)
        if radar and rtk:
            defaults.append({
                'uid': f'mapping-{len(defaults) + 1}',
                'radar_col': radar['name'],
                'rtk_col': rtk['name'],
            })
    if not defaults and radar_fields and rtk_fields:
        defaults.append({
            'uid': 'mapping-1',
            'radar_col': radar_fields[0]['name'],
            'rtk_col': rtk_fields[0]['name'],
        })
    return defaults
def _cmp_mapping_unit_status(radar_field: dict | None, rtk_field: dict | None) -> tuple[str, str]:
    radar_unit = str((radar_field or {}).get('unit', ''))
    rtk_unit = str((rtk_field or {}).get('unit', ''))
    if radar_unit and rtk_unit and radar_unit == rtk_unit:
        return 'compatible', radar_unit
    if radar_unit and rtk_unit and radar_unit != rtk_unit:
        return 'incompatible', f'{radar_unit} / {rtk_unit}'
    return 'unknown', radar_unit or rtk_unit
def _cmp_mapping_options(fields: dict[str, dict]) -> list[dict]:
    # 两侧都允许不选：未选字段不会进入对齐映射，也不会参与误差统计。
    return [{'label': '（不选择）', 'value': None}] + [
        {'label': name, 'value': name} for name in fields
    ]
def _cmp_resolve_mappings(mapping_store: dict | None, state: dict | None) -> list[dict]:
    radar_fields = _cmp_field_lookup(state, 'radar')
    rtk_fields = _cmp_field_lookup(state, 'rtk')
    resolved = []
    for item in (mapping_store or {}).get('items', []):
        radar_col = item.get('radar_col')
        rtk_col = item.get('rtk_col')
        has_radar = radar_col in radar_fields
        has_rtk = rtk_col in rtk_fields
        # 一侧留空时仍是有效的“单线显示”通道；只有两侧都未选或字段
        # 已不存在时才忽略。此前这里要求两侧同时存在，导致单物理量
        # 在执行对齐时被整个丢弃，图表自然没有可画的数据。
        if not has_radar and not has_rtk:
            continue
        if has_radar and has_rtk:
            status, unit = _cmp_mapping_unit_status(
                radar_fields[radar_col], rtk_fields[rtk_col])
        elif has_radar:
            status, unit = 'single', str(radar_fields[radar_col].get('unit', ''))
        else:
            status, unit = 'single', str(rtk_fields[rtk_col].get('unit', ''))
        resolved.append({
            **item,
            'radar_unit': radar_fields[radar_col].get('unit', '') if has_radar else '',
            'rtk_unit': rtk_fields[rtk_col].get('unit', '') if has_rtk else '',
            'unit': unit,
            'unit_status': status,
            'stats_enabled': status == 'compatible',
        })
    return resolved
def _append_auto_performance_mappings(
    resolved_mappings: list[dict],
    state: dict | None,
) -> list[dict]:
    """自动补建性能评估所需的 Ax/Ay 映射通道（问题：RTK 自带 Ax/Ay 列）。

    用户默认只映射 Dx/Dy/Vx/Vy，导致 aligned_df 缺少 radar[Ax]/rtk[Ax]
    等列、Ax/Ay 无法评估。此处检测双方原始字段是否均含 Ax/Ay，是则自动
    追加映射（标记 auto_injected=True）。该类通道：
      - 参与对齐生成 aligned_df 列，供性能评估使用；
      - 不进入对比图、误差统计与分距离统计（由消费方过滤）。
    已被用户显式映射的同名列不会重复注入。
    """
    radar_fields = _cmp_field_lookup(state, 'radar')
    rtk_fields = _cmp_field_lookup(state, 'rtk')
    used_pairs = {
        (str(m.get('radar_col')), str(m.get('rtk_col')))
        for m in resolved_mappings
    }
    extras: list[dict] = []
    for index, field in enumerate(('Ax', 'Ay')):
        if field not in radar_fields or field not in rtk_fields:
            continue
        if (field, field) in used_pairs:
            continue
        status, unit = _cmp_mapping_unit_status(
            radar_fields[field], rtk_fields[field])
        if status != 'compatible':
            continue
        extras.append({
            'uid': f'auto-perf-{field.lower()}',
            'radar_col': field,
            'rtk_col': field,
            'auto_injected': True,
            'radar_unit': radar_fields[field].get('unit', ''),
            'rtk_unit': rtk_fields[field].get('unit', ''),
            'unit': unit,
            'unit_status': status,
            'stats_enabled': False,
        })
    return resolved_mappings + extras
def _cmp_mapping_chart_config(mapping_results: list[dict]) -> tuple[list[str], dict]:
    selected = []
    config = {}
    for index, mapping in enumerate(mapping_results):
        # 自动注入的性能通道（Ax/Ay）仅用于评估，不绘制对比曲线
        if mapping.get('auto_injected'):
            continue
        key = str(mapping.get('uid', f'mapping-{index + 1}'))
        radar_label = str(mapping.get('radar_col', ''))
        rtk_label = str(mapping.get('rtk_col', ''))
        display_label = radar_label or rtk_label
        selected.append(key)
        config[key] = {
            'label': display_label,
            'unit': mapping.get('unit', ''),
            'chart_type': 'overlay',
            'radar_col': mapping.get('radar_output_col', ''),
            'rtk_col': mapping.get('rtk_output_col', ''),
            'rtk_curve_col': mapping.get('rtk_curve_col', ''),
            'radar_source_label': '雷达',
            'rtk_source_label': '真值',
            'rtk_label': rtk_label,
            'show_difference': bool(mapping.get('stats_enabled')),
            'radar_unit': mapping.get('radar_unit', ''),
            'rtk_unit': mapping.get('rtk_unit', ''),
        }
    return selected, config
@callback(
    Output('cmp-mappings', 'data'),
    Input('cmp-state', 'data'),
    State('cmp-mappings', 'data'),
    prevent_initial_call=True,
)
def on_cmp_mapping_sources(state, current_store):
    signature = _cmp_mapping_signature(state)
    if not signature:
        return {'signature': '', 'items': []}
    if (current_store or {}).get('signature') == signature:
        raise PreventUpdate
    return {'signature': signature, 'items': _cmp_default_mappings(state or {})}
@callback(
    Output('cmp-mapping-list', 'children'),
    Output('cmp-mapping-feedback', 'children'),
    Output('cmp-mapping-summary', 'children'),
    Input('cmp-mappings', 'data'),
    State('cmp-state', 'data'),
)
def render_cmp_mappings(mapping_store, state):
    radar_fields = _cmp_field_lookup(state, 'radar')
    rtk_fields = _cmp_field_lookup(state, 'rtk')
    if not radar_fields or not rtk_fields:
        return (
            html.Div('上传雷达与 RTK 文件后生成推荐映射', className='stats-empty'),
            '',
            '等待加载雷达与真值数据',
        )

    resolved_by_uid = {
        item['uid']: item for item in _cmp_resolve_mappings(mapping_store, state)
    }
    rows = []
    for index, item in enumerate((mapping_store or {}).get('items', [])):
        uid = item['uid']
        resolved = resolved_by_uid.get(uid, {})
        has_radar = bool(item.get('radar_col'))
        has_rtk = bool(item.get('rtk_col'))
        status = resolved.get('unit_status', 'unknown') if (has_radar or has_rtk) else 'unselected'
        status_text = {
            'compatible': f'✓ {resolved.get("unit", "")}',
            'incompatible': f'⚠ {resolved.get("unit", "单位不一致")}',
            'unknown': '？待确认',
            'single': '单线显示',
            'unselected': '— 未选择',
        }[status]
        rows.append(html.Div([
            html.Span(str(index + 1), className='mapping-channel-name'),
            html.Div([
                html.Span('雷达', className='mapping-mobile-label'),
                dcc.Dropdown(
                    id={'type': 'cmp-mapping-radar', 'index': uid},
                    options=_cmp_mapping_options(radar_fields),
                    # 不显示下拉框自带“×”清空按钮：它会遮挡窄栏中的物理量名。
                    # 需要留空时，明确选择列表中的“（不选择）”。
                    value=item.get('radar_col'), clearable=False,
                    className='mapping-dropdown',
                ),
            ], className='mapping-field mapping-radar-field',
               title=str(item.get('radar_col', ''))),
            html.Div([
                html.Span('真值', className='mapping-mobile-label'),
                dcc.Dropdown(
                    id={'type': 'cmp-mapping-rtk', 'index': uid},
                    options=_cmp_mapping_options(rtk_fields),
                    value=item.get('rtk_col'), clearable=False,
                    className='mapping-dropdown',
                ),
            ], className='mapping-field mapping-rtk-field',
               title=str(item.get('rtk_col', ''))),
            html.Span(status_text, className=f'mapping-unit-status {status}',
                      title=status_text),
            html.Div([
                html.Button('↑', id={'type': 'cmp-mapping-up', 'index': uid},
                            n_clicks=0, className='mapping-icon-btn', disabled=index == 0,
                            title='上移通道'),
                html.Button('↓', id={'type': 'cmp-mapping-down', 'index': uid},
                            n_clicks=0, className='mapping-icon-btn',
                            disabled=index == len((mapping_store or {}).get('items', [])) - 1,
                            title='下移通道'),
                html.Button('×', id={'type': 'cmp-mapping-delete', 'index': uid},
                            n_clicks=0, className='mapping-delete-btn', title='删除通道'),
            ], className='mapping-row-actions'),
        ], className='mapping-row'))

    if not rows:
        rows = [html.Div('点击“添加通道”创建物理量对比', className='stats-empty')]
    resolved = list(resolved_by_uid.values())
    compatible = sum(item.get('stats_enabled', False) for item in resolved)
    feedback = f'{len(resolved)} 个显示通道 · {compatible} 个可计算误差统计'
    summary = (
        f'{len(resolved)} 个通道 · {compatible} 个可计算误差'
        if resolved else '尚未配置显示通道'
    )
    return rows, feedback, summary
@callback(
    Output('cmp-mappings', 'data', allow_duplicate=True),
    Input('cmp-add-mapping-btn', 'n_clicks'),
    Input({'type': 'cmp-mapping-radar', 'index': ALL}, 'value'),
    Input({'type': 'cmp-mapping-rtk', 'index': ALL}, 'value'),
    Input({'type': 'cmp-mapping-delete', 'index': ALL}, 'n_clicks'),
    Input({'type': 'cmp-mapping-up', 'index': ALL}, 'n_clicks'),
    Input({'type': 'cmp-mapping-down', 'index': ALL}, 'n_clicks'),
    State('cmp-mappings', 'data'),
    State('cmp-state', 'data'),
    prevent_initial_call=True,
)
def on_cmp_mapping_edit(_add, _radar_values, _rtk_values, _delete, _up, _down,
                        mapping_store, state):
    triggered = dash_ctx.triggered_id
    if triggered is None:
        raise PreventUpdate
    store = dict(mapping_store or {'signature': _cmp_mapping_signature(state), 'items': []})
    items = [dict(item) for item in store.get('items', [])]

    if triggered == 'cmp-add-mapping-btn':
        if not _add:
            raise PreventUpdate
        radar_fields = list(_cmp_field_lookup(state, 'radar'))
        rtk_fields = list(_cmp_field_lookup(state, 'rtk'))
        if not radar_fields or not rtk_fields or len(items) >= 8:
            raise PreventUpdate
        numbers = [
            int(item['uid'].rsplit('-', 1)[-1])
            for item in items if str(item.get('uid', '')).rsplit('-', 1)[-1].isdigit()
        ]
        items.append({
            'uid': f'mapping-{max(numbers, default=0) + 1}',
            'radar_col': radar_fields[0],
            'rtk_col': rtk_fields[0],
        })
    elif isinstance(triggered, dict):
        uid = triggered.get('index')
        row_index = next((i for i, item in enumerate(items) if item.get('uid') == uid), None)
        if row_index is None:
            raise PreventUpdate
        control_type = triggered.get('type')
        value = dash_ctx.triggered[0].get('value')
        if control_type == 'cmp-mapping-radar':
            if items[row_index].get('radar_col') == value:
                raise PreventUpdate
            items[row_index]['radar_col'] = value
        elif control_type == 'cmp-mapping-rtk':
            if items[row_index].get('rtk_col') == value:
                raise PreventUpdate
            items[row_index]['rtk_col'] = value
        elif control_type == 'cmp-mapping-delete' and value:
            items.pop(row_index)
        elif control_type == 'cmp-mapping-up' and value and row_index > 0:
            items[row_index - 1], items[row_index] = items[row_index], items[row_index - 1]
        elif control_type == 'cmp-mapping-down' and value and row_index < len(items) - 1:
            items[row_index + 1], items[row_index] = items[row_index], items[row_index + 1]
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

    store['items'] = items
    return store
