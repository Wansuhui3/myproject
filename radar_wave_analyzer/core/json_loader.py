"""
JSON 数据加载模块（雷达目标对象格式）。

输入结构（话题 → 目标ID → 字段数组）：
    {
        "/C798/High/RLR_OBJ_object": {
            "15": {"timestamp": [纳秒epoch...], "Dx": [...], "Track_Age": [...], ...}
        },
        "/C798/High/FLR_OBJ_object": null
    }

展开为与 CSV 一致的长表 DataFrame，复用 data_loader 的预处理与校验逻辑，
使分段、波动计算等下游算法对 JSON/CSV 数据源无感知。
"""
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from ..config import get
from .data_loader import _preprocess_csv

logger = logging.getLogger(__name__)

# 对象内部各字段数组的对齐基准列：无 timestamp 的对象无法参与时序分析
_TIMESTAMP_FIELD = 'timestamp'


def is_json_filename(filename: str | None) -> bool:
    """判断上传文件是否为 JSON 数据文件。"""
    return os.path.splitext(str(filename or ''))[1].strip('.').lower() == 'json'


def build_column_normalizer() -> dict[str, str]:
    """生成 JSON 字段名 → 配置物理量标准名的映射（大小写/空白不敏感）。

    仅映射到 config quantities 中定义的标准列（如 Rx_Front → Rx_front），
    保证主页面物理量勾选框可直接选用；未匹配的字段（Type、ExistProb 等）
    保留原名，供导出与对比页自定义物理量使用。
    """
    normalizer: dict[str, str] = {}
    for standard_name in (get('quantities', {}) or {}):
        lookup_key = str(standard_name).strip().lower()
        normalizer.setdefault(lookup_key, str(standard_name))
    return normalizer


def _normalize_columns(columns: list[str], normalizer: dict[str, str]) -> dict[str, str]:
    """返回需要重命名的列映射 {原名: 标准名}；无变化则返回空 dict。"""
    rename: dict[str, str] = {}
    for column in columns:
        standard = normalizer.get(str(column).strip().lower())
        if standard and standard != column:
            rename[column] = standard
    return rename


def _coerce_id_series(values: pd.Series) -> pd.Series:
    """ID 尝试数值化：整数转 int64，非数值保持原样（与 CSV ID 行为一致）。"""
    numeric = pd.to_numeric(values, errors='coerce')
    # np.floor 兼容 int/float dtype；Series.floor 仅浮点列可用
    if numeric.notna().all() and numeric.eq(np.floor(numeric)).all():
        return numeric.astype('int64')
    return values


def _flatten_objects(objects: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """将 {目标ID: {字段: 数组}} 展开为长表 DataFrame。

    ID 优先取对象内部的 ID 列，缺失时回退到外层键；
    各字段数组长度不齐时按索引对齐，短列产生 NaN 由预处理阶段清洗。
    """
    frames: list[pd.DataFrame] = []
    for object_key, fields in objects.items():
        if not isinstance(fields, dict) or not fields:
            continue
        part = pd.DataFrame(dict(fields))
        if 'ID' not in part.columns:
            part['ID'] = object_key
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        df = frames[0]
    else:
        # ignore_index 重建全局行号；同名列自动纵向拼接
        df = pd.concat(frames, ignore_index=True, sort=False)

    if 'ID' in df.columns:
        df['ID'] = _coerce_id_series(df['ID'])
    return df


def parse_json_topics(content_bytes: bytes, filename: str = '') -> dict[str, list]:
    """解析并展开 JSON 为话题级原始长表（宽松层，不校验业务必需列）。

    仅要求对象含 timestamp 字段；Track_Age 等业务必需列的校验由调用方
    按场景执行（主页面需 Track_Age，RTK 真值数据则不需要）。

    Returns:
        dict:
            topics:   [{'df': 原始长表(列名已规范化), 'topic': 话题路径,
                        'label': '文件名·话题名'}, ...]
            errors:   致命/话题级错误（中文，已含文件名前缀）
            warnings: 非致命提示（空话题跳过等）

    Raises:
        ValueError: 文件不是合法 JSON 或根结构不符时抛出（中文错误）。
    """
    display_name = os.path.basename(str(filename or '<upload>'))
    try:
        root = json.loads(content_bytes.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'JSON 文件解析失败（{exc}），请检查文件格式') from exc

    if not isinstance(root, dict):
        raise ValueError('JSON 根节点必须是对象（话题 → 目标ID → 字段数组）')

    normalizer = build_column_normalizer()
    topics: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []

    for topic, objects in root.items():
        topic_name = str(topic)
        topic_label = os.path.basename(topic_name.rstrip('/')) or topic_name

        if objects is None:
            warnings.append(f'{display_name}·{topic_label}: 话题为空，已跳过')
            continue
        if not isinstance(objects, dict) or not objects:
            warnings.append(f'{display_name}·{topic_label}: 话题无有效目标数据，已跳过')
            continue

        raw_df = _flatten_objects(objects)
        if raw_df.empty or _TIMESTAMP_FIELD not in raw_df.columns:
            errors.append(f'{display_name}·{topic_label}: 缺少 timestamp 字段，无法解析')
            continue

        rename = _normalize_columns(list(raw_df.columns), normalizer)
        if rename:
            raw_df = raw_df.rename(columns=rename)

        topics.append({
            'df': raw_df,
            'topic': topic_name,
            'label': f'{display_name}·{topic_label}',
        })

    if not topics and not errors:
        errors.append(f'{display_name}: 未找到可用的雷达目标话题数据')

    return {'topics': topics, 'errors': errors, 'warnings': warnings}


def load_json_from_bytes(content_bytes: bytes, filename: str = '') -> dict[str, list]:
    """从字节流加载雷达目标 JSON，逐话题展开并执行主页面预处理（严格层）。

    预处理复用 ``_preprocess_csv``：必需列校验（timestamp/ID/Track_Age）、
    Track_Age 0~255 整数校验、纳秒时间戳解析、全局时间排序。

    Returns:
        dict:
            topics:   [{'df': 预处理后DataFrame, 'topic': 话题路径,
                        'label': '文件名·话题名', 'objects': 目标数}, ...]
            errors:   致命/话题级错误（中文，已含文件名前缀）
            warnings: 非致命提示（空话题跳过等）

    Raises:
        ValueError: 文件不是合法 JSON 或根结构不符时抛出（中文错误）。
    """
    result = parse_json_topics(content_bytes, filename)
    topics: list[dict] = []

    for entry in result['topics']:
        try:
            df = _preprocess_csv(entry['df'], epoch_timestamps=True)
        except ValueError as exc:
            result['errors'].append(f'{entry["label"]}: {exc}')
            continue

        if df.empty:
            result['errors'].append(f'{entry["label"]}: 预处理后无有效数据行')
            continue

        objects = int(entry['df']['ID'].nunique()) if 'ID' in entry['df'].columns else 0
        topics.append({
            'df': df,
            'topic': entry['topic'],
            'label': entry['label'],
            'objects': objects,
        })
        logger.info(
            'JSON话题加载: %s, %d行, %d个目标ID',
            entry['label'], len(df),
            df['ID'].nunique() if 'ID' in df.columns else 0,
        )

    if not topics:
        if not result['errors']:
            result['errors'].append('JSON 数据预处理后无有效数据行')
        # 全部话题失败时不再返回半成品
        result['topics'] = []
    else:
        result['topics'] = topics
    return result
