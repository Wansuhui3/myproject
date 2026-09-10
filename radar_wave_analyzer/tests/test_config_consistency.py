"""config.py `_DEFAULTS` 与 config.yaml 一致性校验。

项目硬性约束：config.py 的默认值必须与 config.yaml 保持一致。
本测试拦截两类漂移：
1. yaml 顶层键在 _DEFAULTS 中不存在（多为键名笔误，该配置会被静默忽略语义）
2. 两侧同时存在的键值不等（单侧修改导致另一侧失效）

豁免说明：以下子树以 yaml 为唯一权威来源，_DEFAULTS 中留空/缺省属设计意图，
不做键值比对：
- radar_sources（运行时按雷达型号填充）
- performance_metrics.metrics（阈值规则表，文档规定 yaml 唯一来源）
- comparison.quantities（对比指标定义，仅存在于 yaml）
- comparison 下其余 yaml 独有键（match_threshold 等，有意不设默认值）
"""
import os
from pathlib import Path

import pytest
import yaml

from radar_wave_analyzer import config as config_mod

CONFIG_PATH = Path(config_mod.__file__).with_name('config.yaml')

# 以 yaml 为唯一权威的子树（不要求出现在 _DEFAULTS，也不比对键值）
YAML_AUTHORITATIVE_SUBTREES = {
    ('radar_sources',),
    ('performance_metrics', 'metrics'),
    ('comparison', 'quantities'),
}


@pytest.fixture(scope='module')
def yaml_config():
    assert os.path.isfile(CONFIG_PATH), f'config.yaml 不存在: {CONFIG_PATH}'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), 'config.yaml 根节点必须是对象'
    return data


def _is_authoritative(path: tuple) -> bool:
    """path 是否位于（或正位于）yaml 权威子树。"""
    return any(path[:len(sub)] == sub for sub in YAML_AUTHORITATIVE_SUBTREES)


def _check_values(defaults, yaml_data, path, errors):
    """递归比对两侧共有键；字典深入，其余类型要求相等。"""
    if _is_authoritative(path):
        return
    if isinstance(defaults, dict) and isinstance(yaml_data, dict):
        for key, default_value in defaults.items():
            if key not in yaml_data:
                # 默认值兜底键允许不出现在 yaml（如 WINDOW_TITLE）
                continue
            _check_values(default_value, yaml_data[key], path + (key,), errors)
        return
    if defaults != yaml_data:
        errors.append(
            f'{".".join(path) or "<root>"}: '
            f'config.py={defaults!r} vs config.yaml={yaml_data!r}'
        )


def test_yaml_top_level_keys_exist_in_defaults(yaml_config):
    """yaml 顶层键必须在 _DEFAULTS 中有对应键，否则该配置语义失效。"""
    unknown = sorted(set(yaml_config) - set(config_mod._DEFAULTS))
    assert not unknown, f'config.yaml 存在 _DEFAULTS 未定义的键（疑似笔误）: {unknown}'


def test_shared_values_consistent(yaml_config):
    """两侧同时存在的配置键值必须一致，防止单侧修改造成静默漂移。"""
    errors = []
    _check_values(config_mod._DEFAULTS, yaml_config, (), errors)
    assert not errors, 'config.py 与 config.yaml 存在键值漂移:\n' + '\n'.join(errors)


def test_yaml_authoritative_subtrees_load(yaml_config):
    """权威子树必须能被 _load_config 深度合并保留（不被默认值覆盖丢失）。"""
    cfg = config_mod.reload()
    radar_sources = yaml_config.get('radar_sources') or {}
    for key in radar_sources:
        assert key in cfg['radar_sources'], f'radar_sources.{key} 合并后丢失'
    perf_metrics = yaml_config.get('performance_metrics', {}).get('metrics') or {}
    for key in perf_metrics:
        assert key in cfg['performance_metrics']['metrics'], (
            f'performance_metrics.metrics.{key} 合并后丢失'
        )
    cmp_quantities = yaml_config.get('comparison', {}).get('quantities') or {}
    for key in cmp_quantities:
        assert key in cfg['comparison']['quantities'], (
            f'comparison.quantities.{key} 合并后丢失'
        )
