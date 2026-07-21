"""运行配置的关键参数校验测试。"""
from copy import deepcopy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigValidationError, _DEFAULTS, _validate_config  # noqa: E402


def test_default_configuration_is_valid():
    """内置默认配置必须始终可通过校验。"""
    _validate_config(deepcopy(_DEFAULTS))


def test_invalid_wrap_range_is_rejected():
    """回绕边界倒置会改变分段结果，必须阻止启动。"""
    config = deepcopy(_DEFAULTS)
    config['WRAP_LOW'] = 251
    config['WRAP_HIGH'] = 250

    with pytest.raises(ConfigValidationError, match='WRAP_LOW/WRAP_HIGH'):
        _validate_config(config)


def test_invalid_comparison_coverage_is_rejected():
    """覆盖率不能超出 0~1。"""
    config = deepcopy(_DEFAULTS)
    config['comparison']['delay_min_match_rate'] = 1.2

    with pytest.raises(ConfigValidationError, match='delay_min_match_rate'):
        _validate_config(config)


def test_distance_bins_must_be_strictly_increasing():
    """重复或逆序距离桶会导致统计口径重叠。"""
    config = deepcopy(_DEFAULTS)
    config['comparison']['distance_bins'] = [0, 10, 10]

    with pytest.raises(ConfigValidationError, match='distance_bins'):
        _validate_config(config)


def test_id_matching_coverage_must_be_a_ratio():
    """ID过滤关联覆盖率不允许超出 0~1。"""
    config = deepcopy(_DEFAULTS)
    config['comparison']['id_matching']['min_pair_coverage'] = -0.1

    with pytest.raises(ConfigValidationError, match='min_pair_coverage'):
        _validate_config(config)
