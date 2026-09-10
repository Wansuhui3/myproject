"""
配置参数加载模块。
从 config.yaml 读取配置，提供默认值兜底。
禁止在代码中硬编码任何配置参数值。
"""
import os
import sys
from copy import deepcopy
from numbers import Real
from typing import Any

import yaml

# 默认值，与 config.yaml 保持一致
_DEFAULTS: dict[str, Any] = {
    'SAMPLING_PERIOD_MS': 50,
    'GAP_THRESHOLD': 500,
    'MIN_TRAJ_FRAMES': 20,
    'WRAP_HIGH': 250,
    'WRAP_LOW': 5,
    # ---- 位置不连续 / ID 复用 增强判定 ----
    'SPATIAL_SPLIT_ENABLED': False,     # 是否启用位置跳变强制切分（规则 F），默认关闭保证零回归
    'MAX_TRACK_SPEED': 50.0,            # 帧间最大合理速度（m/s），超过视为不同物理目标
    'POS_JUMP_THRESHOLD': 5.0,          # 同时间戳(Δt=0)位置跳变阈值（m），超过视为不同目标
    'POSITION_COLUMNS': ['Dx', 'Dy'],   # 用于计算空间位移的物理量列
    'LIFECYCLE_COLUMNS': [],            # 生命周期/状态字段（如 Track_Status），留空表示不使用
    'LIFECYCLE_END_TOKENS': ['end', 'dead', 'lost', 'invalid', '0', 'false'],
    'LIFECYCLE_START_TOKENS': ['new', 'begin', 'valid', 'alive', '1', 'true'],
    'TIMESTAMP_WINDOW_SEC': 30,
    'DEFAULT_QUANTITY': 'Dx',
    'MAX_OVERLAY_CURVES': 4,
    'CSV_CHUNK_THRESHOLD_MB': 200,
    'CSV_CHUNK_SIZE_ROWS': 100_000,
    'radar_sources': {},
    'RADAR_FILE_PATTERNS': {
        'flr': {
            'label': '前角雷达',
            'short_label': 'FLR 前角',
            'patterns': [r'(?:^|[_-])flr(?:[_-]|$)'],
        },
        'rlr': {
            'label': '后角雷达',
            'short_label': 'RLR 后角',
            'patterns': [r'(?:^|[_-])rlr(?:[_-]|$)'],
        },
    },
    'UNKNOWN_RADAR_SOURCE_LABEL': '未识别雷达',
    'quantities': {
        'Dx': {'label': 'Dx', 'unit': 'm'},
        'Dy': {'label': 'Dy', 'unit': 'm'},
        'Vx': {'label': 'Vx', 'unit': 'm/s'},
        'Vy': {'label': 'Vy', 'unit': 'm/s'},
        'Ax': {'label': 'Ax', 'unit': 'm/s²'},
        'Ay': {'label': 'Ay', 'unit': 'm/s²'},
        'HeadingAngle': {'label': 'HeadingAngle', 'unit': '°'},
        'Vabs': {'label': 'Vabs', 'unit': 'm/s'},
        'Rx_front': {'label': 'Rx_front', 'unit': 'm'},
        'Rx_rear': {'label': 'Rx_rear', 'unit': 'm'},
        'Ry': {'label': 'Ry', 'unit': 'm'},
    },
    'EXPORT_ENCODING': 'utf-8-sig',
    'WINDOW_TITLE': 'RadarWaveAnalyzer',
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 0,      # 0 = 永不超时
    'CACHE_THRESHOLD': 5000,
    'DISPLAY_DOWNSAMPLING_ENABLED': True,
    # 单条轨迹不超过 5000 帧时完整显示；超过才进入显示降采样。
    'DISPLAY_MAX_POINTS': 5000,
    'GRAPH_RENDERER': 'svg',
    # webgl 模式下允许的最大面板数：plotly.js 每个子图独立创建 WebGL 上下文
    # （无共享池），超过阈值自动回退 svg，避免多 GL 上下文缩放闪烁
    'WEBGL_MAX_SUBPLOTS': 4,
    'REMOVE_ISOLATED_SINGLE_FRAMES': True,
    'RESAMPLER_ENABLED': False,
    'RESAMPLER_MAX_POINTS': 5000,
    'RESAMPLER_DEFAULT_N_SAMPLES': 2000,
    'comparison': {
        'delay_min_matched_frames': 3,
        'delay_min_match_rate': 0.5,
        'id_matching': {
            'enabled': True,
            'min_frames': 3,
            'min_speed_mps': 0.3,
            'min_accel_mps2': 1.0,
            'min_displacement_m': 0.5,
            'min_pair_frames': 3,
            'min_pair_coverage': 0.5,
            'track_age_gap_ms': 500.0,
            'track_age_confirm_frames': 2,
            'track_age_reset_max': 10,
            'lifecycle_position_jump_m': 8.0,
        },
        'radar_gap_break_factor': 2.5,
    },
    # ---- 目标性能指标评估 ----
    # 阈值规则表以 config.yaml 为唯一来源，此处仅兜底全局判定参数，
    # metrics 缺失时对应指标在评估层降级为“不可判定”，不影响其他指标。
    'performance_metrics': {
        'enabled': True,
        'distance_bins': [0, 10, 20, 50, 70, 100, 150],
        'include_last_upper': True,
        'valid_distance_range': [0, 150],
        'accuracy_requirement': 0.9545,
        'three_frame_requirement': 1.0,
        'min_samples_for_conclusion': 3,
        'gap_break_factor': 2.5,
        'gap_percentile': 25,
        'metrics': {},
    },
}

_config: dict[str, Any] = {}
_loaded: bool = False


class ConfigValidationError(ValueError):
    """配置文件包含无法安全运行的参数。"""


def _get_base_dir() -> str:
    """跨模式（开发 / PyInstaller）根目录：frozen → _MEIPASS，开发 → 本文件所在目录。"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


def _find_config_path() -> str:
    """跨模式（开发 / PyInstaller 打包）定位 config.yaml。"""
    # 优先从根目录查找
    path = os.path.join(_get_base_dir(), 'config.yaml')
    if os.path.isfile(path):
        return path
    # 兜底
    return os.path.join(os.getcwd(), 'config.yaml')


def _validate_config(cfg: dict[str, Any]) -> None:
    """校验会影响分段、性能与真值对齐结果的关键配置。"""
    errors: list[str] = []

    def _number(key: str, minimum: float | None = None) -> None:
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, Real):
            errors.append(f'{key} 必须是数值')
        elif minimum is not None and value < minimum:
            operator = '大于等于' if minimum == 0 else '大于'
            limit = minimum if minimum != 0 else 0
            errors.append(f'{key} 必须{operator} {limit}')

    for key in [
        'SAMPLING_PERIOD_MS', 'GAP_THRESHOLD', 'MIN_TRAJ_FRAMES',
        'TIMESTAMP_WINDOW_SEC', 'MAX_OVERLAY_CURVES', 'CSV_CHUNK_THRESHOLD_MB',
        'CSV_CHUNK_SIZE_ROWS', 'MAX_TRACK_SPEED', 'DISPLAY_MAX_POINTS',
    ]:
        _number(key, 1)
    for key in ['POS_JUMP_THRESHOLD', 'CACHE_DEFAULT_TIMEOUT']:
        _number(key, 0)

    wrap_low = cfg.get('WRAP_LOW')
    wrap_high = cfg.get('WRAP_HIGH')

    renderer = cfg.get('GRAPH_RENDERER')
    if renderer not in ('webgl', 'svg'):
        errors.append('GRAPH_RENDERER 必须是 "webgl" 或 "svg"')

    if (isinstance(wrap_low, bool) or not isinstance(wrap_low, Real)
            or isinstance(wrap_high, bool) or not isinstance(wrap_high, Real)):
        errors.append('WRAP_LOW 和 WRAP_HIGH 必须是数值')
    elif not (0 <= wrap_low <= wrap_high <= 255):
        errors.append('WRAP_LOW/WRAP_HIGH 必须满足 0 ≤ WRAP_LOW ≤ WRAP_HIGH ≤ 255')

    comparison = cfg.get('comparison', {})
    if not isinstance(comparison, dict):
        errors.append('comparison 必须是对象')
    else:
        def _cmp_number(key: str, minimum: float | None = None, maximum: float | None = None) -> None:
            if key not in comparison:
                return
            value = comparison[key]
            if isinstance(value, bool) or not isinstance(value, Real):
                errors.append(f'comparison.{key} 必须是数值')
            elif minimum is not None and value < minimum:
                errors.append(f'comparison.{key} 必须大于等于 {minimum}')
            elif maximum is not None and value > maximum:
                errors.append(f'comparison.{key} 必须小于等于 {maximum}')

        for key in ['match_threshold', 'time_gate_ms', 'delay_scan_step',
                    'delay_min_matched_frames', 'coord_bias_threshold']:
            _cmp_number(key, 0 if key in {'time_gate_ms', 'coord_bias_threshold'} else 1)
        _cmp_number('delay_insensitive_ratio', 0, 1)
        _cmp_number('delay_min_match_rate', 0, 1)
        _cmp_number('radar_gap_break_factor', 1)

        delay_range = comparison.get('delay_scan_range')
        if delay_range is not None:
            if (not isinstance(delay_range, (list, tuple)) or len(delay_range) != 2
                    or any(isinstance(v, bool) or not isinstance(v, Real) for v in delay_range)
                    or delay_range[0] > delay_range[1]):
                errors.append('comparison.delay_scan_range 必须是递增的两个数值')

        bins = comparison.get('distance_bins')
        if bins is not None:
            if (not isinstance(bins, (list, tuple)) or len(bins) < 2
                    or any(isinstance(v, bool) or not isinstance(v, Real) for v in bins)
                    or any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1))):
                errors.append('comparison.distance_bins 必须是至少两个严格递增的数值')

        matching = comparison.get('id_matching')
        if matching is not None:
            if not isinstance(matching, dict):
                errors.append('comparison.id_matching 必须是对象')
            else:
                for key in ['min_frames', 'min_pair_frames', 'track_age_confirm_frames']:
                    value = matching.get(key)
                    if isinstance(value, bool) or not isinstance(value, Real) or value < 1:
                        errors.append(f'comparison.id_matching.{key} 必须大于等于 1')
                for key in [
                    'min_speed_mps', 'min_accel_mps2', 'min_displacement_m',
                    'track_age_gap_ms', 'track_age_reset_max', 'lifecycle_position_jump_m',
                ]:
                    value = matching.get(key)
                    if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
                        errors.append(f'comparison.id_matching.{key} 必须大于等于 0')
                coverage = matching.get('min_pair_coverage')
                if (isinstance(coverage, bool) or not isinstance(coverage, Real)
                        or coverage < 0 or coverage > 1):
                    errors.append('comparison.id_matching.min_pair_coverage 必须在 0 到 1 之间')

    _validate_performance_metrics(cfg, errors)

    if errors:
        raise ConfigValidationError('配置校验失败：' + '；'.join(errors))


_LIMIT_MODES = {
    'absolute',
    'percent',
    'max_absolute_percent',
    'scaled_max_absolute_percent',
}


def _validate_limit(metric: str, where: str, limit: Any, errors: list[str]) -> None:
    """校验单个阈值规则：模式合法、百分比非负、倍率大于零。"""
    prefix = f'performance_metrics.metrics.{metric}.{where}'
    if not isinstance(limit, dict):
        errors.append(f'{prefix} 必须是对象')
        return
    mode = limit.get('mode')
    if mode not in _LIMIT_MODES:
        errors.append(f'{prefix}.mode 必须是 {sorted(_LIMIT_MODES)} 之一')
        return

    needs_absolute = mode in {'absolute', 'max_absolute_percent', 'scaled_max_absolute_percent'}
    needs_percent = mode in {'percent', 'max_absolute_percent', 'scaled_max_absolute_percent'}

    if needs_absolute:
        value = limit.get('absolute')
        if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
            errors.append(f'{prefix}.absolute 必须是非负数值')
    if needs_percent:
        value = limit.get('percent')
        if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
            errors.append(f'{prefix}.percent 必须是非负数值')
    if mode == 'scaled_max_absolute_percent':
        value = limit.get('scale')
        if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
            errors.append(f'{prefix}.scale 必须是大于 0 的数值')


def _validate_performance_metrics(cfg: dict[str, Any], errors: list[str]) -> None:
    """校验性能指标评估配置（距离区间连续不重叠、单位存在、阈值参数合法）。"""
    perf = cfg.get('performance_metrics')
    if perf is None:
        return
    if not isinstance(perf, dict):
        errors.append('performance_metrics 必须是对象')
        return

    def _number(key: str, minimum: float | None = None,
                maximum: float | None = None) -> None:
        if key not in perf:
            return
        value = perf[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            errors.append(f'performance_metrics.{key} 必须是数值')
        elif minimum is not None and value < minimum:
            errors.append(f'performance_metrics.{key} 必须大于等于 {minimum}')
        elif maximum is not None and value > maximum:
            errors.append(f'performance_metrics.{key} 必须小于等于 {maximum}')

    _number('accuracy_requirement', 0, 1)
    _number('three_frame_requirement', 0)
    _number('min_samples_for_conclusion', 1)
    _number('gap_break_factor', 1)
    _number('gap_percentile', 0, 100)

    bins = perf.get('distance_bins')
    if bins is not None:
        if (not isinstance(bins, (list, tuple)) or len(bins) < 2
                or any(isinstance(v, bool) or not isinstance(v, Real) for v in bins)
                or any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1))):
            errors.append('performance_metrics.distance_bins 必须是至少两个严格递增的数值')

    valid_range = perf.get('valid_distance_range')
    if valid_range is not None:
        if (not isinstance(valid_range, (list, tuple)) or len(valid_range) != 2
                or any(isinstance(v, bool) or not isinstance(v, Real) for v in valid_range)
                or valid_range[0] >= valid_range[1]):
            errors.append('performance_metrics.valid_distance_range 必须是递增的两个数值')

    metrics = perf.get('metrics')
    if metrics is None:
        return
    if not isinstance(metrics, dict):
        errors.append('performance_metrics.metrics 必须是对象')
        return

    for name, rule in metrics.items():
        prefix = f'performance_metrics.metrics.{name}'
        if not isinstance(rule, dict):
            errors.append(f'{prefix} 必须是对象')
            continue
        if not rule.get('unit'):
            errors.append(f'{prefix}.unit 必须存在')
        mode = rule.get('mode')
        if mode not in {'binned', 'single_limit'}:
            errors.append(f'{prefix}.mode 必须是 "binned" 或 "single_limit"')
            continue

        if mode == 'single_limit':
            value = rule.get('abs_limit')
            if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
                errors.append(f'{prefix}.abs_limit 必须是非负数值')
            if rule.get('operator') not in {'<=', '<'}:
                errors.append(f'{prefix}.operator 必须是 "<=" 或 "<"')
            continue

        # binned：区间必须连续、不重叠，且与 distance_bins 对齐
        if rule.get('percent_basis') not in {'distance', 'truth'}:
            errors.append(f'{prefix}.percent_basis 必须是 "distance" 或 "truth"')
        rule_bins = rule.get('bins')
        if not isinstance(rule_bins, (list, tuple)) or not rule_bins:
            errors.append(f'{prefix}.bins 必须是非空列表')
            continue
        previous_hi = None
        for index, entry in enumerate(rule_bins):
            where = f'bins[{index}]'
            if not isinstance(entry, dict):
                errors.append(f'{prefix}.{where} 必须是对象')
                continue
            span = entry.get('range')
            if (not isinstance(span, (list, tuple)) or len(span) != 2
                    or any(isinstance(v, bool) or not isinstance(v, Real) for v in span)
                    or span[0] >= span[1]):
                errors.append(f'{prefix}.{where}.range 必须是递增的两个数值')
            elif previous_hi is not None and span[0] != previous_hi:
                errors.append(f'{prefix}.{where}.range 必须与上一区间连续（上一上界 {previous_hi}）')
            elif previous_hi is None and bins is not None and span[0] != bins[0]:
                errors.append(f'{prefix}.{where}.range 起点必须等于 distance_bins 首元素')
            if isinstance(span, (list, tuple)) and len(span) == 2 and all(
                    isinstance(v, Real) and not isinstance(v, bool) for v in span):
                previous_hi = span[1]
            _validate_limit(name, f'{where}.normal_limit', entry.get('normal_limit'), errors)
            _validate_limit(name, f'{where}.three_frame_limit',
                            entry.get('three_frame_limit'), errors)
        if bins is not None and previous_hi is not None and previous_hi != bins[-1]:
            errors.append(f'{prefix}.bins 末区间上界必须等于 distance_bins 末元素 {bins[-1]}')


def _load_config() -> dict[str, Any]:
    """从 config.yaml 加载配置，文件不存在时使用默认值。"""
    global _config, _loaded
    if _loaded:
        return _config

    config_path = _find_config_path()
    cfg = deepcopy(_DEFAULTS)

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data and not isinstance(yaml_data, dict):
            raise ConfigValidationError('配置文件根节点必须是对象')
        if yaml_data:
            # 深度合并：yaml 中的值覆盖默认值
            _deep_update(cfg, yaml_data)

    _validate_config(cfg)
    _config = cfg
    _loaded = True
    return _config


def _deep_update(base: dict, override: dict) -> None:
    """递归合并 override 到 base 中。"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def get(key: str, default: Any = None) -> Any:
    """获取单个配置项。"""
    cfg = _load_config()
    return cfg.get(key, default)


def reload() -> dict[str, Any]:
    """强制重新加载配置。"""
    global _loaded
    _loaded = False
    return _load_config()
