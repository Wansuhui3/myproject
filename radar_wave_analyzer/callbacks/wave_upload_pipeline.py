"""波动分析上传解析纯数据管线：多文件解码 → 负载展开 → 合并去重 → 分段与来源缓存。

本模块不导入 Dash（UI 组装见 wave_upload_callbacks）：
  - parse_upload_payloads  解码上传内容并展开为带来源标注的 DataFrame 列表；
  - build_upload_caches    合并去重、全局分段一次扫描，并写入 combined
                           与各来源独立缓存（避免 N 来源重复分段）。
"""
import logging

import numpy as np
import pandas as pd

from ..cache import (
    clear_data_cache,
    set_data_cache,
)
from ..core.data_loader import identify_radar_source, load_csv_from_bytes
from ..core.json_loader import is_json_filename, load_json_from_bytes
from ..core.segmenter import segment_trajectories
from .helpers import _decode_upload_contents

logger = logging.getLogger(__name__)

# 分段缓存只保存源行号，不长期保留每段 DataFrame 的重复副本。
_SOURCE_ROW_COLUMN = '__source_row_index__'


def parse_upload_payloads(contents_list, filenames):
    """解码上传内容并展开为带来源标注的 DataFrame 列表。

    Returns:
        (all_dfs, errors, source_files)
        - all_dfs: 标注了 file_index / radar_source_* 列的 DataFrame 列表；
        - errors: 解析失败消息列表（含空文件提示）；
        - source_files: 来源汇总 {radar_source_group: {key, short_label,
          recognized, filenames}}。
    """
    all_dfs = []
    errors = []
    source_files: dict[str, dict] = {}
    # file_index 按展开后条目递增：JSON 单文件多话题（FLR/RLR 共存）时，
    # 每个话题独立编号，防止不同雷达同 ID 同时间戳的数据在合并去重时被误删。
    file_seq = 0
    for content, fn in zip(contents_list, filenames):
        try:
            data_bytes = _decode_upload_contents(content)
            if is_json_filename(fn):
                json_result = load_json_from_bytes(data_bytes, fn)
                # 错误消息已含 "文件名·话题名" 前缀，无需重复包装
                errors.extend(json_result['errors'])
                # payload: (DataFrame, 来源识别名, 溯源显示名)
                payloads = [
                    (topic['df'], topic['topic'], topic['label'])
                    for topic in json_result['topics']
                ]
            else:
                df = load_csv_from_bytes(data_bytes, fn)
                payloads = [(df, str(fn), str(fn))]
        except ValueError as e:
            errors.append(f'{fn}: {e}')
            continue
        except Exception as e:
            errors.append(f'{fn}: {e}')
            continue

        for df, source_name, display_name in payloads:
            # 空白文件（0 字节/仅表头/全空行）会解析出 0 行 DataFrame，
            # 必须跳过，否则后续 df['radar_source_group'].iloc[0] 会抛
            # IndexError 导致回调 500、前端一直显示加载中。
            if df is None or len(df) == 0:
                errors.append(f'{display_name}: 文件内容为空或无有效数据行')
                continue
            # JSON 用话题名识别雷达来源（如 RLR_OBJ_object 命中既有 rlr 规则）；
            # CSV 保持按文件名识别，行为与历史版本一致。
            source = identify_radar_source(source_name)
            df['file_index'] = file_seq
            df['source_filename'] = display_name
            df['radar_source_key'] = str(source['key'])
            df['radar_source_label'] = str(source['label'])
            df['radar_source_short_label'] = str(source['short_label'])
            df['radar_source_recognized'] = bool(source['recognized'])
            # 未识别来源互相隔离（按条目粒度），防止未知来源的同号 ID 重新交织。
            df['radar_source_group'] = (
                str(source['key']) if source['recognized'] else f'unknown_file_{file_seq}'
            )
            summary_key = str(df['radar_source_group'].iloc[0])
            source_summary = source_files.setdefault(summary_key, {
                'key': str(source['key']),
                'short_label': str(source['short_label']),
                'recognized': bool(source['recognized']),
                'filenames': [],
            })
            source_summary['filenames'].append(display_name)
            all_dfs.append(df)
            file_seq += 1

    return all_dfs, errors, source_files


def build_upload_caches(all_dfs: list, upload_label: str):
    """合并去重 → 全局分段（单次扫描）→ 写 combined 与各来源独立缓存。

    全局分段的结果同时是各来源缓存的唯一来源：不能再为每个来源重复
    segment_trajectories()，否则上传 N 个来源会额外执行 N 次完整分段扫描。

    Returns:
        (merged_df, meta_df, segments, source_keys)
    """
    if len(all_dfs) == 1:
        merged_df = all_dfs[0]
    else:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        # 跨文件、跨雷达的相同 ID/时间戳均保留；仅文件内部去重。
        merged_df = merged_df.drop_duplicates(
            subset=['file_index', 'ID', 'timestamp_parsed'], keep='first')
        merged_df = merged_df.sort_values('timestamp_parsed').reset_index(drop=True)

    merged_df[_SOURCE_ROW_COLUMN] = np.arange(len(merged_df), dtype=np.int64)
    meta_df, segments = segment_trajectories(merged_df)
    # segment_trajectories 已为每段保留独立副本，故可从主表移除内部列；
    # set_data_cache 仍能从 segments 压缩出行号索引。
    merged_df.drop(columns=[_SOURCE_ROW_COLUMN], inplace=True)

    # 混合上传时不能把 FLR+RLR 合并数据只写入当前一个 radar_key。
    # 否则切换到另一个雷达源时会命中空缓存，主可视化页面被清空。
    # 保留一个 combined 缓存，同时为每个已识别来源建立独立缓存。
    clear_data_cache()
    set_data_cache(upload_label, 'combined', merged_df, meta_df, segments)
    source_keys = {
        str(value).strip()
        for value in merged_df.get('radar_source_key', pd.Series(dtype=str)).dropna().unique()
        if str(value).strip()
    }
    for source_key in sorted(source_keys):
        # merged_df 的索引仍是全局源行号；为来源副本显式保留该映射，
        # 之后再 reset_index 生成其本地 iloc 行号。
        source_df = merged_df[
            merged_df['radar_source_key'].astype(str) == source_key
        ].copy()
        source_df[_SOURCE_ROW_COLUMN] = source_df.index.to_numpy(dtype=np.int64)
        source_df.reset_index(drop=True, inplace=True)
        if source_df.empty:
            continue

        # 将全局段内的源行号映射为 source_df 的本地 iloc 行号。这样既能
        # 保留 get_segment() 的紧凑索引契约，又不需要为来源副本重新分段。
        source_meta = meta_df[
            meta_df['radar_source_key'].astype(str) == source_key
        ].copy()
        global_to_local = pd.Series(
            np.arange(len(source_df), dtype=np.int64),
            index=source_df[_SOURCE_ROW_COLUMN].to_numpy(dtype=np.int64),
        )
        source_segments = {}
        for trajectory_id in source_meta.get('trajectory_id', pd.Series(dtype=str)):
            global_segment = segments.get(trajectory_id)
            if global_segment is None:
                continue
            local_rows = global_to_local.reindex(
                global_segment[_SOURCE_ROW_COLUMN].to_numpy(dtype=np.int64)
            ).to_numpy()
            if np.isnan(local_rows).any():
                logger.warning('来源 %s 的轨迹段 %s 索引映射不完整，已跳过',
                               source_key, trajectory_id)
                continue
            source_segments[trajectory_id] = pd.DataFrame({
                _SOURCE_ROW_COLUMN: local_rows.astype(np.int64),
            })
        source_df.drop(columns=[_SOURCE_ROW_COLUMN], inplace=True)
        set_data_cache(upload_label, source_key, source_df, source_meta, source_segments)

    return merged_df, meta_df, segments, source_keys
