"""
修正后轨迹分段算法 - 全量验证脚本
处理多ID、ID复用、uint8回绕的完整场景
"""
import pandas as pd
import numpy as np

CSV_PATH = r'E:\feishuDownLoad\390s\行车路试\20260623-巡库-5.0.001b.0015.AR21\CD701_flr_track_2026_06_23_10_31_42.csv'

WRAP_THRESHOLD = 250   # Track_Age >= 此值时的下降视为回绕
WRAP_RESET_MAX = 5     # 回绕后 Track_Age <= 此值视为回绕延续
SAMPLING_PERIOD_MS = 50
GAP_THRESHOLD_MS = SAMPLING_PERIOD_MS * 10  # 10倍采样周期 = 500ms

def load_and_preprocess(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df = df.dropna(subset=['timestamp', 'ID', 'Track_Age'])
    df['Track_Age'] = df['Track_Age'].astype(int)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def parse_timestamp_ms(ts_str):
    """将 2026_06_23_10_31_42_504 格式转为毫秒整数"""
    s = str(ts_str).replace('_', '').strip()
    # 取最后3位作为毫秒, 其余作为秒级部分
    return int(s)

def detect_breakpoints(sub_df):
    """
    检测单ID分组内的所有断点。
    返回 (wrap_points, reuse_points, gap_points, all_breaks)
    """
    ages = sub_df['Track_Age'].values.astype(int)
    ts_raw = sub_df['timestamp'].values
    ts_ms = np.array([parse_timestamp_ms(t) for t in ts_raw])

    wrap_points = []
    reuse_points = []
    gap_points = []

    for i in range(1, len(ages)):
        diff = ages[i] - ages[i-1]

        # 规则B: uint8回绕检测 (255->0 或 254->0 等接近上限的下降)
        if diff < 0 and ages[i-1] >= WRAP_THRESHOLD and ages[i] <= WRAP_RESET_MAX:
            wrap_points.append(i)
            continue

        # 规则C: 非回绕下降 = ID复用
        if diff < 0:
            reuse_points.append(i)
            continue

        # 规则D: 时间间隔过大
        time_diff = ts_ms[i] - ts_ms[i-1]
        if time_diff > GAP_THRESHOLD_MS:
            gap_points.append(i)
            continue

    # 合并所有断点(仅reuse和gap是切分点,wrap不切分)
    segment_breaks = sorted(set(reuse_points + gap_points))

    return wrap_points, reuse_points, gap_points, segment_breaks

def segment_trajectories(df):
    """修正后的轨迹分段算法"""
    all_segments = []
    seg_counter = {}
    breakpoint_details = []

    for id_val in sorted(df['ID'].unique()):
        sub = df[df['ID'] == id_val].reset_index(drop=True)
        ages = sub['Track_Age'].values.astype(int)

        wraps, reuses, gaps, seg_breaks = detect_breakpoints(sub)

        seg_counter[id_val] = 0

        # 规则A: ID首出现 = 轨迹起点(无论Track_Age值)
        if len(seg_breaks) == 0:
            # 无断点 → 单条轨迹
            seg_counter[id_val] += 1
            traj_id = f"{id_val}_seg{seg_counter[id_val]}"
            segment = {
                'trajectory_id': traj_id,
                'original_id': id_val,
                'start_idx': 0,
                'end_idx': len(sub) - 1,
                'total_frames': len(sub),
                'first_track_age': ages[0],
                'max_raw_age': ages.max(),
                'num_wraps': len(wraps),
                'num_reuses': len(reuses),
                'num_gaps': len(gaps),
            }
            all_segments.append(segment)
        else:
            # 有断点 → 按断点切分
            start_idx = 0
            for break_idx in seg_breaks:
                seg_counter[id_val] += 1
                traj_id = f"{id_val}_seg{seg_counter[id_val]}"

                # 计算此段内的wrap数
                seg_wraps = sum(1 for w in wraps if start_idx <= w < break_idx)
                seg_ages = ages[start_idx:break_idx]

                segment = {
                    'trajectory_id': traj_id,
                    'original_id': id_val,
                    'start_idx': start_idx,
                    'end_idx': break_idx - 1,
                    'total_frames': break_idx - start_idx,
                    'first_track_age': seg_ages[0] if len(seg_ages) > 0 else None,
                    'max_raw_age': seg_ages.max() if len(seg_ages) > 0 else None,
                    'num_wraps': seg_wraps,
                    'num_reuses': 0,
                    'num_gaps': 0,
                }
                all_segments.append(segment)

                # 记录断点详情
                bp_type = 'reuse' if break_idx in reuses else 'gap'
                bp_detail = {
                    'id': id_val,
                    'break_idx': break_idx,
                    'type': bp_type,
                    'age_before': ages[break_idx - 1],
                    'age_after': ages[break_idx],
                    'dx_before': sub.iloc[break_idx-1]['Dx'],
                    'dx_after': sub.iloc[break_idx]['Dx'],
                    'dy_before': sub.iloc[break_idx-1]['Dy'],
                    'dy_after': sub.iloc[break_idx]['Dy'],
                    'ts_before': sub.iloc[break_idx-1]['timestamp'],
                    'ts_after': sub.iloc[break_idx]['timestamp'],
                }
                breakpoint_details.append(bp_detail)

                start_idx = break_idx

            # 最后一段
            seg_counter[id_val] += 1
            traj_id = f"{id_val}_seg{seg_counter[id_val]}"
            seg_wraps = sum(1 for w in wraps if start_idx <= w)
            seg_ages = ages[start_idx:]

            segment = {
                'trajectory_id': traj_id,
                'original_id': id_val,
                'start_idx': start_idx,
                'end_idx': len(sub) - 1,
                'total_frames': len(sub) - start_idx,
                'first_track_age': seg_ages[0] if len(seg_ages) > 0 else None,
                'max_raw_age': seg_ages.max() if len(seg_ages) > 0 else None,
                'num_wraps': seg_wraps,
                'num_reuses': 0,
                'num_gaps': 0,
            }
            all_segments.append(segment)

    return pd.DataFrame(all_segments), pd.DataFrame(breakpoint_details)


def compare_with_old_rule(df):
    """对比原错误规则(Track_Age==1分段)的结果"""
    old_segments = []
    for id_val in sorted(df['ID'].unique()):
        sub = df[df['ID'] == id_val].reset_index(drop=True)
        ages = sub['Track_Age'].values.astype(int)

        # 原规则: Track_Age==1 出现时切分
        ones = np.where(ages == 1)[0]
        break_points = sorted(set([0] + list(ones) + [len(sub)]))

        count = 0
        for i in range(len(break_points) - 1):
            count += 1
            s = break_points[i]
            e = break_points[i+1]
            old_segments.append({
                'trajectory_id': f"{id_val}_old{count}",
                'original_id': id_val,
                'total_frames': e - s,
                'first_track_age': ages[s],
            })

    return pd.DataFrame(old_segments)


if __name__ == '__main__':
    df = load_and_preprocess(CSV_PATH)

    print(f"数据总行数: {len(df)}")
    print(f"唯一ID数量: {len(df['ID'].unique())}")
    print()

    # 修正算法分段
    seg_df, bp_df = segment_trajectories(df)

    print("=" * 70)
    print("修正算法分段结果")
    print("=" * 70)
    print(f"总轨迹段数: {len(seg_df)}")
    print()

    # 按原始ID分组显示
    for id_val in sorted(seg_df['original_id'].unique()):
        id_segs = seg_df[seg_df['original_id'] == id_val]
        print(f"ID={id_val}: {len(id_segs)} 段")
        for _, seg in id_segs.iterrows():
            wrap_note = f" (含{seg['num_wraps']}次回绕)" if seg['num_wraps'] > 0 else ""
            print(f"  {seg['trajectory_id']}: {seg['total_frames']}帧, "
                  f"首Age={seg['first_track_age']}, maxAge={seg['max_raw_age']}{wrap_note}")

    # 断点详情
    if len(bp_df) > 0:
        print()
        print("=" * 70)
        print("断点详情 (ID复用 / 时间间隔)")
        print("=" * 70)
        for _, bp in bp_df.iterrows():
            dx_change = abs(bp['dx_after'] - bp['dx_before'])
            dy_change = abs(bp['dy_after'] - bp['dy_before'])
            print(f"  ID={bp['id']} idx={bp['break_idx']} [{bp['type']}] "
                  f"Age: {bp['age_before']}→{bp['age_after']} "
                  f"Dx: {bp['dx_before']}→{bp['dx_after']} (Δ={dx_change:.1f}) "
                  f"Dy: {bp['dy_before']}→{bp['dy_after']} (Δ={dy_change:.1f})")

    # 原错误规则对比
    old_df = compare_with_old_rule(df)

    print()
    print("=" * 70)
    print("原错误规则(Track_Age==1分段)对比")
    print("=" * 70)
    print(f"修正算法轨迹段数: {len(seg_df)}")
    print(f"原错误规则轨迹段数: {len(old_df)}")

    # 统计差异最大的ID
    for id_val in sorted(df['ID'].unique()):
        new_count = len(seg_df[seg_df['original_id'] == id_val])
        old_count = len(old_df[old_df['original_id'] == id_val])
        if new_count != old_count:
            print(f"  ID={id_val}: 修正={new_count}段 vs 原规则={old_count}段 "
                  f"{'← 差异!' if new_count != old_count else ''}")

    # 统计原规则遗漏的轨迹(首帧Age!=1且无Age==1的段)
    no_age1_segs = old_df[old_df['first_track_age'] != 1]
    print(f"\n原规则中首帧Track_Age!=1的段(可能被误判): {len(no_age1_segs)}")
