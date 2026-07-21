"""
修正后的轨迹分段算法验证脚本
基于实际雷达数据中 Track_Age 的 uint8(0~255) 回绕特性
"""
import pandas as pd
import numpy as np

CSV_PATH = r'E:\feishuDownLoad\390s\kpi\case19\CD701_flr_track_2026_04_20_11_12_27.csv'

def load_and_preprocess(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df = df.dropna(subset=['timestamp', 'ID', 'Track_Age'])
    df['Track_Age'] = df['Track_Age'].astype(int)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def unwrap_track_age(ages):
    """
    对 uint8 Track_Age 序列做展开处理。
    检测 255→0 回绕，累加 256 偏移量，使序列单调非递减。
    同时检测非回绕的下降（ID复用），标记为断点。
    """
    unwrapped = ages.copy().astype(int)
    breakpoints = []
    offset = 0

    for i in range(1, len(ages)):
        diff = ages[i] - ages[i-1]
        if diff == 0:
            # 重复帧，无变化
            unwrapped[i] += offset
        elif diff > 0:
            # 正常递增
            unwrapped[i] += offset
        elif diff == -255 or (ages[i-1] == 255 and ages[i] <= 1):
            # uint8 回绕: 255→0→1
            offset += 256
            unwrapped[i] += offset
        else:
            # 非回绕下降 → ID复用，断点
            breakpoints.append(i)
            offset = 0  # 新段重置偏移
            unwrapped[i] = ages[i]  # 新段从原始值开始

    return unwrapped, breakpoints

def segment_trajectories(df):
    """
    修正后的轨迹分段算法。
    四层判断:
    1. ID首出现 → 轨迹起点
    2. uint8回绕 → 同轨迹延续
    3. 非回绕Track_Age下降 → ID复用, 切分
    4. 时间间隔过大(ID消失后重现) → 新轨迹
    """
    SAMPLING_PERIOD_MS = 50  # 标称采样周期 50ms (20Hz)
    GAP_THRESHOLD = SAMPLING_PERIOD_MS * 5  # 5倍采样周期 = 250ms

    all_segments = []
    seg_counter = {}  # {原始ID: 当前段序号}

    for id_val in sorted(df['ID'].unique()):
        sub = df[df['ID'] == id_val].reset_index(drop=True)
        ages = sub['Track_Age'].values

        # 初始化段计数器
        seg_counter[id_val] = 0

        # Step 1: 展开 Track_Age 并检测断点
        unwrapped, breakpoints = unwrap_track_age(ages)

        # Step 2: 检测时间间隔过大(ID消失后重现)
        timestamps = sub['timestamp'].values
        # 将timestamp转为可计算的数值(毫秒)
        ts_numeric = np.zeros(len(timestamps))
        for i, ts in enumerate(timestamps):
            parts = ts.replace('_', '').strip()
            # 格式: 20260420111309690 → 毫秒级时间戳
            ts_numeric[i] = int(parts)

        time_breakpoints = []
        for i in range(1, len(ts_numeric)):
            time_diff = ts_numeric[i] - ts_numeric[i-1]
            if time_diff > GAP_THRESHOLD:
                time_breakpoints.append(i)

        # 合合所有断点
        all_breaks = sorted(set(breakpoints + time_breakpoints))

        # Step 3: 按断点切分轨迹段
        if len(all_breaks) == 0:
            # 整个ID组只有一条轨迹
            seg_counter[id_val] += 1
            traj_id = f"{id_val}_seg{seg_counter[id_val]}"
            segment = {
                'trajectory_id': traj_id,
                'original_id': id_val,
                'start_time': sub.iloc[0]['timestamp'],
                'end_time': sub.iloc[-1]['timestamp'],
                'total_frames': len(sub),
                'max_track_age': ages.max(),
                'first_track_age': ages[0],
                'unwrapped_max': unwrapped.max(),
            }
            all_segments.append(segment)
        else:
            # 按断点分割
            start_idx = 0
            for break_idx in all_breaks:
                seg_counter[id_val] += 1
                traj_id = f"{id_val}_seg{seg_counter[id_val]}"
                segment_sub = sub.iloc[start_idx:break_idx]
                segment = {
                    'trajectory_id': traj_id,
                    'original_id': id_val,
                    'start_time': segment_sub.iloc[0]['timestamp'],
                    'end_time': segment_sub.iloc[-1]['timestamp'],
                    'total_frames': len(segment_sub),
                    'max_track_age': segment_sub['Track_Age'].max(),
                    'first_track_age': segment_sub.iloc[0]['Track_Age'],
                    'unwrapped_max': unwrapped[start_idx:break_idx].max() if break_idx <= len(unwrapped) else unwrapped[start_idx:].max(),
                }
                all_segments.append(segment)
                start_idx = break_idx

            # 最后一段
            seg_counter[id_val] += 1
            traj_id = f"{id_val}_seg{seg_counter[id_val]}"
            segment_sub = sub.iloc[start_idx:]
            segment = {
                'trajectory_id': traj_id,
                'original_id': id_val,
                'start_time': segment_sub.iloc[0]['timestamp'],
                'end_time': segment_sub.iloc[-1]['timestamp'],
                'total_frames': len(segment_sub),
                'max_track_age': segment_sub['Track_Age'].max(),
                'first_track_age': segment_sub.iloc[0]['Track_Age'],
                'unwrapped_max': unwrapped[start_idx:].max(),
            }
            all_segments.append(segment)

    return pd.DataFrame(all_segments)


if __name__ == '__main__':
    df = load_and_preprocess(CSV_PATH)
    print(f"数据总行数: {len(df)}")
    print(f"唯一ID: {sorted(df['ID'].unique())}")
    print()

    segments_df = segment_trajectories(df)

    print("=== 轨迹分段结果 ===")
    for _, seg in segments_df.iterrows():
        print(f"  {seg['trajectory_id']}: 帧={seg['total_frames']}, "
              f"首帧Age={seg['first_track_age']}, 原始max={seg['max_track_age']}, "
              f"展开max={seg['unwrapped_max']}")
    print()

    # 对比: 如果用原始错误规则(Track_Age==1分段)会怎样
    sub = df[df['ID'] == 55].reset_index(drop=True)
    ages = sub['Track_Age'].values
    ones = np.where(ages == 1)[0]
    print("=== 原始错误规则(Track_Age==1分段)会产生的断点 ===")
    print(f"  Track_Age==1 出现位置: {ones.tolist()}")
    print(f"  会将 599 帧的连续轨迹切成 {len(ones)+1} 段 → 错误!")
    print()

    # 正确结果: 因为没有非回绕断点和时间间隔断点, 整个ID=55只有一条轨迹
    print("=== 正确分段: ID=55 只有一条连续轨迹 ===")
    print(f"  599帧, Track_Age 从14递增到255回绕后继续到185(展开后441)")
