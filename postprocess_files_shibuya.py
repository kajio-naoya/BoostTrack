import os
import pandas as pd
import numpy as np
from postprocess_extract_oneedge import extract_oneedge
from postprocess_remove_shorts import remove_shorts

"""
base_dir/<video_name>/<video_name>.txt にあるすべてのtxtファイルについて、短小軌道削除+特定軌道の人の抽出+px-m変換を行う
"""

def is_in_area1_a(x, y):
    return x < 1120 and y < 720
def is_in_area2_a(x, y):
    return x < 1900 and y > 1660
def is_in_area1_b(x, y):
    return x < 1060 and y < 720
def is_in_area2_b(x, y):
    return x < 2060 and y > 1600
px_per_m_a = 37
px_per_m_b = 40
FRAMES_THRESHOLD = 300

base_dir = "output/shibuya_all"

def downsample_to_5fps(input_file, output_file, downsample_rate=6):
    df = pd.read_csv(input_file)
    df = df[df['frame'] % downsample_rate == 0]
    df['frame'] = df['frame'] // downsample_rate
    df.to_csv(output_file, index=False)

def px_to_m_and_add_velocities_and_remove_unnecessary_columns(input_file, output_file, px_per_m):
    df = pd.read_csv(input_file)[["frame", "id", "x", "y"]]
    df["x"] = df["x"] / px_per_m
    df["y"] = df["y"] / px_per_m
    df["vx"] = df.groupby("id")["x"].diff().fillna(0)
    df["vy"] = df.groupby("id")["y"].diff().fillna(0)
    df.to_csv(output_file, index=False)

def interpolate_missing_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df_per_person = df.groupby("id")
    interpolated_dfs = []
    
    for person_id, df_id in df_per_person:
        df_id = df_id.sort_values("frame").copy()
        min_frame = df_id["frame"].min()
        max_frame = df_id["frame"].max()
        
        # 欠けているフレームを追加
        missing_frames = []
        for frame in range(min_frame, max_frame + 1):
            if frame not in df_id["frame"].values:
                missing_frames.append({
                    "frame": frame,
                    "id": person_id,
                    "x": np.nan,
                    "y": np.nan,
                    "vx": np.nan,
                    "vy": np.nan
                })
        
        if missing_frames:
            missing_df = pd.DataFrame(missing_frames)
            df_id = pd.concat([df_id, missing_df], ignore_index=True)
        
        # フレーム順にソート
        df_id = df_id.sort_values("frame").reset_index(drop=True)
        
        # 補間を実行
        df_id["x"] = df_id["x"].interpolate()
        df_id["y"] = df_id["y"].interpolate()
        df_id["vx"] = df_id["vx"].interpolate()
        df_id["vy"] = df_id["vy"].interpolate()
        
        interpolated_dfs.append(df_id)
    
    # 全員のデータを結合して保存
    if interpolated_dfs:
        result_df = pd.concat(interpolated_dfs, ignore_index=True)
        result_df = result_df.sort_values(["frame", "id"]).reset_index(drop=True)
        result_df.to_csv(output_file, index=False)

def rename_columns(input_file, output_file):
    df = pd.read_csv(input_file)
    df.columns = ["Frame", "NPCID", "X", "Y", "VX", "VY"]
    df.to_csv(output_file, index=False)

for video_name in os.listdir(base_dir):
    video_dir = os.path.join(base_dir, video_name)
    for csv_file in os.listdir(video_dir):
        if csv_file.endswith(".csv") and not csv_file.endswith("_postprocessed.csv"):
            input_file = os.path.join(video_dir, csv_file)
            output_file = os.path.join(video_dir, csv_file.replace(".csv", "_postprocessed.csv"))
            remove_shorts(input_file, output_file, FRAMES_THRESHOLD)
            # カメラ位置の違いで設定値を分岐
            if "DSCF0027" in video_name or "DSCF0028" in video_name:
                is_in_area1 = is_in_area1_a
                is_in_area2 = is_in_area2_a
                px_per_m = px_per_m_a
            else:
                is_in_area1 = is_in_area1_b
                is_in_area2 = is_in_area2_b
                px_per_m = px_per_m_b
            extract_oneedge(output_file, output_file, is_in_area1, is_in_area2)
            downsample_to_5fps(output_file, output_file)
            px_to_m_and_add_velocities_and_remove_unnecessary_columns(output_file, output_file, px_per_m)
            interpolate_missing_data(output_file, output_file)
            rename_columns(output_file, output_file)