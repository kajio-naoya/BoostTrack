import os
import pandas as pd
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

def px_to_m(input_file, output_file, px_per_m):
    df = pd.read_csv(input_file, header=None, names=["frame", "id", "x", "y", "w", "h", "score", "_1", "_2", "_3"])
    df["x"] = df["x"] / px_per_m
    df["y"] = df["y"] / px_per_m
    df.to_csv(output_file, header=None, index=False)

for video_name in os.listdir(base_dir):
    video_dir = os.path.join(base_dir, video_name)
    for txt_file in os.listdir(video_dir):
        if txt_file.endswith(".txt"):
            input_file = os.path.join(video_dir, txt_file)
            output_file = os.path.join(video_dir, txt_file.replace(".txt", "_postprocessed.txt"))
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
            px_to_m(output_file, output_file, px_per_m)