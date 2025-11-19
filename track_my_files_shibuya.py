# track_my_file.pyを使って渋谷のトラッキングを一気に行うスクリプト

import numpy as np
import os
import json
from track_my_file import track_my_file

road_polygon_A_original = np.array([
    [1400, 1400],
    [3280, 1200],
    [3520, 800],
    [3160, 0],
    [3720, 0],
    [4280, 1040],
    [4800, 1360],
    [5600, 1200],
    [5520, 1680],
    [4760, 1880],
    [5040, 2800],
    [4400, 2800],
    [4200, 2160],
    [3720, 1920],
    [1600, 2240]
], dtype=np.float32)
road_polygon_A = road_polygon_A_original - np.array([2300, 440])
road_polygon_B_original = np.array([
    [720, 1400],
    [2720, 1040],
    [2880, 680],
    [2640, 0],
    [3200, 0],
    [3760, 920],
    [4280, 1200],
    [5160, 1080],
    [5080, 1480],
    [4260, 1760],
    [4640, 2800],
    [4040, 2800],
    [3680, 2040],
    [3200, 1800],
    [960, 2280]
], dtype=np.float32)
road_polygon_B = road_polygon_B_original - np.array([1740, 320])

if __name__ == "__main__":
    video_dir = "C:/Users/kajio/CroppedShibuyaVideos"
    output_base_dir = "output/shibuya_all"
    fps = 29.97
    
    for video_file in os.listdir(video_dir):
        if not video_file.endswith(".mp4"):
            continue
        video_path = os.path.join(video_dir, video_file)
        video_name = video_file.split(".")[0]
        if "DSCF0027" in video_name or "DSCF0028" in video_name:
            road_polygon = road_polygon_A
        else:
            road_polygon = road_polygon_B
        output_dir = os.path.join(output_base_dir, video_name)
        if os.path.exists(output_dir):
            print(f"Skipping {video_name} because it already exists")
            continue

        print(f"Tracking {video_name}")
        track_my_file(video_path, output_dir, road_polygon=road_polygon)