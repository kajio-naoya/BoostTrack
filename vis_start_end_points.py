import pandas as pd
import matplotlib.pyplot as plt

track_result_file = "output/shibuya_all/DSCF0027_205_270/DSCF0027_205_270.txt"

df = pd.read_csv(track_result_file, header=None)

df.columns = ["frame", "id", "x", "y", "w", "h", "score", "_1", "_2", "_3"]

# 各歩行者ごとにデータを分割し，その最初のフレームと最後のフレームの(x, y)を取得し，それらをプロットする
df_per_person = df.groupby("id")

for id, df_person in df_per_person:
    start_frame = df_person["frame"].min()
    end_frame = df_person["frame"].max()
    start_point = df_person[df_person["frame"] == start_frame][["x", "y"]].values
    end_point = df_person[df_person["frame"] == end_frame][["x", "y"]].values
    plt.plot(start_point[:, 0], 4320-start_point[:, 1], "ro", markersize=1)
    plt.plot(end_point[:, 0], 4320-end_point[:, 1], "bo", markersize=1)
plt.show()