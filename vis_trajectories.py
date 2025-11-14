import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np

# 軌道データファイルのパス
track_result_file = "output/shibuya_all/DSCF0027_205_270/DSCF0027_205_270.txt"

# データを読み込み
df = pd.read_csv(track_result_file, header=None)
df.columns = ["frame", "id", "x", "y", "w", "h", "score", "_1", "_2", "_3"]

# 各IDごとにデータをグループ化
df_per_person = df.groupby("id")

# 全IDのリストを取得し、ソート
all_ids = sorted(df["id"].unique())
total_ids = len(all_ids)

# 最大IDを取得してスライダーの最大値を計算
max_id = max(all_ids)
max_slider_value = max_id // 100

print(f"Total IDs found: {total_ids}")
print(f"ID range: {min(all_ids)} to {max_id}")
print(f"Slider range: 0 to {max_slider_value}")

# メインの図を作成
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.25)

# 現在のID範囲を表示するテキストボックス
ax_text = plt.axes([0.1, 0.15, 0.8, 0.05])
text_box = ax_text.text(0.5, 0.5, '', transform=ax_text.transAxes, 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
ax_text.set_xticks([])
ax_text.set_yticks([])
ax_text.set_xlim(0, 1)
ax_text.set_ylim(0, 1)

# スライダーを作成
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
slider = widgets.Slider(
    ax=ax_slider,
    valinit=0,
    valmin=0,
    valmax=max_slider_value,
    valstep=1,
    label='ID Range Group'
)

# 軌道を描画する関数
def plot_trajectories(group_index):
    ax.clear()
    
    # ID範囲を計算: 100*group_index から 100*(group_index+1)-1
    start_id = group_index * 100
    end_id = start_id + 100
    
    print(f"Plotting trajectories for IDs {start_id} to {end_id-1}")
    
    # 該当するIDの軌道を描画
    trajectories_plotted = 0
    colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20色のカラーマップ
    
    for i, person_id in enumerate(all_ids):
        if start_id <= person_id < end_id:
            df_person = df_per_person.get_group(person_id)
            
            # 軌道を描画（y座標を反転）
            color = colors[i % 20]
            ax.plot(df_person["x"], 4320 - df_person["y"], 
                   color=color, linewidth=2, alpha=0.7, 
                   label=f'ID {person_id}')
            
            # 開始点と終了点をマーク
            start_point = df_person.iloc[0]
            end_point = df_person.iloc[-1]
            
            ax.plot(start_point["x"], 4320 - start_point["y"], 
                   'o', color=color, markersize=8, markeredgecolor='black')
            ax.plot(end_point["x"], 4320 - end_point["y"], 
                   's', color=color, markersize=8, markeredgecolor='black')
            
            trajectories_plotted += 1
    
    # グラフの設定
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title(f'Trajectory Visualization (ID Range: {start_id}-{end_id-1}, Number of Trajectories: {trajectories_plotted})')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 凡例を表示（軌道数が多い場合は制限）
    if trajectories_plotted <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # テキストボックスを更新
    text_box.set_text(f'Current ID Range: {start_id} - {end_id-1} (Number of Trajectories: {trajectories_plotted})')
    
    plt.draw()

# スライダーの値が変更されたときのコールバック
def on_slider_change(val):
    plot_trajectories(int(val))

# スライダーにコールバックを設定
slider.on_changed(on_slider_change)

# 初期描画
plot_trajectories(0)

plt.show()
