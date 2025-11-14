"""
トラッキング結果から、指定のエリアに始点終点を持つ軌道のみを抽出する
"""

import pandas as pd

def extract_oneedge(input_file, output_file, is_in_area1, is_in_area2):
    """
    input_file: 入力ファイルパス
    output_file: 出力ファイルパス
    is_in_area1: 点がエリア1に含まれるかどうかを判定する関数
    is_in_area2: 点がエリア2に含まれるかどうかを判定する関数
    """
    df = pd.read_csv(input_file, header=None, names=["frame", "id", "x", "y", "w", "h", "score", "_1", "_2", "_3"])

    df_per_person = df.groupby("id")
    valid_person_ids = []
    for person_id, df_person in df_per_person:
        start_point = df_person[df_person["frame"] == df_person["frame"].min()][["x", "y"]].values
        end_point = df_person[df_person["frame"] == df_person["frame"].max()][["x", "y"]].values
        if is_in_area1(start_point[0], start_point[1]) and is_in_area2(end_point[0], end_point[1]):
            valid_person_ids.append(person_id)
        if is_in_area1(end_point[0], end_point[1]) and is_in_area2(start_point[0], start_point[1]):
            valid_person_ids.append(person_id)
            
    new_df = df[df["id"].isin(valid_person_ids)]

    new_df.to_csv(output_file, header=None, index=False)
    print(f"Number of valid person IDs has been reduced from {len(df_per_person)} to {len(valid_person_ids)}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    input_file = "output/sample_movie_1020/sample_movie_postprocessed.txt"
    output_file = "output/sample_movie_1020/sample_movie_postprocessed_oneedge.txt"
    def is_in_area1(x, y):
        return x < 1120 and y < 720
    def is_in_area2(x, y):
        return x < 1900 and y > 1660
    extract_oneedge(input_file, output_file, is_in_area1, is_in_area2)