"""
トラッキング結果から、短い軌道を削除して後続のクラスタリング等を楽にする
FRAMES_THRESHOLD: フレーム数の閾値 これより短い軌道は削除する
"""

FRAMES_THRESHOLD = 300

import pandas as pd

def remove_shorts(input_file, output_file, FRAMES_THRESHOLD):
    df = pd.read_csv(input_file)

    df_per_person = df.groupby("id")
    valid_person_ids = []
    for person_id, df_person in df_per_person:
        if len(df_person) > FRAMES_THRESHOLD:
            valid_person_ids.append(person_id)
    new_df = df[df["id"].isin(valid_person_ids)]

    new_df.to_csv(output_file, index=False)
    print(f"Number of valid person IDs has been reduced from {len(df_per_person)} to {len(valid_person_ids)}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    input_file = "output/sample_movie_1020/sample_movie.csv"
    output_file = "output/sample_movie_1020/sample_movie_postprocessed.csv"
    remove_shorts(input_file, output_file, FRAMES_THRESHOLD)