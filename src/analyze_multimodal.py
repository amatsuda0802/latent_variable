import pympi
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# 因子の時間推移，ファクターの個数，話者ごとのファクターの傾向とか諸々と関連付けて色々考えることができるかも
# 特徴量増やしたり，別のデータで試したり，特に特徴量に関しては現状音声周りのことしか使ってないので見えの特徴とかも使えるかな
# 状態遷移モデル(HMM)とかにもつないでいける？
# とりあえずざっといじって色々見れそうな要素は，「因子数」「使う観測変数，特徴量」「使う対話データ」，「使うモデル」なんかもそうかな，あとwindowサイズとかもそうか

def extract_segment(y, sr, start, end):
    """音声波形から指定時間範囲を抽出"""
    return y[int(start*sr):int(end*sr)]

def extract_audio_features(segment, sr):
    """音声セグメントから特徴量を抽出"""
    if len(segment) == 0:
        return None

    energy = np.mean(segment ** 2)  # エネルギー，振幅の2乗の時間平均：直感的には，声が大きい，強いと値が大きく，小声や沈黙に近いと値が小さい

    zcr = librosa.feature.zero_crossing_rate(segment).mean()  # ゼロ交差率，波形が0を何回またぐか：値が大きいと子音が多くノイズっぽく早口，値が小さいと母音が多くなめらかでゆったり

    f0 = librosa.yin(segment, fmin=50, fmax=300, sr=sr)  # ピッチ，声帯の振動周期，声の高さ：
    f0 = f0[f0 > 0]
    f0_mean = f0.mean() if len(f0) > 0 else np.nan  # ピッチの平均：発話の声の高さの傾向

    return energy, zcr, f0_mean

def load_audio_features(data_eaf, data_wav):
    """音声データから特徴量を抽出"""
    eaf = pympi.Elan.Eaf(data_eaf)
    tiers = eaf.get_tier_names()

    utterances = []
    uid = 0

    for tier in tiers:
        for start, end, label in eaf.get_annotation_data_for_tier(tier):
            utterances.append({
                "utterance_id": uid,
                "speaker": tier,
                "start": start / 1000,  # ms → sec
                "end": end / 1000
            })
            uid += 1

    y, sr = librosa.load(data_wav, sr=16000)

    features = []

    for utt in utterances:
        segment = extract_segment(y, sr, utt["start"], utt["end"])
        feats = extract_audio_features(segment, sr)

        if feats is None:
            continue

        energy, zcr, f0_mean = feats

        features.append({
            "utterance_id": utt["utterance_id"],
            "speaker": utt["speaker"],
            "start": utt["start"],
            "end": utt["end"],
            "time": (utt["start"] + utt["end"]) / 2,  # 発話の中央時刻
            "duration": utt["end"] - utt["start"],  # 発話長
            "energy": energy,
            "zcr": zcr,
            "f0_mean": f0_mean
        })

    df_audio = pd.DataFrame(features)
    return df_audio

def load_gaze_features(gaze_csv_path):
    """視線データを読み込み、特徴量を抽出"""
    df_gaze = pd.read_csv(gaze_csv_path)
    
    # 時間でソート
    df_gaze = df_gaze.sort_values("time_sec")
    
    # 特徴量を選択（数値カラムのみ）
    gaze_features = {
        "time": df_gaze["time_sec"].values,
        "pitch": df_gaze["pitch"].values,
        "yaw": df_gaze["yaw"].values,
        "roll": df_gaze["roll"].values,
        "gaze_velocity": df_gaze["gaze_velocity"].values if "gaze_velocity" in df_gaze.columns else None,
        "pitch_moving_std": df_gaze["pitch_moving_std"].values if "pitch_moving_std" in df_gaze.columns else None,
        "yaw_moving_std": df_gaze["yaw_moving_std"].values if "yaw_moving_std" in df_gaze.columns else None,
        "combined_moving_std": df_gaze["combined_moving_std"].values if "combined_moving_std" in df_gaze.columns else None,
        "is_front": (df_gaze["is_front"] == "True").astype(int).values if "is_front" in df_gaze.columns else None,
        "direction_dist_front": df_gaze["direction_dist_front"].values if "direction_dist_front" in df_gaze.columns else None,
    }
    
    # Noneでない特徴量のみをDataFrameに
    df_gaze_features = pd.DataFrame({"time": gaze_features["time"]})
    
    for key, value in gaze_features.items():
        if key != "time" and value is not None:
            df_gaze_features[key] = value
    
    return df_gaze_features

def load_smile_features(smile_csv_path):
    """笑顔データを読み込み、特徴量を抽出"""
    df_smile = pd.read_csv(smile_csv_path)
    
    # 時間でソート
    df_smile = df_smile.sort_values("time_sec")
    
    # 特徴量を選択
    smile_features = {
        "time": df_smile["time_sec"].values,
        "rank": df_smile["rank"].values,  # 笑顔スコア
    }
    
    df_smile_features = pd.DataFrame(smile_features)
    return df_smile_features

def merge_features_by_time(df_audio, df_gaze, df_smile, time_window=0.1):
    """
    時間軸で特徴量をマージ
    time_window: マッチングする時間窓（秒）
    """
    # 音声特徴量を基準に、最も近い時刻のgazeとsmileをマッチング
    merged_features = df_audio.copy()
    
    # gaze特徴量をマージ
    if df_gaze is not None and len(df_gaze) > 0:
        for col in df_gaze.columns:
            if col == "time":
                continue
            merged_features[f"gaze_{col}"] = np.nan
        
        for idx, row in df_audio.iterrows():
            time = row["time"]
            # 最も近い時刻のgazeデータを検索
            time_diff = np.abs(df_gaze["time"] - time)
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx] <= time_window:
                for col in df_gaze.columns:
                    if col != "time":
                        merged_features.at[idx, f"gaze_{col}"] = df_gaze.at[closest_idx, col]
    
    # smile特徴量をマージ
    if df_smile is not None and len(df_smile) > 0:
        for col in df_smile.columns:
            if col == "time":
                continue
            merged_features[f"smile_{col}"] = np.nan
        
        for idx, row in df_audio.iterrows():
            time = row["time"]
            # 最も近い時刻のsmileデータを検索
            time_diff = np.abs(df_smile["time"] - time)
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx] <= time_window:
                for col in df_smile.columns:
                    if col != "time":
                        merged_features.at[idx, f"smile_{col}"] = df_smile.at[closest_idx, col]
    
    return merged_features

def select_features(df, feature_groups):
    """
    特徴量グループを選択
    feature_groups: 使用する特徴量グループのリスト
        - "audio": 音声特徴量（duration, energy, zcr, f0_mean）
        - "gaze": 視線特徴量（pitch, yaw, roll, gaze_velocity等）
        - "smile": 笑顔特徴量（rank）
    """
    selected_cols = []
    
    if "audio" in feature_groups:
        selected_cols.extend(["duration", "energy", "zcr", "f0_mean"])
    
    if "gaze" in feature_groups:
        gaze_cols = [col for col in df.columns if col.startswith("gaze_")]
        selected_cols.extend(gaze_cols)
    
    if "smile" in feature_groups:
        smile_cols = [col for col in df.columns if col.startswith("smile_")]
        selected_cols.extend(smile_cols)
    
    # 存在するカラムのみを選択
    available_cols = [col for col in selected_cols if col in df.columns]
    
    return df[available_cols]

def main():
    parser = argparse.ArgumentParser(description="マルチモーダル因子分析")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="データディレクトリ名（data/下のディレクトリ名、例: 20250619_g1_s1_main_final）")
    parser.add_argument("--n_factor", type=int, required=True, help="因子数")
    parser.add_argument("--features", type=str, nargs="+", default=["audio"], 
                        choices=["audio", "gaze", "smile"],
                        help="使用する特徴量グループ（audio, gaze, smile）")
    parser.add_argument("--time_window", type=float, default=0.1, 
                        help="特徴量マージ時の時間窓（秒）")
    parser.add_argument("--ma_window", type=int, default=20, 
                        help="移動平均のウィンドウサイズ")
    
    args = parser.parse_args()
    
    # パス設定
    path_data = "../data/"
    path_result = "../result/analyze_multimodal/"
    data_dir = path_data + args.data_dir
    
    # ファイルパスを自動構築
    data_eaf = os.path.join(data_dir, f"{args.data_dir}.eaf")
    data_wav = os.path.join(data_dir, f"{args.data_dir}.wav")
    gaze_csv = os.path.join(data_dir, f"gaze_timeseries_{args.data_dir}.csv")
    smile_csv = os.path.join(data_dir, f"smile_timeseries_{args.data_dir}.csv")
    
    # 結果ディレクトリ名（データディレクトリ名を使用）
    dir_result = path_result + args.data_dir + f"/factor_{args.n_factor}"
    
    # 特徴量グループ名をソートしてディレクトリ名に含める
    feature_str = "_".join(sorted(args.features))
    dir_result = dir_result + f"_features_{feature_str}"
    
    os.makedirs(dir_result, exist_ok=True)
    
    # ファイルの存在確認
    if not os.path.exists(data_eaf):
        raise FileNotFoundError(f"EAFファイルが見つかりません: {data_eaf}")
    if not os.path.exists(data_wav):
        raise FileNotFoundError(f"WAVファイルが見つかりません: {data_wav}")
    
    print(f"音声データを読み込み中...")
    print(f"  EAF: {data_eaf}")
    print(f"  WAV: {data_wav}")
    df_audio = load_audio_features(data_eaf, data_wav)
    print(f"音声特徴量: {len(df_audio)} 発話")
    
    df_gaze = None
    if "gaze" in args.features:
        if not os.path.exists(gaze_csv):
            raise FileNotFoundError(f"gaze CSVファイルが見つかりません: {gaze_csv}")
        print(f"視線データを読み込み中...")
        print(f"  Gaze CSV: {gaze_csv}")
        df_gaze = load_gaze_features(gaze_csv)
        print(f"視線特徴量: {len(df_gaze)} フレーム")
    
    df_smile = None
    if "smile" in args.features:
        if not os.path.exists(smile_csv):
            raise FileNotFoundError(f"smile CSVファイルが見つかりません: {smile_csv}")
        print(f"笑顔データを読み込み中...")
        print(f"  Smile CSV: {smile_csv}")
        df_smile = load_smile_features(smile_csv)
        print(f"笑顔特徴量: {len(df_smile)} フレーム")
    
    # 特徴量の選択とデータの読み込みの整合性チェック
    if "gaze" in args.features and df_gaze is None:
        raise ValueError("--featuresにgazeが含まれていますが、gaze CSVファイルが見つかりません")
    if "smile" in args.features and df_smile is None:
        raise ValueError("--featuresにsmileが含まれていますが、smile CSVファイルが見つかりません")
    
    print(f"特徴量をマージ中...")
    df_merged = merge_features_by_time(df_audio, df_gaze, df_smile, args.time_window)
    
    print(f"特徴量を選択中...")
    X = select_features(df_merged, args.features)
    
    if X.empty or len(X.columns) == 0:
        raise ValueError("選択された特徴量が存在しません。特徴量の選択とデータの読み込みを確認してください。")
    
    # NaNを削除
    df_clean = df_merged.dropna(subset=X.columns)
    X_clean = df_clean[X.columns]
    
    print(f"有効なデータ: {len(df_clean)} / {len(df_merged)}")
    print(f"使用特徴量: {list(X_clean.columns)}")
    
    # 標準化
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_clean)
    
    # 因子分析
    print(f"因子分析を実行中（因子数: {args.n_factor}）...")
    fa = FactorAnalysis(n_components=args.n_factor, random_state=0)
    Z = fa.fit_transform(X_std)
    
    factor_names = [f"Factor{i+1}" for i in range(args.n_factor)]
    
    # 因子負荷量
    loadings = pd.DataFrame(
        fa.components_.T,
        index=X_clean.columns,
        columns=factor_names
    )
    
    pd.set_option("display.max_columns", None)
    pd.options.display.width = 160
    
    print("\n因子負荷量")
    print(loadings)
    
    loadings.to_csv(f"{dir_result}/因子負荷量_{args.n_factor}.csv")
    
    # 因子スコアを追加
    for i, fname in enumerate(factor_names):
        df_clean[fname] = Z[:, i]
    
    df_clean.to_csv(f"{dir_result}/df_clean_{args.n_factor}.csv")
    
    # 移動平均
    df_clean_sorted = df_clean.sort_values("time")
    for fname in factor_names:
        df_clean_sorted[f"{fname}_ma"] = (
            df_clean_sorted[fname]
            .rolling(window=args.ma_window, center=True)
            .mean()
        )
    
    # 可視化
    plt.figure(figsize=(12, 8))
    for fname in factor_names:
        plt.plot(
            df_clean_sorted["time"],
            df_clean_sorted[fname],
            label=fname,
            linewidth=1,
            alpha=0.7
        )
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Factor value")
    plt.legend()
    plt.title(f"Temporal transition of latent factors (n={args.n_factor})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{dir_result}/factor_transition_raw_{args.n_factor}.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for fname in factor_names:
        plt.plot(
            df_clean_sorted["time"],
            df_clean_sorted[f"{fname}_ma"],
            linewidth=2,
            label=f"{fname} (Moving Avg, window={args.ma_window})"
        )
    
    plt.xlabel("Time [sec]")
    plt.ylabel("Factor value")
    plt.title(f"Factor (Moving Average, n={args.n_factor})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{dir_result}/factor_moving_average_{args.n_factor}.png", dpi=300)
    plt.close()
    
    print(f"\n結果を保存しました: {dir_result}")

if __name__ == "__main__":
    main()
