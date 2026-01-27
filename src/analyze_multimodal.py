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

# 特徴量の定義（拡張しやすい構造）
# 新しい特徴量を追加する場合はここに追加する
# 各データの読み込み処理（load_audio_featuresなど）で新しい特徴量を抽出する処理を追加すれば選択可能になる
FEATURE_DEFINITIONS = {
    "audio": {
        "duration": "発話長",
        "energy": "エネルギー，振幅の2乗の時間平均：直感的には，声が大きい，強いと値が大きく，小声や沈黙に近いと値が小さい",
        "zcr": "ゼロ交差率，波形が0を何回またぐか：値が大きいと子音が多くノイズっぽく早口，値が小さいと母音が多くなめらかでゆったり",
        "f0_mean": "ピッチの平均：発話の声の高さの傾向"
    },
    "gaze": {
        "pitch": "視線のピッチ（上下角度）",
        "yaw": "視線のヨー（左右角度）",
        "roll": "視線のロール（回転角度）",
        "gaze_velocity": "視線の速度",
        "pitch_moving_std": "ピッチの移動標準偏差",
        "yaw_moving_std": "ヨーの移動標準偏差",
        "combined_moving_std": "合成移動標準偏差",
        "is_front": "正面を向いているか（0/1）",
        "direction_dist_front": "正面方向の分布（パーセンテージ）"
    },
    "smile": {
        "rank": "笑顔スコア（1.0～10.0、低いほど笑顔）"
    }
}

def get_all_features(group):
    """指定されたグループの全特徴量名を取得"""
    return list(FEATURE_DEFINITIONS.get(group, {}).keys())

def get_available_features(df, group):
    """DataFrameに存在する特徴量のみを取得"""
    if group == "audio":
        # audio特徴量はプレフィックスなし
        prefix = ""
    else:
        # gaze, smile特徴量はプレフィックス付き
        prefix = f"{group}_"
    
    all_features = get_all_features(group)
    available = []
    for feat in all_features:
        col_name = prefix + feat if prefix else feat
        if col_name in df.columns:
            available.append(feat)
    return available

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

def load_elan_features(data_eaf, tier_name=None, annotation_filter=None):
    """
    EAFファイルから注釈を読み込む汎用関数
    
    Parameters:
    -----------
    data_eaf : str
        EAFファイルのパス
    tier_name : str, optional
        読み込むtier名。Noneの場合は全てのtierから読み込む
    annotation_filter : callable, optional
        注釈をフィルタリングする関数。引数は(start_ms, end_ms, label)のタプル
        
    Returns:
    --------
    pd.DataFrame
        注釈データを含むDataFrame。カラム: annotation_id, tier, start, end, label, time
        annotation_id: 0から始まる連番（注釈の順序を表す）
    """
    eaf = pympi.Elan.Eaf(data_eaf)
    
    # tier名のリストを取得
    if tier_name is not None:
        if tier_name not in eaf.get_tier_names():
            raise ValueError(f"tier '{tier_name}' が見つかりません。利用可能なtier: {eaf.get_tier_names()}")
        tier_names = [tier_name]
    else:
        tier_names = eaf.get_tier_names()
    
    annotations = []
    annotation_id = 0
    
    for tier in tier_names:
        for start_ms, end_ms, label in eaf.get_annotation_data_for_tier(tier):
            # フィルタリング関数が指定されている場合は適用
            if annotation_filter is not None:
                if not annotation_filter(start_ms, end_ms, label):
                    continue
            
            annotations.append({
                "annotation_id": annotation_id,
                "tier": tier,
                "start": start_ms / 1000,  # ms → sec
                "end": end_ms / 1000,
                "label": label,
                "time": (start_ms + end_ms) / 2000  # 中央時刻（秒）
            })
            annotation_id += 1
    
    df_annotations = pd.DataFrame(annotations)
    return df_annotations

def load_audio_features(data_eaf, data_wav, tier_name="utterance"):
    """
    音声データから特徴量を抽出
    EAFファイルの指定されたtier（デフォルト: "utterance"）から注釈を取得し、
    各注釈区間の音声特徴量を計算する
    
    Parameters:
    -----------
    data_eaf : str
        EAFファイルのパス
    data_wav : str
        WAVファイルのパス
    tier_name : str, default="utterance"
        読み込むtier名
        
    Returns:
    --------
    pd.DataFrame
        音声特徴量を含むDataFrame
    """
    # EAFファイルから注釈を取得
    df_annotations = load_elan_features(data_eaf, tier_name=tier_name)
    
    if len(df_annotations) == 0:
        raise ValueError(f"tier '{tier_name}' に注釈が見つかりませんでした")
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(data_wav, sr=16000)
    
    features = []
    
    for idx, row in df_annotations.iterrows():
        start = row["start"]
        end = row["end"]
        
        # 音声セグメントを抽出
        segment = extract_segment(y, sr, start, end)
        feats = extract_audio_features(segment, sr)
        
        if feats is None:
            continue
        
        energy, zcr, f0_mean = feats
        
        features.append({
            "annotation_id": row["annotation_id"],
            "tier": row["tier"],
            "label": row["label"],
            "start": start,
            "end": end,
            "time": row["time"],
            "duration": end - start,  # 発話長
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

def select_features(df, feature_groups, selected_features=None):
    """
    特徴量を選択
    feature_groups: 使用する特徴量グループのリスト（"audio", "gaze", "smile"）
    selected_features: 各グループで使用する特徴量の辞書
        - 例: {"audio": ["duration", "energy"], "gaze": ["is_front", "combined_moving_std"]}
        - Noneの場合は全特徴量を使用
    """
    if selected_features is None:
        selected_features = {}
    
    selected_cols = []
    
    for group in feature_groups:
        # 選択された特徴量を取得（指定がない場合は全特徴量）
        if group in selected_features and selected_features[group]:
            group_features = selected_features[group]
        else:
            # デフォルトでは全特徴量を使用
            group_features = get_available_features(df, group)
        
        # カラム名を構築
        if group == "audio":
            # audio特徴量はプレフィックスなし
            for feat in group_features:
                col_name = feat
                if col_name in df.columns:
                    selected_cols.append(col_name)
        else:
            # gaze, smile特徴量はプレフィックス付き
            for feat in group_features:
                col_name = f"{group}_{feat}"
                if col_name in df.columns:
                    selected_cols.append(col_name)
    
    # 存在するカラムのみを選択
    available_cols = [col for col in selected_cols if col in df.columns]
    
    return df[available_cols]

def parse_feature_selection(feature_str):
    """
    特徴量選択文字列をパース
    形式: "group:feature1,feature2" または "group"（全特徴量を使用）
    例: "audio:duration,energy" または "audio"
    """
    if ":" in feature_str:
        group, features = feature_str.split(":", 1)
        features_list = [f.strip() for f in features.split(",") if f.strip()]
        return group, features_list
    else:
        return feature_str, None  # Noneは全特徴量を使用することを意味

def main():
    parser = argparse.ArgumentParser(
        description="マルチモーダル因子分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
特徴量の選択例:
  --features audio gaze smile
    → 全グループの全特徴量を使用
  
  --features audio:duration,energy gaze:is_front,combined_moving_std smile
    → audioからdurationとenergy、gazeからis_frontとcombined_moving_std、smileは全特徴量を使用

利用可能な特徴量:
  audio: duration, energy, zcr, f0_mean
  gaze: pitch, yaw, roll, gaze_velocity, pitch_moving_std, yaw_moving_std, combined_moving_std, is_front, direction_dist_front
  smile: rank
        """
    )
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="データディレクトリ名（data/下のディレクトリ名、例: 20250619_g1_s1_main_final）")
    parser.add_argument("--n_factor", type=int, required=True, help="因子数")
    parser.add_argument("--features", type=str, nargs="+", default=["audio"], 
                        help="使用する特徴量グループと個別特徴量（例: audio:duration,energy または audio）")
    parser.add_argument("--time_window", type=float, default=0.1, 
                        help="特徴量マージ時の時間窓（秒）")
    parser.add_argument("--ma_window", type=int, default=20, 
                        help="移動平均のウィンドウサイズ")
    parser.add_argument("--tier_name", type=str, default="utterance",
                        help="EAFファイルから読み込むtier名（デフォルト: utterance）")
    
    args = parser.parse_args()
    
    # 特徴量選択をパース
    feature_groups = []
    selected_features = {}
    
    for feat_spec in args.features:
        group, features = parse_feature_selection(feat_spec)
        if group not in FEATURE_DEFINITIONS:
            raise ValueError(f"不明な特徴量グループ: {group}。利用可能: {list(FEATURE_DEFINITIONS.keys())}")
        feature_groups.append(group)
        if features is not None:
            # 指定された特徴量が存在するか確認
            available = get_all_features(group)
            invalid = [f for f in features if f not in available]
            if invalid:
                raise ValueError(f"グループ '{group}' に存在しない特徴量: {invalid}。利用可能: {available}")
            selected_features[group] = features
    
    # 重複を除去
    feature_groups = list(set(feature_groups))
    
    # パス設定
    # スクリプトから実行される場合はプロジェクトルートがカレントディレクトリ
    # 直接実行される場合はsrcディレクトリがカレントディレクトリの可能性がある
    # 両方に対応するため、まず相対パスを試し、存在しない場合は絶対パスを構築
    if os.path.exists("data/"):
        path_data = "data/"
        path_result = "outputs/analyze_multimodal/"
    elif os.path.exists("../data/"):
        path_data = "../data/"
        path_result = "../outputs/analyze_multimodal/"
    else:
        # スクリプトの場所から相対パスを構築
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        path_data = os.path.join(project_root, "data") + os.sep
        path_result = os.path.join(project_root, "outputs", "analyze_multimodal") + os.sep
    data_dir = os.path.join(path_data, args.data_dir)
    
    # ファイルパスを自動構築
    data_eaf = os.path.join(data_dir, f"{args.data_dir}.eaf")
    data_wav = os.path.join(data_dir, f"{args.data_dir}.wav")
    gaze_csv = os.path.join(data_dir, f"gaze_timeseries_{args.data_dir}.csv")
    smile_csv = os.path.join(data_dir, f"smile_timeseries_{args.data_dir}.csv")
    
    # 結果ディレクトリ名（データディレクトリ名を使用）
    # os.path.joinを使用して安全にパスを構築
    dir_result = os.path.join(path_result, args.data_dir, f"factor_{args.n_factor}")
    
    # 特徴量グループ名をソートしてディレクトリ名に含める
    feature_str = "_".join(sorted(feature_groups))
    dir_result = os.path.join(dir_result, f"features_{feature_str}")
    
    # 個別特徴量が指定されている場合はディレクトリ名に含める
    if selected_features:
        feat_detail = []
        for group in sorted(feature_groups):
            if group in selected_features:
                feat_detail.append(f"{group}:" + ",".join(selected_features[group]))
            else:
                feat_detail.append(f"{group}:all")
        feat_detail_str = "_".join(feat_detail).replace(":", "-").replace(",", "+")
        dir_result = os.path.join(dir_result, feat_detail_str)
    
    os.makedirs(dir_result, exist_ok=True)
    
    # ファイルの存在確認
    if not os.path.exists(data_eaf):
        raise FileNotFoundError(f"EAFファイルが見つかりません: {data_eaf}")
    if not os.path.exists(data_wav):
        raise FileNotFoundError(f"WAVファイルが見つかりません: {data_wav}")
    
    print(f"音声データを読み込み中...")
    print(f"  EAF: {data_eaf}")
    print(f"  WAV: {data_wav}")
    print(f"  Tier: {args.tier_name}")
    df_audio = load_audio_features(data_eaf, data_wav, tier_name=args.tier_name)
    print(f"音声特徴量: {len(df_audio)} 発話")
    
    df_gaze = None
    if "gaze" in feature_groups:
        if not os.path.exists(gaze_csv):
            raise FileNotFoundError(f"gaze CSVファイルが見つかりません: {gaze_csv}")
        print(f"視線データを読み込み中...")
        print(f"  Gaze CSV: {gaze_csv}")
        df_gaze = load_gaze_features(gaze_csv)
        print(f"視線特徴量: {len(df_gaze)} フレーム")
    
    df_smile = None
    if "smile" in feature_groups:
        if not os.path.exists(smile_csv):
            raise FileNotFoundError(f"smile CSVファイルが見つかりません: {smile_csv}")
        print(f"笑顔データを読み込み中...")
        print(f"  Smile CSV: {smile_csv}")
        df_smile = load_smile_features(smile_csv)
        print(f"笑顔特徴量: {len(df_smile)} フレーム")
    
    # 特徴量の選択とデータの読み込みの整合性チェック
    if "gaze" in feature_groups and df_gaze is None:
        raise ValueError("--featuresにgazeが含まれていますが、gaze CSVファイルが見つかりません")
    if "smile" in feature_groups and df_smile is None:
        raise ValueError("--featuresにsmileが含まれていますが、smile CSVファイルが見つかりません")
    
    print(f"特徴量をマージ中...")
    df_merged = merge_features_by_time(df_audio, df_gaze, df_smile, args.time_window)
    
    print(f"特徴量を選択中...")
    X = select_features(df_merged, feature_groups, selected_features)
    
    # 選択された特徴量を表示
    print(f"選択された特徴量:")
    for group in feature_groups:
        if group in selected_features and selected_features[group]:
            print(f"  {group}: {', '.join(selected_features[group])}")
        else:
            available = get_available_features(df_merged, group)
            print(f"  {group}: {', '.join(available)} (全特徴量)")
    
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
    
    pd.set_option("display.max_columns", None)  # pandasでの表示列数を最大に
    pd.options.display.width = 160  # 表示文字数設定
    
    print("因子負荷量")  # 各特徴量が各ファクターにどのようにどれくらい影響しているか
    print(loadings)
    
    loadings.to_csv(os.path.join(dir_result, f"因子負荷量_{args.n_factor}.csv"))
    
    # 因子スコアを追加
    for i, fname in enumerate(factor_names):
        df_clean[fname] = Z[:, i]
    
    df_clean.to_csv(os.path.join(dir_result, f"df_clean_{args.n_factor}.csv"))
    
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
    plt.savefig(os.path.join(dir_result, f"factor_transition_raw_{args.n_factor}.png"), dpi=300)
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
    plt.savefig(os.path.join(dir_result, f"factor_moving_average_{args.n_factor}.png"), dpi=300)
    plt.close()
    
    print(f"\n結果を保存しました: {dir_result}")

if __name__ == "__main__":
    main()
