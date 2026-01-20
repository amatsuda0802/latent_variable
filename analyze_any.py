import pympi
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import sys
import os
# 因子の時間推移，ファクターの個数，話者ごとのファクターの傾向とか諸々と関連付けて色々考えることができるかも
# 特徴量増やしたり，別のデータで試したり，特に特徴量に関しては現状音声周りのことしか使ってないので見えの特徴とかも使えるかな
# 状態遷移モデル(HMM)とかにもつないでいける？
# とりあえずざっといじって色々見れそうな要素は，「因子数」「使う観測変数，特徴量」「使う対話データ」，「使うモデル」なんかもそうかな，あとwindowサイズとかもそうか

# python3 analyze_any.py [data_eaf] [data_wav] [dir_result] n_factor
n_factor = int(sys.argv[4])
path_data  = "./data/"
path_result = "./result/analyze_any/"
data_eaf = path_data + sys.argv[1]
data_wav = path_data + sys.argv[2]
dir_result = path_result + sys.argv[3] + f"/factor_{n_factor}"
os.makedirs(dir_result, exist_ok=True)

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

print(utterances)

# ---------------------------------
y, sr = librosa.load(data_wav, sr=16000)

# -------------------------------------
def extract_segment(y, sr, start, end):
    return y[int(start*sr):int(end*sr)]

# -------------------------------------
def extract_features(segment, sr): # segment：1発話に対応する音声波形の一部
    if len(segment) == 0:
        return None

    energy = np.mean(segment ** 2) # エネルギー，振幅の2乗の時間平均：直感的には，声が大きい，強いと値が大きく，小声や沈黙に近いと値が小さい

    zcr = librosa.feature.zero_crossing_rate(segment).mean() # ゼロ交差率，波形が0を何回またぐか：値が大きいと子音が多くノイズっぽく早口，値が小さいと母音が多くなめらかでゆったり

    f0 = librosa.yin(segment, fmin=50, fmax=300, sr=sr) # ピッチ，声帯の振動周期，声の高さ：
    f0 = f0[f0 > 0]
    f0_mean = f0.mean() if len(f0) > 0 else np.nan # ピッチの平均：発話の声の高さの傾向

    return energy, zcr, f0_mean # これらに加えて発話長が今回使う特徴量

# ----------------------------------
features = []

for utt in utterances: # utterances：発話id，話者，発話開始時刻，発話終了時刻
    segment = extract_segment(y, sr, utt["start"], utt["end"])
    feats = extract_features(segment, sr)

    if feats is None:
        continue

    energy, zcr, f0_mean = feats

    features.append({
        "utterance_id": utt["utterance_id"],
        "speaker": utt["speaker"],
        "start": utt["start"],
        "end": utt["end"],
        "duration": utt["end"] - utt["start"], # 発話長
        "energy": energy,
        "zcr": zcr,
        "f0_mean": f0_mean
    })

df = pd.DataFrame(features)

# -----------------------------------
df_clean = df.dropna()

X = df_clean[["duration", "energy", "zcr", "f0_mean"]]

scaler = StandardScaler()
X_std = scaler.fit_transform(X) # 特徴量を標準化（どの特徴量も0から1の値になる，もとの特徴量の絶対値の大きさではなく基準が揃ってどの特徴も同じ影響力になる）

# --------------------------------------------
fa = FactorAnalysis(n_components=n_factor, random_state=0)
Z = fa.fit_transform(X_std)

factor_names = [f"Factor{i+1}" for i in range(n_factor)]

loadings = pd.DataFrame(
    fa.components_.T,
    index=X.columns,
    columns=factor_names
)

pd.set_option("display.max_columns", None) # pandasでの表示列数を最大に
pd.options.display.width = 160 # 表示文字数設定

print("=== 因子負荷量 ===") # 各特徴量が各ファクターにどのようにどれくらい影響しているか
print(loadings)

loadings.to_csv(f"{dir_result}/因子負荷量_{n_factor}.csv")

# ------------------------------------------
for i, fname in enumerate(factor_names):
    df_clean[fname] = Z[:, i]

print(df_clean.head()) # 各発話について各ファクターの値が出る，ファクターのあたいの組み合わせ（両方高い，片方高くて片方小さい，など）と実際の発話内容や発話状況，話しやすさを照らし合わせて，このファクターがこういう組み合わせのとき話しやすさが高い状態だと言えるかもねみたいな感じの流れになるかな？

df_clean.to_csv(f"{dir_result}/df_clean_{n_factor}.csv")

# ------------------------------------------------
df_clean["time"] = (df_clean["start"] + df_clean["end"]) / 2

# 移動平均
window = 20  # 発話20個分で平均（調整可）
for fname in factor_names:
    df_clean[f"{fname}_ma"] = (
        df_clean[fname]
        .rolling(window=window, center=True)
        .mean()
    )

df_plot = df_clean.sort_values("time")

plt.figure(figsize=(10, 8))

for fname in factor_names:
    plt.plot(
        df_plot["time"],
        df_plot[fname],
        label=fname,
        linewidth=1
    )

plt.xlabel("Time (sec)")
plt.ylabel("Factor value")
plt.legend()
plt.title("Temporal transition of latent factors")

plt.savefig(f"{dir_result}/factor_transition_raw_{n_factor}.png")
plt.close()

plt.figure(figsize=(10, 8))

for fname in factor_names:
    plt.plot(
        df_plot["time"],
        df_plot[f"{fname}_ma"],
        linewidth=1,
        label=f"{fname} (Moving Avg, window={window})"
    )

plt.xlabel("Time [sec]")
plt.ylabel("Factor value")
plt.title("Factor (Moving Average)")
plt.legend()

plt.tight_layout()
plt.savefig(f"{dir_result}/factor_moving_average_{n_factor}.png", dpi=300)
plt.close()