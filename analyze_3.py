import pympi

eaf = pympi.Elan.Eaf("./data/akao_yamashita/20250619_g1_s1_main_final_comp.eaf")
tiers = eaf.get_tier_names() # A：山下さん B：赤尾さん

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

import librosa

y, sr = librosa.load("./data/akao_yamashita/20250619_g1_s1_main_final.wav", sr=16000)

# -------------------------------------

def extract_segment(y, sr, start, end):
    return y[int(start*sr):int(end*sr)]

# -------------------------------------

import numpy as np
# import librosa

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

import pandas as pd

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

from sklearn.preprocessing import StandardScaler

df_clean = df.dropna()

X = df_clean[["duration", "energy", "zcr", "f0_mean"]]

scaler = StandardScaler()
X_std = scaler.fit_transform(X) # 特徴量を標準化（どの特徴量も0から1の値になる，もとの特徴量の絶対値の大きさではなく基準が揃ってどの特徴も同じ影響力になる）

# --------------------------------------------

from sklearn.decomposition import FactorAnalysis

# fa = FactorAnalysis(n_components=2, random_state=0) # 2因子
fa = FactorAnalysis(n_components=3, random_state=0) # 3因子
Z = fa.fit_transform(X_std)

# loadings = pd.DataFrame(
#     fa.components_.T,
#     index=X.columns,
#     columns=["Factor1", "Factor2"]
# ) # 2因子
loadings = pd.DataFrame(
    fa.components_.T,
    index=X.columns,
    columns=["Factor1", "Factor2", "Factor3"]
) # 3因子

pd.set_option("display.max_columns", None) # pandasでの表示列数を最大に

print("=== 因子負荷量 ===") # 各特徴量が各ファクターにどのようにどれくらい影響しているか
print(loadings)

# ------------------------------------------

df_clean["Factor1"] = Z[:, 0]
df_clean["Factor2"] = Z[:, 1]
df_clean["Factor3"] = Z[:, 2] # 3因子の場合

print(df_clean.head()) # 各発話について各ファクターの値が出る，ファクターのあたいの組み合わせ（両方高い，片方高くて片方小さい，など）と実際の発話内容や発話状況，話しやすさを照らし合わせて，このファクターがこういう組み合わせのとき話しやすさが高い状態だと言えるかもねみたいな感じの流れになるかな？

# ------------------------------------------------
# 因子の時間推移，ファクターの個数，話者ごとのファクターの傾向とか諸々と関連付けて色々考えることができるかも
# 特徴量増やしたり，別のデータで試したり，特に特徴量に関しては現状音声周りのことしか使ってないので見えの特徴とかも使えるかな
# 状態遷移モデル(HMM)とかにもつないでいける？
# とりあえずざっといじって色々見れそうな要素は，「因子数」「使う観測変数，特徴量」「使う対話データ」，「使うモデル」なんかもそうかな

df_clean["time"] = (df_clean["start"] + df_clean["end"]) / 2

# 移動平均
window = 20  # 発話20個分で平均（調整可）
df_clean["Factor1_ma"] = (
    df_clean["Factor1"]
    .rolling(window=window, center=True)
    .mean()
)
df_clean["Factor2_ma"] = (
    df_clean["Factor2"]
    .rolling(window=window, center=True)
    .mean()
)
df_clean["Factor3_ma"] = (
    df_clean["Factor3"]
    .rolling(window=window, center=True)
    .mean()
) # 3因子の場合


import matplotlib.pyplot as plt

df_plot = df_clean.sort_values("time")

plt.figure(figsize=(10, 8))
# plt.plot(df_plot["time"], df_plot["Factor1"], marker="o", label="Factor1")
# plt.plot(df_plot["time"], df_plot["Factor2"], marker="o", label="Factor2")
plt.plot(df_plot["time"], df_plot["Factor1"], label="Factor1", linewidth=1)
plt.plot(df_plot["time"], df_plot["Factor2"], label="Factor2", linewidth=1)
plt.plot(df_plot["time"], df_plot["Factor3"], label="Factor3", linewidth=1) # 3因子の場合


plt.xlabel("Time (sec)")
plt.ylabel("Factor value")
plt.legend()
plt.title("Temporal transition of latent factors")

# plt.savefig("./result/analyze/factor_transition_raw_2.png")
plt.savefig("./result/analyze_3/factor_transition_raw_3.png") # 3因子の場合

plt.close()


plt.figure(figsize=(10, 8))

plt.plot(
    df_plot["time"],
    df_plot["Factor1_ma"],
    linewidth=1,
    label=f"Factor1 (Moving Avg, window={window})"
)
plt.plot(
    df_plot["time"],
    df_plot["Factor2_ma"],
    linewidth=1,
    label=f"Factor2 (Moving Avg, window={window})"
)
plt.plot(
    df_plot["time"],
    df_plot["Factor3_ma"],
    linewidth=1,
    label=f"Factor3 (Moving Avg, window={window})"
) # 3因子の場合

plt.xlabel("Time [sec]")
plt.ylabel("Factor value")
plt.title("Factor (Moving Average)")
plt.legend()

plt.tight_layout()
# plt.savefig("./result/analyze/factor_moving_average_2.png", dpi=300)
plt.savefig("./result/analyze_3/factor_moving_average_3.png", dpi=300) # 3因子の場合
plt.close()
