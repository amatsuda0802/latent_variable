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

def extract_features(segment, sr):
    if len(segment) == 0:
        return None

    energy = np.mean(segment ** 2)

    zcr = librosa.feature.zero_crossing_rate(segment).mean()

    f0 = librosa.yin(segment, fmin=50, fmax=300, sr=sr)
    f0 = f0[f0 > 0]
    f0_mean = f0.mean() if len(f0) > 0 else np.nan

    return energy, zcr, f0_mean

# ----------------------------------

import pandas as pd

features = []

for utt in utterances:
    segment = extract_segment(y, sr, utt["start"], utt["end"])
    feats = extract_features(segment, sr)

    if feats is None:
        continue

    energy, zcr, f0_mean = feats

    features.append({
        "utterance_id": utt["utterance_id"],
        "speaker": utt["speaker"],
        "duration": utt["end"] - utt["start"],
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
X_std = scaler.fit_transform(X)

# --------------------------------------------

from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=2, random_state=0)
Z = fa.fit_transform(X_std)

loadings = pd.DataFrame(
    fa.components_.T,
    index=X.columns,
    columns=["Factor1", "Factor2"]
)

print("=== 因子負荷量 ===")
print(loadings)

# ------------------------------------------

df_clean["Factor1"] = Z[:, 0]
df_clean["Factor2"] = Z[:, 1]

print(df_clean.head())
