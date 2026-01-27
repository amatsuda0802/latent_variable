#!/bin/bash

# マルチモーダル因子分析を実行するスクリプト
# 使用方法:
#   ./scripts/run_multimodal_analysis.sh [data_dir] [n_factor] [features...]
#
# 例:
#   # 全特徴量を使用
#   ./scripts/run_multimodal_analysis.sh 20250619_g1_s1_main_final 3 audio gaze smile
#
#   # 個別特徴量を指定
#   ./scripts/run_multimodal_analysis.sh 20250619_g1_s1_main_final 3 "audio:duration,energy" "gaze:is_front,combined_moving_std" smile

# 引数の確認
if [ $# -lt 2 ]; then
    echo "エラー: 引数が不足しています"
    echo "使用方法: $0 [data_dir] [n_factor] [features...]"
    echo ""
    echo "引数:"
    echo "  data_dir: データディレクトリ名（data/下のディレクトリ名、例: 20250619_g1_s1_main_final）"
    echo "  n_factor: 因子数"
    echo "  features: 使用する特徴量グループと個別特徴量（オプション、デフォルト: audio）"
    echo ""
    echo "特徴量の指定方法:"
    echo "  - グループのみ: \"audio\" → そのグループの全特徴量を使用"
    echo "  - 個別特徴量: \"audio:duration,energy\" → 指定した特徴量のみを使用"
    echo ""
    echo "利用可能な特徴量:"
    echo "  audio: duration, energy, zcr, f0_mean"
    echo "  gaze: pitch, yaw, roll, gaze_velocity, pitch_moving_std, yaw_moving_std, combined_moving_std, is_front, direction_dist_front"
    echo "  smile: rank"
    echo ""
    echo "例:"
    echo "  # 音声のみで因子分析（全特徴量）"
    echo "  $0 20250619_g1_s1_main_final 3 audio"
    echo ""
    echo "  # 音声の特定特徴量のみ"
    echo "  $0 20250619_g1_s1_main_final 3 \"audio:duration,energy\""
    echo ""
    echo "  # 音声 + 視線 + 笑顔（全特徴量）"
    echo "  $0 20250619_g1_s1_main_final 3 audio gaze smile"
    echo ""
    echo "  # 個別特徴量を指定"
    echo "  $0 20250619_g1_s1_main_final 3 \"audio:duration,energy\" \"gaze:is_front,combined_moving_std\" smile"
    echo ""
    echo "  # 特徴量を指定しない場合（デフォルトでaudioのみ）"
    echo "  $0 20250619_g1_s1_main_final 3"
    exit 1
fi

DATA_DIR=$1
N_FACTOR=$2
shift 2
FEATURES="$@"

# 特徴量が指定されていない場合はaudioのみ
if [ -z "$FEATURES" ]; then
    FEATURES="audio"
fi

# プロジェクトのルートディレクトリに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "マルチモーダル因子分析"
echo "=========================================="
echo "データディレクトリ: $DATA_DIR"
echo "因子数: $N_FACTOR"
echo "特徴量: $FEATURES"
echo "=========================================="
echo ""

# Pythonスクリプトを実行
# 特徴量引数を配列として処理（個別特徴量指定に対応）
FEATURE_ARGS=()
if [ -n "$FEATURES" ]; then
    # スペース区切りの引数を個別に処理
    for feat in $FEATURES; do
        FEATURE_ARGS+=("$feat")
    done
else
    FEATURE_ARGS=("audio")
fi

python3 src/analyze_multimodal.py \
    --data_dir "$DATA_DIR" \
    --n_factor "$N_FACTOR" \
    --features "${FEATURE_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "分析が正常に完了しました"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "エラー: 分析中にエラーが発生しました"
    echo "=========================================="
    exit 1
fi
