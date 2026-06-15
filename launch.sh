#!/usr/bin/env bash
# launch.sh – Run all 96 experiments (4α × 4dropout × 2algorithms × 3seeds)
# Usage: bash launch.sh [--dataset cifar] [--model cnn] [--rounds 100]
#
# Output: experiments/results/<exp_name>.json  for each run

set -euo pipefail

DATASET=${1:-mnist}
MODEL=${2:-cnn}
ROUNDS=${3:-50}
LOCAL_EPOCHS=5
NUM_CLIENTS=20
FRAC=0.8
LOCAL_BS=64
LR=0.01

ALPHAS=(10)
DROPOUTS=(0.3)
ALGORITHMS=(fedavg)
SEEDS=(42 43)

MU=0.1   # FedProx proximal term

TOTAL=$(( ${#ALPHAS[@]} * ${#DROPOUTS[@]} * ${#ALGORITHMS[@]} * ${#SEEDS[@]} ))
COUNTER=0

echo "=========================================="
echo "  Federated Learning – Full Experiment Run"
echo "  Dataset   : $DATASET"
echo "  Model     : $MODEL"
echo "  Rounds    : $ROUNDS"
echo "  Total runs: $TOTAL"
echo "=========================================="

for ALGO in "${ALGORITHMS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for DROP in "${DROPOUTS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        COUNTER=$(( COUNTER + 1 ))
        echo ""
        echo "──────────────────────────────────────────"
        echo "  [$COUNTER/$TOTAL] algo=$ALGO  α=$ALPHA  drop=$DROP  seed=$SEED"
        echo "──────────────────────────────────────────"

        python main.py \
          --algorithm "$ALGO" \
          --dataset   "$DATASET" \
          --model     "$MODEL" \
          --rounds    "$ROUNDS" \
          --num_clients "$NUM_CLIENTS" \
          --frac      "$FRAC" \
          --local_epochs "$LOCAL_EPOCHS" \
          --local_bs  "$LOCAL_BS" \
          --lr        "$LR" \
          --alpha     "$ALPHA" \
          --dropout   "$DROP" \
          --mu        "$MU" \
          --seed      "$SEED" \
          --log_interval 5 \
          --verbose

      done
    done
  done
done

echo ""
echo "=========================================="
echo "  All $TOTAL experiments completed."
echo "  Results in: experiments/results/"
echo "=========================================="

# ── Tự động vẽ biểu đồ ────────────────────────────────────────────────────
echo ""
echo "  Đang vẽ biểu đồ..."
echo "=========================================="

if python -c "import matplotlib" 2>/dev/null; then
  python plot.py \
    --results_dir /kaggle/working/experiments/result \
    --save_dir    /kaggle/working/images \
    --charts convergence heatmap bar drift summary

  echo ""
  echo "  Biểu đồ đã lưu vào: images/"
  echo "  Các file:"
  ls images/*.png 2>/dev/null | sed 's/^/    /'
else
  echo "  [!] matplotlib chưa cài. Chạy: pip install matplotlib"
  echo "  Sau đó vẽ thủ công: python plot.py"
fi

echo "=========================================="
