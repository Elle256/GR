#!/usr/bin/env bash
# scripts/compare.sh – Run FedAvg vs FedProx on the same condition
# Usage: bash scripts/compare.sh [alpha] [dropout] [seed]

ALPHA=${1:-0.1}
DROPOUT=${2:-0.4}
SEED=${3:-42}

echo "Running FedAvg ..."
python main.py \
  --algorithm fedavg \
  --dataset   mnist \
  --model     cnn \
  --rounds    100 \
  --num_clients 20 \
  --frac      0.5 \
  --local_epochs 5 \
  --local_bs  32 \
  --lr        0.01 \
  --alpha     "$ALPHA" \
  --dropout   "$DROPOUT" \
  --seed      "$SEED" \
  --log_interval 10 \
  --verbose

echo ""
echo "Running FedProx (μ=0.01) ..."
python main.py \
  --algorithm fedprox \
  --dataset   mnist \
  --model     cnn \
  --rounds    100 \
  --num_clients 20 \
  --frac      0.5 \
  --local_epochs 5 \
  --local_bs  32 \
  --lr        0.01 \
  --alpha     "$ALPHA" \
  --dropout   "$DROPOUT" \
  --mu        0.01 \
  --seed      "$SEED" \
  --log_interval 10 \
  --verbose

echo ""
echo "Results saved in experiments/results/"
echo "Compare with: python -c \""
echo "  import json, glob"
echo "  for f in glob.glob('experiments/results/*alpha${ALPHA}*drop${DROPOUT%.*}*.json'):"
echo "      d = json.load(open(f)); print(f, d['summary'])"
echo "\""