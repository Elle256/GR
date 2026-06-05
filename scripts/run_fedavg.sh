#!/usr/bin/env bash
# scripts/run_fedavg.sh – Quick FedAvg run with configurable parameters
# Usage: bash scripts/run_fedavg.sh [alpha] [dropout] [seed]

ALPHA=${1:-0.5}
DROPOUT=${2:-0.0}
SEED=${3:-42}

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
  --verbose \
  --tensorboard