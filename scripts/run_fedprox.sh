#!/usr/bin/env bash
# scripts/run_fedprox.sh – Quick FedProx run
# Usage: bash scripts/run_fedprox.sh [alpha] [dropout] [mu] [seed]

ALPHA=${1:-0.5}
DROPOUT=${2:-0.0}
MU=${3:-0.01}
SEED=${4:-42}

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
  --mu        "$MU" \
  --seed      "$SEED" \
  --log_interval 10 \
  --verbose \
  --tensorboard