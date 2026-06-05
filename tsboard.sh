#!/usr/bin/env bash
# tsboard.sh – Start TensorBoard to monitor experiments
# Usage: bash tsboard.sh [port]

PORT=${1:-6006}
LOGDIR="experiments/tensorboard"

if [ ! -d "$LOGDIR" ]; then
  echo "No TensorBoard logs found in $LOGDIR"
  echo "Run with --tensorboard flag: python main.py --tensorboard ..."
  exit 1
fi

echo "Starting TensorBoard at http://localhost:$PORT"
echo "Log directory: $LOGDIR"
tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all