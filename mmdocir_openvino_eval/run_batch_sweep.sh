#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chaofan/model_benchmark}
OV=${OV:-$ROOT/benchmark/openvino}
GME=${GME:-$ROOT/gme-Qwen2-VL-2B-Instruct-Running}
PYTHON=${PYTHON:-/home/chaofan/envs/qwen_env/bin/python}
BATCHES=${BATCHES:-1,2,4,8}
PRECISIONS=${PRECISIONS:-fp32,int8_causal_vopt,int4_causal_vopt}
MAX_DOCUMENTS=${MAX_DOCUMENTS:-1}
RESULTS_ROOT=${RESULTS_ROOT:-$GME/mmdocir_openvino_eval/results/batch_sweep}
LOG_DIR=${LOG_DIR:-$RESULTS_ROOT/logs}

mkdir -p "$LOG_DIR"
cd "$GME/mmdocir_openvino_eval"
IFS=',' read -r -a batch_values <<< "$BATCHES"

for batch_size in "${batch_values[@]}"; do
  output_dir="$RESULTS_ROOT/batch_${batch_size}"
  echo "[$(date --iso-8601=seconds)] starting batch=$batch_size precisions=$PRECISIONS"
  ROOT="$ROOT" \
  OV="$OV" \
  GME="$GME" \
  PYTHON="$PYTHON" \
  BATCH_SIZE="$batch_size" \
  BATCH_PROFILE=uniform \
  PRECISIONS="$PRECISIONS" \
  OUTPUT_DIR="$output_dir" \
  ./run_campaign.sh --max-documents "$MAX_DOCUMENTS" --force \
    2>&1 | tee "$LOG_DIR/batch_${batch_size}.log"
  echo "[$(date --iso-8601=seconds)] completed batch=$batch_size"
done