#!/usr/bin/env bash
set -euo pipefail

EXTRA_ARGS=("$@")

ROOT=${ROOT:-/home/chaofan/model_benchmark}
OV=${OV:-$ROOT/benchmark/openvino}
GME=${GME:-$ROOT/gme-Qwen2-VL-2B-Instruct-Running}
PYTHON=${PYTHON:-/home/chaofan/envs/qwen_env/bin/python}
BATCH_SIZE=${BATCH_SIZE:-1}
BATCH_PROFILE=${BATCH_PROFILE:-recommended}
VISUAL_DQ_THRESHOLD=${VISUAL_DQ_THRESHOLD:-}
FULL_DATASET=${FULL_DATASET:-0}
threshold_suffix=""
if [[ -n "$VISUAL_DQ_THRESHOLD" ]]; then
  threshold_suffix="_vdq${VISUAL_DQ_THRESHOLD}"
fi
if [[ "$FULL_DATASET" == "1" ]]; then
  SUBSET_FILE=
  default_suite_name="full_${BATCH_PROFILE}${threshold_suffix}"
else
  SUBSET_FILE=${SUBSET_FILE:-$GME/mmdocir_openvino_eval/subsets/balanced_200.json}
  default_suite_name="$(basename "$SUBSET_FILE" .json)_${BATCH_PROFILE}${threshold_suffix}"
fi
SUITE_NAME=${SUITE_NAME:-$default_suite_name}
RESULTS_ROOT=${RESULTS_ROOT:-$GME/mmdocir_openvino_eval/results/$SUITE_NAME}
LOG_DIR=${LOG_DIR:-$RESULTS_ROOT/logs}

mkdir -p "$LOG_DIR"
cd "$GME/mmdocir_openvino_eval"

run_precision() {
  local precision=$1
  local output_dir="$RESULTS_ROOT/$precision"
  echo "[$(date --iso-8601=seconds)] starting $precision"
  ROOT="$ROOT" \
  OV="$OV" \
  GME="$GME" \
  PYTHON="$PYTHON" \
  BATCH_SIZE="$BATCH_SIZE" \
  BATCH_PROFILE="$BATCH_PROFILE" \
  VISUAL_DQ_THRESHOLD="$VISUAL_DQ_THRESHOLD" \
  PRECISIONS="$precision" \
  SUBSET_FILE="$SUBSET_FILE" \
  OUTPUT_DIR="$output_dir" \
  ./run_campaign.sh "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_DIR/$precision.log"
  echo "[$(date --iso-8601=seconds)] completed $precision"
}

run_precision fp32
run_precision int8_causal_vopt
run_precision int4_causal_vopt

echo "[$(date --iso-8601=seconds)] building combined FP32 delta report from caches"
ROOT="$ROOT" \
OV="$OV" \
GME="$GME" \
PYTHON="$PYTHON" \
BATCH_SIZE="$BATCH_SIZE" \
BATCH_PROFILE="$BATCH_PROFILE" \
VISUAL_DQ_THRESHOLD="$VISUAL_DQ_THRESHOLD" \
PRECISIONS=fp32,int8_causal_vopt,int4_causal_vopt \
SUBSET_FILE="$SUBSET_FILE" \
OUTPUT_DIR="$RESULTS_ROOT/combined" \
./run_campaign.sh "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_DIR/combined.log"
echo "[$(date --iso-8601=seconds)] suite completed: $RESULTS_ROOT"