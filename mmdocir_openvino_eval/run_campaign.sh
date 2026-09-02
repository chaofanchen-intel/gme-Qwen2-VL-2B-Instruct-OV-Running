#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chaofan/model_benchmark}
OV=${OV:-$ROOT/benchmark/openvino}
GME=${GME:-$ROOT/gme-Qwen2-VL-2B-Instruct-Running}
PYTHON=${PYTHON:-/home/chaofan/envs/qwen_env/bin/python}
BATCH_SIZE=${BATCH_SIZE:-1}
BATCH_PROFILE=${BATCH_PROFILE:-recommended}
PRECISIONS=${PRECISIONS:-fp32,int8_causal_vopt,int4_causal_vopt}
VISUAL_DQ_THRESHOLD=${VISUAL_DQ_THRESHOLD:-}
SUBSET_FILE=${SUBSET_FILE:-}
MAX_SAMPLES=${MAX_SAMPLES:-}

if [[ -n "$SUBSET_FILE" && -n "$MAX_SAMPLES" ]]; then
  echo "SUBSET_FILE and MAX_SAMPLES are mutually exclusive" >&2
  exit 2
fi

selection_name=full
selection_args=()
if [[ -n "$SUBSET_FILE" ]]; then
  selection_name=$(basename "$SUBSET_FILE" .json)
  selection_args+=(--subset-file "$SUBSET_FILE")
elif [[ -n "$MAX_SAMPLES" ]]; then
  selection_name="first_${MAX_SAMPLES}"
  selection_args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$BATCH_PROFILE" == "uniform" ]]; then
  profile_name="b${BATCH_SIZE}"
else
  profile_name="$BATCH_PROFILE"
fi
threshold_args=()
threshold_suffix=""
if [[ -n "$VISUAL_DQ_THRESHOLD" ]]; then
  threshold_args+=(--visual-dq-threshold "$VISUAL_DQ_THRESHOLD")
  threshold_suffix="_vdq${VISUAL_DQ_THRESHOLD}"
fi
OUTPUT_DIR=${OUTPUT_DIR:-$GME/mmdocir_openvino_eval/results/${selection_name}_${profile_name}${threshold_suffix}}

cd "$GME/mmdocir_openvino_eval"
exec env -i \
  HOME="$HOME" \
  PATH=/usr/bin:/bin \
  PYTHONPATH="$OV/bin/intel64/Release/python:$GME/mmdocir_openvino_eval:$GME" \
  LD_LIBRARY_PATH="$OV/bin/intel64/Release:$OV/temp/Linux_x86_64/tbb/lib" \
  OPENVINO_LIB_PATHS="$OV/bin/intel64/Release" \
  OPENVINO_PLUGIN_PATH="$OV/bin/intel64/Release" \
  "$PYTHON" -u run_benchmark.py \
    --precisions "$PRECISIONS" \
    --batch-profile "$BATCH_PROFILE" \
    --batch-size "$BATCH_SIZE" \
    "${threshold_args[@]}" \
    --scopes close,open \
    --output-dir "$OUTPUT_DIR" \
    "${selection_args[@]}" \
    "$@"