#!/usr/bin/env bash
# Decompose the cost of the GME INT4 weight-compressed GEMM in oneDNN.
#
# The GME language tower spends ~72% of its GPU time in a handful of oneDNN matmul
# configurations.  This sweep isolates each ingredient of that configuration by
# removing one at a time, so the gap to the plain int8 DPAS ceiling can be
# attributed rather than guessed.
#
# The shapes are the three MLP projections, which together are 88% of the model's
# GEMM FLOPs at 1024 tokens (see docs/02_hotspot_analysis.md).
#
# The weight tag matters: the model runs `wei:u4::blocked:cab`, i.e. the weights are
# reordered at compile time.  Benchmarking with the default `abc` measures a
# different kernel, so `cab` is used throughout and `abc` is kept only as a contrast.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHDNN="${BENCHDNN:-/home/chaofan/model_benchmark/oneDNN-ov2026.3/build/tests/benchdnn/benchdnn}"
OUT="${1:-$HERE/results/gemm_decompose.log}"
MODE="${MODE:-P}"          # P = performance only; use CP to also check correctness

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

# GME MLP shapes at 1024 tokens.  gate/up are K=1536 N=8960, down is K=8960 N=1536.
declare -A SHAPES=(
  [gate_up]="1x1024x1536:1x1536x8960"
  [down]="1x1024x8960:1x8960x1536"
)
# Per-shape src scale group = the full K (per-token dynamic quantization).
declare -A SRC_K=( [gate_up]=1536 [down]=8960 )

run() {   # $1=label  $2=shape_key  $3...=extra benchdnn args
  local label="$1" key="$2"; shift 2
  local shape="${SHAPES[$key]}"
  echo "### $label  [$key $shape]" >> "$OUT"
  "$BENCHDNN" --matmul --engine=gpu --mode="$MODE" --max-ms-per-prb=2000 --stag=abc --dtag=abc \
      "$@" "$shape" >> "$OUT" 2>&1 || echo "  (FAILED)" >> "$OUT"
  echo >> "$OUT"
}

for key in gate_up down; do
  sk="${SRC_K[$key]}"

  # A. Exactly what GME runs today: int8 per-token activations, u4 weights with
  #    per-128-group scales AND per-128-group asymmetric zero points.
  run "A_gme_actual" "$key" --wtag=cab --dt=s8:u4:f16 \
      "--attr-scales=src:per_ocic:f16:1x${sk}+wei:per_ocic:f16:128x1" \
      "--attr-zero-points=wei:per_ocic:u8:128x1"

  # A2. Same as A but with the SwiGLU post-op chain the gate projection carries.
  run "A2_gme_actual_postops" "$key" --wtag=cab --dt=s8:u4:f16 \
      "--attr-scales=src:per_ocic:f16:1x${sk}+wei:per_ocic:f16:128x1" \
      "--attr-zero-points=wei:per_ocic:u8:128x1" \
      "--attr-post-ops=eltwise_swish:1+binary_mul:f16:per_tensor"

  # A3. Same as A but with the default (unreordered) weight tag, to show how much
  #     of the result depends on the compile-time weight reorder.
  run "A3_gme_actual_wtag_abc" "$key" --wtag=abc --dt=s8:u4:f16 \
      "--attr-scales=src:per_ocic:f16:1x${sk}+wei:per_ocic:f16:128x1" \
      "--attr-zero-points=wei:per_ocic:u8:128x1"

  # B. Symmetric weights (no zero points).  On pre-Xe3p this is the branch that
  #    unlocks native mixed s8/s4 DPAS (pd.cpp: s8s4_dpas_ok).
  run "B_u4_symmetric" "$key" --wtag=cab --dt=s8:u4:f16 \
      "--attr-scales=src:per_ocic:f16:1x${sk}+wei:per_ocic:f16:128x1"

  # C. s8 weights with the same grouped scales: removes u4 unpacking, keeps the
  #    per-128-group rescaling of the K loop.
  run "C_s8wei_grouped" "$key" --wtag=cab --dt=s8:s8:f16 \
      "--attr-scales=src:per_ocic:f16:1x${sk}+wei:per_ocic:f16:128x1"

  # D. u4 weights, per-output-channel scales only: removes the K-loop grouping, so
  #    the whole K can accumulate without intermediate rescaling.
  run "D_u4_per_oc" "$key" --wtag=cab --dt=s8:u4:f16 \
      "--attr-scales=src:per_ocic:f16:1x${sk}+wei:per_oc"

  # D2. Grouped scales with a larger group, to price the grouping itself.
  run "D2_u4_group512" "$key" --wtag=cab --dt=s8:u4:f16 \
      "--attr-scales=src:per_ocic:f16:1x${sk}+wei:per_ocic:f16:512x1" \
      "--attr-zero-points=wei:per_ocic:u8:512x1"

  # E. f16 activations against the real u4 weights: shows what the int8 dynamic
  #    quantization of activations actually buys inside the GEMM.
  run "E_f16src_u4wei" "$key" --wtag=cab --dt=f16:u4:f16 \
      "--attr-scales=wei:per_ocic:f16:128x1" \
      "--attr-zero-points=wei:per_ocic:u8:128x1"

  # F. Plain f16 x f16 -- the dense reference.
  run "F_f16_dense" "$key" --wtag=cab --dt=f16:f16:f16

  # G. Plain s8 x s8, no attributes at all -- the int8 DPAS ceiling.
  run "G_s8_dense" "$key" --wtag=cab --dt=s8:s8:f16
done

echo "wrote $OUT"
