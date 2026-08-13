#!/usr/bin/env bash
# Build the standalone oneDNN matmul probe against the oneDNN-ov2026.3 worktree.
#
# That worktree is checked out at the exact commit OpenVINO 2026.3 vendors for its
# GPU fork (babb7375ff), so anything measured here maps 1:1 onto what the model runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DNNL_ROOT="${DNNL_ROOT:-/home/chaofan/model_benchmark/oneDNN-ov2026.3}"
DNNL_BUILD="${DNNL_BUILD:-$DNNL_ROOT/build}"

g++ -O2 -std=c++17 -o "$HERE/gemm_probe" "$HERE/gemm_probe.cpp" \
    -I"$DNNL_ROOT/include" -I"$DNNL_BUILD/include" \
    -L"$DNNL_BUILD/src" -ldnnl -lOpenCL \
    -Wl,-rpath,"$DNNL_BUILD/src"

echo "built $HERE/gemm_probe"
