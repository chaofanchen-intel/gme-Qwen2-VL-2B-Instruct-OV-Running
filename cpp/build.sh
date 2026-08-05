#!/usr/bin/env bash
# Build the GME-Qwen2-VL OpenVINO C++ runtime against the optimized OpenVINO build.
# NOTE: setupvars.sh references unbound vars, so we do NOT use `set -u`.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BENCHMARK_ROOT="${GME_BENCHMARK_ROOT:-$HERE/../../benchmark}"
OV_INSTALL="${OV_INSTALL:-${GME_OPENVINO_INSTALL:-$DEFAULT_BENCHMARK_ROOT/openvino/install}}"

if [[ ! -f "$OV_INSTALL/setupvars.sh" ]]; then
  echo "OpenVINO install not found at $OV_INSTALL (set OV_INSTALL)"; exit 1
fi
# shellcheck disable=SC1091
source "$OV_INSTALL/setupvars.sh"

OpenCV_DIR="$OV_INSTALL/lib/cmake/opencv4"
BUILD_DIR="$HERE/build"

cmake -S "$HERE" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$OpenCV_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "built: $BUILD_DIR/gme_embed"
echo "run with: source $OV_INSTALL/setupvars.sh && $BUILD_DIR/gme_embed --help"
