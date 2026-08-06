#!/usr/bin/env bash
# Build the GME-Qwen2-VL OpenVINO C++ runtime against the optimized OpenVINO build.
# NOTE: setupvars.sh references unbound vars, so we do NOT use `set -u`.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENVINO_INSTALL="$HERE/../../benchmark/openvino/install"
OV_INSTALL="${OV_INSTALL:-$DEFAULT_OPENVINO_INSTALL}"
BUILD_DIR="$HERE/build"
JOBS="$(nproc)"

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Options:
  --openvino-install PATH  OpenVINO installation containing setupvars.sh
  --build-dir PATH         CMake build directory (default: cpp/build)
  --jobs N                 parallel build jobs (default: nproc)
  -h, --help               show this help
EOF
}

while (($#)); do
  case "$1" in
    --openvino-install)
      [[ $# -ge 2 ]] || { echo "missing value for --openvino-install" >&2; exit 2; }
      OV_INSTALL="$2"
      shift 2
      ;;
    --build-dir)
      [[ $# -ge 2 ]] || { echo "missing value for --build-dir" >&2; exit 2; }
      BUILD_DIR="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "missing value for --jobs" >&2; exit 2; }
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$OV_INSTALL" ]]; then
  echo "OpenVINO install directory not found at $OV_INSTALL (pass --openvino-install)" >&2
  exit 1
fi
OV_INSTALL="$(cd "$OV_INSTALL" && pwd)"
BUILD_DIR="$(mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" && pwd)"

if [[ ! -f "$OV_INSTALL/setupvars.sh" ]]; then
  echo "OpenVINO install not found at $OV_INSTALL (pass --openvino-install)"; exit 1
fi
# shellcheck disable=SC1091
source "$OV_INSTALL/setupvars.sh"

OpenCV_DIR="$OV_INSTALL/lib/cmake/opencv4"

cmake -S "$HERE" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$OpenCV_DIR"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo
echo "built: $BUILD_DIR/gme_embed"
echo "run with: source $OV_INSTALL/setupvars.sh && $BUILD_DIR/gme_embed --help"
