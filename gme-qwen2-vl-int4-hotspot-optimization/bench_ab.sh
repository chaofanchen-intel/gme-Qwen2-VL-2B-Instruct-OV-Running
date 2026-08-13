#!/usr/bin/env bash
# Interleaved A/B end-to-end benchmark: stock OpenVINO 2026.3.0 wheel vs the patched local build.
#
# The GPU on this box has a bimodal clock state (see PERFORMANCE_REPORT.md S6.2.1), so the two
# builds are alternated round by round instead of measured back to back.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GME_ROOT="${GME_ROOT:-$(cd "$HERE/.." && pwd)}"
MODEL_BENCH_ROOT="${MODEL_BENCH_ROOT:-$(cd "$GME_ROOT/.." && pwd)}"
OPENVINO_ROOT="${OPENVINO_ROOT:-$MODEL_BENCH_ROOT/benchmark/openvino}"
QWEN_ENV="${QWEN_ENV:-/home/chaofan/envs/qwen_env}"
ROUNDS="${ROUNDS:-3}"
# 128 is here because opt-3's row threshold sits between 64 and 128 rows (docs/06 S6.9.4);
# without a point just above it the speedup curve looks like it jumps from 1.02x to 1.18x
# for no reason.
LENGTHS="${LENGTHS:-1024 512 128 32}"
WARMUP="${WARMUP:-14}"
REPEAT="${REPEAT:-20}"
OUT="$HERE/results/ab"
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$QWEN_ENV/bin/activate"

run_one() {   # $1=build  $2=length  $3=round
  local build="$1" length="$2" round="$3"
  local out="$OUT/${build}_len${length}_r${round}.json"
  if [[ "$build" == "patched" ]]; then
    # setupvars.sh is not written for `set -u`, so relax strict mode around it.
    # shellcheck disable=SC1091
    ( set +u
      source "$OPENVINO_ROOT/install/setupvars.sh" >/dev/null 2>&1
      set -u
      python "$HERE/profile_gme_int4_gpu.py" --length "$length" --warmup "$WARMUP" \
             --repeat "$REPEAT" --output "$out" >/dev/null )
  else
    python "$HERE/profile_gme_int4_gpu.py" --length "$length" --warmup "$WARMUP" \
           --repeat "$REPEAT" --output "$out" >/dev/null
  fi
  echo "  $build len=$length round=$round -> $(python -c "
import json;d=json.load(open('$out'))['workloads']
print(' '.join(f'{k}={v[\"p50_ms\"]}' for k,v in d.items()))")"
}

for length in $LENGTHS; do
  echo "=== length $length ==="
  for round in $(seq 1 "$ROUNDS"); do
    run_one stock   "$length" "$round"
    run_one patched "$length" "$round"
  done
done

python - "$OUT" <<'PY'
import json, sys, statistics, collections, pathlib
out = pathlib.Path(sys.argv[1])
agg = collections.defaultdict(list)
for f in sorted(out.glob("*.json")):
    build, length, _ = f.stem.split("_")
    d = json.load(f.open())
    for name, w in d["workloads"].items():
        agg[(int(length.removeprefix("len")), name, build)].append(w["p50_ms"])

print(f"\n{'tokens':>7} {'workload':>8} {'stock p50':>11} {'patched p50':>12} {'delta':>9} {'speedup':>8}")
for tokens in sorted({k[0] for k in agg}):
    for name in ("text", "query"):
        s = agg.get((tokens, name, "stock"))
        p = agg.get((tokens, name, "patched"))
        if not s or not p:
            continue
        ms, mp = statistics.median(s), statistics.median(p)
        print(f"{tokens:7d} {name:>8} {ms:11.2f} {mp:12.2f} {mp - ms:9.2f} {ms / mp:7.3f}x")
PY
