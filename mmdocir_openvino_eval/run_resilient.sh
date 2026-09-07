#!/usr/bin/env bash
#
# Run a campaign to completion across the OpenVINO GPU plugin's intermittent
# wedge.
#
# INT4 encoding sporadically stops dead inside the GPU plugin -- see README's
# "INT4 挂死：根因与绕过". Nothing recovers a wedged inference except killing the
# process, so the watchdog in run_benchmark.py does exactly that and exits 86.
# This wrapper turns that into a resumable loop: every attempt continues from the
# last checkpoint, so a full-dataset run finishes even if it wedges a dozen times.
#
# Usage mirrors run_campaign.sh; every environment variable it honours is passed
# straight through.
#
#   PRECISIONS=int4_causal_vopt \
#   ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 \
#   OUTPUT_DIR="$PWD/results/full_recommended/int4_causal_vopt" \
#   ./run_resilient.sh
#
# Deliberately NOT `set -e`: the whole point is to inspect exit codes.
set -uo pipefail

# Must match mmdocir_eval.progress.HANG_EXIT_CODE.
HANG_EXIT_CODE=86
MAX_ATTEMPTS=${MAX_ATTEMPTS:-40}

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  echo "[$(date --iso-8601=seconds)] attempt $attempt/$MAX_ATTEMPTS"
  # --resume is unconditional: this script exists to finish the work, and a
  # resume with no usable checkpoint simply starts from scratch. Note that it
  # makes the reported throughput cover only the final attempt's rows.
  ./run_campaign.sh --resume "$@"
  status=$?

  if (( status == 0 )); then
    echo "[$(date --iso-8601=seconds)] completed after $attempt attempt(s)"
    exit 0
  fi
  if (( status != HANG_EXIT_CODE )); then
    echo "[$(date --iso-8601=seconds)] exit $status is not a watchdog abort; not retrying" >&2
    exit $status
  fi

  echo "[$(date --iso-8601=seconds)] watchdog abort (exit $status); resuming from checkpoint"
  attempt=$((attempt + 1))
done

echo "[$(date --iso-8601=seconds)] gave up after $MAX_ATTEMPTS attempts" >&2
exit 1
