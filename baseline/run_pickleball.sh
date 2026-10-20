#!/usr/bin/env bash
# Train v3, v4, v5 on the pickleball data (baseline/pickleball_common.py) in parallel on one GPU.
#
# 1. Build the shared feature cache once ($PB_ARTIFACTS/baseline/pickleball_features),
#    so the three versions do not decode the videos at the same time.
#    If the cache already exists it is only loaded.
# 2. Train v3, v4, v5 concurrently; each loads the cache onto the GPU (~2-3 GB each).
#
# Extra arguments are passed to every version, e.g. ./run_pickleball.sh --spatial mean
# Logs: baseline/logs_pickleball/<version>.log
# Results: $PB_ARTIFACTS/baseline/<version>/checkpoints_pickleball/cv_summary.{json,csv}
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
mkdir -p logs_pickleball

echo "[1/2] Building shared feature cache..."
(cd v3 && python train.py --extract-only --workers 12 "$@") 2>&1 | tee logs_pickleball/extract.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Feature extraction failed; see logs_pickleball/extract.log" >&2
    exit 1
fi

echo "[2/2] Training v3, v4, v5 in parallel..."
declare -A pids
for version in v3 v4 v5; do
    (cd "$version" && python train.py "$@") > "logs_pickleball/$version.log" 2>&1 &
    pids[$version]=$!
    echo "  $version started (pid ${pids[$version]}), log: logs_pickleball/$version.log"
done

status=0
for version in v3 v4 v5; do
    if wait "${pids[$version]}"; then
        echo "  $version done:"
        grep '\[RESULT\]' "logs_pickleball/$version.log" | sed 's/^/    /'
    else
        echo "  $version FAILED; see logs_pickleball/$version.log" >&2
        status=1
    fi
done
exit $status
