#!/usr/bin/env bash
# Train v3, v4, v5 on the NSVA benchmark (1-1 mapping only) in parallel on one GPU.
#
# 1. Build the shared feature cache once ($PB_ARTIFACTS/baseline/benchmark_features),
#    so the three versions do not decode the videos at the same time.
# 2. Train v3, v4, v5 concurrently; each loads the cache onto the GPU (~1 GB each).
#
# Extra arguments are passed to every version, e.g. ./run_benchmark.sh --batch-size 512
# Logs: baseline/logs/<version>.log
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
mkdir -p logs

echo "[1/2] Building shared feature cache..."
(cd v3 && python train_benchmark.py --extract-only "$@") 2>&1 | tee logs/extract.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Feature extraction failed; see logs/extract.log" >&2
    exit 1
fi

echo "[2/2] Training v3, v4, v5 in parallel..."
declare -A pids
for version in v3 v4 v5; do
    (cd "$version" && python train_benchmark.py "$@") > "logs/$version.log" 2>&1 &
    pids[$version]=$!
    echo "  $version started (pid ${pids[$version]}), log: logs/$version.log"
done

status=0
for version in v3 v4 v5; do
    if wait "${pids[$version]}"; then
        echo "  $version done: $(grep '\[RESULT\]' "logs/$version.log" | tail -1)"
    else
        echo "  $version FAILED; see logs/$version.log" >&2
        status=1
    fi
done
exit $status
