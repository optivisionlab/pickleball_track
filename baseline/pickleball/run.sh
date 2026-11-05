#!/usr/bin/env bash
# Train v3, v4, v5 on the pickleball data (pickleball_common.py) in parallel on one GPU,
# with the final configuration (the train.py defaults, e.g. --select loss).
#
# 1. Build the shared feature cache once (cache/),
#    so the three versions do not decode the videos at the same time.
#    If the cache already exists it is only loaded.
# 2. Train v3, v4, v5 concurrently; each loads the cache onto the GPU (~2-3 GB each).
#
# Extra arguments are passed to every version, e.g. ./run.sh --spatial mean
# Logs: baseline/pickleball/logs/<version>.log
# Results: outputs/runs/<version>/cv_summary.{json,csv}
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
mkdir -p logs

echo "[1/2] Building shared feature cache..."
python train.py --version v3 --extract-only --workers 12 "$@" 2>&1 | tee logs/extract.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Feature extraction failed; see logs/extract.log" >&2
    exit 1
fi

echo "[2/2] Training v3, v4, v5 in parallel..."
declare -A pids
for version in v3 v4 v5; do
    python train.py --version "$version" "$@" > "logs/$version.log" 2>&1 &
    pids[$version]=$!
    echo "  $version started (pid ${pids[$version]}), log: logs/$version.log"
done

status=0
for version in v3 v4 v5; do
    if wait "${pids[$version]}"; then
        echo "  $version done:"
        grep '\[RESULT\]' "logs/$version.log" | sed 's/^/    /'
    else
        echo "  $version FAILED; see logs/$version.log" >&2
        status=1
    fi
done
if [ "$status" -ne 0 ]; then
    echo "Some versions failed." >&2
    exit 1
fi

for version in v3 v4 v5; do cp "logs/$version.log" "outputs/runs/$version/train.log"; done
echo "Done: outputs/runs/<version>/{cv_summary.json, fold_<k>/best.pt, train.log}"
