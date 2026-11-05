#!/usr/bin/env bash
# NSVA basketball (made / miss / no_event): v3 / v4 / v5 with the final configuration.
#
#  1. extract_features.py  frozen VideoMAE (mean + 2x2 grid, full / second-half views), CLIP image,
#                          BEATs -> cache/ (resumable; skipped when complete)
#  2. train.py             v3 / v4 / v5, train.py defaults = final configuration, seeds 42, 1, 2
#                          -> outputs/runs/seed_<s>/<version>/{fold_<k>/best.pt, kfold_summary.json}
#  3. summarize.py         results/RESULTS.md
#
# Finished runs (kfold_summary.json) are skipped, so the script is resumable. The GPU is shared:
# every job first waits until enough memory is free.
#   tmux new -d -s basketball 'bash baseline/basketball/run.sh'
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
LOG=logs
mkdir -p "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "[$(date '+%F %T')] $*"; }

wait_gpu() {  # wait_gpu <free MiB>
    local need=$1 free
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "$free" -ge "$need" ] && return 0
        stamp "waiting for GPU: ${free} MiB free, need ${need} MiB"
        sleep 120
    done
}

retry() {  # retry <log> <command...>: up to 4 attempts (e.g. out of memory on the shared GPU)
    local log=$1 attempt; shift
    for attempt in 1 2 3 4; do
        "$@" >> "$log" 2>&1 && return 0
        stamp "$* failed (attempt $attempt, see $log); retry in 5 min"; sleep 300
    done
    return 1
}

if ! python -c "import json, sys; sys.exit(0 if json.load(open('cache/meta.json')).get('complete') else 1)" 2>/dev/null; then
    stamp "START features"
    wait_gpu 3500
    retry "$LOG/extract.log" python extract_features.py --workers "${EXTRACT_WORKERS:-24}" --chunk "${EXTRACT_CHUNK:-32}" || exit 1
fi

for seed in 42 1 2; do
    for v in v3 v4 v5; do
        [ -f "outputs/runs/seed_$seed/$v/kfold_summary.json" ] && { stamp "skip seed_$seed/$v (done)"; continue; }
        stamp "START seed_$seed/$v"
        wait_gpu 1800
        retry "$LOG/seed_${seed}_$v.log" python train.py --version "$v" --run "seed_$seed" --seed "$seed" || exit 1
        grep -h "\[RESULT\]" "$LOG/seed_${seed}_$v.log" | tail -1
    done
done
python summarize.py
stamp "ALL DONE. Results: baseline/basketball/results/RESULTS.md"
