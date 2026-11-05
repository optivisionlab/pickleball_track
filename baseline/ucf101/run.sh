#!/usr/bin/env bash
# UCF-101, final configurations:
#   1. extract_features.py   frozen VideoMAE ssl/k400 (mean + 2x2 grid, 6 views), CLIP image, BEATs -> cache/
#   2. train.py              v3 / v4 / v5 heads on frozen K400 + grid + 6 views + CLIP image + BEATs ft
#                            -> outputs/runs/frozen/<version>/{split<k>.pt, summary.json}
#   3. finetune.py           VideoMAE-K400 fine-tuned end-to-end with the v3 head (video + audio)
#                            -> outputs/runs/ft_k400_av/split<k>/{final.pt, result.json}
#   4. summarize.py          results/RESULTS.md
# Finished steps are skipped (fine-tuning also resumes mid-split from last.pt), and failed steps
# are retried (the GPU is shared).
#   tmux new -d -s ucf101 'bash baseline/ucf101/run.sh'
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
LOG=logs
mkdir -p "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "[$(date '+%F %T')] $*"; }

retry() {  # retry <log> <command...>: up to MAX_ATTEMPTS attempts, 10 min apart
    local log=$1 attempt; shift
    for attempt in $(seq 1 "${MAX_ATTEMPTS:-12}"); do
        "$@" >> "$log" 2>&1 && return 0
        stamp "$* failed (attempt $attempt, see $log); retry in 10 min"; sleep 600
    done
    return 1
}

if ! python -c "import json, sys; sys.exit(0 if json.load(open('cache/meta.json')).get('complete') else 1)" 2>/dev/null; then
    stamp "START features"
    retry "$LOG/extract.log" python extract_features.py --workers 16 || exit 1
fi

for v in v3 v4 v5; do
    [ -f "outputs/runs/frozen/$v/summary.json" ] && { stamp "skip frozen/$v (done)"; continue; }
    stamp "START frozen/$v"
    retry "$LOG/frozen_$v.log" python train.py --version "$v" --run frozen \
        --backbone k400 --spatial grid --views all --beats ft --clip-image || exit 1
    grep -h "\[RESULT\]" "$LOG/frozen_$v.log" | tail -1
done

for k in 1 2 3; do  # ~2-4 h per split alone on the GPU
    [ -f "outputs/runs/ft_k400_av/split$k/result.json" ] && { stamp "skip ft_k400_av/split$k (done)"; continue; }
    stamp "START ft_k400_av/split$k"
    retry "$LOG/ft_k400_av_split$k.log" python finetune.py --backbone k400 --split "$k" --epochs 25 || exit 1
    grep -h "FT RESULT" "$LOG/ft_k400_av_split$k.log" | tail -1
done

python summarize.py
stamp "ALL DONE. Results: baseline/ucf101/results/RESULTS.md"
