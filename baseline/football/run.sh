#!/usr/bin/env bash
# Football (SoccerNet shots, 15 s clips): v3 / v4 / v5 with the best configuration of each version.
#
#   1. extract_features.py  VideoMAE k400 (mean + grid), CLIP image, BEATs -> cache/ (resumable)
#   2. transcribe.py        Whisper commentary -> CLIP text (v4 / v5 text input) -> cache/
#   3. train.py             5 seeds (42-46) per version -> outputs/runs[_binary]/seed_<s>/<version>/
#                           {result.json, probs_{valid,test}.npy, best.pt}
#   4. summarize.py         results[_binary]/RESULTS.md
#
#   ./run.sh            goal / saved / off_target
#   ./run.sh --binary   goal / saved (off_target merged into saved), FB_TASK=binary
#
# The configurations were chosen on the valid split (part1) by a search over inputs and
# hyper-parameters (README.md); test (part2) is only reported. Finished runs are skipped.
#   tmux new -d -s football 'bash baseline/football/run.sh'
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
if [ "${1:-}" = "--binary" ]; then export FB_TASK=binary; fi
TASK=${FB_TASK:-3class}
SUFFIX=${FB_TASK:+_$FB_TASK}
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "[$(date '+%F %T')] $*"; }

retry() {  # retry <log> <command...>: resumable jobs, up to 4 attempts
    local log=$1 attempt; shift
    for attempt in 1 2 3 4; do
        "$@" >> "$log" 2>&1 && return 0
        stamp "$* failed (attempt $attempt, see $log); retry in 2 min"; sleep 120
    done
    return 1
}

# Inputs (same for every version and both tasks).
INPUTS="--backbone k400 --spatial grid --window both --beats base --text asr --clip-image"
declare -A BEST
BEST[3class_v3]="--batch-size 64 --lr 0.0005 --weight-decay 0.1 --label-smoothing 0.0 --class-weight balanced --head-dropout 0.2 --token-drop 0.1 --audio-dropout 0.2 --mixup 0.2 --ema 0.999"
BEST[3class_v4]="--epochs 60 --batch-size 64 --lr 0.0005 --label-smoothing 0.0 --fusion-dim 512 --cross-heads 8 --temporal-heads 8 --dropout 0.3 --head-dropout 0.5 --token-drop 0.2 --text-dropout 0.4 --ema 0.999"
BEST[3class_v5]="--epochs 60 --lr 0.001 --weight-decay 0.05 --label-smoothing 0.2 --class-weight sqrt --fusion-dim 384 --cross-heads 8 --temporal-heads 8 --temporal-layers 1 --token-drop 0.2 --text-dropout 0.2 --mixup 0.2 --ema 0.999"
BEST[binary_v3]="--batch-size 64 --lr 0.0001 --weight-decay 0.05 --label-smoothing 0.0 --head-dropout 0.2 --token-drop 0.1 --mixup 0.2"
BEST[binary_v4]="--epochs 60 --batch-size 256 --lr 0.0005 --weight-decay 0.1 --label-smoothing 0.0 --class-weight sqrt --fusion-dim 512 --cross-heads 8 --temporal-heads 8 --dropout 0.3 --head-dropout 0.5 --token-drop 0.1 --text-dropout 0.4 --mixup 0.2"
BEST[binary_v5]="--epochs 60 --batch-size 64 --lr 0.0005 --weight-decay 0.1 --label-smoothing 0.0 --class-weight sqrt --fusion-dim 384 --cross-heads 8 --temporal-heads 8 --temporal-layers 3 --head-dropout 0.2 --mixup 0.2"

stamp "START features"
retry logs/extract.log python extract_features.py --workers "${EXTRACT_WORKERS:-40}" --batch 4 || exit 1
stamp "START transcribe"
retry logs/transcribe.log python transcribe.py || exit 1

for seed in 42 43 44 45 46; do
    for v in v3 v4 v5; do
        out="outputs/runs$SUFFIX/seed_$seed/$v"
        [ -f "$out/result.json" ] && { stamp "skip $out (done)"; continue; }
        stamp "START $TASK seed_$seed/$v"
        retry "logs/${TASK}_seed_${seed}_$v.log" python train.py --version "$v" --run "seed_$seed" --seed "$seed" \
            --save-ckpt $INPUTS ${BEST[${TASK}_$v]} || exit 1
        grep -h "\[RESULT\]" "logs/${TASK}_seed_${seed}_$v.log" | tail -1
    done
done
python summarize.py
stamp "ALL DONE -> results$SUFFIX/RESULTS.md"
