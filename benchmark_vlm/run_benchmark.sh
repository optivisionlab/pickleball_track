#!/usr/bin/env bash
# Zero-shot Qwen3.5-4B benchmark on the NSVA clips, following Guide.md steps 2-6.
#
#   ./run_benchmark.sh            all stages
#   ./run_benchmark.sh val        only one stage: splits | smoke | val | test | compare
#
# Stages
#   splits   rebuild the custom model's folds -> splits/ (checked against its saved test games)
#   smoke    10 val clips (Guide step 3)
#   val      3 configs on splits/val_fold1.csv, best macro-F1 -> runs/best_config.txt (step 4)
#   test     best config on splits/test.csv = all 5 test folds, resumable (step 5)
#   compare  per-clip predictions of custom v3/v4/v5, then McNemar vs the VLM (step 6)
# Extra arguments after the stage go to every eval_vlm.py run, e.g. ./run_benchmark.sh test --attn sdpa
# BATCH (clips per forward pass, default 8) and WORKERS (CPU decode workers, default 16):
#   BATCH=16 WORKERS=24 ./run_benchmark.sh test
set -euo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate score-video
cd "$(dirname "$0")"

STAGE="${1:-all}"
shift || true
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
BASELINE_DIR="${BASELINE_DIR:-/data/Cuong/pickleball/pickleball_track/baseline}"
VAL_CSV=splits/val_fold1.csv
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-16}"
RUN_ARGS=(--model "$MODEL" --batch-size "$BATCH" --workers "$WORKERS")
mkdir -p runs logs

want() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }

if want splits; then
    python make_splits.py --out splits
fi

if want smoke; then
    python eval_vlm.py run --csv "$VAL_CSV" --out runs/smoke "${RUN_ARGS[@]}" --limit 10 "$@"
fi

# name:num_frames:max_pixels   (200704 = 448x448, 307200 = 640x480)
CONFIGS=("val_16f_448:16:200704" "val_32f_448:32:200704" "val_16f_640:16:307200")

if want val; then
    for c in "${CONFIGS[@]}"; do
        IFS=: read -r name nf mp <<< "$c"
        echo "== $name"
        python eval_vlm.py run --csv "$VAL_CSV" --out "runs/$name" "${RUN_ARGS[@]}" \
            --num-frames "$nf" --max-pixels "$mp" --resume "$@" 2>&1 | tee "logs/$name.log"
    done
    python - "${CONFIGS[@]}" <<'EOF'
import json, sys
best = max(sys.argv[1:], key=lambda c: json.load(open(f"runs/{c.split(':')[0]}/metrics.json"))["macro_f1"])
for c in sys.argv[1:]:
    m = json.load(open(f"runs/{c.split(':')[0]}/metrics.json"))
    print(f"{c:28s} macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
open("runs/best_config.txt", "w").write(best + "\n")
print("best:", best)
EOF
fi

if want test; then
    [[ -f runs/best_config.txt ]] || { echo "run the val stage first" >&2; exit 1; }
    IFS=: read -r name nf mp < runs/best_config.txt
    echo "== test with $name (num_frames=$nf, max_pixels=$mp)"
    python eval_vlm.py run --csv splits/test.csv --out runs/test_final "${RUN_ARGS[@]}" \
        --num-frames "$nf" --max-pixels "$mp" --resume "$@" 2>&1 | tee logs/test_final.log
fi

if want compare; then
    for v in v3 v4 v5; do
        [[ -f "custom_preds_$v.csv" ]] || \
            python export_custom_preds.py --baseline-dir "$BASELINE_DIR" --version "$v" --out "custom_preds_$v.csv"
        python eval_vlm.py compare --csv splits/test.csv --vlm runs/test_final/predictions.csv \
            --custom "custom_preds_$v.csv" | tee "logs/compare_$v.log"
    done
fi
