#!/usr/bin/env bash
# Zero-shot VLM benchmark on the pickleball data (hit / miss / no-event), same models and
# order as the NSVA run, same steps (batch -> smoke -> val sweep -> test -> McNemar vs custom).
#
#   ./run_benchmark_pickleball.sh                  all models below, in this order
#   ./run_benchmark_pickleball.sh qwen3-vl-8b      only some of them
#   STAGES=smoke SMOKE_LIMIT=1 ./run_benchmark_pickleball.sh   check every model runs
#
# Data (make_splits_pickleball.py, run once; built from baseline/pickleball_common.py):
#   one clip of 32 frames at 30 fps (~1 s) around every labelled time: hit / miss events,
#   and no-event clips from the middle of the gaps between strokes;
#   5 folds grouped by user = the custom model's folds (weights_pickleball/<v>/fold_k).
#   Clips: benchmark_vlm/clips_pickleball
# Results:
#   runs_pickleball/<model>/{smoke,val_*,test_final}/  predictions.csv + metrics.json per stage
#   runs_pickleball/<model>/test_final/comparison_custom_preds_v*.json   McNemar vs custom v3/v4/v5
#   runs_pickleball/model_comparison.{md,csv}          one table, updated after every model
#   logs_pickleball/<model>.log, logs_pickleball/prepare.log, logs_pickleball/run_all.log
# Env (see run_models.sh): FORCE=1, SWEEP=0, BATCH_SCALE, WORKERS, STAGES, CPU_FALLBACK=1
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs_pickleball

# Same models, same order as the NSVA benchmark (MiniCPM-V-4.5 was stopped, Keye never ran).
MODELS_PICKLEBALL=(
  qwen3.5-4b
  qwen3-vl-4b
  qwen3-vl-8b
  qwen3.5-9b
  internvl3.5-4b
  gemma4-e4b
  minicpm-v-4
  internvl3.5-8b
)
(( $# )) && MODELS_PICKLEBALL=("$@")

if [[ ! -f splits_pickleball/test.csv || ! -f custom_preds_pickleball/custom_preds_v3.csv ]]; then
    echo "[prepare] building pickleball clips, folds and custom predictions..."
    CUDA_VISIBLE_DEVICES= /data/miniconda/envs/anything/bin/python make_splits_pickleball.py \
        > logs_pickleball/prepare.log 2>&1 || { echo "prepare failed, see logs_pickleball/prepare.log" >&2; exit 1; }
fi
grep -v "clips: video" logs_pickleball/prepare.log | grep -iv warn | tail -12

RUNS=runs_pickleball LOGS=logs_pickleball SPLITS=splits_pickleball CUSTOM_DIR=custom_preds_pickleball \
PROMPT_FILE=prompts/pickleball.txt TITLE="Pickleball" \
    ./run_models.sh "${MODELS_PICKLEBALL[@]}" 2>&1 | tee -a logs_pickleball/run_all.log
