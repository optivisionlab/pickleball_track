#!/usr/bin/env bash
# Zero-shot benchmark of several VLMs on the NSVA clips, following Guide.md steps 2-6.
# Each model gets its own folder runs/<name>/ and runs/model_comparison.{md,csv} compares them all.
#
#   ./run_models.sh                    every model in MODELS, in order (Qwen family first)
#   ./run_models.sh qwen3-vl-8b gemma4-e4b    only these models
#   STAGES=smoke SMOKE_LIMIT=1 ./run_models.sh   only check that every model runs (1 clip each)
#
# Per model (skipped when runs/<name>/STATUS is "done"):
#   batch  largest batch size that fits the GPU (find_batch.py, once per model -> batch_sizes.json);
#          the batch column below is only the fallback
#   smoke  10 val clips; a model that fails here is marked failed and the next one starts
#   val    3 configs on splits/val_fold1.csv, best macro-F1 -> runs/<name>/best_config.txt
#   test   best config on splits/test.csv (all 5 test folds), resumable
#   compare McNemar vs custom v3/v4/v5
# Before each model the script waits until the GPU has enough free memory (other jobs may share it).
#
# Env: BATCH_SCALE (multiply the batch size), WORKERS (decode workers, default 16),
#      SWEEP=0 (skip the val sweep, use FIXED_CONFIG), BASELINE_DIR, FORCE=1 (rerun done models),
#      CPU_FALLBACK=1 (run models that do not fit on the GPU on the CPU instead of skipping them)
# The whole script is one { } block so bash parses it before running: editing this file
# while a benchmark is running does not change the running copy.
{
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate score-video
cd "$(dirname "$0")"

# name | model id | group | fallback batch | VRAM needed (GB) | extra eval_vlm.py args | conda env (default score-video) | device
# device: empty = GPU; "cpu" forces the CPU (very slow: ~1 min per clip, so use SWEEP=0 for it).
# A model that does not fit on the GPU (find_batch.py) or fails the smoke test is skipped and the
# next model starts; CPU_FALLBACK=1 runs a model that does not fit on the CPU instead.
# Remote-code models (MiniCPM-V, Keye) were written for transformers 4.x -> env vlm-tf4.
MODELS=(
  "qwen3.5-4b|Qwen/Qwen3.5-4B|~4B|8|16|"
  "qwen3-vl-4b|Qwen/Qwen3-VL-4B-Instruct|~4B|8|15|"
  "qwen3-vl-8b|Qwen/Qwen3-VL-8B-Instruct|~8B|2|20|"
  "qwen3.5-9b|Qwen/Qwen3.5-9B|~8B|1|22|"
  "internvl3.5-4b|OpenGVLab/InternVL3_5-4B-HF|~4B|4|20|"
  "gemma4-e4b|google/gemma-4-E4B-it|~4B|4|21|"
  "minicpm-v-4|openbmb/MiniCPM-V-4|~4B|1|14||vlm-tf4"
  "internvl3.5-8b|OpenGVLab/InternVL3_5-8B-HF|~8B|4|22|"
  "minicpm-v-4.5|openbmb/MiniCPM-V-4_5|~8B|1|22|--pack 3|vlm-tf4"
  "keye-vl-1.5-8b|Kwai-Keye/Keye-VL-1_5-8B|~8B|2|22||vlm-tf4"
)
# name:num_frames:max_pixels   (200704 = 448x448, 307200 = 640x480)
CONFIGS=("val_16f_448:16:200704" "val_32f_448:32:200704" "val_16f_640:16:307200")
FIXED_CONFIG="${FIXED_CONFIG:-val_16f_640:16:307200}"
STAGES="${STAGES:-batch smoke val test compare}"
WORKERS="${WORKERS:-16}"
BATCH_SCALE="${BATCH_SCALE:-1}"
BASELINE_DIR="${BASELINE_DIR:-/data/Cuong/pickleball/pickleball_track/baseline}"
VAL_CSV=splits/val_fold1.csv
mkdir -p runs logs

has() { [[ " $STAGES " == *" $1 "* ]]; }

wait_for_vram() {   # $1 = GB needed
    local need=$1 free
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        (( free / 1024 >= need )) && return 0
        echo "  waiting for GPU memory: $((free / 1024)) GB free, need $need GB ($(date +%H:%M))"
        sleep 120
    done
}

# shared inputs: splits and per-clip predictions of the custom model
[[ -f splits/test.csv ]] || python make_splits.py --out splits || exit 1
for v in v3 v4 v5; do
    [[ -f "custom_preds_$v.csv" ]] || python export_custom_preds.py --baseline-dir "$BASELINE_DIR" \
        --version "$v" --out "custom_preds_$v.csv" || echo "custom $v predictions not exported" >&2
done

summarize() { python eval_vlm.py summary --runs runs --out runs/model_comparison > logs/summary.log 2>&1; }

run_model() {   # name id group batch need extra env device
    local name=$1 id=$2 group=$3 batch=$4 need=$5 extra=$6 env=${7:-score-video} device=${8:-}
    local PY="/data/miniconda/envs/$env/bin/python"
    local dir="runs/$name" log="logs/$name.log"
    if [[ "${FORCE:-0}" != 1 && -f "$dir/STATUS" && "$(cat "$dir/STATUS")" == done ]]; then
        echo "== $name: already done, skipped"; return 0
    fi
    mkdir -p "$dir"
    echo "running" > "$dir/STATUS"
    echo "== $name ($id) $extra env=$env  [$(date '+%F %H:%M')]" | tee -a "$log"
    [[ "$device" == cpu ]] || wait_for_vram "$need"

    # largest batch that fits on this GPU, measured once per model (find_batch.py -> batch_sizes.json)
    if [[ "$device" != cpu ]] && has batch && ! python -c "import json,sys; sys.exit('$name' not in json.load(open('batch_sizes.json')))" 2>/dev/null; then
        echo "  measuring the largest batch size..." | tee -a "$log"
        # shellcheck disable=SC2086
        "$PY" find_batch.py --name "$name" --model "$id" $extra >> "$log" 2>&1 \
            || echo "  find_batch failed, using the default batch $batch" | tee -a "$log"
    fi
    # batch size per config, measured by find_batch.py; 0 = that config does not fit
    local fits
    fits=$(python -c "import json; d=json.load(open('batch_sizes.json')).get('$name'); print('' if d is None else int(any(v > 0 for v in d.get('batch_by_config', {'all': d['batch']}).values())))" 2>/dev/null)
    if [[ "$fits" == 0 && "$device" != cpu ]]; then
        if [[ "${CPU_FALLBACK:-0}" != 1 ]]; then
            echo "skipped: does not fit on the GPU (CPU_FALLBACK=1 runs it on the CPU)" > "$dir/STATUS"
            echo "  does not fit on the GPU -> skipped" | tee -a "$log"; return 1
        fi
        device=cpu; echo "  does not fit on the GPU -> CPU" | tee -a "$log"
    fi
    local fallback=$batch workers=$WORKERS
    [[ "$device" == cpu ]] && workers=0
    batch_for() {   # $1 = config key (16f_448, 32f_448, 16f_640) -> batch size, 0 = does not fit
        if [[ "$device" == cpu ]]; then echo 1; return; fi
        local b
        b=$(python -c "import json; d=json.load(open('batch_sizes.json')).get('$name') or {}; b=d.get('batch_by_config', {}).get('$1', d.get('batch')); print('' if b is None else b)" 2>/dev/null)
        [[ -z "$b" ]] && b=$fallback
        echo $(( b * BATCH_SCALE ))
    }
    local bjson="" key
    for key in 16f_448 32f_448 16f_640; do bjson+="\"$key\": $(batch_for $key), "; done
    bjson="{${bjson%, }}"
    echo "  device ${device:-cuda}, batch by config $bjson" | tee -a "$log"
    printf '{"model_id": "%s", "group": "%s", "batch_by_config": %s, "extra_args": "%s", "env": "%s", "device": "%s"}\n' \
        "$id" "$group" "$bjson" "$extra" "$env" "${device:-cuda}" > "$dir/model.json"
    # shellcheck disable=SC2206
    local args=(--model "$id" --workers "$workers" --device "${device:-cuda}" $extra)

    if has smoke; then
        local sb; sb=$(batch_for 16f_448); (( sb > 0 )) || sb=1   # smoke uses the 16f/448 default
        "$PY" eval_vlm.py run --csv "$VAL_CSV" --out "$dir/smoke" "${args[@]}" --batch-size "$sb" \
            --limit "${SMOKE_LIMIT:-10}" >> "$log" 2>&1
        if [[ $? -ne 0 || ! -f "$dir/smoke/metrics.json" ]]; then
            echo "failed: smoke (see $log)" > "$dir/STATUS"; echo "  smoke FAILED, see $log"; return 1
        fi
        # letter_mass = share of the next-token probability on A/B/C; when it is low the model
        # does not answer with a letter and its A/B/C probabilities are meaningless
        if ! python -c "
import json, sys
m = json.load(open('$dir/smoke/metrics.json'))
print('  smoke: letter_mass=%.3f errors=%.0f%% vram=%sGB' % (m['mean_letter_mass'], 100 * m['error_rate'], m['peak_vram_gb']))
sys.exit(0 if m['mean_letter_mass'] >= 0.3 and m['error_rate'] < 0.5 else 1)"; then
            echo "failed: smoke (letter_mass or errors, see $dir/smoke)" > "$dir/STATUS"
            echo "  smoke FAILED: letter_mass too low or too many errors"; return 1
        fi
    fi

    if has val; then
        if [[ "${SWEEP:-1}" == 1 ]]; then
            for c in "${CONFIGS[@]}"; do
                IFS=: read -r cname nf mp <<< "$c"
                local vb; vb=$(batch_for "${cname#val_}")
                if (( vb == 0 )); then echo "  $cname: does not fit on the GPU, skipped" | tee -a "$log"; continue; fi
                echo "  $cname (batch $vb)" | tee -a "$log"
                "$PY" eval_vlm.py run --csv "$VAL_CSV" --out "$dir/$cname" "${args[@]}" --batch-size "$vb" \
                    --num-frames "$nf" --max-pixels "$mp" --resume >> "$log" 2>&1 \
                    || echo "  $cname failed, see $log"
            done
            python - "$dir" "${CONFIGS[@]}" <<'EOF'
import json, os, sys
d, configs = sys.argv[1], sys.argv[2:]
scores = {}
for c in configs:
    p = os.path.join(d, c.split(":")[0], "metrics.json")
    if os.path.isfile(p):
        m = json.load(open(p))
        if m["error_rate"] < 0.05:
            scores[c] = m["macro_f1"]
        print(f"    {c:26s} macro-F1={m['macro_f1']:.4f} errors={m['error_rate']:.1%}")
if not scores:
    sys.exit("    no val config finished")
best = max(scores, key=scores.get)
open(os.path.join(d, "best_config.txt"), "w").write(best + "\n")
print("    best:", best)
EOF
            [[ $? -ne 0 ]] && { echo "failed: val" > "$dir/STATUS"; return 1; }
        else
            echo "$FIXED_CONFIG" > "$dir/best_config.txt"
        fi
    fi

    if has test; then
        [[ -f "$dir/best_config.txt" ]] || echo "$FIXED_CONFIG" > "$dir/best_config.txt"
        IFS=: read -r cname nf mp < "$dir/best_config.txt"
        local tb; tb=$(batch_for "${cname#val_}"); (( tb > 0 )) || tb=1
        echo "  test with $cname (batch $tb)" | tee -a "$log"
        "$PY" eval_vlm.py run --csv splits/test.csv --out "$dir/test_final" "${args[@]}" --batch-size "$tb" \
            --num-frames "$nf" --max-pixels "$mp" --resume >> "$log" 2>&1
        if [[ $? -ne 0 || ! -f "$dir/test_final/metrics.json" ]]; then
            echo "failed: test (resume with ./run_models.sh $name)" > "$dir/STATUS"; echo "  test FAILED"; return 1
        fi
    fi

    if has compare && [[ -f "$dir/test_final/predictions.csv" ]]; then
        for v in v3 v4 v5; do
            [[ -f "custom_preds_$v.csv" ]] && python eval_vlm.py compare --csv splits/test.csv \
                --vlm "$dir/test_final/predictions.csv" --custom "custom_preds_$v.csv" >> "$log" 2>&1
        done
    fi
    if has test; then echo done > "$dir/STATUS"; else echo "partial: $STAGES" > "$dir/STATUS"; fi
    summarize
    python -c "import json;m=json.load(open('$dir/test_final/metrics.json'));print('  TEST macro-F1=%.4f acc=%.4f' % (m['macro_f1'], m['accuracy']))" 2>/dev/null
}

selected=("$@")
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name id group batch need extra env device <<< "$entry"
    if (( ${#selected[@]} )) && [[ ! " ${selected[*]} " == *" $name "* ]]; then continue; fi
    run_model "$name" "$id" "$group" "$batch" "$need" "$extra" "$env" "$device"
done
summarize
echo "comparison: runs/model_comparison.md"
cat runs/model_comparison.md
exit
}
