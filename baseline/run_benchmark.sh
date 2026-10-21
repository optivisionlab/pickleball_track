#!/usr/bin/env bash
# Train v3, v4, v5 on the NSVA benchmark (1-1 mapping only) in parallel on one GPU.
#
# 1. Build the shared feature cache once ($PB_ARTIFACTS/baseline/benchmark_features),
#    so the three versions do not decode the videos at the same time.
# 2. Train v3, v4, v5 concurrently; each loads the cache onto the GPU (~1 GB each).
# 3. Copy every version's per-fold best.pt (5 folds each) to
#    $PB_ARTIFACTS/weights_basketball/<version>/fold_<k>/best.pt (replaced on every run).
#
# Extra arguments are passed to every version, e.g. ./run_benchmark.sh --batch-size 512
# Logs: baseline/logs/<version>.log
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
mkdir -p logs
ARTIFACTS_ROOT="${PB_ARTIFACTS:-/data/Cuong/pickleball/artifacts}"
BEST_DIR="$ARTIFACTS_ROOT/weights_basketball"

# Mean test macro-F1 from a kfold_summary.json (-1 if missing).
read_f1() {
    python -c 'import json, sys
try: print(json.load(open(sys.argv[1]))["test_macro_f1_mean"])
except Exception: print(-1)' "$1"
}

echo "[1/3] Building shared feature cache..."
(cd v3 && python train_benchmark.py --extract-only "$@") 2>&1 | tee logs/extract.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Feature extraction failed; see logs/extract.log" >&2
    exit 1
fi

echo "[2/3] Training v3, v4, v5 in parallel..."
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
if [ "$status" -ne 0 ]; then
    echo "Some versions failed; $BEST_DIR left unchanged." >&2
    exit 1
fi

echo "[3/3] Saving weights to $BEST_DIR..."
rm -rf "$BEST_DIR.tmp"
mkdir -p "$BEST_DIR.tmp"
f1s=()
for version in v3 v4 v5; do
    src="$ARTIFACTS_ROOT/baseline/$version/checkpoints_benchmark_1to1"
    cp -r "$src" "$BEST_DIR.tmp/$version"
    cp "logs/$version.log" "$BEST_DIR.tmp/$version/train.log"
    f1=$(read_f1 "$src/kfold_summary.json")
    f1s+=("\"$version\": $f1")
    echo "  $version: $(ls "$BEST_DIR.tmp/$version"/fold_*/best.pt | wc -l) fold best.pt, macro-F1 $f1"
done
cat > "$BEST_DIR.tmp/info.json" <<EOF
{"saved_at": "$(date -Iseconds)", "git_commit": "$(git rev-parse --short HEAD 2>/dev/null)",
 "extra_args": "$*", "test_macro_f1_mean": {$(IFS=,; echo "${f1s[*]}")}}
EOF
rm -rf "$BEST_DIR"
mv "$BEST_DIR.tmp" "$BEST_DIR"
echo "  Done: $BEST_DIR/<version>/fold_<k>/best.pt"
