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
# 3. Copy every version's per-fold best.pt (5 folds each) to
#    $PB_ARTIFACTS/weights_pickleball/<version>/fold_<k>/best.pt (replaced on every run).
set -uo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
mkdir -p logs_pickleball
ARTIFACTS_ROOT="${PB_ARTIFACTS:-/data/Cuong/pickleball/artifacts}"
BEST_DIR="$ARTIFACTS_ROOT/weights_pickleball"

# Headline scores from a cv_summary.json as a JSON object.
read_scores() {
    python -c 'import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps({"pooled_token_macro_f1": d["pooled_token"]["macro_f1"],
                  "fold_token_macro_f1_mean": d["fold_token_macro_f1_mean"],
                  "event_macro_f1": d["event"]["macro_f1"]}))' "$1"
}

echo "[1/3] Building shared feature cache..."
(cd v3 && python train.py --extract-only --workers 12 "$@") 2>&1 | tee logs_pickleball/extract.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Feature extraction failed; see logs_pickleball/extract.log" >&2
    exit 1
fi

echo "[2/3] Training v3, v4, v5 in parallel..."
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
if [ "$status" -ne 0 ]; then
    echo "Some versions failed; $BEST_DIR left unchanged." >&2
    exit 1
fi

echo "[3/3] Saving weights to $BEST_DIR..."
rm -rf "$BEST_DIR.tmp"
mkdir -p "$BEST_DIR.tmp"
scores=()
for version in v3 v4 v5; do
    src="$ARTIFACTS_ROOT/baseline/$version/checkpoints_pickleball"
    cp -r "$src" "$BEST_DIR.tmp/$version"
    cp "logs_pickleball/$version.log" "$BEST_DIR.tmp/$version/train.log"
    score=$(read_scores "$src/cv_summary.json")
    scores+=("\"$version\": $score")
    echo "  $version: $(ls "$BEST_DIR.tmp/$version"/fold_*/best.pt | wc -l) fold best.pt, $score"
done
cat > "$BEST_DIR.tmp/info.json" <<EOF
{"saved_at": "$(date -Iseconds)", "git_commit": "$(git rev-parse --short HEAD 2>/dev/null)",
 "extra_args": "$*", "results": {$(IFS=,; echo "${scores[*]}")}}
EOF
rm -rf "$BEST_DIR"
mv "$BEST_DIR.tmp" "$BEST_DIR"
echo "  Done: $BEST_DIR/<version>/fold_<k>/best.pt"
