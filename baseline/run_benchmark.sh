#!/usr/bin/env bash
# Train v3, v4, v5 on the NSVA benchmark (1-1 mapping only).
# Features are extracted once (first run) into $PB_ARTIFACTS/baseline/benchmark_features
# and reused by the other versions. Extra arguments are passed to every version.
set -euo pipefail
source /data/miniconda/etc/profile.d/conda.sh
conda activate anything
cd "$(dirname "$0")"
for version in v3 v4 v5; do
    (cd "$version" && python train_benchmark.py "$@")
done
