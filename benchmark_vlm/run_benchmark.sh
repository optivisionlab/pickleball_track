#!/usr/bin/env bash
# Kept for the first run (Qwen3.5-4B); run_models.sh now runs every model.
#   ./run_benchmark.sh   ==   ./run_models.sh qwen3.5-4b
exec "$(dirname "$0")/run_models.sh" qwen3.5-4b "$@"
