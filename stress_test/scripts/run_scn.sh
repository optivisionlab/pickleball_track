#!/usr/bin/env bash
# Chạy 1 kịch bản scenario.py kèm giám sát: run_scn.sh <tên> [tham số scenario.py...]
set -u
ulimit -n "$(ulimit -Hn)" 2>/dev/null
HERE=$(cd "$(dirname "$0")/.." && pwd); NAME=$1; shift
R=$HERE/results/v2/$NAME; rm -rf "$R"; mkdir -p "$R"
bash "$HERE/scripts/monitor.sh" "$R/mon" 2 & MON=$!
python3 "$HERE/scripts/scenario.py" --name "$NAME" "$@" 2> "$R/stderr.log" | tee "$R/stdout.log"
kill $MON 2>/dev/null; wait $MON 2>/dev/null
python3 "$HERE/scripts/peaks.py" "$R/mon" | tee -a "$R/stdout.log"
