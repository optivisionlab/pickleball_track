#!/usr/bin/env bash
# Ghi tài nguyên server mỗi INTERVAL giây vào thư mục OUT cho tới khi bị kill.
#   docker.csv : ts,container,cpu%,mem_used,mem%  (docker stats)
#   gpu.csv    : ts,gpu_util%,mem_used_mib
#   ai.csv     : ts,busy_slots,pending_jobs,cached_results (AI /stats)
#   events.log : container die/oom/restart
OUT=${1:-./mon}; INTERVAL=${2:-2}
mkdir -p "$OUT"
docker events --filter type=container --filter event=die --filter event=oom \
  --filter event=restart --filter event=start --format '{{.Time}} {{.Action}} {{.Actor.Attributes.name}}' \
  >> "$OUT/events.log" 2>&1 &
EV=$!
trap 'kill $EV 2>/dev/null; exit 0' INT TERM
while true; do
  ts=$(date +%s)
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}' \
    | grep -E 'haui-fitness|pickleball' | sed "s/^/$ts,/" >> "$OUT/docker.csv"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
    | sed "s/^/$ts,/" >> "$OUT/gpu.csv"
  curl -s -m 2 http://127.0.0.1:8888/stats \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(f\"$ts,{d['busy_slots']},{d['pending_jobs']},{d['cached_results']}\")" \
    >> "$OUT/ai.csv" 2>/dev/null || echo "$ts,ERR,ERR,ERR" >> "$OUT/ai.csv"
  sleep "$INTERVAL"
done
