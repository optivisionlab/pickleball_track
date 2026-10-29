#!/usr/bin/env bash
# Ghi tài nguyên server mỗi INTERVAL giây vào thư mục OUT cho tới khi bị kill.
#   docker.csv  : ts,container,cpu%,mem_used,mem%  (docker stats)
#   ai.csv      : ts,instance,busy_slots,pending_jobs,waiting_short,waiting_long (từng instance AI)
#   kafka.csv   : ts,group,topic,lag (mỗi ~15 s)
#   mongo.csv   : ts,pids
#   host.csv    : ts,cpu_busy%,mem_used_gb,load1,disk_read_mb,disk_write_mb,net_rx_mb,net_tx_mb (delta mỗi chu kỳ)
#   gpu.csv     : ts,gpu_util%,mem_used_mib,temp_c,power_w
#   gpu_procs.csv: ts,pid,mem_mib,process_name
#   events.log  : container die/oom/restart
OUT=${1:-./mon}; INTERVAL=${2:-2}
mkdir -p "$OUT"
docker events --filter type=container --filter event=die --filter event=oom \
  --filter event=restart --filter event=start --format '{{.Time}} {{.Action}} {{.Actor.Attributes.name}}' \
  >> "$OUT/events.log" 2>&1 &
EV=$!
AI_IPS=$(for c in $(docker ps --format '{{.Names}}' | grep -E 'pickleball-ai-[0-9]+$|pickleball-ai-pickleball-ai-[0-9]+$'); do
  echo "$c=$(docker inspect "$c" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"; done)
MONGO_CG=/sys/fs/cgroup/system.slice/docker-$(docker inspect haui-fitness-mongodb-1 --format '{{.Id}}' 2>/dev/null).scope
(
  while true; do
    ts=$(date +%s)
    docker exec haui-fitness-kafka-1 /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server kafka:9093 \
      --describe --all-groups 2>/dev/null | awk -v ts="$ts" '$1 !~ /GROUP|^$/ && $6 ~ /^[0-9]+$/ {lag[$1","$2]+=$6} END {for (k in lag) print ts","k","lag[k]}' \
      >> "$OUT/kafka.csv"
    sleep 15
  done
) & KF=$!
trap 'kill $EV $KF 2>/dev/null; exit 0' INT TERM
while true; do
  ts=$(date +%s)
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}' \
    | grep -E 'haui-fitness|pickleball' | sed "s/^/$ts,/" >> "$OUT/docker.csv"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits \
    | sed "s/^/$ts,/" >> "$OUT/gpu.csv"
  # mọi tiến trình đang dùng GPU (phát hiện tiến trình ngoài hệ thống chen vào)
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits \
    | sed "s/^/$ts,/" >> "$OUT/gpu_procs.csv"
  for pair in $AI_IPS; do
    name=${pair%%=*}; ip=${pair##*=}
    curl -s -m 2 "http://$ip:8888/stats" | python3 -c "import sys,json;d=json.load(sys.stdin);print(f\"$ts,$name,{d['busy_slots']},{d['pending_jobs']},{d.get('waiting_short',0)},{d.get('waiting_long',0)}\")" \
      >> "$OUT/ai.csv" 2>/dev/null || echo "$ts,$name,ERR,ERR,ERR,ERR" >> "$OUT/ai.csv"
  done
  [ -f "$MONGO_CG/pids.current" ] && echo "$ts,$(cat "$MONGO_CG/pids.current")" >> "$OUT/mongo.csv"
  python3 "$(dirname "$0")/host_sample.py" "$OUT/.host_prev" "$ts" >> "$OUT/host.csv" 2>/dev/null
  sleep "$INTERVAL"
done
