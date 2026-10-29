#!/usr/bin/env bash
# Load test V3 (KICH_BAN_TEST_V3.md). run_v3.sh gd1 | gd2 | gd3 | aiscale | h2 | gd4 | gd5 | h1
set -u
cd "$(dirname "$0")/.."
L=$PWD; TOK=$L/jmeter/tokens_v2.csv; ACC=$L/jmeter/accounts_v2.csv; H="-Jhost=127.0.0.1"
K1=loadtest/shot1_b.mp4
K2=loadtest/shot2.mov
AIDIR=/data/Cuong/pickleball/pickleball_track_score_video/score_video_block
BEDIR=/data/Cuong/pickleball/haui_sict_fitness_score
log(){ echo "[$(date '+%F %T')] $*"; }
disk_ok(){ f=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$f" -ge 25 ] || { log "STOP: disk free ${f}G < 25G"; exit 1; }; }
jm(){ disk_ok; log "JM $1"; bash scripts/run_jm.sh "$@" $H | grep -E "^TOTAL|^POST|errors|peak (haui-fitness-(backend|mongodb|frontend)|pickleball-ai)|GPU|HOST|events"; }
scn(){ disk_ok; log "SCN $1"; bash scripts/run_scn.sh "$@" | grep -vE "^peak haui-fitness-(redis|minio)"; }
ai_ips(){ for c in $(docker ps --format '{{.Names}}' | grep -E 'pickleball-ai-pickleball-ai-[0-9]+$'); do docker inspect "$c" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'; done; }
wait_idle(){
  for i in $(seq 1 720); do
    p=$(for ip in $(ai_ips); do curl -s -m 2 "http://$ip:8888/stats" | python3 -c "import sys,json;print(json.load(sys.stdin)['pending_jobs'])" 2>/dev/null || echo 0; done | awk '{s+=$1} END{print s+0}')
    lag=$(docker exec haui-fitness-kafka-1 /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server kafka:9093 --describe --all-groups 2>/dev/null | awk '$1 ~ /^(ai-grading-worker|ai-grading-worker-large|backend-result-consumer)$/ && $6 ~ /^[0-9]+$/ {s+=$6} END{print s+0}')
    [ "$p" = 0 ] && [ "$lag" = 0 ] && { log "idle"; sleep 20; return; }
    sleep 10
  done; log "WARN: not idle after 2h"
}
ai_set(){  # ai_set <replicas> <decoder>
  log "AI -> replicas=$1 decoder=$2"
  (cd $AIDIR && AI_REPLICAS=$1 PICKLEBALL_DECODER=$2 docker compose up -d --force-recreate pickleball-ai >/dev/null 2>&1)
  for i in $(seq 1 120); do [ "$(cd $AIDIR && docker compose ps pickleball-ai --format '{{.Status}}' | grep -c '(healthy)')" -ge "$1" ] && break; sleep 5; done
  (cd $AIDIR && docker compose restart pickleball-lb >/dev/null 2>&1); sleep 10
  log "AI healthy=$(cd $AIDIR && docker compose ps pickleball-ai --format '{{.Status}}' | grep -c '(healthy)') VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
one_line(){ printf '%s\n' "$1" > "$L/one_video.txt"; echo "$L/one_video.txt"; }

gd1(){
  wait_idle
  log "T1.1 health"
  for u in http://127.0.0.1:4200/ http://127.0.0.1:8000/health/ready http://127.0.0.1:8888/lb-health; do
    curl -s -o /dev/null -w "  $u -> %{http_code} %{time_total}s\n" $u; done
  for ip in $(ai_ips); do echo "  AI $ip $(curl -s http://$ip:8888/stats)"; done
  docker ps --format '  {{.Names}} {{.Status}}' | sort
  log "T1.3 single E2E per video type"
  scn v3_T1_3_short --duration 5 --ramp 1 --burst --exam-submits 1 --short-ratio 1 --submit-window 1 --poll 1
  scn v3_T1_3_long60 --duration 5 --ramp 1 --burst --exam-submits 1 --short-ratio 0 --submit-window 1 --poll 1
  for d in 30 90 150; do
    line=$(awk -v d=$d '$1>=d && $1<d+30' real40.txt | head -1)
    scn v3_T1_3_real${d} --duration 5 --ramp 1 --real-list "$(one_line "$line")" --submit-window 1 --poll 1
  done
  log "T1.2 per-endpoint latency"
  rid=$(python3 -c "import csv;print([r for r in csv.DictReader(open('results/v2/v3_T1_3_short/submissions.csv'))][0]['result_id'])")
  python3 scripts/endpoint_latency.py "$rid" 200
  log "T1.4 AI stage breakdown"
  docker run --rm -u "$(id -u):$(id -g)" -e USER=devlab1 -e HOME=/tmp -e TORCHINDUCTOR_CACHE_DIR=/tmp/ind -e HF_HOME=/tmp/hf \
    --gpus all --shm-size 2g --env-file $AIDIR/.env -e PICKLEBALL_DEVICE=cuda \
    -v /data/Cuong/pickleball/pickleball_track/score_video_block/src/weights:/app/src/weights:ro \
    -v $L/videos:/v:ro -v /data/Cuong/pickleball/data/raw:/data:ro -v $L/scripts/bench_ai.py:/tmp/bench_ai.py:ro \
    pickleball-track:gpu-opt python /tmp/bench_ai.py /v/shot1_b.mp4 /v/shot1_b.mp4 /v/shot2.mov \
    "/data/$(basename "$(awk '$1>=90 && $1<120' real40.txt | head -1 | cut -d' ' -f3-)")" 2>&1 | grep -E "^load_models|^/"
}
t25(){  # chạy lại T2.5/T2.6 (AI gọi thẳng) sau khi tải lại video mẫu
  for c in 6 12 24 36 48; do jm v3_T2_5_ai_short_c$c ai_direct.jmx -Jobject_key=$K1 -Jthreads=$c -Jramp=0 -Jloops=4 -Jduration=3600; done
  for c in 3 6 12; do jm v3_T2_6_ai_long_c$c ai_direct.jmx -Jobject_key=$K2 -Jthreads=$c -Jramp=0 -Jloops=1 -Jduration=3600; done
}
gd2(){
  for u in 100 500 1000 2000 5000 10000; do jm v3_T2_2_flow_u$u be_user_flow_tok.jmx -Jtokens=$TOK -Jthreads=$u -Jramp=30 -Jduration=120; done
  for u in 10 50 100 200 500; do jm v3_T2_3_login_u$u be_login.jmx -Jaccounts=$ACC -Jpassword=Loadtest@2026 -Jthreads=$u -Jramp=5 -Jduration=120; done
  for u in 10 50 100 200; do jm v3_T2_4_teacher_u$u teacher_flow.jmx -Jthreads=$u -Jramp=10 -Jduration=120; done
  for c in 6 12 24 36 48; do jm v3_T2_5_ai_short_c$c ai_direct.jmx -Jobject_key=$K1 -Jthreads=$c -Jramp=0 -Jloops=4 -Jduration=3600; done
  for c in 3 6 12; do jm v3_T2_6_ai_long_c$c ai_direct.jmx -Jobject_key=$K2 -Jthreads=$c -Jramp=0 -Jloops=1 -Jduration=3600; done
  for n in 10 50 100; do disk_ok; log "T2.7 upload x$n"; python3 scripts/upload_test.py $n v3_T2_7_upload_$n; done
}
gd3(){
  wait_idle; scn v3_T3_1_normal_day --duration 1200 --ramp 60 --browsers 300 --teachers 5 --submit-every 20 --short-ratio 0.8
  wait_idle; scn v3_T3_2_exam_start --duration 300 --ramp 30 --browsers 200 --think-min 2 --think-max 5 --burst --login-burst 200 --login-window 30 --exam-submits 150 --submit-window 300 --short-ratio 0.7
  wait_idle; scn v3_T3_3_deadline --duration 600 --ramp 30 --browsers 300 --submit-every 6 --short-ratio 0.8 --double-click 0.2
  wait_idle; scn v3_T3_4_poll_storm --duration 300 --ramp 20 --pollers 1000 --poll 2 --burst --exam-submits 300 --submit-window 60 --short-ratio 1.0
  wait_idle; scn v3_T3_5_real_opencv --duration 30 --ramp 1 --real-list $L/real40.txt --submit-window 60 --poll 5
  wait_idle; ai_set 3 ffmpeg
  scn v3_T3_5_real_ffmpeg --duration 30 --ramp 1 --real-list $L/real40.txt --submit-window 60 --poll 5
  wait_idle; ai_set 3 opencv
  scn v3_T3_6_soak --duration 3600 --ramp 60 --browsers 300 --teachers 5 --submit-every 20 --short-ratio 0.8
  wait_idle
}
gd3r(){  # chạy lại các bài GD3 phụ thuộc GPU sau khi dừng tiến trình training dùng chung GPU
  ai_set 3 opencv
  wait_idle; scn v3r_T3_2_exam_start --duration 300 --ramp 30 --browsers 200 --think-min 2 --think-max 5 --burst --login-burst 200 --login-window 30 --exam-submits 150 --submit-window 300 --short-ratio 0.7
  wait_idle; scn v3r_T3_5_real_opencv --duration 30 --ramp 1 --real-list $L/real40.txt --submit-window 60 --poll 5
  wait_idle; ai_set 3 ffmpeg
  scn v3r_T3_5_real_ffmpeg --duration 30 --ramp 1 --real-list $L/real40.txt --submit-window 60 --poll 5
  wait_idle; ai_set 3 opencv
}
gdfix(){  # chạy lại các bài video bị training chiếm GPU (07:05:34–08:51:00) trên cấu hình H1
  A=results/v3_shared_gpu2; mkdir -p $A/v2 $A/jm
  mv results/v2/v3_T3_1_normal_day results/v2/v3_T3_2_exam_start results/v2/v3_T3_3_deadline $A/v2/ 2>/dev/null
  mv results/v3_T2_6_ai_long_c6 results/v3_T2_6_ai_long_c12 $A/jm/ 2>/dev/null
  ai_set 3 opencv
  wait_idle
  for c in 6 12; do jm v3_T2_6_ai_long_c$c ai_direct.jmx -Jobject_key=$K2 -Jthreads=$c -Jramp=0 -Jloops=1 -Jduration=3600; done
  wait_idle; scn v3_T3_1_normal_day --duration 1200 --ramp 60 --browsers 300 --teachers 5 --submit-every 20 --short-ratio 0.8
  wait_idle; scn v3_T3_2_exam_start --duration 300 --ramp 30 --browsers 200 --think-min 2 --think-max 5 --burst --login-burst 200 --login-window 30 --exam-submits 150 --submit-window 300 --short-ratio 0.7
  wait_idle; scn v3_T3_3_deadline --duration 600 --ramp 30 --browsers 300 --submit-every 6 --short-ratio 0.8 --double-click 0.2
  wait_idle
}
t35(){  # chạy lại T3.5 trên lô tài khoản mới (lần trước ffmpeg bị 403 vì tài khoản DONE nên số video không bằng nhau)
  A=results/v3_acc_done; mkdir -p $A; mv results/v2/v3_T3_5_real_opencv results/v2/v3_T3_5_real_ffmpeg $A/ 2>/dev/null
  ai_set 3 opencv; wait_idle; scn v3_T3_5_real_opencv --duration 30 --ramp 1 --real-list $L/real40.txt --submit-window 60 --poll 5
  wait_idle; ai_set 3 ffmpeg; scn v3_T3_5_real_ffmpeg --duration 30 --ramp 1 --real-list $L/real40.txt --submit-window 60 --poll 5
  wait_idle; ai_set 3 opencv
}
aiscale(){  # T5.4: số instance AI tối đa trên 1 GPU; aiscale <decoder> <n...>
  dec=$1; shift
  for n in "$@"; do
    ai_set $n $dec; disk_ok
    bash scripts/bench_ai.sh v3_scale_${dec}_x$n 2>&1 | grep -E "W[123]|GPU"
    docker ps -a --format '{{.Names}} {{.Status}}' | grep pickleball-ai-pickleball-ai | grep -v "(healthy)" || true
  done
}
h2(){  # cấu hình mở rộng H2: h2 <ai_replicas> <decoder>
  log "switch -> H2"
  (cd $BEDIR && BACKEND_WORKERS=16 BACKEND_CPUS=16 BACKEND_MEM_LIMIT=8g MONGODB_CPUS=8 MONGODB_MEM_LIMIT=8g \
     AI_MAX_CONCURRENCY=$(( $1 * 2 + 2 )) AI_POOL_SMALL_CONCURRENCY=$(( $1 * 2 - 1 )) AI_POOL_LARGE_CONCURRENCY=3 \
     docker compose up -d --no-deps mongodb backend ai-worker 2>&1 | tail -1)
  for i in $(seq 1 60); do curl -sf -m 2 localhost:8000/health/ready >/dev/null && break; sleep 3; done
  docker update --cpus 4 --memory 512m --memory-swap 512m --pids-limit 256 haui-fitness-frontend-1 >/dev/null
  docker update --cpus 4 --pids-limit 1024 haui-fitness-kafka-1 >/dev/null
  docker update --cpus 2 haui-fitness-ai-worker-1 >/dev/null
  docker update --cpus 4 --memory 2g --memory-swap 2g --pids-limit 1024 haui-fitness-minio-1 >/dev/null  # 128 luồng làm MinIO sập ở 100 upload
  ai_set $1 $2
  docker inspect haui-fitness-backend-1 haui-fitness-mongodb-1 haui-fitness-frontend-1 haui-fitness-kafka-1 haui-fitness-ai-worker-1 haui-fitness-minio-1 --format '  {{.Name}} cpus={{.HostConfig.NanoCpus}} mem={{.HostConfig.Memory}} pids={{.HostConfig.PidsLimit}}'
}
h1(){  # khôi phục cấu hình H1 (theo compose đã commit)
  log "switch -> H1"
  (cd $BEDIR && docker compose up -d --no-deps --force-recreate mongodb backend ai-worker frontend 2>&1 | tail -1)
  docker update --cpus 1.5 --pids-limit 256 haui-fitness-kafka-1 >/dev/null
  docker update --cpus 1 --memory 512m --memory-swap 512m --pids-limit 128 haui-fitness-minio-1 >/dev/null
  ai_set 3 opencv
}
gd4(){
  wait_idle
  for u in 100 500 1000 2000 3000; do jm v3_H2_fe_u$u fe_page.jmx -Jthreads=$u -Jramp=20 -Jduration=120; done
  for u in 100 500 1000 2000 5000 10000; do jm v3_H2_flow_u$u be_user_flow_tok.jmx -Jtokens=$TOK -Jthreads=$u -Jramp=30 -Jduration=120; done
  for u in 10 50 100 200 500; do jm v3_H2_login_u$u be_login.jmx -Jaccounts=$ACC -Jpassword=Loadtest@2026 -Jthreads=$u -Jramp=5 -Jduration=120; done
  for c in 6 12 24 36 48 72; do jm v3_H2_ai_short_c$c ai_direct.jmx -Jobject_key=$K1 -Jthreads=$c -Jramp=0 -Jloops=4 -Jduration=3600; done
  for c in 3 6 12; do jm v3_H2_ai_long_c$c ai_direct.jmx -Jobject_key=$K2 -Jthreads=$c -Jramp=0 -Jloops=1 -Jduration=3600; done
  for n in 10 50 100; do disk_ok; log "H2 upload x$n"; python3 scripts/upload_test.py $n v3_H2_upload_$n; done
  wait_idle; scn v3_H2_T3_5_real40 --duration 30 --ramp 1 --real-list $L/real40.txt --submit-window 60 --poll 5
  wait_idle; scn v3_H2_T3_2_exam_start --duration 300 --ramp 30 --browsers 200 --think-min 2 --think-max 5 --burst --login-burst 200 --login-window 30 --exam-submits 150 --submit-window 300 --short-ratio 0.7
  wait_idle; scn v3_H2_T3_4_poll_storm --duration 300 --ramp 20 --pollers 1000 --poll 2 --burst --exam-submits 300 --submit-window 60 --short-ratio 1.0
  wait_idle
}
gd4b(){  # chạy lại các bài backend của GD4 sau khi nâng nofile cho MongoDB (lần trước Mongo từ chối kết nối → 500)
  wait_idle
  for u in 100 500 1000 2000 5000 10000; do jm v3_H2b_flow_u$u be_user_flow_tok.jmx -Jtokens=$TOK -Jthreads=$u -Jramp=30 -Jduration=120; done
  wait_idle; scn v3_H2b_T3_2_exam_start --duration 300 --ramp 30 --browsers 200 --think-min 2 --think-max 5 --burst --login-burst 200 --login-window 30 --exam-submits 150 --submit-window 300 --short-ratio 0.7
  wait_idle; scn v3_H2b_T3_4_poll_storm --duration 300 --ramp 20 --pollers 1000 --poll 2 --burst --exam-submits 300 --submit-window 60 --short-ratio 1.0
  wait_idle
}
gd5(){
  export HEAP="-Xms2g -Xmx16g -Xss256k"
  for u in 10000 20000 30000; do jm v3_T5_1_flow_u$u be_user_flow_tok.jmx -Jtokens=$TOK -Jthreads=$u -Jramp=60 -Jduration=120; done
  for u in 500 1000 2000; do jm v3_T5_2_login_u$u be_login.jmx -Jaccounts=$ACC -Jpassword=Loadtest@2026 -Jthreads=$u -Jramp=10 -Jduration=120; done
  unset HEAP
  wait_idle; scn v3_T5_3_deep_queue --duration 30 --ramp 1 --burst --exam-submits 1000 --submit-window 120 --short-ratio 1.0 --poll 5
  wait_idle
  log "T5.5 combined max"
  (sleep 60; HEAP="-Xms2g -Xmx8g" bash scripts/run_jm.sh v3_T5_5_api5000 be_user_flow_tok.jmx $H -Jtokens=$TOK -Jthreads=5000 -Jramp=60 -Jduration=300 >/dev/null) &
  (sleep 60; bash scripts/run_jm.sh v3_T5_5_ai_flood ai_direct.jmx $H -Jobject_key=$K1 -Jthreads=24 -Jramp=0 -Jloops=10 -Jduration=3600 >/dev/null) &
  scn v3_T5_5_combined --duration 360 --ramp 30 --browsers 200 --pollers 1000 --poll 2 --think-min 2 --think-max 5 --burst --login-burst 200 --login-window 30 --exam-submits 150 --submit-window 300 --short-ratio 0.7
  wait; grep -hE "^TOTAL|^POST|errors|events" results/v3_T5_5_api5000/summary.txt results/v3_T5_5_ai_flood/summary.txt
  wait_idle
}
for g in "$@"; do log "===== $g"; case $g in
  gd1) gd1;; gd2) gd2;; t25) wait_idle; t25;; gd3) gd3;; gd4) gd4;; gd4b) gd4b;; gd3r) gd3r;; gdfix) gdfix;; t35) t35;; gd5) gd5;; h1) h1;;
  aiscale*) aiscale ${AISCALE_DECODER:-opencv} ${AISCALE_N:-3 4 5 6};;
  h2) h2 ${H2_AI:-4} ${H2_DECODER:-opencv};;
esac; done; log "DONE $*"
