#!/usr/bin/env bash
# E2E + giờ cao điểm sau tối ưu (kết quả: results/e2e/post_*)
cd ~/loadtest; T1=6ae093a7bf50c057267d76e9; T2=6ae093a7bf50c057267d76ea; mkdir -p results/e2e
sum(){ python3 -c "
import json;s=json.load(open('results/e2e/$1.summary.json'));g=lambda k:s.get(k,{})
print('$1','ok',s['success'],'/',s['total'],'wall',s['wall_s'],'s | vid/min',s['throughput_videos_per_min'],'| submit avg',g('submit_total_ms').get('avg'),'p95',g('submit_total_ms').get('p95'),'| grading avg',g('grading_wait_ms').get('avg'),'max',g('grading_wait_ms').get('max'),'| e2e avg',g('e2e_ms').get('avg'),'p95',g('e2e_ms').get('p95'),'max',g('e2e_ms').get('max'),s['error_samples'])"; }
run(){ n=$1; shift; bash scripts/monitor.sh results/e2e/mon_$n 2 & M=$!; python3 scripts/e2e_video.py --base http://127.0.0.1:8000 --poll 1 --out results/e2e/$n.csv --label $n "$@" >/dev/null 2>results/e2e/$n.err; kill $M; sum $n; echo "   GPU max: $(cut -d, -f2,3 results/e2e/mon_$n/gpu.csv | sort -n | tail -1)"; }
run post_shot1_c1 --video videos/shot1_b.mp4 --task $T1 --concurrency 1 --total 5
run post_shot1_c5 --video videos/shot1_b.mp4 --task $T1 --concurrency 5 --total 10
run post_shot1_c10 --video videos/shot1_b.mp4 --task $T1 --concurrency 10 --total 20
run post_shot1_c20 --video videos/shot1_b.mp4 --task $T1 --concurrency 20 --total 20
run post_shot2_c3 --video videos/shot2.mov --task $T2 --concurrency 3 --total 3
run post_shot2_c6 --video videos/shot2.mov --task $T2 --concurrency 6 --total 6
echo "######## mixed peak"
(SOAK_DUR=300 bash scripts/run_series.sh soak > results/post_series_soak.log 2>&1; mv results/be_soak_100 results/post_soak_100) & S=$!
sleep 30
python3 scripts/e2e_video.py --base http://127.0.0.1:8000 --poll 1 --video videos/shot2.mov --task $T2 --concurrency 2 --total 2 --out results/e2e/post_mixed_shot2.csv >/dev/null 2>&1 &
python3 scripts/e2e_video.py --base http://127.0.0.1:8000 --poll 1 --video videos/shot1_b.mp4 --task $T1 --concurrency 4 --total 40 --out results/e2e/post_mixed_shot1.csv >/dev/null 2>&1
wait
grep -E "^(GET|TOTAL)|errors|backend|mongodb|GPU|events" results/post_soak_100/summary.txt
sum post_mixed_shot1; sum post_mixed_shot2
