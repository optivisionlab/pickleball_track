#!/usr/bin/env bash
# Chạy lại bộ test sau tối ưu (kết quả: results/post_*)
cd ~/loadtest; T=$PWD/jmeter/tokens.csv; H="-Jhost=127.0.0.1"
run(){ echo "######## $1 | GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; bash scripts/run_jm.sh "$@" $H | grep -E "^(TOTAL|POST)|errors|backend|mongodb|events" ; }
for u in 1 5 10 20 50 100; do run post_login_u$u be_login.jmx -Jthreads=$u -Jramp=2 -Jduration=60; done
for u in 10 50 100 200 500 1000; do run post_flow_u$u be_user_flow_tok.jmx -Jtokens=$T -Jthreads=$u -Jramp=20 -Jduration=60; done
run post_spike_1000 be_user_flow_tok.jmx -Jtokens=$T -Jthreads=1000 -Jramp=5 -Jduration=60
for u in 2000 5000; do run post_stress_u$u be_user_flow_tok.jmx -Jtokens=$T -Jthreads=$u -Jramp=30 -Jduration=60; done
t0=$(date +%s); until curl -sf -m 2 http://127.0.0.1:8000/health/ready >/dev/null; do sleep 1; done; echo "healthy after $(( $(date +%s)-t0 ))s"
