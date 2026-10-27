#!/usr/bin/env bash
# Chạy loạt test tải tăng dần số user cho FE và BE (chạy trên server, host=127.0.0.1).
#   run_series.sh <suite> [host]
# suite: fe | be_root | be_login | be_flow | stress | spike | soak
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
SUITE=$1; HOST=${2:-127.0.0.1}; DUR=${DUR:-60}
run() { echo "######## $1"; bash "$HERE/scripts/run_jm.sh" "$@" -Jhost="$HOST" | grep -vE "^peak haui-fitness-(redis|kafka|minio)"; }
case $SUITE in
  fe)       for u in 10 50 100 200 500 1000; do run fe_u$u fe_page.jmx -Jthreads=$u -Jramp=10 -Jduration=$DUR; done ;;
  be_root)  for u in 10 50 100 200 500; do run be_root_u$u be_simple.jmx -Jpath=/ -Jthreads=$u -Jramp=5 -Jduration=$DUR; done ;;
  be_login) for u in 1 5 10 20 50; do run be_login_u$u be_login.jmx -Jthreads=$u -Jramp=2 -Jduration=$DUR; done ;;
  be_flow)  for u in 10 50 100 200 500 1000; do run be_flow_u$u be_user_flow_tok.jmx -Jtokens=$HERE/jmeter/tokens.csv -Jthreads=$u -Jramp=20 -Jduration=$DUR; done ;;
  # Stress đến khi hỏng: tăng user tới 5000, think 0
  stress)   for u in 2000 3000 5000; do run be_stress_u$u be_user_flow_tok.jmx -Jtokens=$HERE/jmeter/tokens.csv -Jthreads=$u -Jramp=30 -Jduration=$DUR; done ;;
  # Spike: 1000 user ập vào trong 5 giây
  spike)    run be_spike_1000 be_user_flow_tok.jmx -Jtokens=$HERE/jmeter/tokens.csv -Jthreads=1000 -Jramp=5 -Jduration=$DUR ;;
  # Soak: 100 user, think 1s, 10 phút
  soak)     run be_soak_100 be_user_flow_tok.jmx -Jtokens=$HERE/jmeter/tokens.csv -Jthreads=100 -Jramp=20 -Jthink=1000 -Jduration=${SOAK_DUR:-600} ;;
esac
