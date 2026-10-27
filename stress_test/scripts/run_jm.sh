#!/usr/bin/env bash
# Chạy 1 test JMeter + giám sát tài nguyên.
#   run_jm.sh <tên_kết_quả> <plan.jmx> [các -J tham số...]
# Ví dụ: run_jm.sh be_flow_100 be_user_flow.jmx -Jthreads=100 -Jramp=10 -Jduration=60
set -u
ulimit -n "$(ulimit -Hn)" 2>/dev/null
HERE=$(cd "$(dirname "$0")/.." && pwd)
JM=${JMETER:-$HERE/apache-jmeter-5.6.3/bin/jmeter}
NAME=$1; PLAN=$2; shift 2
R=$HERE/results/$NAME
rm -rf "$R"; mkdir -p "$R"
bash "$HERE/scripts/monitor.sh" "$R/mon" 2 & MON=$!
cd "$HERE/jmeter"
HEAP=${HEAP:-"-Xms1g -Xmx4g"} JVM_ARGS="${JVM_ARGS:-}" \
  "$JM" -n -t "$PLAN" -Jaccounts="$HERE/jmeter/accounts.csv" "$@" \
  -Jjmeter.save.saveservice.output_format=csv -l "$R/result.jtl" -j "$R/jmeter.log" -e -o "$R/html" > "$R/stdout.log" 2>&1
kill $MON 2>/dev/null; wait $MON 2>/dev/null
echo "$*" > "$R/params.txt"
python3 "$HERE/scripts/summarize_jtl.py" "$R/result.jtl" --exclude-sub --json "$R/summary.json" | tee "$R/summary.txt"
# Đỉnh tài nguyên trong lúc test
python3 - "$R/mon" <<'EOF' | tee -a "$R/summary.txt"
import sys, csv, collections, re
d = sys.argv[1]
mx = collections.defaultdict(lambda: [0.0, 0.0])
try:
    for row in csv.reader(open(f"{d}/docker.csv")):
        ts, name, cpu, mem, memp = row[:5]
        m = mx[name]; m[0] = max(m[0], float(cpu.strip('%') or 0)); m[1] = max(m[1], float(memp.strip('%') or 0))
except FileNotFoundError:
    pass
for n, (c, m) in sorted(mx.items()):
    print(f"peak {n:32} cpu={c:7.1f}%  mem={m:5.1f}% of limit")
try:
    g = [list(map(float, r[1:3])) for r in csv.reader(open(f"{d}/gpu.csv"))]
    print(f"peak GPU util={max(x[0] for x in g):.0f}%  mem={max(x[1] for x in g):.0f}MiB")
except Exception:
    pass
ev = open(f"{d}/events.log").read().strip() if True else ""
print("container events:", ev or "none")
EOF
