#!/usr/bin/env bash
# Benchmark AI qua cổng 8888 với 3 tải chuẩn. bench_ai.sh <nhãn>
L=$1; cd ~/loadtest
K1=loadtest/shot1_b.mp4
K2=loadtest/shot2.mov
J="-Jhost=127.0.0.1 -Jramp=0 -Jduration=3600"
s(){ python3 - "$1" "$2" <<'PY'
import csv,sys,statistics
d,lab=sys.argv[1],sys.argv[2]; r=list(csv.DictReader(open(d+"/result.jtl")))
ok=[x for x in r if x["success"]=="true"]; rej=sum(1 for x in r if x["responseCode"]=="429"); err=len(r)-len(ok)-rej
t0=min(int(x["timeStamp"]) for x in r); t1=max(int(x["timeStamp"])+int(x["elapsed"]) for x in r)
el=sorted(int(x["elapsed"])/1000 for x in ok)
p=lambda q: el[min(len(el)-1,int(round(q*(len(el)-1))))] if el else 0
print(f"{lab:28} sent={len(r):3} ok={len(ok):3} 429={rej:2} err={err:2} avg={statistics.mean(el) if el else 0:7.2f}s p50={p(.5):7.2f} p95={p(.95):7.2f} max={el[-1] if el else 0:7.2f} wall={(t1-t0)/1000:6.1f}s videos/min={len(ok)/((t1-t0)/60000):6.1f}")
PY
}
HEAP="-Xms256m -Xmx1g" bash scripts/run_jm.sh b_${L}_w1 ai_direct.jmx $J -Jobject_key=$K1 -Jthreads=12 -Jloops=4 >/dev/null; s results/b_${L}_w1 "$L W1 short x48 c12"
HEAP="-Xms256m -Xmx1g" bash scripts/run_jm.sh b_${L}_w2 ai_direct.jmx $J -Jobject_key=$K2 -Jthreads=4 -Jloops=1 >/dev/null; s results/b_${L}_w2 "$L W2 long x4 c4"
HEAP="-Xms256m -Xmx1g" bash scripts/run_jm.sh b_${L}_w3l ai_direct.jmx $J -Jobject_key=$K2 -Jthreads=3 -Jloops=1 >/dev/null &
sleep 3
HEAP="-Xms256m -Xmx1g" bash scripts/run_jm.sh b_${L}_w3s ai_direct.jmx $J -Jobject_key=$K1 -Jthreads=4 -Jloops=6 >/dev/null
wait
s results/b_${L}_w3l "$L W3 mixed: long x3"; s results/b_${L}_w3s "$L W3 mixed: short x24 c4"
grep -h "peak GPU\|peak pickleball" results/b_${L}_w*/summary.txt | sort | uniq | head -8
