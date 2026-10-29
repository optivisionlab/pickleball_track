"""In đỉnh tài nguyên từ thư mục mon/ của monitor.sh."""
import csv, sys, collections
d = sys.argv[1]
mx = collections.defaultdict(lambda: [0.0, 0.0])
try:
    for r in csv.reader(open(f"{d}/docker.csv")):
        m = mx[r[1]]; m[0] = max(m[0], float(r[2].strip('%') or 0)); m[1] = max(m[1], float(r[4].strip('%') or 0))
except FileNotFoundError:
    pass
for n, (c, m) in sorted(mx.items()):
    if any(k in n for k in ("backend", "worker", "mongodb", "pickleball-ai", "frontend", "kafka", "minio")):
        print(f"peak {n:34} cpu={c:7.1f}%  mem={m:5.1f}%")
def rows(name):
    try:
        return list(csv.reader(open(f"{d}/{name}")))
    except FileNotFoundError:
        return []
g = [list(map(float, r[1:5])) for r in rows("gpu.csv") if len(r) >= 5]
if g:
    print(f"GPU util max={max(x[0] for x in g):.0f}% avg={sum(x[0] for x in g)/len(g):.0f}%  VRAM max={max(x[1] for x in g):.0f}MiB"
          f"  temp max={max(x[2] for x in g):.0f}C  power max={max(x[3] for x in g):.0f}W avg={sum(x[3] for x in g)/len(g):.0f}W")
h = [list(map(float, r[1:8])) for r in rows("host.csv") if len(r) >= 8]
if h:
    print(f"HOST cpu max={max(x[0] for x in h):.0f}% avg={sum(x[0] for x in h)/len(h):.0f}%  RAM max={max(x[1] for x in h):.1f}GB"
          f"  load1 max={max(x[2] for x in h):.0f}  disk write max={max(x[4] for x in h):.0f}MB/2s  net rx max={max(x[5] for x in h):.0f}MB/2s")
ai = [r for r in rows("ai.csv") if r[2] != "ERR"]
if ai:
    by_ts = collections.defaultdict(lambda: [0, 0])
    for r in ai:
        by_ts[r[0]][0] += int(r[2]); by_ts[r[0]][1] += int(r[3])
    print(f"AI busy slots max={max(v[0] for v in by_ts.values())}  pending jobs max={max(v[1] for v in by_ts.values())}"
          f"  (ERR samples={sum(1 for r in rows('ai.csv') if r[2]=='ERR')})")
k = rows("kafka.csv")
if k:
    lag = collections.defaultdict(int)
    for r in k:
        lag[r[1] + "/" + r[2]] = max(lag[r[1] + "/" + r[2]], int(r[3]))
    print("kafka max lag:", {k2: v for k2, v in lag.items() if v})
m = rows("mongo.csv")
if m:
    print(f"mongo pids max={max(int(r[1]) for r in m)}")
try:
    ev = open(f"{d}/events.log").read().strip()
except FileNotFoundError:
    ev = ""
print("container events:", ev.replace("\n", " | ") or "none")
# Tiến trình GPU không thuộc AI service (/opt/venv/bin/python) → kết quả có thể bị nhiễu
gp = [r for r in rows("gpu_procs.csv") if len(r) >= 4 and "/opt/venv/bin/python" not in r[3]]
if gp:
    foreign = {}
    for r in gp:
        foreign[r[1].strip()] = (r[3].strip(), max(int(r[2] or 0), foreign.get(r[1].strip(), ("", 0))[1]))
    print("WARN foreign GPU processes:", {p: f"{n} {m}MiB" for p, (n, m) in foreign.items()})
else:
    print("GPU processes: AI service only")
