"""1 dòng số liệu host (CPU, RAM, load, I/O đĩa, mạng) tính theo chênh lệch so với lần đo trước."""
import json, re, sys, time
state_path, ts = sys.argv[1], sys.argv[2]
cpu = list(map(int, open("/proc/stat").readline().split()[1:]))
idle, total = cpu[3] + cpu[4], sum(cpu)
mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
used_gb = (mem["MemTotal"] - mem["MemAvailable"]) / 1048576
load1 = float(open("/proc/loadavg").read().split()[0])
rd = wr = 0
for l in open("/proc/diskstats"):
    f = l.split()
    if re.fullmatch(r"sd[a-z]+|vd[a-z]+|nvme\d+n\d+", f[2]):  # chỉ ổ vật lý, không tính phân vùng
        rd += int(f[5]); wr += int(f[9])
rx = tx = 0
for l in open("/proc/net/dev").readlines()[2:]:
    name, rest = l.split(":", 1)
    if name.strip() != "lo" and not name.strip().startswith(("veth", "br-", "docker")):
        v = rest.split(); rx += int(v[0]); tx += int(v[8])
now = {"idle": idle, "total": total, "rd": rd, "wr": wr, "rx": rx, "tx": tx, "t": time.time()}
try:
    prev = json.load(open(state_path))
except Exception:
    prev = None
json.dump(now, open(state_path, "w"))
if prev:
    dt_total = max(1, now["total"] - prev["total"])
    busy = 100 * (1 - (now["idle"] - prev["idle"]) / dt_total)
    mb = lambda k, unit: (now[k] - prev[k]) * unit / 1e6
    print(f"{ts},{busy:.1f},{used_gb:.1f},{load1},{mb('rd',512):.1f},{mb('wr',512):.1f},{mb('rx',1):.1f},{mb('tx',1):.1f}")
