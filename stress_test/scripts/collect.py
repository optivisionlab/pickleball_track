"""Gom summary.json + đỉnh tài nguyên của mọi test trong results/ vào all_results.json."""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "results"
out = {}
for d in sorted(glob.glob(f"{root}/*/")):
    name = os.path.basename(d.rstrip("/"))
    sj = os.path.join(d, "summary.json")
    if not os.path.exists(sj):
        continue
    entry = {"summary": json.load(open(sj))}
    peaks = defaultdict(lambda: {"cpu": 0.0, "mem_pct": 0.0, "mem": ""})
    try:
        for row in csv.reader(open(os.path.join(d, "mon", "docker.csv"))):
            _, cname, cpu, mem, memp = row[:5]
            p = peaks[cname]
            c, m = float(cpu.strip("%") or 0), float(memp.strip("%") or 0)
            p["cpu"] = max(p["cpu"], c)
            if m >= p["mem_pct"]:
                p["mem_pct"], p["mem"] = m, mem.split("/")[0].strip()
    except FileNotFoundError:
        pass
    entry["peaks"] = dict(peaks)
    try:
        g = [list(map(float, r[1:3])) for r in csv.reader(open(os.path.join(d, "mon", "gpu.csv")))]
        entry["gpu"] = {"util_max": max(x[0] for x in g), "util_avg": round(sum(x[0] for x in g) / len(g), 1),
                        "mem_max": max(x[1] for x in g)}
    except Exception:
        pass
    ev = os.path.join(d, "mon", "events.log")
    entry["events"] = open(ev).read().strip().splitlines() if os.path.exists(ev) else []
    pt = os.path.join(d, "params.txt")
    entry["params"] = open(pt).read().strip() if os.path.exists(pt) else ""
    out[name] = entry
for f in sorted(glob.glob(f"{root}/*.summary.json")):
    out[os.path.basename(f).replace(".summary.json", "")] = {"e2e": json.load(open(f))}
json.dump(out, open(f"{root}/all_results.json", "w"), ensure_ascii=False, indent=1)
print(len(out), "tests collected")
