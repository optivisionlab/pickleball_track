"""Tổng hợp file .jtl (CSV) của JMeter thành bảng thống kê theo từng label + tổng.

python3 summarize_jtl.py result.jtl [--json out.json] [--exclude-sub]
"""
import csv
import json
import statistics
import sys
from collections import defaultdict


def pct(v, p):
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 1)


def stats(rows):
    el = sorted(int(r["elapsed"]) for r in rows)
    ts0 = min(int(r["timeStamp"]) for r in rows)
    ts1 = max(int(r["timeStamp"]) + int(r["elapsed"]) for r in rows)
    span = max((ts1 - ts0) / 1000, 0.001)
    errs = [r for r in rows if r["success"] != "true"]
    codes = defaultdict(int)
    for r in errs:
        codes[r["responseCode"][:40]] += 1
    return {
        "samples": len(rows), "errors": len(errs), "error_pct": round(len(errs) * 100 / len(rows), 2),
        "avg_ms": round(statistics.mean(el), 1), "min_ms": el[0], "p50_ms": pct(el, 50), "p90_ms": pct(el, 90),
        "p95_ms": pct(el, 95), "p99_ms": pct(el, 99), "max_ms": el[-1],
        "throughput_rps": round(len(rows) / span, 2),
        "kb_per_sec": round(sum(int(r["bytes"] or 0) for r in rows) / 1024 / span, 1),
        "max_threads": max(int(r["allThreads"] or 0) for r in rows),
        "duration_s": round(span, 1), "error_codes": dict(codes),
    }


def main():
    path = sys.argv[1]
    exclude_sub = "--exclude-sub" in sys.argv
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no samples")
        return
    by = defaultdict(list)
    for r in rows:
        label = r["label"]
        # Bỏ sub-sample (label-0, label-1...) của embedded resources
        if exclude_sub and label.rsplit("-", 1)[-1].isdigit():
            continue
        by[label].append(r)
    # "TOTAL" chỉ tính request HTTP thật, không tính TransactionController (responseMessage chứa "Number of samples")
    http_rows = [r for lab, rs in by.items() for r in rs if not r["responseMessage"].startswith("Number of samples")]
    out = {lab: stats(rs) for lab, rs in sorted(by.items())}
    out["TOTAL"] = stats(http_rows)
    hdr = f"{'label':42} {'n':>7} {'err%':>6} {'avg':>8} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>8} {'max':>7} {'rps':>8}"
    print(hdr)
    for lab, s in out.items():
        print(f"{lab[:42]:42} {s['samples']:>7} {s['error_pct']:>6} {s['avg_ms']:>8} {s['p50_ms']:>7} {s['p90_ms']:>7} "
              f"{s['p95_ms']:>7} {s['p99_ms']:>8} {s['max_ms']:>7} {s['throughput_rps']:>8}")
        if s["error_codes"]:
            print(f"{'':42}   errors: {s['error_codes']}")
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
