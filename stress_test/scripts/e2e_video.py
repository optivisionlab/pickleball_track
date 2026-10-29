"""Test luồng nộp video đầu-cuối (E2E) với N sinh viên nộp đồng thời.

Mỗi lượt nộp: login (1 lần / tài khoản) -> POST upload-url -> PUT video lên MinIO
-> POST confirm-upload -> poll GET /status cho tới khi AI chấm xong.

Ghi ra CSV từng lượt: thời gian từng bước, thời gian chờ AI, tổng E2E, điểm, lỗi.

Ví dụ:
  python3 e2e_video.py --video shot1.mp4 --task 6ae0...e9 --concurrency 4 --total 8 --out r.csv
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import time

import httpx

STUDENTS = [
    "annv@sv.haui.edu.vn", "binhtt@sv.haui.edu.vn", "cuonglh@sv.haui.edu.vn",
    "ducpm@sv.haui.edu.vn", "hoaht@sv.haui.edu.vn", "huynq@sv.haui.edu.vn",
]
PASSWORD = os.environ["HAUI_STUDENT_PASSWORD"]


def probe_duration(path: str) -> int:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]
        )
        return int(float(out.strip()))
    except Exception:
        return 0


def data(resp: httpx.Response):
    body = resp.json()
    return body.get("data", body) if isinstance(body, dict) else body


async def login(client: httpx.AsyncClient, base: str, email: str) -> str:
    r = await client.post(f"{base}/api/auth/login", json={"email": email, "password": PASSWORD})
    r.raise_for_status()
    return data(r)["accessToken"]


async def one_submission(idx, client, base, token, video_bytes, args, duration, sem, rows, t_start):
    async with sem:
        row = {"idx": idx, "start_offset_s": round(time.perf_counter() - t_start, 3)}
        h = {"Authorization": f"Bearer {token}"}
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{base}/api/video-results/upload-url", headers=h, json={
                "task_id": args.task, "file_name": os.path.basename(args.video),
                "content_type": "video/mp4" if args.video.lower().endswith(".mp4") else "video/quicktime",
                "file_size": len(video_bytes), "duration_sec": duration,
            })
            t1 = time.perf_counter()
            row["upload_url_ms"] = round((t1 - t0) * 1000, 1)
            if r.status_code != 200:
                row["error"] = f"upload-url {r.status_code} {r.text[:120]}"
                rows.append(row)
                return
            up = data(r)
            r = await client.put(up["uploadUrl"], content=video_bytes,
                                 headers={"Content-Type": "application/octet-stream"}, timeout=600)
            t2 = time.perf_counter()
            row["put_minio_ms"] = round((t2 - t1) * 1000, 1)
            if r.status_code not in (200, 204):
                row["error"] = f"put {r.status_code}"
                rows.append(row)
                return
            r = await client.post(f"{base}/api/video-results/confirm-upload", headers=h,
                                  json={"upload_id": up["uploadId"]})
            t3 = time.perf_counter()
            row["confirm_ms"] = round((t3 - t2) * 1000, 1)
            if r.status_code not in (200, 201):
                row["error"] = f"confirm {r.status_code} {r.text[:120]}"
                rows.append(row)
                return
            result_id = data(r)["id"]
            row["result_id"] = result_id
            row["submit_total_ms"] = round((t3 - t0) * 1000, 1)
            # Poll trạng thái cho tới khi chấm xong
            seen_processing = None
            status = None
            while time.perf_counter() - t3 < args.max_wait:
                await asyncio.sleep(args.poll)
                tp = time.perf_counter()
                s = await client.get(f"{base}/api/video-results/{result_id}/status", headers=h)
                row.setdefault("poll_ms", []).append(round((time.perf_counter() - tp) * 1000, 1))
                if s.status_code != 200:
                    continue
                st = data(s)
                status = st.get("status")
                if status == "processing" and seen_processing is None:
                    seen_processing = time.perf_counter()
                if not st.get("pending", True) or status in ("completed", "graded", "failed", "error"):
                    row["ai_score"] = st.get("aiScore")
                    row["grading_error"] = st.get("error")
                    break
            t4 = time.perf_counter()
            row["final_status"] = status
            row["grading_wait_ms"] = round((t4 - t3) * 1000, 1)
            if seen_processing:
                row["queue_wait_ms"] = round((seen_processing - t3) * 1000, 1)
            row["e2e_ms"] = round((t4 - t0) * 1000, 1)
            if t4 - t3 >= args.max_wait:
                row["error"] = "grading_timeout"
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"exception {type(exc).__name__}: {exc}"[:200]
        polls = row.pop("poll_ms", [])
        if polls:
            row["poll_avg_ms"] = round(statistics.mean(polls), 1)
            row["poll_count"] = len(polls)
        rows.append(row)


def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return round(values[f] + (values[c] - values[f]) * (k - f), 1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://100.95.228.82:8000")
    ap.add_argument("--video", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--total", type=int, default=1)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--max-wait", type=float, default=1800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    video_bytes = open(args.video, "rb").read()
    duration = probe_duration(args.video)
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        tokens = [await login(client, args.base, e) for e in STUDENTS]
        sem = asyncio.Semaphore(args.concurrency)
        rows: list[dict] = []
        t_start = time.perf_counter()
        await asyncio.gather(*[
            one_submission(i, client, args.base, tokens[i % len(tokens)], video_bytes, args,
                           duration, sem, rows, t_start)
            for i in range(args.total)
        ])
        wall = time.perf_counter() - t_start

    rows.sort(key=lambda r: r["idx"])
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if not r.get("error") and r.get("final_status") in ("completed", "graded")]
    summary = {"label": args.label, "video": os.path.basename(args.video), "size_mb": round(len(video_bytes) / 1e6, 1),
               "duration_s": duration, "concurrency": args.concurrency, "total": args.total,
               "success": len(ok), "errors": len(rows) - len(ok), "wall_s": round(wall, 1),
               "throughput_videos_per_min": round(len(ok) / wall * 60, 2) if wall else None}
    for key in ("upload_url_ms", "put_minio_ms", "confirm_ms", "submit_total_ms", "queue_wait_ms",
                "grading_wait_ms", "e2e_ms", "poll_avg_ms"):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if vals:
            summary[key] = {"avg": round(statistics.mean(vals), 1), "min": min(vals), "p50": pct(vals, 50),
                            "p90": pct(vals, 90), "p95": pct(vals, 95), "max": max(vals)}
    summary["error_samples"] = list({r["error"] for r in rows if r.get("error")})[:5]
    summary["scores"] = [r.get("ai_score") for r in ok][:20]
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    with open(args.out.replace(".csv", ".summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
