"""Mô phỏng người dùng thật cho load test V2 (asyncio + httpx).

Vai (persona):
  browser   SV duyệt web: lặp 1 thao tác ngẫu nhiên rồi nghỉ think-min..think-max giây
  submitter SV nộp bài: upload-url -> PUT MinIO -> confirm -> hỏi /status mỗi poll giây tới khi chấm xong
  teacher   GV: sổ điểm, thống kê, danh sách bài nộp, bài tập của lớp
  poller    SV mở trang chờ kết quả: hỏi /status của 1 bài nộp mỗi poll giây

Ví dụ (B1 ngày học bình thường):
  python3 scenario.py --name B1 --duration 1200 --browsers 300 --teachers 5 \
      --submit-every 20 --short-ratio 0.8
Kết quả: results/v2/<name>/{requests.csv, submissions.csv, summary.json}
"""
import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent.parent
COURSE = "6ae093a7bf50c057267d76e7"
TASKS = {"tx1": ("6ae093a7bf50c057267d76e9", 60), "tx2": ("6ae093a7bf50c057267d76ea", 90),
         "midterm": ("6ae093a7bf50c057267d76eb", 120), "final": ("6ae093a7bf50c057267d76ec", 180)}
SHORT = HERE / "videos" / "shot1_b.mp4"
LONG = HERE / "videos" / "shot2.mov"
TEACHER = ("longdv@haui.edu.vn", os.environ.get("HAUI_TEACHER_PASSWORD", ""))
PASSWORD = "Loadtest@2026"


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    k = (len(v) - 1) * p / 100
    f = int(k)
    return round(v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f), 1)


def data(r):
    try:
        body = r.json()
    except Exception:
        return {}
    return body.get("data", body) if isinstance(body, dict) else body


class Run:
    def __init__(self, a):
        self.a = a
        self.t0 = time.monotonic()
        self.stop_at = self.t0 + a.duration
        self.req: list[tuple] = []          # (t, label, status, ms)
        self.subs: list[dict] = []
        self.result_ids: list[str] = []
        self.own: dict[int, str] = {}  # chỉ số tài khoản -> bài nộp gần nhất của tài khoản đó
        self.accounts = [line.split(",", 1) for line in (HERE / "jmeter" / "tokens_v2.csv").read_text().split()]
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=20),
            limits=httpx.Limits(max_connections=a.max_conn, max_keepalive_connections=a.max_conn),
        )

    def running(self):
        return time.monotonic() < self.stop_at

    async def call(self, label, method, url, **kw):
        t = time.monotonic()
        try:
            r = await self.client.request(method, url, **kw)
            code = r.status_code
        except Exception as exc:
            r, code = None, f"EXC:{type(exc).__name__}"
        self.req.append((round(t - self.t0, 2), label, code, round((time.monotonic() - t) * 1000, 1)))
        return r

    # ---------------------------------------------------------------- personas
    async def browser(self, i):
        email, token = self.accounts[i % len(self.accounts)]
        h = {"Authorization": f"Bearer {token}"}
        b = self.a.base
        actions = [
            ("GET /auth/me", f"{b}/api/auth/me"),
            ("GET /enrollments/my-courses", f"{b}/api/enrollments/my-courses"),
            ("GET /tasks?course", f"{b}/api/tasks?course_id={COURSE}&page_size=20"),
            ("GET /video-results", f"{b}/api/video-results?page_size=10"),
            ("GET /student/grades", f"{b}/api/student/courses/{COURSE}/grades"),
            ("GET /notifications", f"{b}/api/notifications"),
            ("GET /sports", f"{b}/api/sports"),
        ]
        await asyncio.sleep(random.uniform(0, self.a.ramp))
        while self.running():
            label, url = random.choice(actions)
            await self.call(label, "GET", url, headers=h)
            await asyncio.sleep(random.uniform(self.a.think_min, self.a.think_max))

    async def teacher(self, i):
        r = await self.call("POST /auth/login (teacher)", "POST", f"{self.a.base}/api/auth/login",
                            json={"email": TEACHER[0], "password": TEACHER[1]})
        if r is None or r.status_code != 200:
            return
        h = {"Authorization": f"Bearer {data(r)['accessToken']}"}
        b = self.a.base
        actions = [
            ("GET /teacher/gradebook", f"{b}/api/teacher/courses/{COURSE}/gradebook"),
            ("GET /teacher/stats", f"{b}/api/teacher/courses/{COURSE}/stats"),
            ("GET /teacher/course tasks", f"{b}/api/teacher/courses/{COURSE}/tasks"),
            ("GET /tasks/{id}/submissions", f"{b}/api/tasks/{TASKS['tx1'][0]}/submissions"),
        ]
        await asyncio.sleep(random.uniform(0, self.a.ramp))
        while self.running():
            label, url = random.choice(actions)
            await self.call(label, "GET", url, headers=h)
            await asyncio.sleep(random.uniform(self.a.think_min, self.a.think_max))

    async def poller(self, i):
        email, token = self.accounts[i % len(self.accounts)]
        h = {"Authorization": f"Bearer {token}"}
        await asyncio.sleep(random.uniform(0, self.a.ramp))
        while self.running():
            rid = self.own.get(i % len(self.accounts))
            if rid:  # SV chỉ hỏi trạng thái bài nộp của chính mình
                await self.call("GET /status (poll)", "GET", f"{self.a.base}/api/video-results/{rid}/status", headers=h)
            await asyncio.sleep(self.a.poll)

    async def login_user(self, i):
        email, _ = self.accounts[i % len(self.accounts)]
        r = await self.call("POST /auth/login", "POST", f"{self.a.base}/api/auth/login",
                            json={"email": email, "password": PASSWORD})
        return data(r).get("accessToken") if r is not None and r.status_code == 200 else None

    async def submit(self, i, video: Path, task_key: str, token: str | None = None, duration: int | None = None,
                     tag: str = ""):
        email, saved = self.accounts[i % len(self.accounts)]
        token = token or saved
        h = {"Authorization": f"Bearer {token}"}
        b = self.a.base
        task_id, _ = TASKS[task_key]
        size = video.stat().st_size
        row = {"i": i, "video": video.name, "task": task_key, "size_mb": round(size / 1e6, 1), "tag": tag,
               "start_s": round(time.monotonic() - self.t0, 2)}
        t0 = time.monotonic()
        r = await self.call("POST /upload-url", "POST", f"{b}/api/video-results/upload-url", headers=h, json={
            "task_id": task_id, "file_name": video.name,
            "content_type": "video/mp4" if video.suffix.lower() == ".mp4" else "video/quicktime",
            "file_size": size, "duration_sec": duration})
        if r is None or r.status_code != 200:
            row["error"] = f"upload-url {getattr(r, 'status_code', 'EXC')}"
            self.subs.append(row)
            return row
        up = data(r)
        t1 = time.monotonic()
        with open(video, "rb") as f:
            r = await self.call("PUT minio", "PUT", up["uploadUrl"], content=f.read(),
                                headers={"Content-Type": "application/octet-stream"}, timeout=900)
        t2 = time.monotonic()
        if r is None or r.status_code not in (200, 204):
            row["error"] = f"put {getattr(r, 'status_code', 'EXC')}"
            self.subs.append(row)
            return row
        r = await self.call("POST /confirm-upload", "POST", f"{b}/api/video-results/confirm-upload", headers=h,
                            json={"upload_id": up["uploadId"]})
        t3 = time.monotonic()
        if r is None or r.status_code not in (200, 201):
            row["error"] = f"confirm {getattr(r, 'status_code', 'EXC')}"
            self.subs.append(row)
            return row
        rid = data(r)["id"]
        self.result_ids.append(rid)
        self.own[i % len(self.accounts)] = rid
        row.update(result_id=rid, upload_url_ms=round((t1 - t0) * 1000), put_ms=round((t2 - t1) * 1000),
                   confirm_ms=round((t3 - t2) * 1000), submit_ms=round((t3 - t0) * 1000))
        deadline = t3 + self.a.max_wait
        status = None
        while time.monotonic() < deadline:
            await asyncio.sleep(self.a.poll)
            s = await self.call("GET /status (own)", "GET", f"{b}/api/video-results/{rid}/status", headers=h)
            if s is None or s.status_code != 200:
                continue
            st = data(s)
            status = st.get("status")
            if not st.get("pending", True) or status in ("completed", "graded", "failed", "error"):
                row["ai_score"] = st.get("aiScore")
                row["grading_error"] = st.get("error")
                break
        t4 = time.monotonic()
        row.update(final_status=status, grading_ms=round((t4 - t3) * 1000), e2e_ms=round((t4 - t0) * 1000),
                   done_s=round(t4 - self.t0, 2))
        if status not in ("completed", "graded"):
            row["error"] = row.get("grading_error") or "not_graded"
        self.subs.append(row)
        return row

    async def submit_stream(self):
        """Mỗi submit-every giây (phân phối mũ) 1 SV nộp bài; short-ratio là tỷ lệ video ngắn."""
        tasks, i = [], 0
        while self.running():
            await asyncio.sleep(random.expovariate(1 / self.a.submit_every))
            short = random.random() < self.a.short_ratio
            tasks.append(asyncio.create_task(self.submit(
                i, SHORT if short else LONG, "tx1" if short else "tx2", duration=3 if short else 60)))
            if self.a.double_click and random.random() < self.a.double_click:
                tasks.append(asyncio.create_task(self.submit(i, SHORT, "tx1", duration=3, tag="double")))
            i += 1
        return tasks

    async def burst(self):
        """Đầu giờ thi: login-burst SV đăng nhập trong login-window giây, rồi exam-submits bài nộp trong submit-window giây."""
        n = self.a.login_burst
        tokens = await asyncio.gather(*(self._delayed(random.uniform(0, self.a.login_window), self.login_user(i))
                                        for i in range(n)))
        jobs = []
        for k in range(self.a.exam_submits):
            short = random.random() < self.a.short_ratio
            delay = random.uniform(0, self.a.submit_window)
            jobs.append(self._delayed(delay, self.submit(
                k, SHORT if short else LONG, "midterm", token=tokens[k % n] if n else None,
                duration=3 if short else 60)))
        return [asyncio.create_task(j) for j in jobs]

    async def real_videos(self):
        """Nộp danh sách video thật (file real-list: duration size path) vào bài thi cuối kỳ."""
        jobs = []
        for k, line in enumerate(Path(self.a.real_list).read_text().split("\n")):
            if not line.strip():
                continue
            dur, _, path = line.split(" ", 2)
            jobs.append(self._delayed(random.uniform(0, self.a.submit_window), self.submit(
                k, Path(path), "final", duration=int(float(dur)), tag="real")))
        return [asyncio.create_task(j) for j in jobs]

    @staticmethod
    async def _delayed(delay, coro):
        await asyncio.sleep(delay)
        return await coro

    # ---------------------------------------------------------------- run + report
    async def main(self):
        a = self.a
        loops = [asyncio.create_task(self.browser(i)) for i in range(a.browsers)]
        loops += [asyncio.create_task(self.teacher(i)) for i in range(a.teachers)]
        loops += [asyncio.create_task(self.poller(i)) for i in range(a.pollers)]
        sub_tasks = []
        if a.burst:
            sub_tasks += await self.burst()
        if a.real_list:
            sub_tasks += await self.real_videos()
        if a.submit_every:
            stream = asyncio.create_task(self.submit_stream())
        await asyncio.sleep(max(0, self.stop_at - time.monotonic()))
        if a.submit_every:
            sub_tasks += await stream
        # Hết thời gian tải nền: chờ các bài nộp còn đang chấm (đo thời gian xả hàng đợi)
        if sub_tasks:
            await asyncio.gather(*sub_tasks)
        for t in loops:
            t.cancel()
        await asyncio.gather(*loops, return_exceptions=True)
        await self.client.aclose()
        self.report()

    def report(self):
        out = HERE / "results" / "v2" / self.a.name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "requests.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "label", "status", "ms"])
            w.writerows(self.req)
        keys = sorted({k for s in self.subs for k in s})
        with open(out / "submissions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.subs)
        wall = time.monotonic() - self.t0
        by = defaultdict(list)
        for t, label, code, ms in self.req:
            by[label].append((code, ms))
        summary = {"name": self.a.name, "args": vars(self.a), "wall_s": round(wall, 1), "endpoints": {}}
        for label, rows in sorted(by.items()):
            ms = [m for _, m in rows]
            codes = defaultdict(int)
            for c, _ in rows:
                codes[str(c)] += 1
            summary["endpoints"][label] = {"n": len(rows), "avg": round(statistics.mean(ms), 1), "p50": pct(ms, 50),
                                           "p95": pct(ms, 95), "p99": pct(ms, 99), "max": max(ms), "codes": dict(codes)}
        api = [(c, m) for _, label, c, m in self.req if label != "PUT minio"]
        sys_err = sum(1 for c, _ in api if not (isinstance(c, int) and c < 500))
        summary["api"] = {"n": len(api), "rps": round(len(api) / wall, 1), "system_errors": sys_err,
                          "system_error_pct": round(sys_err * 100 / max(1, len(api)), 3),
                          "p50": pct([m for _, m in api], 50), "p95": pct([m for _, m in api], 95),
                          "p99": pct([m for _, m in api], 99)}
        for kind, cond in (("short", lambda s: s["size_mb"] < 20), ("long", lambda s: s["size_mb"] >= 20)):
            rows = [s for s in self.subs if cond(s)]
            ok = [s for s in rows if not s.get("error")]
            if not rows:
                continue
            e2e = [s["e2e_ms"] / 1000 for s in ok]
            summary[f"submissions_{kind}"] = {
                "total": len(rows), "ok": len(ok), "errors": sorted({s["error"] for s in rows if s.get("error")}),
                "submit_p95_ms": pct([s["submit_ms"] for s in ok], 95),
                "e2e_avg_s": round(statistics.mean(e2e), 1) if e2e else None, "e2e_p50_s": pct(e2e, 50),
                "e2e_p95_s": pct(e2e, 95), "e2e_max_s": max(e2e) if e2e else None,
                "last_done_s": max((s["done_s"] for s in ok), default=None)}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print(json.dumps({k: v for k, v in summary.items() if k not in ("args", "endpoints")}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--duration", type=float, default=300, help="giây tải nền (browser/teacher/poller)")
    ap.add_argument("--ramp", type=float, default=30)
    ap.add_argument("--browsers", type=int, default=0)
    ap.add_argument("--teachers", type=int, default=0)
    ap.add_argument("--pollers", type=int, default=0)
    ap.add_argument("--think-min", type=float, default=3)
    ap.add_argument("--think-max", type=float, default=8)
    ap.add_argument("--poll", type=float, default=3)
    ap.add_argument("--submit-every", type=float, default=0, help="trung bình giây giữa 2 bài nộp (0 = tắt)")
    ap.add_argument("--short-ratio", type=float, default=0.8)
    ap.add_argument("--double-click", type=float, default=0, help="xác suất SV bấm nộp 2 lần")
    ap.add_argument("--burst", action="store_true")
    ap.add_argument("--login-burst", type=int, default=0)
    ap.add_argument("--login-window", type=float, default=30)
    ap.add_argument("--exam-submits", type=int, default=0)
    ap.add_argument("--submit-window", type=float, default=300)
    ap.add_argument("--real-list", default="")
    ap.add_argument("--max-wait", type=float, default=7200)
    ap.add_argument("--max-conn", type=int, default=2000)
    a = ap.parse_args()
    random.seed(int(os.getenv("SEED", "42")))
    asyncio.run(Run(a).main())


if __name__ == "__main__":
    main()
