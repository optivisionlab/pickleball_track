"""T1.2: độ trễ đơn lẻ (1 user, tuần tự) của từng API. python3 endpoint_latency.py <result_id> [n]"""
import json, statistics, sys, time
from pathlib import Path
import os
import httpx
B = "http://127.0.0.1:8000/api"; HERE = Path(__file__).resolve().parent.parent
COURSE, TASK = "6ae093a7bf50c057267d76e7", "6ae093a7bf50c057267d76e9"
rid = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
c = httpx.Client(timeout=60)
def tok(email, pw):
    return {"Authorization": "Bearer " + c.post(f"{B}/auth/login", json={"email": email, "password": pw}).json()["data"]["accessToken"]}
student = {"Authorization": "Bearer " + (HERE / "jmeter" / "tokens_v2.csv").read_text().split("\n")[0].split(",")[1]}
teacher, admin = tok("longdv@haui.edu.vn", os.environ["HAUI_TEACHER_PASSWORD"]), tok("admin@haui.edu.vn", os.environ["HAUI_ADMIN_PASSWORD"])
cases = [
  ("SV", "GET /", "http://127.0.0.1:8000/", {}),
  ("SV", "GET /health/ready", "http://127.0.0.1:8000/health/ready", {}),
  ("SV", "GET /auth/me", f"{B}/auth/me", student),
  ("SV", "GET /enrollments/my-courses", f"{B}/enrollments/my-courses", student),
  ("SV", "GET /courses/{id}", f"{B}/courses/{COURSE}", student),
  ("SV", "GET /tasks?course_id", f"{B}/tasks?course_id={COURSE}&page_size=20", student),
  ("SV", "GET /tasks/{id}", f"{B}/tasks/{TASK}", student),
  ("SV", "GET /sports", f"{B}/sports", student),
  ("SV", "GET /video-results", f"{B}/video-results?page_size=10", student),
  ("SV", "GET /video-results/{id}", f"{B}/video-results/{rid}", student),
  ("SV", "GET /video-results/{id}/status", f"{B}/video-results/{rid}/status", student),
  ("SV", "GET /video-results/{id}/video-url", f"{B}/video-results/{rid}/video-url", student),
  ("SV", "GET /student/courses/{id}/grades", f"{B}/student/courses/{COURSE}/grades", student),
  ("SV", "GET /notifications", f"{B}/notifications", student),
  ("GV", "GET /teacher/courses", f"{B}/teacher/courses", teacher),
  ("GV", "GET /teacher/courses/{id}", f"{B}/teacher/courses/{COURSE}", teacher),
  ("GV", "GET /teacher/courses/{id}/gradebook", f"{B}/teacher/courses/{COURSE}/gradebook", teacher),
  ("GV", "GET /teacher/courses/{id}/stats", f"{B}/teacher/courses/{COURSE}/stats", teacher),
  ("GV", "GET /teacher/courses/{id}/tasks", f"{B}/teacher/courses/{COURSE}/tasks", teacher),
  ("GV", "GET /tasks/{id}/submissions", f"{B}/tasks/{TASK}/submissions", teacher),
  ("GV", "GET /enrollments?course_id", f"{B}/enrollments?course_id={COURSE}&page_size=100", teacher),
  ("AD", "GET /users", f"{B}/users?page_size=50", admin),
  ("AD", "GET /courses", f"{B}/courses", admin),
]
out = []
for role, label, url, h in cases:
    ms, codes = [], {}
    for _ in range(N):
        t = time.perf_counter(); r = c.get(url, headers=h); ms.append((time.perf_counter() - t) * 1000)
        codes[r.status_code] = codes.get(r.status_code, 0) + 1
    ms.sort(); q = lambda p: round(ms[min(len(ms) - 1, int(p * (len(ms) - 1)))], 1)
    row = {"role": role, "endpoint": label, "n": N, "min": round(ms[0], 1), "avg": round(statistics.mean(ms), 1),
           "p50": q(.5), "p95": q(.95), "p99": q(.99), "max": round(ms[-1], 1), "codes": codes}
    out.append(row); print(f"{role} {label:42} min={row['min']:6} avg={row['avg']:7} p95={row['p95']:7} p99={row['p99']:7} {codes}")
lat = []
for _ in range(50):
    t = time.perf_counter(); c.post(f"{B}/auth/login", json={"email": "loadtest001@sv.haui.edu.vn", "password": "Loadtest@2026"}); lat.append((time.perf_counter() - t) * 1000)
lat.sort(); print(f"SV POST /auth/login (x50)                     min={lat[0]:.1f} avg={statistics.mean(lat):.1f} p95={lat[47]:.1f}")
out.append({"role": "SV", "endpoint": "POST /auth/login", "n": 50, "min": round(lat[0], 1), "avg": round(statistics.mean(lat), 1), "p95": round(lat[47], 1)})
d = HERE / "results" / "v3" / "T1_2_latency"; d.mkdir(parents=True, exist_ok=True)
(d / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
