"""T2.7: N SV upload video 100 MB lên MinIO cùng lúc (presigned PUT), không xác nhận; xóa file test ngay sau đó.
python3 upload_test.py <N> <tên>"""
import asyncio, json, statistics, subprocess, sys, time
from pathlib import Path
import httpx
HERE = Path(__file__).resolve().parent.parent; B = "http://127.0.0.1:8000/api"
N, NAME = int(sys.argv[1]), sys.argv[2]
VIDEO = HERE / "videos" / "shot2.mov"; BODY = VIDEO.read_bytes()
toks = [l.split(",", 1)[1] for l in (HERE / "jmeter" / "tokens_v2.csv").read_text().split()]
async def one(c, i, rows, keys):
    h = {"Authorization": f"Bearer {toks[i % len(toks)]}"}
    r = await c.post(f"{B}/video-results/upload-url", headers=h, json={"task_id": "6ae093a7bf50c057267d76ea",
        "file_name": "load.mov", "content_type": "video/quicktime", "file_size": len(BODY), "duration_sec": 60})
    if r.status_code != 200:
        rows.append({"error": f"upload-url {r.status_code}"}); return
    up = r.json()["data"]; keys.append(up["objectKey"]); t = time.perf_counter()
    try:
        p = await c.put(up["uploadUrl"], content=BODY, headers={"Content-Type": "application/octet-stream"})
        rows.append({"s": time.perf_counter() - t, "code": p.status_code})
    except Exception as e:
        rows.append({"error": type(e).__name__})
async def main():
    rows, keys = [], []
    async with httpx.AsyncClient(timeout=900, limits=httpx.Limits(max_connections=N + 10)) as c:
        t0 = time.perf_counter(); await asyncio.gather(*(one(c, i, rows, keys) for i in range(N))); wall = time.perf_counter() - t0
    ok = [r["s"] for r in rows if r.get("code") in (200, 204)]
    res = {"name": NAME, "concurrent": N, "ok": len(ok), "errors": [r.get("error") or r.get("code") for r in rows if r.get("code") not in (200, 204)],
           "wall_s": round(wall, 1), "avg_s": round(statistics.mean(ok), 2) if ok else None, "max_s": round(max(ok), 2) if ok else None,
           "aggregate_MBps": round(len(ok) * len(BODY) / 1e6 / wall, 1)}
    print(json.dumps(res))
    d = HERE / "results" / "v3" / NAME; d.mkdir(parents=True, exist_ok=True); (d / "summary.json").write_text(json.dumps(res, indent=1))
    # Xóa ngay các file test vừa upload (không tạo bài nộp)
    code = ("from minio.deleteobjects import DeleteObject\nfrom app.services.minio_service import minio_service as m\n"
            "from app.core.config import settings as s\nimport sys\nks=sys.stdin.read().split()\n"
            "print('deleted', len(ks), 'errors', len(list(m.client.remove_objects(s.MINIO_BUCKET_NAME,[DeleteObject(k) for k in ks]))))")
    print(subprocess.run(["docker", "exec", "-i", "haui-fitness-backend-1", "python", "-c", code], input="\n".join(keys),
                         capture_output=True, text=True).stdout.strip())
asyncio.run(main())
