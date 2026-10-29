"""Xóa video MinIO của các bài nộp test đã kết thúc (result_id lấy từ submissions.csv có summary.json)."""
import csv
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "results"
ids = set()
for sub in root.rglob("submissions.csv"):
    if (sub.parent / "summary.json").exists():  # bài đã chạy xong
        ids |= {r["result_id"] for r in csv.DictReader(open(sub)) if r.get("result_id")}
print("test result ids:", len(ids))
code = """
import sys
from minio.deleteobjects import DeleteObject
from app.services.minio_service import minio_service as m
from app.core.config import settings as s
ids = set(sys.stdin.read().split())
keys = []; size = 0
for o in m.client.list_objects(s.MINIO_BUCKET_NAME, prefix="submissions/", recursive=True):
    if o.object_name.rsplit("/", 1)[-1].split(".")[0] in ids:
        keys.append(o.object_name); size += o.size
dry = %r
print("matched objects", len(keys), "size GB", round(size / 1e9, 1))
if not dry:
    errs = list(m.client.remove_objects(s.MINIO_BUCKET_NAME, [DeleteObject(k) for k in keys]))
    print("deleted", len(keys) - len(errs), "errors", len(errs))
""" % ("--dry" in sys.argv)
r = subprocess.run(["docker", "exec", "-i", "haui-fitness-backend-1", "python", "-c", code],
                   input="\n".join(ids), capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[-500:])
