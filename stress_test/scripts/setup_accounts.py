"""Tạo / xóa tài khoản test cho load test V2.

  python3 setup_accounts.py create --n 300   # đăng ký + ghi danh lớp Pickleball, ghi accounts_v2.json + jmeter/tokens_v2.csv
  python3 setup_accounts.py create --n 300 --start 301   # lô mới (lô cũ đã DONE hết bài nên bị chặn nộp)
  python3 setup_accounts.py delete           # xóa enrollment + user theo accounts_v2.json
"""
import os
import argparse
import asyncio
import json
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000/api"
COURSE_ID = "6ae093a7bf50c057267d76e7"  # Pickleball căn bản & nâng cao
PASSWORD = "Loadtest@2026"
HERE = Path(__file__).resolve().parent.parent
STATE = HERE / "accounts_v2.json"
TOKENS = HERE / "jmeter" / "tokens_v2.csv"


def data(r):
    body = r.json()
    return body.get("data", body) if isinstance(body, dict) else body


async def login(c, email, password):
    r = await c.post(f"{BASE}/auth/login", json={"email": email, "password": password})
    r.raise_for_status()
    return data(r)["accessToken"], data(r)["user"]["id"]


async def create(n, admin_email, admin_password, start=1):
    async with httpx.AsyncClient(timeout=60) as c:
        admin_token, _ = await login(c, admin_email, admin_password)
        h = {"Authorization": f"Bearer {admin_token}"}
        sem = asyncio.Semaphore(8)
        accounts = []

        async def one(i):
            email = f"loadtest{i:03d}@sv.haui.edu.vn"
            async with sem:
                r = await c.post(f"{BASE}/auth/register", json={
                    "name": f"Load Test {i:03d}", "email": email, "password": PASSWORD,
                    "role": "student", "userCode": f"LT2026{i:04d}",
                })
                if r.status_code not in (200, 201):
                    token, uid = await login(c, email, PASSWORD)  # đã tồn tại từ lần chạy trước
                else:
                    uid = data(r)["id"]
                    token, _ = await login(c, email, PASSWORD)
                e = await c.post(f"{BASE}/enrollments", headers=h, json={"userId": uid, "courseId": COURSE_ID})
                enrollment = data(e).get("id") if e.status_code in (200, 201) else None
                accounts.append({"email": email, "userId": uid, "enrollmentId": enrollment, "token": token,
                                 "enrollStatus": e.status_code})

        await asyncio.gather(*(one(i) for i in range(start, start + n)))
    accounts.sort(key=lambda a: a["email"])
    # Gộp với lô cũ để lệnh delete xóa được tất cả; tokens chỉ chứa lô vừa tạo.
    old = json.loads(STATE.read_text()) if STATE.exists() else []
    new_emails = {a["email"] for a in accounts}
    merged = [a for a in old if a["email"] not in new_emails] + [{k: v for k, v in a.items() if k != "token"} for a in accounts]
    STATE.write_text(json.dumps(merged, indent=1))
    TOKENS.write_text("".join(f"{a['email']},{a['token']}\n" for a in accounts))
    bad = [a for a in accounts if a["enrollStatus"] not in (200, 201)]
    print(f"accounts {len(accounts)}, enrolled OK {len(accounts) - len(bad)}, enroll errors {len(bad)}",
          (bad[0]["email"], bad[0]["enrollStatus"]) if bad else "")


async def delete(admin_email, admin_password):
    accounts = json.loads(STATE.read_text())
    async with httpx.AsyncClient(timeout=60) as c:
        admin_token, _ = await login(c, admin_email, admin_password)
        h = {"Authorization": f"Bearer {admin_token}"}
        codes = {}
        for a in accounts:
            if a.get("enrollmentId"):
                await c.delete(f"{BASE}/enrollments/{a['enrollmentId']}", headers=h)
            r = await c.delete(f"{BASE}/users/{a['userId']}", headers=h)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
    print("delete users:", codes)
    TOKENS.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["create", "delete"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=1)  # số thứ tự tài khoản đầu tiên
    ap.add_argument("--admin", default="admin@haui.edu.vn")
    ap.add_argument("--admin-password", default=os.environ.get("HAUI_ADMIN_PASSWORD"))
    a = ap.parse_args()
    asyncio.run(create(a.n, a.admin, a.admin_password, a.start) if a.action == "create" else delete(a.admin, a.admin_password))
