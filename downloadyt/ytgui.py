#!/usr/bin/env python3
"""
Giao diện cho tool download hàng loạt video YouTube (chạy local trên trình duyệt).

Chạy:   ./ytdown        (hoặc: python ytgui.py)
Mọi thao tác (kéo thả CSV, chọn thư mục, bắt đầu/dừng, thử lại...) làm trên giao diện.
Tắt tool: bấm nút "Tắt tool" trên giao diện hoặc Ctrl+C ở terminal.
"""

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import yt_dlp  # noqa: E402
from tooldownyt import JS_RUNTIMES, build_format, is_fatal_error, parse_jobs_text, safe_filename  # noqa: E402

PORT = 8765
API_HEADER = "X-Ytdown"  # header bắt buộc, chặn trang web lạ gọi API qua localhost
CONFIG_PATH = os.path.expanduser("~/.config/ytdown_gui.json")
DEFAULT_SETTINGS = {
    "output": os.path.join(HERE, "downloads"),
    "quality": 2160,          # 0 = cao nhất có thể
    "container": "mp4",
    "jobs": 2,
    "fragments": 4,
    "video_only": False,
    "playlist": False,
    "subs": "",
    "cookies_browser": "",
    "archive": True,
    "retries": 3,
}
ACTIVE = ("starting", "downloading", "processing")


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULT_SETTINGS})
    except (OSError, ValueError):
        pass
    return s


def save_settings(s):
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def clean_settings(raw, base):
    s = dict(base)
    for k, default in DEFAULT_SETTINGS.items():
        if k not in raw:
            continue
        v = raw[k]
        try:
            if isinstance(default, bool):
                s[k] = bool(v)
            elif isinstance(default, int):
                s[k] = max(0, int(v))
            else:
                s[k] = str(v).strip()
        except (TypeError, ValueError):
            pass
    s["jobs"] = min(max(s["jobs"], 1), 10)
    s["fragments"] = min(max(s["fragments"], 1), 16)
    s["retries"] = min(max(s["retries"], 1), 10)
    s["output"] = os.path.abspath(os.path.expanduser(s["output"] or DEFAULT_SETTINGS["output"]))
    return s


def open_path(path):
    opener = "xdg-open" if sys.platform.startswith("linux") else ("open" if sys.platform == "darwin" else None)
    if opener:
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.startfile(path)  # type: ignore[attr-defined]


def pick_folder(current):
    start = current if os.path.isdir(current) else os.path.expanduser("~")
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--directory", "--title=Chọn thư mục lưu video",
               f"--filename={start.rstrip('/')}/"]
    elif shutil.which("kdialog"):
        cmd = ["kdialog", "--getexistingdirectory", start]
    else:
        return None
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() or None


class YDLLogger:
    def __init__(self, engine, tag):
        self.engine, self.tag = engine, tag

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        self.engine.log(f"{self.tag} ⚠ {msg}")

    def error(self, msg):
        pass  # được xử lý trong run_job


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs = []
        self.next_id = 1
        self.logs = deque(maxlen=2000)
        self.log_seq = 0
        self.stop_event = threading.Event()
        self.workers = []
        self.settings = load_settings()
        self.run_settings = dict(self.settings)
        self.last_report = ""

    # ------------------------------------------------------------ helpers
    def log(self, msg):
        with self.lock:
            self.log_seq += 1
            self.logs.append((self.log_seq, f"[{datetime.now():%H:%M:%S}] {msg}"))

    @property
    def running(self):
        return any(t.is_alive() for t in self.workers)

    def find(self, job_id):
        return next((j for j in self.jobs if j["id"] == job_id), None)

    def snapshot(self, since):
        with self.lock:
            return {
                "running": self.running,
                "stopping": self.running and self.stop_event.is_set(),
                "jobs": [dict(j) for j in self.jobs],
                "logs": [line for seq, line in self.logs if seq > since],
                "log_seq": self.log_seq,
                "settings": self.settings,
                "report": self.last_report,
            }

    # ------------------------------------------------------------ list ops
    def add_text(self, text, source):
        parsed = parse_jobs_text(text)
        added = 0
        with self.lock:
            existing = {j["url"] for j in self.jobs}
            for p in parsed:
                if p.url in existing:
                    continue
                existing.add(p.url)
                self.jobs.append({
                    "id": self.next_id, "url": p.url, "name": p.name, "quality": p.quality,
                    "folder": p.folder, "status": "queued", "phase": "", "progress": 0.0,
                    "speed": None, "eta": None, "size": None, "title": "", "resolution": "",
                    "filepath": "", "error": "", "attempts": 0,
                })
                self.next_id += 1
                added += 1
        self.log(f"Thêm từ {source}: {added} link mới, {len(parsed) - added} link trùng bỏ qua")
        return {"added": added, "duplicate": len(parsed) - added}

    def remove(self, ids):
        with self.lock:
            self.jobs = [j for j in self.jobs if j["id"] not in ids or j["status"] in ACTIVE]

    def clear(self, mode):
        with self.lock:
            if mode == "done":
                self.jobs = [j for j in self.jobs if j["status"] not in ("done", "skipped")]
            else:
                self.jobs = [j for j in self.jobs if j["status"] in ACTIVE]

    def requeue(self, ids=None, statuses=("failed", "cancelled")):
        with self.lock:
            for j in self.jobs:
                if (ids is None and j["status"] in statuses) or (ids and j["id"] in ids and j["status"] not in ACTIVE):
                    j.update(status="queued", phase="", progress=0.0, error="", speed=None, eta=None)

    def update_settings(self, raw):
        with self.lock:
            self.settings = clean_settings(raw, self.settings)
            save_settings(self.settings)
            return self.settings

    # ------------------------------------------------------------ run
    def start(self, raw_settings):
        with self.lock:
            if self.running:
                return {"ok": False, "error": "Đang chạy rồi"}
            self.update_settings(raw_settings)
            self.requeue(statuses=("cancelled",))
            queued = sum(j["status"] == "queued" for j in self.jobs)
            if not queued:
                return {"ok": False, "error": "Không có video nào đang chờ tải"}
            os.makedirs(self.settings["output"], exist_ok=True)
            self.stop_event.clear()
            self.run_settings = dict(self.settings)
            n = min(self.settings["jobs"], queued)
            self.workers = [threading.Thread(target=self.worker, daemon=True) for _ in range(n)]
            for t in self.workers:
                t.start()
            threading.Thread(target=self.monitor, args=(list(self.workers), time.time()), daemon=True).start()
        self.log(f"▶ Bắt đầu tải {queued} video | tối đa {self.run_settings['quality'] or 'cao nhất'}p | "
                 f"{n} luồng | lưu tại {self.run_settings['output']}")
        return {"ok": True}

    def stop(self):
        if self.running:
            self.stop_event.set()
            self.log("■ Đang dừng... (các video dở dang sẽ tải tiếp khi bấm Bắt đầu lại)")

    def worker(self):
        while not self.stop_event.is_set():
            with self.lock:
                job = next((j for j in self.jobs if j["status"] == "queued"), None)
                if job is None:
                    return
                job.update(status="starting", phase="Đang lấy thông tin", error="")
            self.run_job(job, self.run_settings)

    def monitor(self, workers, started):
        for t in workers:
            t.join()
        with self.lock:
            jobs = list(self.jobs)
        counts = {k: sum(j["status"] == k for j in jobs) for k in ("done", "skipped", "failed", "cancelled", "queued")}
        self.last_report = self.write_report(self.run_settings["output"])
        self.log(f"■ Kết thúc sau {(time.time() - started) / 60:.1f} phút: {counts['done']} xong, "
                 f"{counts['skipped']} bỏ qua, {counts['failed']} lỗi, "
                 f"{counts['cancelled'] + counts['queued']} chưa tải. Báo cáo: {self.last_report}")

    def write_report(self, outdir):
        path = os.path.join(outdir, f"report_{datetime.now():%Y%m%d_%H%M%S}.csv")
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                f.write(self.report_csv())
            return path
        except OSError:
            return ""

    def report_csv(self):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["index", "url", "status", "title", "resolution", "filepath", "attempts", "error"])
        with self.lock:
            for j in self.jobs:
                w.writerow([j["id"], j["url"], j["status"], j["title"], j["resolution"],
                            j["filepath"], j["attempts"], j["error"]])
        return buf.getvalue()

    def set(self, job, **kw):
        with self.lock:
            job.update(**kw)

    def on_progress(self, job, d):
        if self.stop_event.is_set():
            raise yt_dlp.utils.DownloadCancelled("Đã dừng")
        if d["status"] != "downloading":
            return
        info = d.get("info_dict") or {}
        kind = "tiếng" if info.get("vcodec") == "none" else ("hình" if info.get("acodec") == "none" else "video")
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        pct = d["downloaded_bytes"] * 100 / total if total else 0.0
        self.set(job, status="downloading", phase=f"Đang tải {kind}", progress=round(min(pct, 100), 1),
                 speed=d.get("speed"), eta=d.get("eta"), size=total)

    def on_postprocess(self, job, d):
        if d["status"] != "started":
            return
        pp = d.get("postprocessor", "")
        label = "Đang ghép hình + tiếng" if "Merge" in pp else f"Đang xử lý ({pp})"
        self.set(job, status="processing", phase=label, progress=100.0, speed=None, eta=None)

    def run_job(self, job, s):
        tag = f"#{job['id']}"
        outdir = os.path.join(s["output"], safe_filename(job["folder"])) if job["folder"] else s["output"]
        height = job["quality"] or s["quality"] or None
        name = safe_filename(job["name"]) if job["name"] else "%(title).150B [%(id)s]"
        opts = {
            "format": build_format(height, s["video_only"]),
            "format_sort": ["res", "fps", "hdr:12", "vbr", "abr"],
            "merge_output_format": s["container"],
            "outtmpl": os.path.join(outdir, name + ".%(ext)s"),
            "noplaylist": not s["playlist"],
            "js_runtimes": JS_RUNTIMES,
            "retries": 10,
            "fragment_retries": 10,
            "concurrent_fragment_downloads": s["fragments"],
            "continuedl": True,
            "overwrites": False,
            "windowsfilenames": True,
            "logger": YDLLogger(self, tag),
            "progress_hooks": [lambda d: self.on_progress(job, d)],
            "postprocessor_hooks": [lambda d: self.on_postprocess(job, d)],
            "noprogress": True,
            "quiet": True,
        }
        if s["archive"]:
            opts["download_archive"] = os.path.join(s["output"], ".downloaded_archive.txt")
        if s["cookies_browser"]:
            opts["cookiesfrombrowser"] = (s["cookies_browser"],)
        if s["subs"]:
            opts.update({"writesubtitles": True, "subtitlesformat": "srt/best",
                         "subtitleslangs": [x.strip() for x in s["subs"].split(",") if x.strip()]})

        error = ""
        for attempt in range(1, s["retries"] + 1):
            self.set(job, attempts=attempt)
            try:
                os.makedirs(outdir, exist_ok=True)
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(job["url"], download=False)
                    if info is None or ydl.in_download_archive(info):
                        self.set(job, status="skipped", phase="Đã tải trước đó", progress=100.0)
                        self.log(f"{tag} ⏭ bỏ qua (đã tải trước đó): {job['url']}")
                        return
                    entries = [e for e in (info.get("entries") or [info]) if e]
                    if len(entries) == 1:
                        e = entries[0]
                        res = f"{e.get('width')}x{e.get('height')} {e.get('fps') or ''}fps".replace("None", "?")
                    else:
                        res = f"playlist {len(entries)} video"
                    self.set(job, title=info.get("title", ""), resolution=res, phase="Bắt đầu tải")
                    self.log(f"{tag} ▶ {info.get('title', '')[:80]} | {res}")

                    ydl.process_ie_result(info, download=True)

                    paths = [rd.get("filepath") or rd.get("_filename")
                             for e in entries for rd in (e.get("requested_downloads") or [])]
                    paths = [p for p in paths if p]
                    self.set(job, status="done", phase="", progress=100.0, speed=None, eta=None,
                             filepath=paths[0] if paths else "")
                    self.log(f"{tag} ✔ xong: {os.path.basename(paths[0]) if paths else job['title']}")
                    return
            except yt_dlp.utils.DownloadCancelled:
                self.set(job, status="cancelled", phase="", speed=None, eta=None)
                self.log(f"{tag} ■ đã dừng")
                return
            except yt_dlp.utils.DownloadError as e:
                error = yt_dlp.utils.remove_terminal_sequences(str(e)).replace("ERROR: ", "")
                if self.stop_event.is_set():
                    self.set(job, status="cancelled", phase="", speed=None, eta=None)
                    return
                if is_fatal_error(error):
                    break
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
            if attempt < s["retries"]:
                wait = 5 * attempt
                self.set(job, status="starting", phase=f"Lỗi, thử lại sau {wait}s ({attempt}/{s['retries']})",
                         error=error)
                self.log(f"{tag} ↻ thử lại sau {wait}s: {error[:150]}")
                if self.stop_event.wait(wait):
                    self.set(job, status="cancelled", phase="")
                    return

        self.set(job, status="failed", phase="", speed=None, eta=None, error=error)
        self.log(f"{tag} ✘ LỖI: {job['url']} — {error[:200]}")


ENGINE = Engine()
SERVER = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        data = body if isinstance(body, bytes) else (
            body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def allowed(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self.allowed():
            return self.send(403, {"error": "forbidden"})
        path, _, query = self.path.partition("?")
        if path == "/":
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/ping":
            return self.send(200, {"app": "ytdown"})
        if path == "/api/state":
            since = int(query.split("since=")[-1]) if "since=" in query else 0
            return self.send(200, ENGINE.snapshot(since))
        if path == "/api/report":
            name = f"report_{datetime.now():%Y%m%d_%H%M%S}.csv"
            return self.send(200, "﻿" + ENGINE.report_csv(), "text/csv; charset=utf-8",
                             {"Content-Disposition": f'attachment; filename="{name}"'})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.allowed() or self.headers.get(API_HEADER) != "1":
            return self.send(403, {"error": "forbidden"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self.send(400, {"error": "bad json"})
        try:
            self.send(200, self.route(self.path, body))
        except Exception as e:  # noqa: BLE001
            self.send(500, {"error": f"{type(e).__name__}: {e}"})

    def route(self, path, b):
        E = ENGINE
        if path == "/api/add":
            return E.add_text(b.get("text", ""), b.get("source", "ô dán link"))
        if path == "/api/remove":
            E.remove(set(b.get("ids", [])))
        elif path == "/api/clear":
            E.clear(b.get("mode", "all"))
        elif path == "/api/requeue":
            E.requeue(ids=set(b["ids"]) if b.get("ids") else None)
        elif path == "/api/settings":
            return {"settings": E.update_settings(b)}
        elif path == "/api/start":
            return E.start(b.get("settings", {}))
        elif path == "/api/stop":
            E.stop()
        elif path == "/api/pick-folder":
            folder = pick_folder(E.settings["output"])
            if folder:
                E.update_settings({"output": folder})
            return {"folder": folder}
        elif path == "/api/open":
            target = b.get("target")
            if target == "output":
                p = E.settings["output"]
                os.makedirs(p, exist_ok=True)
            elif target == "report":
                p = E.last_report
            else:
                with E.lock:
                    job = E.find(b.get("id"))
                p = job["filepath"] if job else ""
                if p and not b.get("file"):
                    p = os.path.dirname(p)
            if not p or not os.path.exists(p):
                return {"ok": False, "error": "Không tìm thấy file/thư mục"}
            open_path(p)
        elif path == "/api/shutdown":
            E.stop()
            threading.Thread(target=SERVER.shutdown, daemon=True).start()
        else:
            return {"error": "not found"}
        return {"ok": True}


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YouTube Batch Downloader</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--text:#1d2129;--muted:#6b7280;--line:#e3e5e8;--accent:#2563eb;--ok:#16a34a;--err:#dc2626;--warn:#d97706}
@media (prefers-color-scheme:dark){:root{--bg:#16181d;--card:#1f2228;--text:#e6e7ea;--muted:#9aa0aa;--line:#33363d;--accent:#5b8cff;--ok:#3fbf6b;--err:#f05252;--warn:#f0a030}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text)}
main{max-width:1280px;margin:0 auto;padding:16px}
h1{font-size:19px;margin:0 0 12px;display:flex;align-items:center;gap:10px}
h1 .sp{flex:1}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:12px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:820px){.grid{grid-template-columns:1fr}}
#drop{border:2px dashed var(--line);border-radius:8px;padding:22px;text-align:center;cursor:pointer;color:var(--muted);height:100%;display:flex;flex-direction:column;justify-content:center;gap:4px}
#drop:hover,body.drag #drop{border-color:var(--accent);color:var(--accent);background:color-mix(in srgb,var(--accent) 7%,transparent)}
#drop b{font-size:15px;color:var(--text)}
textarea,input,select{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
textarea{width:100%;height:86px;resize:vertical}
label{display:flex;align-items:center;gap:6px;color:var(--muted)}label>span{white-space:nowrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--accent)}button:disabled{opacity:.45;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button.danger{color:var(--err)}button.sm{padding:2px 7px;font-size:12px}
#output{flex:1;min-width:260px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;min-width:80px}.bar>i{display:block;height:100%;background:var(--accent);transition:width .3s}
.bar.ok>i{background:var(--ok)}.bar.err>i{background:var(--err)}
#overall .bar{height:12px;flex:1}
table{width:100%;border-collapse:collapse}th,td{padding:7px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}
td.n{color:var(--muted);width:36px}td.v{word-break:break-all}td.v small{color:var(--muted);display:block}
td.p{width:170px}td.p small{color:var(--muted)}td.a{white-space:nowrap;width:1px}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;background:var(--line);white-space:nowrap}
.s-done,.s-skipped{background:color-mix(in srgb,var(--ok) 20%,transparent);color:var(--ok)}
.s-failed{background:color-mix(in srgb,var(--err) 18%,transparent);color:var(--err)}
.s-downloading,.s-processing,.s-starting{background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--accent)}
.s-cancelled{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn)}
.err{color:var(--err);font-size:12px}
#tablewrap{max-height:56vh;overflow:auto}
#empty{color:var(--muted);text-align:center;padding:26px}
#log{height:160px;overflow:auto;font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;background:var(--bg);border-radius:6px;padding:8px;margin:8px 0 0}
#toast{position:fixed;right:16px;bottom:16px;background:var(--text);color:var(--bg);padding:10px 14px;border-radius:6px;opacity:0;transition:opacity .3s;pointer-events:none;max-width:420px}
#toast.show{opacity:.95}
.muted{color:var(--muted)}details summary{cursor:pointer;color:var(--muted)}
</style></head><body>
<main>
<h1>📥 YouTube Batch Downloader <span class="sp"></span><button id="btnQuit" class="sm danger">Tắt tool</button></h1>

<div class="grid">
  <div class="card"><div id="drop"><b>Kéo thả file CSV vào đây</b><span>hoặc bấm để chọn file (.csv / .txt, chọn được nhiều file)</span>
    <span style="font-size:12px">Cột nhận diện: url/link (bắt buộc), name, quality, folder — hoặc file chỉ chứa link</span></div>
    <input type="file" id="file" accept=".csv,.txt,.tsv,text/csv,text/plain" multiple hidden></div>
  <div class="card"><textarea id="paste" placeholder="Hoặc dán link YouTube vào đây (mỗi dòng 1 link, hoặc dán nội dung CSV)"></textarea>
    <div class="row" style="margin-top:8px"><button id="btnAdd">Thêm vào danh sách</button><span class="muted" style="font-size:12px">Ctrl+Enter để thêm</span></div></div>
</div>

<div class="card">
  <div class="row"><label style="flex:1"><span>Thư mục lưu</span><input id="output"></label>
    <button id="btnPick">Chọn...</button><button id="btnOpenOut">Mở thư mục</button></div>
  <div class="row" style="margin-top:10px">
    <label><span>Chất lượng tối đa</span><select id="quality">
      <option value="0">Cao nhất có thể</option><option value="4320">8K (4320p)</option><option value="2160">4K (2160p)</option>
      <option value="1440">2K (1440p)</option><option value="1080">1080p</option><option value="720">720p</option>
      <option value="480">480p</option><option value="360">360p</option></select></label>
    <label><span>Định dạng</span><select id="container"><option>mp4</option><option>mkv</option><option>webm</option></select></label>
    <label><span>Tải song song</span><input id="jobs" type="number" min="1" max="10" style="width:58px"></label>
    <label title="Bỏ tích sẽ tải lại cả video đã tải trước đó"><input type="checkbox" id="archive"> Bỏ qua video đã tải</label>
    <label><input type="checkbox" id="video_only"> Chỉ hình (không tiếng)</label>
  </div>
  <details style="margin-top:8px"><summary>Tùy chọn nâng cao</summary><div class="row" style="margin-top:8px">
    <label><input type="checkbox" id="playlist"> Link playlist → tải cả playlist</label>
    <label><span>Phụ đề</span><input id="subs" placeholder="vd: vi,en" style="width:90px"></label>
    <label title="Dùng cho video giới hạn tuổi / thành viên / khi bị YouTube chặn"><span>Cookies từ trình duyệt</span><select id="cookies_browser">
      <option value="">Không dùng</option><option>chrome</option><option>chromium</option><option>firefox</option><option>edge</option><option>brave</option><option>opera</option></select></label>
    <label><span>Fragment/video</span><input id="fragments" type="number" min="1" max="16" style="width:58px"></label>
    <label><span>Số lần thử lại</span><input id="retries" type="number" min="1" max="10" style="width:58px"></label>
  </div></details>
</div>

<div class="card">
  <div class="row">
    <button id="btnStart" class="primary">▶ Bắt đầu tải</button><button id="btnStop">■ Dừng</button>
    <button id="btnRetry">↻ Thử lại các video lỗi</button><button id="btnClearDone">Xóa video đã xong</button>
    <button id="btnClearAll" class="danger">Xóa tất cả</button><span style="flex:1"></span>
    <button id="btnReport">⬇ Tải báo cáo CSV</button>
  </div>
  <div id="overall" class="row" style="margin-top:12px"><div class="bar"><i style="width:0"></i></div><span id="summary" class="muted"></span></div>
</div>

<div class="card" style="padding:0"><div id="tablewrap"><table>
  <thead><tr><th>#</th><th>Video</th><th>Chất lượng</th><th>Trạng thái</th><th>Tiến độ</th><th></th></tr></thead>
  <tbody id="rows"></tbody></table><div id="empty">Chưa có video nào. Kéo thả file CSV hoặc dán link ở trên.</div></div></div>

<div class="card"><details id="logbox" open><summary>Nhật ký</summary><pre id="log"></pre></details></div>
</main>
<div id="toast"></div>
<script>
const $=id=>document.getElementById(id);
const FIELDS=["output","quality","container","jobs","fragments","video_only","playlist","subs","cookies_browser","archive","retries"];
const STATUS={queued:"Chờ tải",starting:"Chuẩn bị",downloading:"Đang tải",processing:"Đang xử lý",done:"Xong",skipped:"Bỏ qua",failed:"Lỗi",cancelled:"Đã dừng"};
let logSeq=0,state=null,settingsLoaded=false,lastHtml="",alive=true;

function toast(m){const t=$("toast");t.textContent=m;t.classList.add("show");clearTimeout(t._h);t._h=setTimeout(()=>t.classList.remove("show"),3500)}
async function api(path,body={}){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Ytdown":"1"},body:JSON.stringify(body)});
  const j=await r.json(); if(j.error) toast(j.error); return j;
}
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function fmtBytes(b){if(!b)return"";const u=["B","KB","MB","GB"];let i=0;while(b>=1024&&i<3){b/=1024;i++}return b.toFixed(i?1:0)+u[i]}
function fmtEta(s){if(s==null)return"";s=Math.round(s);return s>=3600?`${Math.floor(s/3600)}h${String(Math.floor(s%3600/60)).padStart(2,"0")}m`:`${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}`}

function readSettings(){const s={};for(const f of FIELDS){const el=$(f);s[f]=el.type==="checkbox"?el.checked:el.value}return s}
function writeSettings(s){for(const f of FIELDS){const el=$(f);if(el.type==="checkbox")el.checked=!!s[f];else el.value=s[f]??""}}
for(const f of FIELDS) $(f).addEventListener("change",()=>api("/api/settings",readSettings()));

async function addText(text,source){if(!text.trim())return;const r=await api("/api/add",{text,source});
  if(r.added!=null) toast(`Đã thêm ${r.added} link`+(r.duplicate?` (${r.duplicate} link trùng bỏ qua)`:"")+(r.added?"":" — không tìm thấy link mới"));refresh()}
async function addFiles(files){for(const f of files){addText(await f.text(),f.name)}}

const drop=$("drop");drop.onclick=()=>$("file").click();
$("file").onchange=e=>{addFiles(e.target.files);e.target.value=""};
let dragDepth=0;
addEventListener("dragenter",e=>{e.preventDefault();dragDepth++;document.body.classList.add("drag")});
addEventListener("dragleave",()=>{if(--dragDepth<=0){dragDepth=0;document.body.classList.remove("drag")}});
addEventListener("dragover",e=>e.preventDefault());
addEventListener("drop",e=>{e.preventDefault();dragDepth=0;document.body.classList.remove("drag");
  if(e.dataTransfer.files.length)addFiles(e.dataTransfer.files);else addText(e.dataTransfer.getData("text"),"kéo thả")});
$("btnAdd").onclick=async()=>{await addText($("paste").value,"ô dán link");$("paste").value=""};
$("paste").addEventListener("keydown",e=>{if(e.key==="Enter"&&e.ctrlKey)$("btnAdd").click()});

$("btnPick").onclick=async()=>{const r=await api("/api/pick-folder");if(r.folder)$("output").value=r.folder;else if(r.folder===null)toast("Không chọn thư mục (có thể gõ đường dẫn trực tiếp)")};
$("btnOpenOut").onclick=async()=>{await api("/api/settings",readSettings());api("/api/open",{target:"output"})};
$("btnStart").onclick=async()=>{const r=await api("/api/start",{settings:readSettings()});refresh()};
$("btnStop").onclick=()=>api("/api/stop").then(refresh);
$("btnRetry").onclick=()=>api("/api/requeue").then(refresh);
$("btnClearDone").onclick=()=>api("/api/clear",{mode:"done"}).then(refresh);
$("btnClearAll").onclick=()=>{if(confirm("Xóa toàn bộ danh sách? (file đã tải không bị xóa)"))api("/api/clear",{mode:"all"}).then(refresh)};
$("btnReport").onclick=()=>location.href="/api/report";
$("btnQuit").onclick=async()=>{if(!confirm("Tắt tool? Các video đang tải sẽ dừng lại."))return;alive=false;await api("/api/shutdown");
  document.body.innerHTML='<main><div class="card" style="margin-top:40px;text-align:center">Tool đã tắt. Chạy lại lệnh <b>ytdown</b> trong terminal để mở lại.</div></main>'};

$("rows").addEventListener("click",e=>{const b=e.target.closest("button[data-act]");if(!b)return;const id=+b.dataset.id;
  ({open:()=>api("/api/open",{id,file:true}),folder:()=>api("/api/open",{id}),retry:()=>api("/api/requeue",{ids:[id]}),
    del:()=>api("/api/remove",{ids:[id]})})[b.dataset.act]().then(refresh)});

function rowHtml(j){
  const pct=j.progress||0,active=["starting","downloading","processing"].includes(j.status);
  const cls=j.status==="failed"?"err":(j.status==="done"||j.status==="skipped")?"ok":"";
  let info=j.phase||"";
  if(j.status==="downloading"){info+=` · ${pct.toFixed(0)}%`;if(j.speed)info+=` · ${fmtBytes(j.speed)}/s`;if(j.eta!=null)info+=` · còn ${fmtEta(j.eta)}`;if(j.size)info+=` · ${fmtBytes(j.size)}`}
  let acts="";
  if(j.status==="done"&&j.filepath)acts+=`<button class="sm" data-act="open" data-id="${j.id}" title="Mở video">▶</button> <button class="sm" data-act="folder" data-id="${j.id}" title="Mở thư mục chứa">📁</button> `;
  if(["failed","cancelled","done","skipped"].includes(j.status))acts+=`<button class="sm" data-act="retry" data-id="${j.id}" title="Tải lại">↻</button> `;
  if(!active)acts+=`<button class="sm" data-act="del" data-id="${j.id}" title="Xóa khỏi danh sách">✕</button>`;
  const meta=[j.name&&`tên: ${j.name}`,j.quality&&`≤${j.quality}p`,j.folder&&`thư mục: ${j.folder}`].filter(Boolean).join(" · ");
  return `<tr><td class="n">${j.id}</td><td class="v">${j.title?esc(j.title)+`<small>${esc(j.url)}</small>`:esc(j.url)}${meta?`<small>${esc(meta)}</small>`:""}
    ${j.error&&j.status!=="done"?`<div class="err">${esc(j.error.slice(0,300))}</div>`:""}${j.filepath?`<small>${esc(j.filepath)}</small>`:""}</td>
    <td>${esc(j.resolution)}</td><td><span class="badge s-${j.status}">${STATUS[j.status]||j.status}</span></td>
    <td class="p"><div class="bar ${cls}"><i style="width:${j.status==="failed"?100:pct}%"></i></div><small>${esc(info)}</small></td><td class="a">${acts}</td></tr>`;
}

function render(s){
  if(!settingsLoaded){writeSettings(s.settings);settingsLoaded=true}
  const jobs=s.jobs,html=jobs.map(rowHtml).join("");
  if(html!==lastHtml){$("rows").innerHTML=html;lastHtml=html}
  $("empty").style.display=jobs.length?"none":"block";
  const c={};for(const j of jobs)c[j.status]=(c[j.status]||0)+1;
  const finished=(c.done||0)+(c.skipped||0)+(c.failed||0);
  const prog=jobs.length?jobs.reduce((a,j)=>a+(["done","skipped","failed"].includes(j.status)?100:(j.progress||0)),0)/jobs.length:0;
  $("overall").querySelector("i").style.width=prog+"%";
  $("summary").textContent=jobs.length?`${finished}/${jobs.length} · ${c.done||0} xong · ${c.skipped||0} bỏ qua · ${c.failed||0} lỗi · ${(c.downloading||0)+(c.processing||0)+(c.starting||0)} đang tải · ${c.queued||0} chờ`+(s.stopping?" · đang dừng...":""):"";
  $("btnStart").disabled=s.running||!c.queued&&!c.cancelled;$("btnStop").disabled=!s.running||s.stopping;
  $("btnRetry").disabled=!c.failed;$("btnClearDone").disabled=!(c.done||c.skipped);$("btnClearAll").disabled=!jobs.length;$("btnReport").disabled=!jobs.length;
  $("btnStart").textContent=s.running?"Đang tải...":"▶ Bắt đầu tải";
  if(s.logs.length){const l=$("log"),bottom=l.scrollTop+l.clientHeight>=l.scrollHeight-20;l.textContent+=s.logs.join("\n")+"\n";if(bottom)l.scrollTop=l.scrollHeight}
  logSeq=s.log_seq;document.title=s.running?`(${finished}/${jobs.length}) YouTube Downloader`:"YouTube Batch Downloader";
}
async function refresh(){try{const r=await fetch("/api/state?since="+logSeq);state=await r.json();render(state)}catch(e){$("summary").textContent="⚠ Mất kết nối tới tool (đã tắt ở terminal?)"}}
(async function loop(){while(alive){await refresh();await new Promise(r=>setTimeout(r,state&&state.running?700:1500))}})();
</script></body></html>
"""


def main():
    global SERVER
    url = f"http://127.0.0.1:{PORT}/"
    try:
        SERVER = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        # Tool đã chạy sẵn → chỉ mở lại trình duyệt
        try:
            urlopen(Request(url + "api/ping"), timeout=2).read()
            print(f"Tool đang chạy sẵn, mở trình duyệt: {url}")
            webbrowser.open(url)
            return
        except OSError:
            SERVER = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            url = f"http://127.0.0.1:{SERVER.server_address[1]}/"
    if not shutil.which("ffmpeg"):
        print("⚠ Không tìm thấy ffmpeg — cần cài để ghép hình + tiếng: sudo apt install ffmpeg")
    print(f"YouTube Batch Downloader đang chạy tại {url}")
    print("Giao diện sẽ tự mở trên trình duyệt. Nhấn Ctrl+C để tắt.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        SERVER.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE.stop_event.set()
        SERVER.server_close()
        print("\nĐã tắt tool.")


if __name__ == "__main__":
    main()
