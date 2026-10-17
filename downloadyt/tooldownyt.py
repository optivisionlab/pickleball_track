#!/usr/bin/env python3
"""
Tool download hàng loạt video YouTube chất lượng cao từ file CSV.

Yêu cầu:
    pip install -U yt-dlp        # bản mới nhất, YouTube thay đổi liên tục
    ffmpeg                        # để ghép video + audio (sudo apt install ffmpeg)

File CSV:
    - Có header: tool tự tìm cột link (url / link / video / youtube ...).
      Cột tùy chọn: "name" (tên file mong muốn), "quality" (vd 1080, 2160)
      và "folder" (thư mục con).
    - Không header: tool quét mọi ô, lấy tất cả link YouTube tìm được.

Ví dụ:
    python tooldownyt.py links.csv
    python tooldownyt.py links.csv -o videos -q 1080 -j 3
    python tooldownyt.py links.csv --video-only --cookies-from-browser chrome
"""

import argparse
import csv
import io
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

try:
    import yt_dlp
except ImportError:
    sys.exit("Chưa cài yt-dlp. Chạy: pip install -U yt-dlp")

YT_URL_RE = re.compile(
    r"(https?://)?(www\.|m\.|music\.)?(youtube\.com/(watch\?\S*v=|shorts/|live/|embed/|playlist\?\S*list=)"
    r"|youtu\.be/)[\w\-?=&%.#/]+",
    re.IGNORECASE,
)
URL_COLUMNS = ("url", "link", "links", "video", "video_url", "videourl", "youtube", "youtube_url", "href")
NAME_COLUMNS = ("name", "filename", "file_name", "title", "ten")
QUALITY_COLUMNS = ("quality", "height", "resolution", "res")
FOLDER_COLUMNS = ("folder", "dir", "subdir", "category")

# Lỗi không thể thử lại (video private/xóa/bị chặn...)
FATAL_ERRORS = ("private video", "unavailable", "removed", "not available",
                "sign in to confirm your age", "members-only", "copyright")

print_lock = threading.Lock()


def is_fatal_error(msg):
    return any(k in msg.lower() for k in FATAL_ERRORS)


# YouTube cần JS runtime để lấy đủ format chất lượng cao; bật tất cả, yt-dlp tự chọn cái có sẵn
JS_RUNTIMES = {"deno": {}, "node": {}, "quickjs": {}, "bun": {}}


def log(msg):
    with print_lock:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


@dataclass
class Job:
    index: int
    url: str
    name: str = ""
    quality: int | None = None
    folder: str = ""


@dataclass
class Result:
    job: Job
    status: str = "pending"  # ok | skipped | failed
    title: str = ""
    resolution: str = ""
    filepath: str = ""
    error: str = ""
    attempts: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- CSV parsing

def normalize_url(url):
    url = url.strip().strip('"').strip("'")
    if url and not url.lower().startswith("http"):
        url = "https://" + url
    return url


def safe_filename(name):
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name).strip(" .")
    return name[:180]


def find_column(header, candidates):
    lowered = [h.strip().lower().lstrip("﻿") for h in header]
    for cand in candidates:
        if cand in lowered:
            return lowered.index(cand)
    return None


def parse_quality(value):
    m = re.search(r"\d{3,4}", value or "")
    if m:
        return int(m.group())
    if (value or "").strip().lower() in ("4k", "uhd"):
        return 2160
    return None


def read_jobs(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return parse_jobs_text(f.read())


def parse_jobs_text(text):
    # Chuẩn hóa xuống dòng: file từ Excel Mac / Windows dùng \r hoặc \r\n
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        rows = [r for r in csv.reader(io.StringIO(text, newline=""), dialect) if any(c.strip() for c in r)]
    except csv.Error:
        # File không đúng chuẩn CSV: coi mỗi dòng là 1 ô, quét tìm link YouTube
        rows = [[line] for line in text.split("\n") if line.strip()]

    if not rows:
        return []

    header = rows[0]
    url_col = find_column(header, URL_COLUMNS)
    jobs, seen = [], set()

    def add(url, name="", quality=None, folder=""):
        url = normalize_url(url)
        if not url or url in seen:
            return
        seen.add(url)
        jobs.append(Job(len(jobs) + 1, url, name.strip(), quality, folder.strip()))

    if url_col is not None:
        name_col = find_column(header, NAME_COLUMNS)
        q_col = find_column(header, QUALITY_COLUMNS)
        dir_col = find_column(header, FOLDER_COLUMNS)

        def cell(row, col):
            return row[col] if col is not None and col < len(row) else ""

        for row in rows[1:]:
            url = cell(row, url_col)
            if not url.strip():
                continue
            extra = (cell(row, name_col), parse_quality(cell(row, q_col)), cell(row, dir_col))
            # Ô chứa nhiều link (dấu ngoặc kép lỗi, nhiều link 1 ô...) → tách riêng từng link
            found = [m.group(0) for m in YT_URL_RE.finditer(url)]
            if len(found) > 1:
                for u in found:
                    add(u, "", *extra[1:])
            else:
                add(url.strip(), *extra)
    else:
        # Không có header nhận diện được: quét toàn bộ ô để tìm link YouTube
        for row in rows:
            for c in row:
                for m in YT_URL_RE.finditer(c):
                    add(m.group(0))
    return jobs


# ---------------------------------------------------------------- download

def build_format(height, video_only):
    """Chọn stream tốt nhất <= height; fallback dần nếu không có."""
    h = f"[height<={height}]" if height else ""
    if video_only:
        return f"bv*{h}/b{h}/bv*/b"
    return f"bv*{h}+ba/b{h}/bv*+ba/b"


class QuietLogger:
    """Ẩn log rác của yt-dlp, chỉ giữ warning/error quan trọng."""

    def __init__(self, tag):
        self.tag = tag

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        if "nsig" in msg or "Some formats" in msg or "JavaScript runtime" in msg:
            log(f"{self.tag} ⚠ {msg}")

    def error(self, msg):
        pass  # lỗi được xử lý ở download_one


def make_progress_hook(tag):
    state = {"last": -1, "file": None}

    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if not total:
                return
            if d.get("filename") != state["file"]:
                state["file"], state["last"] = d.get("filename"), -1
            pct = int(d["downloaded_bytes"] * 100 / total)
            step = pct // 25 * 25
            if step > state["last"]:
                state["last"] = step
                speed = d.get("speed") or 0
                log(f"{tag} {pct:3d}% của {total / 1048576:.1f}MB @ {speed / 1048576:.1f}MB/s")
        elif d["status"] == "finished":
            state["last"] = -1

    return hook


def download_one(job, args, total):
    tag = f"[{job.index}/{total}]"
    res = Result(job)
    height = job.quality or args.quality
    outdir = os.path.join(args.output, safe_filename(job.folder)) if job.folder else args.output
    os.makedirs(outdir, exist_ok=True)

    name = safe_filename(job.name) if job.name else "%(title).150B [%(id)s]"
    opts = {
        "format": build_format(height, args.video_only),
        # Ưu tiên: độ phân giải > fps > HDR > bitrate
        "format_sort": ["res", "fps", "hdr:12", "vbr", "abr"],
        "merge_output_format": args.container,
        "outtmpl": os.path.join(outdir, name + ".%(ext)s"),
        "noplaylist": not args.playlist,
        "js_runtimes": JS_RUNTIMES,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": args.fragments,
        "continuedl": True,
        "overwrites": False,
        "windowsfilenames": True,
        "logger": QuietLogger(tag),
        "progress_hooks": [make_progress_hook(tag)],
        "noprogress": True,
        "quiet": True,
        "no_warnings": False,
    }
    if args.archive:
        opts["download_archive"] = args.archive
    if args.cookies:
        opts["cookiefile"] = args.cookies
    if args.cookies_from_browser:
        opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    if args.rate_limit:
        opts["ratelimit"] = args.rate_limit
    if args.subs:
        opts.update({"writesubtitles": True, "subtitleslangs": args.subs.split(","), "subtitlesformat": "srt/best"})

    for attempt in range(1, args.retries + 1):
        res.attempts = attempt
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(job.url, download=False)
                if info is None:
                    res.status, res.error = "skipped", "đã có trong archive"
                    log(f"{tag} ⏭  bỏ qua (đã tải trước đó): {job.url}")
                    return res
                entries = list(info["entries"]) if info.get("_type") == "playlist" else [info]
                entries = [e for e in entries if e]
                res.title = info.get("title", "")
                if len(entries) == 1:
                    e = entries[0]
                    res.resolution = f"{e.get('width')}x{e.get('height')}@{e.get('fps')}fps"
                    log(f"{tag} ▶ {res.title[:70]} | {res.resolution} | {e.get('format_id')}")
                else:
                    log(f"{tag} ▶ playlist '{res.title[:60]}' ({len(entries)} video)")

                if ydl.in_download_archive(info):
                    res.status = "skipped"
                    log(f"{tag} ⏭  bỏ qua (archive): {res.title[:70]}")
                    return res

                ydl.process_ie_result(info, download=True)

                paths = []
                for e in entries:
                    for rd in e.get("requested_downloads") or []:
                        p = rd.get("filepath") or rd.get("_filename")
                        if p:
                            paths.append(p)
                res.filepath = " | ".join(paths)
                res.status = "ok"
                log(f"{tag} ✔ xong: {os.path.basename(paths[0]) if paths else res.title}")
                return res
        except yt_dlp.utils.DownloadError as e:
            res.error = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "")
            if is_fatal_error(res.error):
                break
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
        if attempt < args.retries:
            wait = 5 * attempt
            log(f"{tag} ↻ lỗi, thử lại sau {wait}s ({attempt}/{args.retries}): {res.error[:120]}")
            time.sleep(wait)

    res.status = "failed"
    log(f"{tag} ✘ THẤT BẠI: {job.url} — {res.error[:200]}")
    return res


def write_report(results, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["index", "url", "status", "title", "resolution", "filepath", "attempts", "error"])
        for r in sorted(results, key=lambda r: r.job.index):
            w.writerow([r.job.index, r.job.url, r.status, r.title, r.resolution, r.filepath, r.attempts, r.error])


def check_tools():
    from shutil import which
    if not which("ffmpeg"):
        log("⚠ Không tìm thấy ffmpeg — không thể ghép video+audio, chất lượng sẽ bị giới hạn. "
            "Cài: sudo apt install ffmpeg")
    try:
        ver = datetime.strptime(yt_dlp.version.__version__[:10], "%Y.%m.%d")
        if (datetime.now() - ver).days > 60:
            log(f"⚠ yt-dlp {yt_dlp.version.__version__} đã cũ, YouTube có thể lỗi. "
                "Nên chạy: pip install -U yt-dlp")
    except ValueError:
        pass


def parse_rate(value):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMG]?)", value.strip().upper())
    if not m:
        raise argparse.ArgumentTypeError("vd: 5M, 500K")
    return int(float(m.group(1)) * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m.group(2)])


def main():
    p = argparse.ArgumentParser(
        description="Download hàng loạt video YouTube chất lượng cao từ file CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv", help="file CSV chứa danh sách link")
    p.add_argument("-o", "--output", default="downloads", help="thư mục lưu video")
    p.add_argument("-q", "--quality", type=int, default=2160,
                   help="độ phân giải tối đa (2160=4K, 1440, 1080, 720...); 0 = cao nhất có thể")
    p.add_argument("-j", "--jobs", type=int, default=2, help="số video tải song song")
    p.add_argument("--fragments", type=int, default=4, help="số fragment tải song song mỗi video")
    p.add_argument("-c", "--container", default="mp4", choices=["mp4", "mkv", "webm"],
                   help="định dạng file đầu ra")
    p.add_argument("--video-only", action="store_true", help="chỉ tải hình, không có tiếng")
    p.add_argument("--playlist", action="store_true", help="nếu link là playlist thì tải cả playlist")
    p.add_argument("--subs", metavar="LANGS", help="tải phụ đề, vd: vi,en")
    p.add_argument("--retries", type=int, default=3, help="số lần thử lại mỗi video")
    p.add_argument("--rate-limit", type=parse_rate, help="giới hạn tốc độ mỗi luồng, vd 5M")
    p.add_argument("--cookies", help="file cookies.txt (cho video giới hạn tuổi/thành viên)")
    p.add_argument("--cookies-from-browser", metavar="BROWSER", help="lấy cookies từ trình duyệt: chrome, firefox...")
    p.add_argument("--no-archive", action="store_true",
                   help="không dùng archive (mặc định ghi nhớ video đã tải để bỏ qua khi chạy lại)")
    p.add_argument("--report", help="đường dẫn file báo cáo CSV (mặc định trong thư mục output)")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"Không tìm thấy file: {args.csv}")
    os.makedirs(args.output, exist_ok=True)
    args.quality = args.quality or None
    args.archive = None if args.no_archive else os.path.join(args.output, ".downloaded_archive.txt")
    report = args.report or os.path.join(args.output, f"report_{datetime.now():%Y%m%d_%H%M%S}.csv")

    check_tools()
    jobs = read_jobs(args.csv)
    if not jobs:
        sys.exit("Không tìm thấy link nào trong file CSV.")
    log(f"Tìm thấy {len(jobs)} link | chất lượng tối đa: {args.quality or 'cao nhất'}p | "
        f"song song: {args.jobs} | lưu tại: {os.path.abspath(args.output)}")

    results, start = [], time.time()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [pool.submit(download_one, j, args, len(jobs)) for j in jobs]
            for fut in as_completed(futures):
                results.append(fut.result())
    except KeyboardInterrupt:
        log("Đã dừng bởi người dùng (Ctrl+C). Chạy lại lệnh để tải tiếp phần còn lại.")
        os._exit(130)
    finally:
        if results:
            write_report(results, report)

    ok = sum(r.status == "ok" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    failed = [r for r in results if r.status == "failed"]
    mins = (time.time() - start) / 60
    print("\n" + "=" * 60)
    print(f"HOÀN TẤT trong {mins:.1f} phút: {ok} thành công, {skipped} bỏ qua, {len(failed)} lỗi")
    print(f"Báo cáo: {report}")
    if failed:
        print("Các link lỗi:")
        for r in sorted(failed, key=lambda r: r.job.index):
            print(f"  - {r.job.url}: {r.error[:150]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
