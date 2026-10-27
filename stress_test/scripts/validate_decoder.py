"""So sánh bộ giải mã opencv và ffmpeg trên video có nhãn (chạy trong container AI).

python validate_decoder.py <thư mục video+json> <tên video> ...
In ra từng video: số hit/miss, điểm, thời gian theo từng decoder và độ khớp sự kiện với nhãn.
"""
import json
import sys
import time
from pathlib import Path

from pickleball_track.core.config import get_settings
from pickleball_track.models.model_loader import load_models
from pickleball_track.services.inference_service import VideoInferenceService
from pickleball_track.services.video_jobs import score_counts

TOLERANCE = 0.5  # giây


def match(pred, truth):
    """Ghép sự kiện dự đoán với nhãn cùng loại trong ±TOLERANCE; trả (tp, fp, fn)."""
    used, tp = set(), 0
    for p in pred:
        best = None
        for i, t in enumerate(truth):
            if i in used or t["label"] != p["label"]:
                continue
            gap = abs(t["time"] - p["time"])
            if gap <= TOLERANCE and (best is None or gap < best[0]):
                best = (gap, i)
        if best:
            used.add(best[1])
            tp += 1
    return tp, len(pred) - tp, len(truth) - tp


def events_of(result):
    out = []
    for key, label in (("hit_events", "hit"), ("miss_events", "miss")):
        for e in result.get(key) or []:
            t = e.get("time", e.get("peak_time", e.get("start_time")))
            if t is None and "start" in e:
                t = (e["start"] + e.get("end", e["start"])) / 2
            out.append({"time": float(t), "label": label})
    return out


def main():
    root = Path(sys.argv[1])
    names = sys.argv[2:]
    settings = get_settings()
    bundle = load_models(settings)
    services = {d: VideoInferenceService(bundle, settings.inference, slots=1, decoder=d) for d in ("opencv", "ffmpeg")}
    totals = {d: {"tp": 0, "fp": 0, "fn": 0, "sec": 0.0} for d in services}
    same_counts = same_score = 0
    first = True
    for name in names:
        video = root / name
        truth = json.loads(video.with_suffix(".json").read_text())["events"]
        truth = [{"time": float(e["time"]), "label": e["label"]} for e in truth if e["label"] in ("hit", "miss")]
        row = {}
        for decoder, service in services.items():
            started = time.perf_counter()
            result = service.predict(video)
            seconds = time.perf_counter() - started
            score = score_counts(int(result["total_hits"]), int(result["total_misses"]))
            tp, fp, fn = match(events_of(result), truth)
            if first:
                print("keys:", sorted(result)[:12], "| event sample:", (result.get("hit_events") or [None])[0], flush=True)
            first = False
            t = totals[decoder]
            t["tp"] += tp; t["fp"] += fp; t["fn"] += fn; t["sec"] += seconds
            row[decoder] = (score.hit_count, score.miss_count, score.score, tp, fp, fn, seconds)
        o, f = row["opencv"], row["ffmpeg"]
        same_counts += o[:2] == f[:2]
        same_score += o[2] == f[2]
        t_hit = sum(e["label"] == "hit" for e in truth)
        print(f"{name:42} label hit/miss={t_hit}/{len(truth) - t_hit:<3} | opencv {o[0]}/{o[1]} score={o[2]:5} tp/fp/fn={o[3]}/{o[4]}/{o[5]} {o[6]:5.1f}s"
              f" | ffmpeg {f[0]}/{f[1]} score={f[2]:5} tp/fp/fn={f[3]}/{f[4]}/{f[5]} {f[6]:5.1f}s", flush=True)
    n = len(names)
    print(f"\nSame hit/miss counts: {same_counts}/{n}, same score: {same_score}/{n}")
    for d, t in totals.items():
        precision = t["tp"] / max(1, t["tp"] + t["fp"])
        recall = t["tp"] / max(1, t["tp"] + t["fn"])
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        print(f"{d:7} events tp={t['tp']} fp={t['fp']} fn={t['fn']} precision={precision:.3f} recall={recall:.3f} F1={f1:.3f} | total {t['sec']:.0f}s, avg {t['sec'] / n:.1f}s/video")
    for s in services.values():
        s.close()
    bundle.close()


if __name__ == "__main__":
    main()
