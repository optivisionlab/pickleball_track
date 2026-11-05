"""Analyse UCF-101 before training: files, official splits, durations, fps, resolution, audio.

Writes
    <out>/ucf101_videos.csv   one row per video (class, group, duration, fps, audio, loudness, splits)
    <out>/ucf101_analysis.json summary used to choose the training parameters

Usage
    python analyze_ucf101.py --video-root /data/.../UCF-101 --split-dir /data/.../ucfTrainTestlist
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

NAME_RE = re.compile(r"v_(?P<cls>[A-Za-z]+)_g(?P<group>\d+)_c(?P<clip>\d+)\.avi$")
SILENT_RMS = 1e-3  # below this the track is treated as silent (about -60 dBFS)


def probe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        info = json.loads(subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        return {"error": str(exc)}
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    num, _, den = video.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    row = {
        "duration": duration,
        "fps": round(fps, 3),
        "frames": int(video.get("nb_frames") or round(duration * fps)),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "vcodec": video.get("codec_name", ""),
        "has_audio_stream": audio is not None,
        "acodec": audio.get("codec_name", "") if audio else "",
        "audio_sr": int(audio.get("sample_rate") or 0) if audio else 0,
        "audio_rms": 0.0,
        "audio_peak": 0.0,
    }
    if audio is not None:
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
               "-ar", "16000", "-f", "f32le", "-"]
        try:
            wave = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True,
                                                timeout=60).stdout, dtype=np.float32)
            if wave.size:
                row["audio_rms"] = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
                row["audio_peak"] = float(np.abs(wave).max())
        except (subprocess.SubprocessError, OSError):
            pass
    return row


def read_splits(split_dir: Path):
    class_ind = {}
    for line in (split_dir / "classInd.txt").read_text().split("\n"):
        if line.strip():
            idx, name = line.split()
            class_ind[name] = int(idx) - 1
    splits = {}
    for k in (1, 2, 3):
        for part in ("train", "test"):
            names = []
            for line in (split_dir / f"{part}list0{k}.txt").read_text().split("\n"):
                if line.strip():
                    names.append(line.split()[0].split("/")[-1].lower())
            splits[(k, part)] = names
    return class_ind, splits


def stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    q = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    return {"mean": round(float(a.mean()), 3), "min": round(q[0], 3), "p5": round(q[1], 3),
            "p25": round(q[2], 3), "median": round(q[3], 3), "p75": round(q[4], 3),
            "p95": round(q[5], 3), "max": round(q[6], 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "analysis"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=32,
                        help="Frames the feature extractor samples per clip (for coverage stats).")
    args = parser.parse_args()

    root, out = Path(args.video_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    class_ind, splits = read_splits(Path(args.split_dir))
    class_by_lower = {name.lower(): name for name in class_ind}

    files = sorted(root.rglob("*.avi"))
    rows, bad_names = [], []
    for path in files:
        m = NAME_RE.search(path.name)
        if not m:
            bad_names.append(path.name)
            continue
        # The folder name is the class (file names spell HandStandPushups, classInd HandstandPushups).
        cls = class_by_lower.get(path.parent.name.lower()) or class_by_lower.get(m["cls"].lower())
        rows.append({"file": str(path.relative_to(root)), "name": path.name.lower(),
                     "class": cls, "label": class_ind.get(cls, -1),
                     "group": int(m["group"]), "clip": int(m["clip"])})

    with ThreadPoolExecutor(args.workers) as pool:
        for row, info in zip(rows, tqdm(pool.map(lambda r: probe(root / r["file"]), rows),
                                         total=len(rows), desc="ffprobe")):
            row.update(info)

    by_name = {r["name"]: r for r in rows}
    split_report = {}
    for k in (1, 2, 3):
        train, test = splits[(k, "train")], splits[(k, "test")]
        for part, names in (("train", train), ("test", test)):
            for n in names:
                if n in by_name:
                    by_name[n][f"split{k}"] = part
        train_groups = {by_name[n]["group"] for n in train if n in by_name}
        test_groups = {by_name[n]["group"] for n in test if n in by_name}
        tr = [by_name[n] for n in train if n in by_name]
        te = [by_name[n] for n in test if n in by_name]
        split_report[f"split{k}"] = {
            "train": len(train), "test": len(test),
            "train_missing_files": sum(n not in by_name for n in train),
            "test_missing_files": sum(n not in by_name for n in test),
            "train_groups": sorted(train_groups), "test_groups": sorted(test_groups),
            "group_overlap": sorted(train_groups & test_groups),
            "train_audible_frac": round(np.mean([r["audio_rms"] > SILENT_RMS for r in tr]), 4),
            "test_audible_frac": round(np.mean([r["audio_rms"] > SILENT_RMS for r in te]), 4),
            "test_clips_per_class": stats(list(Counter(r["class"] for r in te).values())),
            "train_clips_per_class": stats(list(Counter(r["class"] for r in tr).values())),
        }

    fieldnames = ["file", "class", "label", "group", "clip", "duration", "fps", "frames", "width",
                  "height", "vcodec", "has_audio_stream", "acodec", "audio_sr", "audio_rms",
                  "audio_peak", "split1", "split2", "split3", "error"]
    with (out / "ucf101_videos.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if "error" not in r]
    per_class = defaultdict(list)
    for r in ok:
        per_class[r["class"]].append(r)
    class_table = {}
    for cls in sorted(per_class, key=lambda c: class_ind[c]):
        rs = per_class[cls]
        class_table[cls] = {
            "label": class_ind[cls], "clips": len(rs), "groups": len({r["group"] for r in rs}),
            "duration_median": round(float(np.median([r["duration"] for r in rs])), 2),
            "audio_stream_frac": round(np.mean([r["has_audio_stream"] for r in rs]), 3),
            "audible_frac": round(np.mean([r["audio_rms"] > SILENT_RMS for r in rs]), 3),
        }
    audible_classes = [c for c, v in class_table.items() if v["audible_frac"] >= 0.5]
    mixed_classes = [c for c, v in class_table.items() if 0 < v["audible_frac"] < 1]

    frames = np.array([r["frames"] for r in ok])
    summary = {
        "video_root": str(root),
        "videos_on_disk": len(files),
        "unparsed_names": bad_names[:20],
        "probe_errors": [r["file"] for r in rows if "error" in r][:20],
        "classes": len(class_ind),
        "classes_on_disk": len(per_class),
        "groups": len({r["group"] for r in ok}),
        "clips_per_class": stats([v["clips"] for v in class_table.values()]),
        "clips_per_class_group": stats(list(Counter((r["class"], r["group"]) for r in ok).values())),
        "duration_sec": stats([r["duration"] for r in ok]),
        "frames": stats(frames),
        f"clips_shorter_than_{args.num_frames}_frames": int((frames < args.num_frames).sum()),
        f"sampling_stride_for_{args.num_frames}_frames": stats(frames / args.num_frames),
        "fps": dict(Counter(r["fps"] for r in ok).most_common(10)),
        "resolution": {f"{w}x{h}": n for (w, h), n in
                       Counter((r["width"], r["height"]) for r in ok).most_common(10)},
        "video_codec": dict(Counter(r["vcodec"] for r in ok)),
        "audio": {
            "with_audio_stream": int(sum(r["has_audio_stream"] for r in ok)),
            "audible(rms>%g)" % SILENT_RMS: int(sum(r["audio_rms"] > SILENT_RMS for r in ok)),
            "audio_codec": dict(Counter(r["acodec"] for r in ok if r["has_audio_stream"])),
            "sample_rate": dict(Counter(r["audio_sr"] for r in ok if r["has_audio_stream"])),
            "rms_of_audible": stats([r["audio_rms"] for r in ok if r["audio_rms"] > SILENT_RMS]),
            "classes_mostly_audible": len(audible_classes),
            "classes_mostly_audible_list": audible_classes,
            "classes_mixed_audio": mixed_classes,
        },
        "splits": split_report,
        "per_class": class_table,
    }
    with (out / "ucf101_analysis.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    short = {k: v for k, v in summary.items() if k not in ("per_class", "splits")}
    short["audio"] = {k: v for k, v in summary["audio"].items() if not k.endswith("_list")}
    print(json.dumps(short, indent=2))
    for name, s in split_report.items():
        print(name, {k: v for k, v in s.items() if not k.endswith("_groups")},
              "test groups", s["test_groups"])
    print(f"\nWrote {out / 'ucf101_videos.csv'} and {out / 'ucf101_analysis.json'}")


if __name__ == "__main__":
    main()
