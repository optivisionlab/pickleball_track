from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import cv2


VIDEO_EXTENSIONS = {".mp4", ".MOV", ".mov"}
LABELS_OF_INTEREST = {"hit", "miss"}
DEFAULT_OUTPUT_JSON = "data_analysis.json"
DEFAULT_OUTPUT_CSV = "data_analysis.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze raw pickleball video/json pairs and generate "
            "train-ready summary statistics."
        )
    )
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing videos and matching JSON files.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to save the summary JSON.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to save the summary CSV.",
    )
    p.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Raise an error if any invalid sample is found.",
    )
    return p.parse_args()


def discover_pairs(data_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    video_paths: List[str] = []
    json_paths: List[str] = []
    missing_json: List[str] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        js = path.with_suffix(".json")
        if js.exists():
            video_paths.append(str(path))
            json_paths.append(str(js))
        else:
            missing_json.append(str(path))

    if not video_paths:
        raise RuntimeError(
            f"No video/json pairs found under {data_dir}. "
            "Please check --data-dir."
        )

    return list(zip(video_paths, json_paths)), missing_json


def parse_video_metadata(video_path: Path) -> Dict[str, str]:
    parts = video_path.stem.split("_")

    # Example: 20_12_2025_Lan_Truot_TraiTay.MOV
    # => parts = [20, 12, 2025, Lan, Truot, TraiTay]
    if len(parts) < 6:
        raise ValueError(
            f"Unexpected filename format for {video_path}; "
            "expected DD_MM_YYYY_USER_ACTION_HAND"
        )

    user_id = parts[3].strip()
    action = parts[4].strip().lower()
    hand = normalize_hand(parts[5])

    return {
        "user_id": user_id,
        "user": user_id,
        "action": action,
        "hand": hand,
    }


def normalize_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if "trai" in value or "left" in value:
        return "left"
    if "phai" in value or "right" in value:
        return "right"
    if value in {"", "unknown"}:
        return "unknown"
    return value


def load_events(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        raise ValueError(f"Invalid events list in {json_path}")

    output: List[Dict[str, Any]] = []
    for item in raw_events:
        try:
            if not isinstance(item, dict):
                continue

            time_value = float(item.get("time", -1))
            label = str(item.get("label", "")).strip().lower()

            if label not in LABELS_OF_INTEREST:
                continue

            if time_value < 0:
                continue

            output.append(
                {
                    "time": time_value,
                    "label": label,
                }
            )
        except Exception:
            continue

    output.sort(key=lambda x: float(cast(float, x["time"])))
    return output


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"Invalid metadata for {video_path}: fps={fps}, frames={frame_count}"
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
    }


def compute_window_count(frame_count: int, frame_window: int = 32, overlap: float = 0.5) -> int:
    if frame_count <= frame_window:
        return 1

    stride = max(1, int(round(frame_window * (1.0 - overlap))))
    last = frame_count - frame_window
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return len(starts)


def summarize_one_video(
    video_path: Path,
    json_path: Path,
    frame_window: int = 32,
    overlap: float = 0.5,
) -> Dict[str, object]:
    meta = get_video_metadata(video_path)
    events = load_events(json_path)

    event_counts = Counter(cast(str, event["label"]) for event in events)
    event_times = [float(cast(float, event["time"])) for event in events]

    gap_values: List[float] = []
    for current, nxt in zip(event_times, event_times[1:]):
        gap_values.append(float(nxt - current))

    user_meta = parse_video_metadata(video_path)

    video_summary: Dict[str, Any] = {
        "video_path": str(video_path),
        "json_path": str(json_path),
        "stem": video_path.stem,
        "user": user_meta["user"],
        "user_id": user_meta["user_id"],
        "action": user_meta["action"],
        "hand": user_meta["hand"],
        "fps": round(meta["fps"], 3),
        "frame_count": int(meta["frame_count"]),
        "duration_sec": round(meta["duration_sec"], 3),
        "event_count": len(events),
        "events_by_label": {
            "hit": int(event_counts.get("hit", 0)),
            "miss": int(event_counts.get("miss", 0)),
        },
        "event_rate_per_sec": round(
            len(events) / max(meta["duration_sec"], 1e-6),
            4,
        ),
        "estimated_windows": compute_window_count(
            meta["frame_count"],
            frame_window=frame_window,
            overlap=overlap,
        ),
        "event_times": [round(t, 3) for t in event_times],
    }

    if event_times:
        video_summary["event_time_min"] = round(min(event_times), 3)
        video_summary["event_time_max"] = round(max(event_times), 3)
        video_summary["avg_event_gap_sec"] = round(
            statistics.fmean(gap_values), 3
        ) if gap_values else 0.0
        video_summary["min_event_gap_sec"] = round(
            min(gap_values), 3
        ) if gap_values else 0.0
    else:
        video_summary["event_time_min"] = None
        video_summary["event_time_max"] = None
        video_summary["avg_event_gap_sec"] = None
        video_summary["min_event_gap_sec"] = None

    return video_summary


def build_summary(data_dir: Path, frame_window: int, overlap: float) -> Dict[str, Any]:
    pairs, missing_json = discover_pairs(data_dir)

    video_summaries: List[Dict[str, Any]] = []
    invalid_samples: List[Dict[str, str]] = []
    empty_event_samples: List[str] = []

    for video_path, json_path in pairs:
        video_path = Path(video_path)
        json_path = Path(json_path)

        try:
            info = summarize_one_video(
                video_path,
                json_path,
                frame_window=frame_window,
                overlap=overlap,
            )
            if info["event_count"] == 0:
                empty_event_samples.append(info["video_path"])
            video_summaries.append(info)
        except Exception as exc:
            invalid_samples.append(
                {
                    "video_path": str(video_path),
                    "json_path": str(json_path),
                    "error": str(exc),
                }
            )

    # Global counts
    users = Counter(item["user_id"] for item in video_summaries)
    actions = Counter(item["action"] for item in video_summaries)
    hands = Counter(item["hand"] for item in video_summaries)
    label_counts = Counter()
    total_events = 0
    total_duration = 0.0
    total_windows = 0

    for item in video_summaries:
        total_events += int(cast(int, item["event_count"]))
        total_duration += float(cast(float, item["duration_sec"]))
        total_windows += int(cast(int, item["estimated_windows"]))
        for label_name in ["hit", "miss"]:
            label_counts[label_name] += int(item["events_by_label"].get(label_name, 0))

    global_summary = {
        "total_videos": len(video_summaries),
        "total_pairs": len(pairs),
        "missing_json_count": len(missing_json),
        "missing_json": missing_json,
        "invalid_samples": invalid_samples,
        "empty_event_samples": empty_event_samples,
        "total_events": total_events,
        "total_duration_sec": round(total_duration, 3),
        "avg_duration_sec": round(
            statistics.fmean(
                [float(cast(float, item["duration_sec"])) for item in video_summaries]
            ),
            3,
        ) if video_summaries else 0.0,
        "estimated_total_windows": total_windows,
        "unique_users": len(users),
        "unique_actions": len(actions),
        "unique_hands": len(hands),
        "label_counts": {
            "hit": int(label_counts.get("hit", 0)),
            "miss": int(label_counts.get("miss", 0)),
        },
        "user_counts": dict(sorted(users.items())),
        "action_counts": dict(sorted(actions.items())),
        "hand_counts": dict(sorted(hands.items())),
    }

    total_label_events = max(
        1,
        global_summary["label_counts"]["hit"] + global_summary["label_counts"]["miss"],
    )
    class_weights = {}
    for label in ["hit", "miss"]:
        count = global_summary["label_counts"].get(label, 0)
        class_weights[label] = round(
            total_label_events / max(count, 1),
            4,
        )

    global_summary["class_weights_for_training"] = class_weights

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "config": {
            "frame_window": frame_window,
            "overlap": overlap,
            "labels_of_interest": sorted(LABELS_OF_INTEREST),
        },
        "global": global_summary,
        "videos": video_summaries,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in payload["videos"]:
        rows.append(
            {
                "video_path": item["video_path"],
                "json_path": item["json_path"],
                "stem": item["stem"],
                "user": item["user"],
                "user_id": item["user_id"],
                "action": item["action"],
                "hand": item["hand"],
                "fps": item["fps"],
                "frame_count": item["frame_count"],
                "duration_sec": item["duration_sec"],
                "event_count": item["event_count"],
                "events_by_label_hit": item["events_by_label"]["hit"],
                "events_by_label_miss": item["events_by_label"]["miss"],
                "event_rate_per_sec": item["event_rate_per_sec"],
                "estimated_windows": item["estimated_windows"],
                "event_time_min": item.get("event_time_min"),
                "event_time_max": item.get("event_time_max"),
                "avg_event_gap_sec": item.get("avg_event_gap_sec"),
                "min_event_gap_sec": item.get("min_event_gap_sec"),
            }
        )

    fieldnames = [
        "video_path",
        "json_path",
        "stem",
        "user",
        "user_id",
        "action",
        "hand",
        "fps",
        "frame_count",
        "duration_sec",
        "event_count",
        "events_by_label_hit",
        "events_by_label_miss",
        "event_rate_per_sec",
        "estimated_windows",
        "event_time_min",
        "event_time_max",
        "avg_event_gap_sec",
        "min_event_gap_sec",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: Dict[str, Any]) -> None:
    global_info = summary["global"]
    print("=" * 90)
    print("DATA ANALYSIS SUMMARY")
    print("=" * 90)
    print(f"data_dir        : {summary['data_dir']}")
    print(f"generated_at    : {summary['generated_at']}")
    print(f"videos          : {global_info['total_videos']}")
    print(f"pairs           : {global_info['total_pairs']}")
    print(f"missing JSON    : {global_info['missing_json_count']}")
    print(f"empty events    : {len(global_info['empty_event_samples'])}")
    print(f"invalid samples : {len(global_info['invalid_samples'])}")
    print(f"total events    : {global_info['total_events']}")
    print(f"total duration  : {global_info['total_duration_sec']} sec")
    print(f"avg duration    : {global_info['avg_duration_sec']} sec")
    print(f"estimated windows: {global_info['estimated_total_windows']}")
    print()
    print("Label counts:")
    for label in ["hit", "miss"]:
        print(f"  - {label:>4} : {global_info['label_counts'].get(label, 0)}")
    print()
    print("Class weights for training:")
    for label in ["hit", "miss"]:
        print(
            f"  - {label:>4} : {global_info['class_weights_for_training'].get(label, 0.0)}"
        )
    print()
    print("User counts:")
    for user, count in sorted(global_info["user_counts"].items()):
        print(f"  - {user}: {count}")
    print()
    print("Action counts:")
    for action, count in sorted(global_info["action_counts"].items()):
        print(f"  - {action}: {count}")
    print()
    print("Hand counts:")
    for hand, count in sorted(global_info["hand_counts"].items()):
        print(f"  - {hand}: {count}")

    if global_info["empty_event_samples"]:
        print()
        print("Videos without events:")
        for p in global_info["empty_event_samples"]:
            print(f"  - {p}")

    if global_info["invalid_samples"]:
        print()
        print("Invalid samples:")
        for item in global_info["invalid_samples"][:20]:
            print(f"  - {item['video_path']} | error={item['error']}")

    print("=" * 90)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    summary = build_summary(
        data_dir=data_dir,
        frame_window=32,
        overlap=0.5,
    )

    save_json(output_json, summary)
    save_csv(output_csv, summary)
    print_summary(summary)

    if args.fail_on_invalid and summary["global"]["invalid_samples"]:
        raise RuntimeError(
            "Found invalid samples. Re-check dataset before training."
        )


if __name__ == "__main__":
    main()
