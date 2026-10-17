"""Turn per-window token probabilities into a timeline and hit/miss events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.config import EVENT_MIN_GAP, LABEL2ID, NUM_TEMPORAL_TOKENS, NUM_VIDEO_FRAMES, InferenceParams

ID2LABEL = {int(idx): name for name, idx in LABEL2ID.items()}
ACTION_LABELS = frozenset({"hit", "miss"})


@dataclass(frozen=True)
class WindowPrediction:
    start_frame: int
    probs: torch.Tensor  # (NUM_TEMPORAL_TOKENS, num_classes) softmax probabilities


def label_from_id(label_id: int) -> str:
    try:
        return ID2LABEL[int(label_id)]
    except KeyError:
        raise RuntimeError(f"Unknown class id: {label_id}") from None


def aggregate_predictions(
    window_predictions: Sequence[WindowPrediction], fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average predictions from overlapping windows at the same time."""
    votes = defaultdict(list)
    frames_per_token = NUM_VIDEO_FRAMES // NUM_TEMPORAL_TOKENS

    for item in window_predictions:
        start_frame = int(item.start_frame)
        for token_idx in range(NUM_TEMPORAL_TOKENS):
            center_frame = (
                start_frame + token_idx * frames_per_token
                + frames_per_token / 2.0
            )
            key = round(float(center_frame / fps), 4)
            votes[key].append(item.probs[token_idx])

    times = sorted(votes)
    if not times:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0, len(LABEL2ID)), dtype=np.float32),
        )

    probs = [torch.stack(votes[t], dim=0).mean(dim=0) for t in times]
    return np.asarray(times, dtype=np.float32), torch.stack(probs).numpy()


def smooth_probabilities(probs: Any, radius: int = 1) -> np.ndarray:
    """Light moving-average smoothing on class probabilities."""
    probs = np.asarray(probs, dtype=np.float32)
    if radius <= 0 or len(probs) <= 1:
        return probs.copy()
    out = np.empty_like(probs)
    for i in range(len(probs)):
        lo = max(0, i - radius)
        hi = min(len(probs), i + radius + 1)
        out[i] = probs[lo:hi].mean(axis=0)
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)


def _event_from_indices(
    times: Sequence[float], probs: np.ndarray, label_id: int, indices: list[int]
) -> dict[str, Any] | None:
    if not indices:
        return None
    vals = probs[indices, label_id]
    best_idx = indices[int(np.argmax(vals))]
    return {
        "time": round(float(times[best_idx]), 4),
        "label": label_from_id(label_id),
        "confidence": round(float(probs[best_idx, label_id]), 6),
        "start_time": round(float(times[indices[0]]), 4),
        "end_time": round(float(times[indices[-1]]), 4),
    }


def merge_close_action_events(
    events: Sequence[dict[str, Any]], max_distance: float = EVENT_MIN_GAP
) -> list[dict[str, Any]]:
    """Merge only same-class events closer than max_distance."""
    if not events:
        return []
    merged: list[dict[str, Any]] = []
    for e in sorted((dict(x) for x in events), key=lambda x: x["time"]):
        if (
            merged
            and e["label"] in ACTION_LABELS
            and e["label"] == merged[-1]["label"]
            and e["time"] - merged[-1]["time"] <= max_distance
        ):
            prev = merged[-1]
            best = dict(e if e["confidence"] > prev["confidence"] else prev)
            best["start_time"] = round(min(prev["start_time"], e["start_time"]), 4)
            best["end_time"] = round(max(prev["end_time"], e["end_time"]), 4)
            merged[-1] = best
        else:
            merged.append(e)
    return merged


def build_temporal_events(
    times: Sequence[float],
    probs: Any,
    params: InferenceParams = InferenceParams(),
) -> list[dict[str, Any]]:
    """Convert temporal probabilities to physical Hit/Miss events."""
    if len(times) == 0:
        return []
    no_id, hit_id, miss_id = LABEL2ID["no_event"], LABEL2ID["hit"], LABEL2ID["miss"]
    sm = smooth_probabilities(probs, params.smooth_radius)
    labels = []
    for p in sm:
        h, m, n = float(p[hit_id]), float(p[miss_id]), float(p[no_id])
        if h >= params.hit_threshold and h >= m and h >= n:
            labels.append(hit_id)
        elif m >= params.miss_threshold and m >= h and m >= n:
            labels.append(miss_id)
        else:
            labels.append(no_id)

    # Fill a short no-event gap only when the same action is on both sides.
    i = 0
    while i < len(labels):
        if labels[i] not in (hit_id, miss_id):
            i += 1
            continue
        action = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == no_id:
            j += 1
        if j < len(labels) and labels[j] == action and times[j] - times[i] <= params.event_min_gap:
            labels[i + 1:j] = [action] * (j - i - 1)
            i = j
        else:
            i += 1

    # Group consecutive runs of the same action label into events.
    raw = []
    cur: int | None = None
    idxs: list[int] = []
    for idx, label in enumerate(labels):
        if label in (hit_id, miss_id):
            if cur is None:
                cur, idxs = label, [idx]
            elif label == cur:
                idxs.append(idx)
            else:
                e = _event_from_indices(times, sm, cur, idxs)
                if e:
                    raw.append(e)
                cur, idxs = label, [idx]
        elif cur is not None:
            e = _event_from_indices(times, sm, cur, idxs)
            if e:
                raw.append(e)
            cur, idxs = None, []
    if cur is not None:
        e = _event_from_indices(times, sm, cur, idxs)
        if e:
            raw.append(e)
    return merge_close_action_events(raw, params.event_min_gap)


def build_json_result(
    video_path: str | Path,
    times: Sequence[float],
    probs: Any,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    hit_events = [e for e in events if e["label"] == "hit"]
    miss_events = [e for e in events if e["label"] == "miss"]
    no_event_segments = [e for e in events if e["label"] == "no_event"]
    temporal_predictions = []
    for time, p in zip(times, probs, strict=True):
        label_id = int(np.argmax(p))
        temporal_predictions.append({
            "time": round(float(time), 4),
            "label": label_from_id(label_id),
            "confidence": round(float(p[label_id]), 6),
            "probabilities": {label_from_id(i): round(float(p[i]), 6) for i in range(len(p))},
        })
    return {
        "video": str(video_path),
        "total_hits": len(hit_events),
        "total_misses": len(miss_events),
        "events": events,
        "hit_events": hit_events,
        "miss_events": miss_events,
        "no_event_segments": no_event_segments,
        "temporal_predictions": temporal_predictions,
    }
