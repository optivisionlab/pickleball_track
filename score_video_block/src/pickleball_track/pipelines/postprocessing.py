"""Turn per-window token probabilities into a timeline and hit/miss events.

Both steps follow the training evaluation (baseline/pickleball_common.py:
``stitch_timelines`` and ``decode_events``), so served events are the ones the
reported event macro-F1 was measured on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.config import FRAMES_PER_TOKEN, LABEL2ID, TARGET_FPS, InferenceParams

ID2LABEL = {int(idx): name for name, idx in LABEL2ID.items()}
NO_EVENT = LABEL2ID["no_event"]


@dataclass(frozen=True)
class WindowPrediction:
    start_frame: int  # in sampled frames
    probs: torch.Tensor  # (NUM_TEMPORAL_TOKENS, num_classes) softmax probabilities
    token_valid: torch.Tensor | None = None  # (NUM_TEMPORAL_TOKENS,) bool; None = all valid


def label_from_id(label_id: int) -> str:
    try:
        return ID2LABEL[int(label_id)]
    except KeyError:
        raise RuntimeError(f"Unknown class id: {label_id}") from None


def token_time(token: float) -> float:
    """Centre time (s) of token ``token`` of the video's token grid (may be fractional)."""
    return (token * FRAMES_PER_TOKEN + (FRAMES_PER_TOKEN - 1) / 2) / TARGET_FPS


def aggregate_predictions(
    window_predictions: Sequence[WindowPrediction], num_sampled: int
) -> tuple[np.ndarray, np.ndarray]:
    """Average overlapping windows into one probability per token of the video.

    Token ``k`` covers sampled frames ``2k, 2k + 1``; only valid tokens vote and a
    token no window covers validly counts as no_event.
    """
    size = math.ceil(num_sampled / FRAMES_PER_TOKEN)
    if not window_predictions or size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0, len(LABEL2ID)), dtype=np.float32)
    acc = np.zeros((size, len(LABEL2ID)), dtype=np.float64)
    count = np.zeros(size, dtype=np.int64)
    for item in window_predictions:
        probs = item.probs.numpy()
        slots = item.start_frame // FRAMES_PER_TOKEN + np.arange(len(probs))
        ok = slots < size
        if item.token_valid is not None:
            ok &= item.token_valid.numpy()
        np.add.at(acc, slots[ok], probs[ok])
        np.add.at(count, slots[ok], 1)
    timeline = np.zeros_like(acc)
    timeline[:, NO_EVENT] = 1.0
    covered = count > 0
    timeline[covered] = acc[covered] / count[covered, None]
    times = np.array([token_time(k) for k in range(size)], dtype=np.float32)
    return times, timeline.astype(np.float32)


def event_runs(probs: np.ndarray, params: InferenceParams) -> list[tuple[int, int, int, float]]:
    """Runs of event tokens -> (first token, end token, label id, event token), as in training.

    A token is an event when its argmax is hit/miss (``event_threshold`` None) or when
    the smoothed P(event) = 1 - P(no_event) exceeds the threshold. Runs shorter than
    ``min_event_tokens`` are dropped; the class is hit or miss by summed probability.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if len(probs) == 0:
        return []
    p_event = 1.0 - probs[:, NO_EVENT]
    if params.smooth_tokens > 1:
        p_event = np.convolve(p_event, np.ones(params.smooth_tokens) / params.smooth_tokens, mode="same")
    if params.event_threshold is None:
        mask = probs.argmax(-1) != NO_EVENT
    else:
        mask = p_event > params.event_threshold
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    runs = []
    for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if j - i < params.min_event_tokens:
            continue
        label = 1 + int(probs[i:j, 1:].sum(0).argmax())
        token = i + int(p_event[i:j].argmax()) if params.peak else (i + j - 1) / 2
        runs.append((int(i), int(j), label, float(token)))
    return runs


def build_temporal_events(
    times: Sequence[float],
    probs: Any,
    params: InferenceParams = InferenceParams(),
) -> list[dict[str, Any]]:
    """Hit/miss events of a token timeline (``times`` from ``aggregate_predictions``)."""
    probs = np.asarray(probs, dtype=np.float32)
    events = []
    for first, end, label, token in event_runs(probs, params):
        events.append({
            "time": round(token_time(token), 4),
            "label": label_from_id(label),
            # Mean probability of the event's class over its run.
            "confidence": round(float(probs[first:end, label].mean()), 6),
            "start_time": round(float(times[first]), 4),
            "end_time": round(float(times[end - 1]), 4),
        })
    return events


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
