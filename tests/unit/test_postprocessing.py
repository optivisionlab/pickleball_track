import numpy as np
import pytest
import torch

from pickleball_track.core.config import NUM_TEMPORAL_TOKENS, InferenceParams
from pickleball_track.pipelines.postprocessing import (
    WindowPrediction,
    aggregate_predictions,
    build_json_result,
    build_temporal_events,
    merge_close_action_events,
    smooth_probabilities,
)

NO_SMOOTHING = InferenceParams(smooth_radius=0)


def test_temporal_events_keep_no_event_distinct_from_action_events():
    times = [0.1, 0.2]
    probabilities = [[0.05, 0.90, 0.05], [0.90, 0.05, 0.05]]

    events = build_temporal_events(times, probabilities, NO_SMOOTHING)
    result = build_json_result("sample.mp4", times, probabilities, events)

    assert [event["label"] for event in events] == ["hit"]
    assert result["total_hits"] == 1
    assert result["total_misses"] == 0
    assert [item["label"] for item in result["temporal_predictions"]] == ["hit", "no_event"]
    assert result["temporal_predictions"][0]["probabilities"] == {
        "no_event": 0.05, "hit": 0.9, "miss": 0.05,
    }


def test_only_no_event_yields_no_events():
    times = [0.1, 0.2, 0.3]
    probabilities = [[0.9, 0.05, 0.05]] * 3

    assert build_temporal_events(times, probabilities, NO_SMOOTHING) == []


def test_short_gap_between_same_action_is_filled():
    times = [0.0, 0.1, 0.2]
    probabilities = [[0.1, 0.85, 0.05], [0.8, 0.1, 0.1], [0.05, 0.9, 0.05]]

    events = build_temporal_events(times, probabilities, NO_SMOOTHING)

    assert events == [
        {"time": 0.2, "label": "hit", "confidence": 0.9, "start_time": 0.0, "end_time": 0.2}
    ]


def test_hit_followed_by_miss_are_separate_events():
    times = [0.0, 0.1, 1.0]
    probabilities = [[0.1, 0.85, 0.05], [0.8, 0.1, 0.1], [0.05, 0.1, 0.85]]

    events = build_temporal_events(times, probabilities, NO_SMOOTHING)

    assert [e["label"] for e in events] == ["hit", "miss"]


def test_default_thresholds_come_from_config():
    assert InferenceParams() == InferenceParams(
        hit_threshold=0.8, miss_threshold=0.8, event_min_gap=0.5, smooth_radius=1, overlap=0.5
    )


def test_thresholds_are_respected():
    times = [0.0]
    probabilities = [[0.25, 0.7, 0.05]]

    assert build_temporal_events(times, probabilities, NO_SMOOTHING) == []
    lenient = InferenceParams(hit_threshold=0.6, smooth_radius=0)
    assert [e["label"] for e in build_temporal_events(times, probabilities, lenient)] == ["hit"]


def test_merge_keeps_most_confident_and_widens_range():
    events = [
        {"time": 1.0, "label": "miss", "confidence": 0.6, "start_time": 0.9, "end_time": 1.0},
        {"time": 1.2, "label": "miss", "confidence": 0.9, "start_time": 1.1, "end_time": 1.3},
    ]

    assert merge_close_action_events(events, max_distance=0.35) == [
        {"time": 1.2, "label": "miss", "confidence": 0.9, "start_time": 0.9, "end_time": 1.3}
    ]


def test_smoothing_is_a_normalized_moving_average():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    smoothed = smooth_probabilities(probs, radius=1)

    np.testing.assert_allclose(smoothed[1], [1 / 3, 1 / 3, 1 / 3], rtol=1e-6)
    np.testing.assert_allclose(smoothed[0], [0.5, 0.5, 0.0], rtol=1e-6)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(smooth_probabilities(probs, radius=0), probs)


def test_overlapping_windows_are_averaged_per_timestamp():
    first = torch.zeros(NUM_TEMPORAL_TOKENS, 3)
    first[:, 0] = 1.0
    second = torch.zeros(NUM_TEMPORAL_TOKENS, 3)
    second[:, 1] = 1.0

    times, probs = aggregate_predictions(
        [WindowPrediction(0, first), WindowPrediction(16, second)], fps=10.0
    )

    # Tokens cover 2 frames; windows overlap by 8 tokens.
    assert len(times) == 24
    assert times[0] == pytest.approx(0.1)
    np.testing.assert_allclose(probs[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(probs[8], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(probs[-1], [0.0, 1.0, 0.0])


def test_aggregate_of_no_windows_is_empty():
    times, probs = aggregate_predictions([], fps=30.0)

    assert times.shape == (0,)
    assert probs.shape == (0, 3)
