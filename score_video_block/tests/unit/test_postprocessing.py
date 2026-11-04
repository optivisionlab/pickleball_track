import numpy as np
import pytest
import torch

from pickleball_track.core.config import NUM_TEMPORAL_TOKENS, InferenceParams
from pickleball_track.pipelines.postprocessing import (
    WindowPrediction,
    aggregate_predictions,
    build_json_result,
    build_temporal_events,
    token_time,
)

NO, HIT, MISS = [0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]
# The decoder used in training evaluation: argmax tokens, runs of >= 3.
TRAINING = InferenceParams(event_threshold=None, min_event_tokens=3)


def _times(n):
    return [token_time(k) for k in range(n)]


def test_token_times_are_centres_of_two_sampled_frames():
    assert token_time(0) == pytest.approx(0.5 / 30)
    assert token_time(3) == pytest.approx(6.5 / 30)


def test_runs_of_at_least_three_event_tokens_become_events():
    probs = [NO, HIT, HIT, HIT, NO, MISS, MISS, NO, NO]

    events = build_temporal_events(_times(len(probs)), probs, TRAINING)

    # The two-token miss run is dropped (min_event_tokens = 3).
    assert events == [{
        "time": round(token_time(2), 4), "label": "hit", "confidence": 0.8,
        "start_time": round(token_time(1), 4), "end_time": round(token_time(3), 4),
    }]


def test_event_class_is_decided_by_summed_probability_over_the_run():
    probs = [MISS, MISS, HIT, HIT, HIT]

    events = build_temporal_events(_times(5), probs, TRAINING)

    # hit: 0.2+0.2+0.8*3 = 2.8 > miss: 0.7*2+0.1*3 = 1.7
    assert [e["label"] for e in events] == ["hit"]
    assert events[0]["time"] == round(token_time(2), 4)  # centre of tokens 0..4


def test_even_length_run_is_placed_between_its_middle_tokens():
    events = build_temporal_events(_times(4), [HIT] * 4, TRAINING)

    assert events[0]["time"] == round(token_time(1.5), 4)


def test_peak_places_the_event_at_the_most_confident_token():
    probs = [[0.4, 0.5, 0.1], [0.05, 0.9, 0.05], [0.3, 0.6, 0.1], [0.4, 0.5, 0.1]]

    events = build_temporal_events(_times(4), probs, InferenceParams(event_threshold=None, min_event_tokens=3, peak=True))

    assert events[0]["time"] == round(token_time(1), 4)


def test_threshold_uses_event_probability_instead_of_argmax():
    # P(event) = 0.55 per token, but no_event is the argmax.
    probs = [[0.45, 0.3, 0.25]] * 4

    assert build_temporal_events(_times(4), probs, TRAINING) == []
    events = build_temporal_events(_times(4), probs, InferenceParams(event_threshold=0.5, min_event_tokens=3))
    assert [e["label"] for e in events] == ["hit"]


def test_smoothing_joins_a_run_split_by_one_token():
    probs = [[0.3, 0.7, 0.0]] * 2 + [[0.6, 0.4, 0.0]] + [[0.3, 0.7, 0.0]] * 2

    assert build_temporal_events(_times(5), probs, InferenceParams(event_threshold=0.5, min_event_tokens=3)) == []
    joined = build_temporal_events(
        _times(5), probs, InferenceParams(event_threshold=0.5, min_event_tokens=3, smooth_tokens=3)
    )
    assert len(joined) == 1


def test_json_result_lists_events_and_token_predictions():
    probs = [HIT] * 3 + [NO]
    events = build_temporal_events(_times(4), probs, TRAINING)

    result = build_json_result("sample.mp4", _times(4), probs, events)

    assert (result["total_hits"], result["total_misses"]) == (1, 0)
    assert [p["label"] for p in result["temporal_predictions"]] == ["hit"] * 3 + ["no_event"]
    assert result["temporal_predictions"][3]["probabilities"] == {"no_event": 0.9, "hit": 0.05, "miss": 0.05}


def _window(start, column, valid=None):
    probs = torch.zeros(NUM_TEMPORAL_TOKENS, 3)
    probs[:, column] = 1.0
    return WindowPrediction(start, probs, None if valid is None else torch.tensor(valid))


def test_overlapping_windows_are_averaged_per_token():
    times, probs = aggregate_predictions([_window(0, 0), _window(16, 1)], num_sampled=48)

    # 48 sampled frames -> 24 tokens; the windows overlap on tokens 8..15.
    assert len(times) == 24
    assert times[0] == pytest.approx(token_time(0))
    np.testing.assert_allclose(probs[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(probs[8], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(probs[-1], [0.0, 1.0, 0.0])


def test_invalid_and_uncovered_tokens_count_as_no_event():
    valid = [True] * 4 + [False] * 12

    _, probs = aggregate_predictions([_window(0, 2, valid)], num_sampled=40)

    assert len(probs) == 20
    np.testing.assert_allclose(probs[3], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(probs[4:], np.tile([1.0, 0.0, 0.0], (16, 1)))


def test_aggregate_of_no_windows_is_empty():
    times, probs = aggregate_predictions([], num_sampled=30)

    assert times.shape == (0,)
    assert probs.shape == (0, 3)



def test_default_decoder_needs_half_a_second_of_event_probability():
    # Tuned default: P(event) > 0.4 for at least 8 tokens (~0.53 s).
    short = [NO] + [[0.5, 0.3, 0.2]] * 7 + [NO]
    long = [NO] + [[0.5, 0.3, 0.2]] * 8 + [NO]

    assert build_temporal_events(_times(len(short)), short) == []
    assert [e["label"] for e in build_temporal_events(_times(len(long)), long)] == ["hit"]
