import pytest
import torch

from pickleball_track.core.config import LABEL2ID
from pickleball_track.models import model_loader
from pickleball_track.services import inference


def test_audio_window_padding_matches_expected_shape_and_mask():
    waveform = torch.tensor([[0.25, 0.5]])

    segment, padding_mask = inference.get_audio_window(
        waveform,
        start_time=0.0,
        duration=0.00025,
    )

    assert segment.shape == (1, 4)
    assert torch.equal(padding_mask, torch.tensor([[False, False, True, True]]))
    assert torch.equal(segment, torch.tensor([[0.25, 0.5, 0.0, 0.0]]))


def test_sliding_windows_preserve_half_window_overlap():
    assert inference.build_windows(num_frames=64, num_window_frames=32) == [0, 16, 32]


def test_temporal_events_keep_no_event_distinct_from_action_events():
    times = [0.1, 0.2]
    probabilities = [
        [0.05, 0.90, 0.05],
        [0.90, 0.05, 0.05],
    ]

    events = inference.build_temporal_events(
        times,
        probabilities,
        smoothing_radius=0,
    )
    result = inference.build_json_result("sample.mp4", times, probabilities, events)

    assert [event["label"] for event in events] == ["hit"]
    assert result["total_hits"] == 1
    assert result["total_misses"] == 0
    assert [item["label"] for item in result["temporal_predictions"]] == [
        "hit",
        "no_event",
    ]
    assert LABEL2ID == {"no_event": 0, "hit": 1, "miss": 2}


def test_checkpoint_label_mismatch_is_rejected_before_loading(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    monkeypatch.setattr(
        model_loader.torch,
        "load",
        lambda *args, **kwargs: {"label2id": {"hit": 0, "miss": 1, "no_event": 2}},
    )

    with pytest.raises(RuntimeError, match="label mapping"):
        model_loader.load_trained_model(
            str(checkpoint_path), 768, 768, torch.device("cpu")
        )