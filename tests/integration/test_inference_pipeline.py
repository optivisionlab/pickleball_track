import math

import pytest

from pickleball_track.services.inference_service import VideoInferenceService


def test_full_pipeline_on_real_video_with_controlled_models(tiny_models, clip_path):
    result = VideoInferenceService(tiny_models).predict(clip_path)

    assert set(result) == {
        "video", "total_hits", "total_misses", "events", "hit_events",
        "miss_events", "no_event_segments", "temporal_predictions",
    }
    # 50 frames at 10 fps -> windows at 0, 16, 18 -> 25 distinct token centers.
    predictions = result["temporal_predictions"]
    assert len(predictions) == 25
    assert predictions[0]["time"] == pytest.approx(0.1)
    for item in predictions:
        assert set(item["probabilities"]) == {"no_event", "hit", "miss"}
        assert math.isclose(sum(item["probabilities"].values()), 1.0, abs_tol=1e-4)
        assert item["label"] == max(item["probabilities"], key=item["probabilities"].get)
    assert result["total_hits"] == len(result["hit_events"])
    assert result["total_misses"] == len(result["miss_events"])


def test_pipeline_is_deterministic(tiny_models, clip_path):
    service = VideoInferenceService(tiny_models)

    assert service.predict(clip_path) == service.predict(clip_path)


def test_video_without_audio_fails_instead_of_predicting(tiny_models, clip_no_audio_path):
    with pytest.raises(RuntimeError, match="audio"):
        VideoInferenceService(tiny_models).predict(clip_no_audio_path)


def test_missing_video_fails(tiny_models, tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoInferenceService(tiny_models).predict(tmp_path / "missing.mp4")


def test_frame_count_overestimate_does_not_crash(tiny_models, clip_path, monkeypatch):
    # Containers often report more frames than can be decoded; the last window then
    # reads past the end. The original script crashed here (33 validity flags for 32 frames).
    from dataclasses import replace

    from pickleball_track.pipelines import preprocessing
    from pickleball_track.services import inference_service

    real_info = preprocessing.get_video_info(clip_path)
    monkeypatch.setattr(
        inference_service, "get_video_info", lambda path: replace(real_info, num_frames=60)
    )

    result = VideoInferenceService(tiny_models).predict(clip_path)

    assert len(result["temporal_predictions"]) > 0
