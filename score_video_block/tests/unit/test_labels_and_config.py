from pathlib import Path

import pytest

from pickleball_track.core.config import (
    IMAGE_SIZE,
    LABEL2ID,
    NUM_TEMPORAL_TOKENS,
    NUM_VIDEO_FRAMES,
    SPATIAL_GRID,
    TARGET_FPS,
    WEIGHTS_DIR,
    WINDOW_STRIDE,
    InferenceParams,
    Settings,
)
from pickleball_track.pipelines.postprocessing import ID2LABEL, label_from_id


def test_label_order_matches_checkpoint_contract():
    assert LABEL2ID == {"no_event": 0, "hit": 1, "miss": 2}
    assert [label_from_id(i) for i in range(3)] == ["no_event", "hit", "miss"]
    assert ID2LABEL == {0: "no_event", 1: "hit", 2: "miss"}


def test_unknown_label_id_raises():
    with pytest.raises(RuntimeError, match="Unknown class id"):
        label_from_id(3)


def test_temporal_constants_match_training():
    assert (NUM_VIDEO_FRAMES, NUM_TEMPORAL_TOKENS) == (32, 16)
    assert (TARGET_FPS, WINDOW_STRIDE, IMAGE_SIZE, SPATIAL_GRID) == (30.0, 16, 224, 2)


def test_default_decoder_is_the_tuned_one():
    assert InferenceParams() == InferenceParams(
        event_threshold=0.4, min_event_tokens=8, smooth_tokens=1, peak=False
    )


def test_default_settings_use_the_runtime_weights_directory(monkeypatch):
    for name in (
        "PICKLEBALL_CHECKPOINT", "PICKLEBALL_BEATS_CHECKPOINT", "PICKLEBALL_BEATS_ROOT",
        "PICKLEBALL_VIDEOMAE", "PICKLEBALL_DEVICE", "PICKLEBALL_MAX_VIDEO_BYTES",
        "PICKLEBALL_BATCH_SIZE", "PICKLEBALL_PRECISION", "PICKLEBALL_GPU_PREPROCESS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()

    assert WEIGHTS_DIR == Path(__file__).resolve().parents[2] / "src" / "weights"
    assert settings.checkpoint_path == WEIGHTS_DIR / "pickleball_v4.pt"
    assert settings.beats_checkpoint == WEIGHTS_DIR / "BEATs_iter3_plus_AS2M.pt"
    assert Path(settings.videomae) == WEIGHTS_DIR / "videomae-base-finetuned-kinetics"
    assert settings.device == "auto"
    assert (settings.batch_size, settings.precision, settings.gpu_preprocess) == (0, "auto", False)


def test_settings_read_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PICKLEBALL_CHECKPOINT", str(tmp_path / "model.pt"))
    monkeypatch.setenv("PICKLEBALL_BEATS_ROOT", "vendor/other")
    monkeypatch.setenv("PICKLEBALL_DEVICE", " CPU ")
    monkeypatch.setenv("PICKLEBALL_MAX_VIDEO_BYTES", "1024")
    monkeypatch.setenv("PICKLEBALL_BATCH_SIZE", "24")
    monkeypatch.setenv("PICKLEBALL_PRECISION", " BF16 ")
    monkeypatch.setenv("PICKLEBALL_GPU_PREPROCESS", "1")
    monkeypatch.setenv("MINIO_SECURE", "true")
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", " 100.116.40.89:9000 ")
    monkeypatch.setenv("MINIO_ROOT_USER", "user")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "password")
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.checkpoint_path == tmp_path / "model.pt"
    assert settings.beats_root == WEIGHTS_DIR.parents[1] / "vendor" / "other"
    assert settings.device == "cpu"
    assert settings.max_video_bytes == 1024
    assert (settings.batch_size, settings.precision, settings.gpu_preprocess) == (24, "bf16", True)
    assert settings.minio_secure is True
    assert settings.minio_public_endpoint == "100.116.40.89:9000"
    assert (settings.minio_access_key, settings.minio_secret_key) == ("user", "password")
    assert "password" not in repr(settings)


def test_inference_hyperparameters_read_environment(monkeypatch):
    monkeypatch.setenv("PICKLEBALL_EVENT_THRESHOLD", "0.6")
    monkeypatch.setenv("PICKLEBALL_MIN_EVENT_TOKENS", "4")
    monkeypatch.setenv("PICKLEBALL_EVENT_SMOOTH_TOKENS", "3")
    monkeypatch.setenv("PICKLEBALL_EVENT_AT_PEAK", "true")

    assert Settings.from_env().inference == InferenceParams(
        event_threshold=0.6, min_event_tokens=4, smooth_tokens=3, peak=True
    )
    monkeypatch.setenv("PICKLEBALL_EVENT_THRESHOLD", "argmax")
    assert Settings.from_env().inference.event_threshold is None


@pytest.mark.parametrize(
    "overrides", [{"max_video_bytes": 0}, {"device": "tpu"}, {"batch_size": -1}, {"precision": "int8"}]
)
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


@pytest.mark.parametrize(
    "overrides", [{"event_threshold": 1.0}, {"event_threshold": -0.1}, {"min_event_tokens": 0}, {"smooth_tokens": 0}]
)
def test_invalid_inference_params_are_rejected(overrides):
    with pytest.raises(ValueError):
        InferenceParams(**overrides)
