import pytest
import torch

from pickleball_track.core.config import NUM_TEMPORAL_TOKENS, NUM_VIDEO_FRAMES, Settings
from pickleball_track.core.exceptions import ModelLoadError
from pickleball_track.models.components.openvino_video_encoder import (
    OpenVINOVideoEncoder,
    _match_device,
)
from pickleball_track.models.components.video_encoder import VideoEncoder

from ..helpers import TINY_VIDEO_DIM, build_tiny_videomae


def test_device_names_match_numbered_devices():
    assert _match_device("GPU", ["CPU", "GPU.0", "NPU"]) == "GPU.0"
    assert _match_device("NPU", ["CPU", "GPU.0", "NPU"]) == "NPU"
    assert _match_device("GPU.1", ["CPU", "GPU.0"]) is None


def test_openvino_settings_read_environment(monkeypatch):
    monkeypatch.setenv("PICKLEBALL_VIDEOMAE_BACKEND", "OpenVINO")
    monkeypatch.setenv("PICKLEBALL_OPENVINO_DEVICES", " npu, gpu ")
    monkeypatch.setenv("PICKLEBALL_OPENVINO_GPU_PRECISION", "F16")

    settings = Settings.from_env()

    assert settings.videomae_backend == "openvino"
    assert settings.openvino_devices == ("NPU", "GPU")
    assert settings.openvino_gpu_precision == "f16"


@pytest.mark.parametrize(
    "overrides",
    [
        {"videomae_backend": "onnx"},
        {"videomae_backend": "openvino", "openvino_devices": ()},
        {"openvino_gpu_precision": "int8"},
    ],
)
def test_invalid_openvino_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


def test_matches_pytorch_encoder_and_shares_windows_across_devices():
    pytest.importorskip("openvino")
    torch.manual_seed(0)
    model = build_tiny_videomae()
    pixel_values = torch.randn(5, NUM_VIDEO_FRAMES, 3, 32, 32)
    frame_valid = torch.ones(5, NUM_VIDEO_FRAMES, dtype=torch.bool)
    frame_valid[-1, -3:] = False
    expected, expected_padding = VideoEncoder(model, torch.device("cpu"))(pixel_values, frame_valid)

    # Two compiled copies on the CPU stand in for the NPU and the GPU.
    encoder = OpenVINOVideoEncoder(model, ["CPU", "CPU"], cache_dir=None)
    try:
        features, padding = encoder(pixel_values, frame_valid)
    finally:
        encoder.close()

    assert features.shape == (5, NUM_TEMPORAL_TOKENS, TINY_VIDEO_DIM)
    torch.testing.assert_close(features, expected, atol=1e-4, rtol=1e-4)
    assert torch.equal(padding, expected_padding)
    assert sum(windows for _, windows in encoder.window_counts()) == 5


def test_unavailable_devices_fail_loading():
    pytest.importorskip("openvino")
    with pytest.raises(ModelLoadError, match="No OpenVINO device"):
        OpenVINOVideoEncoder(build_tiny_videomae(), ["NOT_A_DEVICE"], cache_dir=None)
