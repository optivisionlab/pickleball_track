from dataclasses import replace

import pytest
import torch

from pickleball_track.core.config import LABEL2ID
from pickleball_track.core.exceptions import ModelLoadError
from pickleball_track.models import model_loader
from pickleball_track.models.architecture import AVFeatureModel
from pickleball_track.models.model_loader import (
    load_models,
    load_trained_model,
    missing_model_assets,
    resolve_batch_size,
    resolve_device,
    resolve_precision,
)

from ..helpers import TINY_AUDIO_DIM, TINY_VIDEO_DIM, build_tiny_fusion


def _save_checkpoint(path, state_dict, **extra):
    torch.save({"model": state_dict, **extra}, path)
    return path


def test_checkpoint_round_trip_reproduces_outputs(tmp_path):
    reference = build_tiny_fusion(seed=3)
    checkpoint = _save_checkpoint(
        tmp_path / "best.pt", reference.state_dict(), label2id=dict(LABEL2ID)
    )

    loaded = load_trained_model(checkpoint, torch.device("cpu"))

    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())
    assert loaded.video_proj.net[0].in_features == TINY_VIDEO_DIM
    assert loaded.audio_proj.net[0].in_features == TINY_AUDIO_DIM
    torch.manual_seed(0)
    inputs = (
        torch.randn(1, 16, TINY_VIDEO_DIM),
        torch.randn(1, 10, TINY_AUDIO_DIM),
        torch.tensor([[False] * 8 + [True] * 2]),
        torch.tensor([[False] * 12 + [True] * 4]),
    )
    with torch.inference_mode():
        torch.testing.assert_close(loaded(*inputs), reference(*inputs))


def test_checkpoint_label_mismatch_is_rejected(tmp_path):
    checkpoint = _save_checkpoint(
        tmp_path / "best.pt",
        build_tiny_fusion().state_dict(),
        label2id={"hit": 0, "miss": 1, "no_event": 2},
    )

    with pytest.raises(ModelLoadError, match="label mapping"):
        load_trained_model(checkpoint, torch.device("cpu"))


def test_checkpoint_with_missing_keys_is_rejected_strictly(tmp_path):
    state_dict = build_tiny_fusion().state_dict()
    del state_dict["head.net.6.weight"]

    with pytest.raises(ModelLoadError, match="head.net.6.weight"):
        load_trained_model(_save_checkpoint(tmp_path / "best.pt", state_dict), torch.device("cpu"))


def test_checkpoint_with_unexpected_keys_is_rejected_strictly(tmp_path):
    state_dict = build_tiny_fusion().state_dict()
    state_dict["extra.weight"] = torch.zeros(1)

    with pytest.raises(ModelLoadError, match="extra.weight"):
        load_trained_model(_save_checkpoint(tmp_path / "best.pt", state_dict), torch.device("cpu"))


def test_checkpoint_without_model_state_is_rejected(tmp_path):
    path = tmp_path / "best.pt"
    torch.save({"label2id": dict(LABEL2ID)}, path)

    with pytest.raises(ModelLoadError, match="state_dict"):
        load_trained_model(path, torch.device("cpu"))


def test_unreadable_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(ModelLoadError, match="Cannot read"):
        load_trained_model(path, torch.device("cpu"))


def test_missing_assets_are_all_reported(test_settings):
    missing = missing_model_assets(test_settings)

    assert len(missing) == 4
    with pytest.raises(ModelLoadError, match="trained checkpoint not found"):
        load_models(test_settings)


def test_videomae_directory_without_weights_is_reported(test_settings, tmp_path):
    videomae_dir = tmp_path / "videomae"
    videomae_dir.mkdir()
    (videomae_dir / "config.json").write_text("{}")

    missing = missing_model_assets(replace(test_settings, videomae=str(videomae_dir)))

    assert f"VideoMAE weights not found in {videomae_dir}" in missing


def test_videomae_weights_are_downloaded_when_a_repo_is_configured(test_settings, tmp_path, monkeypatch):
    import huggingface_hub

    videomae_dir = tmp_path / "videomae"
    videomae_dir.mkdir()
    for name in model_loader.VIDEOMAE_CONFIG_FILES:
        (videomae_dir / name).write_text("{}")
    downloads = []

    def fake_download(repo_id, filename, local_dir):
        downloads.append((repo_id, filename))
        (local_dir / filename).write_bytes(b"weights")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    settings = replace(test_settings, videomae=str(videomae_dir), videomae_repo="org/videomae")

    assert not any("VideoMAE" in item for item in missing_model_assets(settings))
    model_loader.ensure_videomae_files(settings)
    model_loader.ensure_videomae_files(settings)  # nothing left to fetch

    # Local configs are kept; only the weights are fetched, once.
    assert downloads == [("org/videomae", "model.safetensors")]
    assert (videomae_dir / "config.json").read_text() == "{}"


def test_videomae_download_failure_is_a_model_load_error(test_settings, tmp_path, monkeypatch):
    import huggingface_hub

    videomae_dir = tmp_path / "videomae"
    videomae_dir.mkdir()

    def offline(repo_id, filename, local_dir):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    settings = replace(test_settings, videomae=str(videomae_dir), videomae_repo="org/videomae")

    with pytest.raises(ModelLoadError, match="network unreachable"):
        model_loader.ensure_videomae_files(settings)


def test_load_models_rejects_backbone_checkpoint_dimension_mismatch(monkeypatch, test_settings, tiny_models):
    monkeypatch.setattr(model_loader, "missing_model_assets", lambda settings: [])
    monkeypatch.setattr(
        model_loader, "load_videomae",
        lambda source, device: (tiny_models.video_encoder.model, tiny_models.processor),
    )
    monkeypatch.setattr(
        model_loader, "load_beats", lambda root, checkpoint, device: tiny_models.audio_encoder.model
    )
    mismatched = AVFeatureModel(video_dim=TINY_VIDEO_DIM, audio_dim=TINY_AUDIO_DIM + 1)
    monkeypatch.setattr(model_loader, "load_trained_model", lambda path, device: mismatched)

    with pytest.raises(ModelLoadError, match="BEATs feature size"):
        load_models(test_settings)

    monkeypatch.setattr(
        model_loader, "load_trained_model", lambda path, device: tiny_models.fusion_model
    )
    bundle = load_models(test_settings)
    assert bundle.fusion_model is tiny_models.fusion_model
    assert bundle.device == torch.device("cpu")


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda"}
    with pytest.raises(ValueError):
        resolve_device("tpu")


@pytest.mark.parametrize(
    ("precision", "device", "expected"),
    [
        ("auto", "cuda", torch.float16),
        ("auto", "cpu", None),
        ("fp32", "cuda", None),
        ("bf16", "cuda", torch.bfloat16),
        ("fp16", "cpu", None),  # reduced precision is CUDA-only; falls back to fp32
    ],
)
def test_precision_resolution(precision, device, expected):
    assert resolve_precision(precision, torch.device(device)) is expected


def test_batch_size_resolution():
    assert resolve_batch_size(0, torch.device("cuda")) == 16
    assert resolve_batch_size(0, torch.device("cpu")) == 4
    assert resolve_batch_size(7, torch.device("cuda")) == 7
