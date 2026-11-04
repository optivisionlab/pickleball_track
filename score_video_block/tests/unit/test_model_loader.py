from dataclasses import replace

import pytest
import torch

from pickleball_track.core.config import LABEL2ID
from pickleball_track.core.exceptions import ModelLoadError
from pickleball_track.models import model_loader
from pickleball_track.models.model_loader import (
    load_models,
    load_trained_model,
    missing_model_assets,
    resolve_batch_size,
    resolve_device,
    resolve_precision,
)

from ..helpers import (
    TINY_AUDIO_DIM,
    TINY_VIDEO_DIM,
    build_tiny_checkpoint,
    build_tiny_ensemble,
)


def _save(path, checkpoint):
    torch.save(checkpoint, path)
    return path


def _inputs(batch=2):
    torch.manual_seed(0)
    padding = torch.zeros(batch, 16, dtype=torch.bool)
    padding[1, 12:] = True
    return torch.randn(batch, 16, TINY_VIDEO_DIM), torch.randn(batch, 16, TINY_AUDIO_DIM), padding


def test_checkpoint_round_trip_reproduces_the_ensemble(tmp_path):
    reference = build_tiny_ensemble(folds=2)
    path = _save(tmp_path / "model.pt", build_tiny_checkpoint(folds=2))

    loaded = load_trained_model(path, torch.device("cpu"))

    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())
    assert len(loaded.models) == 2
    assert (loaded.video_dim, loaded.audio_dim) == (TINY_VIDEO_DIM, TINY_AUDIO_DIM)
    video, audio, padding = _inputs()
    with torch.inference_mode():
        probs = loaded(video, audio, padding)
        torch.testing.assert_close(probs, reference(video, audio, padding))
    assert probs.shape == (2, 16, 3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, 16))


def test_ensemble_averages_its_folds():
    ensemble = build_tiny_ensemble(folds=2)
    video, audio, padding = _inputs()

    with torch.inference_mode():
        both = ensemble(video, audio, padding)
        singles = []
        for k in range(2):
            single = type(ensemble)(
                [ensemble.models[k]], ensemble.class_text,
                {"video": ensemble.video_mean[k:k + 1], "audio": ensemble.audio_mean[k:k + 1]},
                {"video": ensemble.video_std[k:k + 1], "audio": ensemble.audio_std[k:k + 1]},
            ).eval()
            singles.append(single(video, audio, padding))

    torch.testing.assert_close(both, (singles[0] + singles[1]) / 2)


def test_missing_audio_is_handled_like_training():
    ensemble = build_tiny_ensemble()
    video, _, padding = _inputs()

    with torch.inference_mode():
        probs = ensemble(video, None, padding)

    assert torch.isfinite(probs).all()


def test_checkpoint_label_mismatch_is_rejected(tmp_path):
    checkpoint = build_tiny_checkpoint()
    checkpoint["label2id"] = {"hit": 0, "miss": 1, "no_event": 2}

    with pytest.raises(ModelLoadError, match="label mapping"):
        load_trained_model(_save(tmp_path / "model.pt", checkpoint), torch.device("cpu"))


def test_checkpoint_with_missing_keys_is_rejected_strictly(tmp_path):
    checkpoint = build_tiny_checkpoint()
    del checkpoint["folds"][1]["model"]["head.bias"]

    with pytest.raises(ModelLoadError, match="head.bias"):
        load_trained_model(_save(tmp_path / "model.pt", checkpoint), torch.device("cpu"))


def test_checkpoint_with_unexpected_keys_is_rejected_strictly(tmp_path):
    checkpoint = build_tiny_checkpoint()
    checkpoint["folds"][0]["model"]["extra.weight"] = torch.zeros(1)

    with pytest.raises(ModelLoadError, match="extra.weight"):
        load_trained_model(_save(tmp_path / "model.pt", checkpoint), torch.device("cpu"))


def test_checkpoint_without_feature_statistics_is_rejected(tmp_path):
    checkpoint = build_tiny_checkpoint()
    del checkpoint["folds"][0]["feature_std"]

    with pytest.raises(ModelLoadError, match="feature_std"):
        load_trained_model(_save(tmp_path / "model.pt", checkpoint), torch.device("cpu"))


def test_old_single_model_checkpoint_is_rejected(tmp_path):
    path = _save(tmp_path / "best.pt", {"model": {}, "label2id": dict(LABEL2ID)})

    with pytest.raises(ModelLoadError, match="export_checkpoint"):
        load_trained_model(path, torch.device("cpu"))


def test_unreadable_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "model.pt"
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
    monkeypatch.setattr(model_loader, "load_videomae", lambda source, device: tiny_models.video_encoder.model)
    monkeypatch.setattr(
        model_loader, "load_beats", lambda root, checkpoint, device: tiny_models.audio_encoder.model
    )
    mismatched = build_tiny_ensemble()
    mismatched.audio_mean = torch.zeros(2, TINY_AUDIO_DIM + 1)
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
