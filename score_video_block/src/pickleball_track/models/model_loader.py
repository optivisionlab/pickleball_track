"""Load pretrained backbones and the trained fusion checkpoint.

Three separate artifacts are involved:

1. Pretrained backbone weights: VideoMAE (Hugging Face directory) and BEATs
   (``BEATs_iter3_plus_AS2M.pt`` + microsoft/unilm source checkout). Both are
   frozen and are not part of the trained checkpoint.
2. The trained checkpoint (``best.pt``): ``{"model": AVFeatureModel state_dict,
   "label2id": ...}``. Projection input sizes are read from its tensors.
3. Architecture/preprocessing constants in ``core.config`` (frame count,
   temporal tokens, audio rate, label order) which must match training.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

from ..core.config import AUDIO_SR, LABEL2ID, WEIGHTS_DIR, Settings
from ..core.exceptions import ModelLoadError
from .architecture import AVFeatureModel
from .components.audio_encoder import AudioEncoder
from .components.video_encoder import VideoEncoder, adapt_videomae_to_frame_count

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    """Everything inference needs, loaded once per worker process."""

    device: torch.device
    processor: VideoMAEImageProcessor
    video_encoder: VideoEncoder
    audio_encoder: AudioEncoder
    fusion_model: AVFeatureModel


def resolve_device(device_name: str) -> torch.device:
    """Resolve auto/CPU/CUDA while rejecting unavailable explicit devices."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("PICKLEBALL_DEVICE must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PICKLEBALL_DEVICE=cuda but CUDA is unavailable")
    return torch.device(device_name)


def _is_remote_model_id(source: str) -> bool:
    path = Path(source).expanduser()
    return not path.exists() and not path.is_absolute() and not source.startswith(".")


def missing_model_assets(settings: Settings) -> list[str]:
    """Describe every configured local model artifact that is absent."""
    missing = []
    if not settings.checkpoint_path.is_file():
        missing.append(f"trained checkpoint not found: {settings.checkpoint_path}")
    if not settings.beats_checkpoint.is_file():
        missing.append(f"BEATs checkpoint not found: {settings.beats_checkpoint}")
    if not (settings.beats_root / "beats" / "BEATs.py").is_file():
        missing.append(
            f"BEATs source not found under {settings.beats_root}; clone the Microsoft "
            "unilm repository and set PICKLEBALL_BEATS_ROOT to that checkout"
        )

    videomae_path = Path(settings.videomae).expanduser()
    if videomae_path.is_dir():
        if not (videomae_path / "config.json").is_file():
            missing.append(f"VideoMAE config not found in {videomae_path}")
        has_weights = any(videomae_path.glob("*.safetensors")) or any(
            videomae_path.glob("pytorch_model*.bin")
        )
        if not has_weights:
            missing.append(f"VideoMAE weights not found in {videomae_path}")
    elif not _is_remote_model_id(settings.videomae):
        missing.append(f"VideoMAE model path not found: {videomae_path}")
    return missing


def _freeze(model: torch.nn.Module, device: torch.device) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)


def load_videomae(
    source: str, device: torch.device
) -> tuple[VideoMAEModel, VideoMAEImageProcessor]:
    logger.info("Loading pretrained VideoMAE", extra={"model": source})
    model_path = Path(source).expanduser()
    if model_path.is_dir():
        model_source = str(model_path.resolve())
        load_options: dict[str, Any] = {"local_files_only": True}
    else:
        model_source = source
        load_options = {"cache_dir": str(WEIGHTS_DIR)}

    processor = VideoMAEImageProcessor.from_pretrained(model_source, **load_options)
    model = VideoMAEModel.from_pretrained(model_source, **load_options)
    adapt_videomae_to_frame_count(model)
    _freeze(model, device)
    return model, processor


def load_beats(beats_root: Path, checkpoint_path: Path, device: torch.device) -> Any:
    beats_package = beats_root / "beats"
    if str(beats_root) not in sys.path:
        sys.path.insert(0, str(beats_root))
    if str(beats_package) not in sys.path:
        sys.path.insert(0, str(beats_package))

    try:
        from beats.BEATs import BEATs, BEATsConfig
    except ImportError:
        from BEATs import BEATs, BEATsConfig

    logger.info("Loading pretrained BEATs", extra={"checkpoint": str(checkpoint_path)})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    _freeze(model, device)
    return model


def load_trained_model(checkpoint_path: str | Path, device: torch.device) -> AVFeatureModel:
    """Build AVFeatureModel from the checkpoint's own tensor shapes and load it strictly."""
    logger.info("Loading trained fusion checkpoint", extra={"checkpoint": str(checkpoint_path)})
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelLoadError(f"Cannot read trained checkpoint {checkpoint_path}: {exc}") from exc

    checkpoint_labels = checkpoint.get("label2id") if isinstance(checkpoint, dict) else None
    if checkpoint_labels is not None and checkpoint_labels != LABEL2ID:
        raise ModelLoadError(
            f"Checkpoint label mapping {checkpoint_labels!r} does not match "
            f"configured mapping {LABEL2ID!r}"
        )

    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise ModelLoadError("Trained checkpoint has no 'model' state_dict")
    try:
        video_dim = int(state_dict["video_proj.net.0.weight"].shape[1])
        audio_dim = int(state_dict["audio_proj.net.0.weight"].shape[1])
    except KeyError as exc:
        raise ModelLoadError(f"Trained checkpoint is missing projection weights: {exc}") from exc

    model = AVFeatureModel(video_dim=video_dim, audio_dim=audio_dim)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        # The message lists missing/unexpected keys and shape mismatches.
        raise ModelLoadError(f"Trained checkpoint does not match AVFeatureModel: {exc}") from exc
    _freeze(model, device)
    return model


def _beats_feature_dim(audio_encoder: AudioEncoder) -> int:
    # One second of silence; the feature size does not depend on the content.
    segment = torch.zeros(1, AUDIO_SR, dtype=torch.float32)
    padding_mask = torch.zeros(1, AUDIO_SR, dtype=torch.bool)
    features, _ = audio_encoder(segment, padding_mask)
    return int(features.shape[-1])


def load_models(settings: Settings) -> ModelBundle:
    """Load all artifacts and verify that backbones match the trained checkpoint."""
    missing = missing_model_assets(settings)
    if missing:
        raise ModelLoadError("; ".join(missing))

    device = resolve_device(settings.device)
    videomae, processor = load_videomae(settings.videomae, device)
    beats = load_beats(settings.beats_root, settings.beats_checkpoint, device)
    fusion_model = load_trained_model(settings.checkpoint_path, device)

    video_encoder = VideoEncoder(videomae, device)
    audio_encoder = AudioEncoder(beats, device)
    audio_dim = _beats_feature_dim(audio_encoder)
    if fusion_model.video_proj.net[0].in_features != video_encoder.hidden_size:
        raise ModelLoadError(
            f"VideoMAE hidden size {video_encoder.hidden_size} does not match checkpoint "
            f"video_dim {fusion_model.video_proj.net[0].in_features}"
        )
    if fusion_model.audio_proj.net[0].in_features != audio_dim:
        raise ModelLoadError(
            f"BEATs feature size {audio_dim} does not match checkpoint "
            f"audio_dim {fusion_model.audio_proj.net[0].in_features}"
        )

    logger.info("Models loaded", extra={"device": str(device)})
    return ModelBundle(
        device=device,
        processor=processor,
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
        fusion_model=fusion_model,
    )
