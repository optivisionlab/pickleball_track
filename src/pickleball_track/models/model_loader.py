"""Load and cache pretrained backbones and the trained fusion model."""

from functools import lru_cache
import logging
from pathlib import Path
import sys

import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

from ..core.config import (
    BEATS_CHECKPOINT,
    BEATS_ROOT,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_VIDEOMAE,
    LABEL2ID,
    MODEL_DEVICE,
    NUM_VIDEO_FRAMES,
    WEIGHTS_DIR,
)
from .fusion import AVFeatureModel

logger = logging.getLogger(__name__)


def model_assets_available() -> bool:
    """Return whether all locally configured model artifacts are present."""
    if not DEFAULT_CHECKPOINT_PATH.is_file():
        return False
    if not Path(BEATS_CHECKPOINT).is_file():
        return False
    if not (Path(BEATS_ROOT) / "beats" / "BEATs.py").is_file():
        return False

    videomae_path = Path(DEFAULT_VIDEOMAE).expanduser()
    if videomae_path.is_dir():
        has_weights = any(videomae_path.glob("*.safetensors")) or any(
            videomae_path.glob("pytorch_model*.bin")
        )
        return (videomae_path / "config.json").is_file() and has_weights
    return not videomae_path.is_absolute() and not str(DEFAULT_VIDEOMAE).startswith(".")


def resolve_device(device_name: str = MODEL_DEVICE) -> torch.device:
    """Resolve auto/CPU/CUDA while rejecting unavailable explicit devices."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PICKLEBALL_DEVICE=cuda but CUDA is unavailable")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("PICKLEBALL_DEVICE must be auto, cpu, or cuda")
    return torch.device(device_name)


def adapt_videomae_to_frame_count(model: VideoMAEModel) -> None:
    """Interpolate temporal position embeddings for the trained frame count."""
    target_frames = NUM_VIDEO_FRAMES
    old_frames = int(model.config.num_frames)
    tubelet_size = int(model.config.tubelet_size)

    if old_frames == target_frames:
        return
    if target_frames % tubelet_size != 0:
        raise RuntimeError("NUM_VIDEO_FRAMES must be divisible by VideoMAE tubelet_size.")

    old_t = old_frames // tubelet_size
    new_t = target_frames // tubelet_size
    image_size = int(model.config.image_size)
    patch_size = int(model.config.patch_size)
    height_patches = image_size // patch_size
    width_patches = image_size // patch_size
    position_embeddings = model.embeddings.position_embeddings
    expected_tokens = old_t * height_patches * width_patches

    if position_embeddings.shape[1] != expected_tokens:
        raise RuntimeError(
            "Unexpected VideoMAE positional embedding shape: "
            f"{tuple(position_embeddings.shape)}"
        )

    hidden_dim = position_embeddings.shape[-1]
    embeddings = position_embeddings.reshape(
        1, old_t, height_patches, width_patches, hidden_dim
    ).permute(0, 4, 1, 2, 3)
    embeddings = torch.nn.functional.interpolate(
        embeddings,
        size=(new_t, height_patches, width_patches),
        mode="trilinear",
        align_corners=False,
    )
    new_embeddings = embeddings.permute(0, 2, 3, 4, 1).reshape(
        1, new_t * height_patches * width_patches, hidden_dim
    ).contiguous()
    model.embeddings.position_embeddings = torch.nn.Parameter(
        new_embeddings,
        requires_grad=False,
    )
    model.config.num_frames = target_frames


@lru_cache(maxsize=1)
def load_videomae(device: torch.device):
    logger.info("Loading pretrained VideoMAE", extra={"model": DEFAULT_VIDEOMAE})
    model_path = Path(DEFAULT_VIDEOMAE).expanduser()

    if model_path.is_dir():
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"VideoMAE config not found in {model_path}")
        has_weights = any(model_path.glob("*.safetensors")) or any(
            model_path.glob("pytorch_model*.bin")
        )
        if not has_weights:
            raise FileNotFoundError(f"VideoMAE weights not found in {model_path}")
        model_source = str(model_path.resolve())
        load_options = {"local_files_only": True}
    else:
        if model_path.is_absolute() or str(DEFAULT_VIDEOMAE).startswith("."):
            raise FileNotFoundError(f"VideoMAE model path not found: {model_path}")
        model_source = DEFAULT_VIDEOMAE
        load_options = {"cache_dir": str(WEIGHTS_DIR)}

    processor = VideoMAEImageProcessor.from_pretrained(model_source, **load_options)
    model = VideoMAEModel.from_pretrained(model_source, **load_options)
    adapt_videomae_to_frame_count(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return model, processor


@lru_cache(maxsize=1)
def load_beats(device: torch.device):
    beats_root = Path(BEATS_ROOT)
    beats_package = beats_root / "beats"
    if not (beats_package / "BEATs.py").is_file():
        raise FileNotFoundError(
            f"BEATs source not found under {beats_root}; clone the Microsoft "
            "unilm repository and set PICKLEBALL_BEATS_ROOT to that checkout"
        )
    if not Path(BEATS_CHECKPOINT).is_file():
        raise FileNotFoundError(f"BEATs checkpoint not found: {BEATS_CHECKPOINT}")

    if str(beats_root) not in sys.path:
        sys.path.insert(0, str(beats_root))
    if str(beats_package) not in sys.path:
        sys.path.insert(0, str(beats_package))

    try:
        from beats.BEATs import BEATs, BEATsConfig
    except ImportError:
        from BEATs import BEATs, BEATsConfig

    logger.info("Loading pretrained BEATs", extra={"checkpoint": BEATS_CHECKPOINT})
    checkpoint = torch.load(BEATS_CHECKPOINT, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return model


@lru_cache(maxsize=2)
def load_trained_model(
    checkpoint_path: str | Path,
    video_dim: int,
    audio_dim: int,
    device: torch.device,
) -> AVFeatureModel:
    logger.info("Loading trained fusion checkpoint", extra={"checkpoint": str(checkpoint_path)})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_labels = checkpoint.get("label2id")
    if checkpoint_labels is not None and checkpoint_labels != LABEL2ID:
        raise RuntimeError(
            f"Checkpoint label mapping {checkpoint_labels!r} does not match "
            f"configured mapping {LABEL2ID!r}"
        )

    model = AVFeatureModel(video_dim=video_dim, audio_dim=audio_dim)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return model