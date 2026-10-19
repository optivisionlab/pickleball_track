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

from ..core.config import AUDIO_SR, LABEL2ID, NUM_VIDEO_FRAMES, WEIGHTS_DIR, Settings
from ..core.exceptions import ModelLoadError
from ..pipelines.frames import (
    FramePreprocessor,
    ProcessFramePool,
    TorchFrames,
    default_preprocess_workers,
)
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
    # Sliding windows per forward pass.
    batch_size: int = 1
    # None: exact Hugging Face processor on the CPU.
    frame_preprocessor: FramePreprocessor | None = None
    # Exact preprocessing on worker processes; used when frame_preprocessor is None.
    frame_pool: ProcessFramePool | None = None

    def close(self) -> None:
        """Stop preprocessing workers; the bundle must not be used afterwards."""
        if self.frame_pool is not None:
            self.frame_pool.close()


# Defaults for PICKLEBALL_BATCH_SIZE=0. 16 windows of 32 frames use ~2-3 GB on the
# GPU in fp16; larger batches give no further speedup on an A30 because frame
# decoding, not the GPU, is then the bottleneck.
DEFAULT_BATCH_SIZE = {"cuda": 16, "cpu": 4}

_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

VIDEOMAE_CONFIG_FILES = ("config.json", "preprocessor_config.json")
VIDEOMAE_WEIGHTS_FILE = "model.safetensors"


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


def resolve_precision(precision: str, device: torch.device) -> torch.dtype | None:
    """Autocast dtype for VideoMAE, or None for fp32. Reduced precision is CUDA-only."""
    if precision == "auto":
        return torch.float16 if device.type == "cuda" else None
    if precision == "fp32":
        return None
    if precision not in _AMP_DTYPES:
        raise ValueError("PICKLEBALL_PRECISION must be auto, fp32, fp16, or bf16")
    if device.type != "cuda":
        logger.warning("Reduced precision needs CUDA; running in fp32", extra={"precision": precision})
        return None
    return _AMP_DTYPES[precision]


def resolve_batch_size(batch_size: int, device: torch.device) -> int:
    return batch_size if batch_size > 0 else DEFAULT_BATCH_SIZE[device.type]


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
        # Files that ensure_videomae_files() can still download are not missing.
        downloadable = bool(settings.videomae_repo)
        if not (videomae_path / "config.json").is_file() and not downloadable:
            missing.append(f"VideoMAE config not found in {videomae_path}")
        if not _has_videomae_weights(videomae_path) and not downloadable:
            missing.append(f"VideoMAE weights not found in {videomae_path}")
    elif not _is_remote_model_id(settings.videomae):
        missing.append(f"VideoMAE model path not found: {videomae_path}")
    return missing


def _has_videomae_weights(directory: Path) -> bool:
    return any(directory.glob("*.safetensors")) or any(directory.glob("pytorch_model*.bin"))


def ensure_videomae_files(settings: Settings) -> None:
    """Download absent VideoMAE files from ``settings.videomae_repo`` into the local directory.

    Only missing files are fetched; existing configs (which define the trained
    preprocessing) are never overwritten.
    """
    directory = Path(settings.videomae).expanduser()
    if not settings.videomae_repo or not directory.is_dir():
        return
    wanted = [name for name in VIDEOMAE_CONFIG_FILES if not (directory / name).is_file()]
    if not _has_videomae_weights(directory):
        wanted.append(VIDEOMAE_WEIGHTS_FILE)
    if not wanted:
        return

    from huggingface_hub import hf_hub_download

    for filename in wanted:
        logger.info(
            "Downloading VideoMAE file",
            extra={"repo": settings.videomae_repo, "file": filename, "directory": str(directory)},
        )
        try:
            hf_hub_download(settings.videomae_repo, filename, local_dir=directory)
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot download {filename} from {settings.videomae_repo} into {directory}: {exc}"
            ) from exc


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


@torch.inference_mode()
def warmup(bundle: ModelBundle, fps: float = 30.0) -> None:
    """Run one full batch so CUDA/cuBLAS initialization is not paid by the first request.

    cuDNN autotuning (``cudnn.benchmark``) is left off: it gave no steady-state
    speedup on an A30 and re-tunes for every new batch size, e.g. the last batch.
    """
    crop = bundle.processor.crop_size
    batch = bundle.batch_size
    pixel_values = torch.zeros(batch, NUM_VIDEO_FRAMES, 3, int(crop["height"]), int(crop["width"]))
    frame_valid = torch.ones(batch, NUM_VIDEO_FRAMES, dtype=torch.bool)
    samples = int(round(NUM_VIDEO_FRAMES / fps * AUDIO_SR))
    segment = torch.zeros(batch, samples)
    padding_mask = torch.zeros(batch, samples, dtype=torch.bool)
    video_features, temporal_padding = bundle.video_encoder(pixel_values, frame_valid)
    audio_features, audio_padding = bundle.audio_encoder(segment, padding_mask)
    bundle.fusion_model(video_features, audio_features, audio_padding, temporal_padding)
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)


def load_models(settings: Settings) -> ModelBundle:
    """Load all artifacts and verify that backbones match the trained checkpoint."""
    ensure_videomae_files(settings)
    missing = missing_model_assets(settings)
    if missing:
        raise ModelLoadError("; ".join(missing))

    device = resolve_device(settings.device)
    amp_dtype = resolve_precision(settings.precision, device)
    batch_size = resolve_batch_size(settings.batch_size, device)
    videomae, processor = load_videomae(settings.videomae, device)
    beats = load_beats(settings.beats_root, settings.beats_checkpoint, device)
    fusion_model = load_trained_model(settings.checkpoint_path, device)

    video_encoder = VideoEncoder(videomae, device, amp_dtype)
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

    frame_preprocessor = (
        TorchFrames(processor, device) if settings.gpu_preprocess and device.type == "cuda" else None
    )
    workers = default_preprocess_workers(settings.preprocess_workers)
    frame_pool = ProcessFramePool(processor, workers) if frame_preprocessor is None and workers else None
    logger.info(
        "Models loaded",
        extra={
            "device": str(device),
            "videomae_precision": str(amp_dtype or torch.float32),
            "batch_size": batch_size,
            "gpu_preprocessing": frame_preprocessor is not None,
            "preprocess_workers": frame_pool.workers if frame_pool else 0,
        },
    )
    bundle = ModelBundle(
        device=device,
        processor=processor,
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
        fusion_model=fusion_model,
        batch_size=batch_size,
        frame_preprocessor=frame_preprocessor,
        frame_pool=frame_pool,
    )
    if device.type == "cuda":
        warmup(bundle)
    return bundle
