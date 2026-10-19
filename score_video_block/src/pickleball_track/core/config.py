"""Runtime configuration.

- MODEL CONSTANTS are fixed by the trained checkpoint; changing them breaks
  compatibility with ``best.pt``.
- INFERENCE HYPERPARAMETERS only affect post-processing / windowing and can be
  tuned freely: edit the defaults below or override them with the matching
  ``PICKLEBALL_*`` environment variable.
- DEPLOYMENT SETTINGS (paths, device, limits, credentials) come from environment
  variables, optionally loaded from ``<project root>/.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = PROJECT_ROOT / "src" / "weights"


# =========================
# MODEL CONSTANTS (fixed by training)
# =========================

AUDIO_SR = 16000

NUM_VIDEO_FRAMES = 32

NUM_TEMPORAL_TOKENS = 16

LABEL2ID = {
    "no_event": 0,
    "hit": 1,
    "miss": 2,
}

SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov"})

SUPPORTED_DEVICES = frozenset({"auto", "cpu", "cuda"})

# auto: fp16 autocast for VideoMAE on CUDA, fp32 on CPU. BEATs and the fusion
# model always run in fp32 (BEATs' fbank overflows fp16).
SUPPORTED_PRECISIONS = frozenset({"auto", "fp32", "fp16", "bf16"})


# =========================
# INFERENCE HYPERPARAMETERS (tunable)
# =========================

# Minimum smoothed probability for a temporal token to count as hit / miss.
HIT_THRESHOLD = 0.8

MISS_THRESHOLD = 0.8

# Same-class events closer than this (seconds) are merged; same-class tokens
# separated by a no_event gap of at most this length are joined.
EVENT_MIN_GAP = 0.5

# Moving-average radius (in temporal tokens) applied before thresholding; 0 disables.
TEMPORAL_SMOOTH_RADIUS = 1

# Overlap between consecutive sliding windows (0 <= overlap < 1).
OVERLAP = 0.5


@dataclass(frozen=True)
class InferenceParams:
    hit_threshold: float = HIT_THRESHOLD
    miss_threshold: float = MISS_THRESHOLD
    event_min_gap: float = EVENT_MIN_GAP
    smooth_radius: int = TEMPORAL_SMOOTH_RADIUS
    overlap: float = OVERLAP

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("PICKLEBALL_OVERLAP must be in [0, 1)")
        if self.smooth_radius < 0:
            raise ValueError("PICKLEBALL_SMOOTH_RADIUS must be >= 0")
        if self.event_min_gap < 0:
            raise ValueError("PICKLEBALL_EVENT_MIN_GAP must be >= 0")

    @classmethod
    def from_env(cls) -> InferenceParams:
        return cls(
            hit_threshold=float(os.getenv("PICKLEBALL_HIT_THRESHOLD", HIT_THRESHOLD)),
            miss_threshold=float(os.getenv("PICKLEBALL_MISS_THRESHOLD", MISS_THRESHOLD)),
            event_min_gap=float(os.getenv("PICKLEBALL_EVENT_MIN_GAP", EVENT_MIN_GAP)),
            smooth_radius=int(os.getenv("PICKLEBALL_SMOOTH_RADIUS", TEMPORAL_SMOOTH_RADIUS)),
            overlap=float(os.getenv("PICKLEBALL_OVERLAP", OVERLAP)),
        )


# =========================
# DEPLOYMENT SETTINGS
# =========================

def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    checkpoint_path: Path = WEIGHTS_DIR / "best.pt"
    beats_checkpoint: Path = WEIGHTS_DIR / "BEATs_iter3_plus_AS2M.pt"
    beats_root: Path = PROJECT_ROOT / "vendor" / "unilm"
    # Local directory, or a Hugging Face model id downloaded into WEIGHTS_DIR.
    videomae: str = str(WEIGHTS_DIR / "videomae-base")
    # Hub repo whose weights are downloaded into a local ``videomae`` directory that
    # has none yet; empty disables the download (offline deployments).
    videomae_repo: str = "MCG-NJU/videomae-base"
    device: str = "auto"
    # Sliding windows per forward pass; 0 picks a default for the device.
    batch_size: int = 0
    precision: str = "auto"
    # Resize/normalize frames on the GPU: ~20% faster end to end, but pixels may
    # differ from the training-time PIL preprocessing by one 8-bit level.
    gpu_preprocess: bool = False
    # Worker processes resizing frames (exact, shared memory); 0 uses threads in
    # this process, which the GIL limits to ~150 frames/s.
    preprocess_workers: int = 8
    max_video_bytes: int = 512 * 1024 * 1024
    minio_secure: bool = False
    # When set, always connect here instead of the request's minio_endpoint (which
    # the backend may fill with a host name only resolvable on its own network).
    minio_public_endpoint: str | None = None
    minio_access_key: str | None = field(default=None, repr=False)
    minio_secret_key: str | None = field(default=None, repr=False)
    log_level: str = "INFO"
    inference: InferenceParams = field(default_factory=InferenceParams)

    def __post_init__(self) -> None:
        if self.max_video_bytes <= 0:
            raise ValueError("PICKLEBALL_MAX_VIDEO_BYTES must be a positive integer")
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError("PICKLEBALL_DEVICE must be auto, cpu, or cuda")
        if self.batch_size < 0:
            raise ValueError("PICKLEBALL_BATCH_SIZE must be >= 0 (0 = auto)")
        if self.preprocess_workers < 0:
            raise ValueError("PICKLEBALL_PREPROCESS_WORKERS must be >= 0")
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError("PICKLEBALL_PRECISION must be auto, fp32, fp16, or bf16")

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return cls(
            checkpoint_path=_project_path(
                os.getenv("PICKLEBALL_CHECKPOINT", str(defaults.checkpoint_path))
            ),
            beats_checkpoint=_project_path(
                os.getenv("PICKLEBALL_BEATS_CHECKPOINT", str(defaults.beats_checkpoint))
            ),
            beats_root=_project_path(
                os.getenv("PICKLEBALL_BEATS_ROOT", str(defaults.beats_root))
            ),
            videomae=os.getenv("PICKLEBALL_VIDEOMAE", defaults.videomae),
            videomae_repo=os.getenv("PICKLEBALL_VIDEOMAE_REPO", defaults.videomae_repo).strip(),
            device=os.getenv("PICKLEBALL_DEVICE", defaults.device).strip().lower(),
            batch_size=int(os.getenv("PICKLEBALL_BATCH_SIZE", str(defaults.batch_size))),
            precision=os.getenv("PICKLEBALL_PRECISION", defaults.precision).strip().lower(),
            gpu_preprocess=_env_bool("PICKLEBALL_GPU_PREPROCESS", defaults.gpu_preprocess),
            preprocess_workers=int(
                os.getenv("PICKLEBALL_PREPROCESS_WORKERS", str(defaults.preprocess_workers))
            ),
            max_video_bytes=int(
                os.getenv("PICKLEBALL_MAX_VIDEO_BYTES", str(defaults.max_video_bytes))
            ),
            minio_secure=_env_bool("MINIO_SECURE", defaults.minio_secure),
            minio_public_endpoint=os.getenv("MINIO_PUBLIC_ENDPOINT", "").strip() or None,
            minio_access_key=os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD"),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
            inference=InferenceParams.from_env(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings.from_env()
