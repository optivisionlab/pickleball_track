import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = PROJECT_ROOT / "src" / "weights"


def _project_path(environment_variable, default):
    path = Path(os.getenv(environment_variable, str(default))).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


DEFAULT_CHECKPOINT_PATH = _project_path(
    "PICKLEBALL_CHECKPOINT",
    WEIGHTS_DIR / "best.pt",
)
BEATS_CHECKPOINT = str(
    _project_path(
        "PICKLEBALL_BEATS_CHECKPOINT",
        WEIGHTS_DIR / "BEATs_iter3_plus_AS2M.pt",
    )
)
BEATS_ROOT = str(
    _project_path("PICKLEBALL_BEATS_ROOT", PROJECT_ROOT / "vendor" / "unilm")
)
DEFAULT_VIDEOMAE = os.getenv(
    "PICKLEBALL_VIDEOMAE",
    str(WEIGHTS_DIR / "videomae-base"),
)
MODEL_DEVICE = os.getenv("PICKLEBALL_DEVICE", "auto").strip().lower()
MAX_VIDEO_BYTES = int(os.getenv("PICKLEBALL_MAX_VIDEO_BYTES", str(512 * 1024 * 1024)))
if MAX_VIDEO_BYTES <= 0:
    raise ValueError("PICKLEBALL_MAX_VIDEO_BYTES must be a positive integer")


AUDIO_SR = 16000


# =========================
# TEMPORAL WINDOW
# =========================

NUM_VIDEO_FRAMES = 32

NUM_TEMPORAL_TOKENS = 16

OVERLAP = 0.5


LABEL2ID = {
    "no_event": 0,
    "hit": 1,
    "miss": 2,
}
