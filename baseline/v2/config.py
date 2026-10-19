import os

# Data (weights, unilm, checkpoints, caches) lives outside the repo so that
# switching git branches only changes code. Override with PB_ARTIFACTS.
ARTIFACTS_ROOT = os.environ.get("PB_ARTIFACTS", "/data/Cuong/pickleball/artifacts")
VERSION_ARTIFACTS = os.path.join(ARTIFACTS_ROOT, "baseline", "v2")

DEFAULT_VIDEOMAE = "MCG-NJU/videomae-base"
VIDEO_EXTENSIONS = {".mp4", ".MOV", ".mov"}
BEATS_CHECKPOINT = f"{ARTIFACTS_ROOT}/weights/BEATs_iter3_plus_AS2M.pt"
IMAGE_SIZE = 224
AUDIO_SR = 16000
OVERLAP = 0.5
EVENT_RADIUS_SEC = 1.0
BEATS_ROOT = f"{ARTIFACTS_ROOT}/unilm"
N_SPLITS = 5
SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 50
WARMUP_EPOCHS = 3

FUSION_LR = 3e-4
VIDEO_LR = 1e-5
AUDIO_LR = 1e-5
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

FREEZE_EPOCHS = 5
UNFREEZE_VIDEO_BLOCKS = 2
UNFREEZE_AUDIO_BLOCKS = 2

PATIENCE = 10
NUM_CLASSES = 3