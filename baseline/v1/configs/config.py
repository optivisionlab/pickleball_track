import os

# Data (weights, unilm, checkpoints, caches) lives outside the repo so that
# switching git branches only changes code. Override with PB_ARTIFACTS.
ARTIFACTS_ROOT = os.environ.get("PB_ARTIFACTS", "/data/Cuong/pickleball/artifacts")
VERSION_ARTIFACTS = os.path.join(ARTIFACTS_ROOT, "baseline", "v1")

import torch

DATA_DIR = "/data/Cuong/pickleball/data/video_capcut"
VIDEO_MODEL_NAME = f"{ARTIFACTS_ROOT}/weights/videomae-base"
BEATS_CHECKPOINT = f"{ARTIFACTS_ROOT}/weights/BEATs_iter3_plus_AS2M.pt"
NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 8
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
INDEX_FILE = "index.json"
BEATS_ROOT = f"{ARTIFACTS_ROOT}/unilm"
SAVE_DIR = f"{VERSION_ARTIFACTS}/checkpoints"