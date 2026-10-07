# =========================
# PRETRAINED MODELS
# =========================

DEFAULT_VIDEOMAE = "MCG-NJU/videomae-base"

BEATS_CHECKPOINT = "./weights/BEATs_iter3_plus_AS2M.pt"

BEATS_ROOT = "/data/Cuong/pickleball/pickleball_track/baseline/v3/unilm"


# =========================
# INPUT / PREPROCESSING
# =========================

IMAGE_SIZE = 224

AUDIO_SR = 16000

VIDEO_EXTENSIONS = {".mp4", ".MOV", ".mov"}


# =========================
# TEMPORAL WINDOW
# =========================

NUM_VIDEO_FRAMES = 32

NUM_TEMPORAL_TOKENS = 16

OVERLAP = 0.5


# IMPORTANT:
# Event duration = exactly 1 second
#
# [event_time - 0.5s,
#  event_time + 0.5s]

EVENT_WINDOW_SEC = 1.0

EVENT_RADIUS_SEC = 0.5


# =========================
# LABELS
# =========================

NUM_CLASSES = 3

LABEL2ID = {
    "no_event": 0,
    "hit": 1,
    "miss": 2,
}

ID2LABEL = {
    0: "no_event",
    1: "hit",
    2: "miss",
}


# =========================
# CROSS VALIDATION
# =========================

N_SPLITS = 5

SEED = 42


# =========================
# TRAINING ONLY
# =========================

BATCH_SIZE = 32

NUM_WORKERS = 0

EPOCHS = 50

WARMUP_EPOCHS = 3

FUSION_LR = 3e-4

WEIGHT_DECAY = 1e-2

GRAD_CLIP = 1.0

PATIENCE = 10


# =========================
# INFERENCE
# =========================

MISS_THRESHOLD = 0.5

HIT_THRESHOLD = 0.5

MISS_MARGIN = 0.0

EVENT_MERGE_DISTANCE = 1.0


# =========================
# OUTPUT
# =========================

OUTPUT_DIR = "./checkpoints"

FEATURE_CACHE_DIR = "./feature_cache"