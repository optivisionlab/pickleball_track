import torch

ROOT_DIR = "/data/Cuong/pickleball/data/video_capcut"
VIDEO_MODEL_NAME = "./weights/videomae-base"
BEATS_CHECKPOINT = "./weights/BEATs_iter3_plus_AS2M.pt"
NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 8
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = False
INDEX_FILE = "index.json"
SAVE_DIR = "./checkpoints"