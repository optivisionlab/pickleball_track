import os

from locateanything_worker import LocateAnythingWorker

# Model weights live outside the repo (shared across branches). Override with
# LA_MODEL_PATH, or PB_ARTIFACTS for the artifacts root. Falls back to the HF
# hub id (cached in ~/.cache/huggingface) if no complete local copy exists.
ARTIFACTS_ROOT = os.environ.get("PB_ARTIFACTS", "/data/Cuong/pickleball/artifacts")
LOCAL_MODEL_DIR = os.path.join(ARTIFACTS_ROOT, "locate-anything", "LocateAnything-3B")
HF_MODEL_ID = "nvidia/LocateAnything-3B"


def resolve_model_path():
    if os.environ.get("LA_MODEL_PATH"):
        return os.environ["LA_MODEL_PATH"]
    if os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        return LOCAL_MODEL_DIR
    return HF_MODEL_ID


MODEL_PATH = resolve_model_path()
worker = LocateAnythingWorker(MODEL_PATH)
