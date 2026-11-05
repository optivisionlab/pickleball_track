"""Shared paths and data loading for the NSVA basketball v2 pipeline.

The clips now live inside two uncompressed (stored) zip files instead of a folder, so
every clip is read from its zip member into /dev/shm, decoded and deleted again.
Rows, labels, games and their order are the same as the v1 pipeline's 1-1 mapping loader,
so make_folds gives exactly the folds of the v1 results.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent))  # baseline root, for common/
from common import media as bc  # noqa: E402

DATA_DIR = "/data/Cuong/pickleball/data/bendmark/basketball"
MAPPING_CSV = f"{DATA_DIR}/1_video_co_nhan_khop_1-1.csv"
ZIPS = [f"{DATA_DIR}/train_part1.zip", f"{DATA_DIR}/train_part2.zip"]
FEATURES = str(HERE / "cache")              # frozen features (extract_features.py)
OUTPUT_ROOT = str(HERE / "outputs" / "runs")  # seed_<s>/<version>/fold_<k>
BEATS_ROOT = bc.BEATS_ROOT
BEATS = bc.BEATS
# The Kinetics-finetuned checkpoint is fine here: Kinetics labels say nothing about NSVA.
BACKBONES = {"ssl": bc.VIDEOMAE_BASE, "k400": "MCG-NJU/videomae-base-finetuned-kinetics"}
CLIP_IMAGE = "openai/clip-vit-base-patch16"
CLIP_TEXT = "openai/clip-vit-base-patch32"
SEED = 42

LABEL2ID = {"no_event": 0, "hit": 1, "miss": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)
DEFAULT_PROMPT = "A basketball game clip."

# View layout of the feature arrays (axis 0).
VIEWS = ["full", "flip", "crop_left", "crop_center", "crop_right", "full_shift", "tail", "tail_flip"]
BASE_VIEWS = [0, 1, 2, 3, 4, 5]   # 32 frames over the whole clip
TAIL_VIEWS = [6, 7]               # 32 frames over the second half (shot outcome)


def tail_view_for(view):
    """Tail view paired with a whole-clip view: flipped views use the flipped tail."""
    import numpy as np
    return np.where(np.asarray(view) == 1, 7, 6)


def zip_index() -> Dict[str, Tuple[str, str]]:
    """mp4 file name -> (zip path, member name)."""
    index = {}
    for path in ZIPS:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.filename.endswith(".mp4"):
                    index[info.filename.rsplit("/", 1)[-1]] = (path, info.filename)
    return index


def load_rows(mapping_csv: str = MAPPING_CSV):
    """Same filtering and order as the v1 1-1 mapping loader, but files are checked in the zips.

    Returns (rows, members): rows are bc.ClipRow with path = mp4 file name.
    """
    members = zip_index()
    with open(mapping_csv, encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    id_count, file_count = {}, {}
    for r in raw:
        id_count[r["video_id"].strip()] = id_count.get(r["video_id"].strip(), 0) + 1
        file_count[r["file"].strip()] = file_count.get(r["file"].strip(), 0) + 1
    rows: List[bc.ClipRow] = []
    for r in raw:
        video_id, name = r["video_id"].strip(), r["file"].strip()
        label, stem = r["label"].strip().lower(), r["gameid_eventid"].strip()
        if (id_count[video_id] > 1 or file_count[name] > 1 or label not in LABEL2ID
                or Path(name).stem != stem or name not in members):
            continue
        rows.append(bc.ClipRow(video_id, stem.split("-")[0], name, LABEL2ID[label]))
    return rows, members


class ZipClip:
    """Context manager: copy one zip member to /dev/shm and yield its path."""

    _open: Dict[str, zipfile.ZipFile] = {}

    def __init__(self, members, name):
        self.zip_path, self.member = members[name]

    def __enter__(self):
        zf = self._open.get(self.zip_path)
        if zf is None:  # one handle per worker process
            zf = self._open[self.zip_path] = zipfile.ZipFile(self.zip_path)
        fd, self.path = tempfile.mkstemp(suffix=".mp4", dir="/dev/shm")
        with os.fdopen(fd, "wb") as out:
            out.write(zf.read(self.member))
        return self.path

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


def encode_texts(texts, device, batch_size: int = 256):
    """[N, 512] f32 numpy, L2-normalised CLIP (ViT-B/32) projected text embeddings."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_TEXT).to(device).eval()
    out = np.zeros((len(texts), 512), np.float32)
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            tok = tokenizer(list(texts[s:s + batch_size]), padding=True, truncation=True, max_length=77,
                            return_tensors="pt").to(device)
            out[s:s + batch_size] = F.normalize(model(**tok).text_embeds.float(), dim=-1).cpu().numpy()
    del model
    return out


def metrics(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "f1_by_class": {ID2LABEL[i]: float(f1[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
    }


def make_folds(labels: np.ndarray, games: np.ndarray, n_splits: int, seed: int):
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(labels, labels, games), start=1):
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + fold)
        fit_rel, val_rel = next(inner.split(train_idx, labels[train_idx], games[train_idx]))
        yield fold, train_idx[fit_rel], train_idx[val_rel], test_idx
