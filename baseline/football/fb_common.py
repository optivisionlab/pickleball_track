"""Shared paths and data loading for the football (SoccerNet shots, 15 s clips) benchmark.

Data: /data/Cuong/pickleball/data/bendmark/football, unzipped into part1..part6:
    part1 = valid, part2 = test, part3-6 = train (a game is never split across parts).
Labels (manifest_15s.csv, column `outcome`): goal / saved / off_target.
Every clip is [anchor - 5 s, anchor + 10 s], so the labelled shot is at t = 5 s.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent))
from common import media as bc  # noqa: E402  (read_audio, pool_tokens, clip_logits, seed_everything)

DATA_DIR = "/data/Cuong/pickleball/data/bendmark/football"
MANIFEST = f"{DATA_DIR}/manifest_15s.csv"
PARTS = {"valid": ["part1"], "test": ["part2"], "train": ["part3", "part4", "part5", "part6"]}
SPLITS = ("train", "valid", "test")
RAW_LABELS = ("goal", "saved", "off_target")  # ids stored in the feature meta.json
# FB_TASK=binary: goal vs saved, with off_target merged into saved ("no goal").
TASK = os.environ.get("FB_TASK", "3class")
assert TASK in ("3class", "binary"), TASK
LABELS = ("goal", "saved") if TASK == "binary" else RAW_LABELS
NUM_CLASSES = len(LABELS)
LABEL2ID = {name: min(i, NUM_CLASSES - 1) for i, name in enumerate(RAW_LABELS)}
TASK_SUFFIX = "_binary" if TASK == "binary" else ""


def task_labels(raw):
    """Raw 3-class ids (meta.json) -> ids of the current task (binary: off_target -> saved)."""
    import numpy as np

    return np.minimum(np.asarray(raw), NUM_CLASSES - 1)

CLIP_SECONDS, ANCHOR_SECONDS = 15.0, 5.0

FEATURES = str(HERE / "cache")
BEATS_ROOT = bc.BEATS_ROOT
BEATS = {
    "base": bc.BEATS["base"],
    "ft": bc.BEATS["ft"],
}
# Kinetics-400 labels say nothing about shot outcomes, so the K400 checkpoint is safe to use.
BACKBONES = {"ssl": bc.VIDEOMAE_BASE, "k400": "MCG-NJU/videomae-base-finetuned-kinetics"}
CLIP_IMAGE = "openai/clip-vit-base-patch16"
CLIP_TEXT = "openai/clip-vit-base-patch32"
DEFAULT_PROMPT = "A soccer match clip of a shot on goal."
# Feature views (axis 0 of the arrays). "focus" = 32 frames over [3 s, 11 s]: the shot (5 s)
# and what follows it, at twice the temporal density of the whole-clip views.
VIEWS = ["full", "full_flip", "focus", "focus_flip"]
FOCUS_SECONDS = (3.0, 11.0)
OUTPUT_ROOT = str(HERE / "outputs" / f"runs{TASK_SUFFIX}")  # seed_<s>/<version>/{result.json, best.pt}
ANALYSIS_DIR = str(HERE / "outputs" / "analysis")
SEED = 42


@dataclass
class Clip:
    index: int
    split: str
    game_idx: int
    label: int
    path: str
    info: Dict[str, str] = field(default_factory=dict)


def file_index() -> Dict[str, str]:
    """'clips15/<split>/<name>.mp4' -> absolute path, over all part folders."""
    out = {}
    for split, parts in PARTS.items():
        for part in parts:
            folder = Path(DATA_DIR) / part / "clips15" / split
            if folder.is_dir():
                for f in folder.glob("*.mp4"):
                    out[f"clips15/{split}/{f.name}"] = str(f)
    return out


def load_clips(require_file: bool = True) -> List[Clip]:
    """Manifest rows in file order; rows without a clip ('failed:' or missing file) are dropped."""
    import pandas as pd

    df = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    files = file_index()
    clips: List[Clip] = []
    for row in df.to_dict("records"):
        rel = row["path"]
        if rel.startswith("failed:") or row["outcome"] not in LABEL2ID:
            continue
        path = files.get(rel, "")
        if require_file and not path:
            continue
        clips.append(Clip(len(clips), row["split"], int(row["game_idx"]), LABEL2ID[row["outcome"]], path, row))
    return clips


def metrics(cm) -> dict:
    import numpy as np

    cm = np.asarray(cm)
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "f1_by_class": {LABELS[i]: float(f1[i]) for i in range(NUM_CLASSES)},
        "recall_by_class": {LABELS[i]: float(recall[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
    }


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
