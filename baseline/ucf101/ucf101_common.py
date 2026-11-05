"""UCF-101 data helpers shared by extract_features.py, train.py and finetune.py.

Data
    Official recognition splits (classInd.txt, trainlist0k.txt, testlist0k.txt).
    The 25 groups (gXX) never cross train/test: split k tests on groups 7k-6..7k.
    Clips without audio or with a silent track (analysis/ucf101_videos.csv from
    analyze_ucf101.py) get has_audio=False.
Text
    3 readable CLIP prompts per class name (no sample information), for the text-anchored
    heads of v4 / v5.
Metrics
    top-1 / top-5 / mean class accuracy / macro-F1; top-1 averaged over the 3 splits is the
    number papers report. Model selection holds out 3 whole training groups.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from common import media as bc  # noqa: E402  (shared media helpers)

NAME_RE = re.compile(r"v_[A-Za-z]+_g(?P<group>\d+)_c\d+\.avi$")
SILENT_RMS = 1e-3


# ============================================================
# Data
# ============================================================

def read_class_index(split_dir: Path) -> List[str]:
    names = {}
    for line in (split_dir / "classInd.txt").read_text().split("\n"):
        if line.strip():
            idx, name = line.split()
            names[int(idx) - 1] = name
    return [names[i] for i in range(len(names))]


def load_dataset(video_root: str, split_dir: str, analysis_csv: Optional[str]):
    """All UCF-101 clips as bc.ClipRow (game_id = group) plus split membership."""
    video_root, split_dir = Path(video_root), Path(split_dir)
    classes = read_class_index(split_dir)
    lower = {c.lower(): i for i, c in enumerate(classes)}

    paths = {p.name.lower(): p for p in video_root.rglob("*.avi")}
    rows, index = [], {}
    for name in sorted(paths):
        path = paths[name]
        label = lower.get(path.parent.name.lower())
        m = NAME_RE.search(path.name)
        if label is None or not m:
            continue
        index[name] = len(rows)
        rows.append(bc.ClipRow(path.stem, f"g{int(m['group']):02d}", str(path), label))

    splits = {}
    for k in (1, 2, 3):
        split = {}
        for part in ("train", "test"):
            names = [l.split()[0].split("/")[-1].lower()
                     for l in (split_dir / f"{part}list0{k}.txt").read_text().split("\n") if l.strip()]
            missing = [n for n in names if n not in index]
            if missing:
                raise FileNotFoundError(f"split{k} {part}: {len(missing)} files missing, e.g. {missing[:3]}")
            split[part] = np.array([index[n] for n in names])
        splits[k] = split

    audible = None
    if analysis_csv and Path(analysis_csv).is_file():
        with open(analysis_csv, newline="") as handle:
            rms = {Path(r["file"]).name.lower(): float(r["audio_rms"] or 0) for r in csv.DictReader(handle)}
        audible = torch.tensor([rms.get(Path(r.path).name.lower(), 0.0) > SILENT_RMS for r in rows])
    return rows, classes, splits, audible


def class_prompts(name: str) -> List[str]:
    """Readable CLIP prompts for a CamelCase UCF class name (no sample information)."""
    words = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name).lower()
    words = words.replace("yo yo", "yo-yo").replace("tai chi", "tai chi")
    return [f"a video of a person {words}.", f"a video of {words}.", f"a photo of a person doing {words}."]


@torch.no_grad()
def encode_class_text(clip_name: str, classes: List[str], device: torch.device) -> torch.Tensor:
    """[C, D] normalised CLIP text embeddings, prompt ensemble per class."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    model = CLIPTextModelWithProjection.from_pretrained(clip_name).to(device).eval()
    rows = []
    for name in classes:
        tokens = tokenizer(class_prompts(name), padding=True, truncation=True, max_length=77,
                           return_tensors="pt").to(device)
        emb = F.normalize(model(**tokens).text_embeds.float(), dim=-1).mean(0)
        rows.append(F.normalize(emb, dim=-1))
    del model
    text = torch.stack(rows)
    sim = (text @ text.t()).fill_diagonal_(-1)
    j = sim.argmax(1)
    worst = sim.max(1).values.topk(5).indices.tolist()
    print("[TEXT] most similar class prompts: "
          + ", ".join(f"{classes[i]}~{classes[j[i]]} ({sim[i, j[i]]:.3f})" for i in worst))
    return text


# ============================================================
# Features (VideoMAE original + flip, BEATs)
# ============================================================


def metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    pred = logits.argmax(-1)
    top5 = logits.topk(5, dim=-1).indices.eq(labels[:, None]).any(-1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    cm.index_put_((labels.cpu(), pred.cpu()), torch.ones(len(labels), dtype=torch.long), accumulate=True)
    cm = cm.numpy().astype(np.float64)
    tp = np.diag(cm)
    recall = tp / np.maximum(cm.sum(1), 1)
    precision = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = cm.sum(1) > 0
    return {
        "top1": float((pred == labels).float().mean()),
        "top5": float(top5.float().mean()),
        "mean_class_acc": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class_acc": recall.tolist(),
    }


def val_groups_split(train_idx: np.ndarray, games: np.ndarray, n_groups: int, seed: int):
    """Hold out whole training groups (every class has all 25 groups) for model selection."""
    groups = sorted(set(games[train_idx].tolist()))
    rng = np.random.default_rng(seed)
    val = set(rng.choice(groups, size=n_groups, replace=False).tolist())
    mask = np.array([games[i] in val for i in train_idx])
    return train_idx[~mask], train_idx[mask], sorted(val)
