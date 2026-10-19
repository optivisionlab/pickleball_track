"""Shared NSVA basketball benchmark pipeline for baseline v3 / v4 / v5.

Data
    Only rows of the explicit 1-1 mapping CSV
    (video_id, gameid_eventid, file, split, label) whose file exists, appears
    exactly once (labels come from the mapping file; differences from
    playable_nsva_labels.csv, e.g. Jump Ball relabeled to no_event, are reported).

Features (extracted once on the GPU, shared by every version)
    VideoMAE: 32 frames sampled uniformly over the WHOLE clip -> 16 tokens.
    BEATs:    16 kHz mono audio of the whole clip, pooled to a fixed token count.

Training
    Clip-level classification: token logits are mean-pooled over valid tokens
    and trained with cross-entropy (no focal loss). Folds are
    StratifiedGroupKFold over game_id so no game appears in train and test;
    early stopping uses an inner validation split carved from the training
    games, and the held-out fold is only used for the reported test metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LABEL2ID = {"no_event": 0, "hit": 1, "miss": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

NUM_VIDEO_FRAMES = 32
NUM_TEMPORAL_TOKENS = 16
IMAGE_SIZE = 224
AUDIO_SR = 16000
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_CACHE_VERSION = "nsva-1to1-v1"

DEFAULT_VIDEO_DIR = "/data/Cuong/pickleball/data/bendmark/train"
DEFAULT_LABELS_CSV = "/data/Cuong/pickleball/data/bendmark/playable_nsva_labels.csv"
DEFAULT_MAPPING_CSV = "/data/Cuong/pickleball/data/bendmark/1_video_co_nhan_khop_1-1.csv"
DEFAULT_CLIP = "openai/clip-vit-base-patch32"
DEFAULT_PROMPT = "A basketball game clip."


# ============================================================
# 1-1 mapping
# ============================================================

@dataclass(frozen=True)
class ClipRow:
    video_id: str
    game_id: str
    path: str
    label: int


def load_mapping(mapping_csv: str, video_dir: str, labels_csv: Optional[str]):
    """Return the clean 1-1 rows plus a report; the mapping file's labels are authoritative."""
    mapping_csv, video_dir = Path(mapping_csv), Path(video_dir)
    if not mapping_csv.is_file():
        raise FileNotFoundError(
            f"Mapping CSV not found: {mapping_csv}. Copy the 1-1 mapping file "
            "(video_id,gameid_eventid,file,split,label) there or pass --mapping-csv."
        )

    with mapping_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video_id", "gameid_eventid", "file", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{mapping_csv} must contain columns {sorted(required)}.")

    reference = {}
    if labels_csv and Path(labels_csv).is_file():
        with open(labels_csv, encoding="utf-8-sig", newline="") as handle:
            reference = {
                r["video_id"].strip(): r["label"].strip().lower()
                for r in csv.DictReader(handle)
            }

    id_count, file_count = {}, {}
    for r in rows:
        id_count[r["video_id"].strip()] = id_count.get(r["video_id"].strip(), 0) + 1
        file_count[r["file"].strip()] = file_count.get(r["file"].strip(), 0) + 1

    drops = {
        "duplicate_video_id": 0, "duplicate_file": 0, "unknown_label": 0,
        "file_missing": 0, "file_name_mismatch": 0,
    }
    relabeled = 0
    clean: List[ClipRow] = []
    split_counts: Dict[str, int] = {}
    for r in rows:
        video_id, file_name = r["video_id"].strip(), r["file"].strip()
        label_name = r["label"].strip().lower()
        stem = r["gameid_eventid"].strip()
        split_counts[r.get("split", "")] = split_counts.get(r.get("split", ""), 0) + 1
        if id_count[video_id] > 1:
            drops["duplicate_video_id"] += 1
        elif file_count[file_name] > 1:
            drops["duplicate_file"] += 1
        elif label_name not in LABEL2ID:
            drops["unknown_label"] += 1
        elif Path(file_name).stem != stem:
            drops["file_name_mismatch"] += 1
        elif not (video_dir / file_name).is_file():
            drops["file_missing"] += 1
        else:
            if reference and reference.get(video_id) not in (None, label_name):
                relabeled += 1  # the mapping file's label is authoritative
            clean.append(ClipRow(video_id, stem.split("-")[0],
                                 str(video_dir / file_name), LABEL2ID[label_name]))

    on_disk = {p.name for p in video_dir.glob("*.mp4")}
    used = {Path(c.path).name for c in clean}
    report = {
        "mapping_csv": str(mapping_csv),
        "mapping_rows": len(rows),
        "rows_used": len(clean),
        "dropped": drops,
        "rows_relabeled_vs_labels_csv": relabeled,
        "split_column": split_counts,
        "label_counts": {ID2LABEL[i]: sum(c.label == i for c in clean) for i in range(NUM_CLASSES)},
        "games": len({c.game_id for c in clean}),
        "video_files_on_disk": len(on_disk),
        "video_files_without_mapping": len(on_disk - used),
        "mapping_rows_not_in_labels_csv": (
            sum(r["video_id"].strip() not in reference for r in rows) if reference else None
        ),
    }
    return clean, report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("1-1 mapping check")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("=" * 60)


# ============================================================
# Decoding (CPU workers)
# ============================================================

def uniform_indices(num_frames: int, count: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros(count, dtype=np.int64)
    return np.linspace(0, num_frames - 1, count).round().astype(np.int64)


def read_frames(path: str, count: int, size: int):
    """Return uint8 [count, size, size, 3] sampled uniformly over the clip."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = uniform_indices(total, count)
    frames = np.zeros((count, size, size, 3), dtype=np.uint8)
    valid = np.zeros(count, dtype=bool)
    position, decoded_at, last = -1, -1, None
    for slot, target in enumerate(wanted):
        while position < target and cap.grab():  # grab() skips the costly retrieve
            position += 1
        if position == target and decoded_at != target:
            ok, frame = cap.retrieve()
            if ok:
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                last, decoded_at = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), target
        valid[slot] = decoded_at == target
        if last is not None:
            frames[slot] = last  # past the real end: repeat the last frame (masked)
    cap.release()
    return frames, valid


def read_audio(path: str, sample_rate: int = AUDIO_SR) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "f32le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
        return np.frombuffer(out, dtype=np.float32).copy()
    except (subprocess.SubprocessError, OSError):
        return np.zeros(0, dtype=np.float32)


class ClipDecodeDataset(Dataset):
    def __init__(self, rows: List[ClipRow]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        path = self.rows[index].path
        frames, valid = read_frames(path, NUM_VIDEO_FRAMES, IMAGE_SIZE)
        return {
            "index": index,
            "frames": torch.from_numpy(frames),
            "frame_valid": torch.from_numpy(valid),
            "audio": torch.from_numpy(read_audio(path)),
        }


def decode_collate(batch):
    length = max(1, max(item["audio"].numel() for item in batch))
    audio = torch.zeros(len(batch), length)
    audio_pad = torch.ones(len(batch), length, dtype=torch.bool)
    for row, item in enumerate(batch):
        n = item["audio"].numel()
        audio[row, :n] = item["audio"]
        audio_pad[row, :n] = False
    return {
        "index": torch.tensor([item["index"] for item in batch]),
        "frames": torch.stack([item["frames"] for item in batch]),
        "frame_valid": torch.stack([item["frame_valid"] for item in batch]),
        "audio": audio,
        "audio_pad": audio_pad,
    }


# ============================================================
# Pretrained encoders (GPU)
# ============================================================

def load_videomae(name: str, device: torch.device):
    from transformers import VideoMAEModel

    model = VideoMAEModel.from_pretrained(name)
    tubelet = int(model.config.tubelet_size)
    old_t, new_t = int(model.config.num_frames) // tubelet, NUM_VIDEO_FRAMES // tubelet
    if old_t != new_t:
        size = model.config.image_size
        patch = model.config.patch_size
        side = (size if isinstance(size, int) else size[0]) // (patch if isinstance(patch, int) else patch[0])
        pos = model.embeddings.position_embeddings
        hidden = pos.shape[-1]
        grid = pos.reshape(1, old_t, side, side, hidden).permute(0, 4, 1, 2, 3)
        grid = F.interpolate(grid, size=(new_t, side, side), mode="trilinear", align_corners=False)
        model.embeddings.position_embeddings = nn.Parameter(
            grid.permute(0, 2, 3, 4, 1).reshape(1, new_t * side * side, hidden).contiguous(),
            requires_grad=False,
        )
        model.config.num_frames = NUM_VIDEO_FRAMES
    return model.to(device).eval().requires_grad_(False)


def load_beats(checkpoint_path: str, beats_root: str, device: torch.device):
    for extra in (beats_root, os.path.join(beats_root, "beats")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from beats.BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_prompt(clip_name: str, prompt: str, device: torch.device) -> torch.Tensor:
    """One CLIP text embedding [D]; the same for every clip (no label information)."""
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    model = CLIPTextModel.from_pretrained(clip_name).to(device).eval()
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=77).to(device)
    feature = model(**tokens).pooler_output[0].float()
    del model
    return feature


def pool_tokens(features: torch.Tensor, valid: torch.Tensor, count: int):
    """Masked adaptive average pooling of [T, D] valid tokens to [count, D]."""
    tokens = features[valid]
    if tokens.shape[0] == 0:
        return torch.zeros(count, features.shape[-1], device=features.device), False
    pooled = F.adaptive_avg_pool1d(tokens.t().unsqueeze(0).float(), count)[0].t()
    return pooled, True


def feature_cache_key(rows: List[ClipRow], videomae: str, beats: str, audio_tokens: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"{FEATURE_CACHE_VERSION}|{videomae}|{beats}|{audio_tokens}|"
                  f"{NUM_VIDEO_FRAMES}|{IMAGE_SIZE}|{IMAGENET_MEAN}|{IMAGENET_STD}".encode())
    for r in rows:
        stat = os.stat(r.path)
        digest.update(f"{r.video_id}|{r.path}|{stat.st_size}|{stat.st_mtime_ns}|{r.label}".encode())
    return digest.hexdigest()


def extract_features(
    rows: List[ClipRow],
    cache_dir: str,
    videomae_name: str,
    beats_checkpoint: str,
    beats_root: str,
    device: torch.device,
    batch_size: int = 16,
    workers: int = 16,
    audio_tokens: int = 32,
    rebuild: bool = False,
) -> dict:
    """Extract (or load) the shared feature file for exactly these rows."""
    key = feature_cache_key(rows, videomae_name, beats_checkpoint, audio_tokens)
    cache_path = Path(cache_dir) / f"features_{key[:16]}.pt"
    if cache_path.is_file() and not rebuild:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        if data.get("key") == key:
            print(f"[FEATURES] Loaded {cache_path} ({len(data['labels'])} clips).")
            return data
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    videomae = load_videomae(videomae_name, device)
    beats = load_beats(beats_checkpoint, beats_root, device)
    hidden = int(videomae.config.hidden_size)
    audio_dim = int(beats.cfg.encoder_embed_dim)
    tubelet = int(videomae.config.tubelet_size)
    temporal = NUM_VIDEO_FRAMES // tubelet
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    n = len(rows)
    video = torch.zeros(n, temporal, hidden, dtype=torch.float16)
    video_pad = torch.ones(n, temporal, dtype=torch.bool)
    audio = torch.zeros(n, audio_tokens, audio_dim, dtype=torch.float16)
    has_audio = torch.zeros(n, dtype=torch.bool)

    loader = DataLoader(
        ClipDecodeDataset(rows), batch_size=batch_size, num_workers=workers,
        collate_fn=decode_collate, pin_memory=device.type == "cuda",
        prefetch_factor=4 if workers > 0 else None, persistent_workers=False,
    )
    start = time.time()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="VideoMAE+BEATs"):
            index = batch["index"]
            frames = batch["frames"].to(device, non_blocking=True)  # [B,T,H,W,3] uint8
            pixels = frames.permute(0, 1, 4, 2, 3).float().div_(255.0)
            pixels = (pixels - mean) / std
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                tokens = videomae(pixel_values=pixels).last_hidden_state
            tokens = tokens.float().reshape(len(index), temporal, -1, hidden).mean(2)
            video[index] = tokens.half().cpu()
            frame_valid = batch["frame_valid"].reshape(len(index), temporal, tubelet).any(-1)
            video_pad[index] = ~frame_valid

            wave = batch["audio"].to(device, non_blocking=True)
            wave_pad = batch["audio_pad"].to(device, non_blocking=True)
            usable = (~wave_pad).sum(1) >= AUDIO_SR // 4
            if usable.any():
                feats, feat_pad = beats.extract_features(wave[usable], padding_mask=wave_pad[usable])
                for row, (f, p) in zip(index[usable.cpu()].tolist(), zip(feats, feat_pad)):
                    pooled, ok = pool_tokens(f, ~p, audio_tokens)
                    audio[row] = pooled.half().cpu()
                    has_audio[row] = ok
    print(f"[FEATURES] Extracted {n} clips in {time.time() - start:.0f}s; "
          f"clips without audio: {int((~has_audio).sum())}.")

    data = {
        "key": key,
        "video_ids": [r.video_id for r in rows],
        "games": [r.game_id for r in rows],
        "labels": torch.tensor([r.label for r in rows]),
        "video": video,
        "video_pad": video_pad,
        "audio": audio,
        "has_audio": has_audio,
    }
    tmp = cache_path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(cache_path)
    print(f"[FEATURES] Saved {cache_path}.")
    del videomae, beats
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return data


# ============================================================
# Clip-level training with game-grouped folds
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip_logits(token_logits: torch.Tensor, video_pad: torch.Tensor) -> torch.Tensor:
    """Mean of token logits over valid tokens: [B,T,C] -> [B,C]."""
    weight = (~video_pad).unsqueeze(-1).to(token_logits.dtype)
    return (token_logits * weight).sum(1) / weight.sum(1).clamp_min(1.0)


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


def _batches(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(indices) if shuffle else indices
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def run_kfold(
    data: dict,
    build_model: Callable[[], nn.Module],
    forward: Callable[[nn.Module, dict], torch.Tensor],
    args,
    device: torch.device,
    output_dir: str,
    extra_inputs: Optional[dict] = None,
) -> dict:
    """Train/evaluate one model per game-grouped fold; all features stay on the GPU."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    labels_np = data["labels"].numpy()
    games_np = np.asarray(data["games"])
    tensors = {
        "video": data["video"].to(device),
        "video_pad": data["video_pad"].to(device),
        "audio": data["audio"].to(device),
        # Clips without audio keep one all-zero token unmasked so attention stays defined.
        "audio_pad": torch.zeros(data["audio"].shape[:2], dtype=torch.bool, device=device),
        "labels": data["labels"].to(device),
    }
    no_audio = ~data["has_audio"].to(device)
    tensors["audio_pad"][no_audio, 1:] = True
    extra_inputs = extra_inputs or {}
    use_amp = device.type == "cuda" and not args.no_amp

    def gather(idx):
        idx_t = torch.as_tensor(idx, device=device)
        batch = {k: v[idx_t] for k, v in tensors.items()}
        batch["video"] = batch["video"].float()
        batch["audio"] = batch["audio"].float()
        batch.update({k: v.expand(len(idx), *v.shape) for k, v in extra_inputs.items()})
        return batch

    def evaluate(model, idx):
        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        total, count = 0.0, 0
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            for chunk in _batches(idx, args.batch_size * 2, False, None):
                batch = gather(chunk)
                logits = clip_logits(forward(model, batch).float(), batch["video_pad"])
                total += float(F.cross_entropy(logits, batch["labels"], reduction="sum"))
                count += len(chunk)
                pred = logits.argmax(-1).cpu().numpy()
                np.add.at(cm, (batch["labels"].cpu().numpy(), pred), 1)
        result = metrics(cm)
        result["loss"] = total / max(count, 1)
        return result

    summary = []
    for fold, fit_idx, val_idx, test_idx in make_folds(labels_np, games_np, args.folds, args.seed):
        seed_everything(args.seed + fold)
        rng = np.random.default_rng(args.seed + fold)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[FOLD {fold}] fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)} clips | "
              f"games fit/val/test={len(set(games_np[fit_idx]))}/{len(set(games_np[val_idx]))}/"
              f"{len(set(games_np[test_idx]))}")

        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(fit_idx) / args.batch_size)
        total_steps = args.epochs * steps_per_epoch
        warmup = args.warmup_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: (s + 1) / max(1, warmup) if s < warmup
            else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup)))),
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best_f1, best_epoch, stale, history = -1.0, 0, 0, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            running = 0.0
            for chunk in _batches(fit_idx, args.batch_size, True, rng):
                batch = gather(chunk)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = clip_logits(forward(model, batch).float(), batch["video_pad"])
                    loss = criterion(logits, batch["labels"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                running += loss.detach().item() * len(chunk)
                np.add.at(cm, (batch["labels"].cpu().numpy(), logits.argmax(-1).cpu().numpy()), 1)
            train_metrics = metrics(cm)
            train_metrics["loss"] = running / len(fit_idx)
            val_metrics = evaluate(model, val_idx)
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            print(f"[FOLD {fold}] epoch {epoch:03d} train loss={train_metrics['loss']:.4f} "
                  f"F1={train_metrics['macro_f1']:.4f} | val loss={val_metrics['loss']:.4f} "
                  f"F1={val_metrics['macro_f1']:.4f}")
            if val_metrics["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics,
                            "label2id": LABEL2ID, "args": vars(args)}, fold_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"[FOLD {fold}] early stopping at epoch {epoch}.")
                    break

        checkpoint = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test_metrics = evaluate(model, test_idx)
        print(f"[FOLD {fold}] best epoch {best_epoch}: val F1={best_f1:.4f} | "
              f"TEST acc={test_metrics['accuracy']:.4f} F1={test_metrics['macro_f1']:.4f} "
              f"{test_metrics['f1_by_class']}")
        result = {
            "fold": fold, "best_epoch": best_epoch, "val_macro_f1": best_f1, "test": test_metrics,
            "test_games": sorted(set(games_np[test_idx].tolist())),
        }
        with (fold_dir / "history.json").open("w") as handle:
            json.dump(history, handle, indent=2)
        with (fold_dir / "result.json").open("w") as handle:
            json.dump(result, handle, indent=2)
        summary.append(result)
        del model, optimizer

    f1s = np.array([r["test"]["macro_f1"] for r in summary])
    accs = np.array([r["test"]["accuracy"] for r in summary])
    report = {
        "folds": summary,
        "test_macro_f1_mean": float(f1s.mean()), "test_macro_f1_std": float(f1s.std()),
        "test_accuracy_mean": float(accs.mean()), "test_accuracy_std": float(accs.std()),
    }
    with (output / "kfold_summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    with (output / "kfold_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "best_epoch", "val_macro_f1", "test_macro_f1", "test_accuracy"])
        for r in summary:
            writer.writerow([r["fold"], r["best_epoch"], r["val_macro_f1"],
                             r["test"]["macro_f1"], r["test"]["accuracy"]])
        writer.writerow(["mean", "", "", report["test_macro_f1_mean"], report["test_accuracy_mean"]])
        writer.writerow(["std", "", "", report["test_macro_f1_std"], report["test_accuracy_std"]])
    print(f"\n[RESULT] test macro-F1 = {report['test_macro_f1_mean']:.4f} ± "
          f"{report['test_macro_f1_std']:.4f} | acc = {report['test_accuracy_mean']:.4f}")
    return report


# ============================================================
# CLI shared by v3 / v4 / v5
# ============================================================

def add_common_args(parser, artifacts_root: str, version_artifacts: str, cfg) -> None:
    parser.add_argument("--mapping-csv", default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    parser.add_argument("--videomae", default=cfg.DEFAULT_VIDEOMAE)
    parser.add_argument("--beats-checkpoint", default=cfg.BEATS_CHECKPOINT)
    parser.add_argument("--beats-root", default=cfg.BEATS_ROOT)
    parser.add_argument("--feature-cache-dir",
                        default=f"{artifacts_root}/baseline/benchmark_features",
                        help="Shared by v3/v4/v5: features are extracted once.")
    parser.add_argument("--output-dir", default=f"{version_artifacts}/checkpoints_benchmark_1to1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=cfg.FUSION_LR)
    parser.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=cfg.WARMUP_EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--grad-clip", type=float, default=cfg.GRAD_CLIP)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 8))
    parser.add_argument("--audio-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--check-only", action="store_true",
                        help="Only validate the 1-1 mapping and exit.")


def prepare(args, device: torch.device):
    """Validate the mapping, write the report, and return the shared features."""
    rows, report = load_mapping(args.mapping_csv, args.video_dir, args.labels_csv)
    print_report(report)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "mapping_report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    if args.check_only:
        return None
    return extract_features(
        rows, args.feature_cache_dir, args.videomae, args.beats_checkpoint, args.beats_root,
        device, args.extract_batch_size, args.workers, args.audio_tokens, args.rebuild_features,
    )
