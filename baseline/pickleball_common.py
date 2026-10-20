"""Shared pickleball pipeline for v3 / v4 / v5.

Each version's train.py only builds its model and says how to call it; everything
else lives here so the three versions are compared on exactly the same data:

* Frames are sampled by time (TARGET_FPS), not by index, so a 32-frame window
  always covers the same ~1.07 s whether the video is 26, 30 or 60 fps.
* Features are extracted once into a shared cache:
    - VideoMAE (fine-tuned checkpoint, ImageNet-normalized input). Every temporal
      token keeps the global mean plus a 2x2 spatial grid, so a small ball is not
      averaged away with 196 patches.
    - BEATs runs once per video on 10 s chunks; its time steps are interpolated at
      the centre time of every video token, so audio and video tokens are aligned.
* Labels are built from the JSON events at training time (radius is a flag).
  The file name (Trung/Truot, TraiTay/PhaiTay, user) is NEVER a model input:
  Trung/Truot alone gives the hit/miss label of 99.8% of the events.
* Evaluation: outer folds grouped by user (test), an inner user-grouped split for
  early stopping (val). Test predictions of all folds are pooled before computing
  metrics, so a fold without a class (e.g. a user with no miss) does not distort
  the score. Metrics are reported per token and per event (detected strokes).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import benchmark_common as bc

LABEL2ID = {"no_event": 0, "hit": 1, "miss": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

TARGET_FPS = 30.0
NUM_VIDEO_FRAMES = 32
FRAMES_PER_TOKEN = 2
NUM_TEMPORAL_TOKENS = NUM_VIDEO_FRAMES // FRAMES_PER_TOKEN
WINDOW_STRIDE = 16  # sampled frames, i.e. 50% overlap; must be even
TOKEN_SEC = FRAMES_PER_TOKEN / TARGET_FPS
IMAGE_SIZE = 224
AUDIO_SR = 16000
BEATS_CHUNK_SEC = 10.0
BEATS_PATCH_SEC = 0.16  # one BEATs time patch = 16 fbank frames of 10 ms
SPATIAL_GRID = 2
VIDEO_EXTENSIONS = {".mp4", ".mov"}
FEATURE_CACHE_VERSION = "pickleball-time30-grid2-v1"

DEFAULT_DATA_DIR = "/data/Cuong/pickleball/data/raw"
DEFAULT_VIDEOMAE = "MCG-NJU/videomae-base-finetuned-kinetics"
DEFAULT_CLIP = "openai/clip-vit-base-patch32"

# Text input of v4/v5: a description of every class (never of the sample).
# Several paraphrases per class are averaged in CLIP space (prompt ensembling).
CLASS_PROMPTS = {
    "no_event": [
        "a pickleball player waiting between shots without hitting the ball",
        "no stroke happens: the player is moving or getting ready",
        "a quiet moment in a pickleball drill with no paddle-ball contact",
        "the ball is not being struck at this moment",
    ],
    "hit": [
        "a pickleball player swings the paddle and hits the ball successfully",
        "a successful pickleball shot with clean paddle-ball contact",
        "the paddle strikes the ball and the shot goes in",
        "a good pickleball stroke that reaches its target",
    ],
    "miss": [
        "a pickleball player swings the paddle but misses the shot",
        "a failed pickleball shot: the ball is mishit or goes out",
        "the player fails to return the ball properly",
        "an unsuccessful pickleball stroke that misses its target",
    ],
}


# ============================================================
# Videos and labels
# ============================================================

@dataclass
class VideoInfo:
    path: str
    json_path: str
    user: str
    fps: float
    num_frames: int
    duration: float
    num_sampled: int
    events: List[Tuple[float, int]] = field(default_factory=list)


def load_events(json_path: str) -> List[Tuple[float, int]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw = data.get("events", []) if isinstance(data, dict) else data
    events = []
    for item in raw if isinstance(raw, list) else []:
        try:
            t, label = float(item["time"]), str(item["label"]).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue
        if label in ("hit", "miss") and t >= 0:
            events.append((t, LABEL2ID[label]))
    return sorted(events)


def discover_videos(data_dir: str) -> List[VideoInfo]:
    videos = []
    for path in sorted(Path(data_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        json_path = path.with_suffix(".json")
        if not json_path.exists():
            print(f"[WARNING] Missing JSON: {path}")
            continue
        parts = path.stem.split("_")
        if len(parts) < 4:
            raise ValueError(f"Cannot read user from {path}; expected DD_MM_YYYY_USER_...")
        cap = cv2.VideoCapture(str(path))
        fps, frames = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"Invalid metadata for {path}: fps={fps}, frames={frames}")
        duration = frames / fps
        events = [e for e in load_events(str(json_path)) if e[0] <= duration]
        videos.append(VideoInfo(
            path=str(path), json_path=str(json_path), user=parts[3], fps=fps,
            num_frames=frames, duration=duration,
            num_sampled=max(1, int(math.floor(duration * TARGET_FPS))), events=events,
        ))
    if not videos:
        raise RuntimeError(f"No video/json pairs found in {data_dir}")
    return videos


def window_starts(num_sampled: int) -> List[int]:
    """Even window starts (in sampled frames) so tokens of all windows share one grid."""
    if num_sampled <= NUM_VIDEO_FRAMES:
        return [0]
    last = (num_sampled - NUM_VIDEO_FRAMES) // 2 * 2
    starts = list(range(0, last + 1, WINDOW_STRIDE))
    if starts[-1] != last:
        starts.append(last)
    return starts


def token_center_times(start: int) -> np.ndarray:
    frames = start + np.arange(NUM_TEMPORAL_TOKENS) * FRAMES_PER_TOKEN
    return (frames + (FRAMES_PER_TOKEN - 1) / 2) / TARGET_FPS


def token_labels(events: List[Tuple[float, int]], start: int, radius: float) -> np.ndarray:
    """Token label = class of an event within `radius` s of one of its frames; hit wins."""
    labels = np.zeros(NUM_TEMPORAL_TOKENS, dtype=np.int64)
    if not events:
        return labels
    times = np.array([e[0] for e in events])
    classes = np.array([e[1] for e in events])
    for token in range(NUM_TEMPORAL_TOKENS):
        frame = start + token * FRAMES_PER_TOKEN
        t0, t1 = frame / TARGET_FPS, (frame + FRAMES_PER_TOKEN - 1) / TARGET_FPS
        near = classes[(times >= t0 - radius) & (times <= t1 + radius)]
        if (near == LABEL2ID["hit"]).any():
            labels[token] = LABEL2ID["hit"]
        elif (near == LABEL2ID["miss"]).any():
            labels[token] = LABEL2ID["miss"]
    return labels


# ============================================================
# Decoding (CPU workers, one whole video per item)
# ============================================================

def decode_sampled(info: VideoInfo, size: int = IMAGE_SIZE):
    """uint8 [num_sampled, size, size, 3] sampled at TARGET_FPS, plus a validity mask."""
    targets = np.minimum(
        np.round(np.arange(info.num_sampled) * info.fps / TARGET_FPS).astype(np.int64),
        info.num_frames - 1,
    )
    frames = np.zeros((info.num_sampled, size, size, 3), dtype=np.uint8)
    valid = np.zeros(info.num_sampled, dtype=bool)
    cap = cv2.VideoCapture(info.path)
    position, decoded_at, last = -1, -1, None
    for slot, target in enumerate(targets):
        while position < target and cap.grab():
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


class VideoDecodeDataset(Dataset):
    def __init__(self, videos: List[VideoInfo], indices: List[int]):
        self.videos, self.indices = videos, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        cv2.setNumThreads(1)
        index = self.indices[item]
        frames, valid = decode_sampled(self.videos[index])
        return {
            "index": index,
            "frames": torch.from_numpy(frames),
            "valid": torch.from_numpy(valid),
            "audio": torch.from_numpy(bc.read_audio(self.videos[index].path, AUDIO_SR)),
        }


# ============================================================
# Feature extraction (GPU)
# ============================================================

@torch.inference_mode()
def beats_timeline(beats, wave: torch.Tensor, device: torch.device):
    """BEATs features of a whole recording: [Ta, D] (mean over frequency) and centre times [Ta]."""
    if wave.numel() < AUDIO_SR // 2:
        return None, None
    chunk = int(BEATS_CHUNK_SEC * AUDIO_SR)
    feats, times = [], []
    for begin in range(0, wave.numel(), chunk):
        segment = wave[begin:begin + chunk]
        length = segment.numel()
        if length < AUDIO_SR // 4:
            break
        if length < AUDIO_SR:
            segment = F.pad(segment, (0, AUDIO_SR - length))
        out, _ = beats.extract_features(segment.unsqueeze(0).to(device), padding_mask=None)
        out = out[0].float()
        steps = out.shape[0] // 8  # tokens are time-major with 8 frequency patches
        out = out[: steps * 8].reshape(steps, 8, -1).mean(1)
        t = begin / AUDIO_SR + (torch.arange(steps, device=device) + 0.5) * BEATS_PATCH_SEC
        keep = t < (begin + length) / AUDIO_SR
        feats.append(out[keep])
        times.append(t[keep])
    if not feats or sum(len(t) for t in times) == 0:
        return None, None
    return torch.cat(feats), torch.cat(times)


def interpolate_timeline(feats: torch.Tensor, times: torch.Tensor, query: torch.Tensor):
    """Linear interpolation of [Ta, D] features at query times [Q] -> [Q, D]."""
    if feats.shape[0] == 1:
        return feats.expand(query.numel(), -1)
    right = torch.searchsorted(times, query).clamp(1, times.numel() - 1)
    left = right - 1
    weight = ((query - times[left]) / (times[right] - times[left]).clamp_min(1e-6)).clamp(0, 1)
    return feats[left] * (1 - weight[:, None]) + feats[right] * weight[:, None]


def feature_cache_key(videos: List[VideoInfo], videomae: str, beats: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{FEATURE_CACHE_VERSION}|{videomae}|{beats}|{TARGET_FPS}|{NUM_VIDEO_FRAMES}|"
                  f"{WINDOW_STRIDE}|{IMAGE_SIZE}|{SPATIAL_GRID}|{BEATS_CHUNK_SEC}".encode())
    for v in videos:
        stat = os.stat(v.path)
        digest.update(f"{v.path}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def extract_features(videos: List[VideoInfo], args, device: torch.device) -> dict:
    """Extract (or load) the shared feature file; per-video parts make it resumable."""
    key = feature_cache_key(videos, args.videomae, args.beats_checkpoint)
    cache_dir = Path(args.feature_cache_dir)
    cache_path = cache_dir / f"features_{key[:16]}.pt"
    if cache_path.is_file() and not args.rebuild_features:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        if data.get("key") == key:
            print(f"[FEATURES] Loaded {cache_path} ({len(data['video_index'])} windows).")
            return data
    parts_dir = cache_dir / f"parts_{key[:16]}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(len(videos))
            if args.rebuild_features or not (parts_dir / f"{i:04d}.pt").is_file()]

    if todo:
        torch.backends.cuda.matmul.allow_tf32 = True
        videomae = bc.load_videomae(args.videomae, device)
        beats = bc.load_beats(args.beats_checkpoint, args.beats_root, device)
        hidden = int(videomae.config.hidden_size)
        side = int(videomae.config.image_size) // int(videomae.config.patch_size)
        mean = torch.tensor(bc.IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor(bc.IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
        loader = DataLoader(
            VideoDecodeDataset(videos, todo), batch_size=None, num_workers=args.workers,
            prefetch_factor=2 if args.workers > 0 else None,
        )
        start_time = time.time()
        with torch.inference_mode():
            for item in tqdm(loader, total=len(todo), desc="Extract videos"):
                index = int(item["index"])
                info = videos[index]
                frames = item["frames"].to(device)  # [S, H, W, 3] uint8
                valid = item["valid"]
                starts = window_starts(info.num_sampled)
                span = torch.arange(NUM_VIDEO_FRAMES)
                video_feats, token_valid = [], []
                for b in range(0, len(starts), args.extract_batch_size):
                    chunk = starts[b:b + args.extract_batch_size]
                    idx = (torch.tensor(chunk)[:, None] + span[None]).clamp(max=info.num_sampled - 1)
                    pixels = frames[idx.to(device)].permute(0, 1, 4, 2, 3).float().div_(255.0)
                    pixels = (pixels - mean) / std
                    with torch.autocast(device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                        tokens = videomae(pixel_values=pixels).last_hidden_state
                    grid = tokens.float().reshape(len(chunk), NUM_TEMPORAL_TOKENS, side, side, hidden)
                    pooled = F.adaptive_avg_pool2d(
                        grid.permute(0, 1, 4, 2, 3).reshape(-1, hidden, side, side), SPATIAL_GRID,
                    ).reshape(len(chunk), NUM_TEMPORAL_TOKENS, hidden, -1).permute(0, 1, 3, 2)
                    feats = torch.cat([grid.mean((2, 3)).unsqueeze(2), pooled], dim=2)
                    video_feats.append(feats.half().cpu())
                    in_range = torch.tensor(chunk)[:, None] + span[None] < info.num_sampled
                    frame_ok = valid[idx] & in_range
                    token_valid.append(frame_ok.reshape(len(chunk), NUM_TEMPORAL_TOKENS, -1).any(-1))

                wave = item["audio"].float()
                feats_a, times_a = beats_timeline(beats, wave, device)
                audio = torch.zeros(len(starts), NUM_TEMPORAL_TOKENS, int(beats.cfg.encoder_embed_dim))
                if feats_a is not None:
                    for w, s in enumerate(starts):
                        query = torch.as_tensor(token_center_times(s), device=device,
                                                dtype=torch.float32)
                        audio[w] = interpolate_timeline(feats_a, times_a, query).cpu()
                part = {
                    "starts": torch.tensor(starts),
                    "video": torch.cat(video_feats),
                    "token_valid": torch.cat(token_valid),
                    "audio": audio.half(),
                    "has_audio": feats_a is not None,
                }
                tmp = parts_dir / f"{index:04d}.tmp"
                torch.save(part, tmp)
                tmp.replace(parts_dir / f"{index:04d}.pt")
        print(f"[FEATURES] Extracted {len(todo)} videos in {time.time() - start_time:.0f}s.")
        del videomae, beats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    parts = [torch.load(parts_dir / f"{i:04d}.pt", map_location="cpu", weights_only=False)
             for i in range(len(videos))]
    data = {
        "key": key,
        "videos": [vars(v) for v in videos],
        "video_index": torch.cat([torch.full((len(p["starts"]),), i) for i, p in enumerate(parts)]),
        "starts": torch.cat([p["starts"] for p in parts]),
        "video": torch.cat([p["video"] for p in parts]),
        "token_valid": torch.cat([p["token_valid"] for p in parts]),
        "audio": torch.cat([p["audio"] for p in parts]),
        "has_audio": torch.cat([torch.full((len(p["starts"]),), bool(p["has_audio"])) for p in parts]),
    }
    tmp = cache_path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(cache_path)
    no_audio = sorted({Path(videos[i].path).name for i, p in enumerate(parts) if not p["has_audio"]})
    print(f"[FEATURES] Saved {cache_path}: {len(data['video_index'])} windows; "
          f"videos without audio: {no_audio or 'none'}.")
    return data


def build_labels(data: dict, radius: float):
    """Token labels [N,16] and window classes [N] from the JSON events."""
    videos = data["videos"]
    labels = np.stack([
        token_labels(videos[int(v)]["events"], int(s), radius)
        for v, s in zip(data["video_index"], data["starts"])
    ])
    labels = torch.from_numpy(labels)
    valid = data["token_valid"]
    masked = labels.masked_fill(~valid, 0)
    window_class = torch.where(
        (masked == LABEL2ID["hit"]).any(1), LABEL2ID["hit"],
        torch.where((masked == LABEL2ID["miss"]).any(1), LABEL2ID["miss"], LABEL2ID["no_event"]),
    )
    return labels, window_class


@torch.no_grad()
def encode_class_prompts(clip_name: str, device: torch.device) -> torch.Tensor:
    """[NUM_CLASSES, D] CLIP text embeddings, one averaged description per class."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    model = CLIPTextModelWithProjection.from_pretrained(clip_name).to(device).eval()
    rows = []
    for c in range(NUM_CLASSES):
        tokens = tokenizer(CLASS_PROMPTS[ID2LABEL[c]], padding=True, truncation=True,
                           max_length=77, return_tensors="pt").to(device)
        emb = F.normalize(model(**tokens).text_embeds.float(), dim=-1).mean(0)
        rows.append(F.normalize(emb, dim=-1))
    del model
    text = torch.stack(rows)
    sim = text @ text.t()
    print("[TEXT] CLIP cosine similarity between class descriptions:\n"
          + "\n".join(f"  {ID2LABEL[i]:>8}: " + " ".join(f"{x:.3f}" for x in sim[i].tolist())
                      for i in range(NUM_CLASSES)))
    return text


# ============================================================
# Metrics
# ============================================================

def token_metrics(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "f1_by_class": {ID2LABEL[i]: float(f1[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
    }


def decode_events(probs: np.ndarray, min_tokens: int) -> List[Tuple[float, int]]:
    """Runs of non-no_event tokens -> (centre time, hit/miss by summed probability)."""
    pred = probs.argmax(-1)
    events, i = [], 0
    while i < len(pred):
        if pred[i] == LABEL2ID["no_event"]:
            i += 1
            continue
        j = i
        while j < len(pred) and pred[j] != LABEL2ID["no_event"]:
            j += 1
        if j - i >= min_tokens:
            scores = probs[i:j, 1:].sum(0)
            centre = ((i + j - 1) / 2 * FRAMES_PER_TOKEN + (FRAMES_PER_TOKEN - 1) / 2) / TARGET_FPS
            events.append((centre, 1 + int(scores.argmax())))
        i = j
    return events


def event_metrics(predicted: Dict[int, list], truth: Dict[int, list], tolerance: float) -> dict:
    """Greedy one-to-one matching within `tolerance` s, per video."""
    det = {"tp": 0, "fp": 0, "fn": 0}
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in (1, 2)}
    for vid in truth:
        pred, gt = predicted.get(vid, []), truth[vid]
        pairs = sorted((abs(p[0] - g[0]), a, b) for a, p in enumerate(pred)
                       for b, g in enumerate(gt) if abs(p[0] - g[0]) <= tolerance)
        used_p, used_g, matches = set(), set(), []
        for _, a, b in pairs:
            if a not in used_p and b not in used_g:
                used_p.add(a)
                used_g.add(b)
                matches.append((a, b))
        det["tp"] += len(matches)
        det["fp"] += len(pred) - len(matches)
        det["fn"] += len(gt) - len(matches)
        correct_p = {a for a, b in matches if pred[a][1] == gt[b][1]}
        correct_g = {b for a, b in matches if pred[a][1] == gt[b][1]}
        for c in (1, 2):
            per_class[c]["tp"] += sum(1 for a in correct_p if pred[a][1] == c)
            per_class[c]["fp"] += sum(1 for a, p in enumerate(pred) if p[1] == c and a not in correct_p)
            per_class[c]["fn"] += sum(1 for b, g in enumerate(gt) if g[1] == c and b not in correct_g)

    def prf(d):
        p = d["tp"] / max(d["tp"] + d["fp"], 1)
        r = d["tp"] / max(d["tp"] + d["fn"], 1)
        return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12), **d}

    result = {"detection": prf(det), **{ID2LABEL[c]: prf(per_class[c]) for c in (1, 2)}}
    result["macro_f1"] = (result["hit"]["f1"] + result["miss"]["f1"]) / 2
    return result


# ============================================================
# Cross-validation
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_folds(window_class: np.ndarray, users: np.ndarray, n_splits: int, seed: int):
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(window_class, window_class, users), 1):
        inner = StratifiedGroupKFold(n_splits=max(3, n_splits), shuffle=True, random_state=seed + fold)
        fit_rel, val_rel = next(inner.split(train_idx, window_class[train_idx], users[train_idx]))
        yield fold, train_idx[fit_rel], train_idx[val_rel], test_idx


def focal_loss(logits, labels, valid, alpha, gamma):
    log_probs = F.log_softmax(logits[valid].float(), dim=-1)
    target = labels[valid]
    log_pt = log_probs.gather(1, target[:, None]).squeeze(1)
    return (-alpha[target] * (1 - log_pt.exp()).pow(gamma) * log_pt).mean()


def run_cv(
    data: dict,
    build_model: Callable[[dict], nn.Module],
    forward: Callable[[nn.Module, dict], torch.Tensor],
    args,
    device: torch.device,
) -> dict:
    """Train one model per user-grouped fold; all features stay on the GPU."""
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type == "cuda" and not args.no_amp

    labels, window_class = build_labels(data, args.event_radius)
    videos = data["videos"]
    users = np.array([videos[int(v)]["user"] for v in data["video_index"]])
    video = data["video"]
    video = video[:, :, 0] if args.spatial == "mean" else video.flatten(2)
    has_audio = data["has_audio"]
    audio_pad = torch.zeros(data["audio"].shape[:2], dtype=torch.bool)
    audio_pad[~has_audio, 1:] = True  # keep one zero token unmasked so attention stays defined
    tensors = {
        "video": video.to(device),
        "audio": data["audio"].to(device),
        "audio_pad": audio_pad.to(device),
        "video_pad": (~data["token_valid"]).to(device),
        "labels": labels.to(device),
    }
    dims = {"video": tensors["video"].shape[-1], "audio": tensors["audio"].shape[-1]}
    token_counts = torch.bincount(labels[data["token_valid"]], minlength=NUM_CLASSES)
    print(f"[DATA] {len(users)} windows, {len(set(users))} users, {len(videos)} videos | "
          f"window classes {np.bincount(window_class.numpy(), minlength=NUM_CLASSES).tolist()} | "
          f"tokens {token_counts.tolist()} (no_event/hit/miss) | video dim {dims['video']}")

    def standardizer(idx):
        """Per-dimension mean/std of the fit windows' valid tokens."""
        stats = {}
        for name in ("video", "audio"):
            total, sq, count = 0.0, 0.0, 0
            for chunk in np.array_split(idx, max(1, len(idx) // 2048)):
                ci = torch.as_tensor(chunk, device=device)
                x = tensors[name][ci].float()
                keep = ~tensors["video_pad"][ci]
                if name == "audio":
                    keep = keep & has_audio.to(device)[ci][:, None]
                x = x[keep]
                total, sq, count = total + x.sum(0), sq + (x * x).sum(0), count + x.shape[0]
            if count == 0:
                dim = tensors[name].shape[-1]
                stats[name] = (torch.zeros(dim, device=device), torch.ones(dim, device=device))
                continue
            mean = total / count
            std = (sq / count - mean * mean).clamp_min(1e-6).sqrt()
            stats[name] = (mean, std)
        return stats

    def gather(idx, stats, train=False):
        it = torch.as_tensor(idx, device=device)
        batch = {k: v[it] for k, v in tensors.items()}
        for name in ("video", "audio"):
            mean, std = stats[name]
            batch[name] = (batch[name].float() - mean) / std
        if train and args.feat_noise > 0:
            batch["video"] = batch["video"] + args.feat_noise * torch.randn_like(batch["video"])
            batch["audio"] = batch["audio"] + args.feat_noise * torch.randn_like(batch["audio"])
        if train and args.audio_drop > 0:
            drop = torch.rand(len(idx), device=device) < args.audio_drop
            batch["audio"][drop] = 0.0
        no_audio = ~has_audio.to(device)[it]
        batch["audio"][no_audio] = 0.0
        return batch

    def predict(model, idx, stats):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            for start in range(0, len(idx), args.batch_size * 4):
                batch = gather(idx[start:start + args.batch_size * 4], stats)
                out.append(torch.softmax(forward(model, batch).float(), -1))
        return torch.cat(out)

    def confusion(probs, idx):
        it = torch.as_tensor(idx, device=device)
        valid = ~tensors["video_pad"][it]
        y, p = tensors["labels"][it][valid].cpu().numpy(), probs.argmax(-1)[valid].cpu().numpy()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(cm, (y, p), 1)
        return cm

    all_probs = torch.zeros(len(users), NUM_TEMPORAL_TOKENS, NUM_CLASSES)
    summary = []
    for fold, fit_idx, val_idx, test_idx in make_folds(window_class.numpy(), users, args.folds, args.seed):
        seed_everything(args.seed + fold)
        rng = np.random.default_rng(args.seed + fold)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_users = {name: sorted(set(users[i].tolist()))
                       for name, i in (("fit", fit_idx), ("val", val_idx), ("test", test_idx))}
        print(f"\n[FOLD {fold}] windows fit/val/test={len(fit_idx)}/{len(val_idx)}/{len(test_idx)} | "
              f"val users={split_users['val']} | test users={split_users['test']}")

        stats = standardizer(fit_idx)
        fit_tokens = tensors["labels"][torch.as_tensor(fit_idx, device=device)][
            ~tensors["video_pad"][torch.as_tensor(fit_idx, device=device)]]
        counts = torch.bincount(fit_tokens, minlength=NUM_CLASSES).float().clamp_min(1)
        alpha = counts.rsqrt()
        alpha = alpha / alpha.mean()

        model = build_model(dims).to(device)
        print(f"[FOLD {fold}] trainable parameters: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} | "
              f"alpha={[round(a, 3) for a in alpha.tolist()]}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(fit_idx) / args.batch_size)
        total_steps, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: (s + 1) / max(1, warmup) if s < warmup
            else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup)))),
        )

        best_f1, best_epoch, stale, history = -1.0, 0, 0, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            running, cm = 0.0, np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            order = rng.permutation(fit_idx)
            for start in range(0, len(order), args.batch_size):
                chunk = order[start:start + args.batch_size]
                batch = gather(chunk, stats, train=True)
                valid = ~batch["video_pad"]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = forward(model, batch)
                loss = focal_loss(logits, batch["labels"], valid, alpha, args.focal_gamma)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                running += loss.item() * len(chunk)
                y, p = batch["labels"][valid].cpu().numpy(), logits.argmax(-1)[valid].cpu().numpy()
                np.add.at(cm, (y, p), 1)
            train_m = token_metrics(cm)
            val_m = token_metrics(confusion(predict(model, val_idx, stats), val_idx))
            history.append({"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                            "train_loss": running / len(fit_idx), "train": train_m, "val": val_m})
            print(f"[FOLD {fold}] epoch {epoch:03d} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"train loss={running / len(fit_idx):.4f} F1={train_m['macro_f1']:.4f} | "
                  f"val F1={val_m['macro_f1']:.4f} {[round(x, 3) for x in val_m['f1_by_class'].values()]}")
            if val_m["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = val_m["macro_f1"], epoch, 0
                torch.save({
                    "model": model.state_dict(), "epoch": epoch, "val": val_m,
                    "feature_mean": {k: v[0].cpu() for k, v in stats.items()},
                    "feature_std": {k: v[1].cpu() for k, v in stats.items()},
                    "label2id": LABEL2ID, "args": vars(args), "users": split_users,
                }, fold_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"[FOLD {fold}] early stopping at epoch {epoch}.")
                    break

        checkpoint = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        probs = predict(model, test_idx, stats)
        all_probs[torch.as_tensor(test_idx)] = probs.cpu()
        test_m = token_metrics(confusion(probs, test_idx))
        print(f"[FOLD {fold}] best epoch {best_epoch}: val F1={best_f1:.4f} | TEST token "
              f"F1={test_m['macro_f1']:.4f} {test_m['f1_by_class']}")
        result = {"fold": fold, "best_epoch": best_epoch, "val_macro_f1": best_f1,
                  "test": test_m, "users": split_users}
        with (fold_dir / "history.json").open("w") as handle:
            json.dump(history, handle, indent=2)
        with (fold_dir / "result.json").open("w") as handle:
            json.dump(result, handle, indent=2)
        summary.append(result)
        del model, optimizer

    return summarize(data, all_probs, summary, users, args, output)


def summarize(data, all_probs, summary, users, args, output: Path) -> dict:
    """Pooled token metrics and event metrics over the test predictions of every fold."""
    labels, _ = build_labels(data, args.event_radius)
    valid = data["token_valid"]
    pred = all_probs.argmax(-1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (labels[valid].numpy(), pred[valid].numpy()), 1)
    pooled = token_metrics(cm)

    per_user = {}
    for user in sorted(set(users.tolist())):
        sel = torch.from_numpy(users == user)[:, None] & valid
        ucm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(ucm, (labels[sel].numpy(), pred[sel].numpy()), 1)
        per_user[user] = token_metrics(ucm)

    # Stitch overlapping windows into one probability timeline per video.
    videos = data["videos"]
    predicted, truth = {}, {}
    for vid, info in enumerate(videos):
        rows = (data["video_index"] == vid).nonzero().flatten()
        length = math.ceil(info["num_sampled"] / FRAMES_PER_TOKEN)
        acc, count = np.zeros((length, NUM_CLASSES)), np.zeros(length)
        for r in rows.tolist():
            slot = int(data["starts"][r]) // FRAMES_PER_TOKEN + np.arange(NUM_TEMPORAL_TOKENS)
            ok = valid[r].numpy() & (slot < length)
            acc[slot[ok]] += all_probs[r].numpy()[ok]
            count[slot[ok]] += 1
        covered = count > 0
        timeline = np.zeros_like(acc)
        timeline[:, 0] = 1.0
        timeline[covered] = acc[covered] / count[covered, None]
        predicted[vid] = decode_events(timeline, args.min_event_tokens)
        truth[vid] = [tuple(e) for e in info["events"]]
    events = event_metrics(predicted, truth, args.event_tolerance)

    fold_f1 = np.array([r["test"]["macro_f1"] for r in summary])
    report = {
        "pooled_token": pooled,
        "event": events,
        "fold_token_macro_f1_mean": float(fold_f1.mean()),
        "fold_token_macro_f1_std": float(fold_f1.std()),
        "folds": summary,
        "per_user_token": per_user,
    }
    with (output / "cv_summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    with (output / "cv_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "best_epoch", "val_macro_f1", "test_token_macro_f1", "test_users"])
        for r in summary:
            writer.writerow([r["fold"], r["best_epoch"], r["val_macro_f1"], r["test"]["macro_f1"],
                             " ".join(r["users"]["test"])])
        writer.writerow(["pooled", "", "", pooled["macro_f1"], ""])
        writer.writerow(["event_macro_f1", "", "", events["macro_f1"], ""])
    torch.save({"probs": all_probs, "starts": data["starts"], "video_index": data["video_index"]},
               output / "test_predictions.pt")
    print(f"\n[RESULT] pooled test token macro-F1 = {pooled['macro_f1']:.4f} "
          f"{pooled['f1_by_class']} | per-fold {fold_f1.mean():.4f} ± {fold_f1.std():.4f}")
    print(f"[RESULT] events (±{args.event_tolerance}s): detection F1={events['detection']['f1']:.4f} "
          f"| hit F1={events['hit']['f1']:.4f} miss F1={events['miss']['f1']:.4f} "
          f"| event macro-F1={events['macro_f1']:.4f}")
    return report


# ============================================================
# CLI shared by v3 / v4 / v5
# ============================================================

def add_common_args(parser, artifacts_root: str, version_artifacts: str, cfg) -> None:
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--videomae", default=DEFAULT_VIDEOMAE)
    parser.add_argument("--beats-checkpoint", default=cfg.BEATS_CHECKPOINT)
    parser.add_argument("--beats-root", default=cfg.BEATS_ROOT)
    parser.add_argument("--feature-cache-dir", default=f"{artifacts_root}/baseline/pickleball_features",
                        help="Shared by v3/v4/v5: features are extracted once.")
    parser.add_argument("--output-dir", default=f"{version_artifacts}/checkpoints_pickleball")
    parser.add_argument("--folds", type=int, default=5, help="Outer folds grouped by user.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=cfg.FUSION_LR)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=cfg.WARMUP_EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--grad-clip", type=float, default=cfg.GRAD_CLIP)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--spatial", choices=["grid", "mean"], default="grid",
                        help="grid: global mean + 2x2 grid per token; mean: global mean only.")
    parser.add_argument("--feat-noise", type=float, default=0.1,
                        help="Gaussian noise on standardized features during training.")
    parser.add_argument("--audio-drop", type=float, default=0.2,
                        help="Probability of zeroing a window's audio during training.")
    parser.add_argument("--event-radius", type=float, default=0.5,
                        help="A token is labelled with an event within this many seconds.")
    parser.add_argument("--event-tolerance", type=float, default=0.5,
                        help="Event metric: max distance (s) between predicted and true event.")
    parser.add_argument("--min-event-tokens", type=int, default=3,
                        help="Event decoding: shortest run of event tokens kept (1 token = 1/15 s).")
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6, help="Video decoding workers.")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--extract-only", action="store_true",
                        help="Build the shared feature cache and exit.")


def prepare(args, device: torch.device) -> Optional[dict]:
    videos = discover_videos(args.data_dir)
    fps = sorted({round(v.fps) for v in videos})
    print(f"[DATA] {len(videos)} videos, {len({v.user for v in videos})} users, source fps {fps} "
          f"-> sampled at {TARGET_FPS:g} fps ({NUM_VIDEO_FRAMES / TARGET_FPS:.2f} s per window)")
    data = extract_features(videos, args, device)
    return None if args.extract_only else data
