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

import sys

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))  # baseline root, for common/
from common import media as bc  # noqa: E402

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
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"      # shared feature cache (v3 / v4 / v5)
OUTPUTS = HERE / "outputs"      # runs/<version> (5 folds), analysis/
BEATS_CHECKPOINT = bc.BEATS["base"]
BEATS_ROOT = bc.BEATS_ROOT
SEED = 42
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


def token_labels(events: List[Tuple[float, int]], start: int, radius: float,
                 length: int = NUM_TEMPORAL_TOKENS) -> np.ndarray:
    """Token label = class of an event within `radius` s of one of its frames; hit wins."""
    labels = np.zeros(length, dtype=np.int64)
    if not events:
        return labels
    times = np.array([e[0] for e in events])
    classes = np.array([e[1] for e in events])
    frames = start + np.arange(length) * FRAMES_PER_TOKEN
    t0, t1 = frames / TARGET_FPS, (frames + FRAMES_PER_TOKEN - 1) / TARGET_FPS
    near = (times[None] >= t0[:, None] - radius) & (times[None] <= t1[:, None] + radius)
    labels[(near & (classes == LABEL2ID["miss"])[None]).any(1)] = LABEL2ID["miss"]
    labels[(near & (classes == LABEL2ID["hit"])[None]).any(1)] = LABEL2ID["hit"]
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
    key = feature_cache_key(videos, args.videomae, Path(args.beats_checkpoint).name)
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


def build_sequences(data: dict, length: int, stride: int) -> dict:
    """Re-cut the cached windows into sequences of `length` tokens (stride in tokens).

    The overlapping 16-token windows are stitched into one token timeline per video
    (features of a token seen by several windows are averaged), then cut again. A
    longer sequence lets the model see what happens after the contact (ball in/out),
    which a 1.07 s window often cuts off. length=16 with the default stride returns
    the cache unchanged.
    """
    stride = stride or max(1, length // 2)
    if length == NUM_TEMPORAL_TOKENS and stride == WINDOW_STRIDE // FRAMES_PER_TOKEN:
        return data
    videos = data["videos"]
    sizes = [math.ceil(v["num_sampled"] / FRAMES_PER_TOKEN) for v in videos]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    total = int(offsets[-1])
    video, audio = data["video"], data["audio"]
    vid_sum = torch.zeros(total, *video.shape[2:])
    aud_sum = torch.zeros(total, audio.shape[-1])
    count = torch.zeros(total)
    span = torch.arange(NUM_TEMPORAL_TOKENS)
    vidx = data["video_index"].long()
    off = torch.from_numpy(offsets)
    for b in range(0, len(vidx), 1024):
        rows = slice(b, b + 1024)
        local = data["starts"][rows].long()[:, None] // FRAMES_PER_TOKEN + span[None]
        size = torch.as_tensor(sizes)[vidx[rows]][:, None]
        ok = data["token_valid"][rows] & (local < size)
        slot = (off[vidx[rows]][:, None] + local)[ok]
        vid_sum.index_add_(0, slot, video[rows][ok].float())
        aud_sum.index_add_(0, slot, audio[rows][ok].float())
        count.index_add_(0, slot, torch.ones(len(slot)))
    covered = count > 0
    vid_sum[covered] /= count[covered].view(-1, *([1] * (vid_sum.dim() - 1)))
    aud_sum[covered] /= count[covered][:, None]
    vid_tl, aud_tl = vid_sum.half(), aud_sum.half()
    del vid_sum, aud_sum

    has_audio_video = torch.zeros(len(videos), dtype=torch.bool)
    has_audio_video[vidx] = data["has_audio"]
    index, valid, starts, owners = [], [], [], []
    ar = np.arange(length)
    for vid, size in enumerate(sizes):
        last = max(0, size - length)
        begins = list(range(0, last + 1, stride))
        if begins[-1] != last:
            begins.append(last)
        for s in begins:
            local = s + ar
            inside = local < size
            idx = offsets[vid] + np.minimum(local, size - 1)
            index.append(idx)
            valid.append(inside & covered[idx].numpy())
            starts.append(s * FRAMES_PER_TOKEN)
            owners.append(vid)
    index = torch.from_numpy(np.stack(index))
    out = {
        "key": data["key"],
        "videos": videos,
        "video_index": torch.tensor(owners),
        "starts": torch.tensor(starts),
        "video": vid_tl[index],
        "token_valid": torch.from_numpy(np.stack(valid)),
        "audio": aud_tl[index],
        "has_audio": has_audio_video[torch.tensor(owners)],
    }
    print(f"[SEQ] {len(data['video_index'])} windows of {NUM_TEMPORAL_TOKENS} tokens -> "
          f"{len(owners)} sequences of {length} tokens ({length * TOKEN_SEC:.2f} s, stride {stride}).")
    return out


def normalize_per_video(data: dict, mode: str) -> dict:
    """Remove each video's own mean (mode 'mean') or mean/std ('std') from its features.

    VideoMAE tokens cluster by video (scene, light, camera), and every video holds only
    hits or only misses, so a model can score well by recognising the session instead of
    the stroke. Uses only the video's own (unlabelled) features, also at test time.
    """
    data = dict(data)
    video, audio = data["video"].clone(), data["audio"].clone()
    vidx = data["video_index"]
    for vid in vidx.unique().tolist():
        rows = (vidx == vid).nonzero().flatten()
        keep = data["token_valid"][rows]
        for name, tensor in (("video", video), ("audio", audio)):
            if name == "audio" and not bool(data["has_audio"][rows[0]]):
                continue
            x = tensor[rows].float()
            mean = x[keep].mean(0)
            x = x - mean
            if mode == "std":
                x = x / x[keep].std(0).clamp_min(1e-3)
            tensor[rows] = x.to(tensor.dtype)
    data["video"], data["audio"] = video, audio
    print(f"[NORM] per-video feature normalisation: {mode}")
    return data


def subset_videos(data: dict, count: int) -> dict:
    """Keep `count` evenly spaced videos (quick smoke tests only)."""
    keep = np.unique(np.linspace(0, len(data["videos"]) - 1, count).round().astype(int))
    remap = {int(old): new for new, old in enumerate(keep)}
    rows = torch.from_numpy(np.isin(data["video_index"].numpy(), keep))
    out = {k: v[rows] for k, v in data.items() if torch.is_tensor(v)}
    out["video_index"] = torch.tensor([remap[int(v)] for v in out["video_index"]])
    out["videos"] = [data["videos"][i] for i in keep]
    out["key"] = data["key"]
    print(f"[DEBUG] subset: {len(keep)} videos, {int(rows.sum())} windows.")
    return out


def build_labels(data: dict, radius: float):
    """Token labels [N,L] and window classes [N] from the JSON events."""
    videos = data["videos"]
    length = data["token_valid"].shape[1]
    labels = np.stack([
        token_labels(videos[int(v)]["events"], int(s), radius, length)
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


def decode_events(probs: np.ndarray, min_tokens: int, threshold: Optional[float] = None,
                  smooth: int = 1, peak: bool = False) -> List[Tuple[float, int]]:
    """Runs of event tokens -> (time, hit/miss by summed probability).

    threshold=None: a token is an event when argmax != no_event (original decoder).
    Otherwise: when the (moving-average smoothed) P(event) = 1 - P(no_event) > threshold.
    peak: the event time is the token with the highest P(event) in the run, not the centre.
    """
    p_event = 1.0 - probs[:, 0]
    if smooth > 1:
        p_event = np.convolve(p_event, np.ones(smooth) / smooth, mode="same")
    mask = probs.argmax(-1) != LABEL2ID["no_event"] if threshold is None else p_event > threshold
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    events = []
    for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if j - i < min_tokens:
            continue
        scores = probs[i:j, 1:].sum(0)
        token = i + int(p_event[i:j].argmax()) if peak else (i + j - 1) / 2
        centre = (token * FRAMES_PER_TOKEN + (FRAMES_PER_TOKEN - 1) / 2) / TARGET_FPS
        events.append((float(centre), 1 + int(scores.argmax())))
    return events


DECODE_GRID = {
    "threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    "min_tokens": [1, 2, 3, 4, 6],
    "smooth": [1, 3, 5],
    "peak": [False, True],
}


def default_decode(args) -> dict:
    return {"threshold": None, "min_tokens": args.min_event_tokens, "smooth": 1, "peak": False}


def stitch_timelines(probs: np.ndarray, data: dict, rows: np.ndarray) -> Dict[int, np.ndarray]:
    """Average overlapping sequence predictions [len(rows), L, C] into one timeline per video."""
    videos = data["videos"]
    length = probs.shape[1]
    vids = data["video_index"].numpy()[rows]
    starts = data["starts"].numpy()[rows] // FRAMES_PER_TOKEN
    valid = data["token_valid"].numpy()[rows]
    out = {}
    for vid in np.unique(vids):
        sel = vids == vid
        size = math.ceil(videos[vid]["num_sampled"] / FRAMES_PER_TOKEN)
        slot = starts[sel][:, None] + np.arange(length)
        ok = valid[sel] & (slot < size)
        acc, count = np.zeros((size, probs.shape[-1])), np.zeros(size)
        np.add.at(acc, slot[ok], probs[sel][ok])
        np.add.at(count, slot[ok], 1)
        timeline = np.zeros_like(acc)
        timeline[:, 0] = 1.0
        covered = count > 0
        timeline[covered] = acc[covered] / count[covered, None]
        out[int(vid)] = timeline
    return out


def decode_all(timelines: Dict[int, np.ndarray], params: dict) -> Dict[int, list]:
    return {vid: decode_events(tl, **params) for vid, tl in timelines.items()}


def video_truth(data: dict, vids) -> Dict[int, list]:
    return {int(v): [tuple(e) for e in data["videos"][int(v)]["events"]] for v in vids}


def tune_decode(timelines: Dict[int, np.ndarray], truth: Dict[int, list], tolerance: float,
                fallback: dict) -> Tuple[dict, float]:
    """Grid-search the decoder on validation videos (event macro-F1)."""
    best, best_f1 = dict(fallback), event_metrics(decode_all(timelines, fallback), truth, tolerance)["macro_f1"]
    for threshold in DECODE_GRID["threshold"]:
        for min_tokens in DECODE_GRID["min_tokens"]:
            for smooth in DECODE_GRID["smooth"]:
                for peak in DECODE_GRID["peak"]:
                    params = {"threshold": threshold, "min_tokens": min_tokens,
                              "smooth": smooth, "peak": peak}
                    f1 = event_metrics(decode_all(timelines, params), truth, tolerance)["macro_f1"]
                    if f1 > best_f1 + 1e-9:
                        best, best_f1 = params, f1
    return best, best_f1


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


def make_folds(window_class: np.ndarray, users: np.ndarray, n_splits: int, seed: int,
               inner_splits: int = 0):
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_n = inner_splits or max(3, n_splits)
    for fold, (train_idx, test_idx) in enumerate(outer.split(window_class, window_class, users), 1):
        inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=seed + fold)
        fit_rel, val_rel = next(inner.split(train_idx, window_class[train_idx], users[train_idx]))
        yield fold, train_idx[fit_rel], train_idx[val_rel], test_idx


def focal_loss(logits, labels, valid, alpha, gamma, smoothing: float = 0.0):
    """Class-weighted focal loss (gamma=0 gives weighted cross-entropy), optional label smoothing."""
    log_probs = F.log_softmax(logits[valid].float(), dim=-1)
    target = labels[valid]
    log_pt = log_probs.gather(1, target[:, None]).squeeze(1)
    nll = -log_pt
    if smoothing > 0:
        nll = (1 - smoothing) * nll - smoothing * log_probs.mean(-1)
    return (alpha[target] * (1 - log_pt.exp()).pow(gamma) * nll).mean()


def class_weights(counts: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        weight = torch.ones_like(counts)
    elif mode == "inv":
        weight = 1.0 / counts
    else:  # rsqrt
        weight = counts.rsqrt()
    return weight / weight.mean()


def run_cv(
    data: dict,
    build_model: Callable[[dict], nn.Module],
    forward: Callable[[nn.Module, dict], torch.Tensor],
    args,
    device: torch.device,
) -> dict:
    """Train one model (or a seed ensemble) per user-grouped fold; features stay on the GPU."""
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type == "cuda" and not args.no_amp

    if args.debug_subset:
        data = subset_videos(data, args.debug_subset)
    data = build_sequences(data, args.seq_tokens, args.seq_stride)
    if args.video_norm != "none":
        data = normalize_per_video(data, args.video_norm)
    length = data["token_valid"].shape[1]

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
    has_audio_dev = has_audio.to(device)
    dims = {"video": tensors["video"].shape[-1], "audio": tensors["audio"].shape[-1]}
    token_counts = torch.bincount(labels[data["token_valid"]], minlength=NUM_CLASSES)
    print(f"[DATA] {len(users)} sequences of {length} tokens, {len(set(users))} users, "
          f"{len(videos)} videos | sequence classes "
          f"{np.bincount(window_class.numpy(), minlength=NUM_CLASSES).tolist()} | "
          f"tokens {token_counts.tolist()} (no_event/hit/miss) | video dim {dims['video']}")

    def standardizer(idx):
        """Per-dimension mean/std of the fit sequences' valid tokens."""
        stats = {}
        for name in ("video", "audio"):
            total, sq, count = 0.0, 0.0, 0
            for chunk in np.array_split(idx, max(1, len(idx) // 2048)):
                ci = torch.as_tensor(chunk, device=device)
                x = tensors[name][ci].float()
                keep = ~tensors["video_pad"][ci]
                if name == "audio":
                    keep = keep & has_audio_dev[ci][:, None]
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
        if train and args.token_mask > 0:
            # Time masking: hide single tokens (set to the feature mean) so the model has to
            # use the neighbouring tokens; labels are unchanged.
            shape = batch["video"].shape[:2]
            batch["video"][torch.rand(shape, device=device) < args.token_mask] = 0.0
            batch["audio"][torch.rand(shape, device=device) < args.token_mask] = 0.0
        if train and args.audio_drop > 0:
            drop = torch.rand(len(idx), device=device) < args.audio_drop
            batch["audio"][drop] = 0.0
        batch["audio"][~has_audio_dev[it]] = 0.0
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

    def evaluate(probs, idx):
        """Validation metrics; 'score' is what checkpoint selection maximises."""
        metrics = token_metrics(confusion(probs, idx))
        it = torch.as_tensor(idx, device=device)
        valid = ~tensors["video_pad"][it]
        target = tensors["labels"][it][valid]
        metrics["loss"] = float(-probs[valid].gather(1, target[:, None]).clamp_min(1e-8).log().mean())
        if args.select in ("event_f1", "combo"):
            timelines = stitch_timelines(probs.cpu().numpy(), data, idx)
            truth = video_truth(data, timelines.keys())
            metrics["event_macro_f1"] = event_metrics(
                decode_all(timelines, default_decode(args)), truth, args.event_tolerance)["macro_f1"]
        metrics["score"] = {
            "token_f1": metrics["macro_f1"],
            "event_f1": metrics.get("event_macro_f1"),
            "combo": (metrics["macro_f1"] + metrics.get("event_macro_f1", 0.0)) / 2,
            "loss": -metrics["loss"],
        }[args.select]
        return metrics

    def train_member(fold, member, fit_idx, val_idx, stats, alpha, path):
        seed_everything(args.seed + fold + 1000 * member)
        rng = np.random.default_rng(args.seed + fold + 1000 * member)
        model = build_model(dims).to(device)
        if member == 0:
            print(f"[FOLD {fold}] trainable parameters: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} | "
                  f"alpha={[round(a, 3) for a in alpha.tolist()]}")
        ema = None
        if args.ema > 0:
            ema = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema),
                use_buffers=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(fit_idx) / args.batch_size)
        total_steps, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: (s + 1) / max(1, warmup) if s < warmup
            else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup)))),
        )
        tag = f"[FOLD {fold}]" if args.n_models == 1 else f"[FOLD {fold}/m{member}]"
        best_score, best_epoch, stale, history = -math.inf, 0, 0, []
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
                loss = focal_loss(logits, batch["labels"], valid, alpha, args.focal_gamma,
                                  args.label_smoothing)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update_parameters(model)
                running += loss.item() * len(chunk)
                y, p = batch["labels"][valid].cpu().numpy(), logits.argmax(-1)[valid].cpu().numpy()
                np.add.at(cm, (y, p), 1)
            eval_model = ema.module if ema is not None else model
            train_m = token_metrics(cm)
            val_m = evaluate(predict(eval_model, val_idx, stats), val_idx)
            history.append({"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                            "train_loss": running / len(fit_idx), "train": train_m, "val": val_m})
            event = f" event={val_m['event_macro_f1']:.4f}" if "event_macro_f1" in val_m else ""
            print(f"{tag} epoch {epoch:03d} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"train loss={running / len(fit_idx):.4f} F1={train_m['macro_f1']:.4f} | "
                  f"val F1={val_m['macro_f1']:.4f} {[round(x, 3) for x in val_m['f1_by_class'].values()]}"
                  f" loss={val_m['loss']:.4f}{event}")
            if val_m["score"] > best_score:
                best_score, best_epoch, stale = val_m["score"], epoch, 0
                torch.save({
                    "model": eval_model.state_dict(), "epoch": epoch, "val": val_m,
                    "feature_mean": {k: v[0].cpu() for k, v in stats.items()},
                    "feature_std": {k: v[1].cpu() for k, v in stats.items()},
                    "label2id": LABEL2ID, "args": vars(args), "seq_tokens": length,
                }, path)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"{tag} early stopping at epoch {epoch}.")
                    break
        del model, optimizer, ema
        return best_score, best_epoch, history

    all_probs = torch.zeros(len(users), length, NUM_CLASSES)
    predicted, predicted_default = {}, {}
    summary = []
    for fold, fit_idx, val_idx, test_idx in make_folds(window_class.numpy(), users, args.folds,
                                                       args.seed, args.inner_splits):
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_users = {name: sorted(set(users[i].tolist()))
                       for name, i in (("fit", fit_idx), ("val", val_idx), ("test", test_idx))}
        print(f"\n[FOLD {fold}] sequences fit/val/test={len(fit_idx)}/{len(val_idx)}/{len(test_idx)} | "
              f"val users={split_users['val']} | test users={split_users['test']}")

        stats = standardizer(fit_idx)
        fit_t = torch.as_tensor(fit_idx, device=device)
        fit_tokens = tensors["labels"][fit_t][~tensors["video_pad"][fit_t]]
        counts = torch.bincount(fit_tokens, minlength=NUM_CLASSES).float().clamp_min(1)
        alpha = class_weights(counts, args.class_weight)

        members, val_probs, test_probs = [], 0.0, 0.0
        for member in range(args.n_models):
            path = fold_dir / ("best.pt" if member == 0 else f"best_m{member}.pt")
            score, epoch, history = train_member(fold, member, fit_idx, val_idx, stats, alpha, path)
            with (fold_dir / ("history.json" if member == 0 else f"history_m{member}.json")).open("w") as h:
                json.dump(history, h, indent=2)
            model = build_model(dims).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
            val_probs = val_probs + predict(model, val_idx, stats).cpu()
            test_probs = test_probs + predict(model, test_idx, stats).cpu()
            members.append({"path": path, "best_epoch": epoch, "val_score": score})
            del model
        val_probs, test_probs = val_probs / args.n_models, test_probs / args.n_models

        # Event decoder: tuned on the validation users of this fold (never on test).
        decode = default_decode(args)
        val_event_f1 = None
        val_tl = stitch_timelines(val_probs.numpy(), data, val_idx)
        val_truth = video_truth(data, val_tl.keys())
        if args.tune_decode:
            decode, val_event_f1 = tune_decode(val_tl, val_truth, args.event_tolerance, decode)
        else:
            val_event_f1 = event_metrics(decode_all(val_tl, decode), val_truth,
                                         args.event_tolerance)["macro_f1"]
        for m in members:
            ckpt = torch.load(m["path"], map_location="cpu", weights_only=False)
            ckpt["decode"], ckpt["users"] = decode, split_users
            torch.save(ckpt, m["path"])

        test_tl = stitch_timelines(test_probs.numpy(), data, test_idx)
        predicted.update(decode_all(test_tl, decode))
        predicted_default.update(decode_all(test_tl, default_decode(args)))
        all_probs[torch.as_tensor(test_idx)] = test_probs
        test_m = token_metrics(confusion(test_probs.to(device), test_idx))
        val_token = token_metrics(confusion(val_probs.to(device), val_idx))["macro_f1"]
        print(f"[FOLD {fold}] best epoch(s) {[m['best_epoch'] for m in members]}: val token "
              f"F1={val_token:.4f} event F1={val_event_f1:.4f} decode={decode} | TEST token "
              f"F1={test_m['macro_f1']:.4f} {test_m['f1_by_class']}")
        result = {"fold": fold, "best_epoch": members[0]["best_epoch"],
                  "best_epochs": [m["best_epoch"] for m in members],
                  "val_macro_f1": val_token, "val_event_macro_f1": val_event_f1,
                  "decode": decode, "test": test_m, "users": split_users}
        with (fold_dir / "result.json").open("w") as handle:
            json.dump(result, handle, indent=2)
        summary.append(result)

    return summarize(data, all_probs, summary, users, args, output, predicted, predicted_default)


def summarize(data, all_probs, summary, users, args, output: Path,
              predicted: Dict[int, list], predicted_default: Dict[int, list]) -> dict:
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

    truth = video_truth(data, range(len(data["videos"])))
    events = event_metrics(predicted, truth, args.event_tolerance)
    events_default = event_metrics(predicted_default, truth, args.event_tolerance)

    fold_f1 = np.array([r["test"]["macro_f1"] for r in summary])
    report = {
        "pooled_token": pooled,
        "event": events,
        "event_default_decode": events_default,
        "fold_token_macro_f1_mean": float(fold_f1.mean()),
        "fold_token_macro_f1_std": float(fold_f1.std()),
        "seq_tokens": int(all_probs.shape[1]),
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
    torch.save({"probs": all_probs, "starts": data["starts"], "video_index": data["video_index"],
                "token_valid": data["token_valid"], "predicted_events": predicted},
               output / "test_predictions.pt")
    print(f"\n[RESULT] pooled test token macro-F1 = {pooled['macro_f1']:.4f} "
          f"{pooled['f1_by_class']} | per-fold {fold_f1.mean():.4f} ± {fold_f1.std():.4f}")
    print(f"[RESULT] events (±{args.event_tolerance}s): detection F1={events['detection']['f1']:.4f} "
          f"| hit F1={events['hit']['f1']:.4f} miss F1={events['miss']['f1']:.4f} "
          f"| event macro-F1={events['macro_f1']:.4f} "
          f"(default decoder: {events_default['macro_f1']:.4f})")
    return report


# ============================================================
# CLI shared by v3 / v4 / v5
# ============================================================

def add_common_args(parser, version: str) -> None:
    """Training flags; the defaults are the final configuration (--select loss)."""

    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--videomae", default=DEFAULT_VIDEOMAE)
    parser.add_argument("--beats-checkpoint", default=BEATS_CHECKPOINT)
    parser.add_argument("--beats-root", default=BEATS_ROOT)
    parser.add_argument("--feature-cache-dir", default=str(CACHE_DIR),
                        help="Shared by v3/v4/v5: features are extracted once.")
    parser.add_argument("--output-dir", default=str(OUTPUTS / "runs" / version))
    parser.add_argument("--folds", type=int, default=5, help="Outer folds grouped by user.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    # --- experiment options (defaults reproduce the original pipeline) ---
    parser.add_argument("--seq-tokens", type=int, default=NUM_TEMPORAL_TOKENS,
                        help="Tokens per training sequence (1 token = 1/15 s). >16 stitches the "
                             "cached windows into a longer context.")
    parser.add_argument("--seq-stride", type=int, default=0,
                        help="Stride between sequences in tokens (0 = seq-tokens / 2).")
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight", choices=["rsqrt", "inv", "none"], default="rsqrt",
                        help="Per-class loss weight from the fit token counts.")
    parser.add_argument("--token-mask", type=float, default=0.0,
                        help="Training: probability of hiding each video/audio token.")
    parser.add_argument("--ema", type=float, default=0.0,
                        help="EMA decay of the weights used for validation/test (0 = off).")
    parser.add_argument("--n-models", type=int, default=1,
                        help="Models (seeds) per fold; their test probabilities are averaged.")
    parser.add_argument("--select", choices=["token_f1", "event_f1", "combo", "loss"],
                        default="loss", help="Validation metric for checkpoint selection.")
    parser.add_argument("--inner-splits", type=int, default=0,
                        help="Inner user-grouped split for validation (0 = max(3, folds)); "
                             "smaller = more validation users.")
    parser.add_argument("--tune-decode", action="store_true",
                        help="Tune the event decoder (threshold/min run/smoothing/peak) on the "
                             "validation users of each fold.")
    parser.add_argument("--video-norm", choices=["none", "mean", "std"], default="none",
                        help="Per-video feature normalisation (removes scene/session appearance).")
    parser.add_argument("--debug-subset", type=int, default=0,
                        help="Keep only this many videos (smoke tests).")
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6, help="Video decoding workers.")
    parser.add_argument("--seed", type=int, default=SEED)
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
