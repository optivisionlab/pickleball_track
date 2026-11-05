"""Media helpers shared by every benchmark: audio decoding, frozen VideoMAE / BEATs
loaders, token pooling, seeding and clip-level logits.

Every task folder adds the baseline root to sys.path and imports this module as `bc`.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pretrained encoders shared by v3 / v4 / v5 of every sport (git-ignored, common/pretrained/):
# BEATs checkpoints + their source code (unilm) and VideoMAE-base (self-supervised, encoder only).
# The other Hugging Face models (VideoMAE K400, CLIP, Whisper) are loaded by name from the HF cache.
PRETRAINED = os.environ.get("PB_PRETRAINED", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained"))
VIDEOMAE_BASE = f"{PRETRAINED}/videomae-base"  # = MCG-NJU/videomae-base encoder (identical weights)
BEATS_ROOT = f"{PRETRAINED}/unilm"  # BEATs source code
BEATS = {
    "base": f"{PRETRAINED}/BEATs_iter3_plus_AS2M.pt",
    "ft": f"{PRETRAINED}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
}

NUM_VIDEO_FRAMES = 32
NUM_TEMPORAL_TOKENS = 16
IMAGE_SIZE = 224
AUDIO_SR = 16000
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClipRow:
    """One labelled clip; game_id is the grouping key of the cross-validation."""
    video_id: str
    game_id: str
    path: str
    label: int


def read_audio(path: str, sample_rate: int = AUDIO_SR) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "f32le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
        return np.frombuffer(out, dtype=np.float32).copy()
    except (subprocess.SubprocessError, OSError):
        return np.zeros(0, dtype=np.float32)


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


def pool_tokens(features: torch.Tensor, valid: torch.Tensor, count: int):
    """Masked adaptive average pooling of [T, D] valid tokens to [count, D]."""
    tokens = features[valid]
    if tokens.shape[0] == 0:
        return torch.zeros(count, features.shape[-1], device=features.device), False
    pooled = F.adaptive_avg_pool1d(tokens.t().unsqueeze(0).float(), count)[0].t()
    return pooled, True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip_logits(token_logits: torch.Tensor, video_pad: torch.Tensor) -> torch.Tensor:
    """Mean of token logits over valid tokens: [B,T,C] -> [B,C]."""
    weight = (~video_pad).unsqueeze(-1).to(token_logits.dtype)
    return (token_logits * weight).sum(1) / weight.sum(1).clamp_min(1.0)
