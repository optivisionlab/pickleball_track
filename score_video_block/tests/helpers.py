"""Small deterministic stand-ins for the real model artifacts.

The real checkpoints are deployment artifacts not stored in Git, so tests run the
real VideoMAE/fusion code with tiny random weights and a controlled BEATs fake.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import VideoMAEConfig, VideoMAEModel

from pickleball_track.core.config import AUDIO_SR, BEATS_PATCH_SEC, LABEL2ID
from pickleball_track.models.architecture import AVTFeatureModel, FoldEnsemble
from pickleball_track.models.components.audio_encoder import FREQUENCY_PATCHES, AudioEncoder
from pickleball_track.models.components.video_encoder import (
    VideoEncoder,
    adapt_videomae_to_frame_count,
)
from pickleball_track.models.model_loader import ModelBundle
from pickleball_track.pipelines.frames import FrameSpec

FIXTURES = Path(__file__).parent / "fixtures"
TINY_IMAGE_SIZE = 32
TINY_HIDDEN = 32
# Global mean + 2x2 grid per token.
TINY_VIDEO_DIM = TINY_HIDDEN * 5
TINY_AUDIO_DIM = 8
TINY_TEXT_DIM = 6


class FakeBEATs(torch.nn.Module):
    """Mimics ``BEATs.extract_features``: per 0.16 s time step, FREQUENCY_PATCHES time-major tokens."""

    step = int(BEATS_PATCH_SEC * AUDIO_SR)

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, TINY_AUDIO_DIM)
        self.freq = torch.nn.Parameter(torch.randn(FREQUENCY_PATCHES, TINY_AUDIO_DIM))

    def extract_features(self, source, padding_mask=None):
        steps = source.shape[-1] // self.step
        chunks = source[:, : steps * self.step].reshape(source.shape[0], steps, self.step)
        stats = torch.stack(
            [chunks.mean(-1), chunks.std(-1), chunks.abs().max(-1).values, chunks.pow(2).mean(-1)], dim=-1
        )
        tokens = self.proj(stats)[:, :, None, :] + self.freq  # (B, steps, 8, D)
        tokens = tokens.reshape(source.shape[0], steps * FREQUENCY_PATCHES, TINY_AUDIO_DIM)
        return tokens, torch.zeros(tokens.shape[:2], dtype=torch.bool)


def build_tiny_videomae(seed: int = 0) -> VideoMAEModel:
    torch.manual_seed(seed)
    config = VideoMAEConfig(
        image_size=TINY_IMAGE_SIZE, patch_size=16, num_channels=3, num_frames=16, tubelet_size=2,
        hidden_size=TINY_HIDDEN, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, use_mean_pooling=False,
    )
    model = VideoMAEModel(config)
    adapt_videomae_to_frame_count(model)
    return model.eval()


def build_tiny_spec() -> FrameSpec:
    return FrameSpec(size=TINY_IMAGE_SIZE)


def build_tiny_fold(seed: int = 0) -> AVTFeatureModel:
    torch.manual_seed(seed)
    return AVTFeatureModel(
        video_dim=TINY_VIDEO_DIM, audio_dim=TINY_AUDIO_DIM, text_dim=TINY_TEXT_DIM,
        fusion_dim=16, heads=2, temporal_layers=1,
    ).eval()


def build_tiny_checkpoint(folds: int = 2, seed: int = 0) -> dict:
    """A checkpoint in the format written by scripts/export_checkpoint.py."""
    generator = torch.Generator().manual_seed(seed)
    entries = []
    for k in range(folds):
        entries.append({
            "fold": k + 1,
            "model": build_tiny_fold(seed + k).state_dict(),
            "feature_mean": {"video": torch.randn(TINY_VIDEO_DIM, generator=generator) * 0.1,
                             "audio": torch.randn(TINY_AUDIO_DIM, generator=generator) * 0.1},
            "feature_std": {"video": torch.rand(TINY_VIDEO_DIM, generator=generator) + 0.5,
                            "audio": torch.rand(TINY_AUDIO_DIM, generator=generator) + 0.5},
        })
    return {
        "label2id": dict(LABEL2ID),
        "config": {"fusion_dim": 16, "heads": 2, "temporal_layers": 1, "seq_tokens": 16},
        "class_text": torch.randn(len(LABEL2ID), TINY_TEXT_DIM, generator=generator),
        "folds": entries,
    }


def build_tiny_ensemble(folds: int = 2, seed: int = 0) -> FoldEnsemble:
    checkpoint = build_tiny_checkpoint(folds, seed)
    models = []
    for fold in checkpoint["folds"]:
        model = build_tiny_fold()
        model.load_state_dict(fold["model"])
        models.append(model)
    stack = {stat: {name: torch.stack([f[stat][name] for f in checkpoint["folds"]]) for name in ("video", "audio")}
             for stat in ("feature_mean", "feature_std")}
    return FoldEnsemble(models, checkpoint["class_text"], stack["feature_mean"], stack["feature_std"]).eval()


def build_fake_beats(seed: int = 0) -> FakeBEATs:
    torch.manual_seed(seed)
    return FakeBEATs().eval()


def build_tiny_models() -> ModelBundle:
    device = torch.device("cpu")
    return ModelBundle(
        device=device,
        frame_spec=build_tiny_spec(),
        video_encoder=VideoEncoder(build_tiny_videomae(), device),
        audio_encoder=AudioEncoder(build_fake_beats(), device),
        fusion_model=build_tiny_ensemble(),
    )
