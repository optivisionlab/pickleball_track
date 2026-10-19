"""Small deterministic stand-ins for the real model artifacts.

The real checkpoints are deployment artifacts not stored in Git, so tests run the
real VideoMAE/fusion code with tiny random weights and a controlled BEATs fake.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import VideoMAEConfig, VideoMAEImageProcessor, VideoMAEModel

from pickleball_track.core.config import AUDIO_SR
from pickleball_track.models.architecture import AVFeatureModel
from pickleball_track.models.components.audio_encoder import AudioEncoder
from pickleball_track.models.components.video_encoder import (
    VideoEncoder,
    adapt_videomae_to_frame_count,
)
from pickleball_track.models.model_loader import ModelBundle

FIXTURES = Path(__file__).parent / "fixtures"
TINY_VIDEO_DIM = 32
TINY_AUDIO_DIM = 8


class FakeBEATs(torch.nn.Module):
    """Mimics ``BEATs.extract_features``: one feature per 1600-sample chunk."""

    chunk = AUDIO_SR // 10

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, TINY_AUDIO_DIM)

    def extract_features(self, source, padding_mask=None):
        usable = source.shape[-1] // self.chunk * self.chunk
        chunks = source[:, :usable].reshape(source.shape[0], -1, self.chunk)
        stats = torch.stack(
            [chunks.mean(-1), chunks.std(-1), chunks.abs().max(-1).values, chunks.pow(2).mean(-1)],
            dim=-1,
        )
        padding = padding_mask[:, :usable].reshape(source.shape[0], -1, self.chunk).all(-1)
        return self.proj(stats), padding


def build_tiny_videomae(seed: int = 0) -> VideoMAEModel:
    torch.manual_seed(seed)
    config = VideoMAEConfig(
        image_size=32, patch_size=16, num_channels=3, num_frames=16, tubelet_size=2,
        hidden_size=TINY_VIDEO_DIM, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, use_mean_pooling=False,
    )
    model = VideoMAEModel(config)
    adapt_videomae_to_frame_count(model)
    return model.eval()


def build_tiny_processor() -> VideoMAEImageProcessor:
    return VideoMAEImageProcessor(
        size={"shortest_edge": 32},
        crop_size={"height": 32, "width": 32},
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )


def build_tiny_fusion(seed: int = 0) -> AVFeatureModel:
    torch.manual_seed(seed)
    return AVFeatureModel(video_dim=TINY_VIDEO_DIM, audio_dim=TINY_AUDIO_DIM).eval()


def build_fake_beats(seed: int = 0) -> FakeBEATs:
    torch.manual_seed(seed)
    return FakeBEATs().eval()


def build_tiny_models() -> ModelBundle:
    device = torch.device("cpu")
    return ModelBundle(
        device=device,
        processor=build_tiny_processor(),
        video_encoder=VideoEncoder(build_tiny_videomae(), device),
        audio_encoder=AudioEncoder(build_fake_beats(), device),
        fusion_model=build_tiny_fusion(),
    )
