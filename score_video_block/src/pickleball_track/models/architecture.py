from __future__ import annotations

import torch
import torch.nn as nn

from ..core.config import NUM_TEMPORAL_TOKENS
from .components.classifier import TemporalPredictionHead
from .components.fusion import CrossAttentionFusion
from .components.projection import Projection
from .components.transformer import TemporalPositionalEncoding, TemporalTransformer


class AVFeatureModel(nn.Module):
    """Temporal audiovisual classifier used by the inference pipeline.

    Attribute names define the checkpoint's state_dict keys; do not rename them.
    """

    def __init__(self, video_dim: int, audio_dim: int, fusion_dim: int = 768, cross_heads: int = 16,
                 temporal_heads: int = 12, temporal_layers: int = 2, dropout: float = 0.1,
                 head_hidden_dim: int = 256, head_dropout: float = 0.3, num_classes: int = 3):
        super().__init__()
        self.video_proj, self.audio_proj = Projection(video_dim, fusion_dim, dropout), Projection(audio_dim, fusion_dim, dropout)
        self.fusion = CrossAttentionFusion(fusion_dim, cross_heads, dropout)
        self.temporal_pos = TemporalPositionalEncoding(length=NUM_TEMPORAL_TOKENS, dim=fusion_dim)
        self.temporal = TemporalTransformer(fusion_dim, temporal_heads, temporal_layers, dropout)
        self.head = TemporalPredictionHead(fusion_dim, head_hidden_dim, num_classes, head_dropout)

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        audio_padding_mask: torch.Tensor | None,
        temporal_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Project both modalities, fuse them, then classify each temporal token.
        video = self.video_proj(video_features)
        audio = self.audio_proj(audio_features)
        x = self.fusion(video, audio, audio_padding_mask, temporal_padding_mask)
        if temporal_padding_mask is not None:
            x = x.masked_fill(temporal_padding_mask.unsqueeze(-1), 0.0)
        x = self.temporal_pos(x)
        if temporal_padding_mask is not None:
            # Positional embeddings must not turn padded timesteps into nonzero inputs.
            x = x.masked_fill(temporal_padding_mask.unsqueeze(-1), 0.0)
        x = self.temporal(x, temporal_padding_mask)
        return self.head(x)
