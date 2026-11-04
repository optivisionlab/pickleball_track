"""The v4 audiovisual + text model and the fold ensemble served in production.

Training code: baseline/v4/model.py (AVTFeatureModel, head_type="text") and
baseline/pickleball_common.py. Attribute names define the checkpoint's state_dict
keys; do not rename them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.config import NUM_TEMPORAL_TOKENS
from .components.classifier import TextAnchoredHead
from .components.fusion import CrossModalAttention, GatedFusion
from .components.projection import Projection
from .components.transformer import TemporalPositionalEncoding, TemporalTransformer


class AVTFeatureModel(nn.Module):
    """Video tokens attend over audio tokens and the class descriptions, then a temporal
    transformer and a text-anchored head classify every token (no_event / hit / miss)."""

    def __init__(self, video_dim: int, audio_dim: int, text_dim: int, fusion_dim: int = 256,
                 heads: int = 8, temporal_layers: int = 2, dropout: float = 0.2,
                 head_hidden_dim: int = 256, head_dropout: float = 0.3, num_classes: int = 3,
                 temporal_length: int = NUM_TEMPORAL_TOKENS):
        super().__init__()
        self.video_proj = Projection(video_dim, fusion_dim, dropout)
        self.audio_proj = Projection(audio_dim, fusion_dim, dropout)
        self.text_proj = Projection(text_dim, fusion_dim, dropout)
        self.audio_cross_attn = CrossModalAttention(fusion_dim, heads, dropout)
        self.text_cross_attn = CrossModalAttention(fusion_dim, heads, dropout)
        self.fusion = GatedFusion(fusion_dim, dropout)
        self.temporal_pos = TemporalPositionalEncoding(length=temporal_length, dim=fusion_dim)
        self.temporal = TemporalTransformer(fusion_dim, heads, temporal_layers, dropout)
        self.head = TextAnchoredHead(fusion_dim, text_dim, head_hidden_dim, num_classes, head_dropout)

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        audio_padding_mask: torch.Tensor | None,
        temporal_padding_mask: torch.Tensor | None,
        class_text: torch.Tensor,
    ) -> torch.Tensor:
        """(B, T, Dv), (B, T, Da), masks (B, T) True = padding, class_text (C, Dt) -> logits (B, T, C)."""
        video = self.video_proj(video_features)
        audio = self.audio_proj(audio_features)
        # Every token attends over the C class descriptions (the same for every sample).
        text = self.text_proj(class_text).unsqueeze(0).expand(video.shape[0], -1, -1)
        audio_context = self.audio_cross_attn(video, audio, audio_padding_mask)
        text_context = self.text_cross_attn(video, text)
        x = self.fusion(video, audio_context, text_context)
        x = self.temporal(self.temporal_pos(x), temporal_padding_mask)
        return self.head(x, class_text)


class FoldEnsemble(nn.Module):
    """Averages the softmax of the per-fold models of a cross-validation run.

    Each fold standardizes its inputs with the statistics of its own training data.
    A window without audio gets zero (standardized) audio with all but one audio
    token masked, as in training.
    """

    def __init__(self, models: list[AVTFeatureModel], class_text: torch.Tensor,
                 feature_mean: dict[str, torch.Tensor], feature_std: dict[str, torch.Tensor]):
        super().__init__()
        if not models:
            raise ValueError("FoldEnsemble needs at least one model")
        self.models = nn.ModuleList(models)
        self.register_buffer("class_text", class_text.float())
        # (folds, D) statistics per modality.
        self.register_buffer("video_mean", feature_mean["video"].float())
        self.register_buffer("video_std", feature_std["video"].float())
        self.register_buffer("audio_mean", feature_mean["audio"].float())
        self.register_buffer("audio_std", feature_std["audio"].float())

    @property
    def video_dim(self) -> int:
        return int(self.video_mean.shape[-1])

    @property
    def audio_dim(self) -> int:
        return int(self.audio_mean.shape[-1])

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor | None,
        temporal_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Video (B, T, Dv), audio (B, T, Da) or None (no audio), padding (B, T) -> probabilities (B, T, C)."""
        batch, length = video_features.shape[:2]
        audio_padding = torch.zeros(batch, length, dtype=torch.bool, device=video_features.device)
        if audio_features is None:
            audio_padding[:, 1:] = True  # keep one zero token so attention stays defined
        probs = 0.0
        for k, model in enumerate(self.models):
            video = (video_features - self.video_mean[k]) / self.video_std[k]
            if audio_features is None:
                audio = torch.zeros(batch, length, self.audio_dim, device=video.device)
            else:
                audio = (audio_features - self.audio_mean[k]) / self.audio_std[k]
            logits = model(video, audio, audio_padding, temporal_padding_mask, self.class_text)
            probs = probs + torch.softmax(logits.float(), dim=-1)
        return probs / len(self.models)
