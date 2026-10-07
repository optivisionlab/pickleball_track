from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_TEMPORAL_TOKENS = 16
FUSION_DIM = 768
NUM_CLASSES = 3


class Projection(nn.Module):
    """Project a cached modality embedding into the fusion space."""

    def __init__(self, in_dim: int, out_dim: int = FUSION_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextConditionedGatedFusion(nn.Module):
    """
    Video-conditioned fusion with one global text vector.

    V: [B,T,D]
    T: [B,D]

    A separate gate is predicted for every temporal video token:
        g_t = sigmoid(MLP([V_t ; T]))
        F_t = LN(V_t + g_t * T)

    This is preferable to cross-attention when text contains only one token.
    """

    def __init__(self, dim: int = FUSION_DIM, dropout: float = 0.1):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.text_scale = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if video.ndim != 3:
            raise ValueError(f"video must be [B,T,D], got {tuple(video.shape)}")
        if text.ndim != 2:
            raise ValueError(f"text must be [B,D], got {tuple(text.shape)}")
        if video.shape[0] != text.shape[0] or video.shape[-1] != text.shape[-1]:
            raise ValueError(
                f"Video/text shape mismatch: {tuple(video.shape)} vs {tuple(text.shape)}"
            )

        text_expanded = text.unsqueeze(1).expand(-1, video.shape[1], -1)
        gate = self.gate(torch.cat([video, text_expanded], dim=-1))

        fused = video + gate * self.text_scale * text_expanded
        return self.dropout(self.norm(fused))


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, length: int = NUM_TEMPORAL_TOKENS, dim: int = FUSION_DIM):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, length, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pos.shape[1]:
            raise ValueError(
                f"Temporal length {x.shape[1]} exceeds {self.pos.shape[1]}"
            )
        return x + self.pos[:, :x.shape[1]]


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        dim: int = FUSION_DIM,
        heads: int = 12,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            return self.norm(self.encoder(x))

        mask = padding_mask.bool()
        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"padding_mask={tuple(mask.shape)} incompatible with x={tuple(x.shape)}"
            )

        # Avoid undefined all-masked rows inside attention.
        out = torch.zeros_like(x)
        valid_rows = ~mask.all(dim=1)

        if valid_rows.any():
            out[valid_rows] = self.encoder(
                x[valid_rows],
                src_key_padding_mask=mask[valid_rows],
            )

        return self.norm(out)


class TemporalPredictionHead(nn.Module):
    def __init__(
        self,
        dim: int = FUSION_DIM,
        hidden_dim: int = 256,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoTextFeatureModel(nn.Module):
    """
    Video + Text temporal event classifier.

    Inputs are PRECOMPUTED features:
        video_features: [B,16,Dv]
        text_features:  [B,Dt]
        temporal_padding_mask: [B,16], True = padding

    Architecture:
        VideoMAE features -> Video projection
        CLIP features     -> Text projection
        Video + Text      -> temporal gated residual fusion
        -> positional encoding
        -> temporal Transformer
        -> temporal classifier

    Outputs:
        logits: [B,16,3]
        classes: 0=no_event, 1=hit, 2=miss
    """

    def __init__(
        self,
        video_dim: int,
        text_dim: int,
        fusion_dim: int = FUSION_DIM,
        temporal_heads: int = 12,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
        temporal_length: int = NUM_TEMPORAL_TOKENS,
    ):
        super().__init__()

        self.video_proj = Projection(video_dim, fusion_dim, dropout)
        self.text_proj = Projection(text_dim, fusion_dim, dropout)
        self.fusion = TextConditionedGatedFusion(fusion_dim, dropout)
        self.temporal_pos = TemporalPositionalEncoding(temporal_length, fusion_dim)
        self.temporal = TemporalTransformer(
            fusion_dim, temporal_heads, temporal_layers, dropout
        )
        self.head = TemporalPredictionHead(
            fusion_dim, head_hidden_dim, num_classes, head_dropout
        )

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        temporal_padding_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        if video_features.ndim != 3:
            raise ValueError(
                f"video_features must be [B,T,D], got {tuple(video_features.shape)}"
            )
        if video_features.shape[1] != NUM_TEMPORAL_TOKENS:
            raise ValueError(
                f"Expected {NUM_TEMPORAL_TOKENS} temporal tokens, "
                f"got {video_features.shape[1]}"
            )

        # CLIP cached feature may be [B,1,D].
        if text_features.ndim == 3:
            if text_features.shape[1] != 1:
                raise ValueError(
                    "This model expects one global text token [B,1,D]."
                )
            text_features = text_features[:, 0]

        video = self.video_proj(video_features)
        text = self.text_proj(text_features)

        fused = self.fusion(video, text)
        fused = self.temporal_pos(fused)
        temporal = self.temporal(fused, temporal_padding_mask)
        logits = self.head(temporal)

        if return_features:
            return logits, {
                "video_projected": video,
                "text_projected": text,
                "fused": fused,
                "temporal": temporal,
            }

        return logits


class TemporalFocalLoss(nn.Module):
    """
    Multi-class focal loss for [B,T,C] temporal logits.

    Padding positions are excluded from the loss.
    """

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

        if alpha is None:
            self.register_buffer("alpha", torch.ones(NUM_CLASSES))
        else:
            a = torch.as_tensor(alpha, dtype=torch.float32)
            if a.numel() != NUM_CLASSES:
                raise ValueError(f"alpha must contain {NUM_CLASSES} values")
            self.register_buffer("alpha", a)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.ndim != 3 or targets.ndim != 2:
            raise ValueError(
                f"Expected logits [B,T,C] and targets [B,T], "
                f"got {tuple(logits.shape)}, {tuple(targets.shape)}"
            )

        b, t, c = logits.shape
        flat_logits = logits.reshape(-1, c)
        flat_targets = targets.reshape(-1).long()

        valid = torch.ones_like(flat_targets, dtype=torch.bool)

        if padding_mask is not None:
            valid &= ~padding_mask.reshape(-1).bool()

        if self.ignore_index is not None:
            valid &= flat_targets != self.ignore_index

        if not valid.any():
            return logits.sum() * 0.0

        flat_logits = flat_logits[valid]
        flat_targets = flat_targets[valid]

        log_probs = F.log_softmax(flat_logits, dim=-1)
        log_pt = log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        alpha_t = self.alpha.to(logits.device)[flat_targets]
        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt
        return loss.mean()