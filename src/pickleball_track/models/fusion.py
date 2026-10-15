from __future__ import annotations

import torch
import torch.nn as nn


# Maps each modality's input features into the shared fusion dimension.
class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


# Lets video tokens incorporate audio context while respecting padding masks.
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, heads=16, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm, self.dropout = nn.LayerNorm(dim), nn.Dropout(dropout)

    def forward(self, video, audio, audio_padding_mask=None, video_padding_mask=None):
        no_audio = audio_padding_mask.all(dim=1) if audio_padding_mask is not None else torch.zeros(
            video.shape[0], dtype=torch.bool, device=video.device
        )
        no_video = video_padding_mask.all(dim=1) if video_padding_mask is not None else torch.zeros(
            video.shape[0], dtype=torch.bool, device=video.device
        )

        out = video.clone()
        valid = ~(no_audio | no_video)
        if valid.any():  # Skip fully padded samples; attention over empty keys is undefined.
            attn_out, _ = self.attn(
                query=video[valid], key=audio[valid], value=audio[valid],
                key_padding_mask=audio_padding_mask[valid] if audio_padding_mask is not None else None,
                need_weights=False,
            )
            fused = video[valid] + self.dropout(attn_out)
            if video_padding_mask is not None:
                fused = fused.masked_fill(video_padding_mask[valid].unsqueeze(-1), 0.0)
            out[valid] = fused
        return self.norm(out)


# Adds a learnable position vector to each temporal token.
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, length=16, dim=768):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, length, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos[:, :x.shape[1]]


# Models relationships between tokens across the clip's timeline.
class TemporalTransformer(nn.Module):
    def __init__(self, dim=768, heads=12, layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder, self.norm = nn.TransformerEncoder(layer, num_layers=layers), nn.LayerNorm(dim)

    def forward(self, x, padding_mask=None):
        all_padding = padding_mask.all(dim=1) if padding_mask is not None else torch.zeros(
            x.shape[0], dtype=torch.bool, device=x.device
        )
        out = torch.zeros_like(x)
        valid = ~all_padding
        if valid.any():
            # Avoid passing all-padding sequences to the transformer, which can yield NaNs.
            out[valid] = self.encoder(
                x[valid], src_key_padding_mask=padding_mask[valid] if padding_mask is not None else None
            )
        return self.norm(out)


# Converts each contextualized token into class logits.
class TemporalPredictionHead(nn.Module):
    def __init__(self, dim=768, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AVFeatureModel(nn.Module):
    """Temporal audiovisual classifier used by the inference pipeline."""

    def __init__(self, video_dim, audio_dim, fusion_dim=768, cross_heads=16,
                 temporal_heads=12, temporal_layers=2, dropout=0.1,
                 head_hidden_dim=256, head_dropout=0.3, num_classes=3):
        super().__init__()
        self.video_proj, self.audio_proj = Projection(video_dim, fusion_dim, dropout), Projection(audio_dim, fusion_dim, dropout)
        self.fusion = CrossAttentionFusion(fusion_dim, cross_heads, dropout)
        self.temporal_pos = TemporalPositionalEncoding(length=16, dim=fusion_dim)
        self.temporal = TemporalTransformer(fusion_dim, temporal_heads, temporal_layers, dropout)
        self.head = TemporalPredictionHead(fusion_dim, head_hidden_dim, num_classes, head_dropout)

    def forward(self, video_features, audio_features, audio_padding_mask, temporal_padding_mask):
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