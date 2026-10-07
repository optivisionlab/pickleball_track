from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, num_heads=16, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video, audio, audio_padding_mask=None):
        attn_out, _ = self.attn(video, audio, audio, key_padding_mask=audio_padding_mask, need_weights=False)
        return self.norm(video + self.dropout(attn_out))


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, length=16, dim=768):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, length, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos[:, :x.shape[1]]


class TemporalTransformer(nn.Module):
    def __init__(self, dim=768, heads=12, layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask=None):
        return self.norm(self.encoder(x, src_key_padding_mask=padding_mask))


class TemporalPredictionHead(nn.Module):
    def __init__(self, dim=768, hidden_dim=256, num_classes=3, dropout=0.3):
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

    def forward(self, x):
        return self.net(x)

class AVFeatureModel(nn.Module):
    def __init__(self, video_dim, audio_dim, fusion_dim=768, cross_heads=16, temporal_heads=12, temporal_layers=2, dropout=0.1, head_hidden_dim=256, head_dropout=0.3, num_classes=3):
        super().__init__()
        self.video_proj = Projection(video_dim, fusion_dim, dropout)
        self.audio_proj = Projection(audio_dim, fusion_dim, dropout)
        self.fusion = CrossAttentionFusion(fusion_dim, cross_heads, dropout)
        self.temporal_pos = TemporalPositionalEncoding(16, fusion_dim)
        self.temporal = TemporalTransformer(fusion_dim, temporal_heads, temporal_layers, dropout)
        self.head = TemporalPredictionHead(fusion_dim, head_hidden_dim, num_classes, head_dropout)
        self._assert_all_trainable()

    def _assert_all_trainable(self):
        if not any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Training model has no trainable parameters.")

    def forward(self, video_features, audio_features, audio_padding_mask, temporal_padding_mask):
        video = self.video_proj(video_features)
        audio = self.audio_proj(audio_features)
        x = self.fusion(video, audio, audio_padding_mask)

        if temporal_padding_mask is not None:
            x = x.masked_fill(temporal_padding_mask.unsqueeze(-1), 0.0)

        x = self.temporal_pos(x)

        if temporal_padding_mask is not None:
            x = x.masked_fill(temporal_padding_mask.unsqueeze(-1), 0.0)

        x = self.temporal(x, temporal_padding_mask)
        return self.head(x)


class TemporalFocalLoss(nn.Module):
    def __init__(self, alpha=(0.25, 0.5, 1.0), gamma=2.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits, labels, padding_mask=None):
        B, T, C = logits.shape
        logits = logits.reshape(B * T, C)
        labels = labels.reshape(B * T).long()

        if padding_mask is not None:
            valid = ~padding_mask.reshape(B * T).bool()
            logits = logits[valid]
            labels = labels[valid]

        if logits.numel() == 0:
            return logits.sum() * 0.0
        
        log_pt = F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -(1.0 - pt).pow(self.gamma) * log_pt
        loss = loss * self.alpha.to(logits.device)[labels]

        return loss.mean()