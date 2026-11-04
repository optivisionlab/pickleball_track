from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Video tokens (query) attend over a context modality; returns only the attended context."""

    def __init__(self, dim: int = 768, heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm, self.dropout = nn.LayerNorm(dim), nn.Dropout(dropout)

    def forward(
        self, video: torch.Tensor, context: torch.Tensor, context_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if context_padding_mask is not None:
            context_padding_mask = context_padding_mask.bool()
            valid = ~context_padding_mask.all(dim=1)
        else:
            valid = torch.ones(video.shape[0], dtype=torch.bool, device=video.device)
        out = torch.zeros_like(video)
        if valid.any():  # attention over fully masked keys is undefined
            attn_out, _ = self.attn(
                query=video[valid], key=context[valid], value=context[valid],
                key_padding_mask=context_padding_mask[valid] if context_padding_mask is not None else None,
                need_weights=False,
            )
            out[valid] = self.dropout(self.norm(attn_out))
        return out


class GatedFusion(nn.Module):
    """F = LN(V + alpha * audio_context + beta * text_context), gates predicted per video token."""

    def __init__(self, dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 2))
        self.norm, self.dropout = nn.LayerNorm(dim), nn.Dropout(dropout)

    def forward(self, video: torch.Tensor, audio_context: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate(torch.cat([video, audio_context, text_context], dim=-1)))
        fused = video + gates[..., 0:1] * audio_context + gates[..., 1:2] * text_context
        return self.dropout(self.norm(fused))
