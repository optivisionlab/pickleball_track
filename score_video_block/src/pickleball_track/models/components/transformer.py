from __future__ import annotations

import torch
import torch.nn as nn


# Adds a learnable position vector to each temporal token.
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, length: int = 16, dim: int = 768):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, length, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos[:, :x.shape[1]]


# Models relationships between tokens across the clip's timeline.
class TemporalTransformer(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder, self.norm = nn.TransformerEncoder(layer, num_layers=layers), nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
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
