from __future__ import annotations

import torch
import torch.nn as nn


# Lets video tokens incorporate audio context while respecting padding masks.
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm, self.dropout = nn.LayerNorm(dim), nn.Dropout(dropout)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
        video_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
