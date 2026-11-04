import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextAnchoredHead(nn.Module):
    """logits[b, t, c] = scale * cos(token_bt, class_text_c) + bias_c.

    ``class_text`` (C, text_dim) holds one CLIP embedding per class description; it
    describes the classes, never the sample.
    """

    def __init__(self, dim: int, text_dim: int, hidden_dim: int = 256, num_classes: int = 3,
                 dropout: float = 0.3, init_temperature: float = 0.07):
        super().__init__()
        self.num_classes = num_classes
        self.video = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor, class_text: torch.Tensor) -> torch.Tensor:
        if class_text.shape[0] != self.num_classes:
            raise ValueError(f"class_text must have one row per class, got {tuple(class_text.shape)}")
        video = F.normalize(self.video(x), dim=-1)
        text = F.normalize(self.text(class_text.to(x.dtype)), dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * video @ text.t() + self.bias
