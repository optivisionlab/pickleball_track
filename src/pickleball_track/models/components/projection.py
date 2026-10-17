import torch
import torch.nn as nn


# Maps each modality's input features into the shared fusion dimension.
class Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
