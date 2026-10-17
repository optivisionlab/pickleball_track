"""BEATs backbone wrapper: padded waveform windows -> audio token features."""

from __future__ import annotations

from typing import Any

import torch


class AudioEncoder:
    """Calls ``BEATs.extract_features`` on a single padded audio window."""

    def __init__(self, model: Any, device: torch.device):
        # `model` is a microsoft/unilm BEATs instance loaded from its source checkout.
        self.model = model
        self.device = device

    @torch.inference_mode()
    def __call__(
        self,
        segment: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return features (1, N, D) and feature padding mask (1, N) on device."""
        features, feature_padding = self.model.extract_features(
            segment.to(self.device, non_blocking=True),
            padding_mask=padding_mask.to(self.device, non_blocking=True),
        )
        return features.float(), feature_padding.bool()
