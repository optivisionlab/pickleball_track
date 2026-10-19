"""VideoMAE backbone wrapper: pixel windows -> temporal token features."""

from __future__ import annotations

import torch
from transformers import VideoMAEModel

from ...core.config import NUM_VIDEO_FRAMES


def adapt_videomae_to_frame_count(model: VideoMAEModel, target_frames: int = NUM_VIDEO_FRAMES) -> None:
    """Interpolate temporal position embeddings for the trained frame count."""
    old_frames = int(model.config.num_frames)
    tubelet_size = int(model.config.tubelet_size)

    if old_frames == target_frames:
        return
    if target_frames % tubelet_size != 0:
        raise RuntimeError("NUM_VIDEO_FRAMES must be divisible by VideoMAE tubelet_size.")

    old_t = old_frames // tubelet_size
    new_t = target_frames // tubelet_size
    image_size = int(model.config.image_size)
    patch_size = int(model.config.patch_size)
    height_patches = image_size // patch_size
    width_patches = image_size // patch_size
    position_embeddings = model.embeddings.position_embeddings
    expected_tokens = old_t * height_patches * width_patches

    if position_embeddings.shape[1] != expected_tokens:
        raise RuntimeError(
            "Unexpected VideoMAE positional embedding shape: "
            f"{tuple(position_embeddings.shape)}"
        )

    hidden_dim = position_embeddings.shape[-1]
    embeddings = position_embeddings.reshape(
        1, old_t, height_patches, width_patches, hidden_dim
    ).permute(0, 4, 1, 2, 3)
    embeddings = torch.nn.functional.interpolate(
        embeddings,
        size=(new_t, height_patches, width_patches),
        mode="trilinear",
        align_corners=False,
    )
    new_embeddings = embeddings.permute(0, 2, 3, 4, 1).reshape(
        1, new_t * height_patches * width_patches, hidden_dim
    ).contiguous()
    model.embeddings.position_embeddings = torch.nn.Parameter(
        new_embeddings,
        requires_grad=False,
    )
    model.config.num_frames = target_frames


class VideoEncoder:
    """Mean-pools VideoMAE spatial tokens into one feature per tubelet."""

    def __init__(self, model: VideoMAEModel, device: torch.device, amp_dtype: torch.dtype | None = None):
        self.model = model
        self.device = device
        # Autocast dtype for the backbone (CUDA only); None runs in fp32.
        self.amp_dtype = amp_dtype

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def __call__(
        self,
        pixel_values: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch of windows (B, F, C, H, W) + validity (B, F) -> features (B, T, D), padding (B, T).

        A 1-D ``frame_valid`` (F,) is accepted for a single window.
        """
        config = self.model.config
        if frame_valid.ndim == 1:
            frame_valid = frame_valid.unsqueeze(0)
        batch = pixel_values.shape[0]
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype or torch.float32,
            enabled=self.amp_dtype is not None,
        ):
            output = self.model(pixel_values=pixel_values.to(self.device, non_blocking=True))
        tokens = output.last_hidden_state
        temporal_tokens = NUM_VIDEO_FRAMES // int(config.tubelet_size)
        spatial_tokens = (int(config.image_size) // int(config.patch_size)) ** 2
        hidden_dim = tokens.shape[-1]
        expected_tokens = temporal_tokens * spatial_tokens
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"VideoMAE token mismatch: got {tokens.shape[1]}, expected {expected_tokens}"
            )

        # Pool in fp32 so reduced-precision activations do not lose accuracy here.
        features = tokens.float().reshape(batch, temporal_tokens, spatial_tokens, hidden_dim).mean(dim=2)
        frames_per_token = NUM_VIDEO_FRAMES // temporal_tokens
        temporal_padding = ~frame_valid.reshape(
            batch, temporal_tokens, frames_per_token
        ).any(dim=2)
        return features, temporal_padding.to(self.device)
