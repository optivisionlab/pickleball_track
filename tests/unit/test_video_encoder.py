import pytest
import torch
from transformers import VideoMAEConfig, VideoMAEModel

from pickleball_track.core.config import NUM_TEMPORAL_TOKENS, NUM_VIDEO_FRAMES
from pickleball_track.models.components.video_encoder import (
    VideoEncoder,
    adapt_videomae_to_frame_count,
)

from ..helpers import TINY_VIDEO_DIM, build_tiny_videomae


def test_position_embeddings_are_interpolated_to_trained_frame_count():
    torch.manual_seed(0)
    model = VideoMAEModel(VideoMAEConfig(
        image_size=32, patch_size=16, num_frames=16, tubelet_size=2,
        hidden_size=TINY_VIDEO_DIM, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
    ))
    before = model.embeddings.position_embeddings.clone()

    adapt_videomae_to_frame_count(model)

    after = model.embeddings.position_embeddings
    assert before.shape == (1, 8 * 4, TINY_VIDEO_DIM)
    assert after.shape == (1, 16 * 4, TINY_VIDEO_DIM)
    assert model.config.num_frames == NUM_VIDEO_FRAMES
    assert not after.requires_grad


def test_incompatible_tubelet_size_is_rejected():
    model = VideoMAEModel(VideoMAEConfig(
        image_size=32, patch_size=16, num_frames=15, tubelet_size=3,
        hidden_size=TINY_VIDEO_DIM, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
    ))

    with pytest.raises(RuntimeError, match="divisible"):
        adapt_videomae_to_frame_count(model)


def test_encoder_pools_spatial_tokens_and_masks_padded_tubelets():
    encoder = VideoEncoder(build_tiny_videomae(), torch.device("cpu"))
    pixel_values = torch.randn(1, NUM_VIDEO_FRAMES, 3, 32, 32)
    frame_valid = torch.tensor([True] * 5 + [False] * 27)

    features, padding = encoder(pixel_values, frame_valid)

    assert features.shape == (1, NUM_TEMPORAL_TOKENS, TINY_VIDEO_DIM)
    assert features.dtype == torch.float32
    # Tubelets cover 2 frames: frames 0-5 touch tubelets 0-2.
    assert padding.tolist() == [[False] * 3 + [True] * 13]
