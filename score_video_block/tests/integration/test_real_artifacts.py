"""Runs only where the deployment artifacts exist (not in Git).

Set PICKLEBALL_TEST_VIDEO to a real pickleball clip to also run real inference.
"""

import os
from pathlib import Path

import pytest

from pickleball_track.core.config import get_settings
from pickleball_track.models.model_loader import load_models, missing_model_assets
from pickleball_track.services.inference_service import VideoInferenceService

settings = get_settings()
pytestmark = pytest.mark.skipif(
    bool(missing_model_assets(settings)),
    reason="model artifacts unavailable: " + "; ".join(missing_model_assets(settings)),
)


@pytest.fixture(scope="module")
def models():
    return load_models(settings)


def test_real_checkpoint_loads_strictly(models):
    assert not models.fusion_model.training
    assert models.fusion_model.video_proj.net[0].in_features == models.video_encoder.hidden_size


@pytest.mark.skipif(not os.getenv("PICKLEBALL_TEST_VIDEO"), reason="PICKLEBALL_TEST_VIDEO not set")
def test_real_inference(models):
    result = VideoInferenceService(models).predict(Path(os.environ["PICKLEBALL_TEST_VIDEO"]))

    assert result["temporal_predictions"]
