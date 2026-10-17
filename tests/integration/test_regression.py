"""Compare the refactored code with outputs recorded from the original inference script.

`fixtures/golden_reference.json` was generated once by running the original
`inference.py` / `model.py` / `config.py` on the same inputs and seeds used here:
tiny seeded VideoMAE, FakeBEATs and AVFeatureModel on `clip_5s.mp4`, plus event
decoding of a fixed probability sequence with the original default hyperparameters.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest

from pickleball_track.core.config import InferenceParams
from pickleball_track.models.architecture import AVFeatureModel
from pickleball_track.pipelines.postprocessing import build_temporal_events
from pickleball_track.services.inference_service import VideoInferenceService

from ..helpers import FIXTURES


@pytest.fixture(scope="module")
def golden():
    return json.loads((FIXTURES / "golden_reference.json").read_text())


def test_default_hyperparameters_match_original_script(golden):
    assert asdict(InferenceParams()) == golden["defaults"]


def test_state_dict_layout_is_unchanged(golden):
    model = AVFeatureModel(video_dim=768, audio_dim=768)
    shapes = {name: list(t.shape) for name, t in model.state_dict().items()}

    assert shapes == golden["architecture_state_dict_shapes"]


def test_event_decoding_matches_original(golden):
    data = golden["postprocessing"]

    events = build_temporal_events(data["times"], np.asarray(data["probs"], dtype=np.float32))

    assert data["events"], "fixture should contain decoded events"
    assert events == data["events"]


def test_full_pipeline_matches_original(golden, tiny_models, clip_path):
    expected = golden["pipeline"]

    result = VideoInferenceService(tiny_models).predict(clip_path)

    assert result["video"].endswith(expected["video"])
    for key in ("total_hits", "total_misses", "events", "hit_events", "miss_events", "no_event_segments"):
        assert result[key] == expected[key], key
    assert len(result["temporal_predictions"]) == len(expected["temporal_predictions"])
    for new, old in zip(result["temporal_predictions"], expected["temporal_predictions"], strict=True):
        assert new["time"] == old["time"]
        assert new["label"] == old["label"]
        for label, probability in old["probabilities"].items():
            assert new["probabilities"][label] == pytest.approx(probability, abs=1e-5)
