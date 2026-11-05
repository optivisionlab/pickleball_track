"""The three trainable heads compared in every benchmark (encoders stay frozen outside):

    v3  AVFeatureModel         video + audio   (cross-attention video <- audio)
    v4  AVTFeatureModel        video + audio + text (two cross-attentions + gated fusion)
    v5  VideoTextFeatureModel  video + text    (text-conditioned gated fusion)
"""

from __future__ import annotations

import importlib

MODEL_CLASSES = {"v3": "AVFeatureModel", "v4": "AVTFeatureModel", "v5": "VideoTextFeatureModel"}


def load_model_module(version: str):
    """The module of one version (common/models/<version>.py)."""
    if version not in MODEL_CLASSES:
        raise ValueError(f"unknown version {version!r}, expected one of {sorted(MODEL_CLASSES)}")
    return importlib.import_module(f"common.models.{version}")


def model_class(version: str):
    return getattr(load_model_module(version), MODEL_CLASSES[version])
