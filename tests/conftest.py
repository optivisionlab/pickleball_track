"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from pickleball_track.core.config import Settings
from pickleball_track.models.model_loader import ModelBundle

from .helpers import FIXTURES, build_tiny_models


@pytest.fixture
def tiny_models() -> ModelBundle:
    return build_tiny_models()


@pytest.fixture
def clip_path() -> Path:
    return FIXTURES / "clip_5s.mp4"


@pytest.fixture
def clip_no_audio_path() -> Path:
    return FIXTURES / "clip_no_audio.mp4"


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        checkpoint_path=tmp_path / "missing.pt",
        beats_checkpoint=tmp_path / "missing_beats.pt",
        beats_root=tmp_path / "missing_unilm",
        videomae=str(tmp_path / "missing_videomae"),
        device="cpu",
        max_video_bytes=1024 * 1024,
        minio_access_key="test-access",
        minio_secret_key="test-secret",
    )


@pytest.fixture
def valid_job_request() -> dict:
    return {
        "job_id": "job-1",
        "result_id": "result-1",
        "task_id": "task-1",
        "course_id": "course-1",
        "sport_id": "sport-1",
        "sport_code": "PB",
        "mode": "VIDEO",
        "bucket": "videos",
        "object_key": "matches/match.MOV",
        "video_url": "https://storage.example/matches/match.MOV",
        "minio_endpoint": "http://minio:9000",
        "user_id": "user-1",
        "attempt": 1,
        "schema_version": 1,
        "detail": {
            "title": "Match",
            "category": "serve",
            "gradingMethod": "ai",
            "timeLimit": "60",
        },
    }
