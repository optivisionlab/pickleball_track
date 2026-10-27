"""FastAPI application factory and model lifespan."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI

from ..core.config import Settings, get_settings
from ..core.logging import configure_logging
from ..integrations.minio import create_minio_client
from ..models.model_loader import load_models
from ..services.inference_service import VideoInferenceService
from ..services.job_gate import JobGate
from .errors import register_exception_handlers
from .routes import health, inference

logger = logging.getLogger(__name__)


def _default_inference_factory(settings: Settings) -> VideoInferenceService:
    return VideoInferenceService(
        load_models(settings),
        settings.inference,
        slots=settings.inference_slots,
        reserved_short_slots=settings.reserved_short_slots,
        short_video_seconds=settings.short_video_seconds,
        slot_aging_seconds=settings.slot_aging_seconds,
        decoder=settings.decoder,
        ffmpeg_threads=settings.ffmpeg_threads,
    )


def create_app(
    settings: Settings | None = None,
    *,
    inference_factory: Callable[[Settings], VideoInferenceService] = _default_inference_factory,
    minio_client_factory: Callable[[str, Settings], Any] = create_minio_client,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Models are loaded exactly once per worker. A failed load keeps the process
        # up so /ready reports 503 and the error is visible in the logs.
        try:
            app.state.inference_service = inference_factory(settings)
        except Exception:
            logger.exception("Model loading failed; service is not ready")
            app.state.inference_service = None
        app.state.job_gate = JobGate(
            settings.max_pending_jobs, settings.busy_retry_after_seconds, settings.result_cache_seconds
        )
        app.state.download_slots = threading.BoundedSemaphore(settings.max_concurrent_downloads)
        try:
            yield
        finally:
            app.state.job_gate.close()
            close_slots = getattr(app.state.inference_service, "close", None)
            if close_slots is not None:
                close_slots()
            models = getattr(app.state.inference_service, "models", None)
            app.state.inference_service = None
            if models is not None:
                models.close()
                if models.device.type == "cuda":
                    torch.cuda.empty_cache()

    app = FastAPI(title="Pickleball AI Pipeline", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.inference_service = None
    app.state.minio_client_factory = minio_client_factory
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(inference.router)
    return app


app = create_app()
