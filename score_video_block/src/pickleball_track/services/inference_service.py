"""Run the trained audiovisual fold ensemble over a whole video with sliding windows."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.config import FRAMES_PER_TOKEN, NUM_TEMPORAL_TOKENS, NUM_VIDEO_FRAMES, TARGET_FPS, InferenceParams
from ..models.components.audio_encoder import AudioTimeline
from ..models.model_loader import ModelBundle
from ..pipelines.frames import (
    CpuFrames,
    ProcessFramePool,
    WindowBatch,
    frame_counts,
    iter_frames,
    iter_window_batches,
    preprocessed_chunks,
    sample_frames,
)
from ..pipelines.postprocessing import (
    WindowPrediction,
    aggregate_predictions,
    build_json_result,
    build_temporal_events,
)
from ..pipelines.preprocessing import (
    VideoInfo,
    get_video_info,
    load_audio,
    num_sampled_frames,
    window_starts,
)
from .slot_scheduler import SlotScheduler

logger = logging.getLogger(__name__)


class VideoInferenceService:
    """Holds loaded models and runs up to ``slots`` videos at the same time.

    The models are shared (inference-mode forwards keep no state). A
    ``ProcessFramePool`` serves one video at a time, so every slot owns one; the
    first slot reuses the bundle's pool. Extra callers wait for a free slot, which
    ``SlotScheduler`` hands out shortest video first, keeping ``reserved_short_slots``
    for videos of at most ``short_video_seconds``.
    """

    def __init__(
        self,
        models: ModelBundle,
        params: InferenceParams = InferenceParams(),
        slots: int = 1,
        *,
        reserved_short_slots: int = 0,
        short_video_seconds: float = 0.0,
        slot_aging_seconds: float = 60.0,
        decoder: str = "opencv",
        ffmpeg_threads: int = 0,
    ):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.models = models
        self.params = params
        self.slots = slots
        self.decoder = decoder
        self.ffmpeg_threads = ffmpeg_threads
        self._extra_pools: list[ProcessFramePool] = []
        pools: list[ProcessFramePool | None] = [models.frame_pool]
        for _ in range(slots - 1):
            pool = None
            if models.frame_pool is not None:
                pool = ProcessFramePool(models.frame_spec, models.frame_pool.workers)
                self._extra_pools.append(pool)
            pools.append(pool)
        self.scheduler: SlotScheduler[ProcessFramePool | None] = SlotScheduler(
            pools,
            reserved_short=min(reserved_short_slots, slots - 1),
            short_seconds=short_video_seconds,
            aging_seconds=slot_aging_seconds,
        )

    @property
    def busy_slots(self) -> int:
        return self.scheduler.busy

    @property
    def waiting(self) -> tuple[int, int]:
        """(short, long) videos waiting for a slot."""
        return self.scheduler.waiting()

    def close(self) -> None:
        """Stop the preprocessing workers of the extra slots (the bundle closes its own)."""
        for pool in self._extra_pools:
            pool.close()
        self._extra_pools = []

    @torch.inference_mode()
    def predict_batch(self, batch: WindowBatch, audio: AudioTimeline | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Probabilities (B, NUM_TEMPORAL_TOKENS, num_classes) and token validity (B, NUM_TEMPORAL_TOKENS)."""
        models = self.models
        video_features, temporal_padding = models.video_encoder(batch.pixel_values, batch.frame_valid)
        audio_features = None
        if audio is not None:
            # Audio features at every token's centre time (training aligns them the same way).
            tokens = torch.tensor(batch.starts)[:, None] // FRAMES_PER_TOKEN + torch.arange(NUM_TEMPORAL_TOKENS)
            audio_features = audio.at(token_time_tensor(tokens))
        probs = models.fusion_model(video_features, audio_features, temporal_padding)
        return probs.float().cpu(), (~temporal_padding).cpu()

    def _frame_chunks(
        self,
        video_path: Path,
        info: VideoInfo,
        counts: np.ndarray,
        pool: ProcessFramePool | None,
    ) -> Iterator[torch.Tensor]:
        """Preprocessed sampled frames in order: the slot's worker processes if available, else threads."""
        models = self.models
        if pool is not None and models.frame_preprocessor is None:
            if pool.fits(info.height, info.width):
                yield from pool.chunks(video_path, counts, decoder=self.decoder, ffmpeg_threads=self.ffmpeg_threads)
                return
            logger.warning(
                "Not enough shared memory for preprocessing workers; using threads "
                "(raise /dev/shm, e.g. docker run --shm-size=1g)",
                extra={"required_bytes": pool.ring_bytes(info.height, info.width)},
            )
        preprocess = models.frame_preprocessor or CpuFrames(models.frame_spec)
        with closing(iter_frames(video_path, decoder=self.decoder, ffmpeg_threads=self.ffmpeg_threads)) as frames:
            yield from preprocessed_chunks(sample_frames(frames, counts), preprocess)

    def predict(self, video_path: str | Path) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        models = self.models
        # Metadata first (cheap): the scheduler orders waiting videos by duration.
        info = get_video_info(video_path)
        waited_at = time.perf_counter()
        pool, short = self.scheduler.acquire(info.duration)
        try:
            started_at = time.perf_counter()
            logger.info(
                "Video metadata loaded",
                extra={
                    "fps": info.fps,
                    "frames": info.num_frames,
                    "video_duration_seconds": info.duration,
                    "short_video": short,
                    "decoder": self.decoder,
                    "slot_wait_seconds": round(started_at - waited_at, 4),
                },
            )

            audio_started_at = time.perf_counter()
            audio = models.audio_encoder.timeline(load_audio(video_path))
            audio_seconds = time.perf_counter() - audio_started_at
            num_sampled = num_sampled_frames(info)
            counts = frame_counts(info.num_frames, info.fps)
            starts = window_starts(num_sampled)
            logger.info(
                "Running temporal windows",
                extra={
                    "window_count": len(starts),
                    "sampled_frames": num_sampled,
                    "has_audio": audio is not None,
                    "batch_size": models.batch_size,
                    "device": str(models.device),
                },
            )
            window_predictions = []
            windows_started_at = time.perf_counter()
            # closing() stops decoding/preprocessing even if inference raises midway.
            with closing(self._frame_chunks(video_path, info, counts, pool)) as chunks:
                batches = iter_window_batches(chunks, starts, NUM_VIDEO_FRAMES, models.batch_size)
                for batch in batches:
                    probs, token_valid = self.predict_batch(batch, audio)
                    window_predictions.extend(
                        WindowPrediction(start_frame=start, probs=window_probs, token_valid=valid)
                        for start, window_probs, valid in zip(batch.starts, probs, token_valid, strict=True)
                    )

            windows_seconds = time.perf_counter() - windows_started_at
            times, probs = aggregate_predictions(window_predictions, num_sampled)
            events = build_temporal_events(times, probs, self.params)
            result = build_json_result(video_path, times, probs, events)
            logger.info(
                "Video inference completed",
                extra={
                    "hit_count": result["total_hits"],
                    "miss_count": result["total_misses"],
                    "audio_seconds": round(audio_seconds, 4),
                    "video_and_model_seconds": round(windows_seconds, 4),
                    "duration_seconds": round(time.perf_counter() - started_at, 4),
                },
            )
        finally:
            self.scheduler.release(pool, short)
        return result


def token_time_tensor(tokens: torch.Tensor) -> torch.Tensor:
    """Centre times (s) of token indices, computed in float64 like training's ``token_center_times``."""
    frames = tokens.double() * FRAMES_PER_TOKEN
    return ((frames + (FRAMES_PER_TOKEN - 1) / 2) / TARGET_FPS).float()
