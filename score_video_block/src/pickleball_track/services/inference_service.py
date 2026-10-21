"""Run the trained audiovisual model over a whole video with sliding windows."""

from __future__ import annotations

import logging
import queue
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import torch

from ..core.config import NUM_VIDEO_FRAMES, InferenceParams
from ..models.model_loader import ModelBundle
from ..pipelines.frames import (
    FrameTimeline,
    ProcessFramePool,
    ProcessorFrames,
    WindowBatch,
    iter_frames,
    iter_window_batches,
    preprocessed_chunks,
)
from ..pipelines.postprocessing import (
    WindowPrediction,
    aggregate_predictions,
    build_json_result,
    build_temporal_events,
)
from ..pipelines.preprocessing import (
    VideoInfo,
    build_windows,
    get_audio_window,
    get_video_info,
    load_audio,
)

logger = logging.getLogger(__name__)


class VideoInferenceService:
    """Holds loaded models and runs up to ``slots`` videos at the same time.

    The models are shared (inference-mode forwards keep no state). A
    ``ProcessFramePool`` serves one video at a time, so every slot owns one; the
    first slot reuses the bundle's pool. Extra callers wait for a free slot.
    """

    def __init__(self, models: ModelBundle, params: InferenceParams = InferenceParams(), slots: int = 1):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.models = models
        self.params = params
        self.slots = slots
        self._extra_pools: list[ProcessFramePool] = []
        self._free: queue.Queue[ProcessFramePool | None] = queue.Queue()
        self._free.put(models.frame_pool)
        for _ in range(slots - 1):
            pool = None
            if models.frame_pool is not None:
                pool = ProcessFramePool(models.processor, models.frame_pool.workers)
                self._extra_pools.append(pool)
            self._free.put(pool)

    @property
    def busy_slots(self) -> int:
        return self.slots - self._free.qsize()

    def close(self) -> None:
        """Stop the preprocessing workers of the extra slots (the bundle closes its own)."""
        for pool in self._extra_pools:
            pool.close()
        self._extra_pools = []

    @torch.inference_mode()
    def predict_batch(
        self,
        batch: WindowBatch,
        waveform: torch.Tensor,
        timeline: FrameTimeline,
        window_duration: float,
    ) -> torch.Tensor:
        """Softmax probabilities (B, NUM_TEMPORAL_TOKENS, num_classes) for a batch of windows."""
        models = self.models
        video_features, temporal_padding = models.video_encoder(batch.pixel_values, batch.frame_valid)
        audio = [
            get_audio_window(waveform, timeline.time(start), window_duration)
            for start in batch.starts
        ]
        segments = torch.cat([segment for segment, _ in audio])
        padding_mask = torch.cat([mask for _, mask in audio])
        audio_features, audio_padding = models.audio_encoder(segments, padding_mask)
        logits = models.fusion_model(
            video_features, audio_features, audio_padding, temporal_padding
        )
        return torch.softmax(logits, dim=-1).float().cpu()

    def _frame_chunks(
        self,
        video_path: Path,
        info: VideoInfo,
        timeline: FrameTimeline,
        pool: ProcessFramePool | None,
    ) -> Iterator[torch.Tensor]:
        """Preprocessed frames in order: the slot's worker processes if available, else threads."""
        models = self.models
        if pool is not None and models.frame_preprocessor is None:
            if pool.fits(info.height, info.width):
                yield from pool.chunks(video_path, timeline)
                return
            logger.warning(
                "Not enough shared memory for preprocessing workers; using threads "
                "(raise /dev/shm, e.g. docker run --shm-size=1g)",
                extra={"required_bytes": pool.ring_bytes(info.height, info.width)},
            )
        preprocess = models.frame_preprocessor or ProcessorFrames(models.processor)
        with closing(iter_frames(video_path)) as frames:
            yield from preprocessed_chunks(frames, preprocess, timeline)

    def predict(self, video_path: str | Path) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        models = self.models
        waited_at = time.perf_counter()
        pool = self._free.get()
        try:
            started_at = time.perf_counter()
            info = get_video_info(video_path)
            logger.info(
                "Video metadata loaded",
                extra={
                    "fps": info.fps,
                    "frames": info.num_frames,
                    "video_duration_seconds": info.duration,
                    "slot_wait_seconds": round(started_at - waited_at, 4),
                },
            )

            audio_started_at = time.perf_counter()
            waveform = load_audio(video_path)
            audio_seconds = time.perf_counter() - audio_started_at
            starts = build_windows(info.num_frames, NUM_VIDEO_FRAMES, self.params.overlap)
            window_duration = NUM_VIDEO_FRAMES / info.fps
            logger.info(
                "Running temporal windows",
                extra={
                    "window_count": len(starts),
                    "batch_size": models.batch_size,
                    "device": str(models.device),
                },
            )
            timeline = FrameTimeline(info.fps)
            window_predictions = []
            windows_started_at = time.perf_counter()
            # closing() stops decoding/preprocessing even if inference raises midway.
            with closing(self._frame_chunks(video_path, info, timeline, pool)) as chunks:
                batches = iter_window_batches(chunks, starts, NUM_VIDEO_FRAMES, models.batch_size)
                for batch in batches:
                    probs = self.predict_batch(batch, waveform, timeline, window_duration)
                    window_predictions.extend(
                        WindowPrediction(start_frame=start, probs=window_probs)
                        for start, window_probs in zip(batch.starts, probs, strict=True)
                    )

            windows_seconds = time.perf_counter() - windows_started_at
            times, probs = aggregate_predictions(window_predictions, info.fps, timeline.time)
            events = build_temporal_events(times, probs, self.params)
            result = build_json_result(video_path, times, probs, events)
            logger.info(
                "Video inference completed",
                extra={
                    "hit_count": result["total_hits"],
                    "miss_count": result["total_misses"],
                    "audio_decode_seconds": round(audio_seconds, 4),
                    "video_and_model_seconds": round(windows_seconds, 4),
                    "duration_seconds": round(time.perf_counter() - started_at, 4),
                },
            )
        finally:
            self._free.put(pool)
        return result
