"""Run the trained audiovisual model over a whole video with sliding windows."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from ..core.config import NUM_VIDEO_FRAMES, InferenceParams
from ..models.model_loader import ModelBundle
from ..pipelines.postprocessing import (
    WindowPrediction,
    aggregate_predictions,
    build_json_result,
    build_temporal_events,
)
from ..pipelines.preprocessing import (
    build_windows,
    get_audio_window,
    get_video_info,
    load_audio,
    read_video_window,
)

logger = logging.getLogger(__name__)


class VideoInferenceService:
    """Holds loaded models; calls are serialized so the device is never shared concurrently."""

    def __init__(self, models: ModelBundle, params: InferenceParams = InferenceParams()):
        self.models = models
        self.params = params
        self._lock = Lock()

    @torch.inference_mode()
    def predict_window(
        self,
        video_path: Path,
        waveform: torch.Tensor,
        start_frame: int,
        fps: float,
        window_duration: float,
    ) -> torch.Tensor:
        """Softmax probabilities (NUM_TEMPORAL_TOKENS, num_classes) for one window."""
        models = self.models
        pixel_values, frame_valid = read_video_window(
            video_path, start_frame, NUM_VIDEO_FRAMES, models.processor
        )
        video_features, temporal_padding = models.video_encoder(pixel_values, frame_valid)
        segment, padding_mask = get_audio_window(waveform, start_frame / fps, window_duration)
        audio_features, audio_padding = models.audio_encoder(segment, padding_mask)
        logits = models.fusion_model(
            video_features, audio_features, audio_padding, temporal_padding
        )
        return torch.softmax(logits, dim=-1).squeeze(0).float().cpu()

    def predict(self, video_path: str | Path) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        with self._lock:
            started_at = time.perf_counter()
            info = get_video_info(video_path)
            logger.info(
                "Video metadata loaded",
                extra={
                    "fps": info.fps,
                    "frames": info.num_frames,
                    "video_duration_seconds": info.duration,
                },
            )

            waveform = load_audio(video_path)
            starts = build_windows(info.num_frames, NUM_VIDEO_FRAMES, self.params.overlap)
            window_duration = NUM_VIDEO_FRAMES / info.fps
            logger.info(
                "Running temporal windows",
                extra={"window_count": len(starts), "device": str(self.models.device)},
            )
            window_predictions = [
                WindowPrediction(
                    start_frame=start_frame,
                    probs=self.predict_window(
                        video_path, waveform, start_frame, info.fps, window_duration
                    ),
                )
                for start_frame in starts
            ]

            times, probs = aggregate_predictions(window_predictions, info.fps)
            events = build_temporal_events(times, probs, self.params)
            result = build_json_result(video_path, times, probs, events)
            logger.info(
                "Video inference completed",
                extra={
                    "hit_count": result["total_hits"],
                    "miss_count": result["total_misses"],
                    "duration_seconds": round(time.perf_counter() - started_at, 4),
                },
            )
        return result
