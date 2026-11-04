"""Video metadata, audio decoding and sliding windows, matching training.

Frame decoding, sampling and preprocessing live in ``frames.py``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..core.config import AUDIO_SR, NUM_VIDEO_FRAMES, TARGET_FPS, WINDOW_STRIDE


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    num_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps


def get_video_info(video_path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS: {fps}")
    return VideoInfo(fps=fps, num_frames=num_frames, width=width, height=height)


def num_sampled_frames(info: VideoInfo) -> int:
    """Frames sampled at TARGET_FPS over the video's duration (at least one)."""
    return max(1, int(np.floor(info.duration * TARGET_FPS)))


def window_starts(num_sampled: int) -> list[int]:
    """Window starts in sampled frames (training's ``window_starts``).

    Starts are even so the tokens of all windows share one grid; the last window
    ends at the last sampled frame (rounded down to an even start).
    """
    if num_sampled <= NUM_VIDEO_FRAMES:
        return [0]
    last = (num_sampled - NUM_VIDEO_FRAMES) // 2 * 2
    starts = list(range(0, last + 1, WINDOW_STRIDE))
    if starts[-1] != last:
        starts.append(last)
    return starts


def load_audio(video_path: str | Path) -> torch.Tensor:
    """Mono float32 waveform at AUDIO_SR, shape (samples,); empty when there is no audio.

    Same command as training (ffmpeg downmixes and resamples). A video without an
    audio stream is scored from the video alone, as training does.
    """
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1",
        "-ar", str(AUDIO_SR), "-f", "f32le", "-",
    ]
    try:
        raw = subprocess.run(command, capture_output=True, check=True, timeout=600).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required to decode audio from video") from exc
    except (subprocess.SubprocessError, OSError):
        return torch.zeros(0, dtype=torch.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
