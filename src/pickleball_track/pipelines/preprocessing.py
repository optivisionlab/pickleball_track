"""Video/audio decoding and sliding-window preparation matching training."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor

from ..core.config import AUDIO_SR, OVERLAP
from .resample import resample


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


def build_windows(num_frames: int, num_window_frames: int, overlap: float = OVERLAP) -> list[int]:
    """Start frames of overlapping windows; the last window ends at the last frame."""
    stride = max(1, int(round(num_window_frames * (1.0 - overlap))))
    if num_frames <= num_window_frames:
        return [0]

    last_start = num_frames - num_window_frames
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def read_video_window(
    video_path: str | Path,
    start_frame: int,
    num_frames: int,
    image_processor: VideoMAEImageProcessor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return VideoMAE pixel values (1, F, C, H, W) and a per-frame validity mask.

    Frames past the end of the video repeat the last frame and are marked invalid.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    valid = []
    for _ in range(num_frames):
        ok, frame = capture.read()
        if not ok:
            # The padding loop below marks the missing frames. (The original script also
            # appended False here, giving 33 flags for 32 frames and crashing the reshape.)
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        valid.append(True)
    capture.release()

    if not frames:
        raise RuntimeError(f"Cannot read frames at {start_frame}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
        valid.append(False)

    inputs = image_processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    if pixel_values.ndim != 5:
        raise RuntimeError(
            f"Unexpected VideoMAE input shape: {tuple(pixel_values.shape)}"
        )
    return pixel_values, torch.tensor(valid, dtype=torch.bool)


def _run(command: list[str]) -> bytes:
    try:
        return subprocess.run(command, check=True, capture_output=True).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required to decode audio from video") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode video audio: {message}") from exc


def decode_audio(video_path: str | Path) -> tuple[torch.Tensor, int]:
    """Decode the first audio stream at its native rate: float32 (channels, samples), sr."""
    probe = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", str(video_path),
    ]).decode().strip()
    if not probe:
        raise RuntimeError("Video contains no decodable audio stream")
    sample_rate, channels = (int(value) for value in probe.split(",")[:2])

    raw = _run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-map", "0:a:0",
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ])
    samples = np.frombuffer(raw, dtype="<f4").copy()
    if samples.size == 0 or samples.size % channels:
        raise RuntimeError("Video contains no decodable audio stream")
    # Interleaved PCM -> channels first, like torchaudio.load.
    return torch.from_numpy(samples).reshape(-1, channels).T.contiguous(), sample_rate


def load_audio(video_path: str | Path) -> torch.Tensor:
    """Mono float32 waveform at AUDIO_SR, shape (1, samples).

    Same steps as the original torchaudio pipeline: decode, average channels,
    then torchaudio's sinc resampling (see ``resample.py``).
    """
    waveform, sr = decode_audio(video_path)
    # Stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.float()
    if sr != AUDIO_SR:
        waveform = resample(waveform, sr, AUDIO_SR)
    return waveform.contiguous()


def get_audio_window(
    waveform: torch.Tensor,
    start_time: float,
    duration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice and zero-pad a window; padding mask is True for padded samples."""
    start_sample = max(0, int(round(start_time * AUDIO_SR)))
    num_samples = max(1, int(round(duration * AUDIO_SR)))
    end_sample = min(waveform.shape[-1], start_sample + num_samples)
    segment = waveform[:, start_sample:end_sample]
    valid_samples = segment.shape[-1]

    if valid_samples < num_samples:
        padded = torch.zeros(1, num_samples, dtype=torch.float32)
        padded[:, :valid_samples] = segment
        segment = padded

    padding_mask = torch.ones(1, num_samples, dtype=torch.bool)
    padding_mask[:, :valid_samples] = False
    return segment, padding_mask
