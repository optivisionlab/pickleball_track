"""Video and audio decoding and feature preparation for the trained model."""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

from ..core.config import AUDIO_SR, NUM_VIDEO_FRAMES


def get_video_info(video_path: str | Path) -> dict[str, float | int]:
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
    return {
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "duration": num_frames / fps,
    }


def read_video_window(
    video_path: str | Path,
    start_frame: int,
    num_frames: int,
    image_processor: VideoMAEImageProcessor,
) -> tuple[torch.Tensor, torch.Tensor]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    valid = []
    for _ in range(num_frames):
        ok, frame = capture.read()
        if not ok:
            valid.append(False)
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


def load_audio(video_path: str | Path) -> torch.Tensor:
    try:
        decoded = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path),
                "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar",
                str(AUDIO_SR), "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required to decode audio from video") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg could not decode video audio: {message}") from exc

    samples = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise RuntimeError("Video contains no decodable audio stream")
    return torch.from_numpy(samples).unsqueeze(0).contiguous()


def get_audio_window(
    waveform: torch.Tensor,
    start_time: float,
    duration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


@torch.inference_mode()
def extract_video_feature(
    videomae: VideoMAEModel,
    processor: VideoMAEImageProcessor,
    video_path: str | Path,
    start_frame: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values, frame_valid = read_video_window(
        video_path, start_frame, NUM_VIDEO_FRAMES, processor
    )
    output = videomae(pixel_values=pixel_values.to(device, non_blocking=True))
    tokens = output.last_hidden_state
    temporal_tokens = NUM_VIDEO_FRAMES // int(videomae.config.tubelet_size)
    spatial_tokens = (
        int(videomae.config.image_size) // int(videomae.config.patch_size)
    ) ** 2
    hidden_dim = tokens.shape[-1]
    expected_tokens = temporal_tokens * spatial_tokens
    if tokens.shape[1] != expected_tokens:
        raise RuntimeError(
            f"VideoMAE token mismatch: got {tokens.shape[1]}, expected {expected_tokens}"
        )

    features = tokens.reshape(1, temporal_tokens, spatial_tokens, hidden_dim).mean(dim=2)
    frames_per_token = NUM_VIDEO_FRAMES // temporal_tokens
    temporal_padding = ~frame_valid.reshape(
        temporal_tokens, frames_per_token
    ).any(dim=1)
    return features.float(), temporal_padding.unsqueeze(0).to(device)


@torch.inference_mode()
def extract_audio_feature(
    beats,
    waveform: torch.Tensor,
    start_time: float,
    duration: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    segment, padding_mask = get_audio_window(waveform, start_time, duration)
    features, feature_padding = beats.extract_features(
        segment.to(device, non_blocking=True),
        padding_mask=padding_mask.to(device, non_blocking=True),
    )
    return features.float(), feature_padding.bool()