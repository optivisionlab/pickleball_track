from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL2ID = {"no_event": 0, "hit": 1, "miss": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
EVENT_LABELS = {"hit", "miss"}

NUM_VIDEO_FRAMES = 32
NUM_TEMPORAL_TOKENS = 16
FRAMES_PER_TEMPORAL_TOKEN = 2

# IMPORTANT:
# One event occupies TOTAL 1.0 second:
# [event_time - 0.5s, event_time + 0.5s].
# Do NOT change this to 1.0 unless you intentionally want a 2-second window.
EVENT_WINDOW_SEC = 1.0
EVENT_RADIUS_SEC = EVENT_WINDOW_SEC / 2.0

DEFAULT_OVERLAP = 0.5
DEFAULT_IMAGE_SIZE = 224
DEFAULT_AUDIO_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Event:
    time: float
    label: int


@dataclass
class VideoInfo:
    video_path: str
    json_path: str
    fps: float
    num_frames: int
    duration: float
    events: List[Event]
    event_times: List[float]
    user: str


@dataclass(frozen=True)
class Window:
    video_index: int
    start_frame: int
    end_frame: int
    valid_start_frame: int
    valid_end_frame: int
    start_time: float
    end_time: float


def extract_user(video_path: str) -> str:
    """
    Expected:
        DD_MM_YYYY_USER_...
    Example:
        20_12_2025_Lan_Truot_TraiTay.MOV -> Lan
    """
    parts = Path(video_path).stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot extract user from video name: {video_path}. "
            "Expected DD_MM_YYYY_USER_..."
        )
    user = parts[3].strip()
    if not user:
        raise ValueError(f"Empty user in video name: {video_path}")
    return user


def load_events(json_path: str) -> List[Event]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events: List[Event] = []
    for item in data.get("events", []):
        try:
            t = float(item["time"])
            label = str(item["label"]).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue

        if label not in EVENT_LABELS or t < 0:
            continue

        events.append(Event(t, LABEL2ID[label]))

    events.sort(key=lambda x: x.time)
    return events


def get_video_metadata(video_path: str) -> Tuple[float, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or n <= 0:
        raise RuntimeError(
            f"Invalid video metadata for {video_path}: fps={fps}, frames={n}"
        )
    return fps, n, n / fps


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self, index: int) -> Optional[Image.Image]:
        if index < 0 or index >= self.num_frames:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class AudioReader:
    """Load audio from the video container when available."""

    def __init__(self, media_path: str, target_sr: int):
        self.waveform = None
        self.sample_rate = target_sr

        try:
            waveform, sr = torchaudio.load(media_path)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != target_sr:
                waveform = torchaudio.functional.resample(
                    waveform, sr, target_sr
                )

            self.waveform = waveform.contiguous().float()
        except Exception:
            self.waveform = None

    @property
    def num_samples(self) -> int:
        return 0 if self.waveform is None else int(self.waveform.shape[-1])

    def get_samples(self, start_sample: int, num_samples: int) -> torch.Tensor:
        if self.waveform is None:
            return torch.empty(1, 0, dtype=torch.float32)

        start = max(0, int(start_sample))
        end = min(self.num_samples, start + int(num_samples))

        if end <= start:
            return torch.empty(1, 0, dtype=torch.float32)

        return self.waveform[:, start:end]

    def close(self):
        self.waveform = None


class PickleballAVDataset(Dataset):
    """
    Dataset contract.

    video:
        (3, 32, 224, 224)

    frame_mask:
        (32,), True = PADDING, False = VALID

    audio:
        (1, fixed_samples)

    audio_mask:
        (fixed_samples,), True = PADDING, False = VALID

    labels:
        (16,), 0=no_event, 1=hit, 2=miss

    temporal_mask:
        (16,), True = PADDING, False = VALID

    Event labeling:
        Each event is active for exactly TOTAL 1 second:
        event_time +/- 0.5 second.
    """

    def __init__(
        self,
        video_paths: Sequence[str],
        json_paths: Sequence[str],
        num_video_frames: int = NUM_VIDEO_FRAMES,
        num_temporal_tokens: int = NUM_TEMPORAL_TOKENS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        overlap: float = DEFAULT_OVERLAP,
        event_radius_sec: float = EVENT_RADIUS_SEC,
        transform=None,
    ):
        super().__init__()

        if len(video_paths) != len(json_paths):
            raise ValueError("video_paths and json_paths must have same length")

        if num_video_frames != 32 or num_temporal_tokens != 16:
            raise ValueError(
                "This architecture requires 32 frames and 16 temporal tokens"
            )

        if num_video_frames % num_temporal_tokens != 0:
            raise ValueError(
                "num_video_frames must be divisible by num_temporal_tokens"
            )

        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1)")

        # Enforce the requested semantics. A radius of 1.0 would mean 2 seconds.
        if abs(float(event_radius_sec) - EVENT_RADIUS_SEC) > 1e-9:
            raise ValueError(
                f"event_radius_sec must be exactly {EVENT_RADIUS_SEC:.1f}s "
                f"so each event spans exactly {EVENT_WINDOW_SEC:.1f}s total "
                f"({EVENT_RADIUS_SEC:.1f}s before + {EVENT_RADIUS_SEC:.1f}s after)."
            )

        self.num_video_frames = num_video_frames
        self.num_temporal_tokens = num_temporal_tokens
        self.frames_per_token = num_video_frames // num_temporal_tokens
        self.image_size = image_size
        self.audio_sample_rate = audio_sample_rate
        self.overlap = overlap
        self.event_radius_sec = EVENT_RADIUS_SEC

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.videos: List[VideoInfo] = []

        for video_path, json_path in zip(video_paths, json_paths):
            video_path = str(Path(video_path))
            json_path = str(Path(json_path))

            if not Path(video_path).exists():
                raise FileNotFoundError(video_path)
            if not Path(json_path).exists():
                raise FileNotFoundError(json_path)

            fps, n, duration = get_video_metadata(video_path)

            events = [
                e for e in load_events(json_path)
                if 0 <= e.time <= duration
            ]

            self.videos.append(
                VideoInfo(
                    video_path=video_path,
                    json_path=json_path,
                    fps=fps,
                    num_frames=n,
                    duration=duration,
                    events=events,
                    event_times=[e.time for e in events],
                    user=extract_user(video_path),
                )
            )

        self.video_readers: Dict[int, VideoReader] = {}
        self.audio_readers: Dict[int, AudioReader] = {}

        # One fixed audio length for every sample.
        self.audio_window_samples = max(
            400,
            max(
                int(round(
                    self.num_video_frames
                    / info.fps
                    * self.audio_sample_rate
                ))
                for info in self.videos
            ),
        )

        self.windows: List[Window] = []
        self._build_windows()

    def _build_windows(self):
        size = self.num_video_frames

        for vi, video in enumerate(self.videos):
            stride = max(1, int(round(size * (1.0 - self.overlap))))

            if video.num_frames <= size:
                starts = [0]
            else:
                starts = list(
                    range(0, video.num_frames - size + 1, stride)
                )
                last = video.num_frames - size

                if starts[-1] != last:
                    starts.append(last)

            self.windows.extend(
                Window(
                    video_index=vi,
                    start_frame=start,
                    end_frame=start + size - 1,
                    valid_start_frame=start,
                    valid_end_frame=min(
                        start + size - 1,
                        video.num_frames - 1,
                    ),
                    start_time=start / video.fps,
                    end_time=(start + size) / video.fps,
                )
                for start in starts
            )

    def _get_video_reader(self, vi: int) -> VideoReader:
        reader = self.video_readers.get(vi)
        if reader is None:
            reader = VideoReader(self.videos[vi].video_path)
            self.video_readers[vi] = reader
        return reader

    def _get_audio_reader(self, vi: int) -> AudioReader:
        reader = self.audio_readers.get(vi)
        if reader is None:
            reader = AudioReader(
                self.videos[vi].video_path,
                self.audio_sample_rate,
            )
            self.audio_readers[vi] = reader
        return reader

    def _load_video(self, window: Window):
        reader = self._get_video_reader(window.video_index)

        frames = []
        frame_mask = torch.ones(
            self.num_video_frames,
            dtype=torch.bool,
        )

        for i in range(self.num_video_frames):
            frame_index = window.start_frame + i
            image = reader.read_frame(frame_index)

            if image is None:
                frames.append(
                    torch.zeros(
                        3,
                        self.image_size,
                        self.image_size,
                    )
                )
            else:
                frames.append(self.transform(image))
                frame_mask[i] = False

        video = torch.stack(frames, dim=0).permute(
            1, 0, 2, 3
        ).contiguous()

        return video, frame_mask

    def _load_audio(self, window: Window):
        fixed_samples = self.audio_window_samples
        audio = torch.zeros(
            1,
            fixed_samples,
            dtype=torch.float32,
        )
        audio_mask = torch.ones(
            fixed_samples,
            dtype=torch.bool,
        )

        reader = self._get_audio_reader(window.video_index)
        start_sample = int(
            round(window.start_time * self.audio_sample_rate)
        )
        segment = reader.get_samples(
            start_sample,
            fixed_samples,
        )

        valid = min(fixed_samples, segment.shape[-1])
        if valid > 0:
            audio[:, :valid] = segment[:, :valid]
            audio_mask[:valid] = False

        return audio, audio_mask

    def _create_frame_labels(
        self,
        info: VideoInfo,
        window: Window,
        frame_mask: torch.Tensor,
    ):
        """
        Efficient event lookup using binary search.

        Exactly 1 second total per event:
            [event_time - 0.5, event_time + 0.5]
        """
        labels = torch.full(
            (self.num_video_frames,),
            LABEL2ID["no_event"],
            dtype=torch.long,
        )

        if not info.events:
            return labels

        times = info.event_times
        radius = self.event_radius_sec

        for i in range(self.num_video_frames):
            if frame_mask[i]:
                continue

            t = (window.start_frame + i) / info.fps

            # Only inspect events whose timestamps can fall inside [t-r, t+r].
            pos = bisect_left(times, t - radius)

            if pos >= len(times):
                continue

            if times[pos] <= t + radius:
                labels[i] = info.events[pos].label
                continue

            # No event in the allowed 1-second interval.
            # The previous event can only be closer when pos > 0.
            if pos > 0:
                prev = pos - 1
                if abs(times[prev] - t) <= radius:
                    labels[i] = info.events[prev].label

        return labels

    def _create_temporal_labels(self, frame_labels, frame_mask):
        labels = torch.full(
            (self.num_temporal_tokens,),
            LABEL2ID["no_event"],
            dtype=torch.long,
        )

        temporal_padding_mask = torch.ones(
            self.num_temporal_tokens,
            dtype=torch.bool,
        )

        for token in range(self.num_temporal_tokens):
            start = token * self.frames_per_token
            end = start + self.frames_per_token

            valid = ~frame_mask[start:end]
            if not valid.any():
                continue

            temporal_padding_mask[token] = False
            token_labels = frame_labels[start:end][valid]

            # Preserve original priority if two event types occur in one token.
            if (token_labels == LABEL2ID["hit"]).any():
                labels[token] = LABEL2ID["hit"]
            elif (token_labels == LABEL2ID["miss"]).any():
                labels[token] = LABEL2ID["miss"]

        return labels, temporal_padding_mask

    def __getitem__(self, index: int):
        window = self.windows[index]
        info = self.videos[window.video_index]

        video, frame_mask = self._load_video(window)
        audio, audio_mask = self._load_audio(window)

        frame_labels = self._create_frame_labels(
            info,
            window,
            frame_mask,
        )

        labels, temporal_mask = self._create_temporal_labels(
            frame_labels,
            frame_mask,
        )

        return {
            "video": video,
            "frame_mask": frame_mask,
            "audio": audio,
            "audio_mask": audio_mask,
            "frame_labels": frame_labels,
            "labels": labels,
            "temporal_mask": temporal_mask,
            "user": info.user,
            "video_id": info.video_path,
            "json_id": info.json_path,
            "video_index": window.video_index,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "start_time": window.start_time,
            "end_time": window.end_time,
        }

    def __len__(self):
        return len(self.windows)

    def close(self):
        for reader in self.video_readers.values():
            reader.close()

        for reader in self.audio_readers.values():
            reader.close()

        self.video_readers.clear()
        self.audio_readers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def pickleball_collate_fn(batch: List[dict]) -> dict:
    return {
        "video": torch.stack([x["video"] for x in batch]),
        "frame_mask": torch.stack([x["frame_mask"] for x in batch]),
        "audio": torch.stack([x["audio"] for x in batch]),
        "audio_mask": torch.stack([x["audio_mask"] for x in batch]),
        "frame_labels": torch.stack([x["frame_labels"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "temporal_mask": torch.stack([x["temporal_mask"] for x in batch]),
        "user": [x["user"] for x in batch],
        "video_id": [x["video_id"] for x in batch],
        "json_id": [x["json_id"] for x in batch],
        "video_index": torch.tensor(
            [x["video_index"] for x in batch],
            dtype=torch.long,
        ),
        "start_frame": torch.tensor(
            [x["start_frame"] for x in batch],
            dtype=torch.long,
        ),
        "end_frame": torch.tensor(
            [x["end_frame"] for x in batch],
            dtype=torch.long,
        ),
        "start_time": torch.tensor(
            [x["start_time"] for x in batch],
            dtype=torch.float32,
        ),
        "end_time": torch.tensor(
            [x["end_time"] for x in batch],
            dtype=torch.float32,
        ),
    }