from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL2ID = {"no_event": 0, "hit": 1, "miss": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
EVENT_LABELS = {"hit", "miss"}

NUM_VIDEO_FRAMES = 32
NUM_TEMPORAL_TOKENS = 16
FRAMES_PER_TOKEN = 2

# EXACTLY 1.0 second total:
# [event_time - 0.5s, event_time + 0.5s]
EVENT_WINDOW_SEC = 1.0
EVENT_RADIUS_SEC = 0.5

DEFAULT_IMAGE_SIZE = 224
DEFAULT_OVERLAP = 0.5


@dataclass(frozen=True)
class Event:
    time: float
    label: int


@dataclass
class VideoInfo:
    path: str
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


def extract_user(video_path: str) -> str:
    # Example:
    # 20_12_2025_Lan_Truot_TraiTay.MOV -> Lan
    parts = Path(video_path).stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot extract user from {video_path}; "
            "expected DD_MM_YYYY_USER_..."
        )
    return parts[3]


def load_events(json_path: str) -> List[Event]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = []
    for item in data.get("events", []):
        try:
            t = float(item["time"])
            label = str(item["label"]).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue

        if label in EVENT_LABELS and t >= 0:
            events.append(Event(t, LABEL2ID[label]))

    events.sort(key=lambda x: x.time)
    return events


def get_video_metadata(path: str) -> Tuple[float, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or n <= 0:
        raise RuntimeError(
            f"Invalid metadata for {path}: fps={fps}, frames={n}"
        )

    return fps, n, n / fps


class SequentialVideoReader:
    """
    Sequential reader. Unlike CAP_PROP_POS_FRAMES per frame,
    this avoids a random seek for every frame.
    """

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.current = -1

    def read_window(
        self,
        start_frame: int,
        num_frames: int,
        image_size: int,
        transform,
    ):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.current = start_frame - 1

        frames = []
        valid = torch.zeros(num_frames, dtype=torch.bool)

        for i in range(num_frames):
            ok, frame = self.cap.read()
            frame_index = start_frame + i

            if not ok:
                frames.append(
                    torch.zeros(
                        3, image_size, image_size
                    )
                )
                continue

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
            frames.append(
                transform(Image.fromarray(rgb))
            )
            valid[i] = True
            self.current = frame_index

        return (
            torch.stack(frames)
            .permute(1, 0, 2, 3)
            .contiguous(),
            ~valid,  # True = padding
        )

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class PickleballFeatureDataset(Dataset):
    """
    Dataset for the OFFLINE FEATURE pipeline.

    It does NOT run VideoMAE or BEATs.

    Expected cache per sample:
        video_features: (16, video_dim)
        audio_features: (A, audio_dim)
        audio_padding_mask: (A,)
        labels: (16,)
        temporal_padding_mask: (16,)

    Before feature extraction, this class can also be used to decode
    the exact 32-frame windows and labels.
    """

    def __init__(
        self,
        video_paths: Sequence[str],
        json_paths: Sequence[str],
        num_video_frames: int = NUM_VIDEO_FRAMES,
        num_temporal_tokens: int = NUM_TEMPORAL_TOKENS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        overlap: float = DEFAULT_OVERLAP,
        cache_dir: Optional[str] = None,
        transform=None,
    ):
        if len(video_paths) != len(json_paths):
            raise ValueError(
                "video_paths and json_paths must have same length"
            )

        if num_video_frames != 32 or num_temporal_tokens != 16:
            raise ValueError(
                "This project expects 32 video frames and 16 temporal tokens"
            )

        if num_video_frames // num_temporal_tokens != FRAMES_PER_TOKEN:
            raise ValueError("Expected 2 frames per temporal token")

        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0, 1)")

        self.num_video_frames = num_video_frames
        self.num_temporal_tokens = num_temporal_tokens
        self.image_size = image_size
        self.overlap = overlap

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
                    path=video_path,
                    json_path=json_path,
                    fps=fps,
                    num_frames=n,
                    duration=duration,
                    events=events,
                    event_times=[e.time for e in events],
                    user=extract_user(video_path),
                )
            )

        self.windows: List[Window] = []
        self._build_windows()

        # These readers are intentionally lazy and are not serialized into
        # DataLoader workers.
        self._readers: Dict[int, SequentialVideoReader] = {}

        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else None
        )

    def _build_windows(self):
        size = self.num_video_frames

        for vi, info in enumerate(self.videos):
            stride = max(
                1,
                int(round(size * (1.0 - self.overlap))),
            )

            if info.num_frames <= size:
                starts = [0]
            else:
                last = info.num_frames - size
                starts = list(range(0, last + 1, stride))
                if starts[-1] != last:
                    starts.append(last)

            for start in starts:
                self.windows.append(
                    Window(
                        video_index=vi,
                        start_frame=start,
                        end_frame=start + size - 1,
                    )
                )

    def _labels_for_window(
        self,
        window: Window,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Labels are generated from frame timestamps.

        One event is active ONLY in:
            event_time - 0.5 <= frame_time <= event_time + 0.5

        Thus the total event duration is exactly 1 second.
        """
        info = self.videos[window.video_index]

        frame_labels = torch.full(
            (self.num_video_frames,),
            LABEL2ID["no_event"],
            dtype=torch.long,
        )
        frame_valid = torch.zeros(
            self.num_video_frames,
            dtype=torch.bool,
        )

        times = info.event_times

        for i in range(self.num_video_frames):
            frame_idx = window.start_frame + i
            if frame_idx >= info.num_frames:
                continue

            frame_valid[i] = True
            t = frame_idx / info.fps

            if not times:
                continue

            pos = bisect_left(
                times,
                t - EVENT_RADIUS_SEC,
            )

            if (
                pos < len(times)
                and times[pos] <= t + EVENT_RADIUS_SEC
            ):
                frame_labels[i] = info.events[pos].label
            elif pos > 0:
                prev = pos - 1
                if (
                    abs(times[prev] - t)
                    <= EVENT_RADIUS_SEC
                ):
                    frame_labels[i] = info.events[prev].label

        labels = torch.full(
            (self.num_temporal_tokens,),
            LABEL2ID["no_event"],
            dtype=torch.long,
        )
        temporal_valid = torch.zeros(
            self.num_temporal_tokens,
            dtype=torch.bool,
        )

        for token in range(self.num_temporal_tokens):
            s = token * FRAMES_PER_TOKEN
            e = s + FRAMES_PER_TOKEN

            valid = frame_valid[s:e]
            if not valid.any():
                continue

            temporal_valid[token] = True
            token_labels = frame_labels[s:e][valid]

            if (token_labels == LABEL2ID["hit"]).any():
                labels[token] = LABEL2ID["hit"]
            elif (token_labels == LABEL2ID["miss"]).any():
                labels[token] = LABEL2ID["miss"]

        return labels, ~temporal_valid

    def get_sample_metadata(self, index: int) -> dict:
        window = self.windows[index]
        info = self.videos[window.video_index]
        labels, temporal_padding_mask = self._labels_for_window(window)

        return {
            "video_index": window.video_index,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "start_time": window.start_frame / info.fps,
            "end_time": (
                (window.end_frame + 1) / info.fps
            ),
            "user": info.user,
            "video_id": info.path,
            "json_id": info.json_path,
            "labels": labels,
            "temporal_padding_mask": temporal_padding_mask,
        }

    def _get_reader(self, video_index: int):
        reader = self._readers.get(video_index)
        if reader is None:
            reader = SequentialVideoReader(
                self.videos[video_index].path
            )
            self._readers[video_index] = reader
        return reader

    def decode_video_window(self, index: int):
        window = self.windows[index]
        reader = self._get_reader(window.video_index)

        video, frame_padding_mask = reader.read_window(
            window.start_frame,
            self.num_video_frames,
            self.image_size,
            self.transform,
        )

        meta = self.get_sample_metadata(index)

        return {
            "video": video,
            "frame_padding_mask": frame_padding_mask,
            **meta,
        }

    def cache_path(self, index: int) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is not configured")
        return self.cache_dir / f"{index:08d}.pt"

    def __getitem__(self, index: int):
        if self.cache_dir is None:
            raise RuntimeError(
                "PickleballFeatureDataset requires cache_dir for training"
            )

        path = self.cache_path(index)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing feature cache: {path}"
            )

        item = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        return item

    def __len__(self):
        return len(self.windows)

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def feature_collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for offline VideoMAE + BEATs features.

    Video features:
        [T_video, D_video]
        T_video = 16 -> fixed

    Audio features:
        [T_audio, D_audio]
        T_audio can differ between samples.

    Therefore audio features are padded to the longest audio sequence
    in the current batch, and audio_padding_mask is padded accordingly.
    """

    # ============================================================
    # 1. Video features
    # ============================================================

    video_features = torch.stack(
        [x["video_features"] for x in batch],
        dim=0,
    )

    # Expected:
    # [B, 16, D_video]
    if video_features.ndim != 3:
        raise RuntimeError(
            f"Unexpected video_features shape: "
            f"{tuple(video_features.shape)}"
        )

    # ============================================================
    # 2. Audio features -- PAD to batch maximum
    # ============================================================

    audio_list = [
        x["audio_features"]
        for x in batch
    ]

    audio_mask_list = [
        x["audio_padding_mask"].bool()
        for x in batch
    ]

    max_audio_len = max(
        x.shape[0]
        for x in audio_list
    )

    audio_dim = audio_list[0].shape[-1]

    batch_size = len(batch)

    # [B, max_audio_len, audio_dim]
    audio_features = torch.zeros(
        batch_size,
        max_audio_len,
        audio_dim,
        dtype=audio_list[0].dtype,
    )

    # True = padding
    # [B, max_audio_len]
    audio_padding_mask = torch.ones(
        batch_size,
        max_audio_len,
        dtype=torch.bool,
    )

    for i, (features, mask) in enumerate(
        zip(audio_list, audio_mask_list)
    ):
        audio_len = features.shape[0]

        if mask.shape[0] != audio_len:
            raise RuntimeError(
                "audio_features and audio_padding_mask "
                f"length mismatch at sample {i}: "
                f"features={audio_len}, mask={mask.shape[0]}"
            )

        audio_features[i, :audio_len] = features
        audio_padding_mask[i, :audio_len] = mask

    # ============================================================
    # 3. Labels
    # ============================================================

    labels = torch.stack(
        [x["labels"] for x in batch],
        dim=0,
    )

    temporal_padding_mask = torch.stack(
        [
            x["temporal_padding_mask"]
            for x in batch
        ],
        dim=0,
    )

    # ============================================================
    # 4. Metadata
    # ============================================================

    out = {
        "video_features": video_features,
        "audio_features": audio_features,
        "audio_padding_mask": audio_padding_mask,
        "labels": labels,
        "temporal_padding_mask": temporal_padding_mask,
    }

    # Numeric metadata
    for key in [
        "video_index",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
    ]:
        values = [x[key] for x in batch]

        out[key] = torch.tensor(values)

    # String metadata
    for key in [
        "user",
        "video_id",
        "json_id",
    ]:
        out[key] = [
            x[key]
            for x in batch
        ]

    return out