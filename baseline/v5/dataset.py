from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    CACHE_VERSION,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OVERLAP,
    EVENT_RADIUS_SEC,
    EVENT_WINDOW_SEC,
    FRAMES_PER_TOKEN,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL2ID,
    NUM_TEMPORAL_TOKENS,
    NUM_VIDEO_FRAMES,
)


ID2LABEL = {v: k for k, v in LABEL2ID.items()}
EVENT_LABELS = {"hit", "miss"}

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
    user_id: str
    hand: str
    action: str


@dataclass(frozen=True)
class Window:
    video_index: int
    start_frame: int
    end_frame: int


def extract_user(video_path: str) -> str:
    parts = Path(video_path).stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot extract user from {video_path}; expected DD_MM_YYYY_USER_..."
        )
    return parts[3]


def normalize_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if "trai" in value or "left" in value:
        return "left"
    if "phai" in value or "right" in value:
        return "right"
    return "unknown"


def parse_video_metadata(video_path: str) -> Dict[str, str]:
    parts = Path(video_path).stem.split("_")
    if len(parts) < 6:
        raise ValueError(
            f"Unexpected filename format: {video_path}; "
            "expected DD_MM_YYYY_USER_ACTION_HAND"
        )

    return {
        "user": parts[3],
        "user_id": parts[3],
        "action": parts[4].strip().lower(),
        "hand": normalize_hand(parts[5]),
    }


def build_metadata_text(user_id: str, hand: str, action: str) -> str:
    return f"user_id={user_id}; hand={hand}; action={action}"


def load_events(json_path: str) -> List[Event]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events: List[Event] = []
    raw_events = data.get("events", data) if isinstance(data, dict) else data

    if not isinstance(raw_events, list):
        return events

    for item in raw_events:
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
        raise RuntimeError(f"Invalid metadata for {path}: fps={fps}, frames={n}")

    return fps, n, n / fps


def _json_signature(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_signature(path: str) -> str:
    p = Path(path)
    stat = p.stat()
    payload = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()


class SequentialVideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

    def read_window(self, start_frame: int, num_frames: int, transform):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: List[torch.Tensor] = []
        valid = torch.zeros(num_frames, dtype=torch.bool)
        last_valid: Optional[torch.Tensor] = None

        for i in range(num_frames):
            ok, frame = self.cap.read()

            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = transform(Image.fromarray(rgb))
                frames.append(tensor)
                last_valid = tensor
                valid[i] = True
            else:
                # Repeat the last valid frame. This avoids feeding an
                # out-of-distribution all-zero/black image to VideoMAE.
                if last_valid is None:
                    raise RuntimeError(
                        f"Could not decode any frame from {self.cap}"
                    )
                frames.append(last_valid.clone())

        video = torch.stack(frames).permute(1, 0, 2, 3).contiguous()
        return video, ~valid

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class PickleballFeatureDataset(Dataset):
    """
    Offline Video+Text feature dataset.

    Cached sample contract:
        video_features: [16, Dv]
        text_features:  [Dt] or [1, Dt]
        labels:         [16]
        temporal_padding_mask: [16], True=padding

    No audio/BEATs fields exist.
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
        super().__init__()

        if len(video_paths) != len(json_paths):
            raise ValueError("video_paths and json_paths must have same length")
        if num_video_frames != 32 or num_temporal_tokens != 16:
            raise ValueError("This architecture requires 32 frames and 16 temporal tokens")
        if num_video_frames // num_temporal_tokens != FRAMES_PER_TOKEN:
            raise ValueError("Expected 2 frames per temporal token")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0,1)")

        self.num_video_frames = num_video_frames
        self.num_temporal_tokens = num_temporal_tokens
        self.frames_per_token = FRAMES_PER_TOKEN
        self.image_size = image_size
        self.overlap = overlap
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.videos: List[VideoInfo] = []

        for video_path, json_path in zip(video_paths, json_paths):
            video_path = str(Path(video_path).resolve())
            json_path = str(Path(json_path).resolve())

            if not Path(video_path).exists():
                raise FileNotFoundError(video_path)
            if not Path(json_path).exists():
                raise FileNotFoundError(json_path)

            fps, n, duration = get_video_metadata(video_path)
            events = [e for e in load_events(json_path) if 0 <= e.time <= duration]
            meta = parse_video_metadata(video_path)

            self.videos.append(
                VideoInfo(
                    path=video_path,
                    json_path=json_path,
                    fps=fps,
                    num_frames=n,
                    duration=duration,
                    events=events,
                    event_times=[e.time for e in events],
                    user=meta["user"],
                    user_id=meta["user_id"],
                    hand=meta["hand"],
                    action=meta["action"],
                )
            )

        self.windows: List[Window] = []
        self._build_windows()
        self._readers: Dict[int, SequentialVideoReader] = {}

    def _build_windows(self):
        for video_index, info in enumerate(self.videos):
            stride = max(
                1,
                int(round(self.num_video_frames * (1.0 - self.overlap))),
            )
            start = 0

            while start < info.num_frames:
                end = start + self.num_video_frames
                self.windows.append(
                    Window(video_index, start, end)
                )
                if end >= info.num_frames:
                    break
                start += stride

    def __len__(self):
        return len(self.windows)

    def _get_reader(self, video_index: int) -> SequentialVideoReader:
        if video_index not in self._readers:
            self._readers[video_index] = SequentialVideoReader(
                self.videos[video_index].path
            )
        return self._readers[video_index]

    def _labels_for_window(
        self,
        info: VideoInfo,
        start_frame: int,
        valid_frames: int,
    ) -> torch.Tensor:
        labels = torch.zeros(self.num_temporal_tokens, dtype=torch.long)

        if valid_frames <= 0:
            return labels

        # Each VideoMAE temporal token corresponds to 2 source frames.
        for token in range(self.num_temporal_tokens):
            token_start = start_frame + token * self.frames_per_token
            token_end = min(
                start_frame + (token + 1) * self.frames_per_token,
                start_frame + valid_frames,
            )

            if token_start >= token_end:
                continue

            t0 = token_start / info.fps
            t1 = (token_end - 1) / info.fps

            token_label = LABEL2ID["no_event"]
            for event in info.events:
                if (
                    event.time < t0 - EVENT_RADIUS_SEC
                    or event.time > t1 + EVENT_RADIUS_SEC
                ):
                    continue

                if event.label == LABEL2ID["hit"]:
                    token_label = LABEL2ID["hit"]
                    break

                if (
                    event.label == LABEL2ID["miss"]
                    and token_label == LABEL2ID["no_event"]
                ):
                    token_label = LABEL2ID["miss"]

            labels[token] = token_label

        return labels

    def decode_video_window(self, index: int) -> dict:
        window = self.windows[index]
        info = self.videos[window.video_index]

        reader = self._get_reader(window.video_index)
        video, frame_padding_mask = reader.read_window(
            window.start_frame,
            self.num_video_frames,
            self.transform,
        )

        valid_frames = int((~frame_padding_mask).sum().item())
        labels = self._labels_for_window(
            info,
            window.start_frame,
            valid_frames,
        )

        temporal_padding_mask = frame_padding_mask.view(
            self.num_temporal_tokens,
            self.frames_per_token,
        ).all(dim=1)

        return {
            "video": video,
            "frame_padding_mask": frame_padding_mask,
            "labels": labels,
            "temporal_padding_mask": temporal_padding_mask,
            "video_index": window.video_index,
            "start_frame": window.start_frame,
            "end_frame": min(window.end_frame, info.num_frames),
            "start_time": window.start_frame / info.fps,
            "end_time": min(window.end_frame, info.num_frames) / info.fps,
            "user": info.user,
            "user_id": info.user_id,
            "hand": info.hand,
            "action": info.action,
            "text": build_metadata_text(info.user_id, info.hand, info.action),
            "video_id": _file_signature(info.path),
            "json_id": _json_signature(info.json_path),
        }

    def get_sample_metadata(self, index: int) -> dict:
        window = self.windows[index]
        info = self.videos[window.video_index]

        valid_frames = max(
            0,
            min(self.num_video_frames, info.num_frames - window.start_frame),
        )

        frame_mask = torch.ones(self.num_video_frames, dtype=torch.bool)
        frame_mask[:valid_frames] = False

        temporal_mask = frame_mask.view(
            self.num_temporal_tokens,
            self.frames_per_token,
        ).all(dim=1)

        labels = self._labels_for_window(
            info,
            window.start_frame,
            valid_frames,
        )

        return {
            "video_index": window.video_index,
            "start_frame": window.start_frame,
            "end_frame": min(window.end_frame, info.num_frames),
            "start_time": window.start_frame / info.fps,
            "end_time": min(window.end_frame, info.num_frames) / info.fps,
            "labels": labels,
            "temporal_padding_mask": temporal_mask,
            "user": info.user,
            "user_id": info.user_id,
            "hand": info.hand,
            "action": info.action,
            "text": build_metadata_text(info.user_id, info.hand, info.action),
            "video_id": _file_signature(info.path),
            "json_id": _json_signature(info.json_path),
        }

    def cache_signature(self, index: int, videomae_name: str, clip_name: str) -> str:
        meta = self.get_sample_metadata(index)
        payload = {
            "cache_version": CACHE_VERSION,
            "video_id": meta["video_id"],
            "json_id": meta["json_id"],
            "video_index": meta["video_index"],
            "start_frame": meta["start_frame"],
            "end_frame": meta["end_frame"],
            "num_video_frames": self.num_video_frames,
            "num_temporal_tokens": self.num_temporal_tokens,
            "image_size": self.image_size,
            "overlap": self.overlap,
            "event_window_sec": EVENT_WINDOW_SEC,
            "event_radius_sec": EVENT_RADIUS_SEC,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "videomae": videomae_name,
            "clip": clip_name,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def __getitem__(self, index: int) -> dict:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is required for feature training")

        path = self.cache_dir / f"{index:08d}.pt"
        item = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "video_features": item["video_features"].float(),
            "text_features": item["text_features"].float(),
            "labels": item["labels"].long(),
            "temporal_padding_mask": item["temporal_padding_mask"].bool(),
            "user": item.get("user", ""),
        }

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
    video = torch.stack([x["video_features"] for x in batch])
    text = torch.stack([
        x["text_features"].squeeze(0) if x["text_features"].ndim == 2 else x["text_features"]
        for x in batch
    ])
    labels = torch.stack([x["labels"] for x in batch])
    mask = torch.stack([x["temporal_padding_mask"] for x in batch])

    return {
        "video_features": video,
        "text_features": text,
        "labels": labels,
        "temporal_padding_mask": mask,
        "user": [x["user"] for x in batch],
    }
