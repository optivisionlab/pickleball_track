from __future__ import annotations

import csv
import hashlib
import json
import re
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
BENCHMARK_TEXT_BY_LABEL = {
    "hit": "A basketball shot goes into the hoop.",
    "miss": "A basketball shot misses the hoop.",
    "no_event": "A basketball game with no shot attempt.",
}

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
    user_id: str
    hand: str
    action: str


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


def normalize_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if "trai" in value or "left" in value:
        return "left"
    if "phai" in value or "right" in value:
        return "right"
    return value


def parse_video_metadata(video_path: str) -> Dict[str, str]:
    parts = Path(video_path).stem.split("_")
    if len(parts) < 6:
        raise ValueError(
            f"Unexpected filename format for metadata: {video_path}; "
            "expected DD_MM_YYYY_USER_ACTION_HAND"
        )

    user_id = parts[3]
    action = parts[4].strip().lower()
    hand = normalize_hand(parts[5])

    return {
        "user": user_id,
        "user_id": user_id,
        "action": action,
        "hand": hand,
    }


def build_metadata_text(user_id: str, hand: str, action: str) -> str:
    return (
        f"user_id={user_id}; "
        f"hand={hand}; "
        f"action={action}"
    )


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

            metadata = parse_video_metadata(video_path)
            user = metadata["user"]
            self.videos.append(
                VideoInfo(
                    path=video_path,
                    json_path=json_path,
                    fps=fps,
                    num_frames=n,
                    duration=duration,
                    events=events,
                    event_times=[e.time for e in events],
                    user=user,
                    user_id=metadata["user_id"],
                    hand=metadata["hand"],
                    action=metadata["action"],
                )
            )

        self.user_to_index = {
            user: idx
            for idx, user in enumerate(
                sorted({info.user_id for info in self.videos})
            )
        }
        self.hand_to_index = {
            "left": 0,
            "right": 1,
            "unknown": 2,
        }
        self.action_to_index = {
            action: idx
            for idx, action in enumerate(
                sorted({info.action for info in self.videos})
            )
        }
        self.text_dim = (
            len(self.user_to_index)
            + len(self.hand_to_index)
            + len(self.action_to_index)
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

    def _metadata_vector(self, user_id: str, hand: str, action: str) -> torch.Tensor:
        vector = torch.zeros(
            self.text_dim,
            dtype=torch.float32,
        )

        user_offset = 0
        hand_offset = len(self.user_to_index)
        action_offset = hand_offset + len(self.hand_to_index)

        user_key = str(user_id)
        if user_key in self.user_to_index:
            vector[self.user_to_index[user_key]] = 1.0

        hand_key = normalize_hand(hand)
        if hand_key in self.hand_to_index:
            vector[hand_offset + self.hand_to_index[hand_key]] = 1.0
        else:
            vector[hand_offset + self.hand_to_index["unknown"]] = 1.0

        action_key = str(action).strip().lower()
        if action_key in self.action_to_index:
            vector[action_offset + self.action_to_index[action_key]] = 1.0

        return vector

    def get_sample_metadata(self, index: int) -> dict:
        window = self.windows[index]
        info = self.videos[window.video_index]
        labels, temporal_padding_mask = self._labels_for_window(window)

        text = build_metadata_text(
            info.user_id,
            info.hand,
            info.action,
        )

        return {
            "video_index": window.video_index,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "start_time": window.start_frame / info.fps,
            "end_time": (
                (window.end_frame + 1) / info.fps
            ),
            "user": info.user,
            "user_id": info.user_id,
            "hand": info.hand,
            "action": info.action,
            "text": text,
            "video_id": info.path,
            "json_id": info.json_path,
            "labels": labels,
            "temporal_padding_mask": temporal_padding_mask,
            "text_features": self._metadata_vector(
                info.user_id,
                info.hand,
                info.action,
            ),
            "text_padding_mask": torch.zeros(
                1,
                dtype=torch.bool,
            ),
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

        if "user_id" not in item or "hand" not in item:
            meta = self.get_sample_metadata(int(index))
            item["user_id"] = meta["user_id"]
            item["hand"] = meta["hand"]
            item["action"] = meta["action"]
            item["text"] = meta["text"]
            item["text_features"] = meta["text_features"]
            item["text_padding_mask"] = meta["text_padding_mask"]

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
    # 4. Metadata text input
    # ============================================================

    text_features = torch.stack(
        [x["text_features"] for x in batch],
        dim=0,
    )

    if text_features.ndim == 1:
        text_features = text_features.unsqueeze(1)
    if text_features.ndim == 2:
        text_features = text_features.unsqueeze(1)

    out = {
        "video_features": video_features,
        "audio_features": audio_features,
        "audio_padding_mask": audio_padding_mask,
        "labels": labels,
        "temporal_padding_mask": temporal_padding_mask,
        "text_features": text_features,
        "text_padding_mask": torch.zeros(
            (len(batch), 1),
            dtype=torch.bool,
        ),
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
        "user_id",
        "hand",
        "action",
        "text",
        "video_id",
        "json_id",
    ]:
        out[key] = [
            x.get(key)
            for x in batch
        ]

    return out


@dataclass(frozen=True)
class BenchmarkClip:
    path: Path
    video_id: str
    numeric_id: int
    label: int
    fps: float
    num_frames: int


@dataclass(frozen=True)
class BenchmarkWindow:
    clip_index: int
    start_frame: int


def benchmark_video_number(value: str) -> int:
    match = re.fullmatch(r"video(\d+)", value.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid video_id {value!r}; expected video123.")
    return int(match.group(1))


def read_benchmark_labels(path: str | Path) -> Dict[int, Tuple[str, int]]:
    labels: Dict[int, Tuple[str, int]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV requires columns {sorted(required)}.")
        for row_number, row in enumerate(reader, start=2):
            video_id = (row.get("video_id") or "").strip()
            label_name = (row.get("label") or "").strip().lower()
            if not video_id or not label_name:
                continue
            if label_name not in LABEL2ID:
                raise ValueError(
                    f"Unknown label {label_name!r} at CSV row {row_number}; "
                    f"expected {sorted(LABEL2ID)}."
                )
            number = benchmark_video_number(video_id)
            label = LABEL2ID[label_name]
            if number in labels and labels[number][1] != label:
                raise ValueError(f"Conflicting labels for {video_id}.")
            labels[number] = (video_id, label)
    if not labels:
        raise RuntimeError(f"No labeled video rows found in {path}.")
    return labels


class BenchmarkDataset(Dataset):
    """Labeled clip windows and cached AV+text features for benchmark training."""

    def __init__(
        self,
        video_dir: str | Path,
        labels_csv: str | Path,
        cache_dir: str | Path,
        videomae_name: str,
        clip_name: str,
        beats_checkpoint: str,
    ):
        video_dir = Path(video_dir)
        labels_csv = Path(labels_csv)
        self.cache_dir = Path(cache_dir)
        if not video_dir.is_dir():
            raise FileNotFoundError(video_dir)
        if not labels_csv.is_file():
            raise FileNotFoundError(labels_csv)

        label_map = read_benchmark_labels(labels_csv)
        self.clips: List[BenchmarkClip] = []
        matched_ids = set()
        extensions = {".mp4", ".mov", ".mkv", ".avi"}
        for path in sorted(video_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            match = re.search(r"-(\d+)$", path.stem)
            if match is None or int(match.group(1)) not in label_map:
                continue
            number = int(match.group(1))
            video_id, label = label_map[number]
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps <= 0 or frame_count <= 0:
                raise RuntimeError(f"Invalid video metadata: {path}")
            self.clips.append(
                BenchmarkClip(path, video_id, number, label, fps, frame_count)
            )
            matched_ids.add(number)
        if not self.clips:
            raise RuntimeError(f"No labeled videos matched under {video_dir}.")

        stride = max(1, round(NUM_VIDEO_FRAMES * (1.0 - DEFAULT_OVERLAP)))
        self.windows: List[BenchmarkWindow] = []
        for clip_index, clip in enumerate(self.clips):
            starts = [0] if clip.num_frames <= NUM_VIDEO_FRAMES else list(
                range(0, clip.num_frames - NUM_VIDEO_FRAMES + 1, stride)
            )
            last = max(0, clip.num_frames - NUM_VIDEO_FRAMES)
            if starts[-1] != last:
                starts.append(last)
            self.windows.extend(
                BenchmarkWindow(clip_index, start) for start in starts
            )

        missing = sorted(set(label_map) - matched_ids)
        if missing:
            print(f"[WARNING] {len(missing)} labeled IDs have no matching video.")
        self.cache_identity = (
            f"v4-benchmark-2|{videomae_name}|{clip_name}|{beats_checkpoint}"
        )
        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Matched {len(self.clips)} video files / {len(matched_ids)} IDs; "
            f"windows={len(self.windows)}."
        )

    def __len__(self) -> int:
        return len(self.windows)

    def metadata(self, index: int) -> dict:
        window = self.windows[index]
        clip = self.clips[window.clip_index]
        valid_frames = min(
            NUM_VIDEO_FRAMES,
            clip.num_frames - window.start_frame,
        )
        frames_per_token = NUM_VIDEO_FRAMES // NUM_TEMPORAL_TOKENS
        valid_tokens = (valid_frames + frames_per_token - 1) // frames_per_token
        temporal_mask = torch.ones(NUM_TEMPORAL_TOKENS, dtype=torch.bool)
        temporal_mask[:valid_tokens] = False
        labels = torch.full((NUM_TEMPORAL_TOKENS,), clip.label, dtype=torch.long)
        labels[temporal_mask] = LABEL2ID["no_event"]
        text = BENCHMARK_TEXT_BY_LABEL[ID2LABEL[clip.label]]
        stat = clip.path.stat()
        payload = (
            f"{self.cache_identity}|{clip.path.resolve()}|{stat.st_size}|"
            f"{stat.st_mtime_ns}|{window.start_frame}|{clip.label}|"
            f"{NUM_VIDEO_FRAMES}|{DEFAULT_IMAGE_SIZE}|{DEFAULT_OVERLAP}"
        )
        return {
            "cache_key": hashlib.sha256(payload.encode()).hexdigest(),
            "video_index": window.clip_index,
            "start_frame": window.start_frame,
            "end_frame": min(window.start_frame + NUM_VIDEO_FRAMES, clip.num_frames),
            "labels": labels,
            "temporal_padding_mask": temporal_mask,
            "video_id": str(clip.path),
            "user": clip.video_id,
            "text": text,
        }

    def cache_path(self, index: int) -> Path:
        return self.cache_dir / f"{index:08d}.pt"

    def cache_for_update(self, index: int) -> dict:
        meta = self.metadata(index)
        path = self.cache_path(index)
        if path.exists():
            try:
                item = torch.load(path, map_location="cpu", weights_only=True)
                if item.get("cache_key") == meta["cache_key"]:
                    return item
            except Exception:
                pass
        return {
            **meta,
            "video_features": None,
            "audio_features": None,
            "audio_padding_mask": None,
            "text_features": None,
            "text_padding_mask": torch.zeros(1, dtype=torch.bool),
        }

    def load_cache(self, index: int) -> dict:
        path = self.cache_path(index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature cache: {path}")
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item.get("cache_key") != self.metadata(index)["cache_key"]:
            raise RuntimeError(f"Stale cache at {path}; rerun without --skip-precompute.")
        return item

    def __getitem__(self, index: int) -> dict:
        return self.load_cache(index)

    def decode(self, index: int) -> torch.Tensor:
        window = self.windows[index]
        clip = self.clips[window.clip_index]
        cap = cv2.VideoCapture(str(clip.path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {clip.path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)
        frames = []
        for _ in range(NUM_VIDEO_FRAMES):
            ok, frame = cap.read()
            if not ok:
                frames.append(torch.zeros(3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(Image.fromarray(rgb)))
        cap.release()
        return torch.stack(frames).permute(1, 0, 2, 3).contiguous()