from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from config import (
    VERSION_ARTIFACTS,
    AUDIO_SR,
    BATCH_SIZE,
    BEATS_CHECKPOINT,
    BEATS_ROOT,
    DEFAULT_VIDEOMAE,
    EPOCHS,
    FUSION_LR,
    GRAD_CLIP,
    ID2LABEL,
    IMAGE_SIZE,
    LABEL2ID,
    NUM_CLASSES,
    NUM_TEMPORAL_TOKENS,
    NUM_VIDEO_FRAMES,
    NUM_WORKERS,
    OVERLAP,
    PATIENCE,
    SEED,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import feature_collate_fn
from model import AVFeatureModel, TemporalFocalLoss
from train import (
    build_optimizer,
    build_scheduler,
    extract_video_features,
    load_beats,
    load_videomae,
    run_epoch,
    seed_everything,
    seed_worker,
)


DEFAULT_VIDEO_DIR = "/data/Cuong/pickleball/data/bendmark/train"
DEFAULT_LABELS_CSV = (
    "/data/Cuong/pickleball/data/bendmark/playable_nsva_labels.csv"
)


@dataclass(frozen=True)
class ClipInfo:
    path: str
    video_id: str
    label: int
    fps: float
    num_frames: int


@dataclass(frozen=True)
class ClipWindow:
    clip_index: int
    start_frame: int
    end_frame: int


def _video_id_number(video_id: str) -> int:
    match = re.fullmatch(r"video(\d+)", video_id.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Invalid video_id {video_id!r}; expected a value like 'video123'."
        )
    return int(match.group(1))


def _filename_id(path: Path) -> Optional[int]:
    match = re.search(r"-(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _read_video_labels(labels_csv: Path) -> Dict[int, Tuple[str, int]]:
    labels: Dict[int, Tuple[str, int]] = {}
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}."
            )

        for row_number, row in enumerate(reader, start=2):
            video_id = (row.get("video_id") or "").strip()
            label_name = (row.get("label") or "").strip().lower()
            if not video_id or not label_name:
                continue
            if label_name not in LABEL2ID:
                raise ValueError(
                    f"Unknown label {label_name!r} at CSV row {row_number}; "
                    f"expected one of {sorted(LABEL2ID)}."
                )

            numeric_id = _video_id_number(video_id)
            label = LABEL2ID[label_name]
            previous = labels.get(numeric_id)
            if previous is not None and previous[1] != label:
                raise ValueError(
                    f"Conflicting labels for video{numeric_id}: "
                    f"{previous[0]} and {label_name}."
                )
            labels[numeric_id] = (video_id, label)

    if not labels:
        raise RuntimeError(f"No labeled videos found in {labels_csv}.")
    return labels


class BenchmarkVideoDataset:
    def __init__(
        self,
        video_dir: Path,
        labels_csv: Path,
        overlap: float = OVERLAP,
        image_size: int = IMAGE_SIZE,
        cache_dir: Optional[Path] = None,
    ):
        if not video_dir.is_dir():
            raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
        if not labels_csv.is_file():
            raise FileNotFoundError(f"Labels CSV does not exist: {labels_csv}")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1).")

        label_by_id = _read_video_labels(labels_csv)
        candidates = sorted(
            path
            for path in video_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mp4"
        )

        matched: List[Tuple[Path, str, int]] = []
        matched_ids = set()
        unlabeled_files = 0
        for path in candidates:
            numeric_id = _filename_id(path)
            if numeric_id is None:
                continue
            if numeric_id not in label_by_id:
                unlabeled_files += 1
                continue
            video_id, label = label_by_id[numeric_id]
            matched.append((path, video_id, label))
            matched_ids.add(numeric_id)

        if not matched:
            raise RuntimeError(
                f"No MP4 files in {video_dir} matched CSV IDs like video123 "
                "to filename suffixes like -123.mp4."
            )

        self.videos: List[ClipInfo] = []
        self.windows: List[ClipWindow] = []
        stride = max(1, int(round(NUM_VIDEO_FRAMES * (1.0 - overlap))))

        for path, video_id, label in matched:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps <= 0 or num_frames <= 0:
                raise RuntimeError(
                    f"Invalid video metadata for {path}: fps={fps}, frames={num_frames}."
                )

            clip_index = len(self.videos)
            self.videos.append(
                ClipInfo(str(path), video_id, label, fps, num_frames)
            )

            if num_frames <= NUM_VIDEO_FRAMES:
                starts = [0]
            else:
                last_start = num_frames - NUM_VIDEO_FRAMES
                starts = list(range(0, last_start + 1, stride))
                if starts[-1] != last_start:
                    starts.append(last_start)

            self.windows.extend(
                ClipWindow(
                    clip_index,
                    start,
                    start + NUM_VIDEO_FRAMES - 1,
                )
                for start in starts
            )

        missing_ids = sorted(set(label_by_id) - matched_ids)
        duplicate_suffix_ids = len(matched) - len(matched_ids)
        print(
            f"Matched {len(matched)} video files across {len(matched_ids)} "
            f"CSV IDs; generated {len(self.windows)} windows."
        )
        if duplicate_suffix_ids:
            print(
                f"[INFO] {duplicate_suffix_ids} extra files share a labeled "
                "numeric suffix; each receives that suffix's CSV label."
            )
        if missing_ids:
            print(
                f"[WARNING] {len(missing_ids)} CSV IDs have no matching MP4; "
                f"first IDs: {[f'video{x}' for x in missing_ids[:10]]}"
            )
        if unlabeled_files:
            print(
                f"[WARNING] Skipped {unlabeled_files} video files whose IDs "
                "are not present in the CSV yet."
            )

        self.labels_csv = str(labels_csv)
        self.cache_dir = cache_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        if self.cache_dir is None:
            raise RuntimeError("BenchmarkVideoDataset requires cache_dir for training.")

        cache_path = self.cache_dir / f"{index:08d}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing feature cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    def get_sample_metadata(self, index: int) -> dict:
        window = self.windows[index]
        clip = self.videos[window.clip_index]
        return {
            "video_index": window.clip_index,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "start_time": window.start_frame / clip.fps,
            "end_time": (window.end_frame + 1) / clip.fps,
            "user": clip.video_id,
            "video_id": clip.path,
            "json_id": self.labels_csv,
            "labels": torch.full(
                (NUM_TEMPORAL_TOKENS,), clip.label, dtype=torch.long
            ),
            "temporal_padding_mask": self._temporal_padding_mask(window),
        }

    def _temporal_padding_mask(self, window: ClipWindow) -> torch.Tensor:
        clip = self.videos[window.clip_index]
        valid_frames = torch.tensor(
            [window.start_frame + i < clip.num_frames for i in range(NUM_VIDEO_FRAMES)],
            dtype=torch.bool,
        )
        valid_tokens = valid_frames.reshape(NUM_TEMPORAL_TOKENS, -1).any(dim=1)
        return ~valid_tokens

    def decode_video_window(self, index: int) -> dict:
        window = self.windows[index]
        clip = self.videos[window.clip_index]
        cap = cv2.VideoCapture(clip.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {clip.path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)

        frames = []
        frame_padding_mask = torch.ones(NUM_VIDEO_FRAMES, dtype=torch.bool)
        for frame_index in range(NUM_VIDEO_FRAMES):
            ok, frame = cap.read()
            if not ok:
                frames.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(Image.fromarray(rgb)))
            frame_padding_mask[frame_index] = False
        cap.release()

        return {
            "video": torch.stack(frames).permute(1, 0, 2, 3).contiguous(),
            "frame_padding_mask": frame_padding_mask,
            **self.get_sample_metadata(index),
        }

    def window_indices_for_clip_indices(self, clip_indices: Sequence[int]) -> np.ndarray:
        selected = set(int(index) for index in clip_indices)
        return np.asarray(
            [i for i, window in enumerate(self.windows) if window.clip_index in selected],
            dtype=np.int64,
        )


def extract_audio_features(
    beats,
    dataset: BenchmarkVideoDataset,
    cache_dir: Path,
    device: torch.device,
    indices,
) -> None:
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError("torchaudio is required for BEATs feature extraction.") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    current_clip_index = None
    waveform = None

    for index in tqdm(indices, desc="Precompute BEATs"):
        cache_path = cache_dir / f"{index:08d}.pt"
        item = torch.load(cache_path, map_location="cpu", weights_only=True)
        if item.get("audio_features") is not None and item.get("audio_padding_mask") is not None:
            continue

        metadata = dataset.get_sample_metadata(index)
        clip = dataset.videos[metadata["video_index"]]
        if current_clip_index != metadata["video_index"]:
            waveform, sample_rate = torchaudio.load(clip.path)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.float()
            if sample_rate != AUDIO_SR:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, AUDIO_SR
                )
            current_clip_index = metadata["video_index"]

        start_sample = round(metadata["start_time"] * AUDIO_SR)
        duration_samples = round(NUM_VIDEO_FRAMES / clip.fps * AUDIO_SR)
        end_sample = min(waveform.shape[-1], start_sample + duration_samples)
        valid_samples = max(0, end_sample - start_sample)
        segment = torch.zeros(1, duration_samples, dtype=torch.float32)
        audio_padding_mask = torch.ones(duration_samples, dtype=torch.bool)
        if valid_samples:
            segment[:, :valid_samples] = waveform[:, start_sample:end_sample]
            audio_padding_mask[:valid_samples] = False

        features, feature_padding_mask = beats.extract_features(
            segment.to(device),
            padding_mask=audio_padding_mask.unsqueeze(0).to(device),
        )
        item["audio_features"] = features.squeeze(0).float().cpu()
        item["audio_padding_mask"] = feature_padding_mask.squeeze(0).bool().cpu()
        torch.save(item, cache_path)


def validate_cache(
    cache_dir: Path,
    dataset: BenchmarkVideoDataset,
    indices,
) -> None:
    for index in indices:
        path = cache_dir / f"{index:08d}.pt"
        if not path.is_file():
            raise RuntimeError(f"Missing feature cache: {path}")
        item = torch.load(path, map_location="cpu", weights_only=True)
        video = item.get("video_features")
        audio = item.get("audio_features")
        audio_mask = item.get("audio_padding_mask")
        labels = item.get("labels")
        temporal_mask = item.get("temporal_padding_mask")
        if (
            video is None
            or audio is None
            or audio_mask is None
            or labels is None
            or temporal_mask is None
        ):
            raise RuntimeError(f"Incomplete feature cache: {path}")
        if video.ndim != 2 or video.shape[0] != NUM_TEMPORAL_TOKENS:
            raise RuntimeError(f"Invalid video feature shape in {path}: {tuple(video.shape)}")
        if audio.ndim != 2 or audio_mask.shape != (audio.shape[0],):
            raise RuntimeError(f"Invalid audio feature shape in {path}.")
        if labels.shape != (NUM_TEMPORAL_TOKENS,) or temporal_mask.shape != labels.shape:
            raise RuntimeError(f"Invalid labels or temporal mask in {path}.")
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise RuntimeError(f"Non-finite features in {path}.")
    print(f"Feature cache validated: {len(indices)} windows.")


def choose_video_ids(dataset: BenchmarkVideoDataset, max_ids, seed: int):
    id_to_label = {
        _video_id_number(clip.video_id): clip.label
        for clip in dataset.videos
    }
    if max_ids is None or max_ids >= len(id_to_label):
        return set(id_to_label)
    if max_ids < NUM_CLASSES * 2:
        raise ValueError("--max-video-ids must be at least 6 for a stratified split.")

    rng = np.random.default_rng(seed)
    by_class = {
        class_id: np.asarray([
            video_id for video_id, label in id_to_label.items()
            if label == class_id
        ])
        for class_id in range(NUM_CLASSES)
    }
    if any(len(values) < 2 for values in by_class.values()):
        raise RuntimeError("Each class needs at least two video IDs.")

    selected = []
    for values in by_class.values():
        selected.extend(rng.permutation(values)[:max_ids // NUM_CLASSES].tolist())
    remaining = max_ids - len(selected)
    if remaining:
        available = [
            value for values in by_class.values()
            for value in values if value not in selected
        ]
        selected.extend(rng.permutation(available)[:remaining].tolist())
    return set(selected)


def compute_alpha(dataset: BenchmarkVideoDataset, train_indices: np.ndarray) -> torch.Tensor:
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for index in train_indices:
        metadata = dataset.get_sample_metadata(int(index))
        valid = ~metadata["temporal_padding_mask"]
        counts += torch.bincount(metadata["labels"][valid], minlength=NUM_CLASSES).double()
    if (counts == 0).any():
        missing = [ID2LABEL[i] for i in range(NUM_CLASSES) if counts[i] == 0]
        raise RuntimeError(f"Training split is missing classes: {missing}")
    alpha = torch.sqrt(counts.sum() / (NUM_CLASSES * counts))
    return (alpha / alpha.mean()).float()


def _flatten_epoch_metrics(epoch: int, train_metrics: dict, val_metrics: dict) -> dict:
    row: dict[str, object] = {"epoch": epoch}
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for name, value in metrics.items():
            key = f"{split}_{name}"
            row[key] = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AV temporal classifier from benchmark CSV labels."
    )
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    parser.add_argument("--videomae", default=DEFAULT_VIDEOMAE)
    parser.add_argument("--beats-checkpoint", default=BEATS_CHECKPOINT)
    parser.add_argument("--beats-root", default=BEATS_ROOT)
    parser.add_argument("--cache-dir", default=f"{VERSION_ARTIFACTS}/benchmark_feature_cache")
    parser.add_argument("--output-dir", default=f"{VERSION_ARTIFACTS}/checkpoints_benchmark")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--max-video-ids",
        type=int,
        default=None,
        help="Optional stratified video-ID limit for smoke runs; default uses all IDs.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers nonnegative.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    dataset = BenchmarkVideoDataset(
        Path(args.video_dir),
        Path(args.labels_csv),
        cache_dir=Path(args.cache_dir),
    )
    selected_ids = choose_video_ids(dataset, args.max_video_ids, args.seed)
    active_indices = np.asarray([
        index
        for index, window in enumerate(dataset.windows)
        if _video_id_number(dataset.videos[window.clip_index].video_id) in selected_ids
    ], dtype=np.int64)
    if args.dry_run:
        counts = {name: 0 for name in LABEL2ID}
        for clip in dataset.videos:
            if _video_id_number(clip.video_id) in selected_ids:
                counts[ID2LABEL[clip.label]] += 1
        print(f"Labeled files by class: {counts}")
        print(f"Selected video IDs={len(selected_ids)} | windows={len(active_indices)}")
        print(f"Unique CSV video IDs: {len({_video_id_number(x.video_id) for x in dataset.videos})}")
        print("Dry run complete; no features extracted and no model trained.")
        return

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_precompute:
        videomae = load_videomae(args.videomae, device)
        extract_video_features(videomae, dataset, active_indices, device, cache_dir)
        del videomae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        beats = load_beats(args.beats_checkpoint, args.beats_root, device)
        extract_audio_features(beats, dataset, cache_dir, device, active_indices)
        del beats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validate_cache(cache_dir, dataset, active_indices)

    id_to_label = {
        _video_id_number(clip.video_id): clip.label
        for clip in dataset.videos
        if _video_id_number(clip.video_id) in selected_ids
    }
    all_video_ids = np.asarray(sorted(id_to_label), dtype=np.int64)
    id_labels = np.asarray([id_to_label[value] for value in all_video_ids])
    train_ids, val_ids = train_test_split(
        all_video_ids,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=id_labels,
    )
    train_clips = [
        index for index, clip in enumerate(dataset.videos)
        if _video_id_number(clip.video_id) in set(train_ids.tolist())
    ]
    val_clips = [
        index for index, clip in enumerate(dataset.videos)
        if _video_id_number(clip.video_id) in set(val_ids.tolist())
    ]
    train_indices = dataset.window_indices_for_clip_indices(train_clips)
    val_indices = dataset.window_indices_for_clip_indices(val_clips)
    print(f"Train windows={len(train_indices)} | validation windows={len(val_indices)}")

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": feature_collate_fn,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()), shuffle=False, **loader_kwargs
    )
    first_item = torch.load(
        cache_dir / f"{int(train_indices[0]):08d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = AVFeatureModel(
        video_dim=first_item["video_features"].shape[-1],
        audio_dim=first_item["audio_features"].shape[-1],
        fusion_dim=768,
        cross_heads=16,
        temporal_heads=12,
        temporal_layers=2,
        dropout=0.1,
        head_hidden_dim=256,
        head_dropout=0.3,
        num_classes=NUM_CLASSES,
    ).to(device)

    optimizer = build_optimizer(model)
    criterion = TemporalFocalLoss(
        alpha=tuple(compute_alpha(dataset, train_indices).tolist()),
        gamma=args.focal_gamma,
    ).to(device)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(
        optimizer,
        args.epochs * steps_per_epoch,
        WARMUP_EPOCHS * steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -math.inf
    best_metrics = None
    stale_epochs = 0
    history = []

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, True,
            optimizer, scaler, scheduler, use_amp,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, False,
            use_amp=use_amp,
        )
        score = val_metrics["macro_f1"]
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} macro_f1={score:.4f}"
        )

        history.append(_flatten_epoch_metrics(epoch + 1, train_metrics, val_metrics))
        with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        if score > best_f1:
            best_f1 = score
            best_metrics = val_metrics
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "video_dim": first_item["video_features"].shape[-1],
                    "audio_dim": first_item["audio_features"].shape[-1],
                    "label2id": LABEL2ID,
                    "best_epoch": epoch + 1,
                    "validation_metrics": val_metrics,
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_metrics is not None:
        with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(best_metrics, handle, indent=2)
        print(f"Best validation macro-F1={best_f1:.5f}")
        print(f"Saved checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()