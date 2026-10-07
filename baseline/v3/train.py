from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import VideoMAEModel
from config import *
from dataset import (
    PickleballFeatureDataset,
    feature_collate_fn,
    LABEL2ID,
    ID2LABEL,
    NUM_VIDEO_FRAMES,
    NUM_TEMPORAL_TOKENS,
    EVENT_RADIUS_SEC,
)
from model import (
    AVFeatureModel,
    TemporalFocalLoss,
)


# ============================================================
# General
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def move_batch(batch, device):
    return {
        k: (
            v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


# ============================================================
# Pair discovery
# ============================================================

def discover_pairs(data_dir):
    root = Path(data_dir)
    videos = []
    jsons = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        js = path.with_suffix(".json")
        if js.exists():
            videos.append(str(path))
            jsons.append(str(js))
        else:
            print(f"[WARNING] Missing JSON: {path}")

    if not videos:
        raise RuntimeError(
            f"No video/json pairs found in {root}"
        )

    return videos, jsons


# ============================================================
# VideoMAE
# ============================================================

def adapt_videomae_to_32_frames(model):
    target_frames = NUM_VIDEO_FRAMES
    old_frames = int(model.config.num_frames)
    tubelet = int(model.config.tubelet_size)

    if old_frames == target_frames:
        return

    if target_frames % tubelet:
        raise ValueError(
            "Target frames must be divisible by tubelet size."
        )

    old_t = old_frames // tubelet
    new_t = target_frames // tubelet

    image_size = model.config.image_size
    patch_size = model.config.patch_size

    if isinstance(image_size, int):
        ih = iw = image_size
    else:
        ih, iw = image_size

    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size

    hp = ih // ph
    wp = iw // pw

    pos = model.embeddings.position_embeddings
    expected = old_t * hp * wp

    if pos.shape[1] != expected:
        raise RuntimeError(
            f"Unexpected VideoMAE position embeddings: "
            f"{tuple(pos.shape)}, expected {expected}"
        )

    hidden = pos.shape[-1]

    x = pos.reshape(
        1,
        old_t,
        hp,
        wp,
        hidden,
    ).permute(0, 4, 1, 2, 3)

    x = nn.functional.interpolate(
        x,
        size=(new_t, hp, wp),
        mode="trilinear",
        align_corners=False,
    )

    new_pos = (
        x.permute(0, 2, 3, 4, 1)
        .reshape(
            1,
            new_t * hp * wp,
            hidden,
        )
        .contiguous()
    )

    model.embeddings.position_embeddings = nn.Parameter(
        new_pos,
        requires_grad=False,
    )
    model.config.num_frames = target_frames


def load_videomae(name, device):
    print("Loading VideoMAE...")
    model = VideoMAEModel.from_pretrained(name)
    adapt_videomae_to_32_frames(model)

    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ============================================================
# BEATs
# ============================================================

def load_beats(checkpoint_path, beats_root, device):
    if beats_root:
        import sys
        sys.path.insert(0, beats_root)
        sys.path.insert(0, os.path.join(beats_root, "beats"))

    try:
        from beats.BEATs import BEATs, BEATsConfig
    except ImportError:
        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError as exc:
            raise ImportError(
                "Cannot import BEATs. Set --beats-root to the "
                "Microsoft/unilm directory containing beats/."
            ) from exc

    print("Loading BEATs...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "cfg" not in checkpoint or "model" not in checkpoint:
        raise KeyError(
            "Invalid BEATs checkpoint: expected cfg and model."
        )

    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ============================================================
# Feature extraction
# ============================================================

@torch.inference_mode()
def extract_video_features(
    videomae,
    dataset,
    indices,
    device,
    cache_dir,
):
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hidden = int(videomae.config.hidden_size)
    tubelet = int(videomae.config.tubelet_size)
    spatial_tokens = (
        int(videomae.config.image_size)
        // int(videomae.config.patch_size)
    ) ** 2
    temporal_tokens = NUM_VIDEO_FRAMES // tubelet

    for index in tqdm(
        indices,
        desc="Precompute VideoMAE",
    ):
        path = cache_dir / f"{index:08d}.pt"

        if path.exists():
            continue

        sample = dataset.decode_video_window(
            int(index)
        )

        video = sample["video"].unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        video = video.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        output = videomae(
            pixel_values=video
        ).last_hidden_state

        expected = (
            temporal_tokens
            * spatial_tokens
        )

        if output.shape[1] != expected:
            raise RuntimeError(
                f"VideoMAE tokens={output.shape[1]}, "
                f"expected={expected}"
            )

        features = output.reshape(
            1,
            temporal_tokens,
            spatial_tokens,
            hidden,
        ).mean(dim=2).squeeze(0).cpu()

        labels = sample["labels"]
        temporal_padding_mask = sample[
            "temporal_padding_mask"
        ]

        torch.save(
            {
                "video_features": features.float(),
                "audio_features": None,
                "audio_padding_mask": None,
                "labels": labels,
                "temporal_padding_mask": (
                    temporal_padding_mask
                ),
                "video_index": sample["video_index"],
                "start_frame": sample["start_frame"],
                "end_frame": sample["end_frame"],
                "start_time": sample["start_time"],
                "end_time": sample["end_time"],
                "user": sample["user"],
                "video_id": sample["video_id"],
                "json_id": sample["json_id"],
            },
            path,
        )


def extract_audio_features(
    beats,
    dataset,
    indices,
    device,
    cache_dir,
):
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "torchaudio is required for BEATs feature extraction."
        ) from exc

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index in tqdm(
        indices,
        desc="Precompute BEATs",
    ):
        path = cache_dir / f"{index:08d}.pt"

        item = torch.load(
            path,
            map_location="cpu",
        )

        if (
            item["audio_features"] is not None
            and item["audio_padding_mask"] is not None
        ):
            continue

        sample = dataset.get_sample_metadata(
            int(index)
        )

        info = dataset.videos[
            sample["video_index"]
        ]

        waveform, sr = torchaudio.load(
            info.path
        )

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        waveform = waveform.float()

        if sr != AUDIO_SR:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                AUDIO_SR,
            )

        start_sample = int(
            round(
                sample["start_time"]
                * AUDIO_SR
            )
        )

        duration_samples = int(
            round(
                NUM_VIDEO_FRAMES
                / info.fps
                * AUDIO_SR
            )
        )

        end_sample = min(
            waveform.shape[-1],
            start_sample + duration_samples,
        )

        if start_sample >= end_sample:
            print(
                f"[WARN] Audio window empty for video={info.path} "
                f"start_sample={start_sample} end_sample={end_sample} "
                f"waveform_len={waveform.shape[-1]} "
                f"start_time={sample['start_time']} duration_samples={duration_samples}"
            )

            segment = torch.zeros(
                1,
                duration_samples,
                dtype=torch.float32,
            )
            audio_padding_mask = torch.ones(
                duration_samples,
                dtype=torch.bool,
            )
        else:
            segment = waveform[
                :,
                start_sample:end_sample,
            ]

            valid = segment.shape[-1]

            if valid < duration_samples:
                padded = torch.zeros(
                    1,
                    duration_samples,
                    dtype=torch.float32,
                )
                padded[:, :valid] = segment
                segment = padded

            audio_padding_mask = torch.ones(
                duration_samples,
                dtype=torch.bool,
            )
            audio_padding_mask[:valid] = False

        segment = segment.to(device)

        padding_device = audio_padding_mask.unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        features, feature_padding_mask = (
            beats.extract_features(
                segment,
                padding_mask=padding_device,
            )
        )

        features = features.squeeze(0).float().cpu()
        feature_padding_mask = (
            feature_padding_mask.squeeze(0)
            .bool()
            .cpu()
        )

        item["audio_features"] = features
        item["audio_padding_mask"] = (
            feature_padding_mask
        )

        torch.save(
            item,
            path,
        )


def validate_cache(cache_dir, dataset):
    invalid = []
    for i in range(len(dataset)):
        path = cache_dir / f"{i:08d}.pt"
        if not path.exists():
            invalid.append(i)
            continue
        try:
            item = torch.load(path, map_location="cpu")
            video = item.get("video_features")
            audio = item.get("audio_features")
            audio_mask = item.get("audio_padding_mask")
            labels = item.get("labels")
            temporal_mask = item.get("temporal_padding_mask")
            if video is None or audio is None or audio_mask is None or labels is None or temporal_mask is None:
                invalid.append(i)
                continue
            if video.ndim != 2 or video.shape[0] != NUM_TEMPORAL_TOKENS:
                invalid.append(i)
                continue
            if audio.ndim != 2 or not torch.isfinite(video).all() or not torch.isfinite(audio).all():
                invalid.append(i)
                continue
            if labels.shape != (NUM_TEMPORAL_TOKENS,) or temporal_mask.shape != (NUM_TEMPORAL_TOKENS,):
                invalid.append(i)
                continue
            if audio_mask.ndim != 1 or audio_mask.shape[0] != audio.shape[0]:
                invalid.append(i)
                continue
            if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                invalid.append(i)
                continue
            if ((labels < 0) | (labels >= NUM_CLASSES)).any():
                invalid.append(i)
        except Exception:
            invalid.append(i)
    if invalid:
        raise RuntimeError(f"Cache invalid: {len(invalid)} samples, first={invalid[:10]}")
    print(f"Feature cache validated: {len(dataset)} samples.")


# ============================================================
# Fold metadata
# ============================================================

def build_fold_metadata(dataset):
    labels = []
    users = []

    for i in tqdm(
        range(len(dataset)),
        desc="Building fold metadata",
    ):
        sample = dataset.get_sample_metadata(i)
        y = sample["labels"]
        mask = ~sample[
            "temporal_padding_mask"
        ]

        valid = y[mask]

        if (valid == LABEL2ID["hit"]).any():
            cls = LABEL2ID["hit"]
        elif (valid == LABEL2ID["miss"]).any():
            cls = LABEL2ID["miss"]
        else:
            cls = LABEL2ID["no_event"]

        labels.append(cls)
        users.append(sample["user"])

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(users, dtype=object),
    )


def compute_alpha(dataset, indices):
    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    for i in indices:
        sample = dataset.get_sample_metadata(
            int(i)
        )

        labels = sample["labels"]
        valid = ~sample[
            "temporal_padding_mask"
        ]

        counts += torch.bincount(
            labels[valid],
            minlength=NUM_CLASSES,
        ).double()

    if (counts == 0).any():
        missing = [
            ID2LABEL[i]
            for i in range(NUM_CLASSES)
            if counts[i] == 0
        ]
        raise RuntimeError(
            f"Missing training classes: {missing}"
        )

    alpha = torch.sqrt(
        counts.sum()
        / (NUM_CLASSES * counts)
    )
    alpha /= alpha.mean()

    return alpha.float()


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(model):
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not params:
        raise RuntimeError(
            "Optimizer received zero trainable parameters."
        )

    optimizer = torch.optim.AdamW(
        params,
        lr=FUSION_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    print(
        "Optimizer parameters:",
        f"{sum(p.numel() for p in params):,}",
    )

    return optimizer


def build_scheduler(
    optimizer,
    total_steps,
    warmup_steps,
):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(
                1,
                warmup_steps,
            )

        progress = (
            step - warmup_steps
        ) / max(
            1,
            total_steps - warmup_steps,
        )

        progress = min(
            1.0,
            max(0.0, progress),
        )

        return 0.5 * (
            1 + math.cos(
                math.pi * progress
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


# ============================================================
# Metrics
# ============================================================

def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    precision = tp / np.maximum(
        tp + fp,
        1e-12,
    )
    recall = tp / np.maximum(
        tp + fn,
        1e-12,
    )

    f1 = (
        2 * precision * recall
        / np.maximum(
            precision + recall,
            1e-12,
        )
    )

    total = cm.sum()
    support = cm.sum(axis=1)
    weighted_f1 = float(
        np.sum(f1 * support) / np.maximum(total, 1e-12)
    ) if total else 0.0

    metrics = {
        "accuracy": float(
            tp.sum() / total
        ) if total else 0.0,
        "macro_precision": float(
            precision.mean()
        ),
        "macro_recall": float(
            recall.mean()
        ),
        "macro_f1": float(
            f1.mean()
        ),
        "weighted_f1": weighted_f1,
        "support_total": int(total),
        "support_by_class": {
            str(ID2LABEL[i]): int(support[i])
            for i in range(len(ID2LABEL))
        },
    }

    for i in range(len(ID2LABEL)):
        label_name = str(ID2LABEL[i])
        metrics[f"precision_{label_name}"] = float(
            precision[i]
        )
        metrics[f"recall_{label_name}"] = float(
            recall[i]
        )
        metrics[f"f1_{label_name}"] = float(
            f1[i]
        )

    return metrics


# ============================================================
# Epoch
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    train,
    optimizer=None,
    scaler=None,
    scheduler=None,
    use_amp=False,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    batches = 0

    cm = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    trainable_params = None
    if train:
        trainable_params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

    pbar = tqdm(
        loader,
        leave=False,
        desc="Train" if train else "Val",
    )

    for batch in pbar:
        batch = move_batch(
            batch,
            device,
        )

        if train:
            optimizer.zero_grad(
                set_to_none=True
            )

        context = torch.enable_grad() if train else torch.inference_mode()
        with context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(
                    batch["video_features"],
                    batch["audio_features"],
                    batch["audio_padding_mask"],
                    batch["temporal_padding_mask"],
                )
                if train and not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite logits detected.")
                loss = criterion(logits, batch["labels"], batch["temporal_padding_mask"])
            if train:
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss: {loss.item()}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        total_loss += float(
            loss.detach()
        )
        batches += 1

        valid = ~batch[
            "temporal_padding_mask"
        ]

        pred = (
            logits.argmax(-1)[valid]
            .detach()
            .cpu()
            .numpy()
        )
        true = (
            batch["labels"][valid]
            .detach()
            .cpu()
            .numpy()
        )

        if len(true):
            cm += confusion_matrix(
                true,
                pred,
                labels=list(range(NUM_CLASSES)),
            )

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    result = metrics_from_cm(cm)
    result["loss"] = (
        total_loss / max(1, batches)
    )

    return result


# ============================================================
# Fold
# ============================================================

def train_fold(
    fold,
    train_idx,
    val_idx,
    dataset,
    args,
    device,
    use_amp,
    users,
):
    train_set = Subset(
        dataset,
        train_idx.tolist(),
    )
    val_set = Subset(
        dataset,
        val_idx.tolist(),
    )

    generator = torch.Generator().manual_seed(
        args.seed + fold
    )

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
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **loader_kwargs,
    )

    sample = dataset[int(train_idx[0])]

    video_dim = sample[
        "video_features"
    ].shape[-1]
    audio_dim = sample[
        "audio_features"
    ].shape[-1]

    model = AVFeatureModel(
        video_dim=video_dim,
        audio_dim=audio_dim,
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

    alpha = compute_alpha(
        dataset,
        train_idx,
    )

    criterion = TemporalFocalLoss(
        alpha=tuple(alpha.tolist()),
        gamma=args.focal_gamma,
    ).to(device)

    steps = max(
        1,
        len(train_loader),
    )

    scheduler = build_scheduler(
        optimizer,
        args.epochs * steps,
        args.warmup_epochs * steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_f1 = -float("inf")
    best_epoch = None
    best_metrics = {}
    stale = 0

    for epoch in range(args.epochs):
        train_m = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            True,
            optimizer,
            scaler,
            scheduler,
            use_amp,
        )

        val_m = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            False,
            None,
            None,
            None,
            use_amp,
        )

        epoch_idx = epoch + 1
        current = val_m["macro_f1"]
        
        print(
            f"Fold {fold} | Epoch {epoch_idx:03d}/{args.epochs} | "
            f"train loss={train_m['loss']:.4f} "
            f"acc={train_m['accuracy']:.4f} macro_f1={train_m['macro_f1']:.4f} "
            f"| val loss={val_m['loss']:.4f} "
            f"acc={val_m['accuracy']:.4f} macro_f1={current:.4f} "
            f"precision={val_m['macro_precision']:.4f} recall={val_m['macro_recall']:.4f}"
        )

        if current > best_f1:
            best_f1 = current
            best_epoch = epoch_idx
            best_metrics = val_m
            stale = 0
            print(
                f"  NEW BEST: epoch={best_epoch} macro_f1={best_f1:.5f}"
            )
        else:
            stale += 1

        if stale >= args.patience:
            print("  Early stopping.")
            break

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "accuracy": best_metrics.get("accuracy", 0),
        "precision": best_metrics.get("macro_precision", 0),
        "recall": best_metrics.get("macro_recall", 0)
    }


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default = "/data/Cuong/pickleball/data/raw",
    )
    p.add_argument(
        "--beats-checkpoint",
        default = BEATS_CHECKPOINT,
    )
    p.add_argument(
        "--beats-root",
        default=BEATS_ROOT,
    )
    p.add_argument(
        "--videomae",
        default=DEFAULT_VIDEOMAE,
    )
    p.add_argument(
        "--cache-dir",
        default="./feature_cache",
    )
    p.add_argument(
        "--output-dir",
        default="./checkpoints",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
    )
    p.add_argument(
        "--val-users",
        type=int,
        default=1,
        help="Số lượng user trong tập val cho mỗi fold (k = 1, 3, 5)",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
    )
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=WARMUP_EPOCHS,
    )
    p.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
    )
    p.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Use existing feature cache.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("=" * 90)
    print(
        "PICKLEBALL OFFLINE AV FEATURE TRAINING"
    )
    print(
        f"Device={device} | AMP={use_amp}"
    )
    print(f"K-Fold Dynamic: k={args.val_users} users per validation fold")
    print(
        "Event window: "
        "[event_time - 0.5s, event_time + 0.5s] "
        "=> exactly 1.0 second"
    )
    print(
        "Training: cached features only"
    )
    print("=" * 90)

    video_paths, json_paths = discover_pairs(
        args.data_dir
    )

    dataset = PickleballFeatureDataset(
        video_paths,
        json_paths,
        num_video_frames=NUM_VIDEO_FRAMES,
        num_temporal_tokens=NUM_TEMPORAL_TOKENS,
        image_size=IMAGE_SIZE,
        overlap=OVERLAP,
        cache_dir=args.cache_dir,
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # PRECOMPUTE ONLY ONCE
    # --------------------------------------------------------

    if not args.skip_precompute:
        print()
        print(
            "=== OFFLINE PRECOMPUTATION ==="
        )

        videomae = load_videomae(
            args.videomae,
            device,
        )

        extract_video_features(
            videomae,
            dataset,
            range(len(dataset)),
            device,
            cache_dir,
        )

        del videomae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        beats = load_beats(
            args.beats_checkpoint,
            args.beats_root,
            device,
        )

        extract_audio_features(
            beats,
            dataset,
            range(len(dataset)),
            device,
            cache_dir,
        )

        del beats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            "=== PRECOMPUTATION COMPLETE ==="
        )

    validate_cache(
        cache_dir,
        dataset,
    )

    # --------------------------------------------------------
    # Metadata / user grouped CV
    # --------------------------------------------------------

    stratify_labels, users = (
        build_fold_metadata(dataset)
    )

    unique_users = sorted(
        set(users.tolist())
    )

    # 1. Trộn và tự động chia nhóm users thành các chunk kích thước k
    np.random.default_rng(args.seed).shuffle(unique_users)
    k = args.val_users
    
    user_chunks = [unique_users[i:i + k] for i in range(0, len(unique_users), k)]
    print(f"[SPLIT] K = {k}: Tự động chia tập dữ liệu thành {len(user_chunks)} folds.")

    # 2. Tạo thư mục con k1, k3, k5...
    out_dir = Path(args.output_dir) / f"k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for fold, val_user_group in enumerate(user_chunks, 1):
        if args.fold is not None and fold != args.fold:
            continue
            
        val_users_set = set(val_user_group)
        train_idx = np.array([i for i, u in enumerate(users) if u not in val_users_set])
        val_idx = np.array([i for i, u in enumerate(users) if u in val_users_set])
        
        print(f"\n{'='*40}")
        print(f"-> Đang chạy Fold {fold}/{len(user_chunks)} | Val Users = {val_user_group}")
        print(f"{'='*40}")

        fold_result = train_fold(
            fold,
            train_idx,
            val_idx,
            dataset,
            args,
            device,
            use_amp,
            users,
        )
        fold_result["val_users"] = ", ".join(val_user_group)
        results.append(fold_result)

    if results:
        csv_path = out_dir / "results.csv"
        
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            fieldnames = ["fold", "val_users", "best_epoch", "best_macro_f1", "accuracy", "precision", "recall"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        scores = np.asarray([x["best_macro_f1"] for x in results], dtype=float)
        
        print("\n" + "=" * 90)
        print(f"HOÀN THÀNH HUẤN LUYỆN CHO K={k}")
        print(f"Đã lưu kết quả tại: {csv_path}")
        print(f"CV mean macro-F1: {scores.mean():.5f} +/- {scores.std():.5f}")
        print("=" * 90)

    dataset.close()


if __name__ == "__main__":
    main()