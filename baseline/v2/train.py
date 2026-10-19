from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import VideoMAEModel

from config import *
from dataset import (
    PickleballAVDataset,
    pickleball_collate_fn,
    LABEL2ID,
    ID2LABEL,
    NUM_VIDEO_FRAMES,
    NUM_TEMPORAL_TOKENS,
    EVENT_RADIUS_SEC,
)
from model import AVModel, TemporalFocalLoss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def discover_video_json_pairs(data_dir):
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(root)

    extensions = set(VIDEO_EXTENSIONS) | {".webm"}
    video_paths = []
    json_paths = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        json_path = path.with_suffix(".json")

        if json_path.exists():
            video_paths.append(str(path))
            json_paths.append(str(json_path))
        else:
            print(f"[WARNING] Missing JSON: {path}")

    if not video_paths:
        raise RuntimeError(
            f"No video/JSON pairs found in {root}"
        )

    return video_paths, json_paths


def frame_label_from_events(events, event_times, t, radius):
    """
    Event interval is exactly 1 second total:
        [event_time - 0.5, event_time + 0.5].

    This function is optimized with binary search instead of scanning
    every event for every frame.
    """
    if not events:
        return LABEL2ID["no_event"]

    import bisect

    pos = bisect.bisect_left(
        event_times,
        t - radius,
    )

    if pos < len(event_times):
        if event_times[pos] <= t + radius:
            return events[pos].label

    if pos > 0:
        prev = pos - 1
        if abs(event_times[prev] - t) <= radius:
            return events[prev].label

    return LABEL2ID["no_event"]


def get_window_labels(dataset, index):
    """Build temporal labels without decoding media."""
    window = dataset.windows[index]
    info = dataset.videos[window.video_index]

    frame_labels = torch.full(
        (NUM_VIDEO_FRAMES,),
        LABEL2ID["no_event"],
        dtype=torch.long,
    )

    valid_frames = torch.zeros(
        NUM_VIDEO_FRAMES,
        dtype=torch.bool,
    )

    radius = EVENT_RADIUS_SEC

    for i in range(NUM_VIDEO_FRAMES):
        frame_index = window.start_frame + i

        if frame_index >= info.num_frames:
            continue

        valid_frames[i] = True

        t = frame_index / info.fps
        frame_labels[i] = frame_label_from_events(
            info.events,
            info.event_times,
            t,
            radius,
        )

    labels = torch.full(
        (NUM_TEMPORAL_TOKENS,),
        LABEL2ID["no_event"],
        dtype=torch.long,
    )

    temporal_valid = torch.zeros(
        NUM_TEMPORAL_TOKENS,
        dtype=torch.bool,
    )

    frames_per_token = (
        NUM_VIDEO_FRAMES // NUM_TEMPORAL_TOKENS
    )

    for token in range(NUM_TEMPORAL_TOKENS):
        start = token * frames_per_token
        end = start + frames_per_token

        valid = valid_frames[start:end]

        if not valid.any():
            continue

        temporal_valid[token] = True
        token_labels = frame_labels[start:end][valid]

        if (token_labels == LABEL2ID["hit"]).any():
            labels[token] = LABEL2ID["hit"]
        elif (token_labels == LABEL2ID["miss"]).any():
            labels[token] = LABEL2ID["miss"]

    return labels, temporal_valid


def build_kfold_metadata(dataset):
    stratify_labels = []
    users = []

    for i in tqdm(
        range(len(dataset)),
        desc="Building fold metadata",
    ):
        labels, valid = get_window_labels(
            dataset,
            i,
        )

        valid_labels = labels[valid]

        if (valid_labels == LABEL2ID["hit"]).any():
            label = LABEL2ID["hit"]
        elif (valid_labels == LABEL2ID["miss"]).any():
            label = LABEL2ID["miss"]
        else:
            label = LABEL2ID["no_event"]

        users.append(
            dataset.videos[
                dataset.windows[i].video_index
            ].user
        )
        stratify_labels.append(label)

    return (
        np.asarray(
            stratify_labels,
            dtype=np.int64,
        ),
        np.asarray(users, dtype=object),
    )


def compute_class_alpha(dataset, indices):
    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    for index in tqdm(
        indices,
        desc="Counting train labels",
    ):
        labels, valid = get_window_labels(
            dataset,
            int(index),
        )

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
            f"Training fold has no samples for: {missing}"
        )

    alpha = torch.sqrt(
        counts.sum()
        / (NUM_CLASSES * counts)
    )
    alpha = alpha / alpha.mean()

    return alpha.float()


def adapt_videomae_to_frames(model, target_frames):
    """Adapt VideoMAE temporal length while preserving spatial tokens."""
    old_frames = int(model.config.num_frames)
    tubelet = int(model.config.tubelet_size)

    if old_frames == target_frames:
        return

    if target_frames % tubelet:
        raise ValueError(
            f"{target_frames=} must be divisible by {tubelet=}"
        )

    image_size = model.config.image_size
    patch_size = model.config.patch_size

    ih, iw = (
        (image_size, image_size)
        if isinstance(image_size, int)
        else image_size
    )
    ph, pw = (
        (patch_size, patch_size)
        if isinstance(patch_size, int)
        else patch_size
    )

    old_t = old_frames // tubelet
    new_t = target_frames // tubelet

    hp, wp = ih // ph, iw // pw

    pos = model.embeddings.position_embeddings
    expected = old_t * hp * wp

    if pos.shape[1] != expected:
        raise RuntimeError(
            f"Unexpected VideoMAE position shape "
            f"{tuple(pos.shape)}; expected {expected}"
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


def load_beats(
    beat_checkpoint=BEATS_CHECKPOINT,
    base_unilm=None,
):
    if base_unilm is not None:
        import sys

        beats_dir = os.path.join(
            base_unilm,
            "beats",
        )

        sys.path.append(base_unilm)
        sys.path.append(beats_dir)

    from beats.BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(
        beat_checkpoint,
        map_location="cpu",
    )

    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])

    # Pretrained backbone is always inference-only.
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    return model


def build_optimizer(model):
    """
    Optimizer receives ONLY parameters with requires_grad=True.

    Frozen VideoMAE/BEATs parameters are never included.
    """
    groups = {
        "video": [],
        "audio": [],
        "fusion": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("video_encoder."):
            groups["video"].append((name, param))
        elif name.startswith("audio_encoder."):
            groups["audio"].append((name, param))
        else:
            groups["fusion"].append((name, param))

    param_groups = []

    lrs = {
        "video": VIDEO_LR,
        "audio": AUDIO_LR,
        "fusion": FUSION_LR,
    }

    for key, items in groups.items():
        if not items:
            continue

        decay = []
        no_decay = []

        for name, param in items:
            if (
                param.ndim == 1
                or name.endswith(".bias")
                or "norm" in name.lower()
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        if decay:
            param_groups.append({
                "params": decay,
                "lr": lrs[key],
                "weight_decay": WEIGHT_DECAY,
                "group": key,
            })

        if no_decay:
            param_groups.append({
                "params": no_decay,
                "lr": lrs[key],
                "weight_decay": 0.0,
                "group": key,
            })

    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. "
            "Check requires_grad configuration."
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    optimized_params = sum(
        p.numel()
        for group in param_groups
        for p in group["params"]
    )

    print(
        f"Optimizer parameters: "
        f"{optimized_params:,} trainable"
    )

    return optimizer


def build_scheduler(
    optimizer,
    total_steps,
    warmup_steps,
):
    def lr_lambda(step):
        if (
            warmup_steps > 0
            and step < warmup_steps
        ):
            return (
                (step + 1)
                / warmup_steps
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
            1.0
            + math.cos(
                math.pi * progress
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


def move_batch(batch, device):
    return {
        k: (
            v.to(
                device,
                non_blocking=True,
            )
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


def metrics_from_confusion(cm):
    tp = np.diag(cm).astype(np.float64)
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

    return {
        "accuracy": (
            float(tp.sum() / total)
            if total
            else 0.0
        ),
        "macro_precision": float(
            precision.mean()
        ),
        "macro_recall": float(
            recall.mean()
        ),
        "macro_f1": float(
            f1.mean()
        ),
        "f1_no_event": float(
            f1[LABEL2ID["no_event"]]
        ),
        "f1_hit": float(
            f1[LABEL2ID["hit"]]
        ),
        "f1_miss": float(
            f1[LABEL2ID["miss"]]
        ),
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    train,
    optimizer,
    scaler,
    scheduler,
    use_amp,
):
    model.train(train)

    total_loss = 0.0
    batches = 0

    cm = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

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

        with torch.set_grad_enabled(train):
            with torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
                enabled=use_amp,
            ):
                logits, padding_mask = model(
                    batch,
                    return_mask=True,
                )

                loss = criterion(
                    logits,
                    batch["labels"],
                    padding_mask,
                )

            if train and not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss detected: "
                    f"loss={loss.item()} "
                    f"logits_finite="
                    f"{torch.isfinite(logits).all().item()}"
                )

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Clip only trainable parameters.
                trainable_params = (
                    p
                    for p in model.parameters()
                    if p.requires_grad
                )

                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    GRAD_CLIP,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        total_loss += loss.detach().item()
        batches += 1

        valid = ~padding_mask

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

    result = metrics_from_confusion(cm)
    result["loss"] = (
        total_loss / max(1, batches)
    )

    return result


@torch.no_grad()
def shape_check(model, loader, device):
    batch = move_batch(
        next(iter(loader)),
        device,
    )

    logits, padding_mask = model(
        batch,
        return_mask=True,
    )

    expected = (
        batch["video"].shape[0],
        NUM_TEMPORAL_TOKENS,
        NUM_CLASSES,
    )

    if tuple(logits.shape) != expected:
        raise RuntimeError(
            f"Logits shape {tuple(logits.shape)} != {expected}"
        )

    if tuple(padding_mask.shape) != expected[:2]:
        raise RuntimeError(
            f"Temporal mask shape "
            f"{tuple(padding_mask.shape)} "
            f"!= {expected[:2]}"
        )

    if padding_mask.dtype != torch.bool:
        raise RuntimeError(
            "Temporal padding mask must be bool"
        )

    print(
        "Shape check PASSED: "
        f"video={tuple(batch['video'].shape)} "
        f"logits={tuple(logits.shape)}"
    )


def verify_trainable_parameters(model):
    """Hard safety check for the requested training policy."""
    frozen_backbone = (
        list(model.video_encoder.parameters())
        + list(model.audio_encoder.parameters())
    )

    if any(
        p.requires_grad
        for p in frozen_backbone
    ):
        raise RuntimeError(
            "Pretrained backbone has trainable parameters."
        )

    trainable = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]

    if not trainable:
        raise RuntimeError(
            "No trainable parameters."
        )

    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for _, p in trainable):,}"
    )

    print(
        f"Frozen VideoMAE: "
        f"{sum(p.numel() for p in model.video_encoder.parameters()):,}"
    )

    print(
        f"Frozen BEATs: "
        f"{sum(p.numel() for p in model.audio_encoder.parameters()):,}"
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_f1,
    fold,
    args,
    history,
):
    torch.save(
        {
            "epoch": epoch,
            "fold": fold,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_macro_f1": best_f1,
            "history": history,
            "args": vars(args),
            "label2id": LABEL2ID,
        },
        path,
    )


def make_loaders(
    dataset,
    train_idx,
    val_idx,
    args,
    device,
    seed,
):
    train_set = Subset(
        dataset,
        train_idx.tolist(),
    )

    val_set = Subset(
        dataset,
        val_idx.tolist(),
    )

    train_gen = torch.Generator().manual_seed(seed)

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": pickleball_collate_fn,
        "worker_init_fn": seed_worker,
        "drop_last": False,
    }

    if args.workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=train_gen,
        **kwargs,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **kwargs,
    )

    return train_loader, val_loader


def build_model(args, device):
    video_model = VideoMAEModel.from_pretrained(
        args.videomae
    )

    adapt_videomae_to_frames(
        video_model,
        NUM_VIDEO_FRAMES,
    )

    audio_model = load_beats(
        args.beats_checkpoint,
        args.beats_root,
    )

    model = AVModel(
        video_model=video_model,
        audio_model=audio_model,
        fusion_dim=768,
        cross_heads=16,
        temporal_heads=12,
        temporal_layers=2,
        temporal_dropout=0.1,
        head_hidden_dim=256,
        head_dropout=0.3,
        num_classes=NUM_CLASSES,
    )

    return model.to(device)


def train_fold(
    fold,
    train_idx,
    val_idx,
    dataset,
    args,
    device,
    use_amp,
    stratify_labels,
    users,
):
    fold_dir = (
        Path(args.output_dir)
        / f"fold_{fold}"
    )
    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_users = set(
        users[train_idx].tolist()
    )
    val_users = set(
        users[val_idx].tolist()
    )

    if train_users & val_users:
        raise RuntimeError(
            "USER LEAKAGE: "
            f"{sorted(train_users & val_users)}"
        )

    print(
        f"\n{'=' * 90}\n"
        f"FOLD {fold}/{args.folds}"
    )

    print(
        f"Train users: {sorted(train_users)}"
    )
    print(
        f"Val users: {sorted(val_users)}"
    )

    train_loader, val_loader = make_loaders(
        dataset,
        train_idx,
        val_idx,
        args,
        device,
        args.seed + fold,
    )

    alpha = compute_class_alpha(
        dataset,
        train_idx,
    )

    print(
        "Focal alpha:",
        {
            ID2LABEL[i]: round(
                alpha[i].item(),
                4,
            )
            for i in range(NUM_CLASSES)
        },
    )

    model = build_model(
        args,
        device,
    )

    # Pretrained backbones are frozen forever.
    model.freeze_pretrained_backbones()
    verify_trainable_parameters(model)

    shape_check(
        model,
        train_loader,
        device,
    )

    criterion = TemporalFocalLoss(
        alpha=tuple(alpha.tolist()),
        gamma=args.focal_gamma,
    ).to(device)

    optimizer = build_optimizer(model)

    steps_per_epoch = max(
        1,
        len(train_loader),
    )

    scheduler = build_scheduler(
        optimizer,
        args.epochs * steps_per_epoch,
        args.warmup_epochs * steps_per_epoch,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_f1 = -float("inf")
    stale = 0
    history = []

    for epoch in range(args.epochs):
        start = time.time()

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
            scaler,
            None,
            use_amp,
        )

        lr = max(
            group["lr"]
            for group in optimizer.param_groups
        )

        row = {
            "epoch": epoch + 1,
            "train": train_m,
            "val": val_m,
            "lr": lr,
            "minutes": (
                time.time() - start
            ) / 60,
        }

        history.append(row)

        with open(
            fold_dir / "history.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                history,
                f,
                indent=2,
            )

        current = val_m["macro_f1"]

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss={train_m['loss']:.4f} "
            f"F1={train_m['macro_f1']:.4f} | "
            f"val loss={val_m['loss']:.4f} "
            f"F1={current:.4f} "
            f"hit={val_m['f1_hit']:.4f} "
            f"miss={val_m['f1_miss']:.4f} | "
            f"lr={lr:.2e}"
        )

        save_checkpoint(
            fold_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_f1,
            fold,
            args,
            history,
        )

        if current > best_f1:
            best_f1 = current
            stale = 0

            save_checkpoint(
                fold_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_f1,
                fold,
                args,
                history,
            )

            print(
                f"NEW BEST | "
                f"macro-F1={best_f1:.5f}"
            )
        else:
            stale += 1

        if stale >= args.patience:
            print("Early stopping.")
            break

    result = {
        "fold": fold,
        "best_macro_f1": best_f1,
        "best_checkpoint": str(
            fold_dir / "best.pt"
        ),
    }

    with open(
        fold_dir / "result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
    )
    parser.add_argument(
        "--beats-checkpoint",
        default=BEATS_CHECKPOINT,
    )
    parser.add_argument(
        "--beats-root",
        default=BEATS_ROOT,
    )
    parser.add_argument(
        "--videomae",
        default=DEFAULT_VIDEOMAE,
    )
    parser.add_argument(
        "--output-dir",
        default=f"{VERSION_ARTIFACTS}/checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=N_SPLITS,
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=WARMUP_EPOCHS,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.folds < 2:
        raise ValueError(
            "--folds must be >= 2"
        )

    if (
        args.fold is not None
        and not 1 <= args.fold <= args.folds
    ):
        raise ValueError(
            f"--fold must be in [1, {args.folds}]"
        )

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

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        f"Device={device} | "
        f"AMP={use_amp} | "
        f"KFold={args.folds} | "
        f"Group=user"
    )
    print(
        "Event labeling: "
        f"TOTAL={EVENT_RADIUS_SEC * 2:.1f}s "
        f"(radius={EVENT_RADIUS_SEC:.1f}s)"
    )
    print(
        "Pretrained VideoMAE/BEATs: "
        "FROZEN + eval() PERMANENTLY"
    )
    print(
        "Optimizer: requires_grad=True ONLY"
    )
    print("=" * 90)

    video_paths, json_paths = (
        discover_video_json_pairs(
            args.data_dir
        )
    )

    dataset = PickleballAVDataset(
        video_paths=video_paths,
        json_paths=json_paths,
        num_video_frames=NUM_VIDEO_FRAMES,
        num_temporal_tokens=NUM_TEMPORAL_TOKENS,
        image_size=IMAGE_SIZE,
        audio_sample_rate=AUDIO_SR,
        overlap=OVERLAP,
        event_radius_sec=EVENT_RADIUS_SEC,
    )

    if not len(dataset):
        raise RuntimeError(
            "Dataset contains zero windows."
        )

    stratify_labels, users = (
        build_kfold_metadata(dataset)
    )

    unique_users = sorted(
        set(users.tolist())
    )

    if len(unique_users) < args.folds:
        raise RuntimeError(
            f"Need at least {args.folds} users; "
            f"found {len(unique_users)}"
        )

    class_groups = {}

    for class_id in range(NUM_CLASSES):
        class_groups[
            ID2LABEL[class_id]
        ] = len(
            set(
                users[
                    stratify_labels == class_id
                ].tolist()
            )
        )

    print(
        f"Videos={len(video_paths)} | "
        f"Windows={len(dataset)} | "
        f"Users={len(unique_users)}"
    )

    print(
        f"Users per stratification class: "
        f"{class_groups}"
    )

    for name, count in class_groups.items():
        if count < args.folds:
            print(
                f"[WARNING] {name} appears in only "
                f"{count} users; perfect "
                f"stratification across "
                f"{args.folds} folds is impossible."
            )

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    results = []

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        splitter.split(
            np.zeros(len(dataset)),
            stratify_labels,
            groups=users,
        ),
        1,
    ):
        if (
            args.fold is not None
            and fold != args.fold
        ):
            continue

        results.append(
            train_fold(
                fold,
                train_idx,
                val_idx,
                dataset,
                args,
                device,
                use_amp,
                stratify_labels,
                users,
            )
        )

    if results:
        scores = np.asarray(
            [
                r["best_macro_f1"]
                for r in results
            ],
            dtype=np.float64,
        )

        summary = {
            "folds": results,
            "mean_best_macro_f1": float(
                scores.mean()
            ),
            "std_best_macro_f1": float(
                scores.std()
            ),
        }

        with open(
            output_dir / "cv_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

        print(
            f"CV macro-F1: "
            f"{summary['mean_best_macro_f1']:.4f} "
            f"+/- "
            f"{summary['std_best_macro_f1']:.4f}"
        )

    dataset.close()


if __name__ == "__main__":
    main()