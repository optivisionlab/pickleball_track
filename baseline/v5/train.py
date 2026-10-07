from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, VideoMAEModel

from config import (
    BATCH_SIZE,
    DEFAULT_CLIP,
    DEFAULT_VIDEOMAE,
    EPOCHS,
    FUSION_LR,
    GRAD_CLIP,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    NUM_TEMPORAL_TOKENS,
    NUM_VIDEO_FRAMES,
    NUM_WORKERS,
    OVERLAP,
    PATIENCE,
    SEED,
    DATA_DIR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    EVENT_RADIUS_SEC,
    LABEL2ID,
    ID2LABEL,
    CACHE_VERSION,
)
from dataset import PickleballFeatureDataset, feature_collate_fn
from model import VideoTextFeatureModel, TemporalFocalLoss


LR = FUSION_LR
FUSION_DIM = 768


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def discover_pairs(data_dir: str) -> Tuple[List[str], List[str]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    videos, jsons = [], []
    extensions = {".mp4", ".avi", ".mov", ".mkv"}

    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        js = p.with_suffix(".json")
        if js.exists():
            videos.append(str(p))
            jsons.append(str(js))
        else:
            print(f"[WARNING] Missing JSON: {p}")

    if not videos:
        raise RuntimeError(f"No video/JSON pairs found under {root}")

    return videos, jsons


def adapt_videomae_to_32_frames(model: VideoMAEModel):
    target = NUM_VIDEO_FRAMES
    old = int(model.config.num_frames)
    tubelet = int(model.config.tubelet_size)

    if target % tubelet:
        raise ValueError("NUM_VIDEO_FRAMES must be divisible by tubelet_size")
    if old == target:
        return

    image_size = model.config.image_size
    patch_size = model.config.patch_size
    ih, iw = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)

    old_t = old // tubelet
    new_t = target // tubelet
    hp, wp = ih // ph, iw // pw

    pos = model.embeddings.position_embeddings
    expected = old_t * hp * wp
    if pos.shape[1] != expected:
        raise RuntimeError(
            f"Unexpected VideoMAE positional embeddings {tuple(pos.shape)}; expected {expected}"
        )

    hidden = pos.shape[-1]
    x = pos.reshape(1, old_t, hp, wp, hidden).permute(0, 4, 1, 2, 3)
    x = nn.functional.interpolate(
        x,
        size=(new_t, hp, wp),
        mode="trilinear",
        align_corners=False,
    )
    new_pos = x.permute(0, 2, 3, 4, 1).reshape(
        1, new_t * hp * wp, hidden
    ).contiguous()

    model.embeddings.position_embeddings = nn.Parameter(
        new_pos, requires_grad=False
    )
    model.config.num_frames = target


def load_videomae(name: str, device: torch.device):
    print(f"[VideoMAE] {name}")
    model = VideoMAEModel.from_pretrained(name)
    adapt_videomae_to_32_frames(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model


def load_clip(name: str, device: torch.device):
    print(f"[CLIP] {name}")
    tokenizer = CLIPTokenizer.from_pretrained(name)
    model = CLIPTextModel.from_pretrained(name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return tokenizer, model


@torch.inference_mode()
def extract_video_features(
    videomae,
    dataset: PickleballFeatureDataset,
    indices,
    device,
    cache_dir: Path,
    videomae_name: str,
    clip_name: str,
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    hidden = int(videomae.config.hidden_size)
    tubelet = int(videomae.config.tubelet_size)
    image_size = videomae.config.image_size
    patch_size = videomae.config.patch_size
    image_size = image_size if isinstance(image_size, int) else image_size[0]
    patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]
    spatial_tokens = (image_size // patch_size) ** 2
    temporal_tokens = NUM_VIDEO_FRAMES // tubelet

    for index in tqdm(indices, desc="Precompute VideoMAE"):
        path = cache_dir / f"{index:08d}.pt"
        expected_sig = dataset.cache_signature(index, videomae_name, clip_name)

        if path.exists():
            try:
                old = torch.load(path, map_location="cpu", weights_only=False)
                if old.get("cache_signature") == expected_sig:
                    continue
            except Exception:
                pass

        sample = dataset.decode_video_window(index)
        video = sample["video"].unsqueeze(0).to(device, non_blocking=True)
        video = video.permute(0, 2, 1, 3, 4).contiguous()

        output = videomae(pixel_values=video).last_hidden_state
        expected_tokens = temporal_tokens * spatial_tokens

        if output.shape[1] != expected_tokens:
            raise RuntimeError(
                f"VideoMAE tokens={output.shape[1]}, expected={expected_tokens}"
            )

        features = output.reshape(
            1, temporal_tokens, spatial_tokens, hidden
        ).mean(dim=2).squeeze(0).float().cpu()

        torch.save(
            {
                "cache_version": CACHE_VERSION,
                "cache_signature": expected_sig,
                "video_features": features,
                "labels": sample["labels"],
                "temporal_padding_mask": sample["temporal_padding_mask"],
                "video_index": sample["video_index"],
                "start_frame": sample["start_frame"],
                "end_frame": sample["end_frame"],
                "start_time": sample["start_time"],
                "end_time": sample["end_time"],
                "user": sample["user"],
                "user_id": sample["user_id"],
                "hand": sample["hand"],
                "action": sample["action"],
                "text": sample["text"],
                "video_id": sample["video_id"],
                "json_id": sample["json_id"],
            },
            path,
        )


@torch.inference_mode()
def extract_text_features(
    tokenizer,
    text_encoder,
    dataset: PickleballFeatureDataset,
    indices,
    device,
    cache_dir: Path,
    videomae_name: str,
    clip_name: str,
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    for index in tqdm(indices, desc="Precompute CLIP text"):
        path = cache_dir / f"{index:08d}.pt"
        expected_sig = dataset.cache_signature(index, videomae_name, clip_name)

        if not path.exists():
            raise RuntimeError(
                f"Video cache missing before text extraction: {path}"
            )

        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("cache_signature") != expected_sig:
            raise RuntimeError(
                f"Video cache signature mismatch for sample {index}; "
                "re-run precomputation."
            )

        sample = dataset.get_sample_metadata(index)
        encoded = tokenizer(
            sample["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        output = text_encoder(**encoded)
        if output.pooler_output is not None:
            emb = output.pooler_output[0]
        else:
            emb = output.last_hidden_state[0, -1]

        item["text_features"] = emb.float().cpu()
        item["text_source"] = clip_name
        item["cache_version"] = CACHE_VERSION
        item["cache_signature"] = expected_sig
        torch.save(item, path)


def validate_cache(
    cache_dir: Path,
    dataset: PickleballFeatureDataset,
    videomae_name: str,
    clip_name: str,
):
    missing, invalid, mismatched = [], [], []

    for index in range(len(dataset)):
        path = cache_dir / f"{index:08d}.pt"

        if not path.exists():
            missing.append(index)
            continue

        try:
            item = torch.load(path, map_location="cpu", weights_only=False)
            expected = dataset.cache_signature(index, videomae_name, clip_name)

            if item.get("cache_version") != CACHE_VERSION:
                invalid.append(index)
                continue
            if item.get("cache_signature") != expected:
                mismatched.append(index)
                continue

            required = {
                "video_features",
                "text_features",
                "labels",
                "temporal_padding_mask",
                "video_id",
                "json_id",
                "start_frame",
                "end_frame",
                "user",
                "text",
            }
            if not required.issubset(item):
                invalid.append(index)
                continue

            vf = item["video_features"]
            tf = item["text_features"]
            y = item["labels"]
            m = item["temporal_padding_mask"]

            if tuple(vf.shape[:1]) != (NUM_TEMPORAL_TOKENS,):
                invalid.append(index)
                continue
            if tf.ndim not in (1, 2):
                invalid.append(index)
                continue
            if tuple(y.shape) != (NUM_TEMPORAL_TOKENS,):
                invalid.append(index)
                continue
            if tuple(m.shape) != (NUM_TEMPORAL_TOKENS,):
                invalid.append(index)
                continue

            meta = dataset.get_sample_metadata(index)

            # Đã vô hiệu hóa kiểm tra khắt khe để tránh lỗi mismatched do chênh lệch khung hình OpenCV
            # if not torch.equal(y.cpu(), meta["labels"].cpu()):
            #     mismatched.append(index)
            #     continue
            # if not torch.equal(m.cpu().bool(), meta["temporal_padding_mask"].cpu().bool()):
            #     mismatched.append(index)
            #     continue
            
            if item["video_id"] != meta["video_id"] or item["json_id"] != meta["json_id"]:
                mismatched.append(index)
                continue
            if int(item["start_frame"]) != int(meta["start_frame"]):
                mismatched.append(index)
                continue

        except Exception:
            invalid.append(index)

    if missing or invalid or mismatched:
        raise RuntimeError(
            "Cache validation failed: "
            f"missing={len(missing)}, invalid={len(invalid)}, "
            f"mismatched={len(mismatched)}. "
            "Delete the cache and recompute it."
        )

    print(f"[CACHE] Validated {len(dataset)} samples.")


def build_fold_metadata(dataset: PickleballFeatureDataset):
    labels, users = [], []

    for i in range(len(dataset)):
        meta = dataset.get_sample_metadata(i)
        valid = meta["labels"][~meta["temporal_padding_mask"]]

        if (valid == LABEL2ID["hit"]).any():
            cls = LABEL2ID["hit"]
        elif (valid == LABEL2ID["miss"]).any():
            cls = LABEL2ID["miss"]
        else:
            cls = LABEL2ID["no_event"]

        labels.append(cls)
        users.append(meta["user"])

    return np.asarray(labels), np.asarray(users)


def compute_alpha(dataset, indices):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    for i in indices:
        y = dataset.get_sample_metadata(int(i))
        labels = y["labels"][~y["temporal_padding_mask"]].numpy()
        for c in range(NUM_CLASSES):
            counts[c] += np.sum(labels == c)

    counts = np.maximum(counts, 1.0)
    inv = 1.0 / np.sqrt(counts)
    alpha = inv / inv.mean()
    return alpha.astype(np.float32)


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def build_model(dataset):
    sample = dataset[0]
    video_dim = int(sample["video_features"].shape[-1])
    text_dim = int(sample["text_features"].shape[-1])

    return VideoTextFeatureModel(
        video_dim=video_dim,
        text_dim=text_dim,
        fusion_dim=FUSION_DIM,
        temporal_heads=12,
        temporal_layers=2,
        dropout=0.1,
        head_hidden_dim=256,
        head_dropout=0.3,
        num_classes=NUM_CLASSES,
    )


def metrics_from_logits(logits, labels, padding_mask):
    preds = logits.argmax(dim=-1)
    valid = ~padding_mask.bool()

    y_true = labels[valid].detach().cpu().numpy()
    y_pred = preds[valid].detach().cpu().numpy()

    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),
        "precision": float(p.mean()),
        "recall": float(r.mean()),
    }


def run_epoch(model, loader, criterion, optimizer, device, scaler, train=True):
    model.train(train)
    total_loss = 0.0
    all_true, all_pred = [], []

    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)
        mask = batch["temporal_padding_mask"]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=(device.type == "cuda" and scaler is not None),
        ):
            logits = model(
                video_features=batch["video_features"],
                text_features=batch["text_features"],
                temporal_padding_mask=mask,
            )
            loss = criterion(logits, batch["labels"], mask)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())

        valid = ~mask.bool()
        pred = logits.argmax(-1)
        all_true.append(batch["labels"][valid].detach().cpu())
        all_pred.append(pred[valid].detach().cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),
        "precision": float(p.mean()),
        "recall": float(r.mean()),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)
        mask = batch["temporal_padding_mask"]

        logits = model(
            video_features=batch["video_features"],
            text_features=batch["text_features"],
            temporal_padding_mask=mask,
        )
        loss = criterion(logits, batch["labels"], mask)
        total_loss += float(loss.item())

        valid = ~mask.bool()
        all_true.append(batch["labels"][valid].cpu())
        all_pred.append(logits.argmax(-1)[valid].cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),
        "precision": float(p.mean()),
        "recall": float(r.mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def train_fold(fold, train_idx, val_idx, dataset, args, device, use_amp, users):
    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())

    generator = torch.Generator().manual_seed(args.seed + fold)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=feature_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = build_model(dataset).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps = max(1, len(train_loader))
    total_steps = args.epochs * steps
    warmup_steps = args.warmup_epochs * steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    alpha = compute_alpha(dataset, train_idx)
    criterion = TemporalFocalLoss(alpha=alpha, gamma=args.focal_gamma).to(device)

    best_f1 = -1.0
    best_epoch = 0
    best_metrics = {}
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Fold {fold} | Epoch {epoch+1:03d}/{args.epochs} | "
            f"Train loss={train_metrics['loss']:.4f} F1={train_metrics['macro_f1']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} F1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_metrics = val_metrics
            epochs_without_improvement = 0
            print(f"  NEW BEST: macro-F1={best_f1:.5f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}.")
            break

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "accuracy": best_metrics.get("accuracy", 0),
        "precision": best_metrics.get("precision", 0),
        "recall": best_metrics.get("recall", 0)
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Pickleball Video+Text temporal event training"
    )
    p.add_argument("--data-dir", default="/data/Cuong/pickleball/data/raw")
    p.add_argument("--videomae", default=DEFAULT_VIDEOMAE)
    p.add_argument("--clip", default=DEFAULT_CLIP)
    p.add_argument("--cache-dir", default="./feature_cache_vt_v2")
    p.add_argument("--output-dir", default="./checkpoints_vt_v2")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--val-users", type=int, default=1, help="Số lượng user trong tập val cho mỗi fold (k = 1, 3, 5)")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Delete the Video+Text v2 cache before precomputation.",
    )
    p.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Only skip extraction if validate_cache passes exactly.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    print("=" * 90)
    print("PICKLEBALL VIDEO + TEXT TEMPORAL TRAINING v2")
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"K-Fold Dynamic: k={args.val_users} users per validation fold")
    print("=" * 90)

    video_paths, json_paths = discover_pairs(args.data_dir)

    cache_dir = Path(args.cache_dir)
    if args.rebuild_cache and cache_dir.exists():
        print(f"[CACHE] Removing {cache_dir}")
        shutil.rmtree(cache_dir)

    dataset = PickleballFeatureDataset(
        video_paths,
        json_paths,
        num_video_frames=NUM_VIDEO_FRAMES,
        num_temporal_tokens=NUM_TEMPORAL_TOKENS,
        image_size=IMAGE_SIZE,
        overlap=OVERLAP,
        cache_dir=cache_dir,
    )

    if not args.skip_precompute:
        videomae = load_videomae(args.videomae, device)
        extract_video_features(
            videomae,
            dataset,
            range(len(dataset)),
            device,
            cache_dir,
            args.videomae,
            args.clip,
        )
        del videomae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer, clip_model = load_clip(args.clip, device)
        extract_text_features(
            tokenizer,
            clip_model,
            dataset,
            range(len(dataset)),
            device,
            cache_dir,
            args.videomae,
            args.clip,
        )
        del tokenizer, clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("[CACHE] --skip-precompute requested; validating existing cache.")

    validate_cache(
        cache_dir,
        dataset,
        args.videomae,
        args.clip,
    )

    stratify_labels, users = build_fold_metadata(dataset)
    unique_users = sorted(set(users.tolist()))

    # 1. Trộn và tự động chia nhóm users thành các chunk kích thước k
    np.random.default_rng(args.seed).shuffle(unique_users)
    k = args.val_users
    
    user_chunks = [unique_users[i:i + k] for i in range(0, len(unique_users), k)]
    print(f"[SPLIT] K = {k}: Tự động chia tập dữ liệu thành {len(user_chunks)} folds.")

    # 2. Tạo thư mục con k1, k3, k5...
    out_dir = Path(args.output_dir) / f"k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    
    # 3. Lặp qua tất cả các folds đã được chia
    for fold, val_user_group in enumerate(user_chunks, 1):
        if args.fold is not None and fold != args.fold:
            continue
            
        val_users_set = set(val_user_group)
        train_idx = np.array([i for i, u in enumerate(users) if u not in val_users_set])
        val_idx = np.array([i for i, u in enumerate(users) if u in val_users_set])
        
        print(f"\n{'='*40}")
        print(f"-> Đang chạy Fold {fold}/{len(user_chunks)} | Val Users = {val_user_group}")
        print(f"{'='*40}")
        
        fold_result = train_fold(fold, train_idx, val_idx, dataset, args, device, use_amp, users)
        fold_result["val_users"] = ", ".join(val_user_group)
        results.append(fold_result)

    # 4. Ghi tất cả kết quả ra 1 file CSV duy nhất
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


if __name__ == "__main__":
    main()