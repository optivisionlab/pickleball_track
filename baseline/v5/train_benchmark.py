from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, VideoMAEModel

from config import (
	VERSION_ARTIFACTS,
	DEFAULT_CLIP,
	DEFAULT_VIDEOMAE,
	EPOCHS,
	FUSION_LR,
	GRAD_CLIP,
	ID2LABEL,
	IMAGENET_MEAN,
	IMAGENET_STD,
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
from model import TemporalFocalLoss, VideoTextFeatureModel
from dataset import BenchmarkDataset
from train import adapt_videomae_to_32_frames


DEFAULT_VIDEO_DIR = "/data/Cuong/pickleball/data/bendmark/train"
DEFAULT_LABELS_CSV = "/data/Cuong/pickleball/data/bendmark/playable_nsva_labels.csv"
FUSION_DIM = 768


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
	seed = torch.initial_seed() % (2**32)
	random.seed(seed)
	np.random.seed(seed)


@torch.inference_mode()
def precompute_video(dataset: BenchmarkDataset, model, device: torch.device, indices) -> None:
	model.eval()
	hidden = int(model.config.hidden_size)
	tubelet = int(model.config.tubelet_size)
	image_size = model.config.image_size
	patch_size = model.config.patch_size
	image_size = image_size if isinstance(image_size, int) else image_size[0]
	patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]
	spatial = (image_size // patch_size) ** 2
	temporal = NUM_VIDEO_FRAMES // tubelet
	for index in tqdm(indices, desc="Precompute VideoMAE"):
		item = dataset.cache_for_update(index)
		if item["video_features"] is not None:
			continue
		video = dataset.decode(index).unsqueeze(0).to(device)
		video = video.permute(0, 2, 1, 3, 4).contiguous()
		output = model(pixel_values=video).last_hidden_state
		expected = temporal * spatial
		if output.shape[1] != expected:
			raise RuntimeError(f"VideoMAE tokens={output.shape[1]}, expected={expected}.")
		item["video_features"] = output.reshape(1, temporal, spatial, hidden).mean(2)[0].float().cpu()
		torch.save(item, dataset.cache_path(index))


@torch.inference_mode()
def precompute_text(dataset: BenchmarkDataset, tokenizer, model, device: torch.device, indices) -> None:
	model.eval()
	for index in tqdm(indices, desc="Precompute CLIP text"):
		item = dataset.cache_for_update(index)
		if item["text_features"] is not None:
			continue
		encoded = tokenizer(
			dataset.metadata(index)["text"], return_tensors="pt",
			padding=True, truncation=True, max_length=77,
		)
		encoded = {key: value.to(device) for key, value in encoded.items()}
		output = model(**encoded)
		feature = output.pooler_output[0] if output.pooler_output is not None else output.last_hidden_state[0, -1]
		item["text_features"] = feature.float().cpu()
		torch.save(item, dataset.cache_path(index))


def validate_cache(dataset: BenchmarkDataset, indices) -> None:
	for index in indices:
		item = dataset.load_cache(index)
		if item.get("video_features") is None or item.get("text_features") is None:
			raise RuntimeError(f"Incomplete cache for window {index}.")
		if item["video_features"].shape[0] != NUM_TEMPORAL_TOKENS:
			raise RuntimeError(f"Invalid video feature shape in window {index}.")
		if not torch.isfinite(item["video_features"]).all() or not torch.isfinite(item["text_features"]).all():
			raise RuntimeError(f"Non-finite feature in window {index}.")
	print(f"[CACHE] Validated {len(indices)} windows.")


def collate_features(batch: List[dict]) -> dict:
	return {
		"video_features": torch.stack([item["video_features"] for item in batch]),
		"text_features": torch.stack([item["text_features"].reshape(-1) for item in batch]),
		"labels": torch.stack([item["labels"] for item in batch]),
		"temporal_padding_mask": torch.stack([item["temporal_padding_mask"].bool() for item in batch]),
	}


def compute_alpha(dataset: BenchmarkDataset, indices: np.ndarray) -> torch.Tensor:
	counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
	for index in indices:
		item = dataset.metadata(int(index))
		valid = ~item["temporal_padding_mask"]
		counts += torch.bincount(item["labels"][valid], minlength=NUM_CLASSES).double()
	missing = [ID2LABEL[i] for i in range(NUM_CLASSES) if counts[i] == 0]
	if missing:
		raise RuntimeError(f"Training split is missing classes: {missing}")
	alpha = torch.sqrt(counts.sum() / (NUM_CLASSES * counts))
	return (alpha / alpha.mean()).float()


def metrics(cm: np.ndarray) -> dict:
	tp = np.diag(cm).astype(np.float64)
	precision = tp / np.maximum(cm.sum(0), 1e-12)
	recall = tp / np.maximum(cm.sum(1), 1e-12)
	f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
	return {
		"accuracy": float(tp.sum() / max(cm.sum(), 1)),
		"macro_f1": float(f1.mean()),
		"f1_by_class": {ID2LABEL[i]: float(f1[i]) for i in range(NUM_CLASSES)},
	}


def flatten_epoch_metrics(epoch: int, train_metrics: dict, val_metrics: dict) -> dict:
	row: dict[str, object] = {"epoch": epoch}
	for split, split_metrics in (("train", train_metrics), ("val", val_metrics)):
		for name, value in split_metrics.items():
			key = f"{split}_{name}"
			row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
	return row


def run_epoch(model, loader, criterion, device, train, optimizer, scaler, scheduler, use_amp):
	model.train(train)
	total_loss = 0.0
	cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
	for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
		batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
		if train:
			optimizer.zero_grad(set_to_none=True)
		with torch.set_grad_enabled(train):
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
				logits = model(
					batch["video_features"], batch["text_features"],
					batch["temporal_padding_mask"],
				)
				loss = criterion(logits, batch["labels"], batch["temporal_padding_mask"])
			if not torch.isfinite(logits).all() or not torch.isfinite(loss):
				raise RuntimeError(
					f"Non-finite {'training' if train else 'validation'} output: "
					f"loss={loss.item()} logits_finite={torch.isfinite(logits).all().item()}"
				)
			if train:
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
				old_scale = scaler.get_scale()
				scaler.step(optimizer)
				scaler.update()
				if scaler.get_scale() >= old_scale:
					scheduler.step()
		total_loss += float(loss.detach())
		valid = ~batch["temporal_padding_mask"]
		pred = logits.argmax(-1)[valid].detach().cpu().numpy()
		target = batch["labels"][valid].detach().cpu().numpy()
		if len(target):
			cm += confusion_matrix(target, pred, labels=list(range(NUM_CLASSES)))
	result = metrics(cm)
	result["loss"] = total_loss / max(len(loader), 1)
	result["confusion_matrix"] = cm.tolist()
	return result


def choose_video_ids(dataset: BenchmarkDataset, max_ids: int | None, seed: int):
	id_to_label = {}
	for clip in dataset.clips:
		if clip.numeric_id in id_to_label and id_to_label[clip.numeric_id] != clip.label:
			raise ValueError(f"Conflicting labels for video{clip.numeric_id}.")
		id_to_label[clip.numeric_id] = clip.label
	if max_ids is None or max_ids >= len(id_to_label):
		return set(id_to_label)
	if max_ids < NUM_CLASSES * 2:
		raise ValueError("--max-video-ids must be at least 6 for a stratified split.")
	rng = np.random.default_rng(seed)
	by_class = {
		class_id: np.asarray([video_id for video_id, label in id_to_label.items() if label == class_id])
		for class_id in range(NUM_CLASSES)
	}
	if any(len(values) < 2 for values in by_class.values()):
		raise RuntimeError("Each class needs at least two video IDs.")
	selected = []
	for values in by_class.values():
		selected.extend(rng.permutation(values)[:max_ids // NUM_CLASSES].tolist())
	remaining = max_ids - len(selected)
	if remaining:
		available = [value for values in by_class.values() for value in values if value not in selected]
		selected.extend(rng.permutation(available)[:remaining].tolist())
	return set(selected)


def make_split(dataset: BenchmarkDataset, val_ratio: float, seed: int, selected_ids):
	id_to_label = {
		clip.numeric_id: clip.label for clip in dataset.clips
		if clip.numeric_id in selected_ids
	}
	ids = np.asarray(sorted(id_to_label), dtype=np.int64)
	labels = np.asarray([id_to_label[int(value)] for value in ids])
	unique, counts = np.unique(labels, return_counts=True)
	if len(unique) != NUM_CLASSES or np.any(counts < 2):
		raise RuntimeError(
			"Stratified train/validation split needs at least two distinct video IDs "
			"for each class; found "
			f"{ {ID2LABEL[int(k)]: int(v) for k, v in zip(unique, counts)} }."
		)
	train_ids, val_ids = train_test_split(
		ids, test_size=val_ratio, random_state=seed, stratify=labels
	)
	train_set, val_set = set(train_ids.tolist()), set(val_ids.tolist())
	train_indices = np.asarray([
		i for i, window in enumerate(dataset.windows)
		if dataset.clips[window.clip_index].numeric_id in train_set
	], dtype=np.int64)
	val_indices = np.asarray([
		i for i, window in enumerate(dataset.windows)
		if dataset.clips[window.clip_index].numeric_id in val_set
	], dtype=np.int64)
	return train_indices, val_indices, train_ids, val_ids


def parse_args():
	parser = argparse.ArgumentParser(description="Train v5 video+text benchmark classifier.")
	parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
	parser.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
	parser.add_argument("--videomae", default=DEFAULT_VIDEOMAE)
	parser.add_argument("--clip", default=DEFAULT_CLIP)
	parser.add_argument("--cache-dir", default=f"{VERSION_ARTIFACTS}/benchmark_feature_cache")
	parser.add_argument("--output-dir", default=f"{VERSION_ARTIFACTS}/checkpoints_benchmark")
	parser.add_argument("--epochs", type=int, default=EPOCHS)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--workers", type=int, default=NUM_WORKERS)
	parser.add_argument("--max-video-ids", type=int, default=None,
					 help="Optional stratified video-ID limit for smoke runs; default uses all IDs.")
	parser.add_argument("--val-ratio", type=float, default=0.2)
	parser.add_argument("--patience", type=int, default=PATIENCE)
	parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
	parser.add_argument("--lr", type=float, default=FUSION_LR)
	parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
	parser.add_argument("--focal-gamma", type=float, default=2.0)
	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--no-amp", action="store_true")
	parser.add_argument("--skip-precompute", action="store_true")
	parser.add_argument("--rebuild-cache", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
		raise ValueError("epochs/batch-size must be positive; workers must be nonnegative.")
	if not 0 < args.val_ratio < 1:
		raise ValueError("--val-ratio must be between 0 and 1.")
	seed_everything(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_amp = device.type == "cuda" and not args.no_amp
	cache_dir = Path(args.cache_dir)
	if args.rebuild_cache and cache_dir.exists():
		for path in cache_dir.glob("*.pt"):
			path.unlink()
	dataset = BenchmarkDataset(
		Path(args.video_dir), Path(args.labels_csv), cache_dir,
		args.videomae, args.clip,
	)
	selected_ids = choose_video_ids(dataset, args.max_video_ids, args.seed)
	active_indices = np.asarray([
		i for i, window in enumerate(dataset.windows)
		if dataset.clips[window.clip_index].numeric_id in selected_ids
	], dtype=np.int64)
	train_indices, val_indices, train_ids, val_ids = make_split(
		dataset, args.val_ratio, args.seed, selected_ids
	)
	print(f"Split by clip ID: train={len(train_ids)} videos/{len(train_indices)} windows, "
		  f"val={len(val_ids)} videos/{len(val_indices)} windows.")
	if args.dry_run:
		counts = {name: 0 for name in LABEL2ID}
		for clip in dataset.clips:
			counts[ID2LABEL[clip.label]] += 1
		print(f"Matched clip counts: {counts}")
		print("Dry run complete; no features extracted and no model trained.")
		return

	if not args.skip_precompute:
		videomae = VideoMAEModel.from_pretrained(args.videomae)
		adapt_videomae_to_32_frames(videomae)
		videomae = videomae.to(device).eval()
		for param in videomae.parameters():
			param.requires_grad_(False)
		precompute_video(dataset, videomae, device, active_indices)
		del videomae
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		tokenizer = CLIPTokenizer.from_pretrained(args.clip)
		clip_model = CLIPTextModel.from_pretrained(args.clip).to(device).eval()
		for param in clip_model.parameters():
			param.requires_grad_(False)
		precompute_text(dataset, tokenizer, clip_model, device, active_indices)
		del tokenizer, clip_model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	else:
		print("[CACHE] Precomputation skipped; checking existing features.")
	validate_cache(dataset, active_indices)

	generator = torch.Generator().manual_seed(args.seed)
	loader_args = dict(
		batch_size=args.batch_size, num_workers=args.workers,
		pin_memory=device.type == "cuda", collate_fn=collate_features,
		worker_init_fn=seed_worker, generator=generator, drop_last=False,
	)
	if args.workers > 0:
		loader_args.update(persistent_workers=True, prefetch_factor=2)
	train_loader = DataLoader(Subset(dataset, train_indices.tolist()), shuffle=True, **loader_args)
	val_loader = DataLoader(Subset(dataset, val_indices.tolist()), shuffle=False, **loader_args)
	sample = dataset.load_cache(int(train_indices[0]))
	model = VideoTextFeatureModel(
		video_dim=sample["video_features"].shape[-1],
		text_dim=sample["text_features"].numel(), fusion_dim=FUSION_DIM,
		temporal_heads=12, temporal_layers=2, dropout=0.1,
		head_hidden_dim=256, head_dropout=0.3, num_classes=NUM_CLASSES,
		temporal_length=NUM_TEMPORAL_TOKENS,
	).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	steps = max(1, len(train_loader))
	warmup_steps = args.warmup_epochs * steps
	total_steps = args.epochs * steps
	scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lambda step: (step + 1) / max(1, warmup_steps) if step < warmup_steps
		else 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))),
	)
	criterion = TemporalFocalLoss(
		alpha=tuple(compute_alpha(dataset, train_indices).tolist()), gamma=args.focal_gamma
	).to(device)
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	best_f1 = -float("inf")
	stale = 0
	history = []
	csv_history = []
	for epoch in range(args.epochs):
		train_metrics = run_epoch(model, train_loader, criterion, device, True, optimizer, scaler, scheduler, use_amp)
		val_metrics = run_epoch(model, val_loader, criterion, device, False, None, scaler, None, use_amp)
		row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
		history.append(row)
		csv_history.append(flatten_epoch_metrics(epoch + 1, train_metrics, val_metrics))
		with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
			writer = csv.DictWriter(handle, fieldnames=list(csv_history[0]))
			writer.writeheader()
			writer.writerows(csv_history)
		with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
			json.dump(history, handle, indent=2)
		print(f"Epoch {epoch + 1:03d}/{args.epochs} | train loss={train_metrics['loss']:.4f} "
			  f"F1={train_metrics['macro_f1']:.4f} | val loss={val_metrics['loss']:.4f} "
			  f"F1={val_metrics['macro_f1']:.4f}")
		torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
					"scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
					"args": vars(args), "metrics": row}, output_dir / "last.pt")
		if val_metrics["macro_f1"] > best_f1:
			best_f1 = val_metrics["macro_f1"]
			stale = 0
			torch.save({"epoch": epoch + 1, "model": model.state_dict(), "args": vars(args),
						"val_metrics": val_metrics, "label2id": LABEL2ID}, output_dir / "best.pt")
		else:
			stale += 1
			if stale >= args.patience:
				print("Early stopping.")
				break
	with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
		json.dump({"best_macro_f1": best_f1, "best_checkpoint": str(output_dir / "best.pt"),
				   "train_video_ids": train_ids.tolist(), "val_video_ids": val_ids.tolist()}, handle, indent=2)


if __name__ == "__main__":
	main()
