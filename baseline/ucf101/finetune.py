"""Stage 2: end-to-end fine-tuning of VideoMAE ViT-B on UCF-101 with the v3 head (video + audio).

    video : 16 frames (one random frame per segment, TSN-style) -> VideoMAE (trainable)
            -> 8 temporal tokens (spatial mean) -> v3 AVFeatureModel head
    audio : cached BEATs features (extract_features.py, frozen), modality dropout 0.3;
            --no-audio trains/tests the same model video-only (the setting of the papers).
    train : random resized crop + flip, AdamW, layer-wise lr decay, cosine, fixed epochs,
            bf16, gradient accumulation (--grad-ckpt if memory is short). Resumes from last.pt.
    test  : 2 temporal offsets x 3 square crops (left/center/right) = 6 views, mean softmax.

Run one split at a time: python finetune.py --backbone ssl --split 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg  # noqa: E402
import extract_features as ef  # noqa: E402
import ucf101_common as uc  # noqa: E402
from common.models import load_model_module  # noqa: E402  (ucf101_common adds the baseline root)

FRAMES, SIZE = 16, 224


def read_indices(path, wanted):
    cap = cv2.VideoCapture(path)
    got, pos, last = {}, -1, None
    for target in sorted(set(wanted)):
        while pos < target and cap.grab():
            pos += 1
        if pos == target:
            ok, frame = cap.retrieve()
            if ok:
                last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                got[target] = last
    cap.release()
    if last is None:
        return np.zeros((len(wanted), 240, 320, 3), np.uint8)
    out, prev = [], last
    for t in wanted:
        f = got.get(t)
        prev = f if f is not None and f.shape == last.shape else prev
        out.append(prev)
    return np.stack(out)


def frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


class TrainSet(Dataset):
    def __init__(self, rows, idx):
        self.rows, self.idx = rows, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        index = int(self.idx[i])
        path = self.rows[index].path
        n = frame_count(path)
        edges = np.linspace(0, n, FRAMES + 1)
        wanted = [min(n - 1, int(random.uniform(edges[s], max(edges[s], edges[s + 1] - 1e-6))))
                  for s in range(FRAMES)]
        frames = read_indices(path, wanted)
        h, w = frames.shape[1:3]
        for _ in range(10):  # random resized crop
            area = h * w * random.uniform(0.3, 1.0)
            ratio = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
            cw, ch = int(round(math.sqrt(area * ratio))), int(round(math.sqrt(area / ratio)))
            if cw <= w and ch <= h:
                break
        else:
            cw, ch = min(h, w), min(h, w)
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        flip = random.random() < 0.5
        out = np.empty((FRAMES, SIZE, SIZE, 3), np.uint8)
        for t in range(FRAMES):
            crop = cv2.resize(frames[t, y0:y0 + ch, x0:x0 + cw], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            out[t] = crop[:, ::-1] if flip else crop
        return index, torch.from_numpy(out)


class TestSet(Dataset):
    """6 views per clip: temporal offset {0, 0.5 segment} x crop {left, center, right}."""

    def __init__(self, rows, idx):
        self.rows, self.idx = rows, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        index = int(self.idx[i])
        path = self.rows[index].path
        n = frame_count(path)
        seg = n / FRAMES
        sets = [[min(n - 1, int(seg * s + seg * off)) for s in range(FRAMES)] for off in (0.25, 0.75)]
        frames = read_indices(path, sets[0] + sets[1])
        frames = frames.reshape(2, FRAMES, *frames.shape[1:])
        h, w = frames.shape[2:4]
        side = min(h, w)
        top = (h - side) // 2
        views = []
        for s in range(2):
            for left in (0, (w - side) // 2, w - side):
                views.append(np.stack([cv2.resize(f[top:top + side, left:left + side], (SIZE, SIZE),
                                                  interpolation=cv2.INTER_AREA) for f in frames[s]]))
        return index, torch.from_numpy(np.stack(views))


class FineTuneModel(nn.Module):
    def __init__(self, backbone_name, head, grad_ckpt=False):
        super().__init__()
        from transformers import VideoMAEModel
        self.backbone = VideoMAEModel.from_pretrained(backbone_name)
        if grad_ckpt:  # saves memory, costs ~30% extra compute
            self.backbone.gradient_checkpointing_enable()
        self.head = head

    def forward(self, pixels, audio, audio_pad):
        tok = self.backbone(pixel_values=pixels).last_hidden_state  # [B, 8*196, 768]
        b = tok.shape[0]
        video = tok.reshape(b, FRAMES // 2, -1, tok.shape[-1]).mean(2)
        pad = torch.zeros(b, FRAMES // 2, dtype=torch.bool, device=tok.device)
        return uc.bc.clip_logits(self.head(video, audio, audio_pad, pad).float(), pad)


def param_groups(model, base_lr, head_lr, layer_decay, wd):
    n_layers = len(model.backbone.encoder.layer)
    groups = {}
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            if ".embeddings." in name:
                layer = 0
            elif ".encoder.layer." in name:
                layer = int(name.split(".encoder.layer.")[1].split(".")[0]) + 1
            else:
                layer = n_layers + 1
            lr = base_lr * layer_decay ** (n_layers + 1 - layer)
        else:
            layer, lr = "head", head_lr
        decay = p.ndim > 1 and "position" not in name
        key = (layer, decay)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd if decay else 0.0})
        groups[key]["params"].append(p)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    return list(groups.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=("ssl", "k400"), required=True)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--beats", choices=("base", "ft"), default="ft")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--warmup-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (only if memory is short)")
    p.add_argument("--blr", type=float, default=1e-3, help="backbone lr = blr * effective batch / 256")
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--audio-dropout", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop after N optimizer steps")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    args = p.parse_args()

    name = f"ft_{args.backbone}_{'video' if args.no_audio else 'av'}"
    out = Path(cfg.OUTPUT_ROOT) / name / f"split{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    # One process per split: a second launcher (e.g. a parallel tmux lane) waits here.
    import fcntl
    lock = open(out / "run.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[FT] {out} is being trained by another process; waiting.", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
    if (out / "result.json").is_file() and not args.max_steps:
        print(f"[FT] {out}/result.json exists, skipping.")
        return
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed + args.split)
    uc.bc.seed_everything(args.seed + args.split)

    rows, classes, splits, audible = uc.load_dataset(cfg.VIDEO_ROOT, cfg.SPLIT_DIR, cfg.ANALYSIS_CSV)
    tr, te = splits[args.split]["train"], splits[args.split]["test"]
    feat = Path(cfg.FEATURES)
    audio_all = np.load(feat / f"audio_{args.beats}.npy", mmap_mode="r")
    has_audio = np.load(feat / "has_audio.npy") & audible.numpy()
    if args.no_audio:
        has_audio[:] = False
    sel = tr[has_audio[tr]]
    a = audio_all[np.sort(sel)].astype(np.float32).reshape(-1, audio_all.shape[-1]) if len(sel) else np.zeros((1, 768))
    a_mean = torch.tensor(a.mean(0), dtype=torch.float32)
    a_std = torch.tensor(a.std(0) + 1e-5, dtype=torch.float32)
    labels = torch.tensor([r.label for r in rows])

    def audio_batch(idx, train):
        x = (torch.from_numpy(audio_all[np.asarray(idx)].astype(np.float32)) - a_mean) / a_std
        on = torch.from_numpy(has_audio[np.asarray(idx)])
        if train and args.audio_dropout > 0:
            on = on & (torch.rand(len(idx)) >= args.audio_dropout)
        x = x * on[:, None, None]
        pad = torch.zeros(x.shape[:2], dtype=torch.bool)
        pad[~on, 1:] = True
        return x.to(device), pad.to(device)

    m = load_model_module("v3")
    head = m.AVFeatureModel(video_dim=768, audio_dim=768, fusion_dim=cfg.FUSION_DIM, cross_heads=cfg.CROSS_HEADS,
                            temporal_heads=cfg.TEMPORAL_HEADS, temporal_layers=cfg.TEMPORAL_LAYERS,
                            dropout=cfg.DROPOUT, head_hidden_dim=cfg.HEAD_HIDDEN, head_dropout=cfg.HEAD_DROPOUT,
                            num_classes=len(classes))
    model = FineTuneModel(ef.BACKBONES[args.backbone], head, args.grad_ckpt).to(device)
    eff_batch = args.batch_size * args.accum
    groups = param_groups(model, args.blr * eff_batch / 256, args.head_lr, args.layer_decay, args.weight_decay)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    loader = DataLoader(TrainSet(rows, tr), batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    steps_per_epoch = len(loader) // args.accum
    total, warm = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mean = torch.tensor(uc.bc.IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(uc.bc.IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    def to_pixels(frames):
        return (frames.to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255.0) - mean) / std

    start_epoch, step = 1, 0
    ckpt_path = out / "last.pt"
    if ckpt_path.is_file() and not args.max_steps:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        print(f"[FT] resumed from epoch {ck['epoch']}", flush=True)
    print(f"[FT] {name} split{args.split}: train={len(tr)} test={len(te)} eff_batch={eff_batch} "
          f"backbone_lr={args.blr * eff_batch / 256:.2e} steps/epoch={steps_per_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (idx, frames) in enumerate(loader):
            audio, apad = audio_batch(idx.numpy(), True)
            y = labels[idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(to_pixels(frames), audio, apad)
                loss = crit(logits.float(), y) / args.accum
            loss.backward()
            run_loss += loss.item() * args.accum * len(idx)
            correct += int((logits.argmax(-1) == y).sum())
            seen += len(idx)
            if (i + 1) % args.accum == 0:
                scale = (step + 1) / max(1, warm) if step < warm else \
                    0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * scale
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if args.max_steps and step >= args.max_steps:
                    print(f"[FT] debug: {step} steps, {seen / (time.time() - t0):.1f} clips/s, "
                          f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, loss {run_loss / seen:.3f}")
                    return
        print(f"[FT] epoch {epoch:03d} loss={run_loss / seen:.4f} train@1={correct / seen:.4f} "
              f"({seen / (time.time() - t0):.1f} clips/s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step},
                   out / "last.tmp")
        (out / "last.tmp").replace(ckpt_path)

    # Test: 6 views per clip, mean softmax.
    model.eval()
    test_loader = DataLoader(TestSet(rows, te), batch_size=4, num_workers=args.workers, prefetch_factor=4)
    probs, order = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, views in test_loader:
            b, v = views.shape[:2]
            audio, apad = audio_batch(np.repeat(idx.numpy(), v), False)
            logits = model(to_pixels(views.reshape(b * v, *views.shape[2:])), audio, apad)
            probs.append(logits.float().softmax(-1).reshape(b, v, -1).mean(1).cpu())
            order.append(idx)
    probs, order = torch.cat(probs), torch.cat(order)
    y = labels[order]
    result = uc.metrics(probs.log(), y, len(classes))
    np.save(out / "probs.npy", probs.numpy())
    np.save(out / "labels.npy", y.numpy())
    torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args)}, out / "final.pt")
    ckpt_path.unlink(missing_ok=True)
    result = {"split": args.split, "name": name, **{k: v for k, v in result.items() if k != "per_class_acc"}}
    json.dump(result, open(out / "result.json", "w"), indent=2)
    (out / "run.lock").unlink(missing_ok=True)
    print(f"[FT RESULT] {name} split{args.split}: top1={result['top1']:.4f} top5={result['top5']:.4f}", flush=True)


if __name__ == "__main__":
    main()
