"""v3 / v4 / v5 heads on frozen football features (extract_features.py, transcribe.py).

Fixed SoccerNet splits: fit on train (part3-6), pick the epoch on valid (part1), report test
(part2). Test is never used for any choice. Labels: goal / saved / off_target.

Inputs (the best setting of each version is in run.sh):
    --backbone k400|ssl     VideoMAE fine-tuned on Kinetics / self-supervised only
    --spatial mean|grid     global mean token, or mean + 2x2 grid (k400 only)
    --window full|focus|both
                            full : 16 tokens over the whole 15 s
                            focus: 16 tokens over [3 s, 11 s] (shot + outcome, 2x denser)
                            both : full and focus tokens concatenated on the feature axis
    --clip-image            concatenate the CLIP ViT-B/16 image embedding per token
    --beats base|ft         BEATs audio tokens (v3 / v4)
    --text prompt|asr|asr_mclip
                            v4 / v5 text: fixed prompt, Whisper transcript in CLIP text space
                            (English tokenizer) or multilingual CLIP text
Training: flips are a train-time augmentation, test = mean of plain + flipped view (--tta).
Feature arrays are loaded once into GPU memory when they fit (--gpu-cache-gb), so an epoch is
a few hundred milliseconds and many runs can share the GPU.

Output: <out-root>/<run>/<version>/result.json, probs_valid.npy, probs_test.npy (+ best.pt with --save-ckpt)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fb_common as fc  # noqa: E402
from common.models import load_model_module  # noqa: E402

bc = fc.bc


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=("v3", "v4", "v5"), required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out-root", default=fc.OUTPUT_ROOT)
    p.add_argument("--features", default=fc.FEATURES)
    p.add_argument("--backbone", choices=("k400", "ssl"), default="k400")
    p.add_argument("--spatial", choices=("mean", "grid"), default="mean")
    p.add_argument("--window", choices=("full", "focus", "both"), default="full")
    p.add_argument("--clip-image", action="store_true")
    p.add_argument("--beats", choices=("base", "ft"), default="base")
    p.add_argument("--text", choices=("prompt", "asr", "asr_mclip"), default="prompt")
    p.add_argument("--prompt", default=fc.DEFAULT_PROMPT)
    p.add_argument("--no-flip", dest="flip", action="store_false", help="no flip augmentation / TTA")
    p.add_argument("--seed", type=int, default=fc.SEED)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-weight", choices=("none", "sqrt", "balanced"), default="none")
    p.add_argument("--fusion-dim", type=int, default=768)
    p.add_argument("--cross-heads", type=int, default=16)
    p.add_argument("--temporal-heads", type=int, default=12)
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--token-drop", type=float, default=0.0, help="zero this share of temporal tokens")
    p.add_argument("--audio-dropout", type=float, default=0.0, help="drop the audio of this share of clips")
    p.add_argument("--text-dropout", type=float, default=0.0, help="replace the transcript by the prompt")
    p.add_argument("--mixup", type=float, default=0.0, help="Beta(a, a) feature mixup; 0 = off")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights; 0 = off")
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--gpu-cache-gb", type=float, default=1.2)
    p.add_argument("--device", default="cuda")
    return p


class Streams:
    """All feature arrays a run needs, as tensors on the GPU (or CPU when too large)."""

    def __init__(self, args, device):
        root = Path(args.features)
        meta = json.load(open(root / "meta.json"))
        if not meta.get("complete"):
            raise RuntimeError(f"{root} is incomplete; run extract_features.py")
        self.meta, self.args, self.device = meta, args, device
        if args.spatial == "grid" and args.backbone != "k400":
            raise ValueError("--spatial grid is stored for k400 only")
        views = {"full": [0, 1], "focus": [2, 3], "both": [0, 1, 2, 3]}[args.window]
        self.views = views
        name = f"video_{args.backbone}.npy"
        mm = np.load(root / name, mmap_mode="r")
        shape = (len(views), mm.shape[1], 16, 768 * (5 if args.spatial == "grid" else 1))
        need = np.prod(shape) * 2 / 2 ** 30
        # small inputs live on the GPU; large ones stay memory-mapped (the OS page cache is
        # shared by all parallel runs, so 8 runs do not need 8 GPU copies of 5-10 GB)
        self.store = device if need < args.gpu_cache_gb else torch.device("cpu")
        t0 = time.time()
        self.mm_video = self.mm_clip = None
        self.clip = None
        if self.store.type == "cpu":
            self.mm_video = mm
            self.video = torch.empty(shape, dtype=torch.float16, device="meta")  # shape only
            if args.clip_image:
                self.mm_clip = np.load(root / "clip_b16.npy", mmap_mode="r")
        else:
            if args.backbone == "k400" and args.spatial == "mean":
                arr = np.stack([np.ascontiguousarray(mm[v, :, :, 0]) for v in views])
            else:
                arr = np.stack([np.asarray(mm[v]).reshape(mm.shape[1], 16, -1) for v in views])
            self.video = torch.from_numpy(arr).to(self.store)
            del arr
            if args.clip_image:
                c = np.load(root / "clip_b16.npy", mmap_mode="r")
                self.clip = torch.from_numpy(np.stack([np.asarray(c[v]) for v in views])).to(self.store)
        self.pad = torch.from_numpy(np.load(root / "pad.npy")[views]).to(device)
        self.audio = torch.from_numpy(np.load(root / f"audio_{args.beats}.npy")).to(device)
        self.has_audio = torch.from_numpy(np.load(root / "has_audio.npy")).to(device)
        self.text = None
        if args.version != "v3":
            prompt = torch.from_numpy(fc.encode_texts([args.prompt], device)[0])
            n = len(meta["labels"])
            if args.text == "prompt":
                text = prompt.expand(n, -1).clone()
                self.has_text = torch.zeros(n, dtype=torch.bool)
            else:
                file = "text_clip.npy" if args.text == "asr" else "text_mclip.npy"
                while not (root / file).is_file():  # transcribe.py may still be running
                    print(f"[TEXT] waiting for {root / file}", flush=True)
                    time.sleep(60)
                time.sleep(5)
                text = torch.from_numpy(np.load(root / file).astype(np.float32))
                self.has_text = torch.from_numpy(np.load(root / "has_text.npy"))
                text[~self.has_text] = prompt
                print(f"[TEXT] {args.text}: transcripts for {int(self.has_text.sum())}/{n} clips", flush=True)
            self.text, self.prompt = text.to(device), prompt.to(device)
            self.has_text = self.has_text.to(device)
        print(f"[DATA] video {tuple(self.video.shape)} on {self.store} ({need:.1f} GB) in {time.time() - t0:.0f}s",
              flush=True)

    def fit_stats(self, fit_idx):
        sample = np.sort(np.random.default_rng(0).choice(fit_idx, min(3000, len(fit_idx)), replace=False))
        v = self.video_part(np.zeros(len(sample), dtype=np.int64), torch.as_tensor(sample)).float()
        v = v.reshape(-1, v.shape[-1])
        self.v_stats = (v.mean(0), v.std(0) + 1e-5)
        a = self.audio[torch.as_tensor(sample, device=self.device)]
        a = a[self.has_audio[torch.as_tensor(sample, device=self.device)]].float().reshape(-1, a.shape[-1])
        self.a_stats = (a.mean(0), a.std(0) + 1e-5)

    def frame_part(self, slot, idx):
        """slot: index into self.views per clip (numpy), idx: clip indices (tensor)."""
        if self.mm_video is not None:  # memory-mapped: sorted reads, then restore the order
            view = np.asarray(self.views)[np.asarray(slot)]
            ids = idx.cpu().numpy()
            order = np.lexsort((ids, view))
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            x = self.mm_video[view[order], ids[order]]
            if self.mm_video.ndim == 5 and self.args.spatial == "mean":
                x = x[:, :, 0]
            x = x.reshape(len(ids), 16, -1)
            if self.mm_clip is not None:
                x = np.concatenate([x, self.mm_clip[view[order], ids[order]]], -1)
            return torch.from_numpy(np.ascontiguousarray(x[inv])).to(self.device, non_blocking=True)
        slot_t = torch.as_tensor(slot, device=self.store)
        idx_t = idx.to(self.store)
        x = self.video[slot_t, idx_t]
        if self.clip is not None:
            x = torch.cat([x, self.clip[slot_t, idx_t]], -1)
        return x.to(self.device, non_blocking=True)

    def video_part(self, flip, idx):
        """flip: 0/1 per clip. With --window both, full and focus are concatenated (same flip)."""
        flip = np.asarray(flip)
        if self.args.window == "both":
            return torch.cat([self.frame_part(flip, idx), self.frame_part(2 + flip, idx)], -1)
        return self.frame_part(flip, idx)

    @property
    def video_dim(self):
        return self.video_part(np.zeros(1, dtype=np.int64), torch.zeros(1, dtype=torch.long)).shape[-1]

    def get(self, flip, idx):
        idx_d = idx.to(self.device)
        m, s = self.v_stats
        video = (self.video_part(flip, idx).float() - m) / s
        pad = self.pad[torch.as_tensor(np.asarray(flip), device=self.device), idx_d]
        if self.args.window == "both":
            pad = pad | self.pad[torch.as_tensor(2 + np.asarray(flip), device=self.device), idx_d]
        am, as_ = self.a_stats
        b = {"video": video, "video_pad": pad, "audio": (self.audio[idx_d].float() - am) / as_,
             "has_audio": self.has_audio[idx_d]}
        if self.text is not None:
            b["text"] = self.text[idx_d]
            b["has_text"] = self.has_text[idx_d]
        return b


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out_root) / args.run / args.version
    if (out / "result.json").is_file():
        print(f"[SKIP] {out} already done")
        return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    bc.seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    st = Streams(args, device)
    labels_np = fc.task_labels(st.meta["labels"])
    labels_all = torch.as_tensor(labels_np, device=device)
    splits = np.asarray(st.meta["splits"])
    fit_idx, val_idx, test_idx = (np.nonzero(splits == s)[0] for s in ("train", "valid", "test"))
    st.fit_stats(fit_idx)
    video_dim = st.video_dim
    m = load_model_module(args.version)
    common = dict(fusion_dim=args.fusion_dim, temporal_heads=args.temporal_heads,
                  temporal_layers=args.temporal_layers, dropout=args.dropout,
                  head_hidden_dim=args.head_hidden, head_dropout=args.head_dropout,
                  num_classes=fc.NUM_CLASSES)

    def build():
        if args.version == "v3":
            return m.AVFeatureModel(video_dim=video_dim, audio_dim=768, cross_heads=args.cross_heads, **common)
        if args.version == "v4":
            return m.AVTFeatureModel(video_dim=video_dim, audio_dim=768, text_dim=512,
                                     cross_heads=args.cross_heads, temporal_length=16, **common)
        return m.VideoTextFeatureModel(video_dim=video_dim, text_dim=512, temporal_length=16, **common)

    def forward(model, b):
        if args.version == "v3":
            return model(b["video"], b["audio"], b["audio_pad"], b["video_pad"])
        if args.version == "v4":
            text_pad = torch.zeros(len(b["video"]), 1, dtype=torch.bool, device=device)
            return model(b["video"], b["audio"], b["audio_pad"], b["video_pad"], b["text"][:, None], text_pad)
        return model(b["video"], b["text"], b["video_pad"])

    def batch(flip, idx, train):
        b = st.get(flip, torch.as_tensor(idx))
        on = b.pop("has_audio")
        if train and args.audio_dropout > 0 and args.version != "v5":
            on = on & (torch.rand(len(idx), device=device) >= args.audio_dropout)
        b["audio"] = b["audio"] * on[:, None, None]
        pad = torch.zeros(b["audio"].shape[:2], dtype=torch.bool, device=device)
        pad[~on, 1:] = True  # no audio: one zero token stays unmasked so attention is defined
        b["audio_pad"] = pad
        if "has_text" in b:
            has = b.pop("has_text")
            if train and args.text_dropout > 0:
                drop = has & (torch.rand(len(idx), device=device) < args.text_dropout)
                b["text"] = torch.where(drop[:, None], st.prompt.expand_as(b["text"]), b["text"])
        if train and args.token_drop > 0:
            drop = torch.rand(b["video"].shape[:2], device=device) < args.token_drop
            b["video"] = b["video"].masked_fill(drop[..., None], 0.0)
        return b

    test_flips = [0, 1] if args.flip else [0]

    def predict(model, idx):
        model.eval()
        probs = 0
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for f in test_flips:
                outs = []
                for s in range(0, len(idx), 1024):
                    chunk = idx[s:s + 1024]
                    b = batch(np.full(len(chunk), f), chunk, False)
                    outs.append(bc.clip_logits(forward(model, b).float(), b["video_pad"]).float())
                probs = probs + torch.cat(outs).softmax(-1)
        return probs / len(test_flips)

    def evaluate(model, idx):
        probs = predict(model, idx)
        y = labels_np[idx]
        cm = np.zeros((fc.NUM_CLASSES, fc.NUM_CLASSES), dtype=np.int64)
        np.add.at(cm, (y, probs.argmax(-1).cpu().numpy()), 1)
        r = fc.metrics(cm)
        r["loss"] = float(F.nll_loss(probs.clamp_min(1e-8).log(), torch.as_tensor(y, device=device)))
        return r, probs

    model = build().to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[MODEL] {args.version} video_dim={video_dim} params={n_params / 1e6:.1f}M | {args.run} "
          f"| fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
    decay = [q for q in model.parameters() if q.ndim > 1]
    no_decay = [q for q in model.parameters() if q.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    spe = math.ceil(len(fit_idx) / args.batch_size)
    total, warm = args.epochs * spe, args.warmup_epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    counts = np.bincount(labels_np[fit_idx], minlength=fc.NUM_CLASSES).astype(np.float64)
    weight = None
    if args.class_weight != "none":
        w = (counts.sum() / counts) ** (0.5 if args.class_weight == "sqrt" else 1.0)
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    ema = None
    if args.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)

    best, best_epoch, stale, best_state, history = -1.0, 0, 0, None, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(fit_idx)
        run_loss, cm = 0.0, np.zeros((fc.NUM_CLASSES, fc.NUM_CLASSES), dtype=np.int64)
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            flips = rng.integers(0, 2, len(idx)) if args.flip else np.zeros(len(idx), dtype=np.int64)
            b = batch(flips, idx, True)
            y = labels_all[idx]
            lam, perm = 1.0, None
            if args.mixup > 0:
                lam = float(rng.beta(args.mixup, args.mixup))
                perm = torch.randperm(len(idx), device=device)
                for key in ("video", "audio", "text"):
                    if key in b:
                        b[key] = lam * b[key] + (1 - lam) * b[key][perm]
                b["video_pad"] = b["video_pad"] & b["video_pad"][perm]
                b["audio_pad"] = b["audio_pad"] & b["audio_pad"][perm]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = bc.clip_logits(forward(model, b).float(), b["video_pad"]).float()
            loss = crit(logits, y) if perm is None else lam * crit(logits, y) + (1 - lam) * crit(logits, y[perm])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            run_loss += loss.item() * len(idx)
            np.add.at(cm, (y.cpu().numpy(), logits.argmax(-1).cpu().numpy()), 1)
        scored = ema.module if ema is not None else model
        val, _ = evaluate(scored, val_idx)
        train_f1 = fc.metrics(cm)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": run_loss / len(fit_idx), "train_f1": train_f1,
                        "val_loss": val["loss"], "val_f1": val["macro_f1"]})
        print(f"epoch {epoch:03d} train loss={run_loss / len(fit_idx):.4f} F1={train_f1:.4f} "
              f"| val loss={val['loss']:.4f} F1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best:
            best, best_epoch, stale = val["macro_f1"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in scored.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    final = build().to(device)
    final.load_state_dict(best_state)
    val, pv = evaluate(final, val_idx)
    test, pt = evaluate(final, test_idx)
    np.save(out / "probs_valid.npy", pv.cpu().numpy().astype(np.float32))
    np.save(out / "probs_test.npy", pt.cpu().numpy().astype(np.float32))
    if args.save_ckpt:
        torch.save({"model": best_state, "epoch": best_epoch, "val": val, "test": test,
                    "labels": list(fc.LABELS), "args": vars(args)}, out / "best.pt")
    result = {"run": args.run, "version": args.version, "args": vars(args), "best_epoch": best_epoch,
              "params_m": n_params / 1e6, "seconds": time.time() - start, "valid": val, "test": test,
              "history": history}
    json.dump(result, open(out / "result.json", "w"), indent=1)
    print(f"[RESULT] {args.run} {args.version}: epoch {best_epoch} | VAL F1={val['macro_f1']:.4f} "
          f"acc={val['accuracy']:.4f} | TEST F1={test['macro_f1']:.4f} acc={test['accuracy']:.4f} "
          f"{ {k: round(v, 3) for k, v in test['f1_by_class'].items()} } | {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
