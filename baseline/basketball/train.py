"""v3 / v4 / v5 heads on frozen NSVA v2 features (extract_features.py), 5-fold game CV.

Folds are nsva_common.make_folds with --fold-seed 42 (independent of --seed): the same
fit/val/test games for every run.

The defaults are the final configuration (chosen on validation, see README.md):
VideoMAE self-supervised (--backbone ssl), mean + 2x2 grid tokens (--spatial grid), train on
full + flip / test on full (--views single), + the second-half view concatenated per token
(--tail), BEATs base, fixed text prompt (v4 / v5), fusion 384, weight decay 0.05,
token-drop 0.1, audio-dropout 0.2.
Other inputs present in the feature cache: --backbone k400, --views all, --clip-image, --beats ft.
Every input stream is standardised with mean/std of the fold's fit clips.
Per fold: best.pt, test probabilities (probs_fold{k}.npy) and indices (test_idx_fold{k}.npy).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nsva_common as nc  # noqa: E402

bc = nc.bc
from common.models import load_model_module  # noqa: E402




class Streams:
    """Memory-mapped v2 features -> standardised float tensors for (view, clip) pairs."""

    def __init__(self, root, args, device):
        root = Path(root)
        meta = json.load(open(root / "meta.json"))
        if not meta.get("complete"):
            raise RuntimeError(f"{root} is incomplete; run extract_features.py")
        self.meta, self.args, self.device = meta, args, device
        self.video = np.load(root / f"video_{args.backbone}.npy", mmap_mode="r")
        self.clip = np.load(root / "clip_b16.npy", mmap_mode="r") if args.clip_image else None
        self.audio = np.load(root / f"audio_{args.beats}.npy", mmap_mode="r")
        self.pad = np.load(root / "pad.npy")
        self.has_audio = np.load(root / "has_audio.npy")
        self.text = None
        if args.version != "v3":
            prompt = nc.encode_texts([args.prompt], device)[0]
            self.text = np.broadcast_to(prompt, (len(self.has_audio), prompt.size)).copy()
        self.stats = None

    def frame_part(self, view, idx):
        x = self.video[view, idx]  # [B,16,5,768]
        x = x[:, :, 0] if self.args.spatial == "mean" else x.reshape(*x.shape[:2], -1)
        parts = [x]
        if self.clip is not None:
            parts.append(self.clip[view, idx])
        return np.concatenate(parts, -1) if len(parts) > 1 else x

    def video_part(self, view, idx):
        x = self.frame_part(view, idx)
        if self.args.tail:
            x = np.concatenate([x, self.frame_part(nc.tail_view_for(view), idx)], -1)
        return x

    def fit_stats(self, fit_idx):
        sample = np.sort(np.random.default_rng(0).choice(fit_idx, min(3000, len(fit_idx)), replace=False))
        v = self.video_part(np.zeros(len(sample), dtype=np.int64), sample).astype(np.float32)
        v = v.reshape(-1, v.shape[-1])
        stats = {"video": (v.mean(0), v.std(0) + 1e-5)}
        a = self.audio[sample[self.has_audio[sample]]].astype(np.float32).reshape(-1, self.audio.shape[-1])
        stats["audio"] = (a.mean(0), a.std(0) + 1e-5)
        self.stats = {k: (torch.tensor(m), torch.tensor(s)) for k, (m, s) in stats.items()}

    @property
    def video_dim(self):
        return self.video_part(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)).shape[-1]

    def get(self, view, idx):
        order = np.argsort(idx, kind="stable")  # sorted memmap reads are much faster
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        idx_s, view_s = idx[order], view[order]

        def norm(x, key):
            m, s = self.stats[key]
            return ((torch.from_numpy(np.ascontiguousarray(x)).float() - m) / s)[inv]

        d = self.device
        batch = {
            "video": norm(self.video_part(view_s, idx_s), "video").to(d, non_blocking=True),
            "video_pad": torch.from_numpy(self.pad[view, idx]).to(d),
            "audio": norm(self.audio[idx_s], "audio").to(d, non_blocking=True),
            "has_audio": torch.from_numpy(self.has_audio[idx]).to(d),
        }
        if self.text is not None:
            batch["text"] = torch.from_numpy(self.text[idx]).to(d)
        return batch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=("v3", "v4", "v5"), required=True)
    p.add_argument("--run", required=True, help="run name (output folder)")
    p.add_argument("--backbone", choices=tuple(nc.BACKBONES), default="ssl")
    p.add_argument("--spatial", choices=("mean", "grid"), default="grid")
    p.add_argument("--views", choices=("single", "all"), default="single")
    p.add_argument("--no-tail", dest="tail", action="store_false", help="drop the second-half view")
    p.add_argument("--clip-image", action="store_true")
    p.add_argument("--beats", choices=tuple(nc.BEATS), default="base")
    p.add_argument("--prompt", default=nc.DEFAULT_PROMPT)
    p.add_argument("--features", default=nc.FEATURES)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=nc.SEED)
    p.add_argument("--seed", type=int, default=nc.SEED)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--fusion-dim", type=int, default=384)
    p.add_argument("--cross-heads", type=int, default=16)
    p.add_argument("--temporal-heads", type=int, default=12)
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--token-drop", type=float, default=0.1, help="zero this share of temporal tokens")
    p.add_argument("--audio-dropout", type=float, default=0.2, help="drop the audio of this share of clips")
    p.add_argument("--mixup", type=float, default=0.0, help="Beta(a, a) feature mixup; 0 = off")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights; 0 = off")
    p.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    args = p.parse_args()
    out = Path(nc.OUTPUT_ROOT) / args.run / args.version
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    st = Streams(args.features, args, device)
    labels_all = torch.tensor(st.meta["labels"])
    labels_np, games = labels_all.numpy(), np.asarray(st.meta["games"])
    train_views = np.array(nc.BASE_VIEWS if args.views == "all" else [0, 1])
    test_views = nc.BASE_VIEWS if args.views == "all" else [0]
    video_dim = st.video_dim
    m = load_model_module(args.version)
    common = dict(fusion_dim=args.fusion_dim, temporal_heads=args.temporal_heads,
                  temporal_layers=args.temporal_layers, dropout=args.dropout,
                  head_hidden_dim=args.head_hidden, head_dropout=args.head_dropout,
                  num_classes=nc.NUM_CLASSES)
    print(f"[MODEL] {args.version} video_dim={video_dim} | {args.run}", flush=True)

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

    def batch(view, idx, train):
        b = st.get(view, idx)
        on = b.pop("has_audio")
        if train and args.audio_dropout > 0 and args.version != "v5":
            on = on & (torch.rand(len(idx), device=device) >= args.audio_dropout)
        b["audio"] = b["audio"] * on[:, None, None]
        pad = torch.zeros(b["audio"].shape[:2], dtype=torch.bool, device=device)
        pad[~on, 1:] = True  # no audio: one zero token stays unmasked so attention is defined
        b["audio_pad"] = pad
        if train and args.token_drop > 0:
            drop = torch.rand(b["video"].shape[:2], device=device) < args.token_drop
            b["video"] = b["video"].masked_fill(drop[..., None], 0.0)
        return b

    def predict(model, idx):
        model.eval()
        probs = 0
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for v in test_views:
                outs = []
                for s in range(0, len(idx), 512):
                    chunk = idx[s:s + 512]
                    b = batch(np.full(len(chunk), v), chunk, False)
                    outs.append(bc.clip_logits(forward(model, b).float(), b["video_pad"]).float())
                probs = probs + torch.cat(outs).softmax(-1)
        return probs / len(test_views)

    def evaluate(model, idx):
        probs = predict(model, idx)
        y = labels_np[idx]
        cm = np.zeros((nc.NUM_CLASSES, nc.NUM_CLASSES), dtype=np.int64)
        np.add.at(cm, (y, probs.argmax(-1).cpu().numpy()), 1)
        r = nc.metrics(cm)
        r["loss"] = float(F.nll_loss(probs.clamp_min(1e-8).log(), torch.as_tensor(y, device=device)))
        return r, probs

    summary = []
    for fold, fit_idx, val_idx, test_idx in nc.make_folds(labels_np, games, args.folds, args.fold_seed):
        seed = args.seed + fold
        bc.seed_everything(seed)
        rng = np.random.default_rng(seed)
        st.fit_stats(fit_idx)
        fold_dir = out / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[FOLD {fold}] fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

        model = build().to(device)
        decay = [q for q in model.parameters() if q.ndim > 1]
        no_decay = [q for q in model.parameters() if q.ndim <= 1]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
        spe = math.ceil(len(fit_idx) / args.batch_size)
        total, warm = args.epochs * spe, args.warmup_epochs * spe
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        ema = None
        if args.ema > 0:
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
        best, best_epoch, stale = -1.0, 0, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(fit_idx)
            run_loss, cm = 0.0, np.zeros((nc.NUM_CLASSES, nc.NUM_CLASSES), dtype=np.int64)
            for s in range(0, len(order), args.batch_size):
                idx = order[s:s + args.batch_size]
                b = batch(rng.choice(train_views, len(idx)), idx, True)
                y = labels_all[idx].to(device)
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
            train_f1 = nc.metrics(cm)["macro_f1"]
            print(f"[FOLD {fold}] epoch {epoch:03d} train loss={run_loss / len(fit_idx):.4f} F1={train_f1:.4f} "
                  f"| val loss={val['loss']:.4f} F1={val['macro_f1']:.4f}", flush=True)
            if val["macro_f1"] > best:
                best, best_epoch, stale = val["macro_f1"], epoch, 0
                torch.save({"model": scored.state_dict(), "epoch": epoch, "val": val,
                            "label2id": nc.LABEL2ID, "args": vars(args)}, fold_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"[FOLD {fold}] early stopping at epoch {epoch}.", flush=True)
                    break

        model = build().to(device)
        model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)["model"])
        test, probs = evaluate(model, test_idx)
        np.save(out / f"probs_fold{fold}.npy", probs.cpu().numpy())
        np.save(out / f"test_idx_fold{fold}.npy", test_idx)
        print(f"[FOLD {fold}] best epoch {best_epoch}: val F1={best:.4f} | TEST acc={test['accuracy']:.4f} "
              f"F1={test['macro_f1']:.4f} {test['f1_by_class']}", flush=True)
        summary.append({"fold": fold, "best_epoch": best_epoch, "val_macro_f1": best,
                        "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
                        "test_f1_by_class": test["f1_by_class"], "confusion_matrix": test["confusion_matrix"]})
        del model, ema
        torch.cuda.empty_cache()

    f1 = np.array([r["test_macro_f1"] for r in summary])
    acc = np.array([r["test_accuracy"] for r in summary])
    report = {"run": args.run, "version": args.version, "args": vars(args), "folds": summary,
              "test_macro_f1_mean": float(f1.mean()), "test_macro_f1_std": float(f1.std(ddof=1)),
              "test_accuracy_mean": float(acc.mean())}
    json.dump(report, open(out / "kfold_summary.json", "w"), indent=2)
    print(f"\n[RESULT] {args.run} {args.version}: test macro-F1 = {f1.mean():.4f} ± {f1.std(ddof=1):.4f} "
          f"| acc = {acc.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
