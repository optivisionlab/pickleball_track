"""Stage 1: v3 / v4 / v5 heads on frozen v2 features (extract_features.py), official 3 splits.

Switches (one ablation step each):
    --backbone ssl|k400        VideoMAE without / with Kinetics labels
    --spatial mean|grid        global mean token, or global mean + 2x2 grid (5 x 768)
    --views single|all         single: train on full+flip, test on full only
                               all:    train on a random view per clip, test = mean over 6 views
    --clip-image               add the CLIP ViT-B/16 image embedding to every temporal token
    --beats base|ft            BEATs pretrained / fine-tuned on AudioSet
Every input stream is standardised with mean/std of the split's training clips.
Test probabilities are saved (probs_split{k}.npy) for ensembling.
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
import config as cfg  # noqa: E402
import ucf101_common as uc  # noqa: E402
from common.models import load_model_module  # noqa: E402  (ucf101_common adds the baseline root)

FEATURES = cfg.FEATURES
ALL_VIEWS = [0, 1, 2, 3, 4, 5]


class Streams:
    """Memory-mapped v2 features; returns standardised float tensors for (view, clip) pairs."""

    def __init__(self, root, args):
        root = Path(root)
        meta = json.load(open(root / "meta.json"))
        if not meta.get("complete"):
            raise RuntimeError(f"{root} is incomplete; run extract_features.py")
        self.meta = meta
        self.video = np.load(root / f"video_{args.backbone}.npy", mmap_mode="r")
        self.clip = np.load(root / "clip_b16.npy", mmap_mode="r") if args.clip_image else None
        self.audio = np.load(root / f"audio_{args.beats}.npy", mmap_mode="r")
        self.pad = np.load(root / "pad.npy")
        self.has_audio = np.load(root / "has_audio.npy")
        self.spatial = args.spatial
        self.stats = None

    def video_part(self, view, idx):
        x = self.video[view, idx]  # [B,16,5,768]
        x = x[:, :, 0] if self.spatial == "mean" else x.reshape(*x.shape[:2], -1)
        return x

    def fit_stats(self, train_idx):
        sample = np.sort(np.random.default_rng(0).choice(train_idx, min(3000, len(train_idx)), replace=False))
        v = self.video_part(np.zeros(len(sample), dtype=np.int64), sample).astype(np.float32)
        stats = {"video": (v.reshape(-1, v.shape[-1]).mean(0), v.reshape(-1, v.shape[-1]).std(0) + 1e-5)}
        if self.clip is not None:
            c = self.clip[0, sample].astype(np.float32).reshape(-1, self.clip.shape[-1])
            stats["clip"] = (c.mean(0), c.std(0) + 1e-5)
        with_audio = sample[self.has_audio[sample]]
        a = self.audio[with_audio].astype(np.float32).reshape(-1, self.audio.shape[-1])
        stats["audio"] = (a.mean(0), a.std(0) + 1e-5)
        self.stats = {k: (torch.tensor(m), torch.tensor(s)) for k, (m, s) in stats.items()}

    def get(self, view, idx, device):
        order = np.argsort(idx, kind="stable")  # sorted reads are much faster on a memmap
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        idx_s, view_s = idx[order], view[order]

        def norm(x, key):
            m, s = self.stats[key]
            return ((torch.from_numpy(np.ascontiguousarray(x)).float() - m) / s)[inv]

        video = norm(self.video_part(view_s, idx_s), "video")
        if self.clip is not None:
            video = torch.cat([video, norm(self.clip[view_s, idx_s], "clip")], -1)
        audio = norm(self.audio[idx_s], "audio")
        return {
            "video": video.to(device, non_blocking=True),
            "video_pad": torch.from_numpy(self.pad[view, idx]).to(device),
            "audio": audio.to(device, non_blocking=True),
            "has_audio": torch.from_numpy(self.has_audio[idx]).to(device),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=("v3", "v4", "v5"), required=True)
    p.add_argument("--run", required=True, help="run name (output folder)")
    p.add_argument("--backbone", choices=("ssl", "k400"), default="k400")
    p.add_argument("--spatial", choices=("mean", "grid"), default="grid")
    p.add_argument("--views", choices=("single", "all"), default="all")
    p.add_argument("--clip-image", action="store_true")
    p.add_argument("--beats", choices=("base", "ft"), default="base")
    p.add_argument("--features", default=FEATURES)
    p.add_argument("--splits", type=int, nargs="+", default=list(cfg.SPLITS))
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--warmup-epochs", type=int, default=cfg.WARMUP_EPOCHS)
    p.add_argument("--patience", type=int, default=cfg.PATIENCE)
    p.add_argument("--label-smoothing", type=float, default=cfg.LABEL_SMOOTHING)
    p.add_argument("--token-drop", type=float, default=cfg.TOKEN_DROP)
    p.add_argument("--audio-dropout", type=float, default=None)
    p.add_argument("--val-groups", type=int, default=cfg.VAL_GROUPS)
    p.add_argument("--no-refit", dest="refit", action="store_false")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    args = p.parse_args()
    if args.audio_dropout is None:
        args.audio_dropout = 0.0 if args.version == "v5" else cfg.AUDIO_DROPOUT
    out = Path(cfg.OUTPUT_ROOT) / args.run / args.version
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    uc.bc.seed_everything(args.seed)
    rows, classes, splits, audible = uc.load_dataset(cfg.VIDEO_ROOT, cfg.SPLIT_DIR, cfg.ANALYSIS_CSV)
    st = Streams(args.features, args)
    if audible is not None:
        st.has_audio = st.has_audio & audible.numpy()
    labels_all = torch.tensor(st.meta["labels"])
    games = np.asarray(st.meta["games"])
    num_classes = len(classes)
    train_views = np.array(ALL_VIEWS if args.views == "all" else [0, 1])
    test_views = ALL_VIEWS if args.views == "all" else [0]

    video_dim = (768 if args.spatial == "mean" else 5 * 768) + (512 if args.clip_image else 0)
    m = load_model_module(args.version)
    common = dict(dropout=cfg.DROPOUT, head_hidden_dim=cfg.HEAD_HIDDEN, head_dropout=cfg.HEAD_DROPOUT,
                  num_classes=num_classes)
    class_text = uc.encode_class_text(cfg.CLIP, classes, device) if args.version != "v3" else None

    def build():
        if args.version == "v3":
            return m.AVFeatureModel(video_dim=video_dim, audio_dim=768, fusion_dim=cfg.FUSION_DIM,
                                    cross_heads=cfg.CROSS_HEADS, temporal_heads=cfg.TEMPORAL_HEADS,
                                    temporal_layers=cfg.TEMPORAL_LAYERS, **common)
        if args.version == "v4":
            return m.AVTFeatureModel(video_dim=video_dim, audio_dim=768, text_dim=class_text.shape[-1],
                                     fusion_dim=cfg.FUSION_DIM, cross_heads=cfg.CROSS_HEADS,
                                     temporal_heads=cfg.TEMPORAL_HEADS, temporal_layers=cfg.TEMPORAL_LAYERS,
                                     temporal_length=16, head_type="text", **common)
        return m.VideoTextFeatureModel(video_dim=video_dim, text_dim=class_text.shape[-1],
                                       fusion_dim=cfg.FUSION_DIM, temporal_heads=cfg.TEMPORAL_HEADS,
                                       temporal_layers=cfg.TEMPORAL_LAYERS, temporal_length=16,
                                       text_tokens=num_classes, head_type="text", **common)

    def forward(model, b):
        if args.version == "v3":
            return model(b["video"], b["audio"], b["audio_pad"], b["video_pad"])
        ctx = class_text.unsqueeze(0).expand(len(b["video"]), -1, -1)
        if args.version == "v4":
            return model(b["video"], b["audio"], b["audio_pad"], b["video_pad"], ctx, None, class_text=class_text)
        return model(b["video"], ctx, b["video_pad"], class_text=class_text)

    def batch(view, idx, train):
        b = st.get(view, idx, device)
        on = b.pop("has_audio")
        if train and args.audio_dropout > 0:
            on = on & (torch.rand(len(idx), device=device) >= args.audio_dropout)
        b["audio"] = b["audio"] * on[:, None, None]
        pad = torch.zeros(b["audio"].shape[:2], dtype=torch.bool, device=device)
        pad[~on, 1:] = True
        b["audio_pad"] = pad
        if train and args.token_drop > 0:
            drop = torch.rand(b["video"].shape[:2], device=device) < args.token_drop
            b["video"] = b["video"].masked_fill(drop[..., None], 0.0)
        return b

    def predict(model, idx):
        model.eval()
        probs = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for v in test_views:
                out_v = []
                for s in range(0, len(idx), 512):
                    chunk = idx[s:s + 512]
                    b = batch(np.full(len(chunk), v), chunk, False)
                    out_v.append(uc.bc.clip_logits(forward(model, b).float(), b["video_pad"]).float())
                probs = probs + torch.cat(out_v).softmax(-1)
        return probs / len(test_views)

    def evaluate(model, idx):
        probs = predict(model, idx)
        y = labels_all[idx].to(device)
        r = uc.metrics(probs.log(), y, num_classes)
        r["loss"] = float(F.nll_loss(probs.clamp_min(1e-8).log(), y))
        return r, probs

    def train(fit_idx, epochs, seed, val_idx=None, tag=""):
        uc.bc.seed_everything(seed)
        rng = np.random.default_rng(seed)
        model = build().to(device)
        decay = [q for q in model.parameters() if q.ndim > 1]
        no_decay = [q for q in model.parameters() if q.ndim <= 1]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
        spe = math.ceil(len(fit_idx) / args.batch_size)
        total, warm = epochs * spe, args.warmup_epochs * spe
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        best, best_epoch, best_state, stale = -1.0, 0, None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(fit_idx)
            run_loss, correct = 0.0, 0
            for s in range(0, len(order), args.batch_size):
                idx = order[s:s + args.batch_size]
                view = rng.choice(train_views, len(idx))
                b = batch(view, idx, True)
                y = labels_all[idx].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = uc.bc.clip_logits(forward(model, b).float(), b["video_pad"])
                    loss = crit(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                opt.step()
                sched.step()
                run_loss += loss.item() * len(idx)
                correct += int((logits.argmax(-1) == y).sum())
            msg = f"[{tag}] epoch {epoch:03d} loss={run_loss / len(fit_idx):.4f} train@1={correct / len(fit_idx):.4f}"
            if val_idx is not None:
                val, _ = evaluate(model, val_idx)
                msg += f" | val@1={val['top1']:.4f}"
                if val["top1"] > best:
                    best, best_epoch, stale = val["top1"], epoch, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(msg, flush=True)
            if val_idx is not None and stale >= args.patience:
                print(f"[{tag}] early stop at {epoch} (best {best_epoch}: {best:.4f})", flush=True)
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, best

    keys = ("top1", "top5", "mean_class_acc", "macro_f1")
    results = []
    for k in args.splits:
        tr, te = splits[k]["train"], splits[k]["test"]
        st.fit_stats(tr)
        fit, val, vg = uc.val_groups_split(tr, games, args.val_groups, args.seed + k)
        print(f"\n[SPLIT {k}] fit={len(fit)} val={len(val)} ({vg}) test={len(te)} | {args.run} {args.version}", flush=True)
        model, best_epoch, best_val = train(fit, args.epochs, args.seed + k, val, f"split{k} select")
        if args.refit:
            model, _, _ = train(tr, max(best_epoch, args.warmup_epochs + 1), args.seed + 100 + k, None, f"split{k} refit")
        test, probs = evaluate(model, te)
        np.save(out / f"probs_split{k}.npy", probs.cpu().numpy())
        np.save(out / f"labels_split{k}.npy", labels_all[te].numpy())
        torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes}, out / f"split{k}.pt")
        print(f"[SPLIT {k}] TEST top1={test['top1']:.4f} top5={test['top5']:.4f} macro-F1={test['macro_f1']:.4f}", flush=True)
        results.append({"split": k, "best_epoch": best_epoch, "val_top1": best_val,
                        **{key: test[key] for key in keys}})
        del model
        torch.cuda.empty_cache()
    report = {"run": args.run, "version": args.version, "args": vars(args), "splits": results}
    for key in keys:
        vals = np.array([r[key] for r in results])
        report[f"{key}_mean"], report[f"{key}_std"] = float(vals.mean()), float(vals.std())
    json.dump(report, open(out / "summary.json", "w"), indent=2)
    print(f"\n[RESULT] {args.run} {args.version}: top1 = {report['top1_mean']:.4f} ± {report['top1_std']:.4f} "
          f"| top5 = {report['top5_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
