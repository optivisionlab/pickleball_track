"""One Markdown report comparing every model: runs/comparison.md

  * custom v3/v4/v5: per-fold test results of the best epoch, read from
    weights_basketball/<v>/fold_k/result.json (the numbers of the training run);
    parameter count = real inference pipeline built on CPU with the real weights
    (VideoMAE + BEATs / CLIP text + the trained head of each fold's best.pt).
  * zero-shot VLMs: per-fold results recomputed from runs/<name>/test_final/predictions.csv
    (best config picked on val); parameter count from the safetensors shapes.
  * Δ vs v3/v4/v5 = difference of mean fold macro-F1; * = McNemar p < 0.05 on the
    8329 per-clip predictions (custom_preds_v*.csv, exported from the same best.pt).

Run in the env used to train the custom models (needs torch + transformers 4.x):
    /data/miniconda/envs/anything/bin/python make_report.py --baseline-dir <baseline/ of branch baseline_shot1>
"""
import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

LABELS = ["made", "miss", "no-event"]
CUSTOM_IDS = {"hit": "made", "miss": "miss", "no_event": "no-event"}   # result.json class names
ALIASES = {"hit": "made", "no_event": "no-event"}
CUSTOM = {   # version -> (head class, pretrained encoders used at inference)
    "v3": ("AVFeatureModel", ["videomae", "beats"]),
    "v4": ("AVTFeatureModel", ["videomae", "beats", "clip_text"]),
    "v5": ("VideoTextFeatureModel", ["videomae", "clip_text"]),
}
ARTIFACTS = "/data/Cuong/pickleball/artifacts"
HF_HUB = os.path.join(os.environ.get("HF_HOME", "/data/prism/huggingface_cache"), "hub")


# ---------------------------------------------------------------- metrics
def scores(cm):
    """macro-F1, accuracy and per-class F1 from a 3x3 confusion matrix (rows = true)."""
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(), tp.sum() / cm.sum(), f1


def custom_folds(weights, v):
    """Per-fold confusion matrices of the best epoch, in LABELS order."""
    out = {}
    for path in sorted(glob.glob(os.path.join(weights, v, "fold_*", "result.json"))):
        r = json.load(open(path))
        names = list(CUSTOM_IDS)                       # result.json order: no_event, hit, miss by id
        order = [r.get("label_order", ["no_event", "hit", "miss"]).index(n) for n in names]
        cm = np.asarray(r["test"]["confusion_matrix"])[np.ix_(order, order)]
        out[int(r["fold"])] = cm                       # now rows/cols = made, miss, no-event
    return out


def vlm_folds(df):
    return {int(f): confusion_matrix(g.label, g.pred, labels=LABELS) for f, g in df.groupby("fold")}


def mcnemar_p(a_ok, b_ok):
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0


# ---------------------------------------------------------------- parameters
def n_params(module):
    return sum(p.numel() for p in module.parameters())


def custom_params(baseline, weights):
    import torch
    sys.path.insert(0, baseline)
    import benchmark_common as bc
    from transformers import CLIPTextModel

    cpu = torch.device("cpu")
    enc = {"videomae": n_params(bc.load_videomae(os.path.join(ARTIFACTS, "weights/videomae-base"), cpu)),
           "beats": n_params(bc.load_beats(os.path.join(ARTIFACTS, "weights/BEATs_iter3_plus_AS2M.pt"),
                                           os.path.join(ARTIFACTS, "unilm"), cpu)),
           "clip_text": n_params(CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32"))}
    out = {}
    for v, (cls_name, used) in CUSTOM.items():
        for name in ("model", "config", "dataset"):
            sys.modules.pop(name, None)
        sys.path.insert(0, os.path.join(baseline, v))
        import model as mod
        sys.path.pop(0)
        kw = dict(fusion_dim=768, temporal_heads=12, temporal_layers=2, dropout=0.1,
                  head_hidden_dim=256, head_dropout=0.3, num_classes=3)
        heads = set()
        for ckpt in sorted(glob.glob(os.path.join(weights, v, "fold_*", "best.pt"))):
            if v == "v3":
                m = mod.AVFeatureModel(video_dim=768, audio_dim=768, cross_heads=16, **kw)
            elif v == "v4":
                m = mod.AVTFeatureModel(video_dim=768, audio_dim=768, text_dim=512, cross_heads=16,
                                        temporal_length=16, **kw)
            else:
                m = mod.VideoTextFeatureModel(video_dim=768, text_dim=512, temporal_length=16, **kw)
            m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
            heads.add(n_params(m))
        assert len(heads) == 1, f"{v}: head size differs across folds {heads}"
        head = heads.pop()
        out[f"custom {v}"] = (head + sum(enc[e] for e in used), head)
        print(f"custom {v}: head {head:,} + " + " + ".join(f"{e} {enc[e]:,}" for e in used))
    return out


def vlm_params(repo):
    """Every tensor of the newest snapshot, minus Qwen3.5 MTP weights (not loaded at inference)."""
    from safetensors import safe_open
    snaps = sorted(glob.glob(os.path.join(HF_HUB, "models--" + repo.replace("/", "--"), "snapshots", "*")),
                   key=os.path.getmtime)
    total = 0
    for f in glob.glob(os.path.join(snaps[-1], "*.safetensors")) if snaps else []:
        with safe_open(f, "pt") as st:
            for k in st.keys():
                if not k.startswith("mtp."):
                    total += int(np.prod(st.get_slice(k).get_shape()))
    return total or None


# ---------------------------------------------------------------- report
def fmt_params(n):
    return "" if n is None else (f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.0f}M")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default="/data/Cuong/pickleball/pickleball_track/baseline",
                    help="baseline/ folder of branch baseline_shot1 (git archive works)")
    ap.add_argument("--weights", default=os.path.join(ARTIFACTS, "weights_basketball"))
    ap.add_argument("--runs", default=os.path.join(here, "runs"))
    ap.add_argument("--test-csv", default=os.path.join(here, "splits/test.csv"))
    ap.add_argument("--exclude", default="minicpm-v-4.5,ref_sequential_16f_448")
    ap.add_argument("--out", default=os.path.join(here, "runs/comparison.md"))
    args = ap.parse_args()

    gt = pd.read_csv(args.test_csv)[["clip_path", "label", "fold"]]
    gt["label"] = gt.label.replace(ALIASES)

    models = {}   # name -> dict(folds={k: cm}, ok=per-clip correctness aligned to gt, params, ...)
    params = custom_params(args.baseline_dir, args.weights)
    for v in CUSTOM:
        c = pd.read_csv(os.path.join(here, f"custom_preds_{v}.csv"))
        c["pred"] = c.pred.replace(ALIASES)
        ok = gt.merge(c[["clip_path", "pred"]], on="clip_path", how="left")
        models[f"custom {v}"] = dict(kind="custom (trained)", config="best epoch / fold",
                                     folds=custom_folds(args.weights, v),
                                     ok=(ok.pred == ok.label).values,
                                     params=params[f"custom {v}"][0], trained=params[f"custom {v}"][1])
    for d in sorted(glob.glob(os.path.join(args.runs, "*", "test_final", "predictions.csv"))):
        name = d.split(os.sep)[-3]
        if name in args.exclude.split(","):
            continue
        info = json.load(open(os.path.join(args.runs, name, "model.json")))
        m = json.load(open(os.path.join(args.runs, name, "test_final", "metrics.json")))
        best = open(os.path.join(args.runs, name, "best_config.txt")).read().split(":")[0].replace("val_", "")
        p = pd.read_csv(d, keep_default_na=False)
        df = gt.merge(p[["clip_path", "pred"]], on="clip_path")
        assert len(df) == len(gt), f"{name}: {len(df)}/{len(gt)} clips"
        models[name] = dict(kind="VLM zero-shot", config=best, folds=vlm_folds(df),
                            ok=(df.pred == df.label).values, params=vlm_params(info["model_id"]),
                            trained=0, speed=m.get("throughput_clips_per_s"))
        print(f"{name}: {fmt_params(models[name]['params'])}")

    for r in models.values():
        f1s, accs = zip(*[scores(r["folds"][k])[:2] for k in sorted(r["folds"])])
        r["f1"], r["acc"] = np.array(f1s), np.array(accs)
        r["cls_f1"] = scores(sum(r["folds"].values()))[2]          # pooled over the 5 folds
    order = sorted(models, key=lambda n: -models[n]["f1"].mean())
    folds = sorted(next(iter(models.values()))["folds"])

    def delta(name, v):
        ref = f"custom {v}"
        if name == ref:
            return "–"
        d = models[name]["f1"].mean() - models[ref]["f1"].mean()
        p = mcnemar_p(models[name]["ok"], models[ref]["ok"])
        return f"{d:+.3f}{'*' if p < 0.05 else ''}"

    L = [f"# NSVA made / miss / no-event: custom models vs zero-shot VLMs", "",
         f"_Generated {datetime.date.today()} by `make_report.py`. 8329 test clips, 96 games, "
         "5 folds `StratifiedGroupKFold` by game (seed 42); every model is scored on the same test clips of each fold._", "",
         "- **Custom v3/v4/v5**: trained per fold; the numbers are the test results of the best epoch "
         "(early-stopped on the fold's val split), from `weights_basketball/<v>/fold_k/result.json`.",
         "- **VLM**: zero-shot, no training; the frame config was picked once on `val_fold1` "
         "(16f/448, 32f/448, 16f/640), then run on all test clips.",
         "- **Params**: counted at inference from the real weights. Custom = pretrained encoders "
         "(VideoMAE-base 88.6M, BEATs 90.3M for v3/v4, CLIP text 63.2M for v4/v5) + the trained head.",
         "- **Mean ± std**: over the 5 folds (population std, as in the training logs).",
         "- **Δ vs vX**: mean macro-F1 of the row minus that of custom vX. `*` = significant "
         "(McNemar exact test on the per-clip predictions, p < 0.05).", "",
         "## Summary", "",
         "| # | Model | Type | Params (trained) | Macro-F1 mean ± std | Acc mean ± std | Δ vs v3 | Δ vs v4 | Δ vs v5 "
         "| F1 made | F1 miss | F1 no-event | Clips/s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, n in enumerate(order, 1):
        r = models[n]
        trained = fmt_params(r["trained"]) if r["trained"] else "0"
        name = f"**{n}**" if i == 1 else n
        speed = f"{r['speed']:.2f}" if r.get("speed") else "–"
        L.append(f"| {i} | {name} | {r['kind']} | {fmt_params(r['params'])} ({trained}) "
                 f"| {r['f1'].mean():.4f} ± {r['f1'].std():.4f} | {r['acc'].mean():.4f} ± {r['acc'].std():.4f} "
                 f"| {delta(n, 'v3')} | {delta(n, 'v4')} | {delta(n, 'v5')} "
                 f"| {r['cls_f1'][0]:.3f} | {r['cls_f1'][1]:.3f} | {r['cls_f1'][2]:.3f} | {speed} |")
    L += ["", "F1 per class is pooled over the 5 test folds. Clips/s: VLM inference on one A30 "
          "(video decoding included); custom models run on cached features, so no comparable number.", "",
          "## Per fold", "", "Macro-F1 / accuracy on each test fold (custom: best epoch of that fold).", "",
          "| Model | Config | " + " | ".join(f"Fold {k} F1 / Acc" for k in folds) + " | Mean F1 | Std F1 | Mean Acc | Std Acc |",
          "|---|---|" + "---|" * (len(folds) + 4)]
    for n in order:
        r = models[n]
        cells = " | ".join(f"{f:.4f} / {a:.4f}" for f, a in zip(r["f1"], r["acc"]))
        L.append(f"| {n} | {r['config']} | {cells} | {r['f1'].mean():.4f} | {r['f1'].std():.4f} "
                 f"| {r['acc'].mean():.4f} | {r['acc'].std():.4f} |")
    L += ["", "## Notes", "",
          "- v3 and v4 are tied (difference not significant); every VLM is significantly below all three custom models.",
          "- Most VLMs almost never answer `no-event` (F1 no-event ≈ 0.03), which caps their macro-F1 near 0.44.",
          "- VLM parameter counts exclude Qwen3.5's MTP weights (not loaded at inference); "
          "Gemma 4 E4B includes its audio tower and per-layer embeddings (all loaded).",
          "- `val_fold1` (used only to pick the VLM frame config) overlaps the test clips of folds 2–5; "
          "fold 1 is fully clean for the VLMs.",
          f"- Not included: {', '.join(args.exclude.split(','))} (stopped / not a model)."]
    open(args.out, "w").write("\n".join(L) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
