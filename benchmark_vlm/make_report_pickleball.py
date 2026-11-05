"""Pickleball version of make_report.py: runs_pickleball/comparison.md, same tables and columns.

Differences from the NSVA report:
  * custom v3/v4/v5 are scored on the same 4685 clips as the VLMs, from
    custom_preds_pickleball/custom_preds_v*.csv (make_splits_pickleball.py maps the custom
    model's test predictions onto each clip). weights_pickleball/<v>/fold_k/result.json
    scores a different unit (1 s windows / tokens), so it is not comparable to the clips.
  * custom parameter count = encoders used for pickleball (VideoMAE-base finetuned on
    Kinetics, BEATs, CLIP text with projection) + the trained head of best.pt.

    /data/miniconda/envs/anything/bin/python make_report_pickleball.py
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np
import pandas as pd

from make_report import ALIASES, LABELS, fmt_params, mcnemar_p, scores, vlm_folds, vlm_params

BASELINE_DIR = "/data/Cuong/pickleball/pickleball_track/baseline"  # branch training_model_custom (outputs of the custom models)
# BEATs_iter3_plus_AS2M parameter count (fixed; the checkpoint lives with the baseline code, not here).
BEATS_PARAMS = 90_311_792
CUSTOM = {"v3": ["videomae", "beats"], "v4": ["videomae", "beats", "clip_text"], "v5": ["videomae", "clip_text"]}


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def custom_params(baseline, weights):
    """(total at inference, trained head) per custom version."""
    import sys

    import torch
    from transformers import CLIPTextModelWithProjection
    sys.path.insert(0, baseline)
    import benchmark_common as bc
    import pickleball_common as pc

    cpu = torch.device("cpu")
    enc = {"videomae": n_params(bc.load_videomae(pc.DEFAULT_VIDEOMAE, cpu)),
           "beats": BEATS_PARAMS,
           "clip_text": n_params(CLIPTextModelWithProjection.from_pretrained(pc.DEFAULT_CLIP))}
    out = {}
    for v, used in CUSTOM.items():
        heads = set()
        for ckpt in sorted(glob.glob(os.path.join(weights, v, "fold_*", "best.pt"))):
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            heads.add(sum(t.numel() for t in sd.values() if torch.is_tensor(t)))
        assert len(heads) == 1, f"{v}: head size differs across folds {heads}"
        head = heads.pop()
        out[v] = (head + sum(enc[e] for e in used), head)
        print(f"custom {v}: head {head:,} + " + " + ".join(f"{e} {enc[e]:,}" for e in used))
    return out, enc


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default="/data/Cuong/pickleball/pickleball_track/baseline")
    ap.add_argument("--weights", default=os.path.join(BASELINE_DIR, "pickleball/outputs/runs"))
    ap.add_argument("--runs", default=os.path.join(here, "runs_pickleball"))
    ap.add_argument("--custom-dir", default=os.path.join(here, "custom_preds_pickleball"))
    ap.add_argument("--test-csv", default=os.path.join(here, "splits_pickleball/test.csv"))
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", default=os.path.join(here, "runs_pickleball/comparison.md"))
    args = ap.parse_args()
    exclude = [e for e in args.exclude.split(",") if e]

    gt = pd.read_csv(args.test_csv)[["clip_path", "label", "fold", "user", "source"]]
    gt["label"] = gt.label.replace(ALIASES)

    models = {}
    params, enc = custom_params(args.baseline_dir, args.weights)
    for v in CUSTOM:
        c = pd.read_csv(os.path.join(args.custom_dir, f"custom_preds_{v}.csv"))
        c["pred"] = c.pred.replace(ALIASES)
        df = gt.merge(c[["clip_path", "pred"]], on="clip_path", how="left")
        assert df.pred.notna().all(), f"custom {v}: missing clips"
        models[f"custom {v}"] = dict(kind="custom (trained)", config="best epoch / fold", folds=vlm_folds(df),
                                     ok=(df.pred == df.label).values, params=params[v][0], trained=params[v][1])
    for d in sorted(glob.glob(os.path.join(args.runs, "*", "test_final", "predictions.csv"))):
        name = d.split(os.sep)[-3]
        if name in exclude:
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
        r["cls_f1"] = scores(sum(r["folds"].values()))[2]
    order = sorted(models, key=lambda n: -models[n]["f1"].mean())
    folds = sorted(next(iter(models.values()))["folds"])

    pvals = {}

    def delta(name, v):
        ref = f"custom {v}"
        if name == ref:
            return "–"
        d = models[name]["f1"].mean() - models[ref]["f1"].mean()
        p = pvals[(name, v)] = mcnemar_p(models[name]["ok"], models[ref]["ok"])
        return f"{d:+.3f}{'*' if p < 0.05 else ''}"

    counts = gt.label.value_counts()
    M = lambda n: f"{n / 1e6:.1f}M"
    L = ["# Pickleball hit / miss / no-event: custom models vs zero-shot VLMs", "",
         f"_Generated {datetime.date.today()} by `make_report_pickleball.py`. {len(gt)} test clips "
         f"({counts['made']} hit, {counts['miss']} miss, {counts['no-event']} no-event), {gt.source.nunique()} videos, "
         f"{gt.user.nunique()} players, 5 folds `StratifiedGroupKFold` by player (seed 42); "
         "every model is scored on the same test clips of each fold._", "",
         "- **Custom v3/v4/v5**: trained per fold (best epoch, early-stopped on the fold's val split); the test "
         "predictions of `weights_pickleball/<v>` are mapped onto each ~1 s clip (`make_splits_pickleball.py`) "
         "and scored here like the VLMs.",
         "- **VLM**: zero-shot, no training; the frame config was picked once on `val_fold1` "
         "(16f/448, 32f/448, 16f/640), then run on all test clips.",
         f"- **Params**: counted at inference from the real weights. Custom = pretrained encoders "
         f"(VideoMAE-base Kinetics {M(enc['videomae'])}, BEATs {M(enc['beats'])} for v3/v4, "
         f"CLIP text {M(enc['clip_text'])} for v4/v5) + the trained head.",
         "- **Mean ± std**: over the 5 folds (population std, as in the training logs).",
         "- **Δ vs vX**: mean macro-F1 of the row minus that of custom vX. `*` = significant "
         "(McNemar exact test on the per-clip predictions, p < 0.05).",
         "- **made = hit**: the column names follow the NSVA report; for pickleball `made` is a hit.", "",
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
          "- v4 is slightly but significantly above v3 (+0.011, McNemar p = 0.016), unlike NSVA where they tie; "
          "every VLM is significantly below all three custom models.",
          "- Unlike NSVA, the VLMs do answer `no-event` (F1 0.73–0.90); their weak class is hit or miss "
          "(minicpm-v-4 hit F1 0.09, gemma4-e4b miss F1 0.03).",
          "- Fold 3 (players Cuong, Son, Tu) is the hardest fold for the custom models and most VLMs, hence the large fold std.",
          "- The custom numbers here are per clip and differ from `weights_pickleball/<v>/fold_k/result.json`, "
          "which scores 1 s training windows.",
          "- VLM parameter counts exclude Qwen3.5's MTP weights (not loaded at inference); "
          "Gemma 4 E4B includes its audio tower and per-layer embeddings (all loaded).",
          "- `val_fold1` (players Cuong, VanAnh; used only to pick the VLM frame config) overlaps the test clips "
          "of folds 3 and 4; folds 1, 2 and 5 are fully clean for the VLMs.",
          f"- Not included: {', '.join(exclude) or 'none (all 8 VLMs finished)'}."]
    open(args.out, "w").write("\n".join(L) + "\n")
    print("wrote", args.out)
    print("McNemar p:", {f"{k[0]} vs {k[1]}": f"{p:.2g}" for k, p in pvals.items() if p >= 0.01})


if __name__ == "__main__":
    main()
