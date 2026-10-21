"""Zero-shot evaluation of video-language models on NSVA basketball clips (made / miss / no-event).

Follows Guide.md / the original eval_vlm.py (first run: Qwen/Qwen3.5-4B). The model
family is picked from the model id (see vlm_backends.py): Qwen3-VL / Qwen3.5,
InternVL3.5-HF, Gemma 4, Keye-VL-1.5, MiniCPM-V 4.0 / 4.5. Additions for this dataset:
  * the CSVs from make_splits.py carry a `fold` column; metrics.json then also has
    per-fold metrics and their mean / std, the same way the custom model reports;
  * --resume continues a stopped run from its predictions.csv;
  * `compare` accepts the custom-model label names (hit / no_event);
  * `summary` collects every model's results into one comparison table.

Input per sample: a video file (.mp4/.avi/...) OR a folder of frames (jpg/png, sorted by name).
Output: predictions.csv (3-class probabilities, predicted label, latency) + metrics.json.

No fine-tuning: the model is asked a multiple-choice question and we read the
probability of the answer letters A/B/C from the first generated token's logits.

Usage:
  python eval_vlm.py run     --csv splits/val_fold1.csv --out runs/qwen3.5-4b/val_16f_448 --model Qwen/Qwen3.5-4B
  python eval_vlm.py compare --csv splits/test.csv --vlm runs/qwen3.5-4b/test_final/predictions.csv --custom custom_preds_v3.csv
  python eval_vlm.py summary --runs runs --out model_comparison

test.csv:          clip_path,label[,fold,...]   (label in made/miss/no-event)
custom_preds.csv:  clip_path,pred[,p_made,p_miss,p_no_event]
"""
import argparse
import functools
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

LABELS = ["made", "miss", "no-event"]
# custom-model / mapping-CSV names -> names used here
ALIASES = {"hit": "made", "no_event": "no-event"}

# Same structure and letter order as the guide; the no-event line follows the
# NSVA labelling (fouls, turnovers, rebounds, jump balls, ... are no_event).
DEFAULT_PROMPT = (
    "You are a basketball referee reviewing a short clip from an NBA game broadcast.\n"
    "Decide what happens in this clip:\n"
    "A. made - a field-goal attempt (jump shot, layup, dunk, ...) goes through the hoop\n"
    "B. miss - a field-goal attempt is taken but the ball does not go through the hoop\n"
    "C. no-event - no field-goal attempt in this clip (e.g. foul, turnover, bad pass, "
    "out of bounds, rebound only, jump ball)\n"
    "Answer with a single letter: A, B or C."
)


# ---------------------------------------------------------------- data loading
class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, paths, backend):
        self.paths, self.backend = list(paths), backend

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            return {"i": i, "path": path, "err": "", **self.backend.load_clip(path)}
        except Exception as e:                        # reported per clip, the run keeps going
            return {"i": i, "path": path, "err": repr(e)[:200]}


def init_worker(_):
    """One thread per decode worker: by default decord and torch each use every core,
    and many workers then overload the CPU."""
    torch.set_num_threads(1)
    try:
        import decord
        if not isinstance(decord.VideoReader, functools.partial):
            decord.VideoReader = functools.partial(decord.VideoReader, num_threads=1)
    except ImportError:
        pass


# ---------------------------------------------------------------- prediction
class Predictor:
    """Letter probabilities for a list of clips; halves the batch for good after an OOM."""

    def __init__(self, backend, batch_size):
        self.backend = backend
        self.max_batch = batch_size if backend.batchable else 1

    def _forward(self, items):
        with torch.inference_mode():
            logits = self.backend.logits(items)
        logp = torch.log_softmax(logits, dim=-1)
        # log-probability of each letter = all its spellings ("A", "\u2581A", ...) added up
        letter = torch.stack([logp[:, g].logsumexp(-1) for g in self.backend.letter_ids], dim=-1)
        probs = torch.softmax(letter, dim=-1).cpu().numpy()
        # sanity signal: how much of the full-vocab mass sits on A/B/C
        mass = letter.exp().sum(-1).cpu().numpy()
        return probs, mass

    def __call__(self, items):
        probs, mass, errs = [], [], []
        k = 0
        while k < len(items):
            chunk = items[k:k + self.max_batch]
            try:
                p, m = self._forward(chunk)
                probs.append(p), mass.append(m), errs.extend([""] * len(chunk))
                k += len(chunk)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(chunk) == 1:                   # one clip does not fit: record and go on
                    probs.append(np.full((1, 3), np.nan)), mass.append(np.full(1, np.nan))
                    errs.append("CUDA out of memory")
                    k += 1
                else:
                    self.max_batch = max(1, len(chunk) // 2)
                    print(f"OOM: batch size lowered to {self.max_batch}", flush=True)
        if not probs:
            return np.zeros((0, 3)), np.zeros(0), []
        return np.concatenate(probs), np.concatenate(mass), errs


# ---------------------------------------------------------------- run
def run(args):
    from vlm_backends import BACKENDS, detect_backend
    df = pd.read_csv(args.csv)
    df["label"] = df.label.replace(ALIASES)
    if args.limit:
        df = df.head(args.limit)
    bad = set(df.label) - set(LABELS)
    assert not bad, f"unknown labels in csv: {bad}"
    prompt = open(args.prompt_file).read().strip() if args.prompt_file else DEFAULT_PROMPT
    extra = [c for c in ("fold", "video_id", "game_id") if c in df]

    os.makedirs(args.out, exist_ok=True)
    pred_path = os.path.join(args.out, "predictions.csv")
    rows, done = [], set()
    if args.resume and os.path.isfile(pred_path):
        old = pd.read_csv(pred_path, keep_default_na=False)
        old = old[old.error == ""]                    # retry clips that failed before
        rows, done = old.to_dict("records"), set(old.clip_path)
        print(f"resume: {len(done)} clips already done")
    todo = df[~df.clip_path.isin(done)].reset_index(drop=True)
    if todo.empty and rows:
        print("nothing left to run; recomputing metrics")

    backend_name = args.backend if args.backend != "auto" else detect_backend(args.model)
    backend = BACKENDS[backend_name](args, prompt)
    attn = backend.load_model()
    print(f"loaded {args.model} (backend={backend_name}, attn={attn})", flush=True)
    predict = Predictor(backend, args.batch_size)
    warm = ClipDataset(df.clip_path.iloc[:1], backend)[0]
    if not warm["err"]:
        predict([warm])                               # warm-up
    cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    # workers decode the next batches on the CPU while the GPU runs the current one
    loader = torch.utils.data.DataLoader(
        ClipDataset(todo.clip_path, backend), batch_size=args.batch_size,
        num_workers=args.workers, collate_fn=list, worker_init_fn=init_worker,
        prefetch_factor=2 if args.workers else None)
    t_start, last_print = time.perf_counter(), 0
    for batch in loader:
        t0 = time.perf_counter()
        ok = [it for it in batch if not it["err"]]
        probs, mass, errs = predict(ok) if ok else (np.zeros((0, 3)), np.zeros(0), [])
        if cuda:
            torch.cuda.synchronize()
        per_clip = (time.perf_counter() - t0) / max(len(ok), 1)   # GPU time per clip in this batch
        res = {it["i"]: (probs[k], mass[k], errs[k]) for k, it in enumerate(ok)}
        for it in batch:
            p, m, err = res.get(it["i"], (np.full(3, np.nan), np.nan, it["err"]))
            r = todo.iloc[it["i"]]
            rows.append(dict(clip_path=r.clip_path, label=r.label,
                             pred=LABELS[int(np.argmax(p))] if not err else "error",
                             p_made=p[0], p_miss=p[1], p_no_event=p[2],
                             letter_mass=m, latency_s=per_clip, error=err,
                             **{c: r[c] for c in extra}))
        if len(rows) - last_print >= 200:
            last_print = len(rows)
            rate = (len(rows) - len(done)) / (time.perf_counter() - t_start)
            print(f"{len(rows)}/{len(df)} done ({rate:.2f} clips/s)", flush=True)
            pd.DataFrame(rows).to_csv(pred_path, index=False)
    wall = time.perf_counter() - t_start

    pred = pd.DataFrame(rows)
    pred["error"] = pred.error.fillna("")
    pred.to_csv(pred_path, index=False)
    m = metrics(pred)
    if "fold" in pred and pred.fold.nunique() > 1:
        m.update(kfold_metrics(pred))
    m.update(model=args.model, backend=backend_name, csv=args.csv, num_frames=args.num_frames,
             max_pixels=args.max_pixels, pack=args.pack, batch_size=args.batch_size,
             workers=args.workers, prompt=prompt,
             throughput_clips_per_s=round((len(rows) - len(done)) / max(wall, 1e-9), 3),
             error_rate=float((pred.error != "").mean()),
             mean_letter_mass=float(pred.letter_mass.mean()),
             latency_p50_s=float(pred.latency_s.quantile(0.5)),
             latency_p95_s=float(pred.latency_s.quantile(0.95)),
             peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps({k: v for k, v in m.items() if k not in ("per_class", "per_fold", "prompt")}, indent=2))


# ---------------------------------------------------------------- metrics
def metrics(df, n_boot=1000, seed=0):
    y, p = df.label.values, df.pred.values
    m = {
        "n": int(len(df)),
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, p)),
        "per_class": classification_report(y, p, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(y, p, labels=LABELS).tolist(),
    }
    cols = ["p_made", "p_miss", "p_no_event"]
    if all(c in df for c in cols) and df[cols].notna().all().all():
        try:
            m["auc_ovr_macro"] = float(roc_auc_score(y, df[cols].values, multi_class="ovr", labels=LABELS))
        except ValueError:
            pass
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots.append(f1_score(y[i], p[i], labels=LABELS, average="macro", zero_division=0))
    m["macro_f1_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return m


def kfold_metrics(df):
    """Per-fold macro-F1 / accuracy and mean ± std, like the custom model's kfold_summary.json."""
    per = {}
    for fold, g in df.groupby("fold"):
        per[int(fold)] = {"n": int(len(g)), "accuracy": float((g.label == g.pred).mean()),
                          "macro_f1": float(f1_score(g.label, g.pred, labels=LABELS,
                                                     average="macro", zero_division=0))}
    f1s = np.array([v["macro_f1"] for v in per.values()])
    accs = np.array([v["accuracy"] for v in per.values()])
    return {"per_fold": per,
            "fold_macro_f1_mean": float(f1s.mean()), "fold_macro_f1_std": float(f1s.std()),
            "fold_accuracy_mean": float(accs.mean()), "fold_accuracy_std": float(accs.std())}


def compare(args):
    from statsmodels.stats.contingency_tables import mcnemar
    gt = pd.read_csv(args.csv)
    gt = gt[[c for c in ("clip_path", "label", "fold") if c in gt]]
    gt["label"] = gt.label.replace(ALIASES)
    vlm = pd.read_csv(args.vlm)
    cus = pd.read_csv(args.custom)
    cus["pred"] = cus.pred.replace(ALIASES)
    df = gt.merge(vlm[["clip_path", "pred"]].rename(columns={"pred": "vlm"}), on="clip_path") \
           .merge(cus[["clip_path", "pred"]].rename(columns={"pred": "custom"}), on="clip_path")
    assert len(df) == len(gt), f"only {len(df)}/{len(gt)} clips matched; both models must cover the same test set"

    out = {}
    for name in ("custom", "vlm"):
        sub = df.rename(columns={name: "pred"})
        mm = metrics(sub[["label", "pred"]])
        out[name] = {k: mm[k] for k in ("accuracy", "macro_f1", "macro_f1_ci95", "balanced_acc",
                                        "confusion_matrix_rows_true_cols_pred")}
        out[name]["f1_per_class"] = {L: mm["per_class"][L]["f1-score"] for L in LABELS}
        if "fold" in sub and sub.fold.nunique() > 1:
            km = kfold_metrics(sub)
            out[name].update({k: km[k] for k in ("fold_macro_f1_mean", "fold_macro_f1_std")})
    ok_v, ok_c = df.vlm == df.label, df.custom == df.label
    table = [[int((ok_v & ok_c).sum()), int((ok_v & ~ok_c).sum())],
             [int((~ok_v & ok_c).sum()), int((~ok_v & ~ok_c).sum())]]
    out["mcnemar"] = {"table_[[both_ok,vlm_only_ok],[custom_only_ok,both_wrong]]": table,
                      "p_value": float(mcnemar(table, exact=True).pvalue)}
    print(json.dumps(out, indent=2))
    out_dir = os.path.dirname(os.path.abspath(args.vlm))
    tag = os.path.splitext(os.path.basename(args.custom))[0]
    with open(os.path.join(out_dir, f"comparison_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    df[ok_v != ok_c].to_csv(os.path.join(out_dir, f"disagreements_{tag}.csv"), index=False)


def summary(args):
    """One table for every model: runs/<name>/test_final/metrics.json (+ val configs, status)."""
    from statsmodels.stats.contingency_tables import mcnemar
    gt = pd.read_csv(args.csv)[["clip_path", "label"]]
    gt["label"] = gt.label.replace(ALIASES)
    ref = {}
    for v in ("v3", "v4", "v5"):
        path = os.path.join(args.custom_dir, f"custom_preds_{v}.csv")
        if os.path.isfile(path):
            c = pd.read_csv(path)
            c["pred"] = c.pred.replace(ALIASES)
            ref[v] = gt.merge(c[["clip_path", "pred", "fold", "p_made", "p_miss", "p_no_event"]], on="clip_path")

    def row_from(name, df, extra):
        mm = metrics(df, n_boot=args.n_boot)
        row = {"model": name, **extra, "n": mm["n"], "macro_f1": mm["macro_f1"],
               "macro_f1_ci95": "%.3f-%.3f" % tuple(mm["macro_f1_ci95"]),
               "accuracy": mm["accuracy"], "balanced_acc": mm["balanced_acc"],
               **{f"f1_{L}": mm["per_class"][L]["f1-score"] for L in LABELS},
               "auc_ovr_macro": mm.get("auc_ovr_macro")}
        if "fold" in df and df.fold.nunique() > 1:
            km = kfold_metrics(df)
            row.update(fold_f1_mean=km["fold_macro_f1_mean"], fold_f1_std=km["fold_macro_f1_std"],
                       fold1_f1=km["per_fold"].get(1, {}).get("macro_f1"))
        if "v3" in ref and name != "custom v3":
            both = df[["clip_path", "pred"]].merge(ref["v3"][["clip_path", "label", "pred"]],
                                                   on="clip_path", suffixes=("", "_v3"))
            ok_a, ok_b = both.pred == both.label, both.pred_v3 == both.label
            table = [[int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
                     [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())]]
            row["mcnemar_p_vs_v3"] = float(mcnemar(table, exact=True).pvalue)
        return row

    rows = [row_from(f"custom {v}", d, {"group": "custom", "status": "done"}) for v, d in ref.items()]
    for d in sorted(glob.glob(os.path.join(args.runs, "*", ""))):
        name = os.path.basename(os.path.normpath(d))
        if name in args.exclude.split(","):
            continue
        info = {}
        if os.path.isfile(os.path.join(d, "model.json")):
            info = json.load(open(os.path.join(d, "model.json")))
        elif not os.path.isdir(os.path.join(d, "test_final")):
            continue                                  # not a model folder
        extra = {"group": info.get("group", ""), "model_id": info.get("model_id", ""),
                 "device": info.get("device", "")}
        best = os.path.join(d, "best_config.txt")
        extra["config"] = open(best).read().strip() if os.path.isfile(best) else ""
        key = extra["config"].split(":")[0].replace("val_", "")
        extra["batch"] = info.get("batch_by_config", {}).get(key, info.get("batch_size"))
        for vm in sorted(glob.glob(os.path.join(d, "val_*", "metrics.json"))):
            extra["val_f1_" + os.path.basename(os.path.dirname(vm))[4:]] = json.load(open(vm))["macro_f1"]
        mpath = os.path.join(d, "test_final", "metrics.json")
        status_file = os.path.join(d, "STATUS")
        status = open(status_file).read().strip() if os.path.isfile(status_file) else ""
        if not os.path.isfile(mpath):
            rows.append({"model": name, **extra, "status": status or "not run"})
            continue
        m = json.load(open(mpath))
        pred = pd.read_csv(os.path.join(d, "test_final", "predictions.csv"), keep_default_na=False)
        pred = pred.drop(columns=["label"]).merge(gt, on="clip_path")
        extra.update(model_id=m.get("model", extra["model_id"]), status=status or "done")
        row = row_from(name, pred, extra)
        row.update({k: m.get(k) for k in ("mean_letter_mass", "error_rate", "throughput_clips_per_s",
                                          "peak_vram_gb")})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out + ".csv", index=False)
    cols = [c for c in ("model", "group", "status", "config", "batch", "device", "macro_f1", "macro_f1_ci95", "fold_f1_mean",
                        "fold_f1_std", "fold1_f1", "accuracy", "f1_made", "f1_miss", "f1_no-event",
                        "auc_ovr_macro", "mcnemar_p_vs_v3", "mean_letter_mass", "throughput_clips_per_s",
                        "peak_vram_gb") if c in out]
    table = out[cols].copy()
    for c in table.columns:
        if table[c].dtype.kind == "f":
            table[c] = table[c].map(lambda x: "" if pd.isna(x) else ("%.2g" % x if "mcnemar" in c else "%.3f" % x))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in table.values]
    with open(args.out + ".md", "w") as f:
        f.write("# Zero-shot VLM vs custom model, NSVA test (5 folds, %d clips)\n\n" % len(gt))
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}.csv and {args.out}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--csv", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="Qwen/Qwen3.5-4B")
    r.add_argument("--num-frames", type=int, default=16)
    r.add_argument("--max-pixels", type=int, default=448 * 448)
    r.add_argument("--frame-fps", type=float, default=2.0, help="real fps of the frames in a frame folder after sampling")
    r.add_argument("--prompt-file", default=None)
    r.add_argument("--attn", default="auto", choices=["auto", "flash_attention_2", "sdpa"],
                   help="auto = flash_attention_2 if flash-attn is installed, else sdpa")
    r.add_argument("--batch-size", type=int, default=8, help="clips per forward pass")
    r.add_argument("--workers", type=int, default=8, help="CPU workers decoding videos ahead of the GPU")
    r.add_argument("--backend", default="auto", choices=["auto", "qwen", "hf", "keye", "minicpm"],
                   help="model family adapter (vlm_backends.py); auto = from the model id")
    r.add_argument("--pack", type=int, default=1,
                   help="MiniCPM-V 4.5: frames packed per group (sees pack x num_frames frames)")
    r.add_argument("--answer-prefix", default="",
                   help="text appended after the generation prompt, before the answer letter (Keye)")
    r.add_argument("--device", default="cuda", help="cuda, cuda:1, or cpu (slow; for checking a model)")
    r.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    r.add_argument("--resume", action="store_true", help="skip clips already in OUT/predictions.csv")
    c = sub.add_parser("compare")
    c.add_argument("--csv", required=True)
    c.add_argument("--vlm", required=True)
    c.add_argument("--custom", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--csv", default="splits/test.csv")
    s.add_argument("--runs", default="runs")
    s.add_argument("--exclude", default="minicpm-v-4.5", help="comma-separated run names to leave out")
    s.add_argument("--custom-dir", default=".", help="folder with custom_preds_v3/v4/v5.csv")
    s.add_argument("--out", default="runs/model_comparison", help="writes OUT.csv and OUT.md")
    s.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    {"run": run, "compare": compare, "summary": summary}[a.cmd](a)
