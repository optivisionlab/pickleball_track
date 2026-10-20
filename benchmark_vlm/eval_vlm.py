"""Zero-shot evaluation of Qwen3.5-4B on NSVA basketball clips (made / miss / no-event).

Follows Guide.md / the original eval_vlm.py; only the model is changed to
Qwen/Qwen3.5-4B. Additions for this dataset:
  * the CSVs from make_splits.py carry a `fold` column; metrics.json then also has
    per-fold metrics and their mean / std, the same way the custom model reports;
  * --resume continues a stopped run from its predictions.csv;
  * `compare` accepts the custom-model label names (hit / no_event).

Input per sample: a video file (.mp4/.avi/...) OR a folder of frames (jpg/png, sorted by name).
Output: predictions.csv (3-class probabilities, predicted label, latency) + metrics.json.

No fine-tuning: the model is asked a multiple-choice question and we read the
probability of the answer letters A/B/C from the first generated token's logits.

Usage:
  python eval_vlm.py run     --csv splits/val_fold1.csv --out runs/val_16f_448 --num-frames 16
  python eval_vlm.py compare --csv splits/test.csv --vlm runs/test_final/predictions.csv --custom custom_preds_v3.csv

test.csv:          clip_path,label[,fold,...]   (label in made/miss/no-event)
custom_preds.csv:  clip_path,pred[,p_made,p_miss,p_no_event]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

LABELS = ["made", "miss", "no-event"]
LETTERS = ["A", "B", "C"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
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


# ---------------------------------------------------------------- model
def load_model(model_id, attn):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if attn == "auto":
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
    # generic class: works for Qwen3-VL and Qwen3.5 (transformers picks the right model)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    tok = processor.tokenizer
    letter_ids = []
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        assert len(ids) == 1, f"letter {L} is not a single token: {ids}"
        letter_ids.append(ids[0])
    print(f"loaded {model_id} (attn={attn})")
    return model, processor, letter_ids


def video_content(path, num_frames, max_pixels, fps_for_frames):
    """Build the Qwen video entry from a video file or a frame folder."""
    if os.path.isdir(path):
        frames = sorted(f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(IMG_EXT))
        if not frames:
            raise FileNotFoundError(f"no frames in {path}")
        idx = np.linspace(0, len(frames) - 1, num=min(num_frames, len(frames))).round().astype(int)
        # qwen-vl-utils needs an even number of frames for a frame list
        if len(idx) % 2:
            idx = np.append(idx, idx[-1])
        return {"type": "video", "video": ["file://" + os.path.abspath(frames[i]) for i in idx],
                "sample_fps": fps_for_frames, "max_pixels": max_pixels}
    return {"type": "video", "video": "file://" + os.path.abspath(path),
            "nframes": num_frames, "max_pixels": max_pixels}


def predict_one(model, processor, letter_ids, path, prompt, args):
    import torch
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        video_content(path, args.num_frames, args.max_pixels, args.frame_fps),
        {"type": "text", "text": prompt},
    ]}]
    # enable_thinking=False: Qwen3.5 answers directly instead of opening a <think> block
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    video_metadata = None
    if videos is not None:
        videos, video_metadata = map(list, zip(*videos))
    inputs = processor(text=[text], images=images, videos=videos, video_metadata=video_metadata,
                       do_resize=False, return_tensors="pt", **video_kwargs).to(model.device)
    with torch.inference_mode():
        # logits_to_keep=1: only the next-token logits after the prompt are computed
        logits = model(**inputs, logits_to_keep=1).logits[0, -1]
    letter_logits = logits[letter_ids].float()
    probs = torch.softmax(letter_logits, dim=-1).cpu().numpy()
    # sanity signal: how much of the full-vocab mass sits on A/B/C
    mass = torch.softmax(logits.float(), dim=-1)[letter_ids].sum().item()
    return probs, mass


# ---------------------------------------------------------------- run
def run(args):
    import torch
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
    todo = df[~df.clip_path.isin(done)]

    model, processor, letter_ids = load_model(args.model, args.attn)
    predict_one(model, processor, letter_ids, df.clip_path.iloc[0], prompt, args)  # warm-up
    torch.cuda.reset_peak_memory_stats()

    for i, r in enumerate(todo.itertuples(index=False)):
        path, label = r.clip_path, r.label
        t0 = time.perf_counter()
        try:
            probs, mass = predict_one(model, processor, letter_ids, path, prompt, args)
            err = ""
        except Exception as e:                       # keep going; failed clips count as wrong
            probs, mass, err = np.array([np.nan] * 3), np.nan, repr(e)[:200]
            if "out of memory" in err.lower():
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        pred = LABELS[int(np.nanargmax(probs))] if not err else "error"
        rows.append(dict(clip_path=path, label=label, pred=pred,
                         p_made=probs[0], p_miss=probs[1], p_no_event=probs[2],
                         letter_mass=mass, latency_s=latency, error=err,
                         **{c: getattr(r, c) for c in extra}))
        if (i + 1) % 50 == 0:
            print(f"{len(rows)}/{len(df)} done", flush=True)
            pd.DataFrame(rows).to_csv(pred_path, index=False)

    pred = pd.DataFrame(rows)
    pred["error"] = pred.error.fillna("")
    pred.to_csv(pred_path, index=False)
    m = metrics(pred)
    if "fold" in pred and pred.fold.nunique() > 1:
        m.update(kfold_metrics(pred))
    m.update(model=args.model, csv=args.csv, num_frames=args.num_frames, max_pixels=args.max_pixels,
             prompt=prompt,
             error_rate=float((pred.error != "").mean()),
             mean_letter_mass=float(pred.letter_mass.mean()),
             latency_p50_s=float(pred.latency_s.quantile(0.5)),
             latency_p95_s=float(pred.latency_s.quantile(0.95)),
             peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
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
    r.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    r.add_argument("--resume", action="store_true", help="skip clips already in OUT/predictions.csv")
    c = sub.add_parser("compare")
    c.add_argument("--csv", required=True)
    c.add_argument("--vlm", required=True)
    c.add_argument("--custom", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
