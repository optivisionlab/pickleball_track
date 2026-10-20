"""Export per-clip test predictions of the custom model (baseline v3/v4/v5) for `eval_vlm.py compare`.

The custom benchmark only saves per-fold confusion matrices, but McNemar needs
per-clip predictions. This script runs the version's own train_benchmark.py with
bc.run_kfold replaced by a prediction pass: for each fold it loads
fold_k/best.pt and predicts that fold's test clips from the shared feature cache.
Nothing is trained and the baseline code is not modified; it is only read from
--baseline-dir (a checkout of branch baseline_shot1).

The macro-F1 of every fold is checked against the saved fold_k/result.json.

Usage:
  python export_custom_preds.py --version v3 --out custom_preds_v3.csv
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID2NAME = {0: "no-event", 1: "made", 2: "miss"}   # custom ids: no_event=0, hit=1, miss=2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", default="/data/Cuong/pickleball/pickleball_track/baseline")
    ap.add_argument("--version", default="v3", choices=["v3", "v4", "v5"])
    ap.add_argument("--splits", default="splits/test.csv", help="from make_splits.py (for clip_path)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    args, rest = ap.parse_known_args()      # `rest` goes to train_benchmark.py (e.g. --output-dir)
    out_csv = args.out or f"custom_preds_{args.version}.csv"

    baseline = Path(args.baseline_dir).resolve()
    version_dir = baseline / args.version
    sys.path[:0] = [str(version_dir), str(baseline)]
    import torch
    import benchmark_common as bc

    rows = []

    def predict_folds(data, build_model, forward, targs, device, output_dir, extra_inputs=None):
        labels = data["labels"].numpy()
        games = np.asarray(data["games"])
        tensors = {
            "video": data["video"].to(device), "video_pad": data["video_pad"].to(device),
            "audio": data["audio"].to(device),
            "audio_pad": torch.zeros(data["audio"].shape[:2], dtype=torch.bool, device=device),
        }
        tensors["audio_pad"][~data["has_audio"].to(device), 1:] = True   # same as run_kfold
        extra_inputs = extra_inputs or {}
        use_amp = device.type == "cuda" and not targs.no_amp
        for fold, _, _, test_idx in bc.make_folds(labels, games, targs.folds, targs.seed):
            fold_dir = Path(output_dir) / f"fold_{fold}"
            model = build_model().to(device)
            ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval()
            probs = []
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                for s in range(0, len(test_idx), args.batch_size):
                    idx = torch.as_tensor(test_idx[s:s + args.batch_size], device=device)
                    batch = {k: v[idx] for k, v in tensors.items()}
                    batch["video"], batch["audio"] = batch["video"].float(), batch["audio"].float()
                    batch.update({k: v.expand(len(idx), *v.shape) for k, v in extra_inputs.items()})
                    logits = bc.clip_logits(forward(model, batch).float(), batch["video_pad"])
                    probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
            probs = np.concatenate(probs)
            pred = probs.argmax(1)
            cm = np.zeros((bc.NUM_CLASSES, bc.NUM_CLASSES), dtype=np.int64)
            np.add.at(cm, (labels[test_idx], pred), 1)
            f1 = bc.metrics(cm)["macro_f1"]
            saved = json.load(open(fold_dir / "result.json"))["test"]["macro_f1"]
            print(f"fold {fold}: test macro-F1 {f1:.4f} (saved {saved:.4f})")
            if abs(f1 - saved) > 0.005:
                print(f"  WARNING: fold {fold} differs from the saved result")
            for j, i in enumerate(test_idx):
                rows.append(dict(video_id=data["video_ids"][i], fold=fold, pred=ID2NAME[int(pred[j])],
                                 p_made=probs[j, 1], p_miss=probs[j, 2], p_no_event=probs[j, 0]))
            del model

    bc.run_kfold = predict_folds
    sys.argv = [str(version_dir / "train_benchmark.py"), *rest]
    runpy.run_path(str(version_dir / "train_benchmark.py"), run_name="__main__")

    splits = pd.read_csv(args.splits)[["video_id", "clip_path"]]
    df = pd.DataFrame(rows).merge(splits, on="video_id", how="left")
    assert df.clip_path.notna().all(), "some clips are missing from --splits"
    df[["clip_path", "pred", "p_made", "p_miss", "p_no_event", "fold", "video_id"]].to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} clips)")


if __name__ == "__main__":
    main()
