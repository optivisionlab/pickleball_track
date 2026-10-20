"""Rebuild the exact val/test splits used by the custom model (baseline v3/v4/v5).

The custom benchmark (baseline/benchmark_common.py on branch baseline_shot1) uses:
  * only the clean rows of the 1-1 mapping CSV (same filtering rules, same order),
  * 5 outer folds: StratifiedGroupKFold(shuffle=True, random_state=42) over game_id,
  * per fold, an inner validation split: the first split of
    StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42 + fold) on the
    training part. It is used for early stopping only.

The logic is copied here (not imported) so this branch has no dependency on the
baseline code. Labels are renamed to the eval_vlm.py names:
hit -> made, miss -> miss, no_event -> no-event.

Outputs (in --out):
  test.csv           all 5 test folds together (every clip once), with a fold column
  test_fold{k}.csv   held-out test clips of fold k
  val_fold{k}.csv    inner validation clips of fold k (disjoint from test_fold{k})
  split_report.json  counts, plus a check against the saved test_games of the custom runs
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABEL2ID = {"no_event": 0, "hit": 1, "miss": 2}           # same ids as the custom model
TO_VLM = {"hit": "made", "miss": "miss", "no_event": "no-event"}

DATA = "/data/Cuong/pickleball/data/bendmark"
ARTIFACTS = "/data/Cuong/pickleball/artifacts"


def load_clean_rows(mapping_csv, video_dir):
    """Same filtering and order as benchmark_common.load_mapping."""
    video_dir = Path(video_dir)
    with open(mapping_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    id_count, file_count = {}, {}
    for r in rows:
        id_count[r["video_id"].strip()] = id_count.get(r["video_id"].strip(), 0) + 1
        file_count[r["file"].strip()] = file_count.get(r["file"].strip(), 0) + 1
    clean = []
    for r in rows:
        video_id, file_name = r["video_id"].strip(), r["file"].strip()
        label = r["label"].strip().lower()
        stem = r["gameid_eventid"].strip()
        if (id_count[video_id] > 1 or file_count[file_name] > 1 or label not in LABEL2ID
                or Path(file_name).stem != stem or not (video_dir / file_name).is_file()):
            continue
        clean.append(dict(video_id=video_id, game_id=stem.split("-")[0],
                          clip_path=str(video_dir / file_name), raw_label=label))
    return clean


def make_folds(labels, games, n_splits, seed):
    """Same as benchmark_common.make_folds."""
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(labels, labels, games), start=1):
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + fold)
        fit_rel, val_rel = next(inner.split(train_idx, labels[train_idx], games[train_idx]))
        yield fold, train_idx[fit_rel], train_idx[val_rel], test_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping-csv", default=f"{DATA}/1_video_co_nhan_khop_1-1.csv")
    ap.add_argument("--video-dir", default=f"{DATA}/train")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-run", default=f"{ARTIFACTS}/baseline/v3/checkpoints_benchmark_1to1",
                    help="custom-model output dir whose fold_k/result.json test_games are compared")
    ap.add_argument("--out", default="splits")
    args = ap.parse_args()

    rows = load_clean_rows(args.mapping_csv, args.video_dir)
    df = pd.DataFrame(rows)
    df["label"] = df.raw_label.map(TO_VLM)
    labels = df.raw_label.map(LABEL2ID).to_numpy()
    games = df.game_id.to_numpy()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["clip_path", "label", "fold", "video_id", "game_id"]
    report = {"clips": len(df), "games": int(df.game_id.nunique()),
              "label_counts": df.label.value_counts().to_dict(), "folds": {}}
    tests = []
    for fold, fit_idx, val_idx, test_idx in make_folds(labels, games, args.folds, args.seed):
        val = df.iloc[val_idx].assign(fold=fold)[cols]
        test = df.iloc[test_idx].assign(fold=fold)[cols]
        val.to_csv(out / f"val_fold{fold}.csv", index=False)
        test.to_csv(out / f"test_fold{fold}.csv", index=False)
        tests.append(test)
        info = {"fit": len(fit_idx), "val": len(val_idx), "test": len(test_idx),
                "test_games": int(test.game_id.nunique()),
                "test_label_counts": test.label.value_counts().to_dict()}
        saved = Path(args.check_run) / f"fold_{fold}" / "result.json"
        if saved.is_file():
            ref = sorted(json.load(open(saved))["test_games"])
            info["matches_custom_test_games"] = ref == sorted(set(test.game_id))
        report["folds"][fold] = info
        print(f"fold {fold}: fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)} "
              f"match={info.get('matches_custom_test_games', 'n/a')}")
    pd.concat(tests).to_csv(out / "test.csv", index=False)
    with open(out / "split_report.json", "w") as f:
        json.dump(report, f, indent=2)
    bad = [k for k, v in report["folds"].items() if v.get("matches_custom_test_games") is False]
    if bad:
        raise SystemExit(f"test folds {bad} differ from the custom-model runs; check sklearn version / inputs")
    print(f"wrote {out}/test.csv ({len(df)} clips) and per-fold val/test csv")


if __name__ == "__main__":
    main()
