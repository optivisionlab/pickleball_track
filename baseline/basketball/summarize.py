"""Results of the final configuration (outputs/runs/seed_<s>/<version>) -> results/RESULTS.md.

Per seed: test macro-F1 = mean over the 5 game folds. Reported: mean ± std over the seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nsva_common as nc  # noqa: E402

VERSIONS = ("v3", "v4", "v5")


def main() -> None:
    runs = sorted(Path(nc.OUTPUT_ROOT).glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    seeds = [p.name.split("_")[1] for p in runs]
    lines = ["# NSVA basketball: v3 / v4 / v5 (final configuration)", "",
             "5-fold cross-validation grouped by game (8 329 clips, 96 games), classes no_event / hit / miss. "
             f"Seeds {', '.join(seeds)}; per seed the score is the mean over the 5 test folds, the table "
             "shows mean ± std over the seeds. Val = inner validation games (used for early stopping).", "",
             "| Version | Val macro-F1 | **Test macro-F1** | Test accuracy | F1 no_event | F1 hit | F1 miss | "
             + " | ".join(f"seed {s}" for s in seeds) + " |",
             "|---" * (7 + len(seeds)) + "|"]
    for v in VERSIONS:
        reps = [json.load(open(r / v / "kfold_summary.json")) for r in runs if (r / v / "kfold_summary.json").is_file()]
        if not reps:
            continue
        f1 = np.array([r["test_macro_f1_mean"] for r in reps])
        acc = np.array([r["test_accuracy_mean"] for r in reps])
        val = np.mean([np.mean([f["val_macro_f1"] for f in r["folds"]]) for r in reps])
        cls = {c: np.mean([np.mean([f["test_f1_by_class"][c] for f in r["folds"]]) for r in reps])
               for c in nc.LABEL2ID}
        lines.append(f"| **{v}** | {val:.4f} | **{f1.mean():.4f} ± {f1.std(ddof=1) if len(f1) > 1 else 0:.4f}** | "
                     f"{acc.mean():.4f} | {cls['no_event']:.4f} | {cls['hit']:.4f} | {cls['miss']:.4f} | "
                     + " | ".join(f"{x:.4f}" for x in f1) + " |")
    lines += ["", "Configuration (train.py defaults): VideoMAE self-supervised, mean + 2x2 grid tokens, "
              "+ second-half view (--tail), BEATs base, fixed text prompt (v4 / v5), fusion 384, weight decay "
              "0.05, token-drop 0.1, audio-dropout 0.2; 50 epochs, patience 10, batch 256, lr 3e-4, "
              "label smoothing 0.1.", ""]
    text = "\n".join(lines)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
