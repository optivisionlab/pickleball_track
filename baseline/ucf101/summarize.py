"""UCF-101 results of the final configurations -> results/RESULTS.md.

    frozen/<version>/summary.json      v3 / v4 / v5 heads on frozen features (train.py)
    ft_k400_av/split<k>/result.json    VideoMAE-K400 fine-tuned end-to-end with the v3 head (finetune.py)

Top-1 (%) averaged over the 3 official splits (± std over the splits).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg  # noqa: E402

ROOT = Path(cfg.OUTPUT_ROOT)


def main() -> None:
    L = ["# UCF-101: v3 / v4 / v5", "",
         "Top-1 / macro-F1 (%) averaged over the 3 official splits (± std over the splits); train / test only "
         "on UCF-101, Kinetics is only the pre-training data of the backbone.", "",
         "| Model | Top-1 | Top-5 | Macro-F1 | Mean class acc. |", "|---|---|---|---|---|"]
    for v in ("v3", "v4", "v5"):
        p = ROOT / "frozen" / v / "summary.json"
        if p.is_file():
            r = json.load(open(p))
            L.append(f"| **{v}** (frozen VideoMAE-K400) | **{100 * r['top1_mean']:.1f} ± {100 * r['top1_std']:.1f}** | "
                     f"{100 * r['top5_mean']:.1f} | {100 * r['macro_f1_mean']:.1f} | {100 * r['mean_class_acc_mean']:.1f} |")
    ft = [json.load(open(p)) for p in sorted((ROOT / "ft_k400_av").glob("split*/result.json"))]
    if ft:
        top1 = np.array([r["top1"] for r in ft])
        L.append(f"| **v3** (VideoMAE-K400 fine-tuned end-to-end, {len(ft)}/3 splits) | "
                 f"**{100 * top1.mean():.1f} ± {100 * top1.std():.1f}** | "
                 f"{100 * np.mean([r['top5'] for r in ft]):.1f} | {100 * np.mean([r['macro_f1'] for r in ft]):.1f} | "
                 f"{100 * np.mean([r['mean_class_acc'] for r in ft]):.1f} |")
    L += ["", "Frozen: VideoMAE-K400 (mean + 2x2 grid, 6 views: random view in training, mean of the 6 at test), "
          "CLIP ViT-B/16 image embedding, BEATs fine-tuned on AudioSet, class-description text for v4 / v5; "
          "3 training groups held out to pick the epoch, then refit on the full training list.",
          "Fine-tuned: 16 frames (TSN sampling), 25 epochs, layer-wise lr decay 0.75, cached BEATs audio, "
          "test = 2 temporal offsets x 3 crops.", ""]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "RESULTS.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
