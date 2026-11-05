"""Results of the best configuration of v3 / v4 / v5 -> results[_binary]/RESULTS.md.

Reads outputs/runs[_binary]/seed_<s>/<version>/result.json (FB_TASK=binary for the 2-label task);
the table shows mean ± std over the seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fb_common as fc  # noqa: E402

VERSIONS = ("v3", "v4", "v5")


def main() -> None:
    root = Path(fc.OUTPUT_ROOT)
    seeds = sorted(int(p.name.split("_")[1]) for p in root.glob("seed_*"))
    binary = fc.TASK == "binary"
    title = "goal / saved (saved = saved + off_target)" if binary else "goal / saved / off_target"
    L = [f"# Football (SoccerNet shots, 15 s clips): v3 / v4 / v5, {title}", "",
         "Train = part3-6 (6 094 clips), valid = part1 (1 931), test = part2 (2 137); the shot is at 5 s. "
         "Inputs and hyper-parameters of each version were chosen on valid macro-F1 (run.sh lists them); "
         f"test is only reported. Seeds {', '.join(map(str, seeds))}: mean ± std over the seeds.", "",
         "| Version | Valid macro-F1 | **Test macro-F1** | Test accuracy | "
         + " | ".join(f"F1 {c}" for c in fc.LABELS) + " | best epoch | trained params |",
         "|---" * (6 + len(fc.LABELS)) + "|"]
    for v in VERSIONS:
        rs = [json.load(open(root / f"seed_{s}" / v / "result.json")) for s in seeds
              if (root / f"seed_{s}" / v / "result.json").is_file()]
        if not rs:
            continue
        f1 = np.array([r["test"]["macro_f1"] for r in rs])
        sd = f1.std(ddof=1) if len(f1) > 1 else 0.0
        cls = " | ".join(f"{np.mean([r['test']['f1_by_class'][c] for r in rs]):.3f}" for c in fc.LABELS)
        L.append(f"| **{v}** | {np.mean([r['valid']['macro_f1'] for r in rs]):.4f} | **{f1.mean():.4f} ± {sd:.4f}** | "
                 f"{np.mean([r['test']['accuracy'] for r in rs]):.4f} | {cls} | "
                 f"{np.mean([r['best_epoch'] for r in rs]):.0f} | {rs[0]['params_m']:.1f}M |")
    L.append("")
    out = HERE / f"results{fc.TASK_SUFFIX}"
    out.mkdir(exist_ok=True)
    (out / "RESULTS.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
