"""List likely annotation errors from cross-validated test predictions (CPU only).

Every video is predicted by a model that never saw its user, so a confident predicted
event with no ground-truth event nearby is often a MISSING label, and a ground-truth
event the model gives no probability to is often a WRONG/SHIFTED label. Check them in
the video and fix the JSON; noisy labels both hurt training and cap the measured score.

    python analysis/find_label_errors.py                    # v4 baseline predictions
    python analysis/find_label_errors.py --pred <ckpt>/test_predictions.pt

Writes label_suspects.csv (one row per suspect, sorted by confidence) and
label_suspects_per_video.csv to --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import pickleball_common as pc  # noqa: E402

PICKLEBALL = Path(__file__).resolve().parents[1]  # baseline/pickleball


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", default=str(PICKLEBALL / "outputs/runs/v4/test_predictions.pt"))
    parser.add_argument("--feature-cache", default="")
    parser.add_argument("--output-dir", default=str(PICKLEBALL / "outputs/analysis"))
    parser.add_argument("--min-prob", type=float, default=0.7,
                        help="Peak P(event) for a predicted event to count as confident.")
    parser.add_argument("--tolerance", type=float, default=0.5)
    args = parser.parse_args()

    pred = torch.load(args.pred, weights_only=False)
    if "token_valid" in pred:  # saved by the new pipeline (any --seq-tokens)
        data = {"video_index": pred["video_index"], "starts": pred["starts"],
                "token_valid": pred["token_valid"]}
        cache = args.feature_cache or sorted(glob.glob(str(PICKLEBALL / "cache/features_*.pt")))[-1]
        data["videos"] = torch.load(cache, map_location="cpu", weights_only=False)["videos"]
    else:
        cache = args.feature_cache or sorted(glob.glob(str(PICKLEBALL / "cache/features_*.pt")))[-1]
        full = torch.load(cache, map_location="cpu", weights_only=False)
        data = {k: full[k] for k in ("videos", "video_index", "starts", "token_valid")}
    timelines = pc.stitch_timelines(pred["probs"].numpy(), data, np.arange(len(data["video_index"])))

    suspects, per_video = [], []
    for vid, tl in timelines.items():
        info = data["videos"][vid]
        name = Path(info["path"]).name
        truth = [tuple(e) for e in info["events"]]
        p_event = 1 - tl[:, 0]
        events = pc.decode_events(tl, min_tokens=2, threshold=0.5, smooth=3, peak=True)
        gt_times = np.array([t for t, _ in truth]) if truth else np.zeros(0)
        n_missing = n_wrong = n_unseen = 0
        for t, c in events:
            token = int(t / pc.TOKEN_SEC)
            conf = float(p_event[max(0, token - 1):token + 2].max())
            near = np.abs(gt_times - t) <= args.tolerance if gt_times.size else np.zeros(0, bool)
            if not near.any() and conf >= args.min_prob:
                n_missing += 1
                suspects.append({"video": name, "time_s": round(t, 2), "type": "missing_label?",
                                 "predicted": pc.ID2LABEL[c], "ground_truth": "-",
                                 "confidence": round(conf, 3)})
            elif near.any():
                g = truth[int(np.flatnonzero(near)[np.abs(gt_times[near] - t).argmin()])]
                cls_conf = float(tl[max(0, token - 1):token + 2, c].max())
                if g[1] != c and cls_conf >= args.min_prob:
                    n_wrong += 1
                    suspects.append({"video": name, "time_s": round(g[0], 2), "type": "class_mismatch?",
                                     "predicted": pc.ID2LABEL[c], "ground_truth": pc.ID2LABEL[g[1]],
                                     "confidence": round(cls_conf, 3)})
        for t, c in truth:
            token = int(t / pc.TOKEN_SEC)
            w = int(args.tolerance / pc.TOKEN_SEC)
            peak = float(p_event[max(0, token - w):token + w + 1].max()) if token < len(p_event) else 0.0
            if peak < 0.2:
                n_unseen += 1
                suspects.append({"video": name, "time_s": round(t, 2), "type": "no_evidence?",
                                 "predicted": "-", "ground_truth": pc.ID2LABEL[c],
                                 "confidence": round(1 - peak, 3)})
        per_video.append({"video": name, "user": info["user"], "gt_events": len(truth),
                          "predicted_events": len(events), "missing_label?": n_missing,
                          "class_mismatch?": n_wrong, "no_evidence?": n_unseen})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suspects.sort(key=lambda r: -r["confidence"])
    with (out / "label_suspects.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(suspects[0]) if suspects else ["video"])
        w.writeheader()
        w.writerows(suspects)
    per_video.sort(key=lambda r: -(r["missing_label?"] + r["class_mismatch?"] + r["no_evidence?"]))
    with (out / "label_suspects_per_video.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(per_video[0]))
        w.writeheader()
        w.writerows(per_video)
    totals = {k: sum(r[k] for r in per_video) for k in ("missing_label?", "class_mismatch?", "no_evidence?")}
    print(f"[SUSPECTS] {totals} over {len(per_video)} videos -> {out / 'label_suspects.csv'}")
    for r in per_video[:12]:
        print(f"  {r['video']:42s} gt={r['gt_events']:3d} pred={r['predicted_events']:3d} "
              f"missing?={r['missing_label?']:3d} mismatch?={r['class_mismatch?']:2d} "
              f"no_evidence?={r['no_evidence?']:2d}")


if __name__ == "__main__":
    main()
