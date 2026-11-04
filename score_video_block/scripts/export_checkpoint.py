"""Pack a v4 cross-validation run into the single checkpoint the service loads.

Reads every ``fold_<k>/best.pt`` written by ``baseline/run_pickleball.sh`` (model
state_dict + that fold's feature mean/std), encodes the class descriptions with
CLIP exactly as training does (``pickleball_common.encode_class_prompts``), and
writes ``{"folds": [...], "class_text": [3, 512], "config": {...}}``. The service
then needs neither the training code nor CLIP.

Run it in the training environment:

    python scripts/export_checkpoint.py \\
        --run-dir /data/Cuong/pickleball/artifacts/weights_pickleball/v4 \\
        --output src/weights/pickleball_v4.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

DEFAULT_BASELINE = "/data/Cuong/pickleball/pickleball_track/baseline"
DEFAULT_RUN = "/data/Cuong/pickleball/artifacts/weights_pickleball/v4"
# Training arguments the served model and preprocessing depend on.
CONFIG_KEYS = ("fusion_dim", "heads", "temporal_layers", "dropout", "seq_tokens", "spatial",
               "videomae", "clip", "video_norm", "event_radius", "min_event_tokens")
# Settings the service implements; a run trained differently cannot be served as is.
SERVED = {"spatial": "grid", "video_norm": "none", "seq_tokens": 16}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=DEFAULT_RUN, help="Directory with fold_<k>/best.pt.")
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE, help="Training code (pickleball_common.py).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    paths = sorted(run_dir.glob("fold_*/best.pt"), key=lambda p: int(p.parent.name.split("_")[1]))
    if not paths:
        raise SystemExit(f"No fold_*/best.pt under {run_dir}")

    folds, config, decode = [], None, None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        train_args = checkpoint["args"]
        fold_config = {key: train_args.get(key) for key in CONFIG_KEYS}
        fold_config["seq_tokens"] = int(checkpoint.get("seq_tokens", fold_config["seq_tokens"]))
        if config is not None and fold_config != config:
            raise SystemExit(f"{path}: training arguments differ from the first fold")
        config = fold_config
        decode = checkpoint.get("decode", decode)
        folds.append({
            "fold": int(path.parent.name.split("_")[1]),
            "epoch": int(checkpoint["epoch"]),
            "model": {k: v.float().contiguous() for k, v in checkpoint["model"].items()},
            "feature_mean": {k: v.float() for k, v in checkpoint["feature_mean"].items()},
            "feature_std": {k: v.float() for k, v in checkpoint["feature_std"].items()},
        })
        label2id = checkpoint["label2id"]
    for key, value in SERVED.items():
        if config[key] != value:
            raise SystemExit(f"The service implements {key}={value!r}; this run used {config[key]!r}")

    sys.path.insert(0, str(Path(args.baseline_dir).resolve()))
    import pickleball_common as pc  # noqa: E402  (training code, for identical prompts)

    class_text = pc.encode_class_prompts(config["clip"], torch.device("cpu")).float()
    summary_path = run_dir / "cv_summary.json"
    results = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        results = {"event_macro_f1": summary["event"]["macro_f1"],
                   "pooled_token_macro_f1": summary["pooled_token"]["macro_f1"]}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "pickleball-fold-ensemble-v1",
        "version": "v4",
        "label2id": label2id,
        "config": config,
        "decode": decode,
        "class_text": class_text,
        "class_prompts": pc.CLASS_PROMPTS,
        "folds": folds,
        "source": {"run_dir": str(run_dir.resolve()), "results": results},
    }, output)
    print(f"Wrote {output}: {len(folds)} folds (epochs {[f['epoch'] for f in folds]}), "
          f"config {config}, decode {decode}, results {results}")


if __name__ == "__main__":
    main()
