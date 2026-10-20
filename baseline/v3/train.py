"""v3 (VideoMAE + BEATs) on the pickleball data.

Data, features, folds and metrics come from baseline/pickleball_common.py and are
shared with v4/v5. No text input.

    python train.py --extract-only   # build the shared feature cache once
    python train.py                  # user-grouped train/val/test CV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
import pickleball_common as pc  # noqa: E402
from model import AVFeatureModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    pc.add_common_args(parser, cfg.ARTIFACTS_ROOT, cfg.VERSION_ARTIFACTS, cfg)
    args = parser.parse_args()
    pc.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pc.prepare(args, device)
    if data is None:
        return

    def build_model(dims):
        return AVFeatureModel(
            video_dim=dims["video"], audio_dim=dims["audio"], fusion_dim=args.fusion_dim,
            cross_heads=args.heads, temporal_heads=args.heads, temporal_layers=2,
            dropout=args.dropout, head_hidden_dim=256, head_dropout=0.3,
            num_classes=pc.NUM_CLASSES,
        )

    def forward(model, batch):
        return model(batch["video"], batch["audio"], batch["audio_pad"], batch["video_pad"])

    pc.run_cv(data, build_model, forward, args, device)


if __name__ == "__main__":
    main()
