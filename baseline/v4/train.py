"""v4 (VideoMAE + BEATs + CLIP class descriptions) on the pickleball data.

Text input: one CLIP description per class (pickleball_common.CLASS_PROMPTS),
identical for every sample, so it carries no label information. It is used
  1. as cross-attention context: each video token attends over the 3 descriptions;
  2. as the classifier: logits = similarity of each token with each description.
Nothing from the file name (Trung/Truot, TraiTay/PhaiTay, user) is a model input.

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
from model import AVTFeatureModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    pc.add_common_args(parser, cfg.ARTIFACTS_ROOT, cfg.VERSION_ARTIFACTS, cfg)
    parser.add_argument("--clip", default=pc.DEFAULT_CLIP)
    args = parser.parse_args()
    pc.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pc.prepare(args, device)
    if data is None:
        return
    class_text = pc.encode_class_prompts(args.clip, device)  # [C, Dt]

    def build_model(dims):
        return AVTFeatureModel(
            video_dim=dims["video"], audio_dim=dims["audio"], text_dim=class_text.shape[-1],
            fusion_dim=args.fusion_dim, cross_heads=args.heads, temporal_heads=args.heads,
            temporal_layers=2, dropout=args.dropout, head_hidden_dim=256, head_dropout=0.3,
            num_classes=pc.NUM_CLASSES, temporal_length=pc.NUM_TEMPORAL_TOKENS,
            head_type="text",
        )

    def forward(model, batch):
        context = class_text.unsqueeze(0).expand(len(batch["video"]), -1, -1)
        return model(batch["video"], batch["audio"], batch["audio_pad"], batch["video_pad"],
                     context, None, class_text=class_text)

    pc.run_cv(data, build_model, forward, args, device)


if __name__ == "__main__":
    main()
