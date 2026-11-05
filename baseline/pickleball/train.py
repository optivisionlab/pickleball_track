"""Train one model version (v3 / v4 / v5) on the pickleball data, 5-fold CV grouped by user.

    v3  video + audio            AVFeatureModel
    v4  video + audio + text     AVTFeatureModel, text = one CLIP description per class
    v5  video + text (no audio)  VideoTextFeatureModel

The class descriptions (pickleball_common.CLASS_PROMPTS) are identical for every sample,
so they carry no label information. Nothing from the file name (Trung/Truot, TraiTay/PhaiTay,
user) is a model input. Data, features, folds and metrics: pickleball_common.py.

    python train.py --version v4 --extract-only   # build the shared feature cache once
    python train.py --version v4                  # CV with the final configuration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pickleball_common as pc  # noqa: E402  (also puts the baseline root on sys.path)
from common.models import model_class  # noqa: E402


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--version", choices=("v3", "v4", "v5"), required=True)
    version = pre.parse_known_args()[0].version
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_common_args(parser, version)
    if version != "v3":
        parser.add_argument("--clip", default=pc.DEFAULT_CLIP)
    args = parser.parse_args()
    pc.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pc.prepare(args, device)
    if data is None:
        return
    Model = model_class(version)
    common = dict(dropout=args.dropout, head_hidden_dim=256, head_dropout=0.3,
                  num_classes=pc.NUM_CLASSES, temporal_length=args.seq_tokens)

    if version == "v3":
        def build_model(dims):
            return Model(video_dim=dims["video"], audio_dim=dims["audio"], fusion_dim=args.fusion_dim,
                         cross_heads=args.heads, temporal_heads=args.heads,
                         temporal_layers=args.temporal_layers, **common)

        def forward(model, batch):
            return model(batch["video"], batch["audio"], batch["audio_pad"], batch["video_pad"])
    else:
        class_text = pc.encode_class_prompts(args.clip, device)  # [C, Dt]

        def context(batch):
            return class_text.unsqueeze(0).expand(len(batch["video"]), -1, -1)

        if version == "v4":
            def build_model(dims):
                return Model(video_dim=dims["video"], audio_dim=dims["audio"], text_dim=class_text.shape[-1],
                             fusion_dim=args.fusion_dim, cross_heads=args.heads, temporal_heads=args.heads,
                             temporal_layers=args.temporal_layers, head_type="text", **common)

            def forward(model, batch):
                return model(batch["video"], batch["audio"], batch["audio_pad"], batch["video_pad"],
                             context(batch), None, class_text=class_text)
        else:
            def build_model(dims):
                return Model(video_dim=dims["video"], text_dim=class_text.shape[-1], fusion_dim=args.fusion_dim,
                             temporal_heads=args.heads, temporal_layers=args.temporal_layers,
                             text_tokens=pc.NUM_CLASSES, head_type="text", **common)

            def forward(model, batch):
                return model(batch["video"], context(batch), batch["video_pad"], class_text=class_text)

    pc.run_cv(data, build_model, forward, args, device)


if __name__ == "__main__":
    main()
