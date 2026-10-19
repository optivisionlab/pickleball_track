"""v4 (VideoMAE + BEATs + CLIP text) on the NSVA basketball benchmark, clip-level, 1-1 mapping only.

The text input is one fixed prompt for every clip: it must never be derived from
the label, so it carries no class information.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import benchmark_common as bc  # noqa: E402
import config as cfg  # noqa: E402
from model import AVTFeatureModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, cfg.ARTIFACTS_ROOT, cfg.VERSION_ARTIFACTS, cfg)
    parser.add_argument("--clip", default=bc.DEFAULT_CLIP)
    parser.add_argument("--prompt", default=bc.DEFAULT_PROMPT)
    args = parser.parse_args()
    bc.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = bc.prepare(args, device)
    if data is None:
        return
    text = bc.encode_prompt(args.clip, args.prompt, device)
    video_dim, audio_dim = data["video"].shape[-1], data["audio"].shape[-1]

    def build_model():
        return AVTFeatureModel(
            video_dim=video_dim, audio_dim=audio_dim, text_dim=text.numel(), fusion_dim=768,
            cross_heads=16, temporal_heads=12, temporal_layers=2, dropout=0.1,
            head_hidden_dim=256, head_dropout=0.3, num_classes=bc.NUM_CLASSES,
            temporal_length=bc.NUM_TEMPORAL_TOKENS,
        )

    def forward(model, batch):
        text_pad = torch.zeros(len(batch["text"]), 1, dtype=torch.bool, device=device)
        return model(batch["video"], batch["audio"], batch["audio_pad"], batch["video_pad"],
                     batch["text"].unsqueeze(1), text_pad)

    bc.run_kfold(data, build_model, forward, args, device, args.output_dir, {"text": text})


if __name__ == "__main__":
    main()
