"""Pickleball single-video inference from the command line.

Usage: python scripts/infer_video.py --video match.mp4 [--output result.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pickleball_track.core.config import InferenceParams, get_settings
from pickleball_track.core.logging import configure_logging
from pickleball_track.models.model_loader import load_models
from pickleball_track.services.inference_service import VideoInferenceService


def parse_args(defaults: InferenceParams) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pickleball single-video inference using pretrained VideoMAE + BEATs "
            "and trained temporal fusion model."
        )
    )
    parser.add_argument("--video", required=True, help="Path to a completely new video.")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint. Default: PICKLEBALL_CHECKPOINT.")
    # Defaults come from core/config.py (or PICKLEBALL_* environment variables).
    parser.add_argument("--miss-threshold", type=float, default=defaults.miss_threshold, help="Miss probability threshold.")
    parser.add_argument("--hit-threshold", type=float, default=defaults.hit_threshold, help="Hit probability threshold.")
    parser.add_argument("--event-gap", type=float, default=defaults.event_min_gap, help="Maximum gap for merging the same action event.")
    parser.add_argument("--smooth-radius", type=int, default=defaults.smooth_radius, help="Probability smoothing radius; 0 disables smoothing.")
    parser.add_argument("--overlap", type=float, default=defaults.overlap, help="Sliding-window overlap in [0, 1).")
    parser.add_argument("--output", default=None, help="Output JSON path. Default: <video>.inference.json")
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    args = parse_args(settings.inference)
    configure_logging(settings.log_level)
    if args.checkpoint:
        settings = replace(settings, checkpoint_path=Path(args.checkpoint).expanduser().resolve())

    service = VideoInferenceService(
        load_models(settings),
        InferenceParams(
            hit_threshold=args.hit_threshold,
            miss_threshold=args.miss_threshold,
            event_min_gap=args.event_gap,
            smooth_radius=args.smooth_radius,
            overlap=args.overlap,
        ),
    )
    video_path = Path(args.video)
    result = service.predict(video_path)

    output_path = Path(args.output) if args.output else video_path.with_suffix(".inference.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(output_path)


if __name__ == "__main__":
    main()
