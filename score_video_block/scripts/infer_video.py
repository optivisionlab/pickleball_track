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
            "Pickleball single-video inference: pretrained VideoMAE + BEATs features "
            "and the trained v4 fold ensemble."
        )
    )
    parser.add_argument("--video", required=True, help="Path to a completely new video.")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint. Default: PICKLEBALL_CHECKPOINT.")
    # Defaults come from core/config.py (or PICKLEBALL_* environment variables).
    parser.add_argument(
        "--event-threshold", type=float, default=defaults.event_threshold,
        help="P(event) threshold for a token (config default 0.4); pass --argmax for the training decoder.",
    )
    parser.add_argument("--argmax", dest="event_threshold", action="store_const", const=None,
                        help="A token is an event when its argmax is hit/miss (training decoder).")
    parser.add_argument("--min-event-tokens", type=int, default=defaults.min_event_tokens,
                        help="Shortest run of event tokens kept as an event (1 token = 1/15 s).")
    parser.add_argument("--smooth-tokens", type=int, default=defaults.smooth_tokens,
                        help="Moving average of P(event) in tokens; 1 disables it.")
    parser.add_argument("--peak", action="store_true", default=defaults.peak,
                        help="Place an event at its most confident token instead of its run centre.")
    parser.add_argument("--output", default=None, help="Output JSON path. Default: <video>.inference.json")
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    args = parse_args(settings.inference)
    configure_logging(settings.log_level)
    if args.checkpoint:
        settings = replace(settings, checkpoint_path=Path(args.checkpoint).expanduser().resolve())

    models = load_models(settings)
    service = VideoInferenceService(
        models,
        InferenceParams(
            event_threshold=args.event_threshold,
            min_event_tokens=args.min_event_tokens,
            smooth_tokens=args.smooth_tokens,
            peak=args.peak,
        ),
    )
    video_path = Path(args.video)
    try:
        result = service.predict(video_path)
    finally:
        models.close()

    output_path = Path(args.output) if args.output else video_path.with_suffix(".inference.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(output_path)


if __name__ == "__main__":
    main()
