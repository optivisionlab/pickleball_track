"""Time and compare scoring configurations on a set of videos.

Scores every video with the configuration from the environment (PICKLEBALL_*),
prints the time and hit/miss counts, saves each result, and compares it with an
earlier run: largest per-token probability difference and label agreement.

Usage (inside the service image, weights mounted):
    python scripts/bench_videos.py --tag openvino --out /out a.mp4 b.mov
    python scripts/bench_videos.py --tag fp16 --out /out --compare-to openvino a.mp4 b.mov
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pickleball_track.core.config import get_settings
from pickleball_track.models.model_loader import load_models
from pickleball_track.services.inference_service import VideoInferenceService


def compare(reference: dict, result: dict) -> str:
    ref = {round(p["time"], 2): p["probabilities"] for p in reference["temporal_predictions"]}
    new = {round(p["time"], 2): p["probabilities"] for p in result["temporal_predictions"]}
    common = ref.keys() & new.keys()
    if not common:
        return "no common timestamps"
    max_diff = max(abs(ref[t][label] - new[t][label]) for t in common for label in ref[t])
    agree = sum(max(ref[t], key=ref[t].get) == max(new[t], key=new[t].get) for t in common)
    return (
        f"reference hit={reference['total_hits']} miss={reference['total_misses']}, "
        f"max |dp|={max_diff:.4f}, labels equal {agree}/{len(common)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--tag", required=True, help="Name of this configuration (output file prefix).")
    parser.add_argument("--out", type=Path, default=Path("bench-results"))
    parser.add_argument("--compare-to", default=None, help="Tag of an earlier run in --out.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    started = time.perf_counter()
    models = load_models(settings)
    service = VideoInferenceService(
        models, settings.inference, slots=1, decoder=settings.decoder, ffmpeg_threads=settings.ffmpeg_threads
    )
    print(
        f"[{args.tag}] models loaded in {time.perf_counter() - started:.1f}s "
        f"(videomae={settings.videomae_backend}, batch={models.batch_size}, decoder={settings.decoder})",
        flush=True,
    )
    try:
        for video in args.videos:
            started = time.perf_counter()
            result = service.predict(video)
            elapsed = time.perf_counter() - started
            (args.out / f"{args.tag}-{video.stem}.json").write_text(json.dumps(result))
            line = f"[{args.tag}] {video.name}: {elapsed:.1f}s hit={result['total_hits']} miss={result['total_misses']}"
            reference = args.out / f"{args.compare_to}-{video.stem}.json" if args.compare_to else None
            if reference and reference.exists():
                line += " | " + compare(json.loads(reference.read_text()), result)
            print(line, flush=True)
    finally:
        service.close()
        models.close()


# Preprocessing workers are spawned processes: they re-import this module.
if __name__ == "__main__":
    main()
