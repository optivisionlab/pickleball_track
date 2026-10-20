"""Measure the largest batch size a model fits on this GPU, and save it for run_models.sh.

The model is loaded once; one real clip, decoded once, is repeated to make batches of
1, 2, 3, 4, 6, 8, 10, ... for each config of the val sweep (all NSVA clips have the same
resolution, so every clip costs the same). For each config the chosen batch is the
largest whose peak reserved memory stays below the GPU total minus --margin-gb.

Usage:
  python find_batch.py --name qwen3-vl-8b --model Qwen/Qwen3-VL-8B-Instruct
  python find_batch.py --name minicpm-v-4.5 --model openbmb/MiniCPM-V-4_5 --pack 3
Result: batch_sizes.json  {name: {"batch_by_config": {"16f_448": B, ...}, "peak_gb": {...}, ...}}
A config with batch 0 does not fit even one clip; run_models.sh skips it.
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

from eval_vlm import DEFAULT_PROMPT
from vlm_backends import BACKENDS, detect_backend

CONFIGS = {"32f_448": (32, 200704), "16f_640": (16, 307200), "16f_448": (16, 200704)}
SIZES = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--pack", type=int, default=1)
    ap.add_argument("--answer-prefix", default="")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--csv", default="splits/val_fold1.csv")
    ap.add_argument("--margin-gb", type=float, default=1.0,
                    help="headroom kept free on the GPU (CUDA context + allocator slack)")
    ap.add_argument("--max-batch", type=int, default=48)
    ap.add_argument("--out", default="batch_sizes.json")
    args = ap.parse_args()
    args.device, args.frame_fps = "cuda", 2.0

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    limit_gb = total_gb - args.margin_gb
    clip = pd.read_csv(args.csv).clip_path.iloc[0]
    name = args.backend if args.backend != "auto" else detect_backend(args.model)

    args.num_frames, args.max_pixels = CONFIGS["32f_448"]
    backend = BACKENDS[name](args, DEFAULT_PROMPT)
    try:
        backend.load_model()
    except torch.OutOfMemoryError:
        print(f"{args.name}: weights do not fit on the GPU -> batch 0")
        save(args, {"batch": 0, "batch_by_config": {c: 0 for c in CONFIGS}, "model_id": args.model, "pack": args.pack, "gpu_gb": round(total_gb, 1),
                    "margin_gb": args.margin_gb, "note": "weights do not fit"})
        return
    base_gb = torch.cuda.memory_allocated() / 1e9
    print(f"{args.name}: weights {base_gb:.1f} GB, GPU {total_gb:.1f} GB, limit {limit_gb:.1f} GB", flush=True)

    sizes = [s for s in SIZES if s <= args.max_batch] if backend.batchable else [1]
    peaks, by_config = {}, {}
    for cfg, (nf, mp) in CONFIGS.items():
        args.num_frames, args.max_pixels = nf, mp
        item = {"i": 0, "path": clip, "err": "", **backend.load_clip(clip)}
        fit = 0
        for b in sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    backend.logits([item] * b)
                torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                print(f"  {cfg} batch {b:2d}: OOM", flush=True)
                break
            peak = torch.cuda.max_memory_reserved() / 1e9
            peaks[f"{cfg}_b{b}"] = round(peak, 2)
            print(f"  {cfg} batch {b:2d}: peak {peak:5.1f} GB  {time.perf_counter() - t0:5.2f} s", flush=True)
            if peak > limit_gb:
                break
            fit = b
        by_config[cfg] = fit
        print(f"  {cfg}: batch {fit}", flush=True)
        del item
    torch.cuda.empty_cache()

    fits = [b for b in by_config.values() if b > 0]
    result = {"batch": min(fits) if fits else 0, "batch_by_config": by_config, "model_id": args.model,
              "pack": args.pack, "weights_gb": round(base_gb, 1), "gpu_gb": round(total_gb, 1),
              "margin_gb": args.margin_gb, "peak_reserved_gb": peaks}
    if not fits:
        print(f"{args.name}: does not fit even with batch 1 within {limit_gb:.1f} GB")
    else:
        print(f"{args.name}: batch by config {by_config}")
    save(args, result)


def save(args, result):
    data = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    data[args.name] = result
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
