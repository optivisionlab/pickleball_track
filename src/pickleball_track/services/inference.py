

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

from ..core.config import (
    DEFAULT_CHECKPOINT_PATH,
    LABEL2ID,
    NUM_TEMPORAL_TOKENS,
    NUM_VIDEO_FRAMES,
    OVERLAP,
)
from ..models.model_loader import (
    load_beats,
    load_trained_model,
    load_videomae,
    resolve_device,
)
from ..pipelines.preprocessing import (
    extract_audio_feature,
    extract_video_feature,
    get_audio_window,
    get_video_info,
    load_audio,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

DEVICE = resolve_device()

# Prediction thresholds
MISS_THRESHOLD = 0.50
HIT_THRESHOLD = 0.50
EVENT_MIN_GAP = 0.35
TEMPORAL_SMOOTH_RADIUS = 1


# ============================================================
# PREDICT ONE WINDOW
# ============================================================

@torch.inference_mode()
def predict_window(
    fusion_model, videomae, processor, beats, waveform, video_path,
    start_frame, fps, duration, device,
):
    video_features, temporal_padding = extract_video_feature(
        videomae, processor, video_path, start_frame, device
    )
    start_time = start_frame / fps
    audio_features, audio_padding = extract_audio_feature(
        beats, waveform, start_time, duration, device
    )
    logits = fusion_model(
        video_features, audio_features, audio_padding, temporal_padding
    )
    return torch.softmax(logits, dim=-1).squeeze(0).float().cpu()


# ============================================================
# SLIDING WINDOWS
# ============================================================

def build_windows(num_frames: int, num_window_frames: int) -> list[int]:

    stride = max(
        1,
        int(
            round(
                num_window_frames
                * (1.0 - OVERLAP)
            )
        ),
    )

    if num_frames <= num_window_frames:
        return [0]

    last_start = (
        num_frames
        - num_window_frames
    )

    starts = list(
        range(
            0,
            last_start + 1,
            stride,
        )
    )

    if starts[-1] != last_start:
        starts.append(
            last_start
        )

    return starts


# ============================================================
# AGGREGATE TEMPORAL PREDICTIONS
# ============================================================

def aggregate_predictions(window_predictions, fps):
    """Average predictions from overlapping windows at the same time."""
    votes = defaultdict(list)
    frames_per_token = NUM_VIDEO_FRAMES // NUM_TEMPORAL_TOKENS

    for item in window_predictions:
        start_frame = int(item["start_frame"])
        probs = item["probs"]
        for token_idx in range(NUM_TEMPORAL_TOKENS):
            center_frame = (
                start_frame + token_idx * frames_per_token
                + frames_per_token / 2.0
            )
            key = round(float(center_frame / fps), 4)
            votes[key].append(probs[token_idx])

    times = sorted(votes)
    if not times:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0, len(LABEL2ID)), dtype=np.float32),
        )

    probs = [torch.stack(votes[t], dim=0).mean(dim=0) for t in times]
    return np.asarray(times, dtype=np.float32), torch.stack(probs).numpy()


def smooth_probabilities(probs, radius=1):
    """Light moving-average smoothing on class probabilities."""
    probs = np.asarray(probs, dtype=np.float32)
    if radius <= 0 or len(probs) <= 1:
        return probs.copy()
    out = np.empty_like(probs)
    for i in range(len(probs)):
        lo=max(0, i-radius); hi=min(len(probs), i+radius+1)
        out[i]=probs[lo:hi].mean(axis=0)
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)


def _label_from_id(label_id):
    for name, idx in LABEL2ID.items():
        if int(idx) == int(label_id):
            return name
    raise RuntimeError(f"Unknown class id: {label_id}")


def _event_from_indices(times, probs, label_id, indices):
    if not indices:
        return None
    vals=probs[indices, label_id]
    best_idx=indices[int(np.argmax(vals))]
    return {
        "time": round(float(times[best_idx]),4),
        "label": _label_from_id(label_id),
        "confidence": round(float(probs[best_idx,label_id]),6),
        "start_time": round(float(times[indices[0]]),4),
        "end_time": round(float(times[indices[-1]]),4),
    }


def merge_close_action_events(events, max_distance=EVENT_MIN_GAP):
    """Merge only same-class events closer than max_distance."""
    if not events: return []
    merged=[]
    for e in sorted((dict(x) for x in events), key=lambda x:x["time"]):
        if merged and e["label"] in {"hit","miss"} and e["label"]==merged[-1]["label"] and e["time"]-merged[-1]["time"] <= max_distance:
            prev=merged[-1]
            best=e if e["confidence"]>prev["confidence"] else prev
            best=dict(best)
            best["start_time"]=round(min(prev["start_time"],e["start_time"]),4)
            best["end_time"]=round(max(prev["end_time"],e["end_time"]),4)
            merged[-1]=best
        else:
            merged.append(e)
    return merged


def build_temporal_events(times, probs, hit_threshold=HIT_THRESHOLD, miss_threshold=MISS_THRESHOLD, smoothing_radius=TEMPORAL_SMOOTH_RADIUS, max_event_gap=EVENT_MIN_GAP):
    """Convert temporal probabilities to physical Hit/Miss events."""
    if len(times)==0: return []
    no_id=LABEL2ID["no_event"]; hit_id=LABEL2ID["hit"]; miss_id=LABEL2ID["miss"]
    sm=smooth_probabilities(probs, smoothing_radius)
    labels=[]
    for p in sm:
        h=float(p[hit_id]); m=float(p[miss_id]); n=float(p[no_id])
        if h>=hit_threshold and h>=m and h>=n: labels.append(hit_id)
        elif m>=miss_threshold and m>=h and m>=n: labels.append(miss_id)
        else: labels.append(no_id)

    # Fill a short no-event gap only when the same action is on both sides.
    i=0
    while i<len(labels):
        if labels[i] not in (hit_id,miss_id): i+=1; continue
        action=labels[i]; j=i+1
        while j<len(labels) and labels[j]==no_id: j+=1
        if j<len(labels) and labels[j]==action and times[j]-times[i] <= max_event_gap:
            labels[i+1:j]=[action]*(j-i-1)
            i=j
        else: i+=1

    raw=[]; cur=None; idxs=[]
    for idx,label in enumerate(labels):
        if label in (hit_id,miss_id):
            if cur is None: cur=label; idxs=[idx]
            elif label==cur: idxs.append(idx)
            else:
                e=_event_from_indices(times,sm,cur,idxs)
                if e: raw.append(e)
                cur=label; idxs=[idx]
        elif cur is not None:
            e=_event_from_indices(times,sm,cur,idxs)
            if e: raw.append(e)
            cur=None; idxs=[]
    if cur is not None:
        e=_event_from_indices(times,sm,cur,idxs)
        if e: raw.append(e)
    return merge_close_action_events(raw, max_event_gap)


def build_json_result(video_path, times, probs, events):
    hit_events=[e for e in events if e["label"]=="hit"]
    miss_events=[e for e in events if e["label"]=="miss"]
    no_event_segments=[e for e in events if e["label"]=="no_event"]
    temporal_predictions=[]
    for time,p in zip(times,probs):
        label_id=int(np.argmax(p))
        temporal_predictions.append({
            "time":round(float(time),4),
            "label":_label_from_id(label_id),
            "confidence":round(float(p[label_id]),6),
            "probabilities":{_label_from_id(i):round(float(p[i]),6) for i in range(len(p))},
        })
    return {
        "video":str(video_path),
        "total_hits":len(hit_events),
        "total_misses":len(miss_events),
        "events":events, "hit_events":hit_events, "miss_events":miss_events,
        "no_event_segments":no_event_segments, "temporal_predictions":temporal_predictions,
    }


def save_json_result(result, output_path):
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    return output_path


# ============================================================
# MAIN INFERENCE
# ============================================================

def inference(
    video_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:

    video_path = Path(
        video_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            video_path
        )

    logger.info("Video inference started", extra={"device": str(DEVICE)})

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    info = get_video_info(
        video_path
    )

    fps = info["fps"]
    num_frames = info[
        "num_frames"
    ]

    duration = info[
        "duration"
    ]

    logger.info(
        "Video metadata loaded",
        extra={"fps": fps, "frames": num_frames, "video_duration_seconds": duration},
    )

    # --------------------------------------------------------
    # PRETRAINED MODELS
    #
    # These immutable models are cached for reuse across jobs.
    # --------------------------------------------------------

    videomae, processor = (
        load_videomae(
            DEVICE
        )
    )

    beats = load_beats(
        DEVICE
    )

    waveform = load_audio(
        video_path
    )

    # --------------------------------------------------------
    # Build fusion model from first feature dimensions.
    #
    # VideoMAE dimension is known directly.
    # BEATs dimension is obtained by extracting one small
    # audio feature before loading fusion checkpoint.
    # --------------------------------------------------------

    dummy_duration = min(
        duration,
        NUM_VIDEO_FRAMES / fps,
    )

    dummy_audio, dummy_padding = (
        get_audio_window(
            waveform,
            0.0,
            dummy_duration,
        )
    )

    dummy_audio = dummy_audio.to(
        DEVICE
    )

    dummy_padding = dummy_padding.to(
        DEVICE
    )

    with torch.inference_mode():

        dummy_features, _ = (
            beats.extract_features(
                dummy_audio,
                padding_mask=dummy_padding,
            )
        )

    video_dim = int(
        videomae.config.hidden_size
    )

    audio_dim = int(
        dummy_features.shape[-1]
    )

    fusion_model = (
        load_trained_model(
            checkpoint_path,
            video_dim,
            audio_dim,
            DEVICE,
        )
    )

    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    starts = build_windows(
        num_frames,
        NUM_VIDEO_FRAMES,
    )

    logger.info("Running temporal windows", extra={"window_count": len(starts)})

    window_predictions = []

    window_duration = (
        NUM_VIDEO_FRAMES / fps
    )

    for start_frame in starts:
        probs = predict_window(
            fusion_model,
            videomae,
            processor,
            beats,
            waveform,
            video_path,
            start_frame,
            fps,
            window_duration,
            DEVICE,
        )

        window_predictions.append(
            {
                "start_frame": start_frame,
                "probs": probs,
            }
        )

    # --------------------------------------------------------
    # Aggregate overlapping windows
    # --------------------------------------------------------

    times, probs = (
        aggregate_predictions(
            window_predictions,
            fps,
        )
    )

    # --------------------------------------------------------
    # Convert all 3-class temporal predictions into events.
    # --------------------------------------------------------

    events = build_temporal_events(
        times, probs,
        hit_threshold=HIT_THRESHOLD,
        miss_threshold=MISS_THRESHOLD,
        smoothing_radius=TEMPORAL_SMOOTH_RADIUS,
        max_event_gap=EVENT_MIN_GAP,
    )

    result = build_json_result(
        video_path,
        times,
        probs,
        events,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = (
        Path(output_path)
        if output_path is not None
        else video_path.with_suffix(".inference.json")
    )

    save_json_result(
        result,
        json_path,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    logger.info(
        "Video inference completed",
        extra={
            "hit_count": result["total_hits"],
            "miss_count": result["total_misses"],
        },
    )

    return result


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Pickleball single-video inference "
            "using pretrained VideoMAE + BEATs "
            "and trained temporal fusion model."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to a completely new video.",
    )

    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to trained fold/best checkpoint.",
    )

    parser.add_argument(
        "--miss-threshold",
        type=float,
        default=MISS_THRESHOLD,
        help="Miss probability threshold.",
    )

    parser.add_argument("--hit-threshold", type=float, default=HIT_THRESHOLD, help="Hit probability threshold.")
    parser.add_argument("--event-gap", type=float, default=EVENT_MIN_GAP, help="Maximum gap for merging the same action event.")
    parser.add_argument("--smooth-radius", type=int, default=TEMPORAL_SMOOTH_RADIUS, help="Probability smoothing radius; 0 disables smoothing.")

    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: <video>.inference.json",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    MISS_THRESHOLD = args.miss_threshold
    HIT_THRESHOLD = args.hit_threshold
    EVENT_MIN_GAP = args.event_gap
    TEMPORAL_SMOOTH_RADIUS = args.smooth_radius

    result = inference(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )