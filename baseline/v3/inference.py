

import argparse
import json
from pathlib import Path
from collections import defaultdict
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from transformers import VideoMAEImageProcessor, VideoMAEModel

from config import (
    AUDIO_SR,
    BEATS_CHECKPOINT,
    BEATS_ROOT,
    DEFAULT_VIDEOMAE,
    EVENT_RADIUS_SEC,
    EVENT_WINDOW_SEC,
    IMAGE_SIZE,
    LABEL2ID,
    NUM_TEMPORAL_TOKENS,
    NUM_VIDEO_FRAMES,
    OVERLAP,
)

from model import AVFeatureModel


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Prediction thresholds
MISS_THRESHOLD = 0.50
HIT_THRESHOLD = 0.50
EVENT_MIN_GAP = 0.35
TEMPORAL_SMOOTH_RADIUS = 1


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    num_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if fps <= 0:
        raise RuntimeError(
            f"Invalid FPS: {fps}"
        )

    duration = num_frames / fps

    return {
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


# ============================================================
# VIDEO READER
# ============================================================

def read_video_window(
    video_path,
    start_frame,
    num_frames,
    image_processor,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    frames = []
    valid = []

    for _ in range(num_frames):

        ok, frame = cap.read()

        if not ok:
            valid.append(False)
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)
        valid.append(True)

    cap.release()

    # Pad missing frames by repeating the last frame.
    if len(frames) == 0:
        raise RuntimeError(
            f"Cannot read frames at {start_frame}"
        )

    while len(frames) < num_frames:
        frames.append(frames[-1])
        valid.append(False)

    inputs = image_processor(
        frames,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"]

    # VideoMAE expects:
    # [B, T, C, H, W]
    if pixel_values.ndim != 5:
        raise RuntimeError(
            f"Unexpected VideoMAE input shape: "
            f"{tuple(pixel_values.shape)}"
        )

    return pixel_values, torch.tensor(
        valid,
        dtype=torch.bool,
    )


# ============================================================
# VIDEOMAE
# ============================================================

def adapt_videomae_to_32_frames(model):

    target_frames = NUM_VIDEO_FRAMES

    old_frames = int(
        model.config.num_frames
    )

    tubelet_size = int(
        model.config.tubelet_size
    )

    if old_frames == target_frames:
        return

    if target_frames % tubelet_size != 0:
        raise RuntimeError(
            "NUM_VIDEO_FRAMES must be divisible "
            "by VideoMAE tubelet_size."
        )

    old_t = old_frames // tubelet_size
    new_t = target_frames // tubelet_size

    image_size = int(
        model.config.image_size
    )

    patch_size = int(
        model.config.patch_size
    )

    hp = image_size // patch_size
    wp = image_size // patch_size

    pos = model.embeddings.position_embeddings

    expected_tokens = (
        old_t * hp * wp
    )

    if pos.shape[1] != expected_tokens:
        raise RuntimeError(
            "Unexpected VideoMAE positional embedding shape: "
            f"{tuple(pos.shape)}"
        )

    hidden_dim = pos.shape[-1]

    x = pos.reshape(
        1,
        old_t,
        hp,
        wp,
        hidden_dim,
    )

    x = x.permute(
        0,
        4,
        1,
        2,
        3,
    )

    x = F.interpolate(
        x,
        size=(
            new_t,
            hp,
            wp,
        ),
        mode="trilinear",
        align_corners=False,
    )

    new_pos = (
        x.permute(
            0,
            2,
            3,
            4,
            1,
        )
        .reshape(
            1,
            new_t * hp * wp,
            hidden_dim,
        )
        .contiguous()
    )

    model.embeddings.position_embeddings = (
        torch.nn.Parameter(
            new_pos,
            requires_grad=False,
        )
    )

    model.config.num_frames = target_frames


def load_videomae(device):

    print(
        f"[VideoMAE] Loading pretrained model: "
        f"{DEFAULT_VIDEOMAE}"
    )

    processor = (
        VideoMAEImageProcessor.from_pretrained(
            DEFAULT_VIDEOMAE
        )
    )

    model = VideoMAEModel.from_pretrained(
        DEFAULT_VIDEOMAE
    )

    adapt_videomae_to_32_frames(model)

    # NEVER TRAIN
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    model.to(device)

    return model, processor


# ============================================================
# BEATS
# ============================================================

def load_beats(device):

    import sys

    if BEATS_ROOT:
        sys.path.insert(
            0,
            BEATS_ROOT,
        )
        sys.path.insert(0, os.path.join(BEATS_ROOT, "beats"))

    try:
        from beats.BEATs import (
            BEATs,
            BEATsConfig,
        )
    except ImportError:

        from BEATs import (
            BEATs,
            BEATsConfig,
        )

    print(
        f"[BEATs] Loading pretrained checkpoint: "
        f"{BEATS_CHECKPOINT}"
    )

    checkpoint = torch.load(
        BEATS_CHECKPOINT,
        map_location="cpu",
    )

    cfg = BEATsConfig(
        checkpoint["cfg"]
    )

    model = BEATs(cfg)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    # NEVER TRAIN
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    model.to(device)

    return model


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(video_path):

    print("[Audio] Loading audio...")

    waveform, sr = torchaudio.load(
        str(video_path)
    )

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    waveform = waveform.float()

    if sr != AUDIO_SR:

        waveform = (
            torchaudio.functional.resample(
                waveform,
                sr,
                AUDIO_SR,
            )
        )

    return waveform.contiguous()


# ============================================================
# AUDIO WINDOW
# ============================================================

def get_audio_window(
    waveform,
    start_time,
    duration,
):
    start_sample = max(
        0,
        int(
            round(
                start_time * AUDIO_SR
            )
        ),
    )

    num_samples = max(
        1,
        int(
            round(
                duration * AUDIO_SR
            )
        ),
    )

    end_sample = min(
        waveform.shape[-1],
        start_sample + num_samples,
    )

    segment = waveform[
        :,
        start_sample:end_sample,
    ]

    valid_samples = segment.shape[-1]

    if valid_samples < num_samples:

        padded = torch.zeros(
            1,
            num_samples,
            dtype=torch.float32,
        )

        padded[
            :,
            :valid_samples,
        ] = segment

        segment = padded

    padding_mask = torch.ones(
        1,
        num_samples,
        dtype=torch.bool,
    )

    padding_mask[
        :,
        :valid_samples,
    ] = False

    return (
        segment,
        padding_mask,
    )


# ============================================================
# EXTRACT VIDEO FEATURE
# ============================================================

@torch.inference_mode()
def extract_video_feature(videomae, processor, video_path, start_frame, device):
    pixel_values, frame_valid = read_video_window(
        video_path, start_frame, NUM_VIDEO_FRAMES, processor
    )
    pixel_values = pixel_values.to(device, non_blocking=True)
    output = videomae(pixel_values=pixel_values)
    tokens = output.last_hidden_state

    temporal_tokens = NUM_VIDEO_FRAMES // int(videomae.config.tubelet_size)
    spatial_tokens = (int(videomae.config.image_size) // int(videomae.config.patch_size)) ** 2
    hidden_dim = tokens.shape[-1]
    expected = temporal_tokens * spatial_tokens

    if tokens.shape[1] != expected:
        raise RuntimeError(
            f"VideoMAE token mismatch: got {tokens.shape[1]}, expected {expected}"
        )

    features = (
        tokens.reshape(1, temporal_tokens, spatial_tokens, hidden_dim)
        .mean(dim=2)
    )

    # VideoMAE temporal tokens cover consecutive groups of frames.
    # A token is padding only when every frame in that group is padded.
    frames_per_token = NUM_VIDEO_FRAMES // temporal_tokens
    temporal_padding = ~(
        frame_valid.reshape(temporal_tokens, frames_per_token).any(dim=1)
    )

    return features.float(), temporal_padding.unsqueeze(0).to(device)


# ============================================================
# EXTRACT AUDIO FEATURE
# ============================================================

@torch.inference_mode()
def extract_audio_feature(
    beats,
    waveform,
    start_time,
    duration,
    device,
):

    segment, padding_mask = (
        get_audio_window(
            waveform,
            start_time,
            duration,
        )
    )

    # BEATs requires:
    #
    # waveform      [B, T]
    # padding_mask  [B, T]

    segment = segment.to(
        device,
        non_blocking=True,
    )

    padding_mask = padding_mask.to(
        device,
        non_blocking=True,
    )

    features, feature_padding = (
        beats.extract_features(
            segment,
            padding_mask=padding_mask,
        )
    )

    return (
        features.float(),
        feature_padding.bool(),
    )


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

def build_windows(
    num_frames,
    num_window_frames,
):

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
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model(
    checkpoint_path,
    video_dim,
    audio_dim,
    device,
):

    print(
        f"[Model] Loading checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = AVFeatureModel(
        video_dim=video_dim,
        audio_dim=audio_dim,
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    # The prediction model itself is also inference-only.
    for p in model.parameters():
        p.requires_grad_(False)

    model.to(device)

    return model


# ============================================================
# MAIN INFERENCE
# ============================================================

def inference(
    video_path,
    checkpoint_path,
    output_path=None,
):

    video_path = Path(
        video_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            video_path
        )

    print("=" * 80)
    print("PICKLEBALL SINGLE VIDEO INFERENCE")
    print("=" * 80)

    print(
        f"Video : {video_path}"
    )

    print(
        f"Device: {DEVICE}"
    )

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

    print(
        f"FPS      : {fps:.3f}"
    )

    print(
        f"Frames   : {num_frames}"
    )

    print(
        f"Duration : {duration:.2f}s"
    )

    # --------------------------------------------------------
    # PRETRAINED MODELS
    #
    # These are intentionally loaded again for every new video.
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

    print(
        "[BEATs] Checking feature dimension..."
    )

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

    print(
        f"[Inference] Windows: "
        f"{len(starts)}"
    )

    window_predictions = []

    window_duration = (
        NUM_VIDEO_FRAMES / fps
    )

    for i, start_frame in enumerate(
        starts,
        1,
    ):

        print(
            f"\rProcessing "
            f"{i}/{len(starts)}",
            end="",
            flush=True,
        )

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

    print()

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

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"HIT COUNT : {result['total_hits']}")
    print(f"MISS COUNT: {result['total_misses']}")

    print("\nDetected events:")

    if events:
        for i, event in enumerate(events, 1):
            print(
                f"  #{i:03d} "
                f"{event['label']:>8s} "
                f"@ {event['time']:.2f}s "
                f"(confidence={event['confidence']:.3f})"
            )
    else:
        print("  No events detected.")

    print(f"\nJSON saved to: {json_path}")
    print("=" * 80)

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
        default = "/data/Cuong/pickleball/data/raw/06_12_2025_Dong_Trung_PhaiTay.MOV",
        help="Path to a completely new video.",
    )

    parser.add_argument(
        "--checkpoint",
        default = "/data/Cuong/pickleball/pickleball_track/baseline/v3/checkpoints/fold_1/best.pt",
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