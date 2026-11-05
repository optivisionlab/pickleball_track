"""Pickleball clips for the zero-shot VLM benchmark, on the same folds as the custom model.

The pickleball labels are timestamps (the moment the ball meets / passes the paddle),
not pre-cut clips. As in training (event radius 0.5 s), one clip = ~1 s around a time:

  * hit / miss: one clip per labelled event, centred on its time;
  * no-event:   the middle of each gap between two strokes (and of the lead-in / tail
    of a video), at least NOEVENT_GAP s from every event, so no event is within 0.5 s
    of any frame of the clip (the custom model's no_event rule). Capped at the number
    of hit clips (random, seed 0).

A clip has 32 frames at 30 fps (1.07 s, the custom model's window length): frame
k of a clip centred at time t is source frame round((round(t*30) - 16 + k) * fps / 30),
clamped to the video, i.e. the same frame sampling as pickleball_common.decode_sampled.
Clips are downscaled to <= 960x540 px and named vNNN_tMMMMMMM.mp4 (no label in the name:
the source names contain Trung/Truot = hit/miss).

Folds: test users of fold k = those of the custom model (make_folds of pickleball_common,
StratifiedGroupKFold by user, seed 42; checked against weights_pickleball/v3/fold_k/result.json).
val_fold1.csv = clips of the users of fold 1's inner val split (picks the VLM frame config).

Custom predictions for the same clips (McNemar): the custom model's test probabilities
(weights_pickleball/<v>/test_predictions.pt) are stitched into one timeline per video
as in pickleball_common.summarize; the tokens covering the clip get its label rule
(any hit token -> hit, else any miss token -> miss, else no_event).

  /data/miniconda/envs/anything/bin/python make_splits_pickleball.py
"""
import argparse
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "/data/Cuong/pickleball/pickleball_track/baseline"
NAMES = {0: "no_event", 1: "hit", 2: "miss"}          # custom ids
FPS = 30.0
N_FRAMES = 32
MAX_PIXELS = 960 * 540
NOEVENT_GAP = 1.1      # s between a no-event clip centre and any event (0.53 half clip + 0.5 radius)


def clip_name(vid, t):
    return f"v{vid:03d}_t{int(round(t * 1000)):07d}.mp4"


def clip_slots(t, n_sampled):
    """Sampled-frame slots (30 fps grid) shown by the clip centred at t."""
    first = int(round(t * FPS)) - N_FRAMES // 2
    return np.clip(first + np.arange(N_FRAMES), 0, n_sampled - 1)


def out_size(w, h):
    scale = min(1.0, (MAX_PIXELS / (w * h)) ** 0.5)
    return int(w * scale) // 2 * 2, int(h * scale) // 2 * 2


def cut_video(job):
    """Decode one video once (like decode_sampled) and write all its clips."""
    import cv2
    cv2.setNumThreads(1)
    vid, info, times, clips_dir = job
    todo = sorted(t for t in times if not (Path(clips_dir) / clip_name(vid, t)).is_file())
    if not todo:
        return vid, 0
    n_sampled = info["num_sampled"]
    targets = np.minimum(np.round(np.arange(n_sampled) * info["fps"] / FPS).astype(np.int64),
                         info["num_frames"] - 1)
    need = {t: clip_slots(t, n_sampled) for t in todo}
    pending = sorted(todo, key=lambda t: need[t].max())
    cap = cv2.VideoCapture(info["path"])
    position, decoded_at, last, size = -1, -1, None, None
    buf, written = {}, 0
    for slot, target in enumerate(targets):
        while position < target and cap.grab():
            position += 1
        if position == target and decoded_at != target:
            ok, frame = cap.retrieve()
            if ok:
                size = size or out_size(frame.shape[1], frame.shape[0])
                last, decoded_at = cv2.resize(frame, size, interpolation=cv2.INTER_AREA), target
        if last is not None:
            buf[slot] = last                           # past the real end: last frame repeated
        while pending and need[pending[0]].max() <= slot:
            t = pending.pop(0)
            frames = [buf.get(int(k), last) for k in need[t]]
            tmp = Path(clips_dir) / (clip_name(vid, t) + ".tmp.mp4")
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-threads", "1",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)]
            subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)
            tmp.replace(Path(clips_dir) / clip_name(vid, t))
            written += 1
        if not pending:
            break
        keep = min(int(need[t].min()) for t in pending)
        for k in [k for k in buf if k < keep]:
            del buf[k]
    cap.release()
    if pending:
        raise RuntimeError(f"video {vid}: {len(pending)} clips not written ({info['path']})")
    return vid, written


def noevent_times(events, duration):
    """Middle of every event-free stretch that is long enough, plus extra centres in long stretches."""
    bounds = [0.0] + sorted(e[0] for e in events) + [duration]
    out = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        lo = a + (NOEVENT_GAP if i > 0 else 0.6)                  # video start: only half a clip
        hi = b - (NOEVENT_GAP if i < len(bounds) - 2 else 0.6)    # video end
        if hi < lo:
            continue
        n = max(1, int((hi - lo) // 2.5) + 1)                     # one centre per ~2.5 s
        out += list(np.linspace(lo, hi, n + 2)[1:-1]) if n > 1 else [(lo + hi) / 2]
    return out


def stitched_timeline(data, probs, vid, info):
    """Average of the overlapping test windows' token probabilities (as summarize())."""
    rows = (data["video_index"] == vid).nonzero().flatten().tolist()
    valid = data["token_valid"]
    length = math.ceil(info["num_sampled"] / 2)
    acc, count = np.zeros((length, 3)), np.zeros(length)
    for r in rows:
        slot = int(data["starts"][r]) // 2 + np.arange(16)
        ok = valid[r].numpy() & (slot < length)
        acc[slot[ok]] += probs[r][ok]
        count[slot[ok]] += 1
    timeline = np.zeros_like(acc)
    timeline[:, 0] = 1.0
    timeline[count > 0] = acc[count > 0] / count[count > 0, None]
    return timeline


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", default=BASELINE)
    ap.add_argument("--features", default=f"{BASELINE}/pickleball/cache/features_b91a91b4244d3b52.pt",
                    help="custom model feature cache: video list, windows (for the folds)")
    ap.add_argument("--weights", default=f"{BASELINE}/pickleball/outputs/runs")
    ap.add_argument("--clips-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips_pickleball"))
    ap.add_argument("--out", default=os.path.join(here, "splits_pickleball"))
    ap.add_argument("--custom-out", default=os.path.join(here, "custom_preds_pickleball"))
    ap.add_argument("--event-radius", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    sys.path[:0] = [args.baseline_dir]
    import torch
    import pickleball_common as pc

    data = torch.load(args.features, map_location="cpu", weights_only=False, mmap=True)
    data = {k: data[k] for k in ("videos", "video_index", "starts", "token_valid")}
    videos = data["videos"]

    # folds exactly as the custom model: make_folds on its windows, grouped by user
    _, window_class = pc.build_labels(data, args.event_radius)
    wusers = np.array([videos[int(v)]["user"] for v in data["video_index"]])
    fold_of_user, val_users = {}, None
    for fold, _, val_idx, test_idx in pc.make_folds(window_class.numpy(), wusers, args.folds, args.seed):
        test_users = sorted(set(wusers[test_idx].tolist()))
        ref = Path(args.weights) / "v3" / f"fold_{fold}" / "result.json"
        if ref.is_file():
            expected = sorted(json.load(open(ref))["users"]["test"])
            assert test_users == expected, f"fold {fold}: test users {test_users} != custom {expected}"
        fold_of_user.update({u: fold for u in test_users})
        if fold == 1:
            val_users = sorted(set(wusers[val_idx].tolist()))
        print(f"fold {fold}: test users {test_users}")

    rows = []
    for vid, info in enumerate(videos):
        events = [tuple(e) for e in info["events"]]
        for t, c in events:
            rows.append((vid, float(t), NAMES[int(c)]))
        for t in noevent_times(events, info["duration"]):
            rows.append((vid, float(t), "no_event"))
    df = pd.DataFrame(rows, columns=["vid", "time", "label"])
    n_hit = int((df.label == "hit").sum())
    ne = df[df.label == "no_event"]
    if len(ne) > n_hit:
        df = df.drop(ne.sample(len(ne) - n_hit, random_state=0).index)
    df = df.sort_values(["vid", "time"]).reset_index(drop=True)
    df["user"] = [videos[v]["user"] for v in df.vid]
    df["fold"] = df.user.map(fold_of_user)
    df["clip_path"] = [os.path.join(args.clips_dir, clip_name(v, t)) for v, t in zip(df.vid, df.time)]
    df["video_id"] = [f"v{v:03d}_t{t:.3f}" for v, t in zip(df.vid, df.time)]
    df["game_id"] = df.user                      # the grouping unit, like game_id for NSVA
    df["source"] = [Path(videos[v]["path"]).name for v in df.vid]
    cols = ["clip_path", "label", "fold", "video_id", "game_id", "user", "time", "source"]
    os.makedirs(args.out, exist_ok=True)
    df[cols].to_csv(os.path.join(args.out, "test.csv"), index=False)
    val = df[df.user.isin(val_users)]
    val[cols].to_csv(os.path.join(args.out, "val_fold1.csv"), index=False)
    print(f"{len(df)} clips {df.label.value_counts().to_dict()}; per fold "
          f"{df.groupby('fold').size().to_dict()}; val_fold1 {len(val)} clips (users {val_users})")

    # custom model predictions on exactly these clips
    os.makedirs(args.custom_out, exist_ok=True)
    for v in ("v3", "v4", "v5"):
        p = Path(args.weights) / v / "test_predictions.pt"
        if not p.is_file():
            continue
        probs = torch.load(p, map_location="cpu", weights_only=False)["probs"].numpy()
        out = []
        for vid, g in df.groupby("vid"):
            tl = stitched_timeline(data, probs, vid, videos[vid])
            for _, r in g.iterrows():
                tok = np.unique(clip_slots(r.time, videos[vid]["num_sampled"]) // 2)
                tok = tok[tok < len(tl)]
                pt = tl[tok]
                arg = pt.argmax(-1)
                cls = 1 if (arg == 1).any() else 2 if (arg == 2).any() else 0
                score = pt.max(0) / pt.max(0).sum()           # only used for AUC
                out.append({"clip_path": r.clip_path, "pred": NAMES[cls], "p_made": score[1],
                            "p_miss": score[2], "p_no_event": score[0], "fold": r.fold,
                            "video_id": r.video_id})
        cdf = pd.DataFrame(out).set_index("clip_path").loc[df.clip_path].reset_index()
        cdf.to_csv(os.path.join(args.custom_out, f"custom_preds_{v}.csv"), index=False)
        print(f"custom {v}: clip accuracy {(cdf.pred.values == df.label.values).mean():.4f}")

    os.makedirs(args.clips_dir, exist_ok=True)
    jobs = [(vid, videos[vid], g.time.tolist(), args.clips_dir) for vid, g in df.groupby("vid")]
    with Pool(args.workers) as pool:
        for k, (vid, n) in enumerate(pool.imap_unordered(cut_video, jobs), 1):
            print(f"  clips: video {vid:3d} +{n} ({k}/{len(jobs)})", flush=True)
    missing = [p for p in df.clip_path if not os.path.isfile(p)]
    assert not missing, f"{len(missing)} clips missing, e.g. {missing[:3]}"
    print(f"all {len(df)} clips in {args.clips_dir}")


if __name__ == "__main__":
    main()
