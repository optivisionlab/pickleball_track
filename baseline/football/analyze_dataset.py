"""Football (SoccerNet shots, 15 s) dataset analysis: figures + DATA_REPORT.md.

    python analyze_dataset.py                 # measure every clip (CPU, cached) + figures
    python analyze_dataset.py --asr-sample 450  # also Whisper on a stratified sample (GPU)

Per clip (cached in ANALYSIS_DIR/per_clip.npz + per_clip.csv):
    ffprobe   : resolution, fps, frames, duration, audio stream, file size
    audio     : RMS loudness (dB) in 0.25 s bins over the 15 s (16 kHz mono)
    visual    : 5 fps grey 64x36 frames -> brightness, mean abs frame difference,
                scene cuts (difference spikes)
Figures go to baseline/football/data_report/ (fig_*.png) next to DATA_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb_common as fc  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent / "data_report"
AUDIO_SR, AUDIO_BIN = 16000, 0.25
N_AUDIO_BINS = int(fc.CLIP_SECONDS / AUDIO_BIN)          # 60
VIS_FPS, VIS_W, VIS_H = 5, 64, 36
N_VIS = int(fc.CLIP_SECONDS * VIS_FPS)                   # 75

# Fixed categorical order (dataviz reference palette, light surface); class colour never changes.
CLASS_COLOR = {"goal": "#2a78d6", "saved": "#eb6834", "off_target": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


# ----------------------------------------------------------------------------- measurement
def probe(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration,sample_rate,channels",
           "-show_entries", "format=duration,size", "-of", "json", path]
    try:
        info = json.loads(subprocess.run(cmd, capture_output=True, timeout=60).stdout or b"{}")
    except Exception:
        info = {}
    out = {"width": 0, "height": 0, "fps": 0.0, "frames": 0, "v_codec": "", "has_audio": False,
           "a_sr": 0, "a_channels": 0, "duration": 0.0, "size_mb": 0.0}
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and not out["width"]:
            num, den = (s.get("r_frame_rate", "0/1") + "/1").split("/")[:2]
            out.update(width=int(s.get("width", 0)), height=int(s.get("height", 0)),
                       fps=float(num) / max(float(den), 1.0), frames=int(s.get("nb_frames", 0) or 0),
                       v_codec=s.get("codec_name", ""))
        elif s.get("codec_type") == "audio":
            out.update(has_audio=True, a_sr=int(s.get("sample_rate", 0) or 0),
                       a_channels=int(s.get("channels", 0) or 0))
    fmt = info.get("format", {})
    out["duration"] = float(fmt.get("duration", 0) or 0)
    out["size_mb"] = float(fmt.get("size", 0) or 0) / 2 ** 20
    return out


def audio_curve(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", path, "-vn", "-ac", "1",
           "-ar", str(AUDIO_SR), "-f", "f32le", "-"]
    curve = np.full(N_AUDIO_BINS, np.nan, np.float32)
    try:
        wave = np.frombuffer(subprocess.run(cmd, capture_output=True, timeout=120).stdout, np.float32)
    except Exception:
        return curve
    step = int(AUDIO_SR * AUDIO_BIN)
    for b in range(N_AUDIO_BINS):
        seg = wave[b * step:(b + 1) * step]
        if len(seg) >= step // 2:
            curve[b] = 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-6)
    return curve


def visual_curves(path: str):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", path,
           "-vf", f"fps={VIS_FPS},scale={VIS_W}:{VIS_H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    bright = np.full(N_VIS, np.nan, np.float32)
    diff = np.full(N_VIS, np.nan, np.float32)
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    except Exception:
        return bright, diff
    n = min(len(raw) // (VIS_W * VIS_H), N_VIS)
    if n == 0:
        return bright, diff
    fr = np.frombuffer(raw[: n * VIS_W * VIS_H], np.uint8).reshape(n, VIS_H, VIS_W).astype(np.float32)
    bright[:n] = fr.mean((1, 2))
    diff[1:n] = np.abs(np.diff(fr, axis=0)).mean((1, 2))
    return bright, diff


def measure(path: str):
    p = probe(path)
    a = audio_curve(path) if p["has_audio"] else np.full(N_AUDIO_BINS, np.nan, np.float32)
    b, d = visual_curves(path)
    return p, a, b, d


def cuts_from_diff(diff: np.ndarray) -> np.ndarray:
    """Scene cut = frame difference spike: > 20 grey levels and > 3x the clip median."""
    med = np.nanmedian(diff, axis=1, keepdims=True)
    return (diff > 20) & (diff > 3 * np.maximum(med, 1.0))


def load_or_measure(clips, workers: int):
    cache = Path(fc.ANALYSIS_DIR) / "per_clip.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        if len(z["audio"]) == len(clips):
            print(f"[MEASURE] cached {cache}")
            return list(z["probe"]), z["audio"], z["bright"], z["diff"]
    print(f"[MEASURE] {len(clips)} clips with {workers} processes (ffprobe + audio + 5 fps grey frames)")
    paths = [c.path for c in clips]
    with Pool(workers) as pool:
        res = []
        for i, r in enumerate(pool.imap(measure, paths, chunksize=4)):
            res.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(paths)}", flush=True)
    probes = [r[0] for r in res]
    audio = np.stack([r[1] for r in res])
    bright = np.stack([r[2] for r in res])
    diff = np.stack([r[3] for r in res])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, probe=np.array(probes, dtype=object), audio=audio, bright=bright, diff=diff)
    return probes, audio, bright, diff


# ----------------------------------------------------------------------------- ASR sample
GOAL_WORDS = r"\b(goal|goals|gol|golo|golazo|gooo+l|tor|toooor|but|rete|scores?|scored|net|гол|забил)\b"
SAVE_WORDS = r"\b(save[sd]?|keeper|goalkeeper|parata|parad[ao]|atajad[ao]|arrêt|pariert|portero|portiere|gardien)\b"
MISS_WORDS = r"\b(wide|over|miss(es|ed)?|off target|fuori|fuera|vorbei|à côté|post|bar|crossbar|palo)\b"


def asr_sample(clips, n: int, seed: int = 0):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    rng = np.random.default_rng(seed)
    labels = np.array([c.label for c in clips])
    pick = np.concatenate([rng.choice(np.where(labels == k)[0], n // fc.NUM_CLASSES, replace=False)
                           for k in range(fc.NUM_CLASSES)])
    name = "openai/whisper-large-v3-turbo"
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    out = []
    for s in range(0, len(pick), 16):
        idx = pick[s:s + 16]
        waves = [fc.bc.read_audio(clips[i].path) for i in idx]
        feats = proc(waves, sampling_rate=AUDIO_SR, return_tensors="pt").input_features.cuda().half()
        with torch.inference_mode():
            ids = model.generate(feats, task="transcribe", max_new_tokens=160)
        raw = proc.batch_decode(ids, skip_special_tokens=False)
        text = proc.batch_decode(ids, skip_special_tokens=True)
        for i, r, t in zip(idx, raw, text):
            lang = re.search(r"<\|([a-z]{2,3})\|>", r.replace("<|startoftranscript|>", ""))
            out.append({"index": int(i), "label": fc.LABELS[labels[i]], "lang": lang.group(1) if lang else "?",
                        "text": t.strip()})
        print(f"  ASR {len(out)}/{len(pick)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.family": "DejaVu Sans", "figure.dpi": 110,
    })
    return plt


def save(fig, name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_DIR / name, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return name


def bars_by_split(plt, df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    x = np.arange(len(fc.SPLITS))
    w = 0.26
    for k, lab in enumerate(fc.LABELS):
        vals = [int(((df.split == s) & (df.outcome == lab)).sum()) for s in fc.SPLITS]
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=CLASS_COLOR[lab], label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 30, f"{v}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"{s}\n({int((df.split == s).sum())} clip)" for s in fc.SPLITS])
    ax.set_ylabel("số clip")
    ax.set_title("Số clip theo split và nhãn")
    ax.legend(ncol=3, loc="upper right")
    ax = axes[1]
    left = np.zeros(len(fc.SPLITS))
    for lab in fc.LABELS:
        share = np.array([((df.split == s) & (df.outcome == lab)).sum() / (df.split == s).sum() for s in fc.SPLITS])
        ax.barh(x, share, left=left, color=CLASS_COLOR[lab], edgecolor=SURFACE, linewidth=2, label=lab)
        for i, (l, v) in enumerate(zip(left, share)):
            ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += share
    ax.set_yticks(x, fc.SPLITS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Tỷ lệ nhãn trong từng split")
    ax.grid(False)
    return save(fig, "fig01_class_by_split.png")


def games_fig(plt, df):
    g = df.groupby(["split", "game_idx"])
    per_game = g.size().rename("n").reset_index()
    goals = df[df.outcome == "goal"].groupby(["split", "game_idx"]).size()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    for ax, s in zip(axes, fc.SPLITS):
        n = per_game[per_game.split == s].n.values
        ax.hist(n, bins=np.arange(0, n.max() + 2) - 0.5, color=SEQ[2], edgecolor=SURFACE, linewidth=1)
        ng = int((per_game.split == s).sum())
        ax.set_title(f"{s}: {ng} trận, {n.mean():.1f} clip/trận")
        ax.set_xlabel("số clip trong 1 trận")
        gs = goals.loc[s] if s in goals.index.get_level_values(0) else []
        ax.text(0.98, 0.95, f"goal/trận: {np.sum(gs) / ng:.1f}", transform=ax.transAxes, ha="right",
                va="top", fontsize=8, color=INK2)
    axes[0].set_ylabel("số trận")
    return save(fig, "fig02_clips_per_game.png")


def league_fig(plt, df):
    leagues = df.league.value_counts().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axes[0]
    y = np.arange(len(leagues))
    left = np.zeros(len(leagues))
    for lab in fc.LABELS:
        v = np.array([((df.league == l) & (df.outcome == lab)).sum() for l in leagues])
        ax.barh(y, v, left=left, color=CLASS_COLOR[lab], edgecolor=SURFACE, linewidth=2, label=lab)
        left += v
    for i, t in enumerate(left):
        ax.text(t + 20, i, f"{int(t)}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, leagues)
    ax.invert_yaxis()
    ax.set_title("Số clip theo giải đấu")
    ax.legend(ncol=3, loc="lower right")
    ax = axes[1]
    mat = np.array([[((df.league == l) & (df.split == s)).sum() / max((df.split == s).sum(), 1)
                     for s in fc.SPLITS] for l in leagues])
    im = ax.imshow(mat, cmap=_seq_cmap(), aspect="auto", vmin=0)
    for i in range(len(leagues)):
        for j in range(len(fc.SPLITS)):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > mat.max() * 0.55 else INK)
    ax.set_xticks(range(len(fc.SPLITS)), fc.SPLITS)
    ax.set_yticks(range(len(leagues)), leagues)
    ax.grid(False)
    ax.set_title("Tỷ lệ giải đấu trong mỗi split")
    return save(fig, "fig03_leagues.png")


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", SEQ)


def stacked_share(ax, df, col, order, title):
    y = np.arange(len(fc.LABELS))
    left = np.zeros(len(fc.LABELS))
    shades = SEQ[1:] if len(order) <= 4 else SEQ
    for k, val in enumerate(order):
        v = np.array([((df.outcome == lab) & (df[col] == val)).sum() / max((df.outcome == lab).sum(), 1)
                      for lab in fc.LABELS])
        ax.barh(y, v, left=left, color=shades[k % len(shades)], edgecolor=SURFACE, linewidth=2, label=str(val))
        for i, (l, s) in enumerate(zip(left, v)):
            if s > 0.07:
                ax.text(l + s / 2, i, f"{s:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if k >= 1 else INK)
        left += v
    ax.set_yticks(y, fc.LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(False)
    ax.set_title(title)
    ax.legend(ncol=len(order), loc="upper center", bbox_to_anchor=(0.5, -0.12))


def label_meta_fig(plt, df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 5.6))
    d = df.copy()
    d["chain"] = d.chain_len.replace("", "goal").map(lambda v: v if v in ("goal", "1", "2") else ("3+" if v else v))
    stacked_share(axes[0, 0], d, "chain", ["goal", "1", "2", "3+"], "Số cú sút trong chuỗi (chain_len)")
    stacked_share(axes[0, 1], d, "visibility", sorted(d.visibility.unique()), "visibility (nhãn SoccerNet)")
    stacked_share(axes[1, 0], d, "team", sorted(d.team.unique()), "Đội sút (home / away)")
    stacked_share(axes[1, 1], d, "video_res", sorted(d.video_res.unique()), "Nguồn video")
    fig.tight_layout()
    return save(fig, "fig04_label_metadata.png")


def minute_fig(plt, df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
    bins = np.arange(0, 56, 2.5)
    for ax, half in zip(axes, ("1", "2")):
        for lab in fc.LABELS:
            m = df[(df.half == half) & (df.outcome == lab)].minute.values
            h, _ = np.histogram(m, bins=bins)
            ax.plot(bins[:-1] + 1.25, h / max(len(df[df.outcome == lab]), 1), color=CLASS_COLOR[lab],
                    lw=2, marker="o", ms=3, label=lab)
        ax.set_title(f"Hiệp {half}: phút của cú sút (tỷ lệ trong mỗi nhãn)")
        ax.set_xlabel("phút trong hiệp")
    axes[0].set_ylabel("tỷ lệ clip")
    axes[0].legend()
    return save(fig, "fig05_game_minute.png")


def overlap_fig(plt, df):
    gaps = np.full(len(df), np.inf)
    for _, g in df.groupby(["game_idx", "half"]):
        pos = g.position_ms.values.astype(np.int64)
        for j, i in enumerate(g.index.values):
            d = np.abs(np.delete(pos, j) - pos[j])
            if len(d):
                gaps[i] = d.min()
    df["gap_s"] = gaps / 1000.0
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = [0, 5, 15, 30, 60, 120, 300, 1e9]
    names = ["<5 s", "5-15 s", "15-30 s", "30-60 s", "1-2 phút", "2-5 phút", ">5 phút"]
    x = np.arange(len(names))
    w = 0.26
    for k, lab in enumerate(fc.LABELS):
        g = df[df.outcome == lab].gap_s.values
        h, _ = np.histogram(g, bins=bins)
        ax.bar(x + (k - 1) * w, h / len(g), w - 0.02, color=CLASS_COLOR[lab], label=lab)
    ax.set_xticks(x, names)
    ax.set_ylabel("tỷ lệ clip")
    ax.set_title("Khoảng cách tới clip gần nhất cùng trận + hiệp (< 15 s = 2 clip chồng lên nhau)")
    ax.legend()
    return save(fig, "fig06_neighbour_gap.png")


def video_props_fig(plt, df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ax = axes[0]
    res = (df.width.astype(str) + "x" + df.height.astype(str)).value_counts()
    ax.barh(range(len(res)), res.values, color=SEQ[2])
    ax.set_yticks(range(len(res)), res.index)
    ax.invert_yaxis()
    for i, v in enumerate(res.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8, color=INK2)
    ax.set_title("Độ phân giải")
    ax = axes[1]
    fps = df.fps.round(2).value_counts()
    ax.barh(range(len(fps)), fps.values, color=SEQ[2])
    ax.set_yticks(range(len(fps)), [f"{f:g} fps" for f in fps.index])
    ax.invert_yaxis()
    for i, v in enumerate(fps.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8, color=INK2)
    ax.set_title("Frame rate")
    ax = axes[2]
    data = [df[df.outcome == lab].size_mb.values for lab in fc.LABELS]
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False)
    for patch, lab in zip(bp["boxes"], fc.LABELS):
        patch.set_facecolor(CLASS_COLOR[lab])
        patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_yticks([1, 2, 3], fc.LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("MB / clip")
    ax.set_title("Dung lượng file theo nhãn")
    fig.tight_layout()
    return save(fig, "fig07_video_properties.png")


def curve_panel(ax, t, mat, labels, title, ylabel, anchor=True):
    for k, lab in enumerate(fc.LABELS):
        m = mat[labels == k]
        mean = np.nanmean(m, 0)
        lo, hi = np.nanpercentile(m, 25, 0), np.nanpercentile(m, 75, 0)
        ax.fill_between(t, lo, hi, color=CLASS_COLOR[lab], alpha=0.12, lw=0)
        ax.plot(t, mean, color=CLASS_COLOR[lab], lw=2, label=lab)
    if anchor:
        ax.axvline(fc.ANCHOR_SECONDS, color=INK2, lw=1, ls="--")
        ax.text(fc.ANCHOR_SECONDS + 0.1, ax.get_ylim()[1], " cú sút (5 s)", va="top", fontsize=7, color=INK2)
    ax.set_xlim(0, fc.CLIP_SECONDS)
    ax.set_xlabel("thời gian trong clip (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def audio_fig(plt, audio, labels):
    t = (np.arange(N_AUDIO_BINS) + 0.5) * AUDIO_BIN
    rel = audio - np.nanmedian(audio[:, :16], axis=1, keepdims=True)  # relative to the first 4 s
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    curve_panel(axes[0], t, audio, labels, "Âm lượng tuyệt đối (trung bình, dải = IQR)", "RMS (dB)")
    curve_panel(axes[1], t, rel, labels, "Âm lượng so với 4 s đầu clip", "Δ dB")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "fig08_audio_loudness.png")


def visual_fig(plt, bright, diff, labels):
    t = np.arange(N_VIS) / VIS_FPS
    cuts = cuts_from_diff(diff)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    curve_panel(axes[0], t, diff, labels, "Chuyển động: chênh lệch giữa 2 frame (5 fps)", "mean |Δ| (0-255)")
    ax = axes[1]
    sec = np.arange(int(fc.CLIP_SECONDS))
    for k, lab in enumerate(fc.LABELS):
        c = cuts[labels == k][:, : len(sec) * VIS_FPS].reshape(-1, len(sec), VIS_FPS).any(-1).mean(0)
        ax.plot(sec + 0.5, c, color=CLASS_COLOR[lab], lw=2, marker="o", ms=3, label=lab)
    ax.axvline(fc.ANCHOR_SECONDS, color=INK2, lw=1, ls="--")
    ax.set_xlim(0, fc.CLIP_SECONDS)
    ax.set_xlabel("giây trong clip")
    ax.set_ylabel("tỷ lệ clip có cắt cảnh")
    ax.set_title("Cắt cảnh (chuyển sang replay / cận cảnh) theo từng giây")
    ax.legend()
    fig.tight_layout()
    return save(fig, "fig09_motion_cuts.png")


def contact_sheet(plt, clips, labels, per_class=3, seed=1):
    import cv2
    times = [0.0, 4.0, 5.0, 6.0, 8.0, 11.0, 14.0]
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(fc.NUM_CLASSES):
        for i in rng.choice(np.where(labels == k)[0], per_class, replace=False):
            cap = cv2.VideoCapture(clips[i].path)
            frames = []
            for t in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, f = cap.read()
                f = cv2.resize(f, (192, 108)) if ok else np.zeros((108, 192, 3), np.uint8)
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
            rows.append((k, Path(clips[i].path).stem, frames))
    fig, axes = plt.subplots(len(rows), len(times), figsize=(len(times) * 1.9, len(rows) * 1.15))
    for r, (k, name, frames) in enumerate(rows):
        for c, f in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(f)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(CLASS_COLOR[fc.LABELS[k]])
                sp.set_linewidth(2)
            if r == 0:
                ax.set_title(f"t = {times[c]:g} s", fontsize=8, fontweight="normal")
            if c == 0:
                ax.set_ylabel(f"{fc.LABELS[k]}\n{name}", fontsize=6, rotation=0, ha="right", va="center")
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    return save(fig, "fig10_contact_sheet.png")


def probe_fig(plt, results):
    fig, ax = plt.subplots(figsize=(8, 3.0))
    names = list(results)
    vals = [results[n]["macro_f1"] for n in names]
    ax.barh(range(len(names)), vals, color=SEQ[2])
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(1 / 3, color=INK2, lw=1, ls="--")
    ax.text(1 / 3, len(names) - 0.4, " ngẫu nhiên", fontsize=7, color=INK2)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1 trên valid (train → valid)")
    ax.set_title("Đặc trưng thô (không deep model) đoán được nhãn tới mức nào?")
    return save(fig, "fig11_simple_probe.png")


def asr_fig(plt, asr):
    import pandas as pd
    d = pd.DataFrame(asr)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    ax = axes[0]
    lang = d.lang.value_counts()
    top = lang.index[:8].tolist()
    ax.barh(range(len(top)), lang.values[:8], color=SEQ[2])
    ax.set_yticks(range(len(top)), top)
    ax.invert_yaxis()
    for i, v in enumerate(lang.values[:8]):
        ax.text(v, i, f" {v / len(d):.0%}", va="center", fontsize=8, color=INK2)
    ax.set_title(f"Ngôn ngữ bình luận (Whisper, mẫu {len(d)} clip)")
    ax = axes[1]
    groups = [("từ 'goal/gol/tor/but…'", GOAL_WORDS), ("từ 'save/keeper…'", SAVE_WORDS), ("từ 'wide/over/miss…'", MISS_WORDS)]
    x = np.arange(len(groups))
    w = 0.26
    for k, lab in enumerate(fc.LABELS):
        sub = d[d.label == lab].text.str.lower()
        vals = [sub.str.contains(p, regex=True).mean() for _, p in groups]
        ax.bar(x + (k - 1) * w, vals, w - 0.02, color=CLASS_COLOR[lab], label=lab)
    ax.set_xticks(x, [g for g, _ in groups])
    ax.set_ylabel("tỷ lệ clip có từ khoá")
    ax.set_title("Lời bình có chứa từ khoá theo nhãn")
    ax.legend()
    fig.tight_layout()
    return save(fig, "fig12_asr_sample.png")


# ----------------------------------------------------------------------------- simple probe
def simple_probe(df, audio, diff, labels):
    """Logistic regression on cheap hand-made curves: how much do loudness / cuts alone reveal?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    a = np.nan_to_num(audio - np.nanmedian(audio[:, :16], 1, keepdims=True), nan=0.0)
    a_sec = a.reshape(len(a), 15, 4).mean(-1)
    cuts = cuts_from_diff(np.nan_to_num(diff, nan=0.0)).reshape(len(diff), 15, VIS_FPS).any(-1).astype(float)
    mot = np.nan_to_num(diff, nan=0.0).reshape(len(diff), 15, VIS_FPS).mean(-1)
    feats = {"âm lượng theo giây": a_sec, "cắt cảnh theo giây": cuts, "chuyển động theo giây": mot,
             "âm lượng + cắt cảnh + chuyển động": np.concatenate([a_sec, cuts, mot], 1)}
    tr, va = (df.split == "train").values, (df.split == "valid").values
    out = {}
    for name, X in feats.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], labels[tr])
        pred = clf.predict(X[va])
        out[name] = {"macro_f1": float(f1_score(labels[va], pred, average="macro")),
                     "f1": [float(v) for v in f1_score(labels[va], pred, average=None)]}
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--asr-sample", type=int, default=0, help="stratified Whisper sample size (GPU)")
    args = p.parse_args()
    import pandas as pd

    raw = pd.read_csv(fc.MANIFEST, dtype=str, keep_default_na=False)
    clips = fc.load_clips(require_file=False)
    missing = [c for c in clips if not c.path]
    clips = [c for c in clips if c.path]
    df = pd.DataFrame([c.info for c in clips]).reset_index(drop=True)
    labels = np.array([c.label for c in clips])
    df["league"] = df.game.str.split("/").str[0]
    df["season"] = df.game.str.split("/").str[1]
    df["minute"] = df.game_time.str.split(" - ").str[1].str.split(":").map(
        lambda t: int(t[0]) + int(t[1]) / 60 if len(t) == 2 else np.nan)
    df["position_ms"] = df.position_ms.astype(np.int64)

    probes, audio, bright, diff = load_or_measure(clips, args.workers)
    for key in probes[0]:
        df[key] = [pr[key] for pr in probes]
    df.to_csv(Path(fc.ANALYSIS_DIR) / "per_clip.csv", index=False)

    asr_path = Path(fc.ANALYSIS_DIR) / "asr_sample.json"
    asr = json.load(open(asr_path)) if asr_path.is_file() else None
    full_asr = Path(fc.FEATURES) / "transcripts.json"  # every clip, from transcribe.py (correct languages)
    if full_asr.is_file():
        tr = json.load(open(full_asr))
        asr = [{"index": i, "label": fc.LABELS[c.label], "lang": tr[c.info["path"]]["lang"] or "?",
                "text": tr[c.info["path"]]["text"]} for i, c in enumerate(clips) if c.info["path"] in tr]
        print(f"[ASR] using full transcripts ({len(asr)} clips)")
    if args.asr_sample and asr is None:
        asr = asr_sample(clips, args.asr_sample)
        json.dump(asr, open(asr_path, "w"), ensure_ascii=False, indent=1)

    plt = setup_mpl()
    figs = [bars_by_split(plt, df), games_fig(plt, df), league_fig(plt, df), label_meta_fig(plt, df),
            minute_fig(plt, df), overlap_fig(plt, df), video_props_fig(plt, df), audio_fig(plt, audio, labels),
            visual_fig(plt, bright, diff, labels), contact_sheet(plt, clips, labels)]
    probe = simple_probe(df, audio, diff, labels)
    figs.append(probe_fig(plt, probe))
    if asr:
        figs.append(asr_fig(plt, asr))

    # ---- numbers for the report
    cuts = cuts_from_diff(diff)
    rel = audio - np.nanmedian(audio[:, :16], 1, keepdims=True)
    stats = {
        "manifest_rows": len(raw), "failed_rows": int(raw.path.str.startswith("failed:").sum()),
        "clips": len(clips), "missing_files": len(missing),
        "by_split_label": {s: {lab: int(((df.split == s) & (df.outcome == lab)).sum()) for lab in fc.LABELS}
                           for s in fc.SPLITS},
        "games": {s: int(df[df.split == s].game_idx.nunique()) for s in fc.SPLITS},
        "games_shared_between_splits": int(len(set(df[df.split == "train"].game_idx) &
                                               (set(df[df.split == "valid"].game_idx) | set(df[df.split == "test"].game_idx)))),
        "leagues": df.league.value_counts().to_dict(), "seasons": df.season.value_counts().to_dict(),
        "resolution": (df.width.astype(str) + "x" + df.height.astype(str)).value_counts().to_dict(),
        "fps": df.fps.round(2).value_counts().to_dict(),
        "duration_not_15s": int((np.abs(df.duration - 15) > 0.2).sum()),
        "no_audio": int((~df.has_audio.astype(bool)).sum()),
        "size_mb_mean": float(df.size_mb.mean()),
        "overlap_lt15s_by_label": {lab: float((df[df.outcome == lab].gap_s < 15).mean()) for lab in fc.LABELS},
        "goal_pairs_lt60s": int(((df.outcome == "goal") & (df.gap_s < 60)).sum()),
        "audio_peak_rel_db_5to10s": {lab: float(np.nanmean(np.nanmax(rel[labels == k][:, 20:40], 1)))
                                     for k, lab in enumerate(fc.LABELS)},
        "cut_rate_by_window": {lab: {w: float(cuts[labels == k][:, a:b].any(1).mean())
                                     for w, (a, b) in {"0-5s": (0, 25), "5-10s": (25, 50), "10-15s": (50, 75)}.items()}
                               for k, lab in enumerate(fc.LABELS)},
        "chain_len": {lab: df[df.outcome == lab].chain_len.value_counts().to_dict() for lab in ("saved", "off_target")},
        "simple_probe_valid": probe,
    }
    if asr:
        d = pd.DataFrame(asr)
        stats["asr_sample"] = {
            "n": len(d), "languages": d.lang.value_counts().to_dict(),
            "words_median": float(d.text.str.split().map(len).median()),
            "goal_word_rate": {lab: float(d[d.label == lab].text.str.lower().str.contains(GOAL_WORDS).mean())
                               for lab in fc.LABELS}}
    stats = json.loads(json.dumps(stats, default=lambda o: o if not hasattr(o, "item") else o.item()))
    json.dump({str(k): v for k, v in stats.items()}, open(REPORT_DIR / "stats.json", "w"), indent=1,
              ensure_ascii=False, default=str)
    print(json.dumps(stats, indent=1, ensure_ascii=False, default=str)[:6000])
    print("[FIGURES]", figs)


if __name__ == "__main__":
    main()
