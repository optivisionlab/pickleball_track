"""Visualize the pickleball dataset, its cached features and the current CV results.

CPU only (it never touches the GPU), so it can run while the GPU is busy:

    CUDA_VISIBLE_DEVICES="" python analysis/visualize_dataset.py

Figures (PNG) and a short numeric summary (summary.json, report.md) are written to
baseline/pickleball/outputs/analysis by default:

  01_overview.png        events per user, minutes per user, durations, label purity per video
  02_timing.png          gaps between strokes, token class balance vs event radius
  03_audio_onsets.png    raw-audio transient around each annotated event (label alignment)
  04_feature_change.png  VideoMAE / BEATs feature change around events (hit vs miss)
  05_embedding.png       t-SNE of event tokens by class and by user (user domain shift)
  06_linear_probe.png    user-grouped linear probes: video vs audio vs longer context
  07_results_per_user.png  test F1 per user and version (from cv_summary.json)
  08_val_vs_test.png     validation vs test F1 per fold (how noisy checkpoint selection is)
  09_timelines.png       predicted probabilities vs ground truth on the worst test videos
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import pickleball_common as pc  # noqa: E402  (puts the baseline root on sys.path)
from common import media as bc  # noqa: E402

PICKLEBALL = Path(__file__).resolve().parents[1]  # baseline/pickleball
COLORS = {"no_event": "#9aa0a6", "hit": "#1a73e8", "miss": "#d93025"}
HIT, MISS = pc.LABEL2ID["hit"], pc.LABEL2ID["miss"]


def savefig(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)
    print(f"[FIG] {out / name}")


# ------------------------------------------------------------------ metadata
def fig_overview(videos, out, summary):
    users = sorted({v.user for v in videos})
    hit = np.array([sum(e[1] == HIT for v in videos if v.user == u for e in v.events) for u in users])
    miss = np.array([sum(e[1] == MISS for v in videos if v.user == u for e in v.events) for u in users])
    minutes = np.array([sum(v.duration for v in videos if v.user == u) / 60 for u in users])

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(users))
    ax[0, 0].bar(x, hit, color=COLORS["hit"], label="hit")
    ax[0, 0].bar(x, miss, bottom=hit, color=COLORS["miss"], label="miss")
    ax[0, 0].set_xticks(x, users, rotation=60)
    ax[0, 0].set_title(f"Events per user (total hit={hit.sum()}, miss={miss.sum()})")
    ax[0, 0].legend()

    ax[0, 1].bar(x, minutes, color="#5f6368")
    ax[0, 1].set_xticks(x, users, rotation=60)
    ax[0, 1].set_title("Minutes of video per user")

    fps = np.array([round(v.fps) for v in videos])
    for f in sorted(set(fps)):
        ax[1, 0].hist([v.duration for v, g in zip(videos, fps) if g == f], bins=20, alpha=0.6,
                      label=f"{f} fps (n={int((fps == f).sum())})")
    ax[1, 0].set_xlabel("duration (s)")
    ax[1, 0].set_title("Video duration by source fps")
    ax[1, 0].legend()

    nh = np.array([sum(e[1] == HIT for e in v.events) for v in videos])
    nm = np.array([sum(e[1] == MISS for e in v.events) for v in videos])
    trung = np.array(["_Trung_" in Path(v.path).name for v in videos])
    ax[1, 1].scatter(nh[trung], nm[trung], c=COLORS["hit"], label="file name 'Trung'")
    ax[1, 1].scatter(nh[~trung], nm[~trung], c=COLORS["miss"], label="file name 'Truot'")
    ax[1, 1].set_xlabel("# hit in video")
    ax[1, 1].set_ylabel("# miss in video")
    ax[1, 1].set_title("Label purity per video")
    ax[1, 1].legend()
    savefig(fig, out, "01_overview.png")

    purity = np.maximum(nh, nm) / np.maximum(nh + nm, 1)
    summary["overview"] = {
        "videos": len(videos), "users": len(users),
        "hours": float(sum(v.duration for v in videos) / 3600),
        "events_hit": int(hit.sum()), "events_miss": int(miss.sum()),
        "per_user_hit": dict(zip(users, hit.tolist())),
        "per_user_miss": dict(zip(users, miss.tolist())),
        "users_without_miss": [u for u, m in zip(users, miss) if m == 0],
        "video_label_purity_mean": float(purity.mean()),
        "videos_mixed_labels": int(((nh > 0) & (nm > 0)).sum()),
        "fps_counts": {int(f): int((fps == f).sum()) for f in sorted(set(fps))},
    }


def fig_timing(videos, out, summary):
    gaps = {HIT: [], MISS: []}
    for v in videos:
        for (t0, _), (t1, c1) in zip(v.events, v.events[1:]):
            gaps[c1].append(t1 - t0)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    bins = np.linspace(0, 8, 65)
    for c, name in ((HIT, "hit"), (MISS, "miss")):
        ax[0].hist(gaps[c], bins=bins, alpha=0.6, color=COLORS[name], label=f"before a {name}")
    ax[0].set_xlabel("time since previous event (s)")
    ax[0].set_title("Gap between consecutive strokes")
    ax[0].axvline(pc.NUM_VIDEO_FRAMES / pc.TARGET_FPS, color="k", ls="--", lw=1)
    ax[0].text(pc.NUM_VIDEO_FRAMES / pc.TARGET_FPS + 0.05, ax[0].get_ylim()[1] * 0.9, "window 1.07 s")
    ax[0].legend()

    radii = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    frac = []
    for r in radii:
        counts = np.zeros(pc.NUM_CLASSES)
        for v in videos:
            size = math.ceil(v.num_sampled / pc.FRAMES_PER_TOKEN)
            lab = pc.token_labels(v.events, 0, r, size)
            counts += np.bincount(lab, minlength=pc.NUM_CLASSES)
        frac.append(counts / counts.sum())
    frac = np.array(frac)
    for c, name in enumerate(pc.LABEL2ID):
        ax[1].plot(radii, frac[:, c], "o-", color=COLORS[name], label=name)
    ax[1].axvline(0.5, color="k", ls="--", lw=1)
    ax[1].set_xlabel("--event-radius (s)")
    ax[1].set_ylabel("fraction of tokens")
    ax[1].set_title("Token class balance vs label radius")
    ax[1].legend()

    all_gaps = np.concatenate([gaps[HIT], gaps[MISS]])
    merged = [float((all_gaps < 2 * r + pc.TOKEN_SEC).mean()) for r in radii]
    ax[2].plot(radii, merged, "o-", color="#5f6368")
    ax[2].set_xlabel("--event-radius (s)")
    ax[2].set_ylabel("fraction of consecutive events")
    ax[2].set_title("Consecutive events whose labelled spans touch\n(decoder may merge them)")
    savefig(fig, out, "02_timing.png")

    summary["timing"] = {
        "gap_median_s": float(np.median(all_gaps)), "gap_p5_s": float(np.percentile(all_gaps, 5)),
        "gap_min_s": float(all_gaps.min()),
        "token_fraction_by_radius": {str(r): f.round(4).tolist() for r, f in zip(radii, frac)},
        "touching_spans_by_radius": dict(zip(map(str, radii), merged)),
    }


# ------------------------------------------------------------------ audio
AUDIO_HOP = 0.005  # s
PERI = 1.0  # s on each side of an event


def _audio_envelope(path):
    wave = bc.read_audio(path, pc.AUDIO_SR)
    if wave.size < pc.AUDIO_SR:
        return None
    hp = np.diff(wave)  # crude high-pass: paddle "pops" are sharp broadband transients
    hop = int(AUDIO_HOP * pc.AUDIO_SR)
    n = hp.size // hop
    env = np.sqrt((hp[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    env = np.log(env)
    return (env - env.mean()) / (env.std() + 1e-6)


def fig_audio(videos, out, summary, workers):
    # spawn, not fork: a forked pool can deadlock OpenMP (t-SNE/PCA) later in this process
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        envs = list(pool.map(_audio_envelope, [v.path for v in videos]))
    k = int(PERI / AUDIO_HOP)
    lags_axis = np.arange(-k, k + 1) * AUDIO_HOP
    curves = {HIT: [], MISS: []}
    lag_video, lag_user = [], {}
    for v, env in zip(videos, envs):
        if env is None:
            continue
        rows = []
        for t, c in v.events:
            i = int(round(t / AUDIO_HOP))
            if i - k < 0 or i + k + 1 > env.size:
                continue
            rows.append(env[i - k:i + k + 1])
            curves[c].append(rows[-1])
        if rows:
            mean = np.mean(rows, 0)
            near = np.abs(lags_axis) <= 0.5
            lag = float(lags_axis[near][mean[near].argmax()])
            lag_video.append(lag)
            lag_user.setdefault(v.user, []).append(lag)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for c, name in ((HIT, "hit"), (MISS, "miss")):
        arr = np.array(curves[c])
        m, s = arr.mean(0), arr.std(0) / math.sqrt(len(arr))
        ax[0].plot(lags_axis, m, color=COLORS[name], label=f"{name} (n={len(arr)})")
        ax[0].fill_between(lags_axis, m - s, m + s, color=COLORS[name], alpha=0.2)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("time relative to annotated event (s)")
    ax[0].set_ylabel("z-scored high-freq audio energy")
    ax[0].set_title("Audio transient around events")
    ax[0].legend()

    ax[1].hist(lag_video, bins=np.arange(-0.5, 0.51, 0.02), color="#5f6368")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("audio peak - annotation (s), per video")
    ax[1].set_title("Annotation offset estimated from audio")

    users = sorted(lag_user)
    ax[2].boxplot([lag_user[u] for u in users], tick_labels=users)
    ax[2].axhline(0, color="k", lw=1)
    ax[2].tick_params(axis="x", rotation=60)
    ax[2].set_ylabel("audio peak - annotation (s)")
    ax[2].set_title("Offset per user")
    savefig(fig, out, "03_audio_onsets.png")

    peak = {name: float(lags_axis[np.array(curves[c]).mean(0).argmax()])
            for c, name in ((HIT, "hit"), (MISS, "miss"))}
    summary["audio"] = {
        "videos_without_audio": [Path(v.path).name for v, e in zip(videos, envs) if e is None],
        "mean_curve_peak_s": peak,
        "per_video_offset_median_s": float(np.median(lag_video)),
        "per_video_offset_abs_gt_0.1s": int((np.abs(lag_video) > 0.1).sum()),
        "per_user_offset_median_s": {u: float(np.median(lag_user[u])) for u in users},
    }


# ------------------------------------------------------------------ cached features
def load_timelines(cache: str):
    """Per-video token timelines (global-mean video feature, audio) stitched from the cache."""
    data = torch.load(cache, map_location="cpu", weights_only=False)
    videos = data["videos"]
    sizes = [math.ceil(v["num_sampled"] / pc.FRAMES_PER_TOKEN) for v in videos]
    vt = [np.zeros((s, data["video"].shape[-1]), np.float32) for s in sizes]
    at = [np.zeros((s, data["audio"].shape[-1]), np.float32) for s in sizes]
    ct = [np.zeros(s, np.float32) for s in sizes]
    vmean = data["video"][:, :, 0].float().numpy()
    grid = data["video"].shape[2]
    vgrid_store = [np.zeros((s, grid * data["video"].shape[-1]), np.float16) for s in sizes]
    audio = data["audio"].float().numpy()
    valid = data["token_valid"].numpy()
    for r, (vid, start) in enumerate(zip(data["video_index"].tolist(), data["starts"].tolist())):
        slot = start // pc.FRAMES_PER_TOKEN + np.arange(pc.NUM_TEMPORAL_TOKENS)
        ok = valid[r] & (slot < sizes[vid])
        vt[vid][slot[ok]] += vmean[r][ok]
        at[vid][slot[ok]] += audio[r][ok]
        ct[vid][slot[ok]] += 1
        vgrid_store[vid][slot[ok]] = data["video"][r][ok].flatten(1).numpy()
    for i in range(len(videos)):
        c = np.maximum(ct[i], 1)[:, None]
        vt[i] /= c
        at[i] /= c
    has_audio = np.zeros(len(videos), bool)
    has_audio[data["video_index"].numpy()] = data["has_audio"].numpy()
    return videos, vt, at, ct, vgrid_store, has_audio


def fig_feature_change(tl, out, summary):
    videos, vt, at, ct, _, has_audio = tl
    k = 15  # tokens = 1 s
    axis = np.arange(-k, k + 1) * pc.TOKEN_SEC
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, (name, feats) in enumerate((("VideoMAE", vt), ("BEATs", at))):
        for c, cname in ((HIT, "hit"), (MISS, "miss")):
            rows = []
            for i, v in enumerate(videos):
                if name == "BEATs" and not has_audio[i]:
                    continue
                d = np.linalg.norm(np.diff(feats[i], axis=0), axis=1)
                d = (d - d.mean()) / (d.std() + 1e-6)
                for t, cc in v["events"]:
                    if cc != c:
                        continue
                    s = int(t / pc.TOKEN_SEC)
                    if s - k >= 0 and s + k + 1 <= d.size:
                        rows.append(d[s - k:s + k + 1])
            arr = np.array(rows)
            ax[j].plot(axis, arr.mean(0), color=COLORS[cname], label=f"{cname} (n={len(arr)})")
            res[f"{name}_{cname}_peak_s"] = float(axis[arr.mean(0).argmax()])
        ax[j].axvline(0, color="k", lw=1)
        ax[j].set_xlabel("time relative to annotated event (s)")
        ax[j].set_title(f"{name}: change between consecutive tokens (z)")
        ax[j].legend()
    savefig(fig, out, "04_feature_change.png")
    summary["feature_change"] = res


def event_token_table(tl, offsets=(0,), max_no_event=20000, seed=0):
    """Tokens at (event + offset) for hit/miss, plus random no_event tokens far from events."""
    videos, vt, at, ct, vg, has_audio = tl
    rng = np.random.default_rng(seed)
    rows = []  # (video, token, class)
    for i, v in enumerate(videos):
        size = vt[i].shape[0]
        ev_tokens = [int(t / pc.TOKEN_SEC) for t, _ in v["events"]]
        for (t, c), s in zip(v["events"], ev_tokens):
            if 0 <= s < size and ct[i][s] > 0:
                rows.append((i, s, c))
        lab = pc.token_labels(v["events"], 0, 0.5, size)
        far = np.flatnonzero((lab == 0) & (ct[i] > 0))
        if far.size:
            rows += [(i, int(s), 0) for s in rng.choice(far, min(far.size, 60), replace=False)]
    rows = np.array(rows)
    no_ev = np.flatnonzero(rows[:, 2] == 0)
    if no_ev.size > max_no_event:
        drop = rng.choice(no_ev, no_ev.size - max_no_event, replace=False)
        rows = np.delete(rows, drop, 0)
    return rows


def context_features(feats, i, s, before, after):
    f = feats[i]
    parts = [f[s]]
    if before:
        parts.append(f[max(0, s - before):s].mean(0) if s > 0 else np.zeros_like(f[s]))
    if after:
        parts.append(f[s + 1:s + 1 + after].mean(0) if s + 1 < f.shape[0] else np.zeros_like(f[s]))
    return np.concatenate(parts)


def fig_embedding(tl, out, summary, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    videos, vt, at, *_ = tl
    rows = event_token_table(tl, max_no_event=3000, seed=seed)
    rng = np.random.default_rng(seed)
    if len(rows) > 9000:
        rows = rows[rng.choice(len(rows), 9000, replace=False)]
    x = np.stack([vt[i][s] for i, s, _ in rows])
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    emb = TSNE(2, init="pca", perplexity=40, random_state=seed).fit_transform(PCA(50).fit_transform(x))
    users = np.array([videos[i]["user"] for i, _, _ in rows])
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    for c, name in enumerate(pc.LABEL2ID):
        m = rows[:, 2] == c
        ax[0].scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, color=COLORS[name], label=name)
    ax[0].set_title("t-SNE of VideoMAE tokens at the event (by class)")
    ax[0].legend(markerscale=5)
    names = sorted(set(users))
    cmap = plt.get_cmap("tab20")
    for j, u in enumerate(names):
        m = users == u
        ax[1].scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, color=cmap(j % 20), label=u)
    ax[1].set_title("Same tokens coloured by user (clusters = user/session domain shift)")
    ax[1].legend(markerscale=5, fontsize=7, ncol=2)
    savefig(fig, out, "05_embedding.png")


def fig_linear_probe(tl, out, summary, seed=0):
    """User-grouped 5-fold logistic regression on single tokens (+ optional context)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    videos, vt, at, ct, vg, has_audio = tl
    rows = event_token_table(tl, max_no_event=12000, seed=seed)
    groups = np.array([videos[i]["user"] for i, _, _ in rows])
    y = rows[:, 2]
    W = 15  # 1 s of context

    def build(kind):
        out_rows = []
        for i, s, _ in rows:
            if kind == "video (mean)":
                out_rows.append(vt[i][s])
            elif kind == "video (grid 2x2)":
                out_rows.append(vg[i][s].astype(np.float32))
            elif kind == "audio":
                out_rows.append(at[i][s])
            elif kind == "video+audio":
                out_rows.append(np.concatenate([vt[i][s], at[i][s]]))
            elif kind == "video+audio, +1 s before":
                out_rows.append(np.concatenate([context_features(vt, i, s, W, 0),
                                                context_features(at, i, s, W, 0)]))
            elif kind == "video+audio, +1 s after":
                out_rows.append(np.concatenate([context_features(vt, i, s, 0, W),
                                                context_features(at, i, s, 0, W)]))
            elif kind == "video+audio, ±1 s":
                out_rows.append(np.concatenate([context_features(vt, i, s, W, W),
                                                context_features(at, i, s, W, W)]))
        return np.stack(out_rows).astype(np.float32)

    kinds = ["video (mean)", "video (grid 2x2)", "audio", "video+audio",
             "video+audio, +1 s before", "video+audio, +1 s after", "video+audio, ±1 s"]
    results = {}
    for kind in kinds:
        x = build(kind)
        pred3 = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(x, y, groups):
            clf = make_pipeline(StandardScaler(), PCA(min(128, x.shape[1]), random_state=seed),
                                LogisticRegression(max_iter=400, class_weight="balanced", C=0.5))
            clf.fit(x[tr], y[tr])
            pred3[te] = clf.predict(x[te])
        ev = y > 0
        results[kind] = {
            "macro_f1_3class": float(f1_score(y, pred3, average="macro")),
            "f1_by_class": f1_score(y, pred3, average=None).round(4).tolist(),
            # hit vs miss among true event tokens (is the outcome visible at that moment?)
            "hit_vs_miss_acc": float((pred3[ev] == y[ev]).mean()),
        }
        print(f"[PROBE] {kind:28s} {results[kind]}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    yy = np.arange(len(kinds))
    ax[0].barh(yy, [results[k]["macro_f1_3class"] for k in kinds], color="#5f6368")
    ax[0].set_yticks(yy, kinds)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 1)
    ax[0].set_title("Linear probe, macro-F1 (no_event / hit / miss), user-grouped CV")
    width = 0.38
    ax[1].barh(yy - width / 2, [results[k]["f1_by_class"][1] for k in kinds], width,
               color=COLORS["hit"], label="hit F1")
    ax[1].barh(yy + width / 2, [results[k]["f1_by_class"][2] for k in kinds], width,
               color=COLORS["miss"], label="miss F1")
    ax[1].set_yticks(yy, kinds)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 1)
    ax[1].legend()
    ax[1].set_title("Per-class F1 of the same probes")
    savefig(fig, out, "06_linear_probe.png")
    summary["linear_probe"] = {"tokens": int(len(y)), "class_counts": np.bincount(y).tolist(),
                               "results": results}


# ------------------------------------------------------------------ results
def load_reports():
    reports = {}
    for v in ("v3", "v4", "v5"):
        p = PICKLEBALL / "outputs" / "runs" / v / "cv_summary.json"
        if p.is_file():
            reports[v] = json.loads(p.read_text())
    return reports


def fig_results(reports, out, summary):
    if not reports:
        print("[SKIP] no cv_summary.json found")
        return
    users = sorted(next(iter(reports.values()))["per_user_token"])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for j, cname in enumerate(pc.LABEL2ID):
        mat = np.array([[reports[v]["per_user_token"][u]["f1_by_class"][cname] for v in reports]
                        for u in users])
        im = ax[j].imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax[j].set_xticks(range(len(reports)), list(reports))
        ax[j].set_yticks(range(len(users)), users)
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                ax[j].text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=8)
        ax[j].set_title(f"test token F1: {cname}")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(out / "07_results_per_user.png", dpi=110)
    plt.close(fig)
    print(f"[FIG] {out / '07_results_per_user.png'}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for v, rep in reports.items():
        val = [f["val_macro_f1"] for f in rep["folds"]]
        test = [f["test"]["macro_f1"] for f in rep["folds"]]
        ax.scatter(val, test, label=v, s=40)
        for f, a, b in zip(rep["folds"], val, test):
            ax.annotate(str(f["fold"]), (a, b), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.plot([0.4, 1], [0.4, 1], "k--", lw=1)
    ax.set_xlabel("validation token macro-F1 (best epoch)")
    ax.set_ylabel("test token macro-F1")
    ax.set_title("Validation vs test per fold\n(far from the diagonal = noisy model selection)")
    ax.legend()
    savefig(fig, out, "08_val_vs_test.png")

    summary["results"] = {
        v: {"pooled_token_macro_f1": rep["pooled_token"]["macro_f1"],
            "event_macro_f1": rep["event"]["macro_f1"],
            "val_users_per_fold": [f["users"]["val"] for f in rep["folds"]],
            "worst_users_miss_f1": sorted(
                ((u, round(m["f1_by_class"]["miss"], 3)) for u, m in rep["per_user_token"].items()),
                key=lambda x: x[1])[:4]}
        for v, rep in reports.items()
    }


def fig_timelines(reports, out, summary, cache, version="v4", worst=4):
    if version not in reports:
        return
    p = PICKLEBALL / "outputs" / "runs" / version / "test_predictions.pt"
    if not p.is_file():
        return
    pred = torch.load(p, weights_only=False)
    data = torch.load(cache, map_location="cpu", weights_only=False)
    data = {k: data[k] for k in ("videos", "video_index", "starts", "token_valid")}
    rows = np.arange(len(data["video_index"]))
    tl = pc.stitch_timelines(pred["probs"].numpy(), data, rows)
    scores = []
    for vid, timeline in tl.items():
        truth = {vid: [tuple(e) for e in data["videos"][vid]["events"]]}
        dec = {vid: pc.decode_events(timeline, 3)}
        m = pc.event_metrics(dec, truth, 0.5)
        scores.append((m["detection"]["f1"] * 0.5 + (m["hit"]["f1"] + m["miss"]["f1"]) / 4, vid))
    scores.sort()
    chosen = [vid for _, vid in scores[:worst]]
    fig, ax = plt.subplots(len(chosen), 1, figsize=(18, 2.6 * len(chosen)))
    for a, vid in zip(np.atleast_1d(ax), chosen):
        timeline = tl[vid]
        t = (np.arange(len(timeline)) * pc.FRAMES_PER_TOKEN + 0.5) / pc.TARGET_FPS
        a.plot(t, timeline[:, HIT], color=COLORS["hit"], lw=1, label="P(hit)")
        a.plot(t, timeline[:, MISS], color=COLORS["miss"], lw=1, label="P(miss)")
        for et, c in data["videos"][vid]["events"]:
            a.axvline(et, color=COLORS["hit" if c == HIT else "miss"], ls=":", lw=1.2)
        a.set_ylim(0, 1)
        name = Path(data["videos"][vid]["path"]).name
        a.set_title(f"{version} test predictions — {name} (dotted = ground-truth events)", fontsize=9)
        a.legend(loc="upper right", fontsize=7)
    savefig(fig, out, "09_timelines.png")
    summary["worst_videos"] = [Path(data["videos"][v]["path"]).name for v in chosen]


def write_report(out: Path, summary: dict) -> None:
    lines = ["# Pickleball dataset analysis", ""]
    for name in sorted(p.name for p in out.glob("*.png")):
        lines += [f"## {name}", f"![{name}]({name})", ""]
    lines += ["## Numbers", "```json", json.dumps(summary, indent=2), "```"]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=pc.DEFAULT_DATA_DIR)
    parser.add_argument("--feature-cache", default="",
                        help="features_*.pt (default: the one in baseline/pickleball/cache)")
    parser.add_argument("--output-dir", default=str(PICKLEBALL / "outputs/analysis"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = args.feature_cache or sorted(
        glob.glob(str(PICKLEBALL / "cache/features_*.pt")))[-1]

    summary = {}
    videos = pc.discover_videos(args.data_dir)
    fig_overview(videos, out, summary)
    fig_timing(videos, out, summary)
    if not args.skip_audio:
        fig_audio(videos, out, summary, args.workers)
    print(f"[LOAD] {cache}")
    tl = load_timelines(cache)
    fig_feature_change(tl, out, summary)
    fig_embedding(tl, out, summary)
    if not args.skip_probe:
        fig_linear_probe(tl, out, summary)
    del tl
    reports = load_reports()
    fig_results(reports, out, summary)
    fig_timelines(reports, out, summary, cache)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out, summary)
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
