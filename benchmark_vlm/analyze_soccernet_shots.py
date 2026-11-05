#!/usr/bin/env python
"""Phân tích bộ SoccerNet Shots: metadata, thông số kỹ thuật của clip, nội dung hình và tiếng.

Chạy:
    python analyze_soccernet_shots.py                      # chạy cả 3 phần
    python analyze_soccernet_shots.py --stages meta        # chỉ metadata
    python analyze_soccernet_shots.py --per-outcome 300    # số clip mẫu mỗi lớp cho phần nội dung

Kết quả (biểu đồ PNG, CSV trung gian, stats.json) nằm trong --out.
Phần probe và nội dung có cache CSV/NPZ; xóa thư mục cache/ để chạy lại từ đầu.
"""
import argparse
import json
import os
import re
import subprocess
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import matplotlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = "/data/Cuong/pickleball/data/bendmark/football/soccernet_shots"
DEFAULT_OUT = os.path.join(HERE, "analysis_soccernet_shots")

# ---------------------------------------------------------------- style
OUTCOMES = ["goal", "saved", "off_target"]
OUTCOME_VI = {"goal": "Bàn thắng", "saved": "Bị cản phá", "off_target": "Ra ngoài"}
# 3 slot đầu của bảng màu tham chiếu: hợp lệ cho mọi cặp, kể cả người mù màu.
COLOR = {"goal": "#2a78d6", "saved": "#eb6834", "off_target": "#1baf7a"}
SEQ = "#2a78d6"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
LEAGUE_VI = {
    "spain_laliga": "La Liga",
    "italy_serie-a": "Serie A",
    "europe_uefa-champions-league": "Champions League",
    "england_epl": "Premier League",
    "germany_bundesliga": "Bundesliga",
    "france_ligue-1": "Ligue 1",
}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print("  ->", path)


def outcome_legend(ax, loc="best", **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[o]) for o in OUTCOMES]
    ax.legend(handles, [OUTCOME_VI[o] for o in OUTCOMES], loc=loc, **kw)


def auc(pos, neg):
    """AUC kiểu Mann-Whitney: xác suất một mẫu pos có điểm cao hơn một mẫu neg."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stacked_share(ax, groups, counts, title):
    """Thanh ngang 100% chia theo outcome. counts: DataFrame index=groups, cột=OUTCOMES."""
    share = counts.div(counts.sum(axis=1), axis=0) * 100
    y = np.arange(len(groups))[::-1]
    left = np.zeros(len(groups))
    for o in OUTCOMES:
        w = share.loc[groups, o].to_numpy()
        ax.barh(y, w, left=left, color=COLOR[o], edgecolor=SURF, linewidth=2, height=0.62)
        for yi, li, wi in zip(y, left, w):
            if wi >= 7:
                ax.text(li + wi / 2, yi, f"{wi:.0f}%", ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  (n={int(counts.loc[g].sum()):,})" for g in groups])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% số clip")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)


# ---------------------------------------------------------------- metadata
KEYWORDS = {
    "Goal! / tỉ số": r"goal!|\b\d+:\d+\b",
    "save / keeper": r"\bsav(e|es|ed|ing)\b|goalkeeper|\bkeeper\b|parr(y|ies|ied)",
    "wide / over / miss": r"\bwide\b|over the (cross)?bar|off target|\bmiss(es|ed)?\b|\bhigh\b",
    "blocked": r"\bblock",
    "post / bar": r"\bpost\b|crossbar|woodwork",
    "corner (phạt góc)": r"corner kick|\b(a|the) corner\b|\bcorners\b",
    "substitution": r"substitut|replac",
    "foul / card": r"\bfoul|\bcard\b|booked|caution",
    "offside": r"offside",
}


def keyword_rule(text):
    t = text.lower()
    if not t:
        return "no_caption"
    if re.search(KEYWORDS["Goal! / tỉ số"], t):
        return "goal"
    if re.search(KEYWORDS["save / keeper"], t):
        return "saved"
    if re.search(KEYWORDS["wide / over / miss"], t) or re.search(KEYWORDS["post / bar"], t):
        return "off_target"
    return "unknown"


def load_manifest(root):
    df = pd.read_csv(os.path.join(root, "manifest_clean.csv"), keep_default_na=False)
    parts = df.game.str.split("/")
    df["league"] = parts.str[0].map(LEAGUE_VI).fillna(parts.str[0])
    df["season"] = parts.str[1]
    teams = parts.str[2].str.extract(r"^\S+ - \S+ (.*?) (\d+) - (\d+) (.*)$")
    df["score_goals"] = pd.to_numeric(teams[1], errors="coerce") + pd.to_numeric(teams[2], errors="coerce")
    # Phút bù giờ (45+, 90+) được gộp vào 5 phút cuối của hiệp đó.
    m = df.position_ms / 60000
    df["minute"] = np.where(df.half == 2, 45 + m.clip(upper=49.99), m.clip(upper=44.99))
    df["has_caption"] = df.caption != ""
    df["caption_offset_s"] = pd.to_numeric(df.caption_offset_s, errors="coerce")
    df["caption_words"] = df.caption.str.split().str.len().where(df.has_caption)
    df["rule_pred"] = df.caption.map(keyword_rule)
    return df


def stage_meta(df, out, stats):
    print("[meta] biểu đồ metadata")
    n = len(df)
    stats["n_clips"] = n
    stats["n_games"] = int(df.game_idx.nunique())
    stats["outcome_counts"] = df.outcome.value_counts().to_dict()
    stats["split_counts"] = df.split.value_counts().to_dict()

    # 01. Tỉ lệ kết quả theo split / đội / nguồn video
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.6), gridspec_kw={"height_ratios": [4, 2, 2]})
    c = pd.crosstab(df.split, df.outcome)[OUTCOMES]
    c.loc["Tất cả"] = c.sum()
    stacked_share(axes[0], ["train", "valid", "test", "Tất cả"], c, "Tỉ lệ kết quả theo split")
    c = pd.crosstab(df.team.map({"home": "Đội nhà", "away": "Đội khách"}), df.outcome)[OUTCOMES]
    stacked_share(axes[1], ["Đội nhà", "Đội khách"], c, "Theo đội sút")
    c = pd.crosstab(df.video_res, df.outcome)[OUTCOMES]
    stacked_share(axes[2], list(c.index), c, "Theo nguồn video")
    outcome_legend(axes[0], loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3)
    fig.tight_layout()
    save(fig, out, "01_outcome_share.png")

    # 02. Giải đấu
    lg = df.league.value_counts()
    order = list(lg.index)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    y = np.arange(len(order))[::-1]
    a1.barh(y, lg[order].to_numpy(), color=SEQ, height=0.62)
    for yi, v in zip(y, lg[order]):
        a1.text(v + 30, yi, f"{v:,}", va="center", color=INK2, fontsize=9)
    a1.set_yticks(y)
    a1.set_yticklabels(order)
    a1.set_xlim(0, lg.max() * 1.18)
    a1.set_xlabel("Số clip")
    a1.grid(axis="y", visible=False)
    a1.set_title("Số clip theo giải đấu")
    c = pd.crosstab(df.league, df.outcome)[OUTCOMES]
    stacked_share(a2, order, c, "Tỉ lệ kết quả theo giải")
    outcome_legend(a2, loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3)
    fig.tight_layout()
    save(fig, out, "02_league.png")
    stats["league_counts"] = lg.to_dict()
    stats["goal_rate_by_league"] = (df.groupby("league").outcome.apply(lambda s: (s == "goal").mean())
                                    .round(4).to_dict())

    # 03. Heatmap giải x mùa
    hm = pd.crosstab(df.league, df.season).loc[order]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.imshow(hm.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#eef4fc", "#9ec5f4", "#2a78d6", "#104281"]), aspect="auto")
    vmax = hm.to_numpy().max()
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.iat[i, j]
            ax.text(j, i, f"{v:,}" if v else "–", ha="center", va="center", fontsize=9,
                    color="white" if v > vmax * 0.55 else INK)
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels(hm.columns)
    ax.set_yticks(range(hm.shape[0]))
    ax.set_yticklabels(hm.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Số clip theo giải và mùa giải")
    save(fig, out, "03_league_season.png")

    # 04. Phút thi đấu
    bins = np.arange(0, 100, 5)
    df["min_bin"] = pd.cut(df.minute, bins, right=False)
    cnt = pd.crosstab(df.min_bin, df.outcome)[OUTCOMES]
    centers = bins[:-1] + 2.5
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1]})
    bottom = np.zeros(len(cnt))
    for o in OUTCOMES:
        a1.bar(centers, cnt[o].to_numpy(), 4.6, bottom=bottom, color=COLOR[o], edgecolor=SURF, linewidth=1.5)
        bottom += cnt[o].to_numpy()
    a1.set_ylabel("Số cú sút")
    a1.set_title("Số cú sút theo phút thi đấu (5 phút một cột; bù giờ 45+ và 90+ gộp vào cột cuối của hiệp)")
    outcome_legend(a1, loc="upper left", ncol=3)
    tot = cnt.sum(axis=1).to_numpy()
    p = cnt["goal"].to_numpy() / np.maximum(tot, 1)
    ci = 1.96 * np.sqrt(p * (1 - p) / np.maximum(tot, 1))
    a2.fill_between(centers, (p - ci) * 100, (p + ci) * 100, color=COLOR["goal"], alpha=0.15, linewidth=0)
    a2.plot(centers, p * 100, color=COLOR["goal"], lw=2, marker="o", ms=4)
    a2.axhline(df.outcome.eq("goal").mean() * 100, color=MUTED, lw=1, ls="--")
    a2.text(1, df.outcome.eq("goal").mean() * 100 + 0.6, "trung bình", color=MUTED, fontsize=8.5)
    a2.axvline(45, color=MUTED, lw=1)
    a1.axvline(45, color=MUTED, lw=1)
    a2.set_ylabel("% cú sút thành bàn")
    a2.set_xlabel("Phút thi đấu")
    a2.set_title("Tỉ lệ thành bàn theo phút (dải mờ = khoảng tin cậy 95%)")
    a2.set_xticks(range(0, 100, 15))
    fig.tight_layout()
    save(fig, out, "04_minute.png")
    stats["shots_by_half"] = df.half.value_counts().to_dict()

    # 05. Theo trận
    g = df.groupby("game_idx").agg(shots=("outcome", "size"), goals=("outcome", lambda s: (s == "goal").sum()),
                                  score_goals=("score_goals", "first"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(g.shots, bins=range(0, g.shots.max() + 2), color=SEQ, edgecolor=SURF, linewidth=1)
    axes[0].axvline(g.shots.median(), color=INK2, lw=1, ls="--")
    axes[0].text(g.shots.median() + 0.4, axes[0].get_ylim()[1] * 0.92, f"trung vị {g.shots.median():.0f}",
                 color=INK2, fontsize=9)
    axes[0].set_xlabel("Số cú sút trong trận")
    axes[0].set_ylabel("Số trận")
    axes[0].set_title("Cú sút mỗi trận")
    axes[1].hist(g.goals, bins=np.arange(-0.5, g.goals.max() + 1.5), color=COLOR["goal"], edgecolor=SURF, linewidth=1)
    axes[1].set_xlabel("Số clip bàn thắng trong trận")
    axes[1].set_ylabel("Số trận")
    axes[1].set_title("Bàn thắng mỗi trận (theo nhãn)")
    gg = g.dropna(subset=["score_goals"])
    tab = pd.crosstab(gg.score_goals.astype(int), gg.goals)
    xs, ys, ss = [], [], []
    for sg in tab.index:
        for lab in tab.columns:
            if tab.loc[sg, lab]:
                xs.append(sg); ys.append(lab); ss.append(tab.loc[sg, lab])
    axes[2].scatter(xs, ys, s=np.array(ss) * 9, color=SEQ, alpha=0.75, edgecolor=SURF, linewidth=1.5)
    m = int(max(gg.score_goals.max(), gg.goals.max())) + 1
    axes[2].plot([0, m], [0, m], color=MUTED, lw=1, ls="--")
    axes[2].set_xlabel("Tổng bàn thắng theo tỉ số chung cuộc")
    axes[2].set_ylabel("Số clip goal trong manifest")
    axes[2].set_title("Nhãn goal so với tỉ số (cỡ chấm = số trận)")
    fig.tight_layout()
    save(fig, out, "05_per_game.png")
    stats["shots_per_game"] = g.shots.describe().round(2).to_dict()
    stats["goal_label_vs_score"] = {
        "equal": int((gg.goals == gg.score_goals).sum()),
        "fewer_labels": int((gg.goals < gg.score_goals).sum()),
        "more_labels": int((gg.goals > gg.score_goals).sum()),
        "missing_goals_total": int((gg.score_goals - gg.goals).clip(lower=0).sum()),
        "score_goals_total": int(gg.score_goals.sum()),
    }

    # 05b. Khoảng cách giữa hai cú sút liên tiếp trong cùng hiệp -> clip chồng lấn và nhãn goal lặp
    d = df.sort_values(["game_idx", "half", "position_ms"]).copy()
    grp = d.groupby(["game_idx", "half"])
    d["gap_prev"] = grp.position_ms.diff() / 1000
    d["prev_outcome"] = grp.outcome.shift()
    gl = d[d.outcome == "goal"].copy()
    gl["gap_prev_goal"] = gl.groupby(["game_idx", "half"]).position_ms.diff() / 1000
    dup_goal = gl[gl.gap_prev_goal < 5]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    lb = np.logspace(-1, np.log10(3000), 45)
    a1.hist(d.gap_prev.dropna(), lb, color=SEQ, alpha=0.9, edgecolor=SURF, linewidth=0.8, label="Mọi cặp cú sút liên tiếp")
    a1.hist(gl.gap_prev_goal.dropna(), lb, histtype="step", lw=2, color=COLOR["saved"], label="Cặp clip goal liên tiếp")
    a1.axvspan(0.1, 10, color=MUTED, alpha=0.12, linewidth=0)
    a1.text(0.13, a1.get_ylim()[1] * 0.9, "< 10 s: hai clip\nchồng lấn nhau", color=INK2, fontsize=9, va="top")
    a1.set_xscale("log")
    a1.set_xlabel("Khoảng cách tới cú sút trước trong cùng hiệp (giây, thang log)")
    a1.set_ylabel("Số cặp")
    a1.set_title("Khoảng cách giữa các cú sút liên tiếp")
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.72))
    gg2 = gg.drop(index=[i for i in gg.index if gg.loc[i, "goals"] - gg.loc[i, "score_goals"] > 8], errors="ignore")
    diff = (gg2.goals - gg2.score_goals).astype(int).value_counts().sort_index()
    a2.bar(diff.index.astype(str), diff.to_numpy(), color=[COLOR["goal"] if k == 0 else COLOR["saved"] for k in diff.index], width=0.6)
    for i, v in enumerate(diff):
        a2.text(i, v + 5, f"{v}", ha="center", color=INK2, fontsize=9)
    a2.set_xlabel("Số nhãn goal trừ số bàn theo tỉ số")
    a2.set_ylabel("Số trận")
    a2.set_title("Nhãn goal so với tỉ số chung cuộc")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "05b_shot_gaps.png")
    stats["quality"] = {
        "pairs_gap_lt_10s": int((d.gap_prev < 10).sum()),
        "clips_overlapping_neighbor": int(((d.gap_prev < 10) | (grp.position_ms.diff(-1).abs() / 1000 < 10)).sum()),
        "goal_after_goal_lt_5s": int(len(dup_goal)),
        "goal_after_goal_lt_5s_by_split": dup_goal.split.value_counts().to_dict(),
        "games_more_goal_labels_than_score": int((gg2.goals > gg2.score_goals).sum()),
        "games_fewer_goal_labels_than_score": int((gg2.goals < gg2.score_goals).sum()),
        "excluded_wrong_score_games": [int(i) for i in gg.index.difference(gg2.index)],
        "goal_diff_hist": {int(k): int(v) for k, v in diff.items()},
        "dup_goal_examples": dup_goal[["path", "game_time", "gap_prev_goal", "caption"]].head(10).to_dict("records"),
    }
    d[["path", "gap_prev", "prev_outcome"]].to_csv(os.path.join(out, "cache", "shot_gaps.csv"), index=False)

    # 06. Caption: độ phủ, độ trễ, độ dài
    cap = df[df.has_caption]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    cov = df.groupby("outcome").has_caption.mean()[OUTCOMES] * 100
    bars = axes[0].bar([OUTCOME_VI[o] for o in OUTCOMES], cov, color=[COLOR[o] for o in OUTCOMES], width=0.6)
    for b, v in zip(bars, cov):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", color=INK2)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("% clip có caption")
    axes[0].set_title("Độ phủ caption")
    axes[0].grid(axis="x", visible=False)
    hb = np.arange(-6, 62, 2)
    for o in OUTCOMES:
        v = cap.loc[cap.outcome == o, "caption_offset_s"].dropna()
        h, _ = np.histogram(v, hb)
        axes[1].step(hb[:-1], h / max(len(v), 1) * 100, where="post", color=COLOR[o], lw=2, label=OUTCOME_VI[o])
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[1].set_xlabel("Caption đăng sau cú sút (giây)")
    axes[1].set_ylabel("% caption (mỗi lớp)")
    axes[1].set_title("Độ trễ caption")
    axes[1].legend()
    data = [cap.loc[cap.outcome == o, "caption_words"].dropna() for o in OUTCOMES]
    bp = axes[2].boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                         medianprops={"color": INK, "lw": 1.5})
    for patch, o in zip(bp["boxes"], OUTCOMES):
        patch.set_facecolor(COLOR[o]); patch.set_alpha(0.85); patch.set_edgecolor(SURF)
    axes[2].set_xticks([1, 2, 3])
    axes[2].set_xticklabels([OUTCOME_VI[o] for o in OUTCOMES])
    axes[2].set_ylabel("Số từ")
    axes[2].set_title("Độ dài caption")
    axes[2].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "06_caption_basic.png")
    stats["caption_coverage"] = cov.round(2).to_dict()
    stats["caption_offset_median"] = float(cap.caption_offset_s.median())
    stats["caption_offset_q"] = cap.caption_offset_s.quantile([0.1, 0.25, 0.75, 0.9]).round(1).to_dict()
    stats["caption_words_median"] = {o: float(d.median()) for o, d in zip(OUTCOMES, data)}

    # 07. Từ khóa lộ kết quả
    rows = {}
    for k, pat in KEYWORDS.items():
        rows[k] = [cap.loc[cap.outcome == o, "caption"].str.lower().str.contains(pat, regex=True).mean() * 100
                   for o in OUTCOMES]
    kw = pd.DataFrame(rows, index=OUTCOMES).T
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.imshow(kw.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f7f9fc", "#9ec5f4", "#2a78d6", "#104281"]), vmin=0, vmax=100, aspect="auto")
    for i in range(kw.shape[0]):
        for j in range(kw.shape[1]):
            v = kw.iat[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9.5, color="white" if v > 55 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([OUTCOME_VI[o] for o in OUTCOMES])
    ax.set_yticks(range(len(kw)))
    ax.set_yticklabels(kw.index)
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("% caption chứa nhóm từ khóa, theo kết quả thật", pad=28)
    save(fig, out, "07_caption_keywords.png")
    stats["keyword_pct"] = kw.round(1).to_dict()

    # 08. Ma trận nhầm lẫn của luật từ khóa
    pred_order = OUTCOMES + ["unknown", "no_caption"]
    cm = pd.crosstab(df.outcome, df.rule_pred).reindex(index=OUTCOMES, columns=pred_order, fill_value=0)
    cmp = cm.div(cm.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.imshow(cmp.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f7f9fc", "#9ec5f4", "#2a78d6", "#104281"]), vmin=0, vmax=100, aspect="auto")
    for i in range(3):
        for j in range(len(pred_order)):
            v = cmp.iat[i, j]
            ax.text(j, i, f"{v:.0f}%\n({cm.iat[i, j]:,})", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 55 else INK)
    ax.set_xticks(range(len(pred_order)))
    ax.set_xticklabels(["→ " + OUTCOME_VI[o] for o in OUTCOMES] + ["→ không rõ", "không caption"])
    ax.set_yticks(range(3))
    ax.set_yticklabels([OUTCOME_VI[o] for o in OUTCOMES])
    ax.set_ylabel("Kết quả thật")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Đoán kết quả chỉ bằng luật từ khóa trên caption (% theo hàng)")
    save(fig, out, "08_caption_rule.png")
    decided = df[df.rule_pred.isin(OUTCOMES)]
    stats["rule"] = {
        "coverage_of_all": round(len(decided) / n, 4),
        "acc_3class_on_decided": round(float((decided.rule_pred == decided.outcome).mean()), 4),
        "acc_hitmiss_on_decided": round(float(((decided.rule_pred == "goal") == (decided.outcome == "goal")).mean()), 4),
        "goal_recall": round(float((df[df.outcome == "goal"].rule_pred == "goal").mean()), 4),
        "goal_precision": round(float((decided[decided.rule_pred == "goal"].outcome == "goal").mean()), 4),
    }
    # Luật nhị phân trên toàn bộ: dự đoán hit nếu có "Goal!/tỉ số", còn lại miss.
    hit_pred = df.rule_pred == "goal"
    hit_true = df.outcome == "goal"
    tp = int((hit_pred & hit_true).sum()); fp = int((hit_pred & ~hit_true).sum()); fn = int((~hit_pred & hit_true).sum())
    stats["rule"]["hit_f1_all_clips"] = round(2 * tp / (2 * tp + fp + fn), 4)
    for s in ["train", "valid", "test"]:
        d = df[df.split == s]
        hp, ht = d.rule_pred == "goal", d.outcome == "goal"
        tp_ = int((hp & ht).sum()); fp_ = int((hp & ~ht).sum()); fn_ = int((~hp & ht).sum())
        stats["rule"][f"hit_f1_{s}"] = round(2 * tp_ / (2 * tp_ + fp_ + fn_), 4)

    # 09. Từ đặc trưng goal vs miss (log-odds có làm mượt)
    tok = lambda s: re.findall(r"[a-z]+!?", s.lower())  # noqa: E731
    cg, cm_ = Counter(), Counter()
    for t, o in zip(cap.caption_anonymized, cap.outcome):
        (cg if o == "goal" else cm_).update(set(tok(t)))
    ng, nm = (cap.outcome == "goal").sum(), (cap.outcome != "goal").sum()
    vocab = [w for w in set(cg) | set(cm_) if cg[w] + cm_[w] >= 40 and len(w) > 2]
    lo = {w: np.log((cg[w] + 1) / (ng - cg[w] + 1)) - np.log((cm_[w] + 1) / (nm - cm_[w] + 1)) for w in vocab}
    s = pd.Series(lo).sort_values()
    top = pd.concat([s.head(12), s.tail(12)])
    fig, ax = plt.subplots(figsize=(8, 7))
    yy = np.arange(len(top))
    ax.barh(yy, top.to_numpy(), color=[COLOR["goal"] if v > 0 else COLOR["saved"] for v in top], height=0.66)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{w}  ({cg[w] / ng * 100:.0f}% / {cm_[w] / nm * 100:.0f}%)" for w in top.index], fontsize=9)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("log-odds: dương = hay gặp ở bàn thắng, âm = hay gặp ở cú sút hỏng")
    ax.set_title("Từ phân biệt goal và miss trong caption\n(trong ngoặc: % caption goal / % caption miss có từ đó)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "09_caption_words.png")


# ---------------------------------------------------------------- probe
def probe_one(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=codec_type,codec_name,width,height,avg_frame_rate,bit_rate,sample_rate,channels,nb_frames:"
           "format=duration,size,bit_rate", "-of", "json", path]
    try:
        j = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
    except Exception:
        return {"file": path, "ok": False}
    r = {"file": path, "ok": True, "duration": float(j["format"].get("duration", "nan")),
         "size_mb": int(j["format"].get("size", 0)) / 1e6, "total_kbps": int(j["format"].get("bit_rate", 0)) / 1e3}
    for st in j.get("streams", []):
        if st["codec_type"] == "video":
            num, den = st.get("avg_frame_rate", "0/1").split("/")
            r.update(vcodec=st.get("codec_name"), width=st.get("width"), height=st.get("height"),
                     fps=float(num) / float(den) if float(den) else np.nan,
                     v_kbps=int(st.get("bit_rate", 0)) / 1e3, nb_frames=int(st.get("nb_frames", 0)))
        elif st["codec_type"] == "audio":
            r.update(acodec=st.get("codec_name"), sample_rate=int(st.get("sample_rate", 0)),
                     channels=st.get("channels"), a_kbps=int(st.get("bit_rate", 0)) / 1e3)
    return r


def stage_probe(df, root, out, stats, workers):
    cache = os.path.join(out, "cache", "probe.csv")
    if os.path.exists(cache):
        pr = pd.read_csv(cache)
    else:
        print(f"[probe] ffprobe {len(df):,} clip")
        paths = [os.path.join(root, p) for p in df.path]
        with ThreadPoolExecutor(workers) as ex:
            pr = pd.DataFrame(list(ex.map(probe_one, paths)))
        pr["path"] = df.path.to_numpy()
        pr.to_csv(cache, index=False)
    pr = pr.merge(df[["path", "outcome", "split"]], on="path")
    pr["res"] = pr.width.astype("Int64").astype(str) + "×" + pr.height.astype("Int64").astype(str)

    on_disk = set()
    for s in ["train", "valid", "test"]:
        on_disk |= {f"clips/{s}/{f}" for f in os.listdir(os.path.join(root, "clips", s))}
    stats["probe"] = {
        "probe_failed": int((~pr.ok.astype(bool)).sum()),
        "files_on_disk": len(on_disk),
        "extra_files_not_in_manifest": len(on_disk - set(df.path)),
        "missing_files": len(set(df.path) - on_disk),
        "vcodec": pr.vcodec.value_counts().to_dict(),
        "acodec": pr.acodec.value_counts().to_dict(),
        "resolution": pr.res.value_counts().to_dict(),
        "fps": pr.fps.round(2).value_counts().to_dict(),
        "duration": pr.duration.describe().round(3).to_dict(),
        "duration_not_10s": int((pr.duration.sub(10).abs() > 0.2).sum()),
        "nb_frames": pr.nb_frames.describe().round(1).to_dict(),
        "sample_rate": pr.sample_rate.value_counts().to_dict(),
        "channels": pr.channels.value_counts().to_dict(),
        "no_audio": int(pr.acodec.isna().sum()),
        "v_kbps_by_res": pr.groupby("res").v_kbps.median().round(0).to_dict(),
        "size_mb": pr.size_mb.describe().round(2).to_dict(),
        "size_total_gb": round(pr.size_mb.sum() / 1e3, 2),
        "v_kbps_by_outcome": pr.groupby("outcome").v_kbps.median().round(0).to_dict(),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    a = axes[0, 0]
    a.hist(pr.duration, bins=np.linspace(pr.duration.min() - 0.05, pr.duration.max() + 0.05, 30),
           color=SEQ, edgecolor=SURF)
    a.set_yscale("log")
    a.set_xlabel("Độ dài clip (giây)")
    a.set_ylabel("Số clip (thang log)")
    a.set_title("Độ dài clip")
    a = axes[0, 1]
    rc = pr.res.value_counts()
    a.bar(rc.index, rc.to_numpy(), color=SEQ, width=0.5)
    for i, v in enumerate(rc):
        a.text(i, v * 1.02, f"{v:,}", ha="center", color=INK2)
    a.set_ylabel("Số clip")
    a.set_title("Độ phân giải")
    a.grid(axis="x", visible=False)
    a = axes[1, 0]
    res_grp = pr.res.where(pr.res.isin(["1280×720", "1920×1080"]), "Khác")
    for r_, col in [("1280×720", COLOR["goal"]), ("1920×1080", COLOR["saved"]), ("Khác", COLOR["off_target"])]:
        v = pr.loc[res_grp == r_, "v_kbps"]
        a.hist(v / 1000, bins=np.linspace(0, pr.v_kbps.quantile(0.995) / 1000, 50), histtype="step", lw=2,
               density=True, label=f"{r_}, n={len(v):,} (trung vị {v.median() / 1000:.1f} Mb/s)", color=col)
    a.set_xlabel("Bitrate video (Mb/s)")
    a.set_ylabel("Mật độ")
    a.set_title("Bitrate video")
    a.legend()
    a = axes[1, 1]
    for o in OUTCOMES:
        v = pr.loc[pr.outcome == o, "size_mb"]
        a.hist(v, bins=np.linspace(0, pr.size_mb.quantile(0.995), 50), histtype="step", lw=2, density=True,
               color=COLOR[o], label=f"{OUTCOME_VI[o]} (trung vị {v.median():.1f} MB)")
    a.set_xlabel("Dung lượng file (MB)")
    a.set_ylabel("Mật độ")
    a.set_title("Dung lượng file theo kết quả")
    a.legend()
    fig.tight_layout()
    save(fig, out, "10_tech_specs.png")


# ---------------------------------------------------------------- content
STEP = 2          # lấy 1 trên 2 frame -> 12.5 fps
SMALL = (192, 108)
N_V = 125         # số mẫu video trên 10 s
WIN_A = 0.1       # cửa sổ audio (giây)
N_A = 100
SHOT_T = 6.0
CUT_THR = 0.45    # ngưỡng khoảng cách Bhattacharyya giữa hai histogram HSV liên tiếp


def analyze_clip(path):
    cv2.setNumThreads(1)
    motion = np.full(N_V, np.nan); green = np.full(N_V, np.nan); bright = np.full(N_V, np.nan)
    hdist = np.full(N_V, np.nan); sat = np.full(N_V, np.nan)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0; k = 0; prev_g = prev_h = None
    while k < N_V and cap.grab():
        if i % STEP == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            idx = int(round(i / fps * N_V / 10.0))
            if idx >= N_V:
                break
            sm = cv2.resize(fr, SMALL, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = cv2.inRange(hsv, (32, 50, 40), (88, 255, 255))
            green[idx] = mask.mean() / 255
            bright[idx] = gray.mean() / 255
            sat[idx] = hsv[..., 1].mean() / 255
            h = cv2.calcHist([hsv], [0, 1], None, [30, 8], [0, 180, 0, 256])
            cv2.normalize(h, h)
            if prev_g is not None:
                motion[idx] = np.abs(gray - prev_g).mean() / 255
                hdist[idx] = cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA)
            prev_g, prev_h = gray, h
            k += 1
        i += 1
    cap.release()
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    w = int(16000 * WIN_A)
    rms_db = np.full(N_A, np.nan)
    flat = np.full(N_A, np.nan)
    m = min(N_A, len(a) // w)
    if m:
        seg = a[: m * w].reshape(m, w)
        rms_db[:m] = 20 * np.log10(np.sqrt((seg ** 2).mean(1)) + 1e-6)
        spec = np.abs(np.fft.rfft(seg * np.hanning(w), axis=1)) + 1e-9
        flat[:m] = np.exp(np.log(spec).mean(1)) / spec.mean(1)   # spectral flatness: tiếng ồn đám đông cao hơn giọng nói
    return dict(motion=motion, green=green, bright=bright, hdist=hdist, sat=sat, rms=rms_db, flat=flat)


def sample_clips(df, per_outcome, seed=0):
    parts = []
    for o in OUTCOMES:
        d = df[df.outcome == o]
        parts.append(d.sample(min(per_outcome, len(d)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def mean_ci(x):
    m = np.nanmean(x, 0)
    n = np.sum(~np.isnan(x), 0)
    se = np.nanstd(x, 0) / np.sqrt(np.maximum(n, 1))
    return m, 1.96 * se


def montage(df, root, out, per_outcome=3, seed=1):
    times = [1.0, 4.0, 6.0, 7.0, 8.0, 9.5]
    tw, th = 320, 180
    rows = []
    for o in OUTCOMES:
        d = df[(df.outcome == o)].sample(per_outcome, random_state=seed)
        for _, r in d.iterrows():
            cap = cv2.VideoCapture(os.path.join(root, r.path))
            cells = []
            for t in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, fr = cap.read()
                fr = cv2.resize(fr, (tw, th)) if ok else np.zeros((th, tw, 3), np.uint8)
                cells.append(cv2.copyMakeBorder(fr, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(251, 252, 252)))
            cap.release()
            label = np.full((th + 6, 150, 3), (251, 252, 252), np.uint8)
            bgr = tuple(int(COLOR[o][i:i + 2], 16) for i in (5, 3, 1))
            cv2.rectangle(label, (8, 10), (20, th - 4), bgr, -1)
            cv2.putText(label, o.upper().replace("_", " "), (28, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
            cv2.putText(label, r.path.split("/")[-1][:-4], (28, th // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (78, 81, 82), 1, cv2.LINE_AA)
            rows.append(np.hstack([label] + cells))
    head = np.full((34, rows[0].shape[1], 3), (251, 252, 252), np.uint8)
    for j, t in enumerate(times):
        rel = t - SHOT_T
        txt = "shot (0s)" if rel == 0 else f"{rel:+.1f}s"
        cv2.putText(head, txt, (150 + j * (tw + 6) + tw // 2 - 30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (11, 11, 11), 1, cv2.LINE_AA)
    img = np.vstack([head] + rows)
    path = os.path.join(out, "13_frames_montage.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("  ->", path)


def stage_content(df, root, out, stats, per_outcome, workers):
    cache = os.path.join(out, "cache", f"content_{per_outcome}.npz")
    smp = sample_clips(df, per_outcome)
    if os.path.exists(cache):
        z = np.load(cache)
        feats = {k: z[k] for k in z.files}
    else:
        print(f"[content] OpenCV + audio trên {len(smp)} clip mẫu, {workers} tiến trình")
        paths = [os.path.join(root, p) for p in smp.path]
        res = []
        with ProcessPoolExecutor(workers) as ex:
            for j, r in enumerate(ex.map(analyze_clip, paths, chunksize=2)):
                res.append(r)
                if (j + 1) % 100 == 0:
                    print(f"    {j + 1}/{len(paths)}")
        feats = {k: np.stack([r[k] for r in res]) for k in res[0]}
        np.savez_compressed(cache, **feats)
    feats["cut"] = (feats["hdist"] > CUT_THR).astype(float)
    feats["cut"][np.isnan(feats["hdist"])] = np.nan
    # Audio so với mức nền trước cú sút (median của 0-5 s) để bỏ khác biệt âm lượng giữa các đài.
    base = np.nanmedian(feats["rms"][:, :50], axis=1, keepdims=True)
    feats["rms_rel"] = feats["rms"] - base
    tv = np.arange(N_V) * 10 / N_V - SHOT_T
    ta = (np.arange(N_A) + 0.5) * WIN_A - SHOT_T
    oc = smp.outcome.to_numpy()

    # 11. Diễn biến theo thời gian quanh cú sút
    panels = [
        ("motion", tv, "Chuyển động\n(chênh lệch frame)", 100, 3),
        ("cut", tv, "Tỉ lệ clip có cắt cảnh\n(%, cửa sổ 0.5 s)", 100, None),
        ("green", tv, "Tỉ lệ mặt cỏ\ntrong khung hình (%)", 100, None),
        ("rms_rel", ta, "Độ ồn so với nền\ntrước cú sút (dB)", 1, 3),
        ("flat", ta, "Độ phẳng phổ âm thanh\n(cao = tiếng ồn đám đông)", 1, 3),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 12.5), sharex=True)
    for ax, (key, t, lab, scale, smooth) in zip(axes, panels):
        for o in OUTCOMES:
            x = feats[key][oc == o]
            if key == "cut":
                nb = 20
                xb = np.nanmax(x[:, : (x.shape[1] // nb) * nb].reshape(len(x), nb, -1), axis=2)  # có cắt trong 0.5 s
                m, ci = mean_ci(xb)
                tt = (np.arange(nb) + 0.5) * 10 / nb - SHOT_T
            else:
                if smooth:
                    x = pd.DataFrame(x).T.rolling(smooth, center=True, min_periods=1).mean().T.to_numpy()
                # Bỏ mẫu đầu/cuối: frame đầu chưa có frame trước, cửa sổ audio cuối bị thiếu.
                x = x.copy(); x[:, :2] = np.nan; x[:, -2:] = np.nan
                m, ci = mean_ci(x)
                tt = t
            ax.fill_between(tt, (m - ci) * scale, (m + ci) * scale, color=COLOR[o], alpha=0.15, linewidth=0)
            ax.plot(tt, m * scale, color=COLOR[o], lw=2, label=OUTCOME_VI[o])
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.set_ylabel(lab)
    axes[0].text(0.08, axes[0].get_ylim()[1] * 0.97, "cú sút", color=INK2, fontsize=9, va="top")
    axes[0].legend(ncol=3, loc="upper left")
    axes[0].set_title(f"Diễn biến trung bình quanh thời điểm sút ({per_outcome} clip mỗi lớp, dải mờ = KTC 95%)")
    axes[-1].set_xlabel("Thời gian so với cú sút (giây)")
    axes[-1].set_xticks(range(-6, 5))
    fig.tight_layout()
    save(fig, out, "11_timeline.png")

    # 12. Đặc trưng một con số mỗi clip, trước và sau cú sút
    pre_v, post_v = tv < 0, tv >= 0
    pre_a, post_a = ta < 0, ta >= 0
    scal = pd.DataFrame({
        "outcome": oc,
        "Số lần cắt cảnh sau sút": np.nansum(feats["cut"][:, post_v], 1),
        "Số lần cắt cảnh trước sút": np.nansum(feats["cut"][:, pre_v], 1),
        "Tỉ lệ cỏ sau sút (%)": np.nanmean(feats["green"][:, post_v], 1) * 100,
        "Tỉ lệ cỏ trước sút (%)": np.nanmean(feats["green"][:, pre_v], 1) * 100,
        "Độ ồn sau sút (dB so với nền)": np.nanmean(feats["rms_rel"][:, post_a], 1),
        "Độ ồn trước sút (dB so với nền)": np.nanmean(feats["rms_rel"][:, (ta >= -1) & (ta < 0)], 1),
        "Chuyển động sau sút": np.nanmean(feats["motion"][:, post_v], 1) * 100,
        "Chuyển động trước sút": np.nanmean(feats["motion"][:, pre_v], 1) * 100,
    })
    scal["path"] = smp.path.to_numpy()
    scal.to_csv(os.path.join(out, "cache", "content_scalars.csv"), index=False)
    cols = [c for c in scal.columns if c not in ("outcome", "path")]
    aucs = {}
    for c in cols:
        a_ = auc(scal.loc[scal.outcome == "goal", c], scal.loc[scal.outcome != "goal", c])
        aucs[c] = round(a_, 3)
    stats["content"] = {"n_per_outcome": per_outcome, "auc_goal_vs_miss": aucs,
                        "median_by_outcome": scal.groupby("outcome")[cols].median().round(2).to_dict()}

    order = ["Số lần cắt cảnh trước sút", "Số lần cắt cảnh sau sút", "Tỉ lệ cỏ trước sút (%)", "Tỉ lệ cỏ sau sút (%)",
             "Độ ồn trước sút (dB so với nền)", "Độ ồn sau sút (dB so với nền)", "Chuyển động trước sút",
             "Chuyển động sau sút"]
    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    for ax, c in zip(axes.ravel(), order):
        data = [scal.loc[scal.outcome == o, c].dropna() for o in OUTCOMES]
        if "cắt cảnh" in c:
            # Phần lớn clip có 0 lần cắt -> boxplot vô nghĩa; vẽ % clip có ít nhất 1 lần cắt.
            v = [(d > 0).mean() * 100 for d in data]
            ax.bar([1, 2, 3], v, color=[COLOR[o] for o in OUTCOMES], width=0.5)
            for xi, vi in zip([1, 2, 3], v):
                ax.text(xi, vi + 1, f"{vi:.0f}%", ha="center", color=INK2, fontsize=9)
            ax.set_ylim(0, 100)
            ax.set_ylabel("% clip có ≥ 1 lần cắt cảnh")
        else:
            bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": INK, "lw": 1.5})
            for patch, o in zip(bp["boxes"], OUTCOMES):
                patch.set_facecolor(COLOR[o]); patch.set_alpha(0.85); patch.set_edgecolor(SURF)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([OUTCOME_VI[o] for o in OUTCOMES])
        ax.set_title(f"{c}   ·   AUC goal vs miss = {aucs[c]:.2f}", fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Một số đặc trưng đơn giản: cột trái = trước cú sút, cột phải = sau cú sút\n"
                 "(AUC 0.5 = không phân biệt được; càng xa 0.5 càng phân biệt tốt)", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "12_shortcut_features.png")

    # 13. Ảnh ghép frame mẫu quanh cú sút
    montage(df, root, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--stages", default="meta,probe,content")
    ap.add_argument("--per-outcome", type=int, default=300)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True)
    stats_path = os.path.join(args.out, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    df = load_manifest(args.root)
    st = args.stages.split(",")
    if "meta" in st:
        stage_meta(df, args.out, stats)
    if "probe" in st:
        stage_probe(df, args.root, args.out, stats, args.workers)
    if "content" in st:
        stage_content(df, args.root, args.out, stats, args.per_outcome, args.workers)
    json.dump(stats, open(stats_path, "w"), indent=2, ensure_ascii=False, default=str)
    print("stats ->", stats_path)


if __name__ == "__main__":
    main()
