"""Generate the polished plot set used by README.md and BLOG.md.

Reads `phase1_eval_metrics.json` artifacts and writes PNGs to `assets/`.
Run with the project's training venv (matplotlib must be installed):

    /Users/ehsaan/Documents/lab/circuit_detective/.venv-train/bin/python scripts/make_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts"
ASSETS = REPO / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


ACCENT = "#4C5FD5"
ACCENT_DARK = "#2E3A8C"
ACCENT_LIGHT = "#A8B1E8"
GRAY = "#9CA3AF"
GRAY_LIGHT = "#E5E7EB"
GREEN = "#16A34A"
RED = "#DC2626"
AMBER = "#D97706"
TEXT = "#1F2937"

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def load(run: str) -> tuple[dict, dict]:
    payload = json.loads((ARTIFACTS / run / "phase1_eval_metrics.json").read_text())
    return payload.get("before") or {}, payload.get("after") or {}


def annotate_bars(ax, bars, values, fmt: str = "{:.1%}", offset: float = 0.012):
    ymax = ax.get_ylim()[1]
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset * ymax,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT,
            fontweight="600",
        )


def grouped_before_after(
    *,
    title: str,
    subtitle: str,
    metrics: list[tuple[str, float, float, str]],
    out: Path,
    rollouts: int,
):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    n = len(metrics)
    x = np.arange(n)
    width = 0.36

    before_vals = [m[1] for m in metrics]
    after_vals = [m[2] for m in metrics]
    fmts = [m[3] for m in metrics]

    b1 = ax.bar(x - width / 2, before_vals, width, color=GRAY_LIGHT, edgecolor=GRAY,
                linewidth=0.8, label="Before training")
    b2 = ax.bar(x + width / 2, after_vals, width, color=ACCENT, edgecolor=ACCENT_DARK,
                linewidth=0.8, label="After training")

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Rate / value")
    ymax = max(max(before_vals), max(after_vals)) * 1.22
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    for bar, val, fmt in zip(b1, before_vals, fmts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012 * ymax,
                fmt.format(val), ha="center", va="bottom", fontsize=9.5, color=TEXT)
    for bar, val, fmt in zip(b2, after_vals, fmts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012 * ymax,
                fmt.format(val), ha="center", va="bottom", fontsize=10, color=ACCENT_DARK,
                fontweight="700")

    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(axis="y", color=GRAY_LIGHT, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle(title, x=0.06, ha="left", fontsize=15, fontweight="700", color=TEXT, y=0.99)
    ax.set_title(subtitle + f"   ·   n = {rollouts} eval rollouts",
                 loc="left", fontsize=11, fontweight="400", color=GRAY, pad=4)

    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.relative_to(REPO)}")


def phase1_before_after():
    _, a = load("phase1_sft64_grpo200_a10g_large")
    b = json.loads(
        (ARTIFACTS / "phase1_sft64_grpo200_a10g_large" / "phase1_eval_metrics.json").read_text()
    )["before"]
    metrics = [
        ("Success rate", b["eval_before_success_rate"], a["eval_after_success_rate"], "{:.1%}"),
        ("Submit rate", b["eval_before_submit_rate"], a["eval_after_submit_rate"], "{:.1%}"),
        ("Mean F1", b["eval_before_mean_f1"], a["eval_after_mean_f1"], "{:.2f}"),
        ("Mean reward", b["eval_before_mean_reward"], a["eval_after_mean_reward"], "{:.2f}"),
    ]
    grouped_before_after(
        title="Phase 1 — induction localization (canonical run)",
        subtitle="SFT64 + GRPO200 on Qwen3.5-2B, target = L1H6 in attn-only-2l",
        metrics=metrics,
        out=ASSETS / "phase1_before_after.png",
        rollouts=int(a["eval_after_rollouts"]),
    )


def phase2_before_after():
    b, a = load("planted_lite_naive_max_sft1536_grpo300_ctx1024")
    metrics = [
        ("Causal success", b["eval_before_causal_success_rate"],
         a["eval_after_causal_success_rate"], "{:.1%}"),
        ("Submit rate", b["eval_before_submit_rate"], a["eval_after_submit_rate"], "{:.1%}"),
        ("Ablate rate", b["eval_before_ablate_rate"], a["eval_after_ablate_rate"], "{:.1%}"),
        ("Mean F1", b["eval_before_mean_f1"], a["eval_after_mean_f1"], "{:.2f}"),
    ]
    grouped_before_after(
        title="Phase 2 — planted-lite causal chain (canonical run)",
        subtitle="SFT1536 + GRPO300, agent must ablate both candidates and submit max-delta head",
        metrics=metrics,
        out=ASSETS / "phase2_before_after.png",
        rollouts=int(a["eval_after_rollouts"]),
    )


def phase1_progression():
    runs_in_order = [
        ("phase1_shaped_smoke_20b", "GRPO only, 20 steps"),
        ("phase1_submit_tuned_50", "+ submit shaping, 50 steps"),
        ("phase1_submit_tuned_150", "+ submit shaping, 150 steps"),
        ("phase1_sft_grpo_75_a10g_large", "+ SFT warm-start, 75 steps"),
        ("phase1_sft_grpo_150_a10g_large", "+ SFT warm-start, 150 steps"),
        ("phase1_sft64_grpo200_a10g_large", "SFT64 + GRPO200 (canonical)"),
    ]
    after_success = []
    rollouts = []
    labels = []
    for run, label in runs_in_order:
        _, a = load(run)
        after_success.append(a["eval_after_success_rate"])
        rollouts.append(int(a["eval_after_rollouts"]))
        labels.append(label)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(runs_in_order))
    colors = [GRAY_LIGHT] * (len(runs_in_order) - 1) + [ACCENT]
    edge = [GRAY] * (len(runs_in_order) - 1) + [ACCENT_DARK]
    bars = ax.bar(x, after_success, color=colors, edgecolor=edge, linewidth=0.8, width=0.62)

    ax.axhline(0.40, color=GREEN, linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    ax.text(-0.45, 0.41, "Phase 1 gate (>=40%)", ha="left", va="bottom",
            fontsize=9, color=GREEN, fontweight="600")

    for bar, v, n in zip(bars, after_success, rollouts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10,
                color=ACCENT_DARK if v == max(after_success) else TEXT, fontweight="700")
        ax.text(bar.get_x() + bar.get_width() / 2, -0.05,
                f"n={n}", ha="center", va="top", fontsize=8.5, color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=9.5)
    ax.set_ylabel("Eval success rate (after training)")
    ax.set_ylim(-0.08, 1.0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRAY_LIGHT, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("Phase 1 iteration journey", x=0.06, ha="left", fontsize=15,
                 fontweight="700", color=TEXT, y=0.99)
    ax.set_title("Six runs from shaped GRPO smoke to the canonical SFT+GRPO result",
                 loc="left", fontsize=11, fontweight="400", color=GRAY, pad=4)

    fig.savefig(ASSETS / "phase1_progression.png")
    plt.close(fig)
    print(f"saved {(ASSETS / 'phase1_progression.png').relative_to(REPO)}")


def phase2_journey():
    runs = [
        ("phase2_ablation_required_grpo120_a10g_large",
         "Phase 2 v1\nablation-required",
         "Shortcut: solves task\nwithout ablating", RED),
        ("phase2_strict_sft64_grpo160_a10g_large",
         "Phase 2 v2\nstrict reward",
         "Freeze: ablates 96%,\nbut won't submit", AMBER),
        ("planted_lite_naive_max_sft1536_grpo300_ctx1024",
         "planted-lite\nfinal (canonical)",
         "Solved: full\ncausal chain", GREEN),
    ]
    causal_success = []
    submit_rate = []
    ablate_rate = []
    rollouts = []
    labels = []
    captions = []
    accent_colors = []
    for run, label, caption, color in runs:
        _, a = load(run)
        causal_success.append(a.get("eval_after_causal_success_rate") or 0.0)
        submit_rate.append(a.get("eval_after_submit_rate") or 0.0)
        ablate_rate.append(a.get("eval_after_ablate_rate") or 0.0)
        rollouts.append(int(a["eval_after_rollouts"]))
        labels.append(label)
        captions.append(caption)
        accent_colors.append(color)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    n = len(runs)
    x = np.arange(n)
    width = 0.25

    series = [
        ("Ablate rate", ablate_rate, ACCENT_LIGHT, ACCENT),
        ("Submit rate", submit_rate, ACCENT, ACCENT_DARK),
        ("Causal success", causal_success, ACCENT_DARK, "#1E2657"),
    ]
    for i, (name, vals, color, edge) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=color, edgecolor=edge,
                      linewidth=0.8, label=name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=TEXT,
                    fontweight="600")

    for i, (caption, color) in enumerate(zip(captions, accent_colors)):
        ax.text(i, -0.13, caption, ha="center", va="top", fontsize=9.5,
                color=color, fontweight="700")
        ax.text(i, -0.27, f"n={rollouts[i]}", ha="center", va="top", fontsize=9, color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="600")
    ax.set_ylim(-0.32, 1.15)
    ax.set_ylabel("Eval rate (after training)")
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.grid(axis="y", color=GRAY_LIGHT, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.suptitle("Phase 2 reward-design journey", x=0.06, ha="left", fontsize=15,
                 fontweight="700", color=TEXT, y=0.995)
    ax.set_title(
        "Each reward design exposes a different failure mode until planted-lite produces causal solves",
        loc="left", fontsize=10.5, fontweight="400", color=GRAY, pad=4,
    )

    fig.savefig(ASSETS / "phase2_journey.png")
    plt.close(fig)
    print(f"saved {(ASSETS / 'phase2_journey.png').relative_to(REPO)}")


def sft_vs_grpo_contribution():
    """For each phase: bars showing pre-SFT, post-SFT (= before-GRPO), post-GRPO."""
    _, p1_a = load("phase1_sft64_grpo200_a10g_large")
    p1_b = json.loads(
        (ARTIFACTS / "phase1_sft64_grpo200_a10g_large" / "phase1_eval_metrics.json").read_text()
    )["before"]
    _, p1_no_sft = load("phase1_shaped_smoke_20b")

    _, p2_a = load("planted_lite_naive_max_sft1536_grpo300_ctx1024")
    p2_b = json.loads(
        (ARTIFACTS / "planted_lite_naive_max_sft1536_grpo300_ctx1024" / "phase1_eval_metrics.json").read_text()
    )["before"]
    _, p2_no_sft = load("phase2_ablation_required_grpo120_a10g_large")

    phases = [
        ("Phase 1 — induction localization\nMetric: success rate",
         [p1_no_sft["eval_after_success_rate"],
          p1_b["eval_before_success_rate"],
          p1_a["eval_after_success_rate"]],
         "Phase 1 progresses from 0% (no SFT) to 10% (SFT only) to 79% (SFT+GRPO).\nGRPO contributes most of the gain."),
        ("Phase 2 — planted-lite causal chain\nMetric: causal success rate",
         [p2_no_sft.get("eval_after_causal_success_rate", 0.0),
          p2_b["eval_before_causal_success_rate"],
          p2_a["eval_after_causal_success_rate"]],
         "Phase 2 progresses from 0% (no terminal-clinic SFT) to 95% (SFT only) to 98% (SFT+GRPO).\nSFT carries most of the gain; GRPO provides a final cleanup."),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    stages = ["No SFT", "SFT only\n(before GRPO)", "SFT + GRPO\n(after training)"]
    colors = [GRAY_LIGHT, ACCENT_LIGHT, ACCENT]
    edges = [GRAY, ACCENT, ACCENT_DARK]

    for ax, (title, vals, caption) in zip(axes, phases):
        x = np.arange(len(stages))
        bars = ax.bar(x, vals, color=colors, edgecolor=edges, linewidth=0.8, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.022,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=11,
                    color=ACCENT_DARK if v == max(vals) else TEXT, fontweight="700")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=10)
        ax.set_ylim(0, 1.12)
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="700", color=TEXT, pad=8)
        ax.grid(axis="y", color=GRAY_LIGHT, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.0, -0.32, caption, transform=ax.transAxes, fontsize=9.5, color=GRAY,
                ha="left", va="top")

    fig.suptitle("SFT vs GRPO contribution",
                 x=0.06, ha="left", fontsize=15, fontweight="700", color=TEXT, y=1.0)
    fig.text(0.06, 0.94, "Decomposing the canonical results into the SFT and GRPO stages",
             fontsize=10.5, color=GRAY)
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.18)
    fig.savefig(ASSETS / "sft_vs_grpo_contribution.png")
    plt.close(fig)
    print(f"saved {(ASSETS / 'sft_vs_grpo_contribution.png').relative_to(REPO)}")


def phase2_reward_distribution():
    b, a = load("planted_lite_naive_max_sft1536_grpo300_ctx1024")
    rc_before = b.get("eval_before_reward_counts") or {}
    rc_after = a.get("eval_after_reward_counts") or {}

    keys = sorted(set(rc_before) | set(rc_after), key=lambda s: float(s))
    before_vals = [rc_before.get(k, 0) for k in keys]
    after_vals = [rc_after.get(k, 0) for k in keys]
    total_b = sum(before_vals)
    total_a = sum(after_vals)

    labels = []
    for k in keys:
        f = float(k)
        if f >= 5.0:
            labels.append(f"+{f:.2f}\n(full causal\ncredit)")
        elif f > 0:
            labels.append(f"+{f:.2f}")
        else:
            labels.append(f"{f:.2f}")

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    n = len(keys)
    x = np.arange(n)
    width = 0.4

    b1 = ax.bar(x - width / 2, before_vals, width, color=GRAY_LIGHT, edgecolor=GRAY,
                linewidth=0.8, label=f"Before training (n={total_b})")
    b2 = ax.bar(x + width / 2, after_vals, width, color=ACCENT, edgecolor=ACCENT_DARK,
                linewidth=0.8, label=f"After training (n={total_a})")

    for bars, vals, total in [(b1, before_vals, total_b), (b2, after_vals, total_a)]:
        for bar, v in zip(bars, vals):
            if v == 0:
                continue
            pct = v / total
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(after_vals) * 0.012,
                    f"{v}\n{pct:.1%}", ha="center", va="bottom", fontsize=9, color=TEXT,
                    fontweight="600")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Eval rollout count")
    ax.set_ylim(0, max(after_vals) * 1.18)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(axis="y", color=GRAY_LIGHT, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("Phase 2 — reward-bucket distribution",
                 x=0.06, ha="left", fontsize=15, fontweight="700", color=TEXT, y=0.99)
    ax.set_title(
        "After training, 250 of 256 eval rollouts hit the full causal-credit reward (5.00).",
        loc="left", fontsize=10.5, fontweight="400", color=GRAY, pad=4,
    )

    fig.savefig(ASSETS / "phase2_reward_distribution.png")
    plt.close(fig)
    print(f"saved {(ASSETS / 'phase2_reward_distribution.png').relative_to(REPO)}")


if __name__ == "__main__":
    phase1_before_after()
    phase1_progression()
    phase2_before_after()
    phase2_journey()
    sft_vs_grpo_contribution()
    phase2_reward_distribution()
    print("done.")
