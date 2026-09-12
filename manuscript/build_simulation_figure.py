#!/usr/bin/env python3
"""Build paper figures from archived summaries; never rerun simulations."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from boundary_contamination_study import analyze_boundary_contamination_study as analysis

BLUE = "#2C6EAD"
RED = "#B24745"
GREEN = "#39835B"
PURPLE = "#8562A6"
SAMPLE_LABELS = ["250", "1k", "4k", "16k", "64k", "256k", "1024k"]


def sample_axis(ax, summary):
    ax.set_xscale("log", base=4)
    ax.set_xticks(summary.n, SAMPLE_LABELS, rotation=45, ha="right")
    ax.set_xlabel(r"Sample size, $n=N$")
    ax.grid(alpha=0.18)


def intervals(ax, summary, bounds, title):
    lo, hi = bounds["aux_lower"], bounds["aux_upper"]
    ax.axvspan(lo, hi, color="#D8D8D8", alpha=0.65, zorder=0)
    for end in (lo, hi):
        ax.axvline(end, color="#6D6D6D", linestyle="--", linewidth=1.25)
    y = np.arange(len(summary))
    segments = ax.hlines(y, summary.mean_lower, summary.mean_upper,
                         color=BLUE, linewidth=2.7, zorder=3)
    ax.scatter(summary.mean_lower, y, color=BLUE, s=27, zorder=4)
    ax.scatter(summary.mean_upper, y, color=BLUE, s=27, zorder=4)
    ax.set_yticks(y, [f"{n:,}" for n in summary.n])
    ax.set_ylim(len(summary) - 0.45, -0.55)
    ax.set_xlim(0.025, 0.137)
    ax.set_xticks([0.04, 0.08, 0.12])
    ax.set_xlabel("ATE")
    ax.set_ylabel(r"Sample size, $n=N$")
    ax.set_title(title, loc="left", pad=12)
    ax.grid(axis="y", alpha=0.15)
    # Verify that every plotted segment uses the archived endpoint means.
    for j, seg in enumerate(segments.get_segments()):
        np.testing.assert_array_equal(seg[:, 0],
            [summary.mean_lower.iloc[j], summary.mean_upper.iloc[j]])
    np.testing.assert_allclose(summary.mean_width,
                               summary.mean_upper - summary.mean_lower, atol=1e-14)


def errors(ax, summary):
    keys = ["scaled_lower_rmse", "scaled_upper_rmse", "scaled_joint_q95"]
    labels = ["Lower-endpoint RMSE", "Upper-endpoint RMSE", "95th pct. joint error"]
    for key, label, color, marker in zip(keys, labels, [BLUE, RED, GREEN], ["o", "s", "^"]):
        line, = ax.plot(summary.n, summary[key], label=label, color=color,
                        marker=marker, linewidth=2, markersize=5)
        np.testing.assert_array_equal(line.get_ydata(), summary[key])
    ax.set_ylim(0.8, 3.1)
    ax.set_ylabel(r"Error multiplied by $\sqrt{n}$")
    ax.set_title("(b) Errors: Design 2, $K=4$", loc="left", pad=12)
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.54, 0.43),
              fontsize=12.5, labelspacing=0.55)
    sample_axis(ax, summary)


def feasibility(ax, summaries):
    for (name, frame), color, marker in zip(summaries.items(),
            [BLUE, GREEN, PURPLE, RED], ["o", "s", "^", "D"]):
        line, = ax.plot(frame.n, frame.raw_feasibility_rate, label=name,
                        color=color, marker=marker, linewidth=2, markersize=5,
                        linestyle="--" if name.startswith("Design 2") else "-")
        np.testing.assert_array_equal(line.get_ydata(), frame.raw_feasibility_rate)
    ax.set_ylim(-0.035, 1.055)
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1"])
    ax.set_ylabel("Proportion feasible")
    ax.legend(frameon=False, loc="center right", fontsize=13, labelspacing=0.4)
    sample_axis(ax, next(iter(summaries.values())))
    assert not ax.containers  # No feasibility intervals or reference curve.


def main():
    summaries = {}
    for k in (2, 4, 16):
        summaries[f"Design 1, $K={k}$"] = pd.read_csv(
            ROOT / f"small_sample_extension/regular_K{k}/combined_summary.csv")
    summaries["Design 2, $K=4$"] = pd.read_csv(
        ROOT / "small_sample_extension/boundary_K4/combined_summary.csv")
    design1 = summaries["Design 1, $K=4$"]
    design2 = summaries["Design 2, $K=4$"]
    bounds = json.loads((ROOT / "support_size_study/population_bounds.json").read_text())["4"]
    for frame in summaries.values():
        assert frame.n.tolist() == [250, 1000, 4000, 16000, 64000, 256000, 1024000]
        assert (frame.repetitions == 1500).all()
        assert (frame.success_rate == 1).all()

    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    analysis.configure_plot_style()
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 14.5,
                         "axes.labelsize": 14, "xtick.labelsize": 12.5,
                         "ytick.labelsize": 12.5})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    intervals(axes[0], design1, bounds, "(a) Intervals: Design 1, $K=4$")
    errors(axes[1], design2)
    fig.tight_layout(w_pad=1.3)
    analysis.save_figure(fig, HERE / "figures", "endpoint_estimation")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    feasibility(ax, summaries)
    fig.tight_layout()
    analysis.save_figure(fig, HERE / "figures", "empirical_feasibility")
    plt.close(fig)

    # A larger standalone copy makes the requested comparison easy to inspect.
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    intervals(ax, design1, bounds, "Oracle and mean estimated intervals: Design 1, $K=4$")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#D8D8D8", label="Oracle interval"),
                       Line2D([0], [0], color=BLUE, marker="o", linewidth=2.7,
                              label="Mean estimated interval")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=12)
    fig.tight_layout()
    analysis.save_figure(fig, HERE / "figures", "oracle_estimated_intervals_K4")
    plt.close(fig)
    print("Built manuscript and standalone figures; all plotted values match archived results.")


if __name__ == "__main__":
    main()
