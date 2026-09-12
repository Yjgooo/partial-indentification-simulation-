#!/usr/bin/env python3
"""Summarize and plot the active-boundary contamination experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def rmse(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / trials + z**2 / (4.0 * trials**2)) / denominator
    return float(center - half_width), float(center + half_width)


def summarize(replications: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for n, all_cell in replications.groupby("n", sort=True):
        cell = all_cell.loc[all_cell.success].copy()
        feasible_count = int(cell.raw_feasible.sum())
        feasible_lo, feasible_hi = wilson_interval(feasible_count, len(cell))
        activated = cell.loc[cell.penalty_activated, "mean_endpoint_slack"]
        raw_differences = pd.concat(
            [
                (cell.loc[cell.raw_feasible, "lower_hat"] - cell.loc[cell.raw_feasible, "raw_lower"]).abs(),
                (cell.loc[cell.raw_feasible, "upper_hat"] - cell.loc[cell.raw_feasible, "raw_upper"]).abs(),
            ],
            ignore_index=True,
        )
        rows.append(
            {
                "n": int(n),
                "repetitions": len(cell),
                "success_rate": float(all_cell.success.mean()),
                "coherent_rate": float(cell.coherent.mean()),
                "raw_feasibility_rate": float(cell.raw_feasible.mean()),
                "raw_feasibility_ci_lower": feasible_lo,
                "raw_feasibility_ci_upper": feasible_hi,
                "penalty_activation_rate": float(cell.penalty_activated.mean()),
                "mean_lower": float(cell.lower_hat.mean()),
                "mean_upper": float(cell.upper_hat.mean()),
                "mean_width": float(cell.width_hat.mean()),
                "lower_bias": float(cell.lower_error.mean()),
                "upper_bias": float(cell.upper_error.mean()),
                "lower_rmse": rmse(cell.lower_error),
                "upper_rmse": rmse(cell.upper_error),
                "joint_q95": float(cell.joint_abs_error.quantile(0.95)),
                "scaled_lower_rmse": np.sqrt(n) * rmse(cell.lower_error),
                "scaled_upper_rmse": np.sqrt(n) * rmse(cell.upper_error),
                "scaled_joint_q95": float(cell.scaled_joint_abs_error.quantile(0.95)),
                "mean_endpoint_slack": float(cell.mean_endpoint_slack.mean()),
                "mean_slack_when_activated": float(activated.mean()),
                "scaled_mean_endpoint_slack": float(np.sqrt(n) * cell.mean_endpoint_slack.mean()),
                "scaled_mean_slack_when_activated": float(np.sqrt(n) * activated.mean()),
                "mean_control_exposed_margin": float(cell.control_exposed_margin.mean()),
                "mean_treated_exposed_margin": float(cell.treated_exposed_margin.mean()),
                "scaled_sd_control_exposed_margin": float(np.sqrt(n) * cell.control_exposed_margin.std(ddof=1)),
                "scaled_sd_treated_exposed_margin": float(np.sqrt(n) * cell.treated_exposed_margin.std(ddof=1)),
                "max_feasible_penalty_raw_difference": float(raw_differences.max()),
                "median_runtime_seconds": float(cell.runtime_seconds.median()),
                "max_lower_primary_gap": float(cell.lower_primary_gap.abs().max()),
                "max_upper_primary_gap": float(cell.upper_primary_gap.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def decay_exponent(n: np.ndarray, metric: np.ndarray) -> float:
    return float(-np.polyfit(np.log(n), np.log(metric), 1)[0])


def bootstrap_slopes(
    replications: pd.DataFrame,
    draws: int = 1_000,
    seed: int = 3_026_090_702,
) -> pd.DataFrame:
    cells = {
        int(n): cell.loc[cell.success].reset_index(drop=True)
        for n, cell in replications.groupby("n")
    }
    sizes = np.array(sorted(cells), dtype=float)
    rng = np.random.default_rng(seed)
    specifications = {
        "lower RMSE": lambda frame: rmse(frame.lower_error),
        "upper RMSE": lambda frame: rmse(frame.upper_error),
        "joint 95th-percentile error": lambda frame: float(frame.joint_abs_error.quantile(0.95)),
        "mean row-normalized total slack": lambda frame: float(
            frame.mean_endpoint_slack.mean()
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    for label, statistic in specifications.items():
        observed = np.array([statistic(cells[int(n)]) for n in sizes])
        bootstrap = np.empty(draws)
        for draw in range(draws):
            metrics = []
            for n in sizes:
                cell = cells[int(n)]
                indices = rng.integers(0, len(cell), size=len(cell))
                metrics.append(statistic(cell.iloc[indices]))
            bootstrap[draw] = decay_exponent(sizes, np.asarray(metrics))
        rows.append(
            {
                "metric": label,
                "decay_exponent": decay_exponent(sizes, observed),
                "bootstrap_ci_lower": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12.0,
            "axes.titlesize": 13.0,
            "axes.labelsize": 12.5,
            "legend.fontsize": 10.5,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_sample_size_axis(axis: plt.Axes, x: np.ndarray) -> None:
    axis.set_xscale("log", base=4)
    axis.set_xticks(x, [f"{int(n):,}" for n in x], rotation=28)
    axis.set_xlabel(r"Sample size per source, $n=N$")
    axis.grid(alpha=0.20)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    """Write each figure atomically and verify that the PNG can be decoded."""

    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    temporary_png = output / f".{stem}.tmp.png"
    temporary_pdf = output / f".{stem}.tmp.pdf"
    fig.savefig(
        temporary_png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        temporary_pdf,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    temporary_png.replace(png)
    temporary_pdf.replace(pdf)
    plt.imread(png)


def add_feasibility_panel(axis: plt.Axes, summary: pd.DataFrame) -> None:
    red = "#B24745"
    green = "#39835B"
    gray = "#555555"
    x = summary.n.to_numpy(dtype=float)
    axis.errorbar(
        x,
        summary.raw_feasibility_rate,
        yerr=np.vstack(
            [
                summary.raw_feasibility_rate - summary.raw_feasibility_ci_lower,
                summary.raw_feasibility_ci_upper - summary.raw_feasibility_rate,
            ]
        ),
        color=red,
        marker="o",
        linewidth=2,
        capsize=3,
        label="Direct plug-in LP feasible",
    )
    axis.plot(
        x,
        summary.success_rate,
        color=green,
        marker="s",
        linewidth=2,
        label="Penalized endpoints computed",
    )
    axis.axhline(
        0.25,
        color=gray,
        linestyle=":",
        linewidth=1.4,
        label="0.25 asymptotic reference",
    )
    axis.set_ylim(0.0, 1.04)
    axis.set_ylabel("Monte Carlo probability")
    axis.set_title("(a) Empirical incompatibility", loc="left")
    axis.legend(frameon=False, loc="best")
    format_sample_size_axis(axis, x)


def add_rate_panel(axis: plt.Axes, summary: pd.DataFrame) -> None:
    blue = "#2C6EAD"
    red = "#B24745"
    green = "#39835B"
    x = summary.n.to_numpy(dtype=float)
    axis.plot(
        x,
        summary.scaled_lower_rmse,
        marker="o",
        linewidth=2,
        color=blue,
        label=r"$\sqrt{n}$ lower RMSE",
    )
    axis.plot(
        x,
        summary.scaled_upper_rmse,
        marker="s",
        linewidth=2,
        color=red,
        label=r"$\sqrt{n}$ upper RMSE",
    )
    axis.plot(
        x,
        summary.scaled_joint_q95,
        marker="^",
        linewidth=2,
        color=green,
        label=r"95th percentile of $\sqrt{n}d_H$",
    )
    axis.set_ylabel("Root-n-scaled endpoint error")
    axis.set_title("(b) Root-n scaling diagnostic", loc="left")
    axis.legend(frameon=False, loc="best")
    format_sample_size_axis(axis, x)


def plot_main_diagnostics(summary: pd.DataFrame, output: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))
    add_feasibility_panel(axes[0], summary)
    add_rate_panel(axes[1], summary)
    fig.tight_layout()
    save_figure(fig, output, "boundary_penalty_main")
    plt.close(fig)


def plot_full_diagnostics(summary: pd.DataFrame, output: Path) -> None:
    configure_plot_style()
    blue = "#2C6EAD"
    red = "#B24745"
    x = summary.n.to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    add_feasibility_panel(axes[0], summary)
    add_rate_panel(axes[1], summary)
    axes[2].plot(
        x,
        summary.scaled_mean_endpoint_slack,
        marker="o",
        linewidth=2,
        color=blue,
        label="All replications",
    )
    axes[2].plot(
        x,
        summary.scaled_mean_slack_when_activated,
        marker="s",
        linewidth=2,
        color=red,
        label="Conditional on activation",
    )
    axes[2].set_ylabel(r"$\sqrt{n}$ times mean row-normalized total slack")
    axes[2].set_title("(c) Constraint repair magnitude", loc="left")
    axes[2].legend(frameon=False, loc="best")
    format_sample_size_axis(axes[2], x)
    fig.tight_layout()
    save_figure(fig, output, "boundary_penalty_diagnostics")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    replications = pd.read_csv(args.input / "replications.csv.gz")
    oracle = pd.read_csv(args.input / "oracle_bounds.csv").iloc[0]
    exactness = pd.read_csv(args.input / "population_penalty_exactness.csv")
    summary = summarize(replications)
    slopes = bootstrap_slopes(replications, draws=args.bootstrap_draws)
    summary.to_csv(args.output / "monte_carlo_summary.csv", index=False)
    slopes.to_csv(args.output / "rate_slopes.csv", index=False)
    plot_main_diagnostics(summary, args.output)
    plot_full_diagnostics(summary, args.output)

    successful = replications.loc[replications.success]
    activation_matches = successful.penalty_activated.eq(~successful.raw_feasible)
    checks = {
        "expected_rows": int(summary.repetitions.sum()),
        "actual_rows": int(len(replications)),
        "all_penalty_solves_successful": bool(replications.success.all()),
        "all_estimated_intervals_coherent": bool(successful.coherent.all()),
        "population_penalty_exact_at_all_n": bool(exactness.success.all()),
        "true_ate_in_population_interval": bool(oracle.oracle_lower <= oracle.true_ate <= oracle.oracle_upper),
        "actual_nonzero_contamination_in_all_strata": bool(
            all(oracle[f"tv_a{a}_y{y}"] > 0 for a in range(2) for y in range(2))
        ),
        "both_exposed_faces_active": bool(
            abs(oracle.control_exposed_margin) <= 1e-12
            and abs(oracle.treated_exposed_margin) <= 1e-12
        ),
        "raw_feasibility_rates_between_0.15_and_0.35": bool(
            summary.raw_feasibility_rate.between(0.15, 0.35).all()
        ),
        "penalty_activation_matches_raw_infeasibility": bool(activation_matches.all()),
        "max_feasible_penalty_raw_difference": float(summary.max_feasible_penalty_raw_difference.max()),
        "max_population_exactness_error": float(exactness.max_abs_error.max()),
        "max_numerical_primary_gap": float(max(successful.lower_primary_gap.abs().max(), successful.upper_primary_gap.abs().max())),
        "root_n_scaled_lower_rmse_range_ratio": float(summary.scaled_lower_rmse.max() / summary.scaled_lower_rmse.min()),
        "root_n_scaled_upper_rmse_range_ratio": float(summary.scaled_upper_rmse.max() / summary.scaled_upper_rmse.min()),
    }
    (args.output / "integrity_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print("\nRate diagnostics", flush=True)
    print(slopes.to_string(index=False), flush=True)
    print("\nIntegrity checks", flush=True)
    print(json.dumps(checks, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    parser.add_argument("--bootstrap-draws", type=int, default=1_000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
