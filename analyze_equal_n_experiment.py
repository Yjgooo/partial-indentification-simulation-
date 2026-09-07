#!/usr/bin/env python3
"""Analyze and plot the equal-n identification-interval experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def rmse(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def summarize(replications: pd.DataFrame) -> pd.DataFrame:
    successful = replications.loc[replications.success].copy()
    rows = []
    for n, cell in successful.groupby("n", sort=True):
        rows.append(
            {
                "n": int(n),
                "repetitions": len(cell),
                "success_rate": float(
                    replications.loc[replications.n == n, "success"].mean()
                ),
                "coherent_rate": float(cell.coherent.mean()),
                "raw_feasibility_rate": float(cell.raw_feasible.mean()),
                "mean_lower": float(cell.lower_hat.mean()),
                "lower_q05": float(cell.lower_hat.quantile(0.05)),
                "lower_q50": float(cell.lower_hat.quantile(0.50)),
                "lower_q95": float(cell.lower_hat.quantile(0.95)),
                "mean_upper": float(cell.upper_hat.mean()),
                "upper_q05": float(cell.upper_hat.quantile(0.05)),
                "upper_q50": float(cell.upper_hat.quantile(0.50)),
                "upper_q95": float(cell.upper_hat.quantile(0.95)),
                "mean_width": float(cell.width_hat.mean()),
                "lower_bias": float(cell.lower_error.mean()),
                "upper_bias": float(cell.upper_error.mean()),
                "lower_rmse": rmse(cell.lower_error),
                "upper_rmse": rmse(cell.upper_error),
                "joint_q95": float(cell.joint_abs_error.quantile(0.95)),
                "scaled_lower_rmse": np.sqrt(n) * rmse(cell.lower_error),
                "scaled_upper_rmse": np.sqrt(n) * rmse(cell.upper_error),
                "scaled_joint_q95": float(cell.scaled_joint_abs_error.quantile(0.95)),
                "median_runtime_seconds": float(cell.runtime_seconds.median()),
                "max_lower_primary_gap": float(cell.lower_primary_gap.abs().max()),
                "max_upper_primary_gap": float(cell.upper_primary_gap.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def decay_exponent(n: np.ndarray, metric: np.ndarray) -> float:
    return float(-np.polyfit(np.log(n), np.log(metric), 1)[0])


def bootstrap_slopes(
    replications: pd.DataFrame, draws: int = 1_000, seed: int = 2026090702
) -> pd.DataFrame:
    cells = {
        int(n): cell.reset_index(drop=True)
        for n, cell in replications.loc[replications.success].groupby("n")
    }
    sizes = np.array(sorted(cells), dtype=float)
    rng = np.random.default_rng(seed)
    specifications = {
        "lower RMSE": lambda frame: rmse(frame.lower_error),
        "upper RMSE": lambda frame: rmse(frame.upper_error),
        "joint 95th-percentile error": lambda frame: float(
            frame.joint_abs_error.quantile(0.95)
        ),
    }
    rows = []
    for label, statistic in specifications.items():
        observed = np.array([statistic(cells[int(n)]) for n in sizes])
        boot = np.empty(draws)
        for draw in range(draws):
            sampled_metrics = []
            for n in sizes:
                cell = cells[int(n)]
                indices = rng.integers(0, len(cell), size=len(cell))
                sampled_metrics.append(statistic(cell.iloc[indices]))
            boot[draw] = decay_exponent(sizes, np.asarray(sampled_metrics))
        rows.append(
            {
                "metric": label,
                "sample_sizes": "all five",
                "decay_exponent": decay_exponent(sizes, observed),
                "bootstrap_ci_lower": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(boot, 0.975)),
                "bootstrap_draws": draws,
            }
        )

        large_sizes = sizes[-3:]
        observed_large = observed[-3:]
        boot_large = np.empty(draws)
        for draw in range(draws):
            sampled_metrics = []
            for n in large_sizes:
                cell = cells[int(n)]
                indices = rng.integers(0, len(cell), size=len(cell))
                sampled_metrics.append(statistic(cell.iloc[indices]))
            boot_large[draw] = decay_exponent(
                large_sizes, np.asarray(sampled_metrics)
            )
        rows.append(
            {
                "metric": label,
                "sample_sizes": "largest three",
                "decay_exponent": decay_exponent(large_sizes, observed_large),
                "bootstrap_ci_lower": float(np.quantile(boot_large, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(boot_large, 0.975)),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(
    summary: pd.DataFrame, oracle: pd.Series, output: Path
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13.0,
            "axes.titlesize": 14.0,
            "axes.labelsize": 13.5,
            "legend.fontsize": 11.0,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.8),
        gridspec_kw={"width_ratios": [1.42, 1.0]},
    )

    labels = [
        "Population bounds without auxiliary data",
        "Population bounds with auxiliary data",
        *[f"Estimated bounds, n = N = {int(n):,}" for n in summary.n],
    ]
    positions = np.arange(len(labels))[::-1]
    noaux_y, aux_y = positions[0], positions[1]
    estimate_y = positions[2:]

    left.hlines(
        noaux_y,
        oracle.noaux_lower,
        oracle.noaux_upper,
        color="#777777",
        linewidth=6,
        zorder=2,
    )
    left.scatter(
        [oracle.noaux_lower, oracle.noaux_upper],
        [noaux_y, noaux_y],
        color="#555555",
        s=35,
        zorder=3,
    )
    left.hlines(
        aux_y,
        oracle.aux_lower,
        oracle.aux_upper,
        color="#111111",
        linewidth=6,
        zorder=2,
    )
    left.scatter(
        [oracle.aux_lower, oracle.aux_upper],
        [aux_y, aux_y],
        color="#111111",
        s=35,
        zorder=3,
    )

    color = "#2C6EAD"
    for y, row in zip(estimate_y, summary.itertuples(index=False)):
        left.hlines(
            y,
            row.mean_lower,
            row.mean_upper,
            color=color,
            linewidth=5,
            alpha=0.80,
            zorder=2,
        )
        left.errorbar(
            row.mean_lower,
            y,
            xerr=np.array(
                [[row.mean_lower - row.lower_q05], [row.lower_q95 - row.mean_lower]]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            markersize=5,
            zorder=4,
        )
        left.errorbar(
            row.mean_upper,
            y,
            xerr=np.array(
                [[row.mean_upper - row.upper_q05], [row.upper_q95 - row.mean_upper]]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            markersize=5,
            zorder=4,
        )

    left.axvline(
        oracle.true_ate,
        color="#B24745",
        linestyle=":",
        linewidth=1.8,
        zorder=1,
    )
    left.set_yticks(positions, labels)
    left.set_xlabel("ATE")
    left.set_title("(a) Identification and finite-sample estimation", loc="left")
    left.grid(axis="x", alpha=0.20)
    left.set_ylim(-0.7, positions[0] + 0.7)
    pad = 0.012
    left.set_xlim(oracle.noaux_lower - pad, oracle.noaux_upper + pad)
    legend_handles = [
        Line2D([0], [0], color=color, linewidth=5, marker="o", label="Monte Carlo mean estimated bounds"),
        Line2D([0], [0], color=color, linewidth=1.2, marker="o", label="5th–95th endpoint percentiles"),
        Line2D([0], [0], color="#B24745", linestyle=":", linewidth=1.8, label="True ATE"),
    ]
    left.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
        ncol=3,
    )

    right.plot(
        summary.n,
        summary.scaled_lower_rmse,
        marker="o",
        linewidth=2,
        label=r"$\sqrt{n}$ lower RMSE",
    )
    right.plot(
        summary.n,
        summary.scaled_upper_rmse,
        marker="s",
        linewidth=2,
        label=r"$\sqrt{n}$ upper RMSE",
    )
    right.plot(
        summary.n,
        summary.scaled_joint_q95,
        marker="^",
        linewidth=2,
        label=r"95th percentile of $\sqrt{n}d_H$",
    )
    right.set_xscale("log", base=4)
    right.set_xticks(summary.n, [f"{int(n):,}" for n in summary.n], rotation=28)
    right.set_xlabel(r"Sample size per source, $n=N$")
    right.set_ylabel("Root-n-scaled error")
    right.set_title("(b) Root-n diagnostic", loc="left")
    right.grid(alpha=0.20)
    right.legend(frameon=False, loc="best")

    fig.suptitle(
        "Population and estimated ATE bounds",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output / "identification_interval_comparison.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / "identification_interval_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    replications = pd.read_csv(args.input / "replications.csv.gz")
    oracle = pd.read_csv(args.input / "oracle_bounds.csv").iloc[0]
    exactness = pd.read_csv(args.input / "population_penalty_exactness.csv")
    certificates_path = args.input / "oracle_certificates.csv"
    certificates = (
        pd.read_csv(certificates_path)
        if certificates_path.exists() and certificates_path.stat().st_size > 1
        else pd.DataFrame()
    )

    summary = summarize(replications)
    slopes = bootstrap_slopes(replications)
    summary.to_csv(args.output / "monte_carlo_summary.csv", index=False)
    slopes.to_csv(args.output / "rate_slopes.csv", index=False)
    plot_comparison(summary, oracle, args.output)

    checks = {
        "expected_rows": int(len(summary) * summary.repetitions.iloc[0]),
        "actual_rows": int(len(replications)),
        "all_penalty_solves_successful": bool(replications.success.all()),
        "all_estimated_intervals_coherent": bool(
            replications.loc[replications.success, "coherent"].all()
        ),
        "population_penalty_exact_at_all_n": bool(exactness.success.all()),
        "oracle_nested": bool(
            oracle.noaux_lower <= oracle.aux_lower
            and oracle.aux_upper <= oracle.noaux_upper
        ),
        "true_ate_in_aux_oracle": bool(
            oracle.aux_lower <= oracle.true_ate <= oracle.aux_upper
        ),
        "max_profile_shgo_difference": float(
            max(
                abs(oracle.lower_crosscheck_difference),
                abs(oracle.upper_crosscheck_difference),
            )
        ),
        "max_population_exactness_error": float(exactness.max_abs_error.max()),
        "max_numerical_primary_gap": float(
            max(
                replications.lower_primary_gap.abs().max(),
                replications.upper_primary_gap.abs().max(),
            )
        ),
        "certificate_statuses": (
            certificates.status.tolist() if not certificates.empty else []
        ),
        "max_certificate_gap": (
            float(certificates.optimality_gap.max())
            if not certificates.empty
            else None
        ),
        "oracle_certificates_pass": bool(
            not certificates.empty
            and certificates.status.eq("certified").all()
            and certificates.optimality_gap.le(2.0e-6 + 1.0e-12).all()
        ),
    }
    (args.output / "integrity_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    if not all(
        [
            checks["actual_rows"] == checks["expected_rows"],
            checks["all_penalty_solves_successful"],
            checks["all_estimated_intervals_coherent"],
            checks["population_penalty_exact_at_all_n"],
            checks["oracle_nested"],
            checks["true_ate_in_aux_oracle"],
            checks["oracle_certificates_pass"],
        ]
    ):
        raise RuntimeError("One or more integrity checks failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "final_analysis")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
