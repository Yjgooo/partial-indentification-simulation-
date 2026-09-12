#!/usr/bin/env python3
"""Monte Carlo study of penalization under active-boundary contamination.

The latent clean law is the smooth K=4 DGP from the support-size study. In
every (A,Y) stratum, rho probability mass is moved out of category 3 of X and
distributed equally across the other categories when generating X_tilde.
Consequently every true clean conditional distribution is exactly rho away
from its observed center, and both auxiliary mixture laws lie on exposed faces
of the feasible TV region. Tangential simplex directions keep the population
identified set nondegenerate.

The primary sample observes (X_tilde,A,Y), the independent auxiliary sample
observes (X,A), and both samples have size n.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "vendor"))

import finite_support_engine as engine  # noqa: E402


K = 4
RHO = 0.10
SHIFTED_CATEGORY = 2  # Zero-based index: the third category.
SAMPLE_SIZES = (4_000, 16_000, 64_000, 256_000, 1_024_000)
BASE_SEED = 3_026_090_711
SLACK_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BoundaryDGP:
    """Clean K=4 DGP plus the reported-confounder distribution."""

    clean: engine.SmoothDGP
    tilde_q: np.ndarray
    main: np.ndarray
    shift_direction: np.ndarray
    tv_distances: np.ndarray


def make_boundary_dgp(rho: float = RHO) -> BoundaryDGP:
    """Construct the exposed-boundary contamination mechanism.

    With category index j=2, the shift from clean q to observed tilde-q is

        d_j = -rho,  d_k = rho/(K-1) for k != j.

    This direction sums to zero and has L1 norm 2*rho, so the TV distance is
    exactly rho. Category 3 is chosen because it has the largest minimum clean
    probability across the four (A,Y) strata, avoiding sparse primary cells.
    """

    clean = engine.make_smooth_dgp(K)
    direction = np.full(K, float(rho) / (K - 1), dtype=float)
    direction[SHIFTED_CATEGORY] = -float(rho)
    tilde_q = clean.q + direction[None, None, :]
    if np.any((tilde_q <= 0.0) | (tilde_q >= 1.0)):
        raise ValueError("contamination shift moves a probability outside (0,1)")
    np.testing.assert_allclose(tilde_q.sum(axis=2), 1.0, atol=1e-14)
    tv = 0.5 * np.abs(clean.q - tilde_q).sum(axis=2)
    np.testing.assert_allclose(tv, rho, atol=1e-14)
    main = engine.joint_main_probability(clean.alpha, clean.p, tilde_q)
    return BoundaryDGP(clean, tilde_q, main, direction, tv)


def exposed_face_margins(
    p: np.ndarray,
    g: np.ndarray,
    tilde_q: np.ndarray,
    rho: float = RHO,
) -> dict[str, float]:
    """Distance from each arm's mixture to the exposed upper face.

    For any q within TV rho of tilde-q, q_j <= tilde-q_j+rho. Hence the
    auxiliary mixture coordinate g[a,j] cannot exceed the corresponding
    mixture of tilde-q_j+rho. The constructed population attains equality in
    both arms. A negative empirical margin certifies raw-LP infeasibility.
    """

    j = SHIFTED_CATEGORY
    upper = np.minimum(1.0, tilde_q[:, :, j] + rho)
    maximum_mixture = p * upper[:, 1] + (1.0 - p) * upper[:, 0]
    margins = maximum_mixture - g[:, j]
    return {
        "control_exposed_margin": float(margins[0]),
        "treated_exposed_margin": float(margins[1]),
        "minimum_exposed_margin": float(margins.min()),
    }


def compute_oracle() -> pd.DataFrame:
    dgp = make_boundary_dgp()
    clean = dgp.clean
    result = engine.direct_aux_pair(clean.alpha, clean.p, clean.g, dgp.tilde_q, RHO)
    if not result.feasible:
        raise RuntimeError("population auxiliary LP is infeasible")
    margins = exposed_face_margins(clean.p, clean.g, dgp.tilde_q)
    if max(abs(margins["control_exposed_margin"]), abs(margins["treated_exposed_margin"])) > 1e-12:
        raise RuntimeError("constructed population is not on both exposed faces")
    if not result.lower <= clean.true_ate <= result.upper:
        raise RuntimeError("true ATE is outside the population identified set")

    row: dict[str, float | int | str] = {
        "dgp_id": "k4_all_strata_tv_boundary",
        "dgp_label": "K=4 active TV boundary in all (A,Y) strata",
        "K": K,
        "rho": RHO,
        "shifted_category_one_based": SHIFTED_CATEGORY + 1,
        "alpha0": clean.alpha[0],
        "alpha1": clean.alpha[1],
        "p_y1_a0": clean.p[0],
        "p_y1_a1": clean.p[1],
        "true_ate": clean.true_ate,
        "oracle_lower": result.lower,
        "oracle_upper": result.upper,
        "oracle_width": result.upper - result.lower,
        "minimum_tilde_q_cell": float(dgp.tilde_q.min()),
        "minimum_main_joint_cell": float(dgp.main.min()),
        **margins,
    }
    for k in range(K):
        row[f"shift_direction_x{k + 1}"] = dgp.shift_direction[k]
        row[f"p_x{k + 1}"] = clean.px[k]
        row[f"propensity_x{k + 1}"] = clean.propensity[k]
        row[f"outcome_a0_x{k + 1}"] = clean.outcome[0, k]
        row[f"outcome_a1_x{k + 1}"] = clean.outcome[1, k]
        row[f"p_x{k + 1}_a0"] = clean.g[0, k]
        row[f"p_x{k + 1}_a1"] = clean.g[1, k]
    for a in range(2):
        for y in range(2):
            row[f"tv_a{a}_y{y}"] = dgp.tv_distances[a, y]
            for k in range(K):
                row[f"clean_q_a{a}_y{y}_x{k + 1}"] = clean.q[a, y, k]
                row[f"tilde_q_a{a}_y{y}_x{k + 1}"] = dgp.tilde_q[a, y, k]
    return pd.DataFrame([row])


def population_exactness(n: int, oracle: dict[str, float]) -> dict[str, object]:
    dgp = make_boundary_dgp()
    lower, neg_upper, w_n = engine.penalty_pair(
        dgp.clean.alpha,
        dgp.clean.p,
        dgp.clean.g,
        dgp.tilde_q,
        RHO,
        n,
        n,
    )
    lower_error = lower.selected_value - oracle["oracle_lower"]
    upper_error = -neg_upper.selected_value - oracle["oracle_upper"]
    max_error = max(abs(lower_error), abs(upper_error))
    return {
        "n": n,
        "penalty_level": w_n,
        "penalty_lower": lower.selected_value,
        "penalty_upper": -neg_upper.selected_value,
        "oracle_lower": oracle["oracle_lower"],
        "oracle_upper": oracle["oracle_upper"],
        "lower_total_slack": lower.total_slack,
        "upper_total_slack": neg_upper.total_slack,
        "max_abs_error": max_error,
        "success": bool(lower.success and neg_upper.success and max_error <= 1e-7),
    }


def simulate_one(payload: tuple[int, int, int, dict[str, float]]) -> dict[str, object]:
    n, rep, seed, oracle = payload
    started = time.perf_counter()
    dgp = make_boundary_dgp()
    sequence = np.random.SeedSequence(seed)
    main_sequence, auxiliary_sequence = sequence.spawn(2)
    main_rng = np.random.default_rng(main_sequence)
    auxiliary_rng = np.random.default_rng(auxiliary_sequence)
    base = {
        "n": n,
        "n_main": n,
        "n_aux": n,
        "rep": rep,
        "seed": seed,
        "K": K,
        "rho": RHO,
        "oracle_lower": oracle["oracle_lower"],
        "oracle_upper": oracle["oracle_upper"],
        "true_ate": oracle["true_ate"],
    }
    try:
        main_counts = main_rng.multinomial(n, dgp.main.ravel()).reshape(2, 2, K)
        auxiliary_counts = auxiliary_rng.multinomial(n, dgp.clean.aux.ravel()).reshape(2, K)
        alpha, p, tilde_q = engine.estimate_main(main_counts)
        g_hat = engine.estimate_aux(auxiliary_counts)
        margins = exposed_face_margins(p, g_hat, tilde_q)
        lower, neg_upper, w_n = engine.penalty_pair(
            alpha, p, g_hat, tilde_q, RHO, n, n
        )
        raw = engine.direct_aux_pair(alpha, p, g_hat, tilde_q, RHO)
        if not (lower.success and neg_upper.success):
            return {
                **base,
                "success": False,
                "error": "penalty_lp_failure",
                "raw_feasible": raw.feasible,
                **margins,
                "runtime_seconds": time.perf_counter() - started,
            }

        lower_hat = lower.selected_value
        upper_hat = -neg_upper.selected_value
        lower_error = lower_hat - oracle["oracle_lower"]
        upper_error = upper_hat - oracle["oracle_upper"]
        joint_abs_error = max(abs(lower_error), abs(upper_error))
        max_total_slack = max(lower.total_slack, neg_upper.total_slack)
        return {
            **base,
            "success": True,
            "error": "",
            "penalty_level": w_n,
            "lower_hat": lower_hat,
            "upper_hat": upper_hat,
            "width_hat": upper_hat - lower_hat,
            "lower_error": lower_error,
            "upper_error": upper_error,
            "joint_abs_error": joint_abs_error,
            "scaled_lower_error": math.sqrt(n) * lower_error,
            "scaled_upper_error": math.sqrt(n) * upper_error,
            "scaled_joint_abs_error": math.sqrt(n) * joint_abs_error,
            "coherent": bool(lower_hat <= upper_hat + 1e-10),
            "raw_feasible": raw.feasible,
            "raw_lower": raw.lower,
            "raw_upper": raw.upper,
            "raw_lower_status": raw.lower_status,
            "raw_upper_status": raw.upper_status,
            "penalty_activated": bool(max_total_slack > SLACK_TOLERANCE),
            "lower_total_slack": lower.total_slack,
            "upper_total_slack": neg_upper.total_slack,
            "mean_endpoint_slack": 0.5 * (lower.total_slack + neg_upper.total_slack),
            "max_total_slack": max_total_slack,
            "lower_max_slack": lower.max_slack,
            "upper_max_slack": neg_upper.max_slack,
            "lower_primary_gap": lower.primary_gap,
            "upper_primary_gap": neg_upper.primary_gap,
            "lower_original_violation": lower.selected_original_violation,
            "upper_original_violation": neg_upper.selected_original_violation,
            **margins,
            "runtime_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            **base,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_feasible": False,
            "runtime_seconds": time.perf_counter() - started,
        }


def parse_sample_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size < 16 for size in sizes):
        raise argparse.ArgumentTypeError("sample sizes must be comma-separated integers >=16")
    return sizes


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    oracle_frame = compute_oracle()
    oracle_frame.to_csv(args.output / "oracle_bounds.csv", index=False)
    oracle = oracle_frame.iloc[0].to_dict()

    exactness = pd.DataFrame([population_exactness(n, oracle) for n in args.sample_sizes])
    exactness.to_csv(args.output / "population_penalty_exactness.csv", index=False)
    if not exactness.success.all():
        raise RuntimeError("population penalty exactness gate failed")

    all_rows: list[dict[str, object]] = []
    started = time.time()
    for cell_index, n in enumerate(args.sample_sizes):
        payloads = [
            (n, rep, args.base_seed + cell_index * 10_000_000 + rep * 7_919, oracle)
            for rep in range(args.repetitions)
        ]
        cell_started = time.time()
        if args.workers == 1:
            rows = [simulate_one(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(simulate_one, payloads, chunksize=4))
        frame = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
        frame.to_csv(
            args.output / f"replications_n{n}.csv.gz",
            index=False,
            compression="gzip",
        )
        all_rows.extend(rows)
        raw_rate = float(frame.loc[frame.success, "raw_feasible"].mean())
        print(
            f"n={n}: {int(frame.success.sum())}/{len(frame)} successful; "
            f"raw feasible={raw_rate:.3f}; elapsed={time.time() - cell_started:.1f}s",
            flush=True,
        )

    replications = pd.DataFrame(all_rows).sort_values(["n", "rep"]).reset_index(drop=True)
    replications.to_csv(args.output / "replications.csv.gz", index=False, compression="gzip")
    metadata = {
        "dgp": "K=4 active TV boundary in all (A,Y) strata",
        "K": K,
        "rho_ay": RHO,
        "shifted_category_one_based": SHIFTED_CATEGORY + 1,
        "shift_direction": make_boundary_dgp().shift_direction.tolist(),
        "sample_sizes_per_source": list(args.sample_sizes),
        "independent_sources": True,
        "total_observations_per_replication": "2n",
        "repetitions_per_sample_size": args.repetitions,
        "total_replications": len(replications),
        "base_seed": args.base_seed,
        "workers": args.workers,
        "penalty_rule": "w_n=5*log(log(min(n,N)))/log(log(30000))",
        "smoothing": "Jeffreys 0.5 cell pseudocount",
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--repetitions", type=int, default=1_500)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--sample-sizes",
        type=parse_sample_sizes,
        default=SAMPLE_SIZES,
        help="comma-separated sizes per source",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
