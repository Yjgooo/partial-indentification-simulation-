#!/usr/bin/env python3
"""Equal-sample comparison of two population bounds and penalized estimates.

The experiment uses the regular binary-X DGP from the audited auxiliary-value
study.  The main sample observes (X_tilde, A, Y), the independent auxiliary
sample observes (X, A), and both samples have size n.  The population
no-auxiliary oracle differs from the auxiliary oracle only by deleting the
mixture restrictions that identify P(X|A).
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
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import aux_value_engine as engine  # noqa: E402
from global_k2 import certify_interval, solve_noaux_shgo  # noqa: E402


SAMPLE_SIZES = (4_000, 16_000, 64_000, 256_000, 1_024_000)
RHO = 0.10
BASE_SEED = 2026090711


def compute_oracle(certificate_tol: float, skip_certificate: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    dgp = engine.reference_dgp()
    aux = engine.solve_aux_binary(
        dgp.alpha1,
        dgp.p0,
        dgp.p1,
        dgp.centers,
        RHO,
        dgp.auxiliary_r,
    )
    profile_lower, profile_upper = engine.profile_pair(
        dgp.alpha1,
        dgp.p0,
        dgp.p1,
        dgp.centers,
        RHO,
        n=4096,
        iters=3,
    )
    independent = solve_noaux_shgo(
        dgp.alpha1,
        dgp.p0,
        dgp.p1,
        dgp.centers,
        RHO,
        n=1200,
        iters=5,
    )
    noaux_lower = min(profile_lower.value, independent.lower.value)
    noaux_upper = max(profile_upper.value, independent.upper.value)
    if not (
        noaux_lower <= aux.lower.value <= aux.upper.value <= noaux_upper
        and aux.lower.value <= dgp.true_ate <= aux.upper.value
    ):
        raise RuntimeError("Oracle nesting or true-ATE containment failed")

    oracle = pd.DataFrame(
        [
            {
                "dgp_id": dgp.dgp_id,
                "dgp_label": dgp.label,
                "rho": RHO,
                "true_ate": dgp.true_ate,
                "aux_lower": aux.lower.value,
                "aux_upper": aux.upper.value,
                "aux_width": aux.upper.value - aux.lower.value,
                "noaux_lower": noaux_lower,
                "noaux_upper": noaux_upper,
                "noaux_width": noaux_upper - noaux_lower,
                "width_reduction": (noaux_upper - noaux_lower)
                - (aux.upper.value - aux.lower.value),
                "width_ratio": (noaux_upper - noaux_lower)
                / (aux.upper.value - aux.lower.value),
                "profile_lower": profile_lower.value,
                "profile_upper": profile_upper.value,
                "independent_lower": independent.lower.value,
                "independent_upper": independent.upper.value,
                "lower_crosscheck_difference": profile_lower.value
                - independent.lower.value,
                "upper_crosscheck_difference": profile_upper.value
                - independent.upper.value,
            }
        ]
    )

    certificate_rows: list[dict[str, object]] = []
    if not skip_certificate:
        for endpoint, incumbent in (
            ("lower", independent.lower.x),
            ("upper", independent.upper.x),
        ):
            certificate = certify_interval(
                dgp.alpha1,
                dgp.p0,
                dgp.p1,
                dgp.centers,
                RHO,
                endpoint=endpoint,
                abs_tol=certificate_tol,
                max_boxes=2_000_000,
                incumbent_x=incumbent,
            )
            certificate_rows.append(
                {
                    "endpoint": endpoint,
                    "reported_value": noaux_lower if endpoint == "lower" else noaux_upper,
                    "certificate_lower": certificate.certified_lower,
                    "certificate_upper": certificate.certified_upper,
                    "optimality_gap": certificate.optimality_gap,
                    "boxes_created": certificate.boxes_created,
                    "boxes_processed": certificate.boxes_processed,
                    "active_boxes": certificate.active_boxes,
                    "status": certificate.status,
                }
            )
    return oracle, pd.DataFrame(certificate_rows)


def population_exactness(n: int, oracle: dict[str, float]) -> dict[str, object]:
    dgp = engine.reference_dgp()
    true_g = np.array(
        [
            [1.0 - dgp.auxiliary_r[0], dgp.auxiliary_r[0]],
            [1.0 - dgp.auxiliary_r[1], dgp.auxiliary_r[1]],
        ]
    )
    lower, neg_upper, w_n = engine.penalty_pair(
        dgp.alpha1,
        dgp.p0,
        dgp.p1,
        np.asarray(dgp.centers),
        true_g,
        RHO,
        n,
        n,
    )
    lower_error = lower.selected_value - oracle["aux_lower"]
    upper_error = -neg_upper.selected_value - oracle["aux_upper"]
    return {
        "n": n,
        "penalty_level": w_n,
        "penalty_lower": lower.selected_value,
        "penalty_upper": -neg_upper.selected_value,
        "oracle_lower": oracle["aux_lower"],
        "oracle_upper": oracle["aux_upper"],
        "lower_error": lower_error,
        "upper_error": upper_error,
        "max_abs_error": max(abs(lower_error), abs(upper_error)),
        "success": bool(
            lower.success
            and neg_upper.success
            and max(abs(lower_error), abs(upper_error)) <= 1e-7
        ),
    }


def simulate_one(payload: tuple[int, int, int, dict[str, float]]) -> dict[str, object]:
    n, rep, seed, oracle = payload
    started = time.perf_counter()
    dgp = engine.reference_dgp()
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
        "rho": RHO,
        "oracle_lower": oracle["aux_lower"],
        "oracle_upper": oracle["aux_upper"],
        "noaux_lower": oracle["noaux_lower"],
        "noaux_upper": oracle["noaux_upper"],
        "true_ate": oracle["true_ate"],
    }
    try:
        main_counts = main_rng.multinomial(
            n, engine.binary_main_probability(dgp).ravel()
        ).reshape(2, 2, 2)
        auxiliary_counts = auxiliary_rng.multinomial(
            n, engine.binary_aux_probability(dgp).ravel()
        ).reshape(2, 2)
        alpha1, p0, p1, centers, _ = engine.estimate_main(main_counts)
        g_hat = engine.estimate_aux_g(auxiliary_counts)
        lower, neg_upper, w_n = engine.penalty_pair(
            alpha1,
            p0,
            p1,
            centers,
            g_hat,
            RHO,
            n,
            n,
        )
        raw_lower, raw_upper, raw_feasible = engine.raw_aux_pair(
            alpha1, p0, p1, centers, g_hat, RHO
        )
        success = bool(lower.success and neg_upper.success)
        if not success:
            return {
                **base,
                "success": False,
                "error": "penalty_lp_failure",
                "raw_feasible": raw_feasible,
                "runtime_seconds": time.perf_counter() - started,
            }
        lower_hat = lower.selected_value
        upper_hat = -neg_upper.selected_value
        lower_error = lower_hat - oracle["aux_lower"]
        upper_error = upper_hat - oracle["aux_upper"]
        joint_abs_error = max(abs(lower_error), abs(upper_error))
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
            "raw_feasible": raw_feasible,
            "raw_lower": raw_lower,
            "raw_upper": raw_upper,
            "lower_total_slack": lower.total_slack,
            "upper_total_slack": neg_upper.total_slack,
            "lower_primary_gap": lower.primary_gap,
            "upper_primary_gap": neg_upper.primary_gap,
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


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    oracle_frame, certificates = compute_oracle(
        args.certificate_tol, args.skip_certificate
    )
    oracle_frame.to_csv(args.output / "oracle_bounds.csv", index=False)
    certificates.to_csv(args.output / "oracle_certificates.csv", index=False)
    oracle = oracle_frame.iloc[0].to_dict()

    exactness = pd.DataFrame(
        [population_exactness(n, oracle) for n in SAMPLE_SIZES]
    )
    exactness.to_csv(args.output / "population_penalty_exactness.csv", index=False)
    if not exactness.success.all():
        raise RuntimeError("Population penalty exactness gate failed")

    all_rows: list[dict[str, object]] = []
    started = time.time()
    for cell_index, n in enumerate(SAMPLE_SIZES):
        payloads = [
            (
                n,
                rep,
                args.base_seed + cell_index * 10_000_000 + rep * 7_919,
                oracle,
            )
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
        print(
            f"n={n}: {int(frame.success.sum())}/{len(frame)} successful; "
            f"elapsed={time.time() - cell_started:.1f}s",
            flush=True,
        )

    replications = (
        pd.DataFrame(all_rows)
        .sort_values(["n", "rep"])
        .reset_index(drop=True)
    )
    replications.to_csv(
        args.output / "replications.csv.gz", index=False, compression="gzip"
    )
    metadata = {
        "dgp": "regular binary-X reference",
        "rho_ay": RHO,
        "sample_sizes_per_source": list(SAMPLE_SIZES),
        "independent_sources": True,
        "total_observations_per_replication": "2n",
        "repetitions_per_sample_size": args.repetitions,
        "total_replications": len(replications),
        "base_seed": args.base_seed,
        "workers": args.workers,
        "penalty_rule": "w_n=5*log(log(n))/log(log(30000))",
        "smoothing": "Jeffreys 0.5 cell pseudocount",
        "certificate_tolerance": None if args.skip_certificate else args.certificate_tol,
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
    parser.add_argument("--certificate-tol", type=float, default=2e-6)
    parser.add_argument("--skip-certificate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
