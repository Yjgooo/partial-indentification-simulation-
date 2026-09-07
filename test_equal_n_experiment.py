#!/usr/bin/env python3
"""Deterministic checks for the equal-n comparison experiment."""

from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))

import run_equal_n_experiment as study  # noqa: E402
import analyze_equal_n_experiment as analysis  # noqa: E402
import aux_value_engine as engine  # noqa: E402


def test_dgp_and_oracle_values() -> None:
    dgp = engine.reference_dgp()
    np.testing.assert_allclose(dgp.true_ate, 0.08, atol=1e-14)
    oracle, _ = study.compute_oracle(2e-5, skip_certificate=True)
    row = oracle.iloc[0]
    np.testing.assert_allclose(row.aux_lower, 0.0475, atol=2e-12)
    np.testing.assert_allclose(row.aux_upper, 0.1125, atol=2e-12)
    np.testing.assert_allclose(row.noaux_lower, -0.001561242274987, atol=2e-10)
    np.testing.assert_allclose(row.noaux_upper, 0.14, atol=2e-10)
    assert row.noaux_lower <= row.aux_lower <= row.aux_upper <= row.noaux_upper


def test_dgp_probability_laws() -> None:
    dgp = engine.reference_dgp()
    main = engine.binary_main_probability(dgp)
    auxiliary = engine.binary_aux_probability(dgp)
    np.testing.assert_allclose(main.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(auxiliary.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(main.sum(axis=(1, 2)), [0.5, 0.5], atol=1e-14)
    np.testing.assert_allclose(auxiliary.sum(axis=1), [0.5, 0.5], atol=1e-14)
    np.testing.assert_allclose(
        auxiliary / auxiliary.sum(axis=1, keepdims=True),
        [[0.6, 0.4], [0.4, 0.6]],
        atol=1e-14,
    )


def test_ate_formula_against_direct_standardization() -> None:
    dgp = engine.reference_dgp()
    alpha = np.array([1.0 - dgp.alpha1, dgp.alpha1])
    p = np.array([dgp.p0, dgp.p1])
    u1, v1, u0, v0 = dgp.centers
    q = np.array(
        [
            [[1.0 - v0, v0], [1.0 - u0, u0]],
            [[1.0 - v1, v1], [1.0 - u1, u1]],
        ]
    )
    g = np.vstack([p[a] * q[a, 1] + (1.0 - p[a]) * q[a, 0] for a in range(2)])
    px = alpha @ g
    outcome_regression = np.vstack([p[a] * q[a, 1] / g[a] for a in range(2)])
    direct = float(px @ (outcome_regression[1] - outcome_regression[0]))
    np.testing.assert_allclose(
        engine.ate_binary(dgp.centers, dgp.alpha1, dgp.p0, dgp.p1),
        direct,
        atol=1e-14,
    )


def test_population_penalty_exactness() -> None:
    oracle, _ = study.compute_oracle(2e-5, skip_certificate=True)
    values = oracle.iloc[0].to_dict()
    for n in study.SAMPLE_SIZES:
        check = study.population_exactness(n, values)
        assert check["success"]
        assert check["max_abs_error"] <= 1e-7


def test_seeded_replication_is_reproducible() -> None:
    oracle, _ = study.compute_oracle(2e-5, skip_certificate=True)
    values = oracle.iloc[0].to_dict()
    first = study.simulate_one((4_000, 0, 123456, values))
    second = study.simulate_one((4_000, 0, 123456, values))
    for key in first:
        if key != "runtime_seconds":
            assert first[key] == second[key]
    assert first["success"]
    assert first["coherent"]
    assert first["n_main"] == first["n_aux"] == 4_000


def test_saved_confirmatory_outputs() -> None:
    results = HERE / "results"
    if not (results / "replications.csv.gz").exists():
        return
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(results / "replications.csv.gz").sort_values(
        ["n", "rep"]
    ).reset_index(drop=True)
    assert metadata["base_seed"] == study.BASE_SEED
    assert metadata["repetitions_per_sample_size"] == 1_500
    assert len(frame) == 7_500
    assert frame.seed.nunique() == len(frame)
    assert frame.groupby("n").size().eq(1_500).all()
    assert set(frame.n.unique()) == set(study.SAMPLE_SIZES)
    assert ((frame.n_main == frame.n) & (frame.n_aux == frame.n)).all()
    assert frame.success.all() and frame.coherent.all() and frame.raw_feasible.all()
    np.testing.assert_allclose(frame.lower_hat, frame.raw_lower, atol=1e-12)
    np.testing.assert_allclose(frame.upper_hat, frame.raw_upper, atol=1e-12)
    assert (
        frame[
            [
                "lower_total_slack",
                "upper_total_slack",
                "lower_primary_gap",
                "upper_primary_gap",
            ]
        ]
        .abs()
        .to_numpy()
        .max()
        == 0.0
    )

    cell_files = sorted(
        results.glob("replications_n*.csv.gz"),
        key=lambda path: int(
            path.name.removeprefix("replications_n").split(".", 1)[0]
        ),
    )
    cells = pd.concat([pd.read_csv(path) for path in cell_files], ignore_index=True)
    cells = cells.sort_values(["n", "rep"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame, cells, check_exact=True)

    pilot_path = HERE / "pilot_results" / "replications.csv.gz"
    if pilot_path.exists():
        pilot = pd.read_csv(pilot_path, usecols=["seed"])
        assert set(frame.seed).isdisjoint(set(pilot.seed))


def test_saved_analysis_reproduces_from_raw_data() -> None:
    results = HERE / "results" / "replications.csv.gz"
    stored_path = HERE / "final_analysis" / "monte_carlo_summary.csv"
    if not (results.exists() and stored_path.exists()):
        return
    frame = pd.read_csv(results)
    reproduced = analysis.summarize(frame)
    stored = pd.read_csv(stored_path)
    pd.testing.assert_frame_equal(reproduced, stored, check_exact=False, atol=1e-14)


def main() -> None:
    tests = [
        test_dgp_and_oracle_values,
        test_dgp_probability_laws,
        test_ate_formula_against_direct_standardization,
        test_population_penalty_exactness,
        test_seeded_replication_is_reproducible,
        test_saved_confirmatory_outputs,
        test_saved_analysis_reproduces_from_raw_data,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} tests")


if __name__ == "__main__":
    main()
