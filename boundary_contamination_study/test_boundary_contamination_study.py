#!/usr/bin/env python3
"""Deterministic checks for the active-boundary contamination study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "vendor"))

import analyze_boundary_contamination_study as analysis  # noqa: E402
import finite_support_engine as generic  # noqa: E402
import run_boundary_contamination_study as study  # noqa: E402


def test_generic_engine_matches_existing_k4_dgp() -> None:
    sys.path.insert(0, str(ROOT / "support_size_study"))
    import run_support_size_study as existing  # noqa: E402

    new = generic.make_smooth_dgp(4)
    old = existing.make_dgp(4)
    for name in (
        "px",
        "propensity",
        "outcome",
        "alpha",
        "p",
        "g",
        "q",
        "main",
        "aux",
    ):
        np.testing.assert_allclose(getattr(new, name), getattr(old, name), atol=1e-14)
    np.testing.assert_allclose(new.true_ate, old.true_ate, atol=1e-14)


def test_generic_engine_matches_binary_audited_engine() -> None:
    import aux_value_engine as binary

    dgp = generic.make_smooth_dgp(2)
    reference = binary.reference_dgp()
    direct = generic.direct_aux_pair(dgp.alpha, dgp.p, dgp.g, dgp.q, 0.1)
    audited = binary.solve_aux_binary(
        reference.alpha1,
        reference.p0,
        reference.p1,
        reference.centers,
        0.1,
        reference.auxiliary_r,
    )
    assert direct.feasible
    np.testing.assert_allclose(direct.lower, audited.lower.value, atol=2e-12)
    np.testing.assert_allclose(direct.upper, audited.upper.value, atol=2e-12)


def test_dgp_probability_laws_and_tv_distances() -> None:
    dgp = study.make_boundary_dgp()
    np.testing.assert_allclose(dgp.main.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(dgp.clean.aux.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(dgp.main.sum(axis=(1, 2)), [0.5, 0.5], atol=1e-14)
    np.testing.assert_allclose(dgp.clean.aux.sum(axis=1), [0.5, 0.5], atol=1e-14)
    np.testing.assert_allclose(dgp.tv_distances, np.full((2, 2), 0.1), atol=1e-14)
    np.testing.assert_allclose(dgp.clean.true_ate, 0.08, atol=1e-14)
    assert dgp.tilde_q.min() > 0.09
    assert dgp.main.min() > 0.025


def test_latent_law_satisfies_auxiliary_mixture() -> None:
    dgp = study.make_boundary_dgp()
    reconstructed = np.empty_like(dgp.clean.g)
    for a in range(2):
        reconstructed[a] = (
            dgp.clean.p[a] * dgp.clean.q[a, 1]
            + (1.0 - dgp.clean.p[a]) * dgp.clean.q[a, 0]
        )
    np.testing.assert_allclose(reconstructed, dgp.clean.g, atol=1e-14)


def test_population_is_on_both_exposed_faces() -> None:
    oracle = study.compute_oracle().iloc[0]
    np.testing.assert_allclose(oracle.control_exposed_margin, 0.0, atol=1e-14)
    np.testing.assert_allclose(oracle.treated_exposed_margin, 0.0, atol=1e-14)


def test_oracle_values() -> None:
    oracle = study.compute_oracle().iloc[0]
    np.testing.assert_allclose(oracle.oracle_lower, 0.06286744149056057, atol=2e-12)
    np.testing.assert_allclose(oracle.oracle_upper, 0.10064927973107418, atol=2e-12)
    np.testing.assert_allclose(oracle.oracle_width, 0.03778183824051361, atol=2e-12)
    assert oracle.oracle_lower < oracle.true_ate < oracle.oracle_upper


def test_population_penalty_exactness() -> None:
    oracle = study.compute_oracle().iloc[0].to_dict()
    for n in study.SAMPLE_SIZES:
        result = study.population_exactness(n, oracle)
        assert result["success"]
        assert result["max_abs_error"] <= 1e-7
        assert result["lower_total_slack"] == 0.0
        assert result["upper_total_slack"] == 0.0


def test_seeded_replication_is_reproducible() -> None:
    oracle = study.compute_oracle().iloc[0].to_dict()
    payload = (4_000, 0, 987_654_321, oracle)
    first = study.simulate_one(payload)
    second = study.simulate_one(payload)
    for key in first:
        if key != "runtime_seconds":
            assert first[key] == second[key]
    assert first["success"] and first["coherent"]


def test_seeded_batch_contains_feasible_and_infeasible_cases() -> None:
    oracle = study.compute_oracle().iloc[0].to_dict()
    rows = [
        study.simulate_one((4_000, rep, 1_234_000 + rep * 7_919, oracle))
        for rep in range(20)
    ]
    assert all(row["success"] and row["coherent"] for row in rows)
    assert any(row["raw_feasible"] for row in rows)
    assert any(not row["raw_feasible"] for row in rows)
    assert all(row["penalty_activated"] == (not row["raw_feasible"]) for row in rows)


def test_saved_confirmatory_outputs() -> None:
    results = HERE / "results"
    if not (results / "replications.csv.gz").exists():
        return
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(results / "replications.csv.gz").sort_values(["n", "rep"]).reset_index(drop=True)
    assert metadata["base_seed"] == study.BASE_SEED
    assert metadata["repetitions_per_sample_size"] == 1_500
    assert len(frame) == 7_500
    assert frame.seed.nunique() == len(frame)
    assert frame.groupby("n").size().eq(1_500).all()
    assert set(frame.n.unique()) == set(study.SAMPLE_SIZES)
    assert frame.success.all() and frame.coherent.all()
    assert frame.penalty_activated.eq(~frame.raw_feasible).all()
    assert frame.groupby("n").raw_feasible.mean().between(0.15, 0.35).all()
    feasible = frame.loc[frame.raw_feasible]
    np.testing.assert_allclose(feasible.lower_hat, feasible.raw_lower, atol=1e-10)
    np.testing.assert_allclose(feasible.upper_hat, feasible.raw_upper, atol=1e-10)

    cell_files = sorted(
        results.glob("replications_n*.csv.gz"),
        key=lambda path: int(path.name.removeprefix("replications_n").split(".", 1)[0]),
    )
    cells = pd.concat([pd.read_csv(path) for path in cell_files], ignore_index=True)
    cells = cells.sort_values(["n", "rep"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame, cells, check_exact=True)


def test_saved_analysis_reproduces_from_raw_data() -> None:
    raw_path = HERE / "results" / "replications.csv.gz"
    stored_path = HERE / "analysis" / "monte_carlo_summary.csv"
    if not (raw_path.exists() and stored_path.exists()):
        return
    reproduced = analysis.summarize(pd.read_csv(raw_path))
    stored = pd.read_csv(stored_path)
    pd.testing.assert_frame_equal(reproduced, stored, check_exact=False, atol=1e-14)


def main() -> None:
    tests = [
        test_generic_engine_matches_existing_k4_dgp,
        test_generic_engine_matches_binary_audited_engine,
        test_dgp_probability_laws_and_tv_distances,
        test_latent_law_satisfies_auxiliary_mixture,
        test_population_is_on_both_exposed_faces,
        test_oracle_values,
        test_population_penalty_exactness,
        test_seeded_replication_is_reproducible,
        test_seeded_batch_contains_feasible_and_infeasible_cases,
        test_saved_confirmatory_outputs,
        test_saved_analysis_reproduces_from_raw_data,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} tests")


if __name__ == "__main__":
    main()
