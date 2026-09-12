#!/usr/bin/env python3
"""Validate the added cells and update the boundary figure and bootstrap slopes."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from boundary_contamination_study import analyze_boundary_contamination_study as analysis


def main():
    seeds = []
    bounds = json.loads((ROOT / "support_size_study/population_bounds.json").read_text())
    checks = {}
    for design in ("regular_K2", "regular_K4", "regular_K16", "boundary_K4"):
        frame = pd.read_csv(HERE / design / "replications.csv.gz")
        assert frame.groupby("n").size().to_dict() == {250: 1500, 1000: 1500}
        assert not frame.duplicated(["n", "rep"]).any()
        assert frame.success.all() and frame.coherent.all()
        numeric = frame[["lower_hat", "upper_hat", "lower_error", "upper_error", "joint_abs_error",
                         "lower_total_slack", "upper_total_slack", "lower_primary_gap", "upper_primary_gap"]]
        assert np.isfinite(numeric.to_numpy()).all()
        assert (frame.width_hat >= -1e-10).all()
        if design.startswith("regular"):
            oracle = bounds[design.split("K")[1]]
            lower, upper = oracle["aux_lower"], oracle["aux_upper"]
        else:
            oracle = pd.read_csv(ROOT / "boundary_contamination_study/results/oracle_bounds.csv").iloc[0]
            lower, upper = oracle.oracle_lower, oracle.oracle_upper
        assert np.allclose(frame.lower_hat - lower, frame.lower_error, rtol=0, atol=1e-14)
        assert np.allclose(frame.upper_hat - upper, frame.upper_error, rtol=0, atol=1e-14)
        assert np.allclose(frame.joint_abs_error, np.maximum(frame.lower_error.abs(), frame.upper_error.abs()), atol=1e-14)
        assert numeric[["lower_primary_gap", "upper_primary_gap"]].abs().max().max() <= 1.01e-8
        seeds.extend(frame.seed.tolist())
        checks[design] = {str(n): {
            "replications": len(cell), "direct_feasible_count": int(cell.raw_feasible.sum()),
            "penalized_success_count": int(cell.success.sum()),
            "coherent_count": int(cell.coherent.sum()),
        } for n, cell in frame.groupby("n")}
    assert len(seeds) == len(set(seeds)) == 12000
    output = HERE / "boundary_K4"
    combined = pd.read_csv(output / "combined_replications.csv.gz")
    assert len(combined) == 10500 and combined.seed.is_unique
    summary = analysis.summarize(combined)
    archived_summary = pd.read_csv(output / "combined_summary.csv")
    for key in ("lower_rmse", "upper_rmse", "raw_feasibility_rate", "joint_q95"):
        assert np.allclose(summary[key], archived_summary[key], rtol=0, atol=1e-13)
    slopes = analysis.bootstrap_slopes(combined, draws=1000, seed=5_026_091_102)
    slopes.to_csv(output / "bootstrap_rate_slopes.csv", index=False)
    analysis.plot_main_diagnostics(summary, output)
    (HERE / "integrity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(slopes.to_string(index=False))
    print("Validated 12,000 added replications and updated boundary diagnostics.")


if __name__ == "__main__":
    main()
