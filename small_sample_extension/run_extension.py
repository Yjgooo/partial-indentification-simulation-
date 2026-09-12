#!/usr/bin/env python3
"""Add n=N=250,1000 using the original runners; retain the archived larger cells."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import run_equal_n_experiment as binary
import analyze_equal_n_experiment as binary_analysis
from support_size_study import run_support_size_study as support
from boundary_contamination_study import run_boundary_contamination_study as boundary
from boundary_contamination_study import analyze_boundary_contamination_study as boundary_analysis

SIZES = (250, 1000)
REPETITIONS = 1500
BASE_SEED = 5_000_000_000_000
ARCHIVES = {
    "regular_K2": ROOT / "final_analysis/monte_carlo_summary.csv",
    "regular_K4": ROOT / "support_size_study/summary_K4.csv",
    "regular_K16": ROOT / "support_size_study/summary_K16.csv",
    "boundary_K4": ROOT / "boundary_contamination_study/analysis/monte_carlo_summary.csv",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_replications(frame, path):
    raw = frame.to_csv(index=False).encode()
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(gzip.compress(raw))
        stream.flush()
        os.fsync(stream.fileno())
    assert gzip.decompress(temporary.read_bytes()) == raw
    temporary.replace(path)


def run(workers):
    started = time.time()
    tracked = list(ARCHIVES.values()) + [
        ROOT / "run_equal_n_experiment.py",
        ROOT / "support_size_study/run_support_size_study.py",
        ROOT / "boundary_contamination_study/run_boundary_contamination_study.py",
        ROOT / "vendor/aux_value_engine.py",
        ROOT / "vendor/finite_support_engine.py",
        ROOT / "support_size_study/population_bounds.json",
        ROOT / "boundary_contamination_study/results/replications.csv.gz",
    ]
    original_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in tracked}
    oracles = json.loads((ROOT / "support_size_study/population_bounds.json").read_text())
    binary_oracle = pd.read_csv(ROOT / "results/oracle_bounds.csv").iloc[0].to_dict()
    boundary_oracle = pd.read_csv(ROOT / "boundary_contamination_study/results/oracle_bounds.csv").iloc[0].to_dict()
    all_seeds = set()
    checks = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for design_index, (design, archive) in enumerate(ARCHIVES.items()):
            output = HERE / design
            output.mkdir(parents=True, exist_ok=True)
            if design == "regular_K2":
                oracle, worker, summarizer = binary_oracle, binary.simulate_one, binary_analysis.summarize
                exactness = [binary.population_exactness(n, oracle) for n in SIZES]
            elif design == "boundary_K4":
                oracle, worker, summarizer = boundary_oracle, boundary.simulate_one, boundary_analysis.summarize
                exactness = [boundary.population_exactness(n, oracle) for n in SIZES]
            else:
                k = int(design.split("K")[1])
                oracle, worker, summarizer = oracles[str(k)], support.simulate_one, support.summarize
                exactness = [support.pop_exactness(k, n, oracle) for n in SIZES]
            exact = pd.DataFrame(exactness)
            exact.to_csv(output / "population_penalty_exactness.csv", index=False)
            assert exact.success.all(), f"Population exactness failed: {design}"
            frames = []
            for n in SIZES:
                seeds = [BASE_SEED + design_index * 100_000_000 + n * 10_000 + rep for rep in range(REPETITIONS)]
                assert not all_seeds.intersection(seeds)
                all_seeds.update(seeds)
                if design in ("regular_K2", "boundary_K4"):
                    payloads = [(n, rep, seed, oracle) for rep, seed in enumerate(seeds)]
                else:
                    payloads = [(k, n, rep, seed, oracle) for rep, seed in enumerate(seeds)]
                cell_start = time.time()
                frame = pd.DataFrame(pool.map(worker, payloads, chunksize=8))
                save_replications(frame, output / f"replications_n{n}.csv.gz")
                frames.append(frame)
                print(f"{design} n={n}: {int(frame.success.sum())}/{len(frame)} solved; "
                      f"coherent={frame.coherent.mean():.4f}, feasible={frame.raw_feasible.mean():.4f}; "
                      f"{time.time()-cell_start:.1f}s", flush=True)
            new = pd.concat(frames, ignore_index=True)
            save_replications(new, output / "replications.csv.gz")
            summary = summarizer(new)
            summary.to_csv(output / "small_sample_summary.csv", index=False)
            archived = pd.read_csv(archive)
            combined = pd.concat([summary, archived], ignore_index=True).sort_values("n")
            assert combined.n.tolist() == [250, 1000, 4000, 16000, 64000, 256000, 1024000]
            combined.to_csv(output / "combined_summary.csv", index=False)
            slope_rows = []
            for sample_range, cell in (("all_seven", combined), ("original_five", archived)):
                for metric in ("lower_rmse", "upper_rmse", "joint_q95"):
                    slope_rows.append({"sample_range": sample_range, "metric": metric,
                                       "decay_exponent": float(-np.polyfit(np.log(cell.n), np.log(cell[metric]), 1)[0])})
            pd.DataFrame(slope_rows).to_csv(output / "rate_slopes.csv", index=False)
            checks[design] = {
                "new_replications": len(new),
                "all_solved": bool(new.success.all()),
                "all_coherent": bool(new.coherent.all()),
                "minimum_width": float(new.width_hat.min()),
                "max_absolute_primary_gap": float(new[["lower_primary_gap", "upper_primary_gap"]].abs().max().max()),
                "max_total_slack": float(new[["lower_total_slack", "upper_total_slack"]].max().max()),
                "population_max_error": float(exact.max_abs_error.max()),
            }
            if design == "boundary_K4":
                old = pd.read_csv(ROOT / "boundary_contamination_study/results/replications.csv.gz")
                assert not set(old.seed).intersection(new.seed)
                merged = pd.concat([new, old], ignore_index=True).sort_values(["n", "rep"])
                save_replications(merged, output / "combined_replications.csv.gz")
    assert all(sha256(ROOT / name) == digest for name, digest in original_hashes.items())
    metadata = {
        "new_sample_sizes": list(SIZES), "repetitions_per_setting": REPETITIONS,
        "designs": list(ARCHIVES), "new_replications": len(all_seeds),
        "base_seed": BASE_SEED,
        "seed_rule": "base + design_index*100000000 + n*10000 + rep",
        "new_seeds_unique": True,
        "seed_note": "New scalar seed range is disjoint from the original runners' published seed schedules.",
        "original_files_unchanged": original_hashes,
        "elapsed_seconds": time.time() - started, "checks": checks,
    }
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args().workers)
