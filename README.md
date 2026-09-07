# Identification-interval comparison experiment

This experiment compares, under one fixed binary-covariate DGP:

1. the population ATE identified set after removing the auxiliary restriction;
2. the population ATE identified set with auxiliary knowledge of `P(X|A)`;
3. the proposed penalized estimated auxiliary-data interval at five equal sample
   sizes `n_main = N_aux = n`.

The two samples are independent. The sensitivity radius is `rho=0.10` in all
four `(A,Y)` strata. There are 1,500 Monte Carlo replications at each of
`n = 4,000, 16,000, 64,000, 256,000, 1,024,000`.
The confirmatory seeds are disjoint from the pilot seeds.

## Reproduce

From this directory:

```bash
python test_equal_n_experiment.py
python run_equal_n_experiment.py \
  --output results \
  --repetitions 1500 \
  --workers 6 \
  --certificate-tol 2e-6
python analyze_equal_n_experiment.py \
  --input results \
  --output final_analysis
```

Dependencies are Python 3.12+, NumPy, pandas, SciPy, and Matplotlib.

The copied files under `vendor/` are the audited binary-X auxiliary-value
engine and its interval branch-and-bound implementation.
