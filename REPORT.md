# Population and estimated interval comparison

## Result in one sentence

In the regular binary-covariate design, auxiliary knowledge of `P(X|A)` reduces the population identified-set width by 54.1%, and the proposed penalized endpoint estimator approaches that narrower population set at a clean root-n rate.

## Design

We use a fixed binary-covariate DGP:

- `P(X=0)=P(X=1)=0.5`;
- `P(A=1|X=0)=0.4` and `P(A=1|X=1)=0.6`;
- `P(Y=1|A=0,X)=(0.20,0.50)`;
- `P(Y=1|A=1,X)=(0.28,0.58)`.

Consequently, the true ATE is `0.08`. At the center of the sensitivity model, the reported covariate equals the clean covariate, while the model permits a TV radius `rho=0.10` in every `(A,Y)` stratum.

Each replication independently draws `n` observations of `(X_tilde,A,Y)` from the main law and `N=n` observations of `(X,A)` from the auxiliary law. The sample-size ladder is `4,000, 16,000, 64,000, 256,000, 1,024,000`, with 1,500 fresh Monte Carlo replications per value.

The empirical endpoints use the full-sample debiased exact-L1-penalty estimator with lexicographic debiasing and `w_n = 5 log(log n) / log(log 30000)`. Joint cell probabilities use a Jeffreys `0.5` pseudocount per cell.

## Population oracles

The population auxiliary-data oracle is

`I_aux = [0.047500, 0.112500]`,

with width `0.065000`. Removing only the auxiliary mixture restrictions that identify `P(X|A)` gives

`I_noaux = [-0.00156, 0.14000]` (approximately),

with width `0.141561`. Thus access to the auxiliary law reduces width by `0.076561`, or 54.1% relative to the no-auxiliary width. The specialized two-dimensional profile and an independent four-dimensional SHGO search agree within `2.0e-15`; interval branch-and-bound brackets each reported no-auxiliary endpoint within `2.0e-6`.

## Monte Carlo results

| n=N per source | Mean estimated interval | Lower RMSE | Upper RMSE | 95th percentile Hausdorff error |
|---:|:---:|---:|---:|---:|
| 4,000 | [0.049111, 0.111516] | 0.018659 | 0.016700 | 0.036769 |
| 16,000 | [0.047835, 0.112385] | 0.009132 | 0.008235 | 0.018070 |
| 64,000 | [0.047672, 0.112564] | 0.004441 | 0.004014 | 0.008872 |
| 256,000 | [0.047448, 0.112461] | 0.002198 | 0.002033 | 0.004452 |
| 1,024,000 | [0.047528, 0.112540] | 0.001091 | 0.001006 | 0.002116 |

The root-n-scaled diagnostics remain roughly level across the sample-size ladder. Log-log decay exponents across all five sizes are 0.5123 for lower-endpoint RMSE, 0.5062 for upper-endpoint RMSE, and 0.5129 for the joint 95th-percentile error, versus the root-n benchmark 0.5.

All 7,500 endpoint pairs solved successfully and were coherent. In this regular design, the ordinary plug-in LP was feasible in every replication and penalty slack was zero, so this experiment isolates the value of auxiliary information and regular root-n recovery rather than the penalty method's infeasibility repair.

## Scope

This is transparent evidence for one fixed, regular binary-X DGP under independent, equally sized samples. It does not establish uniformity over DGPs, growing support, asymptotic normality, confidence-interval coverage, or the practical benefit of penalization under empirical infeasibility.
