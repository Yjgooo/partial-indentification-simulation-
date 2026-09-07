# Support-size robustness study: K = 4 and K = 16

## Design

This repeats the existing equal-sample experiment while changing only the support size of the discrete confounder. To make the comparison as close as possible across support sizes, define an equally spaced score and standardize it within each K:

- `z_k = standardized(linspace(-1, 1, K))`, so `E[z]=0` and `E[z^2]=1`;
- `P(X=k)=1/K`;
- `P(A=1|X=k)=0.5 + 0.1 z_k`;
- `P(Y=1|A=0,X=k)=0.35 + 0.15 z_k`;
- `P(Y=1|A=1,X=k)=0.43 + 0.15 z_k`.

For `K=2` this is exactly the DGP in the original study: propensity `(0.4,0.6)`, untreated outcome `(0.20,0.50)`, and treated outcome `(0.28,0.58)`. For every K the true ATE is 0.08 and the treatment-arm outcome marginals remain `P(Y=1|A=0)=0.32` and `P(Y=1|A=1)=0.46`.

As before, the sensitivity radius is `rho=0.10` in every `(A,Y)` stratum. The main and auxiliary samples are independent and equally sized, with `n=N` in `{4,000, 16,000, 64,000, 256,000, 1,024,000}`. Each cell uses 1,500 Monte Carlo replications. Main and auxiliary empirical cell probabilities use the same Jeffreys `0.5` pseudocount rule and the endpoint estimator uses the same exact-L1 penalty sequence as the K=2 study.

## Population identification

| K | Auxiliary interval | Auxiliary width | No-auxiliary interval | No-aux width | Width reduction |
|---:|:---:|---:|:---:|---:|---:|
| 2 | [0.047500, 0.112500] | 0.065000 | [-0.001561, 0.140000] | 0.141561 | 54.1% |
| 4 | [0.036009, 0.125107] | 0.089098 | [-0.071267, 0.160321] | 0.231587 | at least 61.5% |
| 16 | [0.032937, 0.135104] | 0.102168 | [-0.098015, 0.201558] | 0.299573 | at least 65.9% |

The auxiliary intervals are exact LP solutions. The K=2 no-auxiliary endpoints are the previously certified values. For K=4 and K=16, the no-auxiliary problem is nonconvex and the specialized K=2 global certificate is unavailable. The displayed K=4 values are extremely stable across 24 multistart SLSQP runs. The K=16 lower endpoint repeatedly converged to approximately `-0.098015`, while the upper endpoint has multiple local optima; `0.201558` is the strongest feasible candidate found. Hence the K>2 no-auxiliary widths are best-found lower bounds on the true no-auxiliary width, and the reported width reductions are conservative.

## Monte Carlo results: K=4

| n=N | Mean estimated interval | Lower RMSE | Upper RMSE | 95th percentile joint error |
|---:|:---:|---:|---:|---:|
| 4,000 | [0.037902, 0.123606] | 0.018913 | 0.016291 | 0.037061 |
| 16,000 | [0.036242, 0.124841] | 0.009250 | 0.008061 | 0.018023 |
| 64,000 | [0.035999, 0.125092] | 0.004571 | 0.004082 | 0.009438 |
| 256,000 | [0.036014, 0.125099] | 0.002284 | 0.002003 | 0.004683 |
| 1,024,000 | [0.035972, 0.125089] | 0.001139 | 0.001012 | 0.002286 |

The log-log decay exponents are `0.5063` for lower RMSE, `0.5013` for upper RMSE, and `0.4992` for the joint 95th-percentile error.

## Monte Carlo results: K=16

| n=N | Mean estimated interval | Lower RMSE | Upper RMSE | 95th percentile joint error |
|---:|:---:|---:|---:|---:|
| 4,000 | [0.033525, 0.131501] | 0.018383 | 0.016898 | 0.037433 |
| 16,000 | [0.032609, 0.133445] | 0.009394 | 0.008394 | 0.019146 |
| 64,000 | [0.032908, 0.134890] | 0.004740 | 0.004255 | 0.009512 |
| 256,000 | [0.032954, 0.135122] | 0.002275 | 0.002084 | 0.004638 |
| 1,024,000 | [0.032954, 0.135101] | 0.001160 | 0.001043 | 0.002372 |

The log-log decay exponents are `0.5009` for lower RMSE, `0.5023` for upper RMSE, and `0.5003` for the joint 95th-percentile error.

## Numerical checks

All 15,000 K=4/K=16 endpoint pairs solved successfully and coherently. The ordinary plug-in auxiliary LP was feasible in every replication at both support sizes. Mean penalty slack is zero to numerical precision. At the population probabilities, the penalized LP reproduces the direct auxiliary LP endpoints at every sample-size tuning level, with maximum discrepancies below `4e-16`.

## Takeaway

Increasing K weakens population identification under a fixed TV radius: the auxiliary identified-set width rises from `0.065` at K=2 to `0.089` at K=4 and `0.102` at K=16. The no-auxiliary problem deteriorates much faster. In contrast, once the auxiliary restrictions are imposed, the finite-sample endpoint estimator continues to exhibit an essentially exact root-n rate through K=16, with root-n-scaled error constants of similar magnitude across support sizes.

This experiment therefore supports two distinct claims: (i) auxiliary knowledge of `P(X|A)` becomes increasingly valuable as the discrete support grows, and (ii) for fixed finite K and adequately populated cells, increasing K from 2 to 16 does not visibly compromise the root-n behavior of the proposed endpoint estimator in this regular design.
