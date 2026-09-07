# Independent audit of the identification-interval experiment

## Audit outcome

The statistical and optimization implementations pass the audit. One
provenance issue was found and corrected: the original confirmatory run reused
the 500 pilot seeds as its first 100 replications at each sample size. The
experiment was rerun with base seed `2026090711`. The corrected confirmatory
dataset contains 7,500 unique seeds and has zero seed overlap with either the
pilot or the superseded run.

The rerun preserves the substantive conclusion: auxiliary knowledge of
`P(X|A)` reduces the population identified-set width by 54.1%, and the
estimated endpoints exhibit the pointwise root-n pattern expected in this
regular fixed DGP.

## Model and DGP reconstruction

Starting only from the primitive probabilities in the design gives

- `P(A=1)=0.5`;
- `P(X=1|A=0)=0.4` and `P(X=1|A=1)=0.6`;
- `P(Y=1|A=0)=0.32` and `P(Y=1|A=1)=0.46`;
- `P(X=1|A=1,Y=1)=0.7565217391`;
- `P(X=1|A=1,Y=0)=0.4666666667`;
- `P(X=1|A=0,Y=1)=0.625`;
- `P(X=1|A=0,Y=0)=0.2941176471`;
- true ATE `0.08`.

The primary and auxiliary multinomial laws reconstruct these probabilities
exactly. Separate child streams of a NumPy `SeedSequence` generate the primary
and auxiliary samples, so the two sources are independent by construction.

## Population-bound checks

An independently written binary-covariate corner enumeration reproduces the
auxiliary-data population bounds as

`[0.0475000000, 0.1125000000]`.

An independent direct-standardization objective optimized by differential
evolution reproduces the no-auxiliary bounds as

`[-0.001561242275, 0.140000000000]`.

These agree with the specialized two-dimensional profile and the separate
four-dimensional SHGO calculation. Outward-rounded interval branch-and-bound
encloses each no-auxiliary endpoint within `2e-6`.

The population widths are therefore `0.065` with auxiliary information and
`0.1415612423` without it. The relative reduction is

`1 - 0.065 / 0.1415612423 = 0.54083477`,

or 54.1% after rounding.

## Penalized-estimator check

The code implements the debiased penalty construction for a lower LP endpoint:
minimize the original objective plus the weighted sum of positive constraint
violations, then maximize the original objective over the resulting penalized
argmin set. The upper endpoint applies the same construction to the negative
ATE objective and negates the selected value.

The estimated objective is represented by an epigraph variable. Giving its
linking row unit weight makes minimization over the epigraph variable exactly
recover the estimated linear objective; the substantive constraint violations
receive weight `w_n`. The secondary maximization then returns the largest
original objective over the penalized argmin, as required. The chosen sequence

`w_n = 5 log(log n) / log(log 30000)`

diverges and is `o(sqrt(n))`. The padded compact domain places every feasible
probability vector and ATE value in its interior. With independent equal-sized
sources, observations can be paired into `n` i.i.d. two-source blocks, so the
joint nuisance estimator has the root-n rate required by the theorem.

## Raw-output and figure checks

- There are exactly 1,500 replications at each of five sample sizes.
- All 7,500 seeds and `(n,rep)` keys are unique.
- The combined replication file equals the five concatenated cell files.
- Every penalized solve succeeded and every endpoint pair satisfies
  `lower_hat <= upper_hat`.
- Stored errors, scaled errors, RMSEs, quantiles, and log-log slopes reproduce
  directly from the raw endpoint columns.
- Replaying 25 randomly selected seeds and using an independently coded corner
  solution reproduces both stored endpoints within `4e-16`.
- The PNG and PDF are generated from the corrected confirmatory data. PDF fonts
  are embedded TrueType fonts rather than Type 3 fonts.

In this deliberately regular design, the empirical plug-in LP is feasible in
every replication. Penalty slack is consequently zero, and penalized and
plug-in endpoints agree to machine precision. The figure tests the value of the
auxiliary information and regular root-n recovery; it does not test the
penalty's infeasibility repair, confidence-interval coverage, uniformity over
DGPs, growing support, or asymptotic normality.
