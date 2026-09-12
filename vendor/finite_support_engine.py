#!/usr/bin/env python3
"""Generic finite-support DGP, direct LP, and penalty wrappers.

Arrays use the order ``[a, y, k]`` throughout, with A,Y in {0,1} and
``k=0,...,K-1``. The low-level inequality and debiased-penalty routines are
the audited implementations in :mod:`aux_value_engine`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

import aux_value_engine as penalty_engine


TINY = 1e-12


@dataclass(frozen=True)
class SmoothDGP:
    K: int
    px: np.ndarray
    propensity: np.ndarray
    outcome: np.ndarray
    alpha: np.ndarray
    p: np.ndarray
    g: np.ndarray
    q: np.ndarray
    main: np.ndarray
    aux: np.ndarray
    true_ate: float


@dataclass(frozen=True)
class DirectBoundPair:
    lower: float
    upper: float
    feasible: bool
    lower_status: int
    upper_status: int


def make_smooth_dgp(K: int) -> SmoothDGP:
    """Construct the smooth K-category extension of the binary reference DGP."""

    if K < 2:
        raise ValueError("K must be at least 2")
    z = np.linspace(-1.0, 1.0, K)
    z /= np.sqrt(np.mean(z**2))
    px = np.ones(K) / K
    propensity = 0.5 + 0.1 * z
    y0 = 0.35 + 0.15 * z
    y1 = y0 + 0.08
    outcome = np.vstack([y0, y1])
    if np.any((propensity <= 0.0) | (propensity >= 1.0)):
        raise ValueError("propensity is outside (0,1)")
    if np.any((outcome <= 0.0) | (outcome >= 1.0)):
        raise ValueError("outcome probability is outside (0,1)")

    alpha1 = float(px @ propensity)
    alpha = np.array([1.0 - alpha1, alpha1])
    g = np.empty((2, K))
    p = np.empty(2)
    q = np.empty((2, 2, K))
    for a in range(2):
        pa_x = propensity if a == 1 else 1.0 - propensity
        g[a] = px * pa_x / alpha[a]
        p[a] = float(g[a] @ outcome[a])
        q[a, 1] = g[a] * outcome[a] / p[a]
        q[a, 0] = g[a] * (1.0 - outcome[a]) / (1.0 - p[a])

    main = joint_main_probability(alpha, p, q)
    aux = alpha[:, None] * g
    true_ate = float(px @ (y1 - y0))
    np.testing.assert_allclose(aux.sum(), 1.0, atol=1e-14)
    return SmoothDGP(
        K, px, propensity, outcome, alpha, p, g, q, main, aux, true_ate
    )


def joint_main_probability(
    alpha: np.ndarray, p: np.ndarray, conditional_x: np.ndarray
) -> np.ndarray:
    """Return joint P(A,Y,X) from P(A), P(Y=1|A), and P(X|A,Y)."""

    alpha = np.asarray(alpha, dtype=float)
    p = np.asarray(p, dtype=float)
    conditional_x = np.asarray(conditional_x, dtype=float)
    if alpha.shape != (2,) or p.shape != (2,) or conditional_x.ndim != 3:
        raise ValueError("incompatible alpha, p, or conditional-X shape")
    if conditional_x.shape[:2] != (2, 2):
        raise ValueError("conditional-X array must have shape (2,2,K)")
    np.testing.assert_allclose(conditional_x.sum(axis=2), 1.0, atol=1e-12)
    law = np.empty_like(conditional_x)
    for a in range(2):
        for y in range(2):
            py = p[a] if y == 1 else 1.0 - p[a]
            law[a, y] = alpha[a] * py * conditional_x[a, y]
    np.testing.assert_allclose(law.sum(), 1.0, atol=1e-12)
    return law


def smoothed_probability(counts: np.ndarray, smooth: float = 0.5) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return (counts + smooth) / (counts.sum() + smooth * counts.size)


def estimate_main(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    law = smoothed_probability(counts)
    alpha = law.sum(axis=(1, 2))
    p = law[:, 1].sum(axis=1) / alpha
    q = law / law.sum(axis=2, keepdims=True)
    return alpha, p, q


def estimate_aux(counts: np.ndarray) -> np.ndarray:
    law = smoothed_probability(counts)
    return law / law.sum(axis=1, keepdims=True)


def direct_aux_pair(
    alpha: np.ndarray,
    p: np.ndarray,
    g: np.ndarray,
    tilde_q: np.ndarray,
    rho: float,
) -> DirectBoundPair:
    """Solve both sample-analogue auxiliary LP endpoints directly."""

    alpha = np.asarray(alpha, dtype=float)
    p = np.asarray(p, dtype=float)
    g = np.asarray(g, dtype=float)
    tilde_q = np.asarray(tilde_q, dtype=float)
    K = g.shape[1]
    if g.shape != (2, K) or tilde_q.shape != (2, 2, K):
        raise ValueError("g and tilde_q have incompatible shapes")
    n_q = 4 * K
    n_u = 4 * K
    n_variables = n_q + n_u

    def qi(a: int, y: int, k: int) -> int:
        return (2 * a + y) * K + k

    def ui(a: int, y: int, k: int) -> int:
        return n_q + (2 * a + y) * K + k

    px = alpha[0] * g[0] + alpha[1] * g[1]
    objective = np.zeros(n_variables)
    for k in range(K):
        if g[0, k] <= TINY or g[1, k] <= TINY:
            raise ValueError("auxiliary cell probability violates overlap")
        objective[qi(1, 1, k)] = px[k] * p[1] / g[1, k]
        objective[qi(0, 1, k)] = -px[k] * p[0] / g[0, k]

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for a in range(2):
        for y in range(2):
            row = np.zeros(n_variables)
            row[[qi(a, y, k) for k in range(K)]] = 1.0
            a_eq.append(row)
            b_eq.append(1.0)
        for k in range(K):
            row = np.zeros(n_variables)
            row[qi(a, 0, k)] = 1.0 - p[a]
            row[qi(a, 1, k)] = p[a]
            a_eq.append(row)
            b_eq.append(float(g[a, k]))

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for a in range(2):
        for y in range(2):
            for k in range(K):
                row = np.zeros(n_variables)
                row[qi(a, y, k)] = 1.0
                row[ui(a, y, k)] = -1.0
                a_ub.append(row)
                b_ub.append(float(tilde_q[a, y, k]))
                row = np.zeros(n_variables)
                row[qi(a, y, k)] = -1.0
                row[ui(a, y, k)] = -1.0
                a_ub.append(row)
                b_ub.append(float(-tilde_q[a, y, k]))
            row = np.zeros(n_variables)
            row[[ui(a, y, k) for k in range(K)]] = 1.0
            a_ub.append(row)
            b_ub.append(float(2.0 * rho))

    kwargs = {
        "A_ub": np.asarray(a_ub),
        "b_ub": np.asarray(b_ub),
        "A_eq": np.asarray(a_eq),
        "b_eq": np.asarray(b_eq),
        "bounds": [(0.0, 1.0)] * n_q + [(0.0, None)] * n_u,
        "method": "highs",
        "options": {
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    }
    lower = linprog(objective, **kwargs)
    upper = linprog(-objective, **kwargs)
    feasible = bool(lower.success and upper.success)
    return DirectBoundPair(
        lower=float(lower.fun) if lower.success else float("nan"),
        upper=float(-upper.fun) if upper.success else float("nan"),
        feasible=feasible,
        lower_status=int(lower.status),
        upper_status=int(upper.status),
    )


def penalty_pair(
    alpha: np.ndarray,
    p: np.ndarray,
    g: np.ndarray,
    tilde_q: np.ndarray,
    rho: float,
    n_main: int,
    n_aux: int,
) -> tuple[
    penalty_engine.PenaltySolve,
    penalty_engine.PenaltySolve,
    float,
]:
    """Apply the audited debiased penalty construction for generic K."""

    w_n = penalty_engine.penalty_level(n_main, n_aux)
    tolerance = penalty_engine.face_tolerance(n_main, n_aux)
    lower_system = penalty_engine.endpoint_inequality_system(
        alpha, p, g, tilde_q, rho, 1.0
    )
    neg_upper_system = penalty_engine.endpoint_inequality_system(
        alpha, p, g, tilde_q, rho, -1.0
    )
    return (
        penalty_engine.solve_debiased_value(
            *lower_system, w_n, face_tolerance=tolerance
        ),
        penalty_engine.solve_debiased_value(
            *neg_upper_system, w_n, face_tolerance=tolerance
        ),
        w_n,
    )
