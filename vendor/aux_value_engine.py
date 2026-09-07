#!/usr/bin/env python3
"""Audited binary-X auxiliary LP and Voronin-inspired penalty engine."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linprog, shgo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from global_k2 import (  # noqa: E402
    ate_binary,
    solve_aux_binary,
    stress_dgp,
    variable_bounds,
)

TINY = 1e-12


@dataclass(frozen=True)
class BinaryDGP:
    dgp_id: str
    label: str
    alpha1: float
    p0: float
    p1: float
    centers: tuple[float, float, float, float]
    auxiliary_r: tuple[float, float]
    true_ate: float


@dataclass(frozen=True)
class ProfileResult:
    endpoint: str
    value: float
    g0: float
    g1: float
    success: bool
    nfev: int


@dataclass(frozen=True)
class PenaltySolve:
    selected_value: float
    arbitrary_value: float
    preliminary_value: float
    total_slack: float
    max_slack: float
    primary_gap: float
    selected_original_violation: float
    success: bool


def reference_dgp() -> BinaryDGP:
    """Balanced regular DGP used in the identification-interval experiment."""
    px = np.array([0.5, 0.5])
    propensity = np.array([0.4, 0.6])
    outcome = np.array([[0.20, 0.50], [0.28, 0.58]])
    alpha1 = float(np.dot(px, propensity))
    alpha = np.array([1.0 - alpha1, alpha1])
    p = np.empty(2)
    g_one = np.empty(2)
    q_one = np.empty((2, 2))
    for a in range(2):
        pa_x = propensity if a == 1 else 1.0 - propensity
        gx = px * pa_x / alpha[a]
        g_one[a] = gx[1]
        p[a] = float(np.dot(gx, outcome[a]))
        q_one[a, 1] = gx[1] * outcome[a, 1] / p[a]
        q_one[a, 0] = gx[1] * (1.0 - outcome[a, 1]) / (1.0 - p[a])
    centers = (
        float(q_one[1, 1]),
        float(q_one[1, 0]),
        float(q_one[0, 1]),
        float(q_one[0, 0]),
    )
    true_ate = float(np.dot(px, outcome[1] - outcome[0]))
    return BinaryDGP(
        "regular",
        "Regular reference",
        alpha1,
        float(p[0]),
        float(p[1]),
        centers,
        (float(g_one[0]), float(g_one[1])),
        true_ate,
    )


def binary_main_probability(dgp: BinaryDGP) -> np.ndarray:
    """Return joint P(A,Y,X_tilde) in [a,y,x] order."""
    u1, v1, u0, v0 = dgp.centers
    q_one = np.array([[v0, u0], [v1, u1]], dtype=float)
    alpha = np.array([1.0 - dgp.alpha1, dgp.alpha1])
    p = np.array([dgp.p0, dgp.p1])
    law = np.empty((2, 2, 2), dtype=float)
    for a in range(2):
        for y in range(2):
            py = p[a] if y == 1 else 1.0 - p[a]
            law[a, y] = alpha[a] * py * np.array(
                [1.0 - q_one[a, y], q_one[a, y]]
            )
    np.testing.assert_allclose(law.sum(), 1.0, atol=1e-14)
    return law


def binary_aux_probability(dgp: BinaryDGP) -> np.ndarray:
    """Return joint P(A,X) in [a,x] order."""
    alpha = np.array([1.0 - dgp.alpha1, dgp.alpha1])
    g0, g1 = dgp.auxiliary_r
    g = np.array([[1.0 - g0, g0], [1.0 - g1, g1]])
    law = alpha[:, None] * g
    np.testing.assert_allclose(law.sum(), 1.0, atol=1e-14)
    return law


def smoothed_probability(counts: np.ndarray, smooth: float = 0.5) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return (counts + smooth) / (counts.sum() + smooth * counts.size)


def estimate_main(counts: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    law = smoothed_probability(counts)
    alpha = law.sum(axis=(1, 2))
    p = law[:, 1].sum(axis=1) / alpha
    q_one = law[:, :, 1] / law.sum(axis=2)
    centers = np.array([q_one[1, 1], q_one[1, 0], q_one[0, 1], q_one[0, 0]])
    tilde_q = law / alpha[:, None, None]
    tilde_q /= tilde_q.sum(axis=2, keepdims=True)
    return float(alpha[1]), float(p[0]), float(p[1]), centers, tilde_q


def estimate_aux_g(counts: np.ndarray) -> np.ndarray:
    law = smoothed_probability(counts)
    return law / law.sum(axis=1, keepdims=True)


def _profile_problem(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float | Sequence[float],
    endpoint: str,
):
    if endpoint not in {"lower", "upper"}:
        raise ValueError("endpoint must be lower or upper")
    box = variable_bounds(centers, rho)
    arm = [
        (p0, box[2][0], box[2][1], box[3][0], box[3][1]),
        (p1, box[0][0], box[0][1], box[1][0], box[1][1]),
    ]

    def g_range(a: int) -> tuple[float, float]:
        p, ulo, uhi, vlo, vhi = arm[a]
        return p * ulo + (1.0 - p) * vlo, p * uhi + (1.0 - p) * vhi

    g_bounds = [g_range(0), g_range(1)]
    if min(g_bounds[0][0], g_bounds[1][0]) <= 0.0:
        raise ValueError("profile permits a zero denominator")
    if max(g_bounds[0][1], g_bounds[1][1]) >= 1.0:
        raise ValueError("profile permits a unit denominator")

    def r_bound(a: int, g: float, kind: str) -> float:
        p, ulo, uhi, vlo, vhi = arm[a]
        if kind == "lower":
            return max(p * ulo, g - (1.0 - p) * vhi)
        return min(p * uhi, g - (1.0 - p) * vlo)

    def objective(g: np.ndarray) -> float:
        g0, g1 = map(float, g)
        if endpoint == "lower":
            kind = "lower" if g0 >= g1 else "upper"
        else:
            kind = "upper" if g0 >= g1 else "lower"
        r0 = r_bound(0, g0, kind)
        r1 = r_bound(1, g1, kind)
        value = (
            alpha1 * p1
            - (1.0 - alpha1) * p0
            + (1.0 - alpha1)
            * (g0 * r1 / g1 + (1.0 - g0) * (p1 - r1) / (1.0 - g1))
            - alpha1
            * (g1 * r0 / g0 + (1.0 - g1) * (p0 - r0) / (1.0 - g0))
        )
        return float(value)

    return objective, g_bounds


def solve_noaux_profile(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float,
    endpoint: str,
    n: int = 64,
    iters: int = 1,
) -> ProfileResult:
    if abs(float(rho)) <= 1e-15:
        value = ate_binary(centers, alpha1, p0, p1)
        g0 = p0 * centers[2] + (1.0 - p0) * centers[3]
        g1 = p1 * centers[0] + (1.0 - p1) * centers[1]
        return ProfileResult(endpoint, value, float(g0), float(g1), True, 1)
    objective, bounds = _profile_problem(alpha1, p0, p1, centers, rho, endpoint)
    sign = 1.0 if endpoint == "lower" else -1.0
    result = shgo(
        lambda g: sign * objective(g),
        bounds=bounds,
        n=int(n),
        iters=int(iters),
        sampling_method="sobol",
        options={"minimize_every_iter": True, "f_tol": 1e-12},
    )
    return ProfileResult(
        endpoint,
        sign * float(result.fun),
        float(result.x[0]),
        float(result.x[1]),
        bool(result.success),
        int(getattr(result, "nfev", -1)),
    )


def profile_pair(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float,
    n: int = 64,
    iters: int = 1,
) -> tuple[ProfileResult, ProfileResult]:
    return (
        solve_noaux_profile(alpha1, p0, p1, centers, rho, "lower", n, iters),
        solve_noaux_profile(alpha1, p0, p1, centers, rho, "upper", n, iters),
    )


def endpoint_inequality_system(
    alpha: np.ndarray,
    p1: np.ndarray,
    g: np.ndarray,
    tilde_q: np.ndarray,
    rho: float,
    sign: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Write min sign*ATE as min t subject to Mx>=c."""
    k_count = g.shape[1]
    n_q = 4 * k_count
    n_u = 4 * k_count
    n_x = 1 + n_q + n_u

    def qi(a: int, y: int, k: int) -> int:
        return 1 + (2 * a + y) * k_count + k

    def ui(a: int, y: int, k: int) -> int:
        return 1 + n_q + (2 * a + y) * k_count + k

    px = alpha[0] * g[0] + alpha[1] * g[1]
    beta = np.zeros(n_x)
    for k in range(k_count):
        beta[qi(1, 1, k)] = px[k] * p1[1] / max(g[1, k], TINY)
        beta[qi(0, 1, k)] = -px[k] * p1[0] / max(g[0, k], TINY)

    matrix: list[np.ndarray] = []
    rhs: list[float] = []
    row = np.zeros(n_x)
    row[0] = 1.0
    row -= sign * beta
    matrix.append(row)
    rhs.append(0.0)

    for a in range(2):
        for y in range(2):
            row = np.zeros(n_x)
            row[[qi(a, y, k) for k in range(k_count)]] = 1.0
            matrix.extend([row, -row])
            rhs.extend([1.0, -1.0])
        for k in range(k_count):
            row = np.zeros(n_x)
            row[qi(a, 0, k)] = 1.0 - p1[a]
            row[qi(a, 1, k)] = p1[a]
            matrix.extend([row, -row])
            rhs.extend([float(g[a, k]), float(-g[a, k])])

    for a in range(2):
        for y in range(2):
            for k in range(k_count):
                row = np.zeros(n_x)
                row[ui(a, y, k)] = 1.0
                row[qi(a, y, k)] = -1.0
                matrix.append(row)
                rhs.append(float(-tilde_q[a, y, k]))
                row = np.zeros(n_x)
                row[ui(a, y, k)] = 1.0
                row[qi(a, y, k)] = 1.0
                matrix.append(row)
                rhs.append(float(tilde_q[a, y, k]))
            row = np.zeros(n_x)
            row[[ui(a, y, k) for k in range(k_count)]] = -1.0
            matrix.append(row)
            rhs.append(float(-2.0 * rho))

    original_lower = np.r_[-1.0, np.zeros(n_q + n_u)]
    original_upper = np.r_[1.0, np.ones(n_q + n_u)]
    for index in range(n_x):
        row = np.zeros(n_x)
        row[index] = 1.0
        matrix.append(row)
        rhs.append(float(original_lower[index]))
        matrix.append(-row)
        rhs.append(float(-original_upper[index]))
    bounds = [(-1.25, 1.25)] + [(-0.25, 1.25)] * (n_q + n_u)
    return np.asarray(matrix), np.asarray(rhs), bounds


def solve_debiased_value(
    matrix: np.ndarray,
    rhs: np.ndarray,
    bounds_x: list[tuple[float, float]],
    w_n: float,
    face_tolerance: float = 1e-8,
) -> PenaltySolve:
    """Voronin-inspired exact-L1 penalty followed by max-t debiasing."""
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    q_constraints, n_x = matrix.shape
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= TINY):
        return PenaltySolve(*(np.nan,) * 7, False)
    normalized_m = matrix / norms[:, None]
    normalized_c = rhs / norms
    normalized_m[0] = matrix[0]
    normalized_c[0] = rhs[0]
    weights = np.full(q_constraints, w_n)
    weights[0] = 1.0
    p = np.r_[1.0, np.zeros(n_x - 1)]
    penalty_objective = np.r_[p, weights]
    a_ub = np.hstack([-normalized_m, -np.eye(q_constraints)])
    b_ub = -normalized_c
    bounds = bounds_x + [(0.0, None)] * q_constraints
    options = {
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
    }
    stage1 = linprog(
        penalty_objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options=options,
    )
    if not stage1.success:
        return PenaltySolve(*(np.nan,) * 7, False)
    stage2_objective = np.zeros(n_x + q_constraints)
    stage2_objective[0] = -1.0
    stage2 = linprog(
        stage2_objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=penalty_objective[None, :],
        b_eq=np.array([stage1.fun]),
        bounds=bounds,
        method="highs",
        options=options,
    )
    if not stage2.success:
        face_ub = np.vstack([a_ub, penalty_objective, -penalty_objective])
        face_rhs = np.r_[
            b_ub,
            stage1.fun + face_tolerance,
            -stage1.fun + face_tolerance,
        ]
        stage2 = linprog(
            stage2_objective,
            A_ub=face_ub,
            b_ub=face_rhs,
            bounds=bounds,
            method="highs",
            options=options,
        )
    if not stage2.success:
        return PenaltySolve(*(np.nan,) * 7, False)
    selected = np.asarray(stage2.x[:n_x], dtype=float)
    slacks = np.asarray(stage2.x[n_x:], dtype=float)
    return PenaltySolve(
        float(stage2.x[0]),
        float(stage1.x[0]),
        float(stage1.fun),
        float(slacks.sum()),
        float(slacks.max(initial=0.0)),
        float(penalty_objective @ stage2.x - stage1.fun),
        float(np.maximum(rhs - matrix @ selected, 0.0).max()),
        True,
    )


def penalty_level(n_main: int, n_aux: int | None, multiplier: float = 1.0) -> float:
    """Use the slower of the two sample sizes in Voronin's tuning sequence."""
    n_eff = n_main if n_aux is None else min(n_main, n_aux)
    if n_eff < 16:
        raise ValueError("effective sample size must be at least 16")
    anchor = math.log(math.log(30_000.0))
    return multiplier * 5.0 * math.log(math.log(float(n_eff))) / anchor


def face_tolerance(n_main: int, n_aux: int | None) -> float:
    n_eff = n_main if n_aux is None else min(n_main, n_aux)
    return min(1e-8, float(n_eff) ** (-0.75))


def penalty_pair(
    alpha1: float,
    p0: float,
    p1: float,
    centers: np.ndarray,
    g: np.ndarray,
    rho: float,
    n_main: int,
    n_aux: int | None,
    multiplier: float = 1.0,
) -> tuple[PenaltySolve, PenaltySolve, float]:
    alpha = np.array([1.0 - alpha1, alpha1])
    p = np.array([p0, p1])
    u1, v1, u0, v0 = centers
    tilde_q = np.array(
        [
            [[1.0 - v0, v0], [1.0 - u0, u0]],
            [[1.0 - v1, v1], [1.0 - u1, u1]],
        ]
    )
    w_n = penalty_level(n_main, n_aux, multiplier)
    tolerance = face_tolerance(n_main, n_aux)
    lower_system = endpoint_inequality_system(alpha, p, g, tilde_q, rho, 1.0)
    neg_upper_system = endpoint_inequality_system(alpha, p, g, tilde_q, rho, -1.0)
    return (
        solve_debiased_value(*lower_system, w_n, face_tolerance=tolerance),
        solve_debiased_value(*neg_upper_system, w_n, face_tolerance=tolerance),
        w_n,
    )


def raw_aux_pair(
    alpha1: float,
    p0: float,
    p1: float,
    centers: np.ndarray,
    g: np.ndarray,
    rho: float,
) -> tuple[float, float, bool]:
    try:
        result = solve_aux_binary(
            alpha1,
            p0,
            p1,
            centers,
            rho,
            (float(g[0, 1]), float(g[1, 1])),
        )
        return result.lower.value, result.upper.value, True
    except (ValueError, FloatingPointError):
        return np.nan, np.nan, False
