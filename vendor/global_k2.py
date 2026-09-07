"""Global benchmarks for the binary-X, no-auxiliary-data ATE problem.

The four optimization variables are ordered as

    z = (u1, v1, u0, v0),

where ``ua = P(X=1 | Y=1, A=a)`` and
``va = P(X=1 | Y=0, A=a)``.  For binary X, a total-variation ball is
just an interval in each of these four variables.

This module deliberately separates three notions:

* ``solve_local_binary`` returns a local nonlinear-programming solution;
* ``solve_noaux_shgo`` uses SciPy's SHGO global-search heuristic;
* ``certify_interval`` uses inclusion interval arithmetic and spatial
  branch-and-bound to enclose the true global optimum.

SHGO is useful for finding a strong incumbent, but only the last routine
returns an objective-value certificate.  The interval implementation uses
outward rounding with ``numpy.nextafter``.  It treats the supplied binary64
inputs as the exact problem data to be certified.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import minimize, shgo


Endpoint = Literal["lower", "upper"]
Box = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class EndpointResult:
    """A feasible endpoint value returned by a numerical optimizer."""

    endpoint: Endpoint
    value: float
    x: np.ndarray
    success: bool
    nfev: int
    message: str
    method: str


@dataclass(frozen=True)
class BoundPair:
    """Lower- and upper-endpoint results."""

    lower: EndpointResult
    upper: EndpointResult


@dataclass(frozen=True)
class IntervalCertificate:
    """Certified enclosure of one global endpoint.

    For ``endpoint='lower'``, the population minimum lies in
    ``[certified_lower, certified_upper]``.  The analogous statement holds
    for the population maximum when ``endpoint='upper'``.  ``value`` is the
    feasible incumbent (upper side for a minimum, lower side for a maximum).
    """

    endpoint: Endpoint
    value: float
    x: np.ndarray
    certified_lower: float
    certified_upper: float
    optimality_gap: float
    boxes_created: int
    boxes_processed: int
    active_boxes: int
    status: str


@dataclass(frozen=True)
class StressDGP:
    """Full-precision deliberately selected multimodal stress design."""

    alpha1: float
    p0: float
    p1: float
    centers: tuple[float, float, float, float]
    rho: float

    @property
    def clean_z(self) -> np.ndarray:
        return np.asarray(self.centers, dtype=float)

    @property
    def auxiliary_r(self) -> tuple[float, float]:
        u1, v1, u0, v0 = self.centers
        r1 = self.p1 * u1 + (1.0 - self.p1) * v1
        r0 = self.p0 * u0 + (1.0 - self.p0) * v0
        return r0, r1

    @property
    def true_ate(self) -> float:
        return ate_binary(self.clean_z, self.alpha1, self.p0, self.p1)


def stress_dgp() -> StressDGP:
    """Return the full-precision K=2 stress DGP used in the audit."""

    return StressDGP(
        alpha1=0.82130764966610537,
        p0=0.35607305146126755,
        p1=0.68244985849884232,
        centers=(
            0.70553947044971455,
            0.51064326507503210,
            0.34581955401962128,
            0.59177769083205933,
        ),
        rho=0.24963250543669208,
    )


def ate_binary(
    z: Sequence[float], alpha1: float, p0: float, p1: float
) -> float:
    """Evaluate the binary-X standardized ATE.

    Parameters
    ----------
    z
        ``(u1, v1, u0, v0)`` in the ordering documented above.
    alpha1
        ``P(A=1)``.
    p0, p1
        ``P(Y=1 | A=0)`` and ``P(Y=1 | A=1)``.
    """

    u1, v1, u0, v0 = np.asarray(z, dtype=float)
    r1 = p1 * u1 + (1.0 - p1) * v1
    r0 = p0 * u0 + (1.0 - p0) * v0
    if not (0.0 < r0 < 1.0 and 0.0 < r1 < 1.0):
        raise ValueError("ATE is undefined without positive cell overlap")

    # This expanded form makes clear which cross-arm ratios create the
    # nonconvexity.  It is algebraically identical to direct standardization.
    mu1 = p1 * (
        alpha1
        + (1.0 - alpha1)
        * (u1 * r0 / r1 + (1.0 - u1) * (1.0 - r0) / (1.0 - r1))
    )
    mu0 = p0 * (
        (1.0 - alpha1)
        + alpha1
        * (u0 * r1 / r0 + (1.0 - u0) * (1.0 - r1) / (1.0 - r0))
    )
    return float(mu1 - mu0)


def _as_centers(centers: Sequence[float]) -> np.ndarray:
    arr = np.asarray(centers, dtype=float)
    if arr.shape != (4,) or not np.all(np.isfinite(arr)):
        raise ValueError("centers must be four finite values in (u1,v1,u0,v0) order")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("centers must lie in [0,1]")
    return arr


def _as_radii(rho: float | Sequence[float]) -> np.ndarray:
    arr = np.asarray(rho, dtype=float)
    if arr.ndim == 0:
        out = np.repeat(float(arr), 4)
    elif arr.shape == (2,):
        # Treatment-arm ordering: (rho_0, rho_1).
        out = np.array([arr[1], arr[1], arr[0], arr[0]], dtype=float)
    elif arr.shape == (4,):
        out = arr.astype(float, copy=True)
    else:
        raise ValueError("rho must be scalar, (rho_0,rho_1), or four stratum radii")
    if np.any(~np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError("TV radii must be finite and nonnegative")
    return out


def variable_bounds(
    centers: Sequence[float], rho: float | Sequence[float]
) -> Box:
    """Return the four clipped binary-TV intervals."""

    c = _as_centers(centers)
    radii = _as_radii(rho)
    lo = np.maximum(0.0, c - radii)
    hi = np.minimum(1.0, c + radii)
    if np.any(lo > hi):
        raise ValueError("empty TV interval")
    return tuple((float(a), float(b)) for a, b in zip(lo, hi))


def _validate_probabilities(alpha1: float, p0: float, p1: float) -> None:
    if not (0.0 < alpha1 < 1.0):
        raise ValueError("alpha1 must lie strictly between zero and one")
    if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0):
        raise ValueError("p0 and p1 must lie strictly between zero and one")


def _validate_regular_box(box: Box, p0: float, p1: float) -> None:
    """Reject boxes on which a denominator can vanish."""

    (u1l, u1h), (v1l, v1h), (u0l, u0h), (v0l, v0h) = box
    r1l = p1 * u1l + (1.0 - p1) * v1l
    r1h = p1 * u1h + (1.0 - p1) * v1h
    r0l = p0 * u0l + (1.0 - p0) * v0l
    r0h = p0 * u0h + (1.0 - p0) * v0h
    if r0l <= 0.0 or r1l <= 0.0 or r0h >= 1.0 or r1h >= 1.0:
        raise ValueError(
            "the feasible box permits a zero denominator; impose quantitative "
            "overlap or use a singularity-aware formulation"
        )


def _endpoint_sign(endpoint: Endpoint) -> float:
    if endpoint == "lower":
        return 1.0
    if endpoint == "upper":
        return -1.0
    raise ValueError("endpoint must be 'lower' or 'upper'")


def solve_local_binary(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float | Sequence[float],
    endpoint: Endpoint = "lower",
    start: Sequence[float] | None = None,
    method: str = "L-BFGS-B",
) -> EndpointResult:
    """Solve one endpoint from one local start.

    A successful return certifies only local stationarity, not globality.
    """

    _validate_probabilities(alpha1, p0, p1)
    box = variable_bounds(centers, rho)
    _validate_regular_box(box, p0, p1)
    sign = _endpoint_sign(endpoint)
    if start is None:
        x0 = _as_centers(centers)
    else:
        x0 = np.asarray(start, dtype=float)
        if x0.shape != (4,):
            raise ValueError("start must contain four values")
    x0 = np.array([np.clip(x, lo, hi) for x, (lo, hi) in zip(x0, box)])

    result = minimize(
        lambda x: sign * ate_binary(x, alpha1, p0, p1),
        x0,
        bounds=box,
        method=method,
        options={"maxiter": 10_000, "ftol": 1e-14, "gtol": 1e-11},
    )
    x = np.asarray(result.x, dtype=float)
    return EndpointResult(
        endpoint=endpoint,
        value=ate_binary(x, alpha1, p0, p1),
        x=x,
        success=bool(result.success),
        nfev=int(getattr(result, "nfev", -1)),
        message=str(result.message),
        method=f"local:{method}",
    )


def _solve_one_shgo(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float | Sequence[float],
    endpoint: Endpoint,
    n: int,
    iters: int,
) -> EndpointResult:
    box = variable_bounds(centers, rho)
    _validate_regular_box(box, p0, p1)
    sign = _endpoint_sign(endpoint)
    result = shgo(
        lambda x: sign * ate_binary(x, alpha1, p0, p1),
        bounds=box,
        n=int(n),
        iters=int(iters),
        sampling_method="simplicial",
        options={"minimize_every_iter": True},
    )
    x = np.asarray(result.x, dtype=float)
    return EndpointResult(
        endpoint=endpoint,
        value=ate_binary(x, alpha1, p0, p1),
        x=x,
        success=bool(result.success),
        nfev=int(getattr(result, "nfev", -1)),
        message=str(result.message),
        method=f"SHGO:simplicial(n={n},iters={iters})",
    )


def solve_noaux_shgo(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float | Sequence[float],
    endpoint: Endpoint | None = None,
    n: int = 625,
    iters: int = 4,
) -> EndpointResult | BoundPair:
    """Find no-auxiliary endpoint incumbent(s) using simplicial SHGO.

    SHGO is a strong low-dimensional global-search method, but callers should
    use :func:`certify_interval` when a finite objective certificate is needed.
    """

    _validate_probabilities(alpha1, p0, p1)
    if endpoint is not None:
        return _solve_one_shgo(alpha1, p0, p1, centers, rho, endpoint, n, iters)
    lower = _solve_one_shgo(alpha1, p0, p1, centers, rho, "lower", n, iters)
    upper = _solve_one_shgo(alpha1, p0, p1, centers, rho, "upper", n, iters)
    return BoundPair(lower=lower, upper=upper)


def solve_aux_binary(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float | Sequence[float],
    auxiliary_r: tuple[float, float],
) -> BoundPair:
    """Solve the auxiliary-data K=2 LP analytically.

    ``auxiliary_r`` is ordered as ``(r0,r1)`` with
    ``ra=P(X=1|A=a)``.  The mixture equality fixes ``va`` as an affine
    function of ``ua``; the ATE is then affine in ``(u1,u0)``, so evaluating
    the four interval corners is exact.
    """

    _validate_probabilities(alpha1, p0, p1)
    box = variable_bounds(centers, rho)
    r0, r1 = map(float, auxiliary_r)
    if not (0.0 < r0 < 1.0 and 0.0 < r1 < 1.0):
        raise ValueError("auxiliary marginals must lie strictly between 0 and 1")

    def u_interval(
        s: float, r: float, u_bounds: tuple[float, float], v_bounds: tuple[float, float]
    ) -> tuple[float, float]:
        ulo, uhi = u_bounds
        vlo, vhi = v_bounds
        induced_lo = (r - (1.0 - s) * vhi) / s
        induced_hi = (r - (1.0 - s) * vlo) / s
        lo = max(ulo, induced_lo)
        hi = min(uhi, induced_hi)
        if lo > hi + 1e-13:
            raise ValueError("auxiliary marginal is incompatible with the TV balls")
        if lo > hi:  # absorb a sub-tolerance roundoff-only reversal
            lo = hi = (lo + hi) / 2.0
        return float(lo), float(hi)

    i1 = u_interval(p1, r1, box[0], box[1])
    i0 = u_interval(p0, r0, box[2], box[3])
    candidates: list[tuple[float, np.ndarray]] = []
    for u1, u0 in itertools.product(i1, i0):
        v1 = (r1 - p1 * u1) / (1.0 - p1)
        v0 = (r0 - p0 * u0) / (1.0 - p0)
        x = np.array([u1, v1, u0, v0], dtype=float)
        candidates.append((ate_binary(x, alpha1, p0, p1), x))

    lower_value, lower_x = min(candidates, key=lambda pair: pair[0])
    upper_value, upper_x = max(candidates, key=lambda pair: pair[0])
    lower = EndpointResult(
        endpoint="lower",
        value=lower_value,
        x=lower_x,
        success=True,
        nfev=4,
        message="exact affine corner solution",
        method="analytic auxiliary K=2 LP",
    )
    upper = EndpointResult(
        endpoint="upper",
        value=upper_value,
        x=upper_x,
        success=True,
        nfev=4,
        message="exact affine corner solution",
        method="analytic auxiliary K=2 LP",
    )
    return BoundPair(lower=lower, upper=upper)


# ---- Outward-rounded interval arithmetic ---------------------------------


@dataclass(frozen=True)
class _Interval:
    lo: float
    hi: float

    @staticmethod
    def point(x: float) -> "_Interval":
        return _Interval(float(x), float(x))


def _down(x: float) -> float:
    return float(np.nextafter(float(x), -np.inf))


def _up(x: float) -> float:
    return float(np.nextafter(float(x), np.inf))


def _iadd(a: _Interval, b: _Interval) -> _Interval:
    return _Interval(_down(a.lo + b.lo), _up(a.hi + b.hi))


def _ineg(a: _Interval) -> _Interval:
    return _Interval(_down(-a.hi), _up(-a.lo))


def _isub(a: _Interval, b: _Interval) -> _Interval:
    return _iadd(a, _ineg(b))


def _imul(a: _Interval, b: _Interval) -> _Interval:
    products = (
        a.lo * b.lo,
        a.lo * b.hi,
        a.hi * b.lo,
        a.hi * b.hi,
    )
    return _Interval(_down(min(products)), _up(max(products)))


def _ireciprocal(a: _Interval) -> _Interval:
    if a.lo <= 0.0 <= a.hi:
        raise ZeroDivisionError("interval denominator contains zero")
    endpoints = (1.0 / a.lo, 1.0 / a.hi)
    return _Interval(_down(min(endpoints)), _up(max(endpoints)))


def _idiv(a: _Interval, b: _Interval) -> _Interval:
    return _imul(a, _ireciprocal(b))


def _iscale(c: float, a: _Interval) -> _Interval:
    return _imul(_Interval.point(c), a)


def _ione_minus(a: _Interval) -> _Interval:
    return _isub(_Interval.point(1.0), a)


def _ate_interval(
    box: Box, alpha1: float, p0: float, p1: float
) -> _Interval:
    """Natural interval extension of :func:`ate_binary`."""

    u1, v1, u0, v0 = (_Interval(lo, hi) for lo, hi in box)
    r1 = _iadd(_iscale(p1, u1), _iscale(1.0 - p1, v1))
    r0 = _iadd(_iscale(p0, u0), _iscale(1.0 - p0, v0))

    treated_cross = _iadd(
        _idiv(_imul(u1, r0), r1),
        _idiv(_imul(_ione_minus(u1), _ione_minus(r0)), _ione_minus(r1)),
    )
    control_cross = _iadd(
        _idiv(_imul(u0, r1), r0),
        _idiv(_imul(_ione_minus(u0), _ione_minus(r1)), _ione_minus(r0)),
    )

    mu1 = _iscale(
        p1,
        _iadd(
            _Interval.point(alpha1),
            _iscale(1.0 - alpha1, treated_cross),
        ),
    )
    mu0 = _iscale(
        p0,
        _iadd(
            _Interval.point(1.0 - alpha1),
            _iscale(alpha1, control_cross),
        ),
    )
    return _isub(mu1, mu0)


def _transformed_interval(value: _Interval, endpoint: Endpoint) -> _Interval:
    return value if endpoint == "lower" else _ineg(value)


def _initial_incumbent(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float | Sequence[float],
    endpoint: Endpoint,
    incumbent_x: Sequence[float] | None,
) -> tuple[float, np.ndarray]:
    """Build a strong feasible incumbent without assuming globality."""

    box = variable_bounds(centers, rho)
    sign = _endpoint_sign(endpoint)
    starts: list[np.ndarray] = [_as_centers(centers)]
    starts.extend(
        np.array([box[j][bit] for j, bit in enumerate(bits)], dtype=float)
        for bits in itertools.product((0, 1), repeat=4)
    )
    if incumbent_x is not None:
        starts.append(np.asarray(incumbent_x, dtype=float))

    best = math.inf
    best_x: np.ndarray | None = None
    for start in starts:
        result = solve_local_binary(
            alpha1,
            p0,
            p1,
            centers,
            rho,
            endpoint=endpoint,
            start=start,
        )
        transformed = sign * result.value
        if transformed < best:
            best = transformed
            best_x = result.x
    if best_x is None:
        raise RuntimeError("failed to construct a feasible incumbent")
    return best, best_x


def certify_interval(
    alpha1: float,
    p0: float,
    p1: float,
    centers: Sequence[float],
    rho: float | Sequence[float],
    endpoint: Endpoint = "lower",
    abs_tol: float = 1e-6,
    max_boxes: int = 2_000_000,
    incumbent_x: Sequence[float] | None = None,
) -> IntervalCertificate:
    """Certify one global endpoint by interval branch-and-bound.

    The method minimizes either ``ATE`` or ``-ATE``.  A box is discarded only
    when its outward-rounded interval lower bound is no better than the
    incumbent up to ``abs_tol``.  If ``max_boxes`` is reached, the returned
    bracket remains valid but may be wider than ``abs_tol``.

    This routine is intended for K=2 audits.  Natural interval bounds converge
    under subdivision, but their cost grows exponentially and they should not
    be mistaken for a scalable K=24 algorithm.
    """

    _validate_probabilities(alpha1, p0, p1)
    if not (abs_tol > 0.0 and np.isfinite(abs_tol)):
        raise ValueError("abs_tol must be finite and positive")
    if max_boxes < 1:
        raise ValueError("max_boxes must be positive")
    root = variable_bounds(centers, rho)
    _validate_regular_box(root, p0, p1)
    sign = _endpoint_sign(endpoint)
    incumbent, best_x = _initial_incumbent(
        alpha1, p0, p1, centers, rho, endpoint, incumbent_x
    )

    root_interval = _transformed_interval(
        _ate_interval(root, alpha1, p0, p1), endpoint
    )
    heap: list[tuple[float, int, Box]] = [(root_interval.lo, 0, root)]
    counter = 1
    processed = 0
    status = "certified"

    while heap:
        lower, _, box = heapq.heappop(heap)
        processed += 1
        if lower >= incumbent - abs_tol:
            continue
        if counter >= max_boxes:
            heapq.heappush(heap, (lower, counter, box))
            status = "max_boxes_reached"
            break

        midpoint = np.array([(lo + hi) / 2.0 for lo, hi in box], dtype=float)
        midpoint_value = sign * ate_binary(midpoint, alpha1, p0, p1)
        if midpoint_value < incumbent:
            incumbent = midpoint_value
            best_x = midpoint

        widths = np.array([hi - lo for lo, hi in box])
        split_dim = int(np.argmax(widths))
        lo, hi = box[split_dim]
        mid = (lo + hi) / 2.0
        for child_interval in ((lo, mid), (mid, hi)):
            child_list = list(box)
            child_list[split_dim] = child_interval
            child = tuple(child_list)
            enclosure = _transformed_interval(
                _ate_interval(child, alpha1, p0, p1), endpoint
            )
            if enclosure.lo < incumbent - abs_tol:
                heapq.heappush(heap, (enclosure.lo, counter, child))
                counter += 1

    # Every discarded box has transformed lower bound at least
    # final_incumbent-abs_tol.  If active boxes remain after a resource stop,
    # their smallest lower bound must also enter the global lower certificate.
    transformed_lower = incumbent - abs_tol
    if heap:
        transformed_lower = min(transformed_lower, heap[0][0])
    transformed_gap = max(0.0, incumbent - transformed_lower)

    if endpoint == "lower":
        certified_lower = transformed_lower
        certified_upper = incumbent
        value = incumbent
    else:
        certified_lower = -incumbent
        certified_upper = -transformed_lower
        value = -incumbent

    return IntervalCertificate(
        endpoint=endpoint,
        value=float(value),
        x=np.asarray(best_x, dtype=float),
        certified_lower=float(certified_lower),
        certified_upper=float(certified_upper),
        optimality_gap=float(transformed_gap),
        boxes_created=int(counter),
        boxes_processed=int(processed),
        active_boxes=len(heap),
        status=status,
    )
