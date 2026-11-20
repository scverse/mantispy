"""Combine a gene's one-sided guide p-values into one, in closed form.

Shared by :func:`~mantispy.tl.aggregate_guides`, which calibrates the combined statistic against control guides, and
:func:`~mantispy.tl.bootstrap_features`, which reads its closed-form p-value.
"""

from __future__ import annotations

from math import erfc, exp, lgamma, log, log1p, sqrt

import numpy as np
from numba import njit

STOUFFER, FISHER, ORDER = 0, 1, 2
#: The rules by name, in the order of their codes.
METHODS = ("stouffer", "fisher", "order")
#: A guide's p-value is clipped to ``[P_CLIP, 1 - P_CLIP]``, so one guide at 0 or 1 cannot carry its gene alone.
P_CLIP = 1e-12

_SQRT2 = sqrt(2.0)
_SQRT2PI = sqrt(2.0 * np.pi)
# Acklam's rational approximation to the normal quantile, refined below by one Halley step.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01)
_A5 = 2.506628277459239e00
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
)
_C5 = 2.938163982698783e00
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)


@njit(cache=True, nogil=True)
def norm_sf(z: float) -> float:
    """``P(Z > z)`` for a standard normal ``Z``, with its digits kept far into the upper tail."""
    return 0.5 * erfc(z / _SQRT2)


@njit(cache=True, nogil=True)
def _lower_quantile(p: float) -> float:
    """The standard normal quantile of ``p`` at most one half, to rounding."""
    if p < 0.02425:
        q = sqrt(-2.0 * log(p))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C5
        x = num / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A5) * q
        x = num / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    e = 0.5 * erfc(-x / _SQRT2) - p
    u = e * _SQRT2PI * exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


@njit(cache=True, nogil=True)
def _z(p: float, q: float) -> float:
    """The z-score of a one-sided p-value ``p`` whose complement is ``q``, read from the smaller of the two."""
    if p <= q:
        return np.inf if p <= 0.0 else -_lower_quantile(p)
    return -np.inf if q <= 0.0 else _lower_quantile(q)


@njit(cache=True, nogil=True)
def guide_z(p: float, q: float) -> float:
    """The z-score of a guide's one-sided p-value ``p`` whose complement is ``q``, each clipped to ``P_CLIP``.

    The z-score is read from the smaller of the two, so a p-value near one keeps its digits.
    """
    return _z(max(p, P_CLIP), max(q, P_CLIP))


@njit(cache=True, nogil=True)
def combine(p: np.ndarray, z: np.ndarray, w: np.ndarray, k: int, method: int) -> float:
    """The combined statistic of ``k`` guides, larger for a stronger phenotype.

    ``"stouffer"`` is the weighted sum of the guides' z-scores ``z`` over the root of their summed squared weights ``w``.
    ``"fisher"`` is ``-2`` times the summed log of the one-sided p-values ``p``.
    ``"order"`` is the z-score of the chance that the ``ceil(k / 2)``-th smallest of ``k`` uniform p-values falls at
    or below the observed one.
    """
    if method == STOUFFER:
        num = den = 0.0
        for i in range(k):
            num += w[i] * z[i]
            den += w[i] * w[i]
        return num / sqrt(den)
    if method == FISHER:
        s = 0.0
        for i in range(k):
            s -= 2.0 * log(min(max(p[i], P_CLIP), 1.0 - P_CLIP))
        return s
    pv = np.sort(np.minimum(np.maximum(p[:k], P_CLIP), 1.0 - P_CLIP))
    r = (k + 1) // 2
    return _z(_binom_upper(r, k, pv[r - 1]), _binom_lower(r, k, pv[r - 1]))


@njit(cache=True, nogil=True)
def combined_pvalue(statistic: float, k: int, method: int) -> float:
    """The one-sided p-value of a :func:`combine` statistic of ``k`` independent uniform guides, in closed form."""
    if method == FISHER:  # chi-squared on 2k degrees of freedom: P(Poisson(s / 2) < k)
        h = 0.5 * statistic
        if h <= 0.0:
            return 1.0
        # the terms h^i e^-h / i! peak at i = h, so they are summed relative to the largest that they never overflow
        top = min(k - 1, int(h))
        log_top = top * log(h) - h - lgamma(top + 1.0)
        total = 0.0
        for i in range(k):
            total += exp(i * log(h) - h - lgamma(i + 1.0) - log_top)
        return min(exp(log_top) * total, 1.0)
    return norm_sf(statistic)


@njit(cache=True, nogil=True)
def _binom_term(i: int, n: int, x: float) -> float:
    return exp(lgamma(n + 1.0) - lgamma(i + 1.0) - lgamma(n - i + 1.0) + i * log(x) + (n - i) * log1p(-x))


@njit(cache=True, nogil=True)
def _binom_upper(r: int, n: int, x: float) -> float:
    """``P(Binomial(n, x) >= r)``, summed term by term so that a small one keeps its digits."""
    total = 0.0
    for i in range(r, n + 1):
        total += _binom_term(i, n, x)
    return min(total, 1.0)


@njit(cache=True, nogil=True)
def _binom_lower(r: int, n: int, x: float) -> float:
    """``P(Binomial(n, x) < r)``, the complement of :func:`_binom_upper`."""
    total = 0.0
    for i in range(r):
        total += _binom_term(i, n, x)
    return min(total, 1.0)


@njit(cache=True, nogil=True)
def combine_groups(p: np.ndarray, z: np.ndarray, w: np.ndarray, offsets: np.ndarray, method: int) -> np.ndarray:
    """:func:`combine` over each group of guides ``offsets[i]:offsets[i + 1]``."""
    n = offsets.size - 1
    out = np.empty(n)
    for i in range(n):
        a, b = offsets[i], offsets[i + 1]
        out[i] = combine(p[a:b], z[a:b], w[a:b], b - a, method)
    return out
