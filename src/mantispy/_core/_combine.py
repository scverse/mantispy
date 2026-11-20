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


@njit(cache=True, nogil=True)
def combine(p: np.ndarray, z: np.ndarray, w: np.ndarray, k: int, method: int) -> float:
    """The combined statistic of ``k`` guides, larger for a stronger phenotype.

    ``"stouffer"`` is the weighted sum of the guides' z-scores ``z`` over the root of their summed squared weights ``w``.
    ``"fisher"`` is ``-2`` times the summed log of the one-sided p-values ``p``.
    ``"order"`` is minus the log of the chance that the ``k // 2 + 1``-th smallest of ``k`` uniform p-values, a strict
    majority of them, falls at or below the observed one.
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
    r = k // 2 + 1
    return -log(_binom_upper(r, k, pv[r - 1]))


@njit(cache=True, nogil=True)
def combined_pvalue(statistic: float, k: int, method: int) -> float:
    """The one-sided p-value of a :func:`combine` statistic of ``k`` independent uniform guides, in closed form."""
    if method == STOUFFER:
        return 0.5 * erfc(statistic / _SQRT2)
    if method == ORDER:
        return exp(-statistic)
    # Fisher, chi-squared on 2k degrees of freedom: P(Poisson(s / 2) < k)
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


@njit(cache=True, nogil=True)
def _binom_pmf(k: int, n: int, p: float) -> float:
    """``P(Binomial(n, p) == k)`` for ``0 < p < 1``, in log space so it does not underflow for large ``n``."""
    return exp(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0) + k * log(p) + (n - k) * log1p(-p))


@njit(cache=True, nogil=True)
def _binom_upper(r: int, n: int, x: float) -> float:
    """``P(Binomial(n, x) >= r)``, summed term by term so that a small one keeps its digits."""
    total = 0.0
    for i in range(r, n + 1):
        total += _binom_pmf(i, n, x)
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
