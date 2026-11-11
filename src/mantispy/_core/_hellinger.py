"""Hellinger distance between two Gaussian fits, the mp-value statistic, and their permutation null.

This is BioProfiling.jl's robust Hellinger distance when each side is fitted with the minimum covariance determinant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from mantispy._core._mcd import _moments, mcd

if TYPE_CHECKING:
    from builtins import range as prange
else:
    from numba import prange


@njit(cache=True, nogil=True)
def _pooled(centre_a: np.ndarray, cov_a: np.ndarray, centre_b: np.ndarray, cov_b: np.ndarray) -> tuple[float, float]:
    """Log-determinant of ``S = (A + B) / 2`` and the squared Mahalanobis distance between the centres in it; ``NaN`` when a fit failed or ``S`` is not positive definite."""
    if not (
        np.isfinite(cov_a).all()
        and np.isfinite(cov_b).all()
        and np.isfinite(centre_a).all()
        and np.isfinite(centre_b).all()
    ):
        return np.nan, np.nan
    pooled = (cov_a + cov_b) / 2
    sign, log_pooled = np.linalg.slogdet(pooled)
    if sign <= 0:
        return np.nan, np.nan
    shift = centre_a - centre_b
    return log_pooled, shift @ np.linalg.solve(pooled, shift)


@njit(cache=True, nogil=True)
def hellinger(centre_a: np.ndarray, cov_a: np.ndarray, centre_b: np.ndarray, cov_b: np.ndarray) -> float:
    """Squared Hellinger distance between two Gaussians, ``NaN`` when a fit failed or a covariance is not positive definite.

    ``1 - det(A)^(1/4) det(B)^(1/4) / det(S)^(1/2) * exp(-(m_a - m_b)' S^-1 (m_a - m_b) / 8)`` with ``S = (A + B) / 2``,
    in log-determinants so that it does not underflow in more than a few dimensions.
    It is zero for identical Gaussians and approaches one as they stop overlapping, through either the means or the covariances.
    """
    log_pooled, quadratic = _pooled(centre_a, cov_a, centre_b, cov_b)
    if np.isnan(quadratic):
        return np.nan
    sign_a, log_a = np.linalg.slogdet(cov_a)
    sign_b, log_b = np.linalg.slogdet(cov_b)
    if sign_a <= 0 or sign_b <= 0:
        return np.nan
    return 1.0 - np.exp(0.25 * log_a + 0.25 * log_b - 0.5 * log_pooled - quadratic / 8)


@njit(cache=True, nogil=True)
def mp_value(centre_a: np.ndarray, cov_a: np.ndarray, centre_b: np.ndarray, cov_b: np.ndarray) -> float:
    """Squared Mahalanobis distance between the centres in ``S = (A + B) / 2``, the statistic of the mp-value.

    Only ``S`` must be positive definite, so either side may have fewer rows than features.
    """
    return _pooled(centre_a, cov_a, centre_b, cov_b)[1]


@njit(cache=True, nogil=True)
def _distance(centre_a, cov_a, centre_b, cov_b, mp):
    if mp:
        return mp_value(centre_a, cov_a, centre_b, cov_b)
    return hellinger(centre_a, cov_a, centre_b, cov_b)


@njit(cache=True, nogil=True)
def _fit(x, rows, robust, h, raw, rew, exhaustive, cutoff, n_samp, seed):
    if robust:
        k = rows.size
        return mcd(np.ascontiguousarray(x[rows]), h[k], raw[k], rew[k], cutoff, exhaustive[k], n_samp, seed)
    return _moments(x, rows)


@njit(cache=True, nogil=True)
def statistic(x, rows_a, rows_b, mp, robust, h, raw, rew, exhaustive, cutoff, n_samp, seed, min_rows) -> float:
    """Hellinger distance, or the mp-value statistic when ``mp``, between the fits to two row sets of ``x``; ``NaN`` when either has fewer than ``min_rows``."""
    if rows_a.size < min_rows or rows_b.size < min_rows:
        return np.nan
    centre_a, cov_a = _fit(x, rows_a, robust, h, raw, rew, exhaustive, cutoff, n_samp, seed)
    centre_b, cov_b = _fit(x, rows_b, robust, h, raw, rew, exhaustive, cutoff, n_samp, seed)
    return _distance(centre_a, cov_a, centre_b, cov_b, mp)


@njit(cache=True, nogil=True)
def _from_sums(k, total, cross):
    centre = total / k
    return centre, (cross - k * np.outer(centre, centre)) / (k - 1)


@njit(cache=True, parallel=True)
def null(x, units, draws, mp, robust, h, raw, rew, exhaustive, cutoff, n_samp, seed, min_rows) -> np.ndarray:
    """Statistic for each row of ``draws``: the rows of ``x`` whose unit was drawn against the rest.

    ``units`` labels each row of ``x`` with a unit numbered from zero (a replicate, or the row itself for a row-level null).
    The empirical fit sums the drawn rows' moments and takes the rest's as the pool's minus those, so a draw costs the rows it holds rather than the pool; the robust fit refits the MCD on both sides.
    """
    n_units = units.max() + 1
    out = np.empty(draws.shape[0])
    if robust:
        for b in prange(draws.shape[0]):
            drawn = np.zeros(n_units, dtype=np.bool_)
            drawn[draws[b]] = True
            inside = drawn[units]
            out[b] = statistic(
                x,
                np.flatnonzero(inside),
                np.flatnonzero(~inside),
                mp,
                robust,
                h,
                raw,
                rew,
                exhaustive,
                cutoff,
                n_samp,
                seed,
                min_rows,
            )
        return out
    n, p = x.shape
    # Centring first keeps the sums of squares from cancelling.
    centred = x - x.sum(axis=0) / n
    all_sums = centred.sum(axis=0)
    all_cross = centred.T @ centred
    order = np.argsort(units, kind="mergesort")
    offsets = np.zeros(n_units + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(units, minlength=n_units))
    for b in prange(draws.shape[0]):
        k = 0
        total = np.zeros(p)
        products = np.zeros((p, p))
        for u in draws[b]:
            for j in range(offsets[u], offsets[u + 1]):
                i = order[j]
                k += 1
                for a in range(p):
                    total[a] += centred[i, a]
                    for c in range(a + 1):
                        products[a, c] += centred[i, a] * centred[i, c]
        if k < min_rows or n - k < min_rows:
            out[b] = np.nan
            continue
        for a in range(p):
            for c in range(a):
                products[c, a] = products[a, c]
        centre_a, cov_a = _from_sums(k, total, products)
        centre_b, cov_b = _from_sums(n - k, all_sums - total, all_cross - products)
        out[b] = _distance(centre_a, cov_a, centre_b, cov_b, mp)
    return out
