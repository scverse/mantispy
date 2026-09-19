"""ECOD: empirical-cumulative-distribution outlier detection :cite:p:`Li_2023`.

A row's score is the sum, over features, of how far into a tail its value sits.
Reimplemented here instead of depending on pyod, and checked against pyod's formulation by an equivalence test.
Columns stream through a numba kernel, so memory grows with the number of rows and not with rows times features.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from mantispy._core._numba import prange

AGGREGATIONS = ("pyod", "paper")
#: Fixed rather than the thread count, so the summation order, and with it every tie at a cutoff, is the same on any machine.
_CHUNKS = 16
_EPS = float(np.finfo(np.float64).eps)


@njit(cache=True, nogil=True)
def _skew_sign(column: np.ndarray) -> int:
    """Sign of the column's skewness, 0 where scipy's ``skew`` is zero or undefined."""
    mean = column.mean()
    m2 = m3 = 0.0
    for value in column:
        deviation = value - mean
        m2 += deviation * deviation
        m3 += deviation * deviation * deviation
    if m2 / column.size <= (_EPS * mean) ** 2:
        return 0
    return 1 if m3 > 0 else -1 if m3 < 0 else 0


@njit(parallel=True, cache=True, nogil=True)
def _ecod(X: np.ndarray, paper: bool) -> np.ndarray:
    """Per-row tail sums: one row for ``"pyod"``, the left, right and skew-directed rows for ``"paper"``.

    Chunk ``c`` owns every ``_CHUNKS``-th column and an accumulator of its own, so threads never write to the same row.
    """
    n_obs, n_vars = X.shape
    sums = np.zeros((_CHUNKS, 3 if paper else 1, n_obs))
    log_n = np.log(n_obs)
    for chunk in prange(_CHUNKS):
        column = np.empty(n_obs)
        for j in range(chunk, n_vars, _CHUNKS):
            total, count = 0.0, 0
            for i in range(n_obs):
                column[i] = X[i, j]
                if np.isfinite(column[i]):
                    total += column[i]
                    count += 1
            # A missing value has no tail probability of its own, so it takes the mean of the finite values.
            for i in range(n_obs):
                if np.isnan(column[i]):
                    column[i] = total / count if count else 0.0

            sign = _skew_sign(column)
            order = np.argsort(column)
            start = 0
            while start < n_obs:
                # Equal values share a rank: <= counts to the end of their run, < to its start.
                stop = start + 1
                while stop < n_obs and column[order[stop]] == column[order[start]]:
                    stop += 1
                left, right = log_n - np.log(stop), log_n - np.log(n_obs - start)
                for row in order[start:stop]:
                    if paper:
                        sums[chunk, 0, row] += left
                        sums[chunk, 1, row] += right
                        sums[chunk, 2, row] += left if sign < 0 else right
                    else:
                        # pyod's skew term adds both tails where the skewness is zero or undefined.
                        sums[chunk, 0, row] += max(left, right) if sign else left + right
                start = stop
    return sums.sum(axis=0)


def ecod_scores(X: np.ndarray, aggregation: str = "pyod") -> np.ndarray:
    """ECOD outlier score per row of ``X``, higher meaning more outlying, aggregated as one of :data:`AGGREGATIONS`."""
    X = np.asarray(X)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float64)
    if not len(X):
        return np.zeros(0)
    return _ecod(X, aggregation == "paper").max(axis=0)
