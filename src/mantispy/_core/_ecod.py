"""ECOD: empirical-cumulative-distribution outlier detection :cite:p:`Li_2023`.

Parameter-free and interpretable: a row's score is the sum, over features, of how far into a tail its value sits.
Reimplemented here instead of depending on pyod, and checked against pyod's formulation by an equivalence test.

The score is the sum over dimensions of the elementwise maximum of the three tail matrices, as pyod computes it.
Algorithm 1 of :cite:t:`Li_2023` takes the maximum of the three per-row sums instead, which gives different scores.
"""

from __future__ import annotations

import numpy as np


def column_ecdf(X: np.ndarray) -> np.ndarray:
    """Column-wise empirical CDF, with ties taking the highest rank.

    Counts values less than or equal to each entry, which is the tie handling pyod's explicit backwards pass produces.
    """
    return _tail_counts(X)[0] / X.shape[0]


def _tail_counts(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(#values <= x, #values < x)`` per column, from a single sort.

    ECOD needs the empirical CDF of both ``X`` and ``-X``.
    Since ``#(-X <= -x)`` is ``n - #(X < x)``, both counts come from one ``argsort`` and the run structure of the sorted column.
    """
    n_obs = X.shape[0]
    order = np.argsort(X, axis=0, kind="stable")
    ordered = np.take_along_axis(X, order, axis=0)

    ends = np.empty(X.shape, dtype=bool)  # last member of a run of equal values
    ends[:-1] = ordered[:-1] != ordered[1:]
    ends[-1] = True
    starts = np.empty(X.shape, dtype=bool)
    starts[0] = True
    starts[1:] = ends[:-1]

    position = np.arange(1, n_obs + 1, dtype=np.float64)[:, None]
    # The end of each entry's run is how many values are <= it; the start, less one, how
    # many are strictly below.
    highest = np.minimum.accumulate(np.where(ends, position, n_obs + 1)[::-1], axis=0)[::-1]
    lowest = np.maximum.accumulate(np.where(starts, position, 0.0), axis=0) - 1

    at_or_below, below = np.empty(X.shape), np.empty(X.shape)
    np.put_along_axis(at_or_below, order, highest, axis=0)
    np.put_along_axis(below, order, lowest, axis=0)
    return at_or_below, below


def ecod_scores(X: np.ndarray) -> np.ndarray:
    """ECOD outlier score per row of ``X``. Higher is more outlying.

    NaN is imputed with the column mean first, since a missing value has no tail probability of its own.
    """
    from scipy.stats import skew as _skew

    X = np.asarray(X, dtype=np.float64)
    missing = np.isnan(X)
    if missing.any():
        column_means = np.nanmean(np.where(missing, np.nan, X), axis=0)
        X = np.where(missing, np.nan_to_num(column_means, nan=0.0), X)

    at_or_below, below = _tail_counts(X)
    left = -np.log(at_or_below / X.shape[0])
    right = -np.log((X.shape[0] - below) / X.shape[0])

    # skewness in {-1, 0, 1}: negative picks the left tail, positive the right, and a
    # feature with zero skew contributes both.
    skewness = np.sign(np.nan_to_num(_skew(X, axis=0)))
    tail = left * -1 * np.sign(skewness - 1) + right * np.sign(skewness + 1)

    combined = np.maximum(np.maximum(left, right), tail)
    return combined.sum(axis=1)
