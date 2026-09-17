"""NaN-skipping grouped reduction kernels.

Two design choices matter at JUMP scale:

* The row permutation is passed as an index array and applied inside the kernel.
  Materializing ``X[order]`` would copy the whole matrix for every statistic, twice for ``mad_robustize`` and four times for ``robustize``.
* Parallelism is over groups, not over (group, feature) pairs, so the per-column scratch buffer is allocated once per group instead of once per output cell.
  Allocating inside a hot ``prange`` body serializes the loop on numba's allocator lock.

Input stays ``float32``; accumulation is in ``float64`` scalars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    # numba's prange iterates like range, but its own type says nothing about that.
    from builtins import range as prange
else:
    from numba import prange

# Statistic selectors, kept as ints so one kernel serves all of them.
MEAN, MEDIAN, MAD, STD, QUANTILE = 0, 1, 2, 3, 4


def group_offsets(codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable row ordering by group, plus the start/stop offsets of each group."""
    codes = np.ascontiguousarray(codes, dtype=np.int32)
    order = np.argsort(codes, kind="stable").astype(np.int64)
    offsets = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(np.bincount(codes, minlength=n_groups), out=offsets[1:])
    return order, offsets


def group_counts(codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Number of rows per group."""
    return np.bincount(np.asarray(codes), minlength=n_groups).astype(np.int64)


@njit(cache=True, nogil=True)
def _median_of(buffer: np.ndarray, n: int) -> float:
    if n == 0:
        return np.nan
    values = np.sort(buffer[:n])
    middle = n // 2
    if n % 2 == 1:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


@njit(cache=True, nogil=True)
def _quantile_of(buffer: np.ndarray, n: int, q: float) -> float:
    if n == 0:
        return np.nan
    values = np.sort(buffer[:n])
    position = q * (n - 1)
    low = int(np.floor(position))
    high = int(np.ceil(position))
    if low == high:
        return values[low]
    return values[low] + (position - low) * (values[high] - values[low])


@njit(parallel=True, cache=True, nogil=True)
def _grouped(
    X: np.ndarray, order: np.ndarray, offsets: np.ndarray, n_groups: int, stat: int, q: float, ddof: int
) -> np.ndarray:
    n_vars = X.shape[1]
    out = np.empty((n_groups, n_vars), dtype=np.float64)
    longest = 0
    for group in range(n_groups):
        length = offsets[group + 1] - offsets[group]
        if length > longest:
            longest = length

    for group in prange(n_groups):
        start, stop = offsets[group], offsets[group + 1]
        buffer = np.empty(longest, dtype=np.float64)
        for j in range(n_vars):
            n = 0
            total = 0.0
            for i in range(start, stop):
                value = X[order[i], j]
                if not np.isnan(value):
                    buffer[n] = value
                    total += value
                    n += 1
            if n == 0:
                out[group, j] = np.nan
            elif stat == 0:  # mean
                out[group, j] = total / n
            elif stat == 1:  # median
                out[group, j] = _median_of(buffer, n)
            elif stat == 2:  # median absolute deviation, unscaled
                centre = _median_of(buffer, n)
                for k in range(n):
                    buffer[k] = abs(buffer[k] - centre)
                out[group, j] = _median_of(buffer, n)
            elif stat == 3:  # standard deviation
                if n - ddof <= 0:
                    out[group, j] = np.nan
                else:
                    mean = total / n
                    squares = 0.0
                    for k in range(n):
                        squares += (buffer[k] - mean) ** 2
                    out[group, j] = np.sqrt(squares / (n - ddof))
            else:  # quantile
                out[group, j] = _quantile_of(buffer, n, q)
    return out


def grouped_stat(
    X: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    stat: int,
    q: float = 0.5,
    ddof: int = 1,
) -> np.ndarray:
    """Per-group NaN-skipping statistic, shaped ``(n_groups, n_vars)``.

    Empty groups and all-NaN columns yield ``NaN`` rather than raising.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    order, offsets = group_offsets(codes, n_groups)
    return _grouped(X, order, offsets, n_groups, stat, float(q), int(ddof))


@njit(cache=True, nogil=True)
def _count_below(column: np.ndarray, stop: int, value: float) -> tuple[int, int]:
    """``(#values < value, #values == value)`` in ``column[:stop]``, which is sorted."""
    low, high = 0, stop
    while low < high:  # first index not less than value
        middle = (low + high) // 2
        if column[middle] < value:
            low = middle + 1
        else:
            high = middle
    less = low
    high = stop
    while low < high:  # first index greater than value
        middle = (low + high) // 2
        if column[middle] <= value:
            low = middle + 1
        else:
            high = middle
    return less, low - less


@njit(parallel=True, cache=True, nogil=True)
def _mwu_moments(
    treated: np.ndarray, control: np.ndarray, n_control: np.ndarray, control_ties: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mann-Whitney ``U1``, the pooled tie term and the finite count, per feature.

    ``treated`` and ``control`` are ``(n_vars, n_obs)``, feature-major so each row is contiguous for the binary search.
    ``control`` is sorted ascending with missing values last, ``n_control`` is how many of each row were measured, and ``control_ties`` is that row's ``sum(c ** 3 - c)`` over its runs of equal values.

    The control is the same for every group, but ``scipy.stats.mannwhitneyu`` called per group re-ranks it each time.
    Ranked once and searched into, each group costs its own size rather than the reference's.
    """
    n_vars, n_treated = treated.shape
    statistic = np.empty(n_vars, dtype=np.float64)
    tie_term = np.empty(n_vars, dtype=np.float64)
    counts = np.empty(n_vars, dtype=np.int64)

    for j in prange(n_vars):
        buffer = np.empty(n_treated, dtype=np.float64)
        size = 0
        for i in range(n_treated):
            value = treated[j, i]
            if not np.isnan(value):
                buffer[size] = value
                size += 1
        values = np.sort(buffer[:size])

        rank_sum = 0.0
        ties = control_ties[j]
        start = 0
        while start < size:
            stop = start + 1
            while stop < size and values[stop] == values[start]:
                stop += 1
            repeats = stop - start
            fewer, equal = _count_below(control[j], n_control[j], values[start])
            # Average rank of this value in the pooled sample, times how many share it.
            rank_sum += repeats * (fewer + start + (equal + repeats + 1) / 2.0)
            pooled = equal + repeats
            ties += (pooled**3 - pooled) - (equal**3 - equal)
            start = stop

        statistic[j] = rank_sum - size * (size + 1) / 2.0
        tie_term[j] = ties
        counts[j] = size
    return statistic, tie_term, counts


@njit(parallel=True, cache=True, nogil=True)
def _polish_planes(planes: np.ndarray, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Tukey median polish of every ``(rows, columns)`` plane of ``planes``.

    ``planes`` is ``(n_features, n_rows, n_columns)``, feature-major so each plane is contiguous and handled by one thread.
    Returns the fitted row and column effects with the grand level held out of both, as ``(n_features, n_rows)`` and ``(n_features, n_columns)``.

    The arithmetic matches ``np.median`` over a stacked array.
    The medians here are over 16 and 24 values, where numpy's per-slice dispatch costs more than the median.
    """
    n_features, n_rows, n_columns = planes.shape
    row_out = np.zeros((n_features, n_rows), dtype=np.float64)
    column_out = np.zeros((n_features, n_columns), dtype=np.float64)

    for feature in prange(n_features):
        residual = planes[feature].copy()
        rows = np.zeros(n_rows, dtype=np.float64)
        columns = np.zeros(n_columns, dtype=np.float64)
        buffer = np.empty(max(n_rows, n_columns), dtype=np.float64)
        previous = np.inf

        for _ in range(max_iter):
            for r in range(n_rows):
                n = 0
                for c in range(n_columns):
                    value = residual[r, c]
                    if not np.isnan(value):
                        buffer[n] = value
                        n += 1
                middle = _median_of(buffer, n)
                if np.isnan(middle):  # a row of the plate with nothing measured
                    middle = 0.0
                for c in range(n_columns):
                    residual[r, c] -= middle
                rows[r] += middle
            for r in range(n_rows):
                buffer[r] = rows[r]
            level = _median_of(buffer, n_rows)
            for r in range(n_rows):
                rows[r] -= level

            for c in range(n_columns):
                n = 0
                for r in range(n_rows):
                    value = residual[r, c]
                    if not np.isnan(value):
                        buffer[n] = value
                        n += 1
                middle = _median_of(buffer, n)
                if np.isnan(middle):
                    middle = 0.0
                for r in range(n_rows):
                    residual[r, c] -= middle
                columns[c] += middle
            for c in range(n_columns):
                buffer[c] = columns[c]
            level = _median_of(buffer, n_columns)
            for c in range(n_columns):
                columns[c] -= level

            current = 0.0
            for r in range(n_rows):
                for c in range(n_columns):
                    value = residual[r, c]
                    if not np.isnan(value):
                        current += abs(value)
            if abs(previous - current) < tol:
                break
            previous = current

        for r in range(n_rows):
            row_out[feature, r] = rows[r]
        for c in range(n_columns):
            column_out[feature, c] = columns[c]
    return row_out, column_out


@njit(parallel=True, cache=True, nogil=True)
def _wasserstein_against(treated: np.ndarray, control: np.ndarray, n_control: np.ndarray) -> np.ndarray:
    """Wasserstein-1 distance per feature, against a reference sorted once.

    ``W1`` is the integral of ``|F_treated - F_control|`` over the merged support, computed in one walk through both sorted samples.
    A per-group ``argsort`` of the concatenation re-sorts the reference each time, which dominates the cost when the reference is the eight thousand control wells of a screen.

    Args:
        treated: ``(n_vars, n_treated)`` float64 sample, feature-major so each row is contiguous, and holding no missing value: every entry of a row counts towards that feature's treated CDF.
        control: ``(n_vars, n_obs)`` float64 reference, feature-major, each row sorted ascending over its first ``n_control`` entries, as :func:`~mantispy._core._stats.sorted_control` returns it.
        n_control: How many entries of each row of `control` were measured, so that the unmeasured tail a row was padded with is left out of the reference CDF.

    Returns:
        ``(n_vars,)`` float64, the distance between the two samples of each feature in that feature's own units, and ``NaN`` for a feature with no treated rows or no measured reference value.
    """
    n_vars, n_treated = treated.shape
    out = np.empty(n_vars, dtype=np.float64)

    for j in prange(n_vars):
        values = np.sort(treated[j])
        m = n_control[j]
        if n_treated == 0 or m == 0:
            out[j] = np.nan
            continue

        total = 0.0
        i = 0  # into control
        k = 0  # into treated
        seen_treated = 0
        seen_control = 0
        previous = min(control[j, 0], values[0])
        while i < m or k < n_treated:
            # Ties go to the treated sample, matching a stable sort of the concatenation
            # with the treated block first. The gap is zero either way.
            if k < n_treated and (i >= m or values[k] <= control[j, i]):
                current = values[k]
                k += 1
                from_treated = True
            else:
                current = control[j, i]
                i += 1
                from_treated = False
            # The step from the previous point to this one is weighted by the two CDFs
            # as they stood at the previous point, so the counts advance afterwards.
            total += abs(seen_treated / n_treated - seen_control / m) * (current - previous)
            if from_treated:
                seen_treated += 1
            else:
                seen_control += 1
            previous = current
        out[j] = total
    return out
