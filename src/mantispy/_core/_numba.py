"""NaN-skipping grouped reduction kernels.

The row permutation is passed as an index array and applied inside the kernel, never materialized as ``X[order]``.
Scratch buffers are allocated once per group, because allocating inside a ``prange`` body serializes the loop on numba's allocator lock.
Medians and quantiles are read with an in-place quickselect rather than a full sort; it lands the same order statistics, so the result is bit-identical.
Input stays ``float32`` or ``float64`` as given, and any other dtype is read as ``float64``; accumulation is in ``float64`` scalars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import get_num_threads, njit

if TYPE_CHECKING:
    # numba's prange iterates like range, but its own type says nothing about that.
    from builtins import range as prange
else:
    from numba import prange

MEAN, MEDIAN, MAD, STD, QUANTILE = 0, 1, 2, 3, 4
#: Spread selector for :func:`grouped_median_spread` only, alongside :data:`MAD`.
IQR = 5


@njit(cache=True, nogil=True)
def _counting_order(codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows bucketed by group code in one counting pass, visiting rows in order so each group stays stable."""
    n = codes.size
    offsets = np.zeros(n_groups + 1, dtype=np.int64)
    for i in range(n):
        code = codes[i]
        if code < 0 or code >= n_groups:
            # A negative code once raised in np.bincount; without that guard it would index out of bounds below.
            raise ValueError("group codes must be in [0, n_groups)")
        offsets[code + 1] += 1
    for group in range(n_groups):
        offsets[group + 1] += offsets[group]
    order = np.empty(n, dtype=np.int64)
    cursor = offsets[:n_groups].copy()
    for i in range(n):
        code = codes[i]
        order[cursor[code]] = i
        cursor[code] += 1
    return order, offsets


def group_offsets(codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable row ordering by group, plus the start/stop offsets of each group.

    The codes already index the groups, so a counting sort buckets the rows in linear time and gives
    the same ordering a stable ``argsort`` would, without its ``n log n``.
    """
    codes = np.ascontiguousarray(codes, dtype=np.int32)
    return _counting_order(codes, n_groups)


def group_counts(codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Number of rows per group."""
    return np.bincount(np.asarray(codes), minlength=n_groups).astype(np.int64)


@njit(cache=True, nogil=True)
def _longest_group(offsets: np.ndarray) -> int:
    """Rows in the largest group, which is how wide a per-thread gather buffer has to be."""
    longest = 0
    for group in range(offsets.size - 1):
        length = offsets[group + 1] - offsets[group]
        if length > longest:
            longest = length
    return longest


@njit(cache=True, nogil=True)
def _median_sorted(values: np.ndarray) -> float:
    """Median of ``values``, already sorted ascending."""
    n = values.size
    if n == 0:
        return np.nan
    middle = n // 2
    if n % 2 == 1:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


@njit(cache=True, nogil=True)
def _quantile_sorted(values: np.ndarray, q: float) -> float:
    """Linearly interpolated quantile of ``values``, already sorted ascending."""
    n = values.size
    if n == 0:
        return np.nan
    position = q * (n - 1)
    low = int(np.floor(position))
    high = int(np.ceil(position))
    if low == high:
        return values[low]
    return values[low] + (position - low) * (values[high] - values[low])


@njit(cache=True, nogil=True)
def _median_of(buffer: np.ndarray, n: int) -> float:
    """Median of the ``n`` gathered values in ``buffer``, which this sorts."""
    return _median_sorted(np.sort(buffer[:n]))


@njit(cache=True, nogil=True)
def _select_kth(a: np.ndarray, n: int, k: int) -> float:
    """The ``k``-th smallest of ``a[:n]``, partitioning ``a`` in place so ``a[k]`` holds it.

    Quickselect (the Numerical Recipes in-place scheme, iterative with a median-of-three pivot):
    it reads an order statistic without the ``n log n`` of a full sort, and on return every entry
    left of ``k`` is no greater than ``a[k]`` and every entry right of it no smaller. The value is
    the exact same float a sort would land at ``k``, so a median or quantile read off it is bit-identical.
    """
    lo, hi = 0, n - 1
    while True:
        if hi <= lo + 1:
            if hi == lo + 1 and a[lo] > a[hi]:
                a[lo], a[hi] = a[hi], a[lo]
            return a[k]
        mid = (lo + hi) >> 1
        a[mid], a[lo + 1] = a[lo + 1], a[mid]
        if a[lo] > a[hi]:
            a[lo], a[hi] = a[hi], a[lo]
        if a[lo + 1] > a[hi]:
            a[lo + 1], a[hi] = a[hi], a[lo + 1]
        if a[lo] > a[lo + 1]:
            a[lo], a[lo + 1] = a[lo + 1], a[lo]
        pivot = a[lo + 1]
        i, j = lo + 1, hi
        while True:
            i += 1
            while a[i] < pivot:
                i += 1
            j -= 1
            while a[j] > pivot:
                j -= 1
            if i > j:
                break
            a[i], a[j] = a[j], a[i]
        a[lo + 1], a[j] = a[j], a[lo + 1]
        if j >= k:
            hi = j - 1
        if j <= k:
            lo = i


@njit(cache=True, nogil=True)
def _median_select(buffer: np.ndarray, n: int) -> float:
    """Median of the ``n`` gathered values in ``buffer`` by selection, which this partitions.

    For an even count the upper middle is selected, then the lower middle is the largest value left
    of it; ``0.5 * (lower + upper)`` is taken in ``float64``, matching :func:`_median_sorted`.
    """
    if n == 0:
        return np.nan
    middle = n // 2
    upper = _select_kth(buffer, n, middle)
    if n % 2 == 1:
        return upper
    lower = buffer[0]
    for k in range(1, middle):
        if buffer[k] > lower:
            lower = buffer[k]
    return 0.5 * (np.float64(lower) + np.float64(upper))


@njit(cache=True, nogil=True)
def _quantile_select(buffer: np.ndarray, n: int, q: float) -> float:
    """Linearly interpolated quantile of the ``n`` gathered values by selection, which this partitions.

    The two order statistics the interpolation needs are selected rather than taken from a full sort,
    matching :func:`_quantile_sorted`.
    """
    if n == 0:
        return np.nan
    position = q * (n - 1)
    low = int(np.floor(position))
    high = int(np.ceil(position))
    high_value = _select_kth(buffer, n, high)
    if low == high:
        return high_value
    low_value = buffer[0]  # the largest value left of ``high`` is the ``low``-th order statistic
    for k in range(1, high):
        if buffer[k] > low_value:
            low_value = buffer[k]
    return low_value + (position - low) * (high_value - low_value)


@njit(parallel=True, cache=True, nogil=True)
def _grouped(
    X: np.ndarray, order: np.ndarray, offsets: np.ndarray, n_groups: int, stat: int, q: float, ddof: int
) -> np.ndarray:
    n_vars = X.shape[1]
    out = np.empty((n_groups, n_vars), dtype=np.float64)
    longest = _longest_group(offsets)

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
                out[group, j] = _median_select(buffer, n)
            elif stat == 2:  # median absolute deviation, unscaled
                centre = _median_select(buffer, n)
                for k in range(n):
                    buffer[k] = abs(buffer[k] - centre)
                out[group, j] = _median_select(buffer, n)
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
                out[group, j] = _quantile_select(buffer, n, q)
    return out


def _kernel_input(X: np.ndarray) -> np.ndarray:
    """``X`` as a C-contiguous array the kernels compile for, without rounding a ``float64`` matrix to ``float32``."""
    X = np.asarray(X)
    return np.ascontiguousarray(X, dtype=X.dtype if X.dtype in (np.float32, np.float64) else np.float64)


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
    X = _kernel_input(X)
    order, offsets = group_offsets(codes, n_groups)
    return _grouped(X, order, offsets, n_groups, stat, float(q), int(ddof))


@njit(parallel=True, cache=True, nogil=True)
def _grouped_median_spread(
    X: np.ndarray, order: np.ndarray, offsets: np.ndarray, n_groups: int, spread: int
) -> tuple[np.ndarray, np.ndarray]:
    n_vars = X.shape[1]
    centre = np.empty((n_groups, n_vars), dtype=np.float64)
    scale = np.empty((n_groups, n_vars), dtype=np.float64)
    longest = _longest_group(offsets)

    for group in prange(n_groups):
        start, stop = offsets[group], offsets[group + 1]
        buffer = np.empty(longest, dtype=np.float64)
        for j in range(n_vars):
            n = 0
            for i in range(start, stop):
                value = X[order[i], j]
                if not np.isnan(value):
                    buffer[n] = value
                    n += 1
            if n == 0:
                centre[group, j] = np.nan
                scale[group, j] = np.nan
                continue
            values = np.sort(buffer[:n])
            middle = _median_sorted(values)
            centre[group, j] = middle
            if spread == MAD:
                # Deviations taken in sorted order rather than in gather order are a permutation of the same numbers, so their median is the same.
                for k in range(n):
                    buffer[k] = abs(values[k] - middle)
                scale[group, j] = _median_of(buffer, n)
            else:
                scale[group, j] = _quantile_sorted(values, 0.75) - _quantile_sorted(values, 0.25)
    return centre, scale


def grouped_median_spread(
    X: np.ndarray, codes: np.ndarray, n_groups: int, spread: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group median and robust spread, both read off one sort per (group, feature) slice.

    The same pair out of :func:`grouped_stat` costs three sorts of every slice: for the MAD because the spread pass re-finds the median it measures deviations from, and for the IQR because each of the three quantiles is its own pass.
    Sorting is nearly the whole cost of a robust normalization at screen scale, so this nearly halves ``mad_robustize`` and cuts ``robustize`` to a third.

    Args:
        X: ``(n_obs, n_vars)`` matrix, read in its own dtype if that is ``float32`` or ``float64`` and as ``float64`` otherwise.
        codes: Per-row group code in ``[0, n_groups)``.
        n_groups: Number of groups, so that a group with no rows still gets its row of ``NaN`` instead of being dropped.
        spread: :data:`MAD` for the unscaled median absolute deviation, or :data:`IQR` for the interquartile range.

    Returns:
        ``(median, spread)``, each ``(n_groups, n_vars)`` float64, holding ``NaN`` wherever a group has no usable value for a feature.

    Raises:
        ValueError: If ``spread`` is neither :data:`MAD` nor :data:`IQR`.
    """
    if spread not in (MAD, IQR):
        raise ValueError(f"spread must be MAD or IQR, got {spread!r}")
    X = _kernel_input(X)
    order, offsets = group_offsets(codes, n_groups)
    return _grouped_median_spread(X, order, offsets, n_groups, int(spread))


@njit(parallel=True, cache=True, nogil=True)
def _column_max_robust_z(X: np.ndarray, columns: np.ndarray, mad_to_sigma: float) -> np.ndarray:
    """Largest robust z-score of each selected column, columns shared out over the threads.

    For column ``j`` this is ``max|value - median| / (mad_to_sigma * MAD)`` over its finite values, where ``MAD`` is
    the median of ``|value - median|``. A column with no finite value, or a zero or non-finite scale (a constant
    feature), scores ``0``, matching ``np.nan_to_num(..., nan=0.0)`` on the division.

    ``columns`` indexes into ``X``'s features, so the whole matrix is read without first copying the kept block.
    Each thread gathers into one buffer reused across its columns, since allocating inside the ``prange`` body
    serializes the loop on numba's allocator lock, and the median and MAD come from the same selection
    ``np.nanmedian`` uses, so the result matches it per column.
    """
    n_obs = X.shape[0]
    n_columns = columns.size
    out = np.zeros(n_columns, dtype=np.float64)
    n_threads = get_num_threads()
    for thread in prange(n_threads):
        buffer = np.empty(n_obs, dtype=np.float64)
        for c in range(thread, n_columns, n_threads):
            j = columns[c]
            n = 0
            for i in range(n_obs):
                value = X[i, j]
                if not np.isnan(value):
                    buffer[n] = value
                    n += 1
            if n == 0:
                continue
            median = _median_select(buffer, n)
            largest = 0.0
            for k in range(n):
                deviation = abs(buffer[k] - median)
                buffer[k] = deviation
                if deviation > largest:
                    largest = deviation
            scale = mad_to_sigma * _median_select(buffer, n)
            if scale != 0.0 and np.isfinite(scale):
                out[c] = largest / scale
    return out


def column_max_robust_z(X: np.ndarray, columns: np.ndarray, mad_to_sigma: float) -> np.ndarray:
    """Per-column largest robust z-score over the given feature indices, read column-major in parallel.

    Args:
        X: ``(n_obs, n_vars)`` matrix, read in its own dtype if that is ``float32`` or ``float64`` and as ``float64`` otherwise.
        columns: Feature indices to score, each read straight from ``X`` without copying the kept block out.
        mad_to_sigma: The scale factor turning a median absolute deviation into a standard-deviation estimate.

    Returns:
        ``(columns.size,)`` float64, the largest ``|value - median| / (mad_to_sigma * MAD)`` of each column, and ``0``
        for a column with no finite value or a zero/non-finite scale.
    """
    X = np.asarray(X)
    X = X if X.dtype in (np.float32, np.float64) else X.astype(np.float64)
    return _column_max_robust_z(X, np.ascontiguousarray(columns, dtype=np.int64), float(mad_to_sigma))


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


@njit(cache=True, nogil=True)
def _mwu_exact_sf(n1: int, n2: int) -> np.ndarray:
    """Survival function ``P(U >= k)`` for ``k`` in ``0 .. n1 * n2`` of the exact untied Mann-Whitney null.

    The untied null depends only on the two sample sizes, so one call serves every feature that shares them.
    The arrangement counts are the Gaussian binomial coefficients ``[n1 + n2, n1]_q``, built by multiplying the
    coefficient polynomial by ``(1 - q^(m + i))`` and dividing by ``(1 - q^i)`` for ``i`` up to the smaller size.
    The polynomial is rescaled to sum to one each step, which leaves the ratios exact and keeps the coefficients from
    overflowing when the samples are large.
    """
    size = n1 * n2 + 1
    freq = np.zeros(size)
    freq[0] = 1.0
    smaller = n1 if n1 <= n2 else n2
    larger = n2 if n1 <= n2 else n1
    for i in range(1, smaller + 1):
        shift = larger + i
        nxt = freq.copy()
        for u in range(size - 1, shift - 1, -1):  # multiply by (1 - q^shift)
            nxt[u] = freq[u] - freq[u - shift]
        for u in range(i, size):  # divide by (1 - q^i): a prefix sum with step i
            nxt[u] += nxt[u - i]
        total = 0.0
        for u in range(size):
            total += nxt[u]
        if total > 0.0:
            for u in range(size):
                nxt[u] /= total
        freq = nxt
    grand = 0.0
    for u in range(size):
        grand += freq[u]
    sf = np.empty(size)
    acc = 0.0
    for u in range(size - 1, -1, -1):
        acc += freq[u] / grand
        sf[u] = acc
    return sf


@njit(parallel=True, cache=True, nogil=True)
def _polish_planes(planes: np.ndarray, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Tukey median polish of every ``(rows, columns)`` plane of ``planes``.

    ``planes`` is ``(n_features, n_rows, n_columns)``, feature-major so each plane is contiguous and handled by one thread.
    Returns the fitted row and column effects with the grand level held out of both, as ``(n_features, n_rows)`` and ``(n_features, n_columns)``.

    The arithmetic matches ``np.median`` over a stacked array.
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

    ``W1`` is the integral of ``|F_treated - F_control|`` over the merged support.

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
            # Ties go to the treated sample, as in a stable sort with the treated block first; the gap is zero either way.
            if k < n_treated and (i >= m or values[k] <= control[j, i]):
                current = values[k]
                k += 1
                from_treated = True
            else:
                current = control[j, i]
                i += 1
                from_treated = False
            # The step is weighted by the CDFs as they stood at the previous point, so the counts advance afterwards.
            total += abs(seen_treated / n_treated - seen_control / m) * (current - previous)
            if from_treated:
                seen_treated += 1
            else:
                seen_control += 1
            previous = current
        out[j] = total
    return out
