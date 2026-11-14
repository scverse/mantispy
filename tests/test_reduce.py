import numpy as np
import pytest

from mantispy._core._numba import (
    MAD,
    MEDIAN,
    QUANTILE,
    grouped_median_spread,
    grouped_stat,
)


def test_median_spread_refuses_a_stat_that_is_not_a_spread():
    """MEDIAN reads as a valid selector everywhere else, and would silently give the IQR here."""
    with pytest.raises(ValueError, match="MAD or IQR"):
        grouped_median_spread(np.ones((4, 2), dtype=np.float32), np.zeros(4, dtype=np.int32), 1, MEDIAN)


def _reference(matrix, codes, n_groups, stat, q):
    """Per-group statistic from a plain sort in float64, the definition the kernel's quickselect must match."""
    matrix = matrix.astype(np.float64)
    out = np.full((n_groups, matrix.shape[1]), np.nan)
    for group in range(n_groups):
        rows = np.flatnonzero(codes == group)
        for col in range(matrix.shape[1]):
            values = np.sort(matrix[rows, col][~np.isnan(matrix[rows, col])])
            if values.size == 0:
                continue
            if stat == MEDIAN:
                out[group, col] = np.median(values)
            elif stat == MAD:
                out[group, col] = np.median(np.abs(values - np.median(values)))
            else:
                position = q * (values.size - 1)
                low, high = int(np.floor(position)), int(np.ceil(position))
                out[group, col] = (
                    values[low] if low == high else values[low] + (position - low) * (values[high] - values[low])
                )
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("stat,q", [(MEDIAN, 0.5), (MAD, 0.5), (QUANTILE, 0.25), (QUANTILE, 0.9)])
def test_grouped_stat_quickselect_matches_a_full_sort(dtype, stat, q):
    """The quickselect median/quantile lands the same order statistics a full sort would, bit for bit.

    Covers even- and odd-count groups, singletons, ties, NaN-skipping, all-NaN and empty groups.
    """
    rng = np.random.default_rng(0)
    blocks, codes = [], []
    # group 0: even count, group 1: odd count, group 2: singleton, group 3: all NaN,
    # group 5 left empty so a gap in the codes still yields a NaN row.
    sizes = [8, 7, 1, 4, 6]
    for group, size in enumerate(sizes):
        block = rng.standard_normal((size, 5)).astype(dtype)
        if group == 3:
            block[:] = np.nan
        else:
            block[rng.random((size, 5)) < 0.3] = np.nan
        block[: size // 2, 0] = block[0, 0]  # ties, including a column that is all one value
        blocks.append(block)
        codes.extend([group] * size)
    matrix = np.ascontiguousarray(np.concatenate(blocks))
    codes = np.array(codes, dtype=np.int32)
    n_groups = len(sizes) + 1

    got = grouped_stat(matrix, codes, n_groups, stat, q=q)
    expected = _reference(matrix, codes, n_groups, stat, q)
    assert np.array_equal(got, expected, equal_nan=True)
