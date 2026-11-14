import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mantispy._core._numba import (
    MAD,
    MEAN,
    MEDIAN,
    QUANTILE,
    grouped_median_spread,
    grouped_stat,
)
from mantispy._core._reduce import group_codes, reduce_grouped, transform_grouped


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


@pytest.fixture
def mixed_adata():
    """An object whose obs mixes string and numeric grouping keys, with repeated leading keys."""
    rng = np.random.default_rng(0)
    n = 400
    obs = pd.DataFrame(
        {
            "Metadata_Plate": rng.choice(["p10", "p2", "p1"], n),
            "Metadata_Well": rng.choice(["A01", "B12", "A2", "H09"], n),
            # A numeric key so 0.4 and 0.40 stay one group and the dtype is not stringified.
            "Metadata_Dose": rng.choice([0.4, 10.0, 2.0], n),
        }
    )
    return ad.AnnData(X=rng.standard_normal((n, 5)).astype(np.float32), obs=obs)


@pytest.mark.parametrize(
    "by",
    [
        ["Metadata_Plate", "Metadata_Well"],
        ["Metadata_Plate", "Metadata_Dose"],
        ["Metadata_Plate", "Metadata_Well", "Metadata_Dose"],
        # A column named twice, as when a caller joins its own keys to a by= that already holds one of them.
        ["Metadata_Plate", "Metadata_Well", "Metadata_Plate"],
    ],
)
def test_group_codes_matches_the_multiindex_path(mixed_adata, by):
    """The multi-column fast path must reproduce MultiIndex.factorize(sort=True), codes and keys alike."""
    codes, keys = group_codes(mixed_adata, by)
    ref_codes, ref_uniques = pd.MultiIndex.from_frame(mixed_adata.obs[by]).factorize(sort=True)
    assert np.array_equal(codes, ref_codes)
    assert keys.equals(ref_uniques)


def test_reduce_grouped_reuses_a_precomputed_grouping(mixed_adata):
    """Passing grouping= must give the same statistic as letting reduce_grouped factorize itself."""
    by = ["Metadata_Plate", "Metadata_Well"]
    values, keys, counts = reduce_grouped(mixed_adata, by, MEAN)
    grouping = group_codes(mixed_adata, by)
    values2, keys2, counts2 = reduce_grouped(mixed_adata, by, MEAN, grouping=grouping)
    assert np.array_equal(values, values2, equal_nan=True)
    assert keys.equals(keys2)
    assert np.array_equal(counts, counts2)


def test_transform_grouped_reuses_a_precomputed_grouping(mixed_adata):
    """Passing grouping= must rewrite the matrix identically to the self-factorizing path."""
    by = ["Metadata_Plate", "Metadata_Dose"]
    centred = transform_grouped(mixed_adata, by, lambda key, block: block - block.mean(axis=0))
    grouping = group_codes(mixed_adata, by)
    centred2 = transform_grouped(mixed_adata, by, lambda key, block: block - block.mean(axis=0), grouping=grouping)
    assert np.array_equal(centred, centred2, equal_nan=True)
