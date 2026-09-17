import numpy as np
import pandas as pd
import pytest

from mantispy._core._numba import IQR, MAD, MEAN, MEDIAN, QUANTILE, STD, grouped_median_spread, grouped_stat
from mantispy._core._reduce import (
    get_matrix,
    group_codes,
    iter_groups,
    reduce_grouped,
    transform_grouped,
)
from mantispy.ds import synthetic_plate


@pytest.fixture
def adata():
    return synthetic_plate(n_plates=2, n_wells=8, n_cells=5, n_features=10, seed=0)


# --- kernels ---------------------------------------------------------------


@pytest.fixture
def kernel_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 7)).astype(np.float32)
    codes = rng.integers(0, 4, size=200).astype(np.int32)
    return X, codes, 4


@pytest.mark.parametrize(
    ("stat", "reference"),
    [
        (MEAN, lambda block: np.nanmean(block, axis=0)),
        (MEDIAN, lambda block: np.nanmedian(block, axis=0)),
        (STD, lambda block: np.nanstd(block, axis=0, ddof=1)),
        (
            MAD,
            lambda block: np.nanmedian(np.abs(block - np.nanmedian(block, axis=0)), axis=0),
        ),
    ],
    ids=["mean", "median", "std", "mad"],
)
def test_kernels_match_numpy(kernel_data, stat, reference):
    X, codes, n_groups = kernel_data
    expected = np.stack([reference(X[codes == g]) for g in range(n_groups)])
    np.testing.assert_allclose(grouped_stat(X, codes, n_groups, stat), expected, rtol=1e-5)


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_quantile_matches_numpy(kernel_data, q):
    X, codes, n_groups = kernel_data
    expected = np.stack([np.nanquantile(X[codes == g], q, axis=0) for g in range(n_groups)])
    np.testing.assert_allclose(grouped_stat(X, codes, n_groups, QUANTILE, q=q), expected, rtol=1e-5)


def test_std_honours_ddof():
    """pycytominer's standardize uses the population SD; mad_robustize does not care."""
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    codes = np.zeros(6, dtype=np.int32)
    np.testing.assert_allclose(grouped_stat(X, codes, 1, STD, ddof=0)[0], np.std(X, axis=0, ddof=0), rtol=1e-6)
    np.testing.assert_allclose(grouped_stat(X, codes, 1, STD, ddof=1)[0], np.std(X, axis=0, ddof=1), rtol=1e-6)


def test_nan_is_skipped_not_propagated():
    X = np.array([[1.0, 1.0], [np.nan, 2.0], [3.0, 3.0]], dtype=np.float32)
    codes = np.zeros(3, dtype=np.int32)
    np.testing.assert_allclose(grouped_stat(X, codes, 1, MEDIAN), [[2.0, 2.0]])


def test_all_nan_column_and_empty_group_yield_nan():
    X = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=np.float32)
    codes = np.zeros(2, dtype=np.int32)
    out = grouped_stat(X, codes, 3, MEDIAN)
    assert np.isnan(out[0, 0]) and out[0, 1] == 1.5
    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()


@pytest.mark.parametrize("spread", [MAD, IQR], ids=["mad", "iqr"])
def test_median_spread_matches_the_separate_passes_exactly(kernel_data, spread):
    """One sort per slice replaced a median pass plus a spread pass that sorted it again.

    The agreement has to be exact rather than close: pp.normalize divides every value by this
    scale, so a last-bit difference would move every normalized profile in the screen.
    """
    X, codes, n_groups = kernel_data
    X = X.copy()
    X[:20, 3] = np.nan  # measured in some rows of every group, but not all
    X[:, 5] = np.nan  # measured nowhere, so the slice is empty
    n_groups += 1  # and one group with no rows at all

    centre, scale = grouped_median_spread(X, codes, n_groups, spread)
    if spread == MAD:
        expected = grouped_stat(X, codes, n_groups, MAD)
    else:
        upper = grouped_stat(X, codes, n_groups, QUANTILE, q=0.75)
        expected = upper - grouped_stat(X, codes, n_groups, QUANTILE, q=0.25)

    np.testing.assert_array_equal(centre, grouped_stat(X, codes, n_groups, MEDIAN))
    np.testing.assert_array_equal(scale, expected)
    assert np.isnan(centre[:, 5]).all() and np.isnan(scale[:, 5]).all()
    assert np.isnan(centre[-1]).all(), "a group with no rows has no median"


def test_median_spread_refuses_a_stat_that_is_not_a_spread():
    """MEDIAN reads as a valid selector everywhere else, and would silently give the IQR here."""
    with pytest.raises(ValueError, match="MAD or IQR"):
        grouped_median_spread(np.ones((4, 2), dtype=np.float32), np.zeros(4, dtype=np.int32), 1, MEDIAN)


# --- the seam --------------------------------------------------------------


def test_group_codes_single_and_multi_column(adata):
    codes, keys = group_codes(adata, "Metadata_Plate")
    assert list(keys) == ["Plate01", "Plate02"]
    assert codes.dtype == np.int32

    codes, keys = group_codes(adata, ["Metadata_Plate", "Metadata_Well"])
    assert isinstance(keys, pd.MultiIndex)
    assert len(keys) == 16


def test_group_codes_preserves_numeric_dtype(adata):
    """Stringifying would make 0.4 and 0.40 different groups."""
    adata.obs["Metadata_Concentration"] = np.tile([0.4, 0.40, 1.0, 2.0], adata.n_obs // 4)
    _, keys = group_codes(adata, ["Metadata_Plate", "Metadata_Concentration"])
    assert keys.levels[1].dtype.kind == "f"
    assert len(keys.levels[1]) == 3


def test_group_codes_rejects_missing_values(adata):
    obs = adata.obs.copy()
    obs["Metadata_Plate"] = obs["Metadata_Plate"].astype(object)
    obs.iloc[0, obs.columns.get_loc("Metadata_Plate")] = None
    adata.obs = obs
    with pytest.raises(ValueError, match="missing values"):
        group_codes(adata, ["Metadata_Plate", "Metadata_Well"])


def test_group_codes_none_is_one_group(adata):
    codes, keys = group_codes(adata, None)
    assert len(keys) == 1 and (codes == 0).all()


def test_iter_groups_covers_every_row_exactly_once(adata):
    seen = np.zeros(adata.n_obs, dtype=int)
    for _key, rows, block in iter_groups(adata, "Metadata_Plate"):
        seen[rows] += 1
        assert block.shape == (len(rows), adata.n_vars)
        np.testing.assert_array_equal(block, adata.X[rows])
    np.testing.assert_array_equal(seen, 1)


def test_reduce_grouped_matches_a_manual_loop(adata):
    values, keys, counts = reduce_grouped(adata, "Metadata_Plate", MEDIAN)
    for index, plate in enumerate(keys):
        rows = (adata.obs["Metadata_Plate"] == plate).to_numpy()
        np.testing.assert_allclose(values[index], np.nanmedian(adata.X[rows], axis=0), rtol=1e-5)
        assert counts[index] == rows.sum()


def test_reduce_grouped_mask_restricts_contributing_rows(adata):
    mask = adata.obs["Metadata_Control"].to_numpy()
    values, keys, counts = reduce_grouped(adata, "Metadata_Plate", MEAN, mask=mask)
    for index, plate in enumerate(keys):
        rows = ((adata.obs["Metadata_Plate"] == plate) & adata.obs["Metadata_Control"]).to_numpy()
        np.testing.assert_allclose(values[index], np.nanmean(adata.X[rows], axis=0), rtol=1e-5)
        assert counts[index] == rows.sum()


def test_transform_grouped_writes_every_row_and_keeps_float32(adata):
    out = transform_grouped(adata, "Metadata_Plate", lambda _key, block: block * 2.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, adata.X * 2.0, rtol=1e-6)


def test_transform_grouped_sees_each_group_separately(adata):
    seen = []
    transform_grouped(adata, "Metadata_Plate", lambda key, block: seen.append((key, len(block))) or block)
    assert [key for key, _ in seen] == ["Plate01", "Plate02"]


def test_get_matrix_reads_a_layer(adata):
    adata.layers["doubled"] = adata.X * 2
    np.testing.assert_array_equal(get_matrix(adata, "doubled"), adata.X * 2)
