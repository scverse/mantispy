"""Correctness of the column-blocked kernels: block iteration and float32 correlation.

Chunking is a re-association of the same math, so the result must equal the whole-matrix
reference. float32 correlation only diverges near the threshold, which these tests keep clear of.
"""

import numpy as np
import pandas as pd
import pytest

from mantispy._core._corr import column_block, corr_matrix, correlated_pairs
from mantispy._core._stats import nanvar
from mantispy.pp._select import _op_drop_outliers


def _scatter_nans(values: np.ndarray, rng: np.random.Generator, denom: int = 20) -> np.ndarray:
    """Set roughly ``1 / denom`` of the entries to NaN, in place, with an int8 mask to stay cheap."""
    values[rng.integers(0, denom, size=values.shape, dtype=np.int8) == 0] = np.nan
    return values


def _pair_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in pairs.tolist()}


def test_nanvar_chunked_matches_numpy_across_shapes():
    rng = np.random.default_rng(0)
    for shape, ddof in [((500, 300), 0), ((500, 300), 1), ((128, 4096), 0)]:
        X = _scatter_nans(rng.standard_normal(shape).astype(np.float32), rng)
        expected = np.nanvar(X, axis=0, ddof=ddof)
        np.testing.assert_allclose(nanvar(X, ddof=ddof), expected, rtol=1e-6)


def test_nanvar_chunked_spans_multiple_blocks():
    """A shape whose feature count exceeds one block, so the loop runs more than once."""
    rng = np.random.default_rng(1)
    X = _scatter_nans(rng.standard_normal((20000, 4000)).astype(np.float32), rng)
    assert column_block(X.shape[0], X.dtype.itemsize) < X.shape[1]
    np.testing.assert_allclose(nanvar(X, ddof=0), np.nanvar(X, axis=0, ddof=0), rtol=1e-6)


def test_drop_outliers_matches_reference():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 60)).astype(np.float32)
    X[:, 3] = 1000.0  # a clear outlier column
    X[:, 10] *= 2000.0  # large values
    X[:, 20] = np.nan  # all-NaN column
    X = _scatter_nans(X, rng)
    cutoff = 500.0
    expected = ~(np.nan_to_num(np.nanmax(np.abs(X), axis=0), nan=0.0) > cutoff)
    np.testing.assert_array_equal(_op_drop_outliers(X, cutoff), expected)


def test_drop_outliers_spans_multiple_blocks():
    """A shape whose feature count exceeds one block, so the chunk loop runs more than once."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((20000, 4000)).astype(np.float32)
    X[:, 1234] = 1000.0  # an outlier column beyond the first block
    assert column_block(X.shape[0], X.dtype.itemsize) < X.shape[1]
    cutoff = 500.0
    expected = ~(np.nan_to_num(np.nanmax(np.abs(X), axis=0), nan=0.0) > cutoff)
    np.testing.assert_array_equal(_op_drop_outliers(X, cutoff), expected)


def test_exact_correlated_pairs_are_dtype_invariant():
    """The exact path upcasts to float64, so a float32 and a float64 input give the same pair set."""
    rng = np.random.default_rng(3)
    n_obs, n_vars = 800, 12
    base = rng.standard_normal((n_obs, n_vars))
    # A couple of near-perfect pairs, everything else independent, so both sides of 0.9 are far away.
    base[:, 1] = base[:, 0] + rng.standard_normal(n_obs) * 0.01
    base[:, 3] = base[:, 2] + rng.standard_normal(n_obs) * 0.01

    pairs32, _ = correlated_pairs(base.astype(np.float32), 0.9)
    pairs64, _ = correlated_pairs(base.astype(np.float64), 0.9)
    assert _pair_set(pairs32) == _pair_set(pairs64)
    assert _pair_set(pairs64) == {(0, 1), (2, 3)}


def test_corr_matrix_float64_default_matches_pandas():
    """The default work_dtype stays float64: a float32 input still gives the float64-accurate result."""
    rng = np.random.default_rng(8)
    X = rng.standard_normal((300, 10)).astype(np.float32)
    expected = pd.DataFrame(X).corr().to_numpy()
    np.testing.assert_allclose(corr_matrix(X), expected, atol=1e-6)


def test_corr_matrix_float32_is_close():
    """work_dtype=float32 matches the float64 result to ~1e-4."""
    rng = np.random.default_rng(9)
    X = rng.standard_normal((300, 10)).astype(np.float32)
    np.testing.assert_allclose(corr_matrix(X, work_dtype=np.float32), corr_matrix(X), atol=1e-4)


def test_windowed_pairs_equals_exact_when_window_covers_all():
    """A window at least as wide as the data tests every pair, so it matches the exact set."""
    rng = np.random.default_rng(4)
    n_obs, n_vars = 400, 15
    X = rng.standard_normal((n_obs, n_vars))
    X[:, 5] = X[:, 2] + rng.standard_normal(n_obs) * 0.01
    X[:, 11] = X[:, 8] + rng.standard_normal(n_obs) * 0.01

    exact, _ = correlated_pairs(X, 0.9)
    windowed, _ = correlated_pairs(X, 0.9, window=n_vars)
    assert _pair_set(windowed) == _pair_set(exact)
    assert _pair_set(exact) == {(2, 5), (8, 11)}


def test_windowed_pairs_finds_adjacent_pairs():
    """Correlated features that sort adjacently are recovered by a small overlapping window."""
    rng = np.random.default_rng(5)
    n_obs = 400
    base = rng.standard_normal((n_obs, 6))
    base[:, 1] = base[:, 0] + rng.standard_normal(n_obs) * 0.01
    base[:, 3] = base[:, 2] + rng.standard_normal(n_obs) * 0.01

    windowed, _ = correlated_pairs(base, 0.9, window=2, stride=1)
    assert _pair_set(windowed) == {(0, 1), (2, 3)}


def test_windowed_misses_only_cross_window_pairs():
    """Two correlated features placed far apart with no overlap fall in different windows and are missed."""
    rng = np.random.default_rng(6)
    n_obs, n_vars = 400, 12
    X = rng.standard_normal((n_obs, n_vars))
    X[:, 8] = X[:, 0] + rng.standard_normal(n_obs) * 0.01

    exact, _ = correlated_pairs(X, 0.9)
    windowed, _ = correlated_pairs(X, 0.9, window=4, stride=4)
    assert (0, 8) in _pair_set(exact)
    assert (0, 8) not in _pair_set(windowed)


def test_two_pass_matches_exact_on_cross_window_redundancy():
    """The fast two-pass path recovers cross-window redundancy the windowed primitive alone misses.

    A correlated pair placed far apart in the column order falls into different windows, so pass 1
    never tests it; the exact pass 2 on the survivors does, and the final keep mask equals the exact
    full pass. Exact duplicates give each pair's two members an identical total correlation, so the
    greedy keep is decided by index alone, the same in the exact pass and in the two-pass refine.
    """
    from mantispy.pp._select import _op_correlation_threshold

    rng = np.random.default_rng(12)
    n_obs, n_vars = 800, 8
    X = rng.standard_normal((n_obs, n_vars))
    X[:, 3] = X[:, 2]  # within-window pair: adjacent, caught by pass 1
    X[:, 7] = X[:, 0]  # cross-window pair: seven columns apart, missed by a small window

    exact = _op_correlation_threshold(X, 0.9, window=None)
    fast = _op_correlation_threshold(X, 0.9, window=3)
    np.testing.assert_array_equal(fast, exact)
    # The windowed primitive alone would leave column 7 in; pass 2 on the survivors is what drops it.
    assert not exact[7] and not exact[3]
    assert exact[0] and exact[2]


def test_windowed_rejects_bad_window_and_stride():
    """The fast path validates its knobs: window and stride must be positive integers."""
    rng = np.random.default_rng(13)
    X = rng.standard_normal((50, 6))
    for bad_window in (0, -1):
        with pytest.raises(ValueError):
            correlated_pairs(X, 0.9, window=bad_window)
    for bad_stride in (0, -2):
        with pytest.raises(ValueError):
            correlated_pairs(X, 0.9, window=3, stride=bad_stride)


def test_feature_select_corr_window_drops_one_of_a_pair():
    """The windowed correlation path actually drops a redundant feature and writes a boolean var column."""
    from anndata import AnnData

    import mantispy as mt

    rng = np.random.default_rng(7)
    n_obs = 60
    X = rng.standard_normal((n_obs, 6)).astype(np.float32)
    X[:, 1] = X[:, 0] + rng.standard_normal(n_obs).astype(np.float32) * 0.01
    names = [f"Cells_Intensity_{i}" for i in range(6)]
    adata = AnnData(X=X, var=pd.DataFrame(index=names))

    mt.pp.feature_select(adata, operations=("correlation_threshold",), corr_window=3, corr_stride=1)
    selected = adata.var["selected"]
    assert selected.dtype == bool
    assert selected.shape == (6,)
    assert int(selected.sum()) == 5  # one of the correlated pair is dropped
    assert not (bool(selected.iloc[0]) and bool(selected.iloc[1]))  # not both members kept
