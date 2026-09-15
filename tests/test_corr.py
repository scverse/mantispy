import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mantispy._core._corr import corr_matrix, correlated_pairs


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((500, 8))
    values[:, 1] = values[:, 0] * 0.9 + rng.standard_normal(500) * 0.1
    return values.astype(np.float32)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_matches_pandas(X, method):
    expected = pd.DataFrame(X).corr(method=method).to_numpy()
    np.testing.assert_allclose(corr_matrix(X, method=method), expected, rtol=1e-5, atol=1e-6)


def test_chunking_does_not_change_the_result(X):
    """The chunked Gram accumulation is what a streamed implementation would use."""
    np.testing.assert_allclose(corr_matrix(X, chunk_size=37), corr_matrix(X, chunk_size=10_000), rtol=1e-9)


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda X: X.__setitem__((slice(None, 50), 0), np.nan), id="one_column"),
        pytest.param(lambda X: [X.__setitem__((slice(None, 50), c), np.nan) for c in (0, 3, 4)], id="several"),
        pytest.param(lambda X: X.__setitem__((slice(None), 1), np.nan), id="all_nan_column"),
        pytest.param(
            lambda X: (X.__setitem__((slice(None, 20), 0), np.nan), X.__setitem__((slice(None), 2), 3.0)),
            id="nan_plus_constant",
        ),
    ],
)
def test_pairwise_complete_matches_pandas_exactly(X, damage):
    """Only pairs touching a NaN column need their own row set; the rest go through the
    chunked Gram. This gives pandas' numbers in seconds on 50 640 JUMP wells by 3634
    features, where pandas takes over fourteen minutes."""
    X = X.copy()
    damage(X)
    actual = corr_matrix(X)
    expected = pd.DataFrame(X).corr().to_numpy()
    # NaN in the same places, and equal everywhere else.
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-9, atol=1e-10)


def test_constant_column_correlates_with_nothing(X):
    X = X.copy()
    X[:, 2] = 3.0
    result = corr_matrix(X)
    assert np.isnan(result[2]).all()
    np.testing.assert_allclose(np.diag(result)[[0, 1]], 1.0, rtol=1e-9)


def test_unknown_method(X):
    with pytest.raises(ValueError, match="method must be"):
        corr_matrix(X, method="kendall")


@pytest.mark.parametrize("n_vars", [40, 300])
def test_column_blocking_agrees_with_the_full_matrix(n_vars):
    """The blocked pair search must find the same pairs as the full matrix.

    Block sizes that do not divide the feature count leave a narrower last block, which is
    covered here.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, n_vars))
    X[:, 1] = X[:, 0] * 0.98 + rng.normal(scale=0.05, size=200)  # a pair over threshold
    X[rng.integers(0, 200, 3), 5] = np.nan  # and one column with missing values

    full = corr_matrix(X)
    rows, columns = np.tril_indices(n_vars, k=-1)
    expected = {
        tuple(sorted(pair))
        for pair in zip(
            rows[np.nan_to_num(full[rows, columns]) > 0.9],
            columns[np.nan_to_num(full[rows, columns]) > 0.9],
            strict=True,
        )
    }

    for block_size in (7, 13, n_vars):
        pairs, total = correlated_pairs(X, 0.9, block_size=block_size)
        assert {tuple(sorted(pair)) for pair in pairs} == expected
        np.testing.assert_allclose(total, np.nan_to_num(np.abs(full), nan=0.0).sum(axis=0), rtol=1e-9)


@pytest.mark.parametrize("offset", [1.0, 65535.0, 1e7, 1e8])
def test_correlation_survives_a_feature_measured_far_from_zero(offset):
    """Four independent columns whose mean dwarfs their spread, as in a saturated 16-bit
    channel before normalization. The uncentred one-pass identity E[xy]-E[x]E[y] loses the
    covariance to cancellation here and returns 1.0, which makes pp.feature_select drop
    independent features.

    The reference is ``np.corrcoef`` rather than ``pandas.DataFrame.corr``. pandas defines
    this module's pairwise-complete deletion rule but loses precision at these offsets:
    against a ``longdouble`` ground truth its error reaches 4.9e-06 at 1e8 and 3.8e-05 at
    1e9, while this function's is 7.6e-10 and np.corrcoef's is 1.4e-17.
    """
    values = offset + 1e-3 * np.random.default_rng(0).standard_normal((800, 4))
    np.testing.assert_allclose(corr_matrix(values), np.corrcoef(values, rowvar=False), atol=1e-6)


def test_blocked_correlation_survives_the_same_offset():
    """_gram_block computes the correlation separately from _gram_corr and needs the same check."""
    values = 65535.0 + 1e-3 * np.random.default_rng(1).standard_normal((600, 12))
    expected = pd.DataFrame(values).corr().to_numpy()
    rows, columns = correlated_pairs(values, threshold=0.5, block_size=5)
    assert not len(rows), f"independent columns reported as correlated: {list(zip(rows, columns, strict=True))}"
    np.testing.assert_allclose(corr_matrix(values, chunk_size=97), expected, atol=1e-6)


@pytest.mark.parametrize("pattern", ["shared", "distinct"])
def test_spearman_reranks_each_pair_over_the_rows_it_shares(pattern):
    """A rank depends on which rows are present, so a global ranking is not Spearman.

    pandas and scipy both re-rank each pair over its shared rows. A single global ranking
    gets the sign wrong in 8 of 300 trials of two columns at 25% missing. The two
    missingness patterns exercise different branches of the grouping.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=(80, 12))
    if pattern == "shared":  # a feature family undefined for the same rows
        values[np.ix_(rng.random(80) < 0.2, np.arange(5))] = np.nan
    else:  # scattered, so every column carries its own pattern
        values[rng.random(values.shape) < 0.15] = np.nan

    expected = pd.DataFrame(values).corr(method="spearman").to_numpy()
    actual = corr_matrix(values, method="spearman")
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-9, atol=1e-12)


def test_spearman_sign_agrees_with_scipy_on_gappy_columns():
    """Per-pair re-ranking keeps the sign of the correlation, which a global ranking can flip."""
    rng = np.random.default_rng(0)
    disagreements = 0
    for _ in range(200):
        left = rng.normal(size=40)
        values = np.column_stack([left, 0.3 * left + rng.normal(size=40)])
        values[rng.random(values.shape) < 0.25] = np.nan
        ours = corr_matrix(values, method="spearman")[0, 1]
        reference = stats.spearmanr(values[:, 0], values[:, 1], nan_policy="omit").statistic
        if np.isfinite(ours) and np.isfinite(reference):
            assert ours == pytest.approx(reference, abs=1e-12)
            disagreements += np.sign(ours) != np.sign(reference)
    assert disagreements == 0


def test_correlated_pairs_uses_the_same_spearman_as_the_matrix():
    """correlated_pairs reaches the same pairwise Spearman path as corr_matrix."""
    rng = np.random.default_rng(1)
    values = rng.normal(size=(60, 20))
    values[rng.random(values.shape) < 0.1] = np.nan

    reference = pd.DataFrame(values).corr(method="spearman").to_numpy()
    expected = {(min(i, j), max(i, j)) for i in range(20) for j in range(i) if reference[i, j] > 0.3}
    pairs, _ = correlated_pairs(values, 0.3, method="spearman")
    assert {(min(a, b), max(a, b)) for a, b in pairs} == expected
