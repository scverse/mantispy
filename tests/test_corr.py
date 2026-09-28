import numpy as np
import pandas as pd
import pytest

from mantispy._core._corr import corr_matrix, correlated_pairs


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((500, 8))
    values[:, 1] = values[:, 0] * 0.9 + rng.standard_normal(500) * 0.1
    return values.astype(np.float32)


def test_unknown_method(X):
    with pytest.raises(ValueError, match="method must be"):
        corr_matrix(X, method="kendall")


@pytest.mark.parametrize("n_vars", [40, 300])
def test_column_blocking_agrees_with_the_full_matrix(n_vars):
    """The blocked pair search must find the same pairs as the full matrix.

    Block sizes that do not divide the feature count leave a narrower last block, which is covered here.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, n_vars))
    X[:, 1] = X[:, 0] * 0.98 + rng.normal(scale=0.05, size=200)
    X[rng.integers(0, 200, 3), 5] = np.nan

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


def test_correlated_pairs_uses_the_same_spearman_as_the_matrix():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(60, 20))
    values[rng.random(values.shape) < 0.1] = np.nan

    reference = pd.DataFrame(values).corr(method="spearman").to_numpy()
    expected = {(min(i, j), max(i, j)) for i in range(20) for j in range(i) if reference[i, j] > 0.3}
    pairs, _ = correlated_pairs(values, 0.3, method="spearman")
    assert {(min(a, b), max(a, b)) for a, b in pairs} == expected
