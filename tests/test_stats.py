import numpy as np
import pytest

from mantispy._core._stats import benjamini_hochberg, permutation_pvalue, robust_zscore, split_reference


def test_benjamini_hochberg_matches_the_textbook_example():
    pvalues = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    expected = np.array([0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0846, 0.205])
    np.testing.assert_allclose(benjamini_hochberg(pvalues), expected, rtol=1e-3)


def test_benjamini_hochberg_is_bounded_and_order_independent():
    rng = np.random.default_rng(0)
    pvalues = rng.random(200)
    q = benjamini_hochberg(pvalues)
    assert (q >= 0).all() and (q <= 1).all()

    shuffle = rng.permutation(200)
    np.testing.assert_allclose(benjamini_hochberg(pvalues[shuffle]), q[shuffle], rtol=1e-12)


def test_benjamini_hochberg_passes_nan_through():
    q = benjamini_hochberg(np.array([0.01, np.nan, 0.5]))
    assert np.isnan(q[1]) and np.isfinite(q[[0, 2]]).all()


def test_permutation_pvalue_never_returns_zero():
    """A permutation p-value of 0 claims more precision than the null has."""
    assert permutation_pvalue(np.array([10.0]), np.zeros((1, 99)))[0] == pytest.approx(1 / 100)


def test_permutation_pvalue_is_right_tailed():
    null = np.tile(np.linspace(0, 1, 100), (2, 1))
    np.testing.assert_allclose(permutation_pvalue(np.array([0.5, 2.0]), null), [0.505, 0.0099], atol=0.02)


def test_robust_zscore_handles_zero_spread():
    z = robust_zscore(np.column_stack([np.ones(10), np.arange(10.0)]))
    assert np.isfinite(z).all()
    assert (z[:, 0] == 0).all()


def test_benjamini_hochberg_corrects_a_table_as_one_family():
    """A features-by-groups table is one family of tests, and it must keep its shape."""
    grid = np.array([[0.001, 0.2, 0.5], [0.03, 0.4, 0.9]])
    q = benjamini_hochberg(grid)
    assert q.shape == grid.shape
    np.testing.assert_allclose(q, benjamini_hochberg(grid.ravel()).reshape(grid.shape))


def test_permutation_pvalue_passes_nan_through():
    """`nan >= x` is False, which would otherwise hand an unmeasured value the best p."""
    assert np.isnan(permutation_pvalue(np.array([np.nan]), np.zeros((1, 99)))[0])


def test_robust_zscore_keeps_an_infinite_value_extreme():
    values = np.column_stack([np.arange(10.0), np.arange(10.0)])
    values[3, 0] = np.inf
    assert np.isinf(robust_zscore(values)[3, 0])


def test_split_reference_partitions_the_rows():
    generator = np.random.default_rng(0)
    rows = np.arange(10, 60)
    fit, null = split_reference(rows, generator)

    assert set(fit) | set(null) == set(rows)
    assert not set(fit) & set(null)
    assert list(fit) == sorted(fit) and list(null) == sorted(null)
    assert abs(fit.size - null.size) <= 1


def test_split_reference_is_reproducible_from_the_generator():
    first = split_reference(np.arange(40), np.random.default_rng(3))
    second = split_reference(np.arange(40), np.random.default_rng(3))
    np.testing.assert_array_equal(first[0], second[0])


def test_split_reference_refuses_a_reference_it_cannot_halve():
    """One row per side is the least that can carry a null at all."""
    with pytest.raises(ValueError, match="at least four rows"):
        split_reference(np.arange(3), np.random.default_rng(0))
