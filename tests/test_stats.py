import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from mantispy._core._stats import (
    benjamini_hochberg,
    mannwhitney_pvalues,
    permutation_pvalue,
    robust_zscore,
    sorted_control,
    split_reference,
)


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


def test_sorted_control_counts_every_measured_value_including_infinities():
    """np.sort puts -inf first, so counting only finite values sliced the reference one short
    and cut the largest control off its end."""
    values, counts, _ = sorted_control(np.array([[-np.inf, 1.0], [1.0, np.nan], [2.0, 2.0], [np.nan, 3.0]]))

    assert counts.tolist() == [3, 3]
    assert values[0][:3].tolist() == [-np.inf, 1.0, 2.0]
    assert np.isnan(values[0][3])


def test_mannwhitney_matches_scipy_when_the_reference_holds_an_infinity():
    """CellProfiler ratio features produce +-inf. A -inf shifted the sorted reference by one,
    which moved the p-values by up to 41% and always toward significance."""
    rng = np.random.default_rng(0)
    control = rng.normal(size=(40, 4))
    treated = rng.normal(loc=0.4, size=(25, 4))
    control[5, 0] = -np.inf
    control[[7, 11], 1] = -np.inf  # repeated, so the tie correction has to see both
    control[9, 2] = np.inf
    control[13, 3] = np.nan

    got = mannwhitney_pvalues(treated, sorted_control(control))

    # scipy ranks an infinity as the extreme value it is and omits only the missing ones.
    expected = [
        mannwhitneyu(treated[:, j], control[~np.isnan(control[:, j]), j], method="asymptotic").pvalue
        for j in range(control.shape[1])
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-12)
