"""The private solver picker that keeps an internal PCA bounded on a tall, narrow table."""

import pytest

from mantispy._core._pca import pca_solver


def test_auto_picks_covariance_eigh_for_a_tall_narrow_table():
    """Many more rows than columns, and few enough columns, is where eigendecomposing the covariance wins."""
    assert pca_solver(100_000, 50, fallback="arpack") == "covariance_eigh"


def test_auto_keeps_the_fallback_when_the_table_is_wide():
    """Without the row-to-column headroom the full decomposition stays, so the covariance is never squared."""
    assert pca_solver(60, 50, fallback="arpack") == "arpack"


def test_auto_keeps_the_fallback_when_there_are_too_many_columns():
    """A covariance that large is itself expensive to form and eigendecompose, so the table stays wide-handled."""
    assert pca_solver(10_000_000, 5_000, fallback="full") == "full"


@pytest.mark.parametrize("override", ["arpack", "full", "covariance_eigh", "randomized"])
def test_a_named_solver_is_returned_unchanged(override):
    """Anything but "auto" is the caller forcing a solver, which the picker must not second-guess."""
    assert pca_solver(100_000, 50, override, fallback="arpack") == override
    # Even when the shape would otherwise keep the fallback.
    assert pca_solver(60, 50, override, fallback="arpack") == override
