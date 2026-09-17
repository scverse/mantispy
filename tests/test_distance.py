import numpy as np
import pytest
from scipy.spatial.distance import cdist

from mantispy._core._distance import energy_distance, mahalanobis_transform, pairwise_sqeuclidean


@pytest.fixture
def blocks():
    rng = np.random.default_rng(0)
    return rng.standard_normal((60, 5)), rng.standard_normal((40, 5))


def test_pairwise_matches_scipy_and_chunking_is_irrelevant(blocks):
    A, B = blocks
    np.testing.assert_allclose(pairwise_sqeuclidean(A, B), cdist(A, B, "sqeuclidean"), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(pairwise_sqeuclidean(A, B, chunk_size=7), pairwise_sqeuclidean(A, B), rtol=1e-9)


def test_pairwise_is_never_negative(blocks):
    """Floating point can push a self-distance slightly below zero."""
    A, _ = blocks
    assert (pairwise_sqeuclidean(A, A) >= 0).all()


def test_energy_distance_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(1)
    assert abs(energy_distance(rng.standard_normal((300, 4)), rng.standard_normal((300, 4)))) < 0.5


def test_energy_distance_grows_with_separation():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 4))
    assert energy_distance(A, A + 1.0) < energy_distance(A, A + 5.0)


def test_mahalanobis_transform_whitens_the_reference():
    rng = np.random.default_rng(2)
    reference = rng.standard_normal((500, 3)) @ np.array([[2.0, 1, 0], [0, 3, 0], [0, 0, 1]])
    centre, whitening = mahalanobis_transform(reference)
    # Whitening this reference lands within 1.1e-06 of the identity. Skipping it misses by 9.3
    # and scaling by the standard deviation alone misses by 0.33, so atol=0.15 accepted both.
    np.testing.assert_allclose(np.cov((reference - centre) @ whitening, rowvar=False), np.eye(3), atol=1e-5)


def test_mahalanobis_transform_survives_a_missing_value():
    """A single NaN would make np.cov all-NaN and eigh fail to converge."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 4))
    reference[3, 1] = np.nan
    centre, whitening = mahalanobis_transform(reference)
    assert np.isfinite(centre).all() and np.isfinite(whitening).all()

    with pytest.raises(ValueError, match="too few to estimate a covariance"):
        mahalanobis_transform(np.full((5, 3), np.nan))
