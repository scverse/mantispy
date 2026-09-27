import numpy as np
import pytest

from mantispy._core._distance import mahalanobis_transform


def test_mahalanobis_transform_survives_a_missing_value():
    """A single NaN would make np.cov all-NaN and eigh fail to converge."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 4))
    reference[3, 1] = np.nan
    centre, whitening = mahalanobis_transform(reference)
    assert np.isfinite(centre).all() and np.isfinite(whitening).all()

    with pytest.raises(ValueError, match="too few to estimate a covariance"):
        mahalanobis_transform(np.full((5, 3), np.nan))
