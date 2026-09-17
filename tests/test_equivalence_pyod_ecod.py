"""ECOD equivalence against pyod, a test-only dependency."""

import numpy as np
import pytest

from mantispy._core._ecod import ecod_scores

ecod = pytest.importorskip("pyod.models.ecod")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scores_match_pyod(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((400, 8))
    X[:8] += 10.0
    X[0, 0] -= 25.0  # one dimension outlying the other way, which sum-of-max needs
    np.testing.assert_allclose(ecod_scores(X), ecod.ECOD().fit(X).decision_scores_, rtol=1e-9)


def test_symmetric_feature_and_ties_match_pyod():
    """skew == 0 contributes both tails; repeated values share the highest rank."""
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.standard_normal(200), np.tile([1.0, 2.0], 100), np.zeros(200)])
    X[:3, 0] += 8.0
    np.testing.assert_allclose(ecod_scores(X), ecod.ECOD().fit(X).decision_scores_, rtol=1e-9)


def test_matches_pyod_on_a_synthetic_plate():
    import mantispy as mt
    from mantispy.ds import synthetic_plate

    wells = mt.tl.aggregate(synthetic_plate(n_wells=48, n_cells=10, n_features=12, seed=0), min_cells=0)
    X = wells.X.astype(np.float64)
    np.testing.assert_allclose(ecod_scores(X), ecod.ECOD().fit(X).decision_scores_, rtol=1e-9)
