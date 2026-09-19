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
    X = np.column_stack([rng.standard_normal(200), np.tile([1.0, 2.0, 3.0, 2.0], 50), np.zeros(200)])
    X[:3, 0] += 8.0
    np.testing.assert_allclose(ecod_scores(X), ecod.ECOD().fit(X).decision_scores_, rtol=1e-9)


def test_matches_pyod_on_a_synthetic_plate():
    import mantispy as mt
    from mantispy.ds import synthetic_plate

    wells = mt.tl.aggregate(synthetic_plate(n_wells=48, n_cells=10, n_features=12, seed=0), min_cells=0)
    X = wells.X.astype(np.float64)
    np.testing.assert_allclose(ecod_scores(X), ecod.ECOD().fit(X).decision_scores_, rtol=1e-9)


def test_float32_and_missing_values_match_pyod_on_the_imputed_matrix():
    """CellProfiler output arrives as float32 with gaps; a gap takes its column's mean."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    X[:4] += 9.0
    X[rng.random(X.shape) < 0.02] = np.nan
    imputed = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    np.testing.assert_allclose(ecod_scores(X), ecod.ECOD().fit(imputed).decision_scores_, rtol=1e-9)
