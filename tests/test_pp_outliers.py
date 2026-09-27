import numpy as np
import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def adata():
    return synthetic_plate(n_wells=16, n_cells=40, n_features=15, seed=0)


@pytest.mark.parametrize("method", ["ecod", "isolation_forest", "mad"])
def test_every_method_writes_a_flag_and_a_score(adata, method):
    mt.pp.outliers(adata, method=method, contamination=0.05)
    assert adata.obs["qc_outlier"].dtype == bool
    assert 0 < adata.obs["qc_outlier"].sum() < adata.n_obs
    assert np.isfinite(adata.obs["qc_outlier_score"]).all()


def test_score_cutoff_thresholds_absolutely(adata):
    """With method='mad' the score is a robust z, so 5 is the familiar rule."""
    mt.pp.outliers(adata, method="mad", score_cutoff=5.0)
    assert (adata.obs["qc_outlier_score"][adata.obs["qc_outlier"]] > 5.0).all()
    mt.pp.outliers(adata, method="mad", score_cutoff=1e9, key_added="none_flagged")
    assert not adata.obs["none_flagged"].any()


def test_rejects_a_bad_method_or_contamination(adata):
    with pytest.raises(ValueError, match="method must be"):
        mt.pp.outliers(adata, method="nonsense")
    with pytest.raises(ValueError, match="contamination"):
        mt.pp.outliers(adata, contamination=1.5)
    with pytest.raises(ValueError, match="ecod_aggregation must be"):
        mt.pp.outliers(adata, ecod_aggregation="nonsense")


def test_ecod_does_not_depend_on_which_way_round_a_feature_is_measured(adata):
    """#72: CellProfiler's feature directions are arbitrary, so the default aggregation must not
    care about them. ecod_aggregation="paper" does, and fails this."""
    flipped = adata.copy()
    flipped.X[:, ::2] *= -1
    mt.pp.outliers(adata)
    mt.pp.outliers(flipped)
    np.testing.assert_allclose(flipped.obs["qc_outlier_score"], adata.obs["qc_outlier_score"], rtol=1e-12)
