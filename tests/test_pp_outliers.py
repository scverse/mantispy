import logging

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def adata():
    return synthetic_plate(n_wells=16, n_cells=40, n_features=15, seed=0)


@pytest.mark.parametrize("method", ["ecod", "isolation_forest", "lof", "mad"])
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
    """#72: CellProfiler's feature directions are arbitrary, so the default aggregation must not care about them.

    ecod_aggregation="paper" does, and fails this.
    """
    flipped = adata.copy()
    flipped.X[:, ::2] *= -1
    mt.pp.outliers(adata)
    mt.pp.outliers(flipped)
    np.testing.assert_allclose(flipped.obs["qc_outlier_score"], adata.obs["qc_outlier_score"], rtol=1e-12)


def test_key_picks_the_features_scored_and_key_added_names_the_output(adata):
    """`key` is the input var flag; an extreme value in a feature it leaves out cannot be flagged."""
    adata.X[0, 0] = 1e6
    adata.var["usable"] = [name != adata.var_names[0] for name in adata.var_names]
    mt.pp.outliers(adata, method="mad", score_cutoff=50.0, key="usable", key_added="flagged_usable")
    mt.pp.outliers(adata, method="mad", score_cutoff=50.0, key=None, key_added="flagged_all")
    assert not adata.obs["flagged_usable"].iloc[0]
    assert adata.obs["flagged_all"].iloc[0]


def test_outliers_join_qc_fail_so_filter_cells_drops_them(adata):
    # filter_cells reads qc_fail, so an outlier left only in qc_outlier survived filtering (#272).
    mt.pp.calculate_qc_metrics(adata)
    failed_before = adata.obs["qc_fail"].to_numpy(dtype=bool).copy()
    mt.pp.outliers(adata, method="mad", contamination=0.05)
    flagged = adata.obs["qc_outlier"].to_numpy(dtype=bool)
    assert (adata.obs["qc_fail"].to_numpy(dtype=bool) == (failed_before | flagged)).all()
    kept = mt.pp.filter_cells(adata, min_cells_per_well=0, copy=True)
    assert not kept.obs["qc_outlier"].any()


def test_lof_flags_what_sklearn_and_brieflow_flag():
    """Brieflow's intensity_filter (workflow/lib/aggregate/filter.py, v1.5.1) keeps fit_predict(...) == 1."""
    from sklearn.neighbors import LocalOutlierFactor

    adata = synthetic_plate(n_wells=25, n_cells=40, n_features=6, seed=1)
    X = np.asarray(adata.X, dtype=np.float64)
    brieflow = LocalOutlierFactor(contamination=0.01, n_jobs=-1).fit_predict(X) == -1
    mt.pp.outliers(adata, method="lof", contamination=0.01)
    assert brieflow.sum() == 10
    np.testing.assert_array_equal(adata.obs["qc_outlier"].to_numpy(), brieflow)


def test_lof_scores_a_group_of_one_or_two_rows():
    # One well of one cell and one of two, so both the single-row path and the real two-row (n_neighbors=1) path run.
    X = np.random.default_rng(0).normal(size=(3, 4))
    obs = pd.DataFrame({"Metadata_Well": ["A", "B", "B"]}, index=["c0", "c1", "c2"])
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"f{j}" for j in range(4)]))
    assert sorted(adata.obs["Metadata_Well"].value_counts().tolist()) == [1, 2]
    mt.pp.outliers(adata, method="lof", by="Metadata_Well")
    assert np.isfinite(adata.obs["qc_outlier_score"]).all()
    assert adata.obs["qc_outlier"].sum() == 2  # one flagged per group


def test_lof_excludes_all_missing_rows_and_still_flags_real_outliers(caplog):
    """#2: filling all-missing rows with zero collapses them onto the origin, inflating nearby cells' LOF so the
    real outliers go unflagged. They must be excluded from the fit, left unscored, and the true outliers flagged."""
    rng = np.random.default_rng(0)
    n_core, n_missing, n_outliers, n_vars = 200, 30, 6, 6
    core = rng.normal(size=(n_core, n_vars))
    far = rng.normal(size=(n_outliers, n_vars)) + 15.0
    gone = np.full((n_missing, n_vars), np.nan)
    X = np.vstack([core, far, gone])
    obs = pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"f{j}" for j in range(n_vars)]))
    outlier_rows = np.arange(n_core, n_core + n_outliers)
    missing_rows = np.arange(n_core + n_outliers, X.shape[0])

    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.pp.outliers(adata, method="lof", contamination=0.03)

    flagged = adata.obs["qc_outlier"].to_numpy()
    scores = adata.obs["qc_outlier_score"].to_numpy()
    assert flagged[outlier_rows].all()  # the genuine outliers are flagged
    assert not flagged[missing_rows].any()  # the all-missing rows are not
    assert np.isnan(scores[missing_rows]).all()  # and are left unscored
    assert np.isfinite(scores[: n_core + n_outliers]).all()
    assert f"{n_missing} cell" in caplog.text
