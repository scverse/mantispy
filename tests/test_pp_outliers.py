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


@pytest.mark.parametrize("method", ["ecod", "isolation_forest", "mad"])
def test_contamination_controls_the_flagged_fraction(adata, method):
    """It must mean the same thing for every method, including mad."""
    mt.pp.outliers(adata, method=method, contamination=0.10)
    assert 0.05 < adata.obs["qc_outlier"].mean() < 0.15


def test_injected_extremes_are_flagged(adata):
    values = adata.X.copy()
    values[:10] += 50.0
    adata.X = values
    mt.pp.outliers(adata, method="ecod", contamination=0.02)
    assert adata.obs["qc_outlier"].to_numpy()[:10].all()


def test_score_cutoff_thresholds_absolutely(adata):
    """With method='mad' the score is a robust z, so 5 is the familiar rule."""
    mt.pp.outliers(adata, method="mad", score_cutoff=5.0)
    assert (adata.obs["qc_outlier_score"][adata.obs["qc_outlier"]] > 5.0).all()
    mt.pp.outliers(adata, method="mad", score_cutoff=1e9, key_added="none_flagged")
    assert not adata.obs["none_flagged"].any()


def test_by_thresholds_within_each_group(adata):
    values = adata.X.copy()
    values[(adata.obs["Metadata_Plate"] == "Plate01").to_numpy()] += 100.0
    adata.X = values
    mt.pp.outliers(adata, method="ecod", contamination=0.05, by="Metadata_Plate")
    per_plate = adata.obs.groupby("Metadata_Plate", observed=True)["qc_outlier"].mean()
    assert per_plate.max() - per_plate.min() < 0.05


def test_uses_selected_features_when_present(adata):
    adata.var["selected"] = False
    adata.var.iloc[:3, adata.var.columns.get_loc("selected")] = True
    mt.pp.outliers(adata, method="mad")
    assert adata.uns["mantispy"]["params"]["outliers"]["key"] == "selected"


def test_rejects_a_bad_method_or_contamination(adata):
    with pytest.raises(ValueError, match="method must be"):
        mt.pp.outliers(adata, method="nonsense")
    with pytest.raises(ValueError, match="contamination"):
        mt.pp.outliers(adata, contamination=1.5)


def test_an_infinite_feature_value_is_flagged_not_hidden(adata):
    """CellProfiler ratio features produce inf. Mapping it to zero would score the most
    extreme cell as the most ordinary one."""
    values = adata.X.copy()
    values[7, 2] = np.inf
    adata.X = values
    mt.pp.outliers(adata, method="mad", contamination=0.05)
    assert adata.obs["qc_outlier"].to_numpy()[7]
    assert np.isinf(adata.obs["qc_outlier_score"].to_numpy()[7])


def test_contamination_bounds_the_flagged_fraction_within_small_groups(adata):
    """Rounding the per-group count up flags at least one cell in every group, so
    contamination stopped meaning anything below 1/group_size: 0.01 and 0.001 both flagged
    24 of 360 cells with by='Metadata_Well', a 6.7x overshoot."""
    mt.pp.outliers(adata, method="ecod", contamination=0.01, by="Metadata_Well")
    sparse = int(adata.obs["qc_outlier"].sum())
    mt.pp.outliers(adata, method="ecod", contamination=0.10, by="Metadata_Well")
    dense = int(adata.obs["qc_outlier"].sum())

    assert sparse <= 0.02 * adata.n_obs, "contamination has to bound the flagged fraction"
    assert sparse < dense
