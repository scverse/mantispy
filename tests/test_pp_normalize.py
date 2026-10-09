import numpy as np
import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def adata():
    return synthetic_plate(n_plates=2, n_wells=24, n_cells=10, n_features=12, seed=0)


@pytest.mark.parametrize("method", ["standardize", "robustize", "mad_robustize"])
def test_constant_feature_yields_zero_not_inf(adata, method):
    """An unguarded zero scale would make the whole column inf or NaN."""
    values = adata.X.copy()
    values[:, 0] = 3.0
    adata.X = values
    mt.pp.normalize(adata, method=method, by="Metadata_Plate")
    assert np.isfinite(adata.X).all()
    np.testing.assert_allclose(adata.X[:, 0], 0.0, atol=1e-6)


def test_reference_rejects_a_non_boolean_column(adata):
    """A string column read as bool is all True, which would fit on every row."""
    adata.obs["treatment"] = "trt"
    with pytest.raises(TypeError, match="must be boolean"):
        mt.pp.normalize(adata, reference="treatment")


def test_missing_control_column_points_at_annotate_controls(adata):
    adata.obs = adata.obs.drop(columns="Metadata_Control_Type")
    with pytest.raises(KeyError, match="annotate_controls"):
        mt.pp.normalize(adata, reference="negcon")


def test_group_without_reference_rows_names_the_group(adata):
    obs = adata.obs.copy()
    obs.loc[obs["Metadata_Plate"] == "Plate01", "Metadata_Control_Type"] = "treatment"
    adata.obs = obs
    with pytest.raises(ValueError, match="Plate01"):
        mt.pp.normalize(adata, by="Metadata_Plate", reference="negcon")


def test_reference_column_with_missing_values_is_refused(adata):
    """A named boolean reference column with NaN coerces to True, which would select those rows as controls."""
    obs = adata.obs.copy()
    flag = obs["Metadata_Control"].astype(object)
    flag.iloc[:5] = None
    obs["held_out"] = flag
    adata.obs = obs
    with pytest.raises(ValueError, match="missing value"):
        mt.pp.normalize(adata, reference="held_out")


def test_string_true_false_survives_a_round_trip(adata):
    """h5ad can bring a bool column back as a category of 'True'/'False' strings."""
    obs = adata.obs.copy()
    obs["Metadata_Control"] = obs["Metadata_Control"].astype(str).astype("category")
    adata.obs = obs
    mt.pp.normalize(adata, by="Metadata_Plate", reference="negcon")
    controls = np.asarray(adata[adata.obs["Metadata_Control"] == "True"].X)
    np.testing.assert_allclose(np.nanmedian(controls, axis=0), 0.0, atol=1e-5)


@pytest.mark.parametrize("method", ["mad_robustize", "standardize", "robustize"])
def test_a_feature_unmeasured_among_one_groups_controls_keeps_its_measured_values(adata, method):
    """Repairing the zero scale to 1.0 but leaving the centre NaN turns every measured value in that group into NaN: the first plate's treated wells went from [-0.62, -0.22, -0.54] to all-NaN, 6 of 6 rows on that plate against 0 of 6 on the other."""
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    control = adata.obs["Metadata_Control"].to_numpy(dtype=bool)
    first = sorted(set(plate))[0]

    values = adata.X.copy()
    values[(plate == first) & control, 1] = np.nan
    adata.X = values
    measured = (plate == first) & ~control
    assert np.isfinite(values[measured, 1]).all(), "the feature is measured outside the controls"

    with pytest.warns(UserWarning, match="no reference values"):
        mt.pp.normalize(adata, method=method, by="Metadata_Plate", reference="negcon")

    assert np.isfinite(np.asarray(adata.X)[measured, 1]).all()
    assert adata.var["degenerate_scale"].to_numpy()[1], "and the feature is still flagged"
