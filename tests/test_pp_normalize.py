import warnings

import numpy as np
import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def adata():
    return synthetic_plate(n_plates=2, n_wells=24, n_cells=10, n_features=12, seed=0)


@pytest.mark.parametrize("method", ["standardize", "robustize", "mad_robustize"])
def test_a_feature_without_spread_among_the_controls_becomes_nan(adata, method):
    """Dividing by 1.4826 * 0 + 1e-18 left values near 1e18 that dominated every distance (#268)."""
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    control = adata.obs["Metadata_Control"].to_numpy(dtype=bool)
    first = plate == sorted(set(plate))[0]
    values = adata.X.copy()
    values[first & control, 0] = 3.0
    adata.X = values
    with pytest.warns(UserWarning, match="no spread"):
        mt.pp.normalize(adata, method=method, by="Metadata_Plate")
    assert np.isnan(adata.X[first, 0]).all()
    assert np.isfinite(adata.X[~first, 0]).all(), "the plate where the feature does vary is normalized as usual"
    assert np.isfinite(adata.X[:, 1:]).all()
    assert adata.var["degenerate_scale"].tolist() == [True] + [False] * (adata.n_vars - 1)


def test_normalize_raises_when_no_feature_can_be_scaled(adata):
    adata.X = np.ones_like(adata.X)
    with pytest.raises(ValueError, match="every feature has no spread"):
        mt.pp.normalize(adata, by="Metadata_Plate")


def test_center_leaves_a_constant_feature_at_zero(adata):
    values = adata.X.copy()
    values[:, 0] = 3.0
    adata.X = values
    mt.pp.normalize(adata, method="center", by="Metadata_Plate")
    np.testing.assert_allclose(adata.X[:, 0], 0.0, atol=1e-6)


def test_the_no_reference_warning_names_each_group_once(adata):
    # One warning per screen, naming none of them, on an 88-screen build (#278).
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    control = adata.obs["Metadata_Control"].to_numpy(dtype=bool)
    values = adata.X.copy()
    for name in sorted(set(plate)):
        values[(plate == name) & control, 1] = np.nan
    adata.X = values
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mt.pp.normalize(adata, by="Metadata_Plate")
    messages = [str(w.message) for w in caught if "no reference values" in str(w.message)]
    assert len(messages) == 1
    for name in sorted(set(plate)):
        count = int(((plate == name) & control).sum())
        assert f"{name} ({count} reference rows)" in messages[0]


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


@pytest.mark.parametrize("method", ["mad_robustize", "standardize", "robustize", "center"])
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


def test_default_reference_is_the_negative_controls(adata):
    """pp.normalize fits on the controls by default, like pp.sphere and the tl functions; reference=None fits on every row."""
    default = mt.pp.normalize(adata, by="Metadata_Plate", copy=True)
    on_controls = mt.pp.normalize(adata, by="Metadata_Plate", reference="negcon", copy=True)
    on_everything = mt.pp.normalize(adata, by="Metadata_Plate", reference=None, copy=True)
    np.testing.assert_array_equal(default.X, on_controls.X)
    assert not np.allclose(default.X, on_everything.X)


def test_normalize_reads_the_recorded_control_column():
    adata = synthetic_plate(n_plates=2, n_wells=24, n_cells=10, n_features=12, seed=0)
    adata.obs["Ctl"] = adata.obs["Metadata_Control_Type"].astype(str)
    expected = mt.pp.normalize(adata, reference="negcon", copy=True).X
    adata.obs["Metadata_Control_Type"] = "treatment"
    adata.obs["Metadata_Control"] = False
    mt.io.stamp(adata, perturbation={"column": "Metadata_Perturbation", "control_column": "Ctl"})
    assert np.allclose(mt.pp.normalize(adata, reference="negcon", copy=True).X, expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_normalize_writes_its_result_in_the_dtype_of_x(adata, dtype):
    """float32 in is float32 out and float64 in is float64 out, with no float32 rounding on the float64 path."""
    adata.X = adata.X.astype(dtype)
    values = adata.X.astype(np.float64)
    mt.pp.normalize(adata, method="standardize", by=None, reference=None)
    assert adata.X.dtype == dtype
    expected = (values - values.mean(axis=0)) / values.std(axis=0)
    tolerance = 1e-12 if dtype is np.float64 else 1e-5
    np.testing.assert_allclose(adata.X, expected, rtol=tolerance, atol=tolerance)


def test_center_subtracts_the_reference_median_and_keeps_the_units(adata):
    """method="center" removes each group's control median and leaves the spread as it was."""
    before = np.asarray(adata.X).copy()
    mt.pp.normalize(adata, method="center", by="Metadata_Plate", reference="negcon")
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    control = adata.obs["Metadata_Control_Type"].astype(str).to_numpy() == "negcon"
    for value in np.unique(plate):
        rows = plate == value
        np.testing.assert_allclose(np.median(np.asarray(adata.X)[rows & control], axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.asarray(adata.X)[rows].std(axis=0), before[rows].std(axis=0), rtol=1e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_center_keeps_the_dtype_of_x(adata, dtype):
    adata.X = adata.X.astype(dtype)
    mt.pp.normalize(adata, method="center", by="Metadata_Plate", reference="negcon")
    assert adata.X.dtype == dtype


def test_normalizing_within_groups_that_each_hold_one_treatment_warns(adata):
    """Each well holds one perturbation, so normalizing per well removes the perturbation effect."""
    with pytest.warns(UserWarning, match="Metadata_Perturbation.*constant within every group"):
        mt.pp.normalize(adata, method="center", by="Metadata_Well", reference=None)


def test_the_treatment_warning_reads_the_recorded_perturbation_column(adata):
    """A perturbation column whose name no heuristic knows is still found through the perturbation record."""
    adata.obs["Metadata_Drug"] = adata.obs["Metadata_Perturbation"].astype(str)
    mt.io.stamp(adata, perturbation={"column": "Metadata_Drug"})
    with pytest.warns(UserWarning, match="Metadata_Drug.*constant within every group"):
        mt.pp.normalize(adata, by="Metadata_Well", reference=None, copy=True)


@pytest.mark.parametrize("column", ["Metadata_pert_iname", "Metadata_JCP2022", "Metadata_mmoles_per_liter"])
def test_the_treatment_warning_knows_the_jump_column_names(adata, column):
    adata.obs[column] = adata.obs["Metadata_Perturbation"].astype(str)
    with pytest.warns(UserWarning, match=f"{column}.*constant within every group"):
        mt.pp.normalize(adata, by="Metadata_Well", reference=None, copy=True)


def test_per_plate_normalization_with_plate_level_labels_does_not_warn(adata):
    """A plate-level column whose name merely contains a treatment word is not taken for the treatment, and a
    concentration shared by a plate's treated wells but left empty on its controls is not constant there."""
    plate = adata.obs["Metadata_Plate"].astype(str)
    adata.obs["Metadata_growth_condition"] = plate
    concentration = plate.map({value: float(i + 1) for i, value in enumerate(sorted(plate.unique()))})
    adata.obs["Metadata_Concentration"] = concentration.where(~adata.obs["Metadata_Control"].astype(bool))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mt.pp.normalize(adata, by="Metadata_Plate", copy=True)
    assert not [w for w in caught if "constant within every group" in str(w.message)]
