import numpy as np
import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def adata():
    return synthetic_plate(n_plates=2, n_wells=24, n_cells=10, n_features=12, seed=0)


def _per_plate(adata, fn):
    expected = adata.X.astype(np.float64).copy()
    for plate in adata.obs["Metadata_Plate"].unique():
        rows = (adata.obs["Metadata_Plate"] == plate).to_numpy()
        expected[rows] = fn(expected[rows])
    return expected


def test_mad_robustize_centres_each_plate(adata):
    mt.pp.normalize(adata, method="mad_robustize", by="Metadata_Plate")
    for plate in adata.obs["Metadata_Plate"].unique():
        # np.asarray: nanmedian writes through an AnnData view's .X, corrupting it
        block = np.asarray(adata[adata.obs["Metadata_Plate"] == plate].X)
        np.testing.assert_allclose(np.nanmedian(block, axis=0), 0.0, atol=1e-5)


def test_standardize_uses_the_population_sd(adata):
    """pycytominer standardises with sklearn's StandardScaler, which is ddof=0."""
    expected = _per_plate(adata, lambda b: (b - np.nanmean(b, axis=0)) / np.nanstd(b, axis=0, ddof=0))
    mt.pp.normalize(adata, method="standardize", by="Metadata_Plate")
    np.testing.assert_allclose(adata.X, expected, rtol=1e-4, atol=1e-5)


def test_robustize_uses_the_iqr(adata):
    def scale(block):
        q75, q25 = np.nanquantile(block, 0.75, axis=0), np.nanquantile(block, 0.25, axis=0)
        return (block - np.nanmedian(block, axis=0)) / (q75 - q25)

    expected = _per_plate(adata, scale)
    mt.pp.normalize(adata, method="robustize", by="Metadata_Plate")
    np.testing.assert_allclose(adata.X, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("method", ["standardize", "robustize", "mad_robustize"])
def test_constant_feature_yields_zero_not_inf(adata, method):
    """An unguarded zero scale would make the whole column inf or NaN."""
    values = adata.X.copy()
    values[:, 0] = 3.0
    adata.X = values
    mt.pp.normalize(adata, method=method, by="Metadata_Plate")
    assert np.isfinite(adata.X).all()
    np.testing.assert_allclose(adata.X[:, 0], 0.0, atol=1e-6)


def test_reference_negcon_centres_the_controls(adata):
    mt.pp.normalize(adata, by="Metadata_Plate", reference="negcon")
    controls = adata[adata.obs["Metadata_Control"]]
    for plate in controls.obs["Metadata_Plate"].unique():
        block = np.asarray(controls[controls.obs["Metadata_Plate"] == plate].X)
        np.testing.assert_allclose(np.nanmedian(block, axis=0), 0.0, atol=1e-5)


def test_reference_rejects_a_non_boolean_column(adata):
    """A string column read as bool is all True, which would fit on every row."""
    adata.obs["treatment"] = "trt"
    with pytest.raises(TypeError, match="must be boolean"):
        mt.pp.normalize(adata, reference="treatment")


def test_missing_control_column_points_at_annotate_controls(adata):
    adata.obs = adata.obs.drop(columns="Metadata_Control")
    with pytest.raises(KeyError, match="annotate_controls"):
        mt.pp.normalize(adata, reference="negcon")


def test_group_without_reference_rows_names_the_group(adata):
    obs = adata.obs.copy()
    obs.loc[obs["Metadata_Plate"] == "Plate01", "Metadata_Control"] = False
    adata.obs = obs
    with pytest.raises(ValueError, match="Plate01"):
        mt.pp.normalize(adata, by="Metadata_Plate", reference="negcon")


def test_by_none_normalizes_globally(adata):
    mt.pp.normalize(adata, by=None)
    np.testing.assert_allclose(np.nanmedian(adata.X, axis=0), 0.0, atol=1e-5)


def test_key_added_leaves_x_untouched_and_keep_raw_is_off_by_default(adata):
    before = adata.X.copy()
    mt.pp.normalize(adata, key_added="normalized")
    np.testing.assert_array_equal(adata.X, before)
    assert adata.layers["normalized"].shape == adata.shape
    assert "raw" not in adata.layers

    mt.pp.normalize(adata, keep_raw=True)
    np.testing.assert_array_equal(adata.layers["raw"], before)


def test_output_stays_float32_and_records_params(adata):
    mt.pp.normalize(adata, method="standardize")
    assert adata.X.dtype == np.float32
    assert adata.uns["mantispy"]["params"]["normalize"]["method"] == "standardize"


def test_nan_is_preserved_not_invented(adata):
    values = adata.X.copy()
    values[0, 0] = np.nan
    adata.X = values
    mt.pp.normalize(adata)
    assert np.isnan(adata.X[0, 0])
    assert int(np.isnan(adata.X).sum()) == 1


def test_reference_column_with_missing_values_is_refused(adata):
    """NaN coerces to True, which would select those rows as controls."""
    obs = adata.obs.copy()
    control = obs["Metadata_Control"].astype(object)
    control.iloc[:5] = None
    obs["Metadata_Control"] = control
    adata.obs = obs
    with pytest.raises(ValueError, match="missing value"):
        mt.pp.normalize(adata, reference="negcon")


def test_string_true_false_survives_a_round_trip(adata):
    """h5ad can bring a bool column back as a category of 'True'/'False' strings."""
    obs = adata.obs.copy()
    obs["Metadata_Control"] = obs["Metadata_Control"].astype(str).astype("category")
    adata.obs = obs
    mt.pp.normalize(adata, by="Metadata_Plate", reference="negcon")
    controls = np.asarray(adata[adata.obs["Metadata_Control"] == "True"].X)
    np.testing.assert_allclose(np.nanmedian(controls, axis=0), 0.0, atol=1e-5)


def test_a_feature_constant_among_the_controls_is_flagged(cells):
    """mad_robustize divides by epsilon rather than by zero, which multiplies such a
    feature by 1e18 and lets it dominate every distance downstream. Real screens have
    these: 65 of rohban's 3634 features, 129 of pki's 5857."""
    values = cells.X.copy()
    control = cells.obs["Metadata_Control"].to_numpy(dtype=bool)
    values[control, 3] = 7.0  # constant among the controls, varying elsewhere
    cells.X = values

    with pytest.warns(UserWarning, match="no spread"):
        mt.pp.normalize(cells, method="mad_robustize", by="Metadata_Plate", reference="negcon")

    assert cells.var["degenerate_scale"].to_numpy()[3]
    assert cells.var["degenerate_scale"].sum() < cells.n_vars
    assert np.abs(np.asarray(cells.X)[:, 3]).max() > 1e6  # the multiplier the warning describes
    kept = cells[:, ~cells.var["degenerate_scale"].to_numpy()]
    assert np.abs(np.asarray(kept.X)).max() < 1e6


def test_a_feature_both_constant_and_unmeasured_reports_both_conditions(adata):
    """Suppressing the spread warning whenever a feature was also uncentred somewhere hid the
    1e18 multiplication exactly when it was happening: a feature constant among the first
    plate's controls and unmeasured among the second's left the first plate's treated rows at
    1.0e19, with only the centring warning to show for it."""
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    control = adata.obs["Metadata_Control"].to_numpy(dtype=bool)
    first, second = sorted(set(plate))

    values = adata.X.copy()
    values[(plate == first) & control, 3] = 7.0  # constant among these controls, varying outside them
    values[(plate == second) & control, 3] = np.nan  # never measured among those
    adata.X = values
    treated = (plate == first) & ~control

    with pytest.warns(UserWarning) as record:
        mt.pp.normalize(adata, method="mad_robustize", by="Metadata_Plate", reference="negcon")
    messages = [str(warning.message) for warning in record]

    assert any("no spread" in message and "1e18" in message for message in messages), "the multiplication"
    assert any("no reference values" in message for message in messages), "and the missing centre"
    assert adata.var["degenerate_scale"].to_numpy()[3]
    assert np.abs(np.asarray(adata.X)[treated, 3]).max() > 1e6


@pytest.mark.parametrize("method", ["mad_robustize", "standardize", "robustize"])
def test_a_feature_unmeasured_among_one_groups_controls_keeps_its_measured_values(adata, method):
    """Repairing the zero scale to 1.0 but leaving the centre NaN turns every measured value
    in that group into NaN: the first plate's treated wells went from [-0.62, -0.22, -0.54]
    to all-NaN, 6 of 6 rows on that plate against 0 of 6 on the other."""
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


def test_two_layers_keep_their_own_degenerate_flags(adata):
    """One unsuffixed flag column for every layer lets a second call unflag a feature whose
    value in the first layer is 8.0e18, so the remedy the docstrings give (drop the flagged
    features) keeps it."""
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    control = adata.obs["Metadata_Control"].to_numpy(dtype=bool)

    values = adata.X.copy()
    for name in sorted(set(plate)):
        rows = np.flatnonzero((plate == name) & control)
        # A quantized feature: most controls share one value, so the MAD is 0 while the SD is not.
        values[rows, 3] = 4.0
        values[rows[: rows.size // 5], 3] = 12.0
        values[rows[rows.size // 5 : 2 * (rows.size // 5)], 3] = -4.0
    adata.X = values

    with pytest.warns(UserWarning, match="no spread"):
        mt.pp.normalize(adata, method="mad_robustize", by="Metadata_Plate", reference="negcon", key_added="one")
    assert adata.var["degenerate_scale_one"].to_numpy()[3]
    assert np.abs(np.asarray(adata.layers["one"])[:, 3]).max() > 1e6

    mt.pp.normalize(adata, method="standardize", by="Metadata_Plate", reference="negcon", key_added="two")
    assert adata.var["degenerate_scale_one"].to_numpy()[3], "the second call unflagged the first layer"
    assert not adata.var["degenerate_scale_two"].to_numpy()[3]
