import numpy as np
import pytest

from mantispy._core.schema import validate
from mantispy.ds import synthetic_plate


def test_shape_schema_and_determinism():
    adata = synthetic_plate(n_wells=24, n_cells=10, n_features=20, seed=0)
    assert adata.shape == (240, 20)
    assert adata.X.dtype == np.float32
    assert validate(adata).ok, validate(adata).errors
    np.testing.assert_array_equal(adata.X, synthetic_plate(n_wells=24, n_cells=10, n_features=20, seed=0).X)


def test_controls_exist_and_are_unshifted():
    adata = synthetic_plate(n_wells=48, n_cells=20, n_features=20, effect_size=5.0, seed=0)
    assert adata.obs["Metadata_Control"].sum() > 0
    assert abs(float(np.mean(adata[adata.obs["Metadata_Control"]].X))) < 0.2


def test_injected_perturbation_effect_is_recoverable():
    adata = synthetic_plate(n_wells=48, n_cells=50, n_features=20, n_perturbations=2, effect_size=3.0, seed=0)
    truth = adata.uns["mantispy"]["truth"]["affected_features"]
    perturbation = next(p for p, features in truth.items() if features)
    feature = truth[perturbation][0]
    treated = adata[adata.obs["Metadata_Perturbation"] == perturbation, feature].X.ravel()
    control = adata[adata.obs["Metadata_Control"], feature].X.ravel()
    assert float(np.mean(treated) - np.mean(control)) > 1.5


@pytest.mark.parametrize("channels", [["Hoechst", "GFP"], ["w1"], list("ABCDEFG")])
def test_any_channel_vocabulary_works(channels):
    """Cell Painting channel names are the default, and other vocabularies work too."""
    adata = synthetic_plate(n_wells=8, n_cells=5, n_features=12, channels=channels, seed=0)
    assert adata.uns["mantispy"]["channels"] == channels
    observed = set(adata.var["channel"].dropna().unique())
    assert observed and observed <= set(channels)
    assert validate(adata).ok


def test_too_many_features_for_the_channel_set_raises():
    with pytest.raises(ValueError, match="at most"):
        synthetic_plate(n_wells=4, n_cells=2, n_features=1000, channels=["only"])


def test_image_table_is_written_and_keyed_to_cells():
    adata = synthetic_plate(n_wells=8, n_cells=10, n_features=10, n_images_per_well=2, seed=0)
    table = adata.uns["mantispy"]["image_table"]
    assert len(table) == 16
    assert set(adata.obs["Metadata_ImageNumber"]) == set(table.index)
    # image QC needs the plate column to threshold per plate rather than pooling
    assert {"Metadata_Plate", "Metadata_Well"} <= set(table.columns)


def test_bad_images_are_recorded_and_extreme():
    adata = synthetic_plate(n_wells=16, n_cells=10, n_features=10, n_images_per_well=2, n_bad_images=3, seed=0)
    bad = adata.uns["mantispy"]["truth"]["bad_images"]
    focus = adata.uns["mantispy"]["image_table"]["Image_ImageQuality_FocusScore_DNA"]
    assert len(bad) == 3
    assert focus.loc[bad].mean() < focus.drop(index=bad).mean()


def test_correlated_and_constant_features():
    adata = synthetic_plate(n_wells=8, n_cells=50, n_features=20, n_correlated_pairs=2, n_constant_features=2, seed=0)
    for original, copy in adata.uns["mantispy"]["truth"]["correlated_pairs"]:
        assert abs(np.corrcoef(adata[:, original].X.ravel(), adata[:, copy].X.ravel())[0, 1]) > 0.95
    for name in adata.uns["mantispy"]["truth"]["constant_features"]:
        assert np.nanstd(adata[:, name].X) == 0.0


def test_position_gradients_and_batch_effects_are_injected():
    adata = synthetic_plate(n_wells=96, n_cells=5, n_features=10, row_gradient=2.0, seed=0)
    rows = adata.obs["Metadata_Row"].to_numpy()
    assert abs(np.corrcoef(rows, adata.X[:, 0])[0, 1]) > 0.4

    batched = synthetic_plate(n_plates=4, n_wells=8, n_cells=10, n_features=10, n_batches=2, batch_effect=5.0, seed=0)
    means = batched.obs.assign(v=batched.X[:, 0]).groupby("Metadata_Batch", observed=True)["v"].mean()
    assert abs(means.iloc[0] - means.iloc[1]) > 1.0
    assert len(batched.uns["mantispy"]["truth"]["batch_offsets"]) == 2


def test_confounder_effect_correlates_with_cell_count():
    adata = synthetic_plate(n_wells=48, n_cells=30, n_features=10, confounder_effect=3.0, seed=0)
    confounded = adata.uns["mantispy"]["truth"]["confounded_features"]
    counts = adata.obs["Metadata_CellCount"].to_numpy(dtype=float)
    assert abs(np.corrcoef(counts, adata[:, confounded[0]].X.ravel())[0, 1]) > 0.3


def test_nan_fraction():
    adata = synthetic_plate(n_wells=8, n_cells=10, n_features=20, nan_fraction=0.1, seed=0)
    assert 0.05 < float(np.isnan(adata.X).mean()) < 0.15
