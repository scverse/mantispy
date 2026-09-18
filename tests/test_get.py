import numpy as np
import pytest

import mantispy as mt


def test_features_filters_combine(cells):
    names = mt.get.features(cells, object="Cells", feature_group="Intensity", channel="DNA")
    assert names
    for name in names:
        row = cells.var.loc[name]
        assert (row["object"], row["feature_group"]) == ("Cells", "Intensity")
        assert "DNA" in str(row["channel"]).split("|")
    assert len(mt.get.features(cells)) == cells.n_vars


def test_features_key_restricts(cells):
    cells.var["selected"] = False
    cells.var.iloc[:3, cells.var.columns.get_loc("selected")] = True
    assert len(mt.get.features(cells, key="selected")) == 3
    with pytest.raises(KeyError, match="nope"):
        mt.get.features(cells, key="nope")


def test_canonical_channels_match_across_naming_conventions():
    """A dataset calling its nuclear stain Hoechst should answer to channel='DNA'."""
    from mantispy.ds import synthetic_plate

    adata = synthetic_plate(n_wells=8, n_cells=4, n_features=12, channels=["Hoechst", "GFP"], seed=0)
    assert mt.get.features(adata, channel="DNA") == []
    assert mt.get.features(adata, channel="DNA", canonical_channels=True)


def test_to_dataframe_shape_and_column_order(cells):
    frame = mt.get.to_dataframe(cells)
    metadata = [c for c in frame.columns if c.startswith("Metadata_")]
    assert list(frame.columns[: len(metadata)]) == metadata
    assert len(frame) == cells.n_obs
    np.testing.assert_allclose(frame[cells.var_names[0]].to_numpy(), cells.X[:, 0], rtol=1e-6)

    assert list(mt.get.to_dataframe(cells, metadata=False).columns) == list(cells.var_names)


def test_to_dataframe_subsets_features_and_reads_layers(cells):
    cells.layers["doubled"] = cells.X * 2
    subset = mt.get.to_dataframe(cells, layer="doubled", features=list(cells.var_names[:2]))
    assert list(subset.columns)[-2:] == list(cells.var_names[:2])
    np.testing.assert_allclose(subset[cells.var_names[0]].to_numpy(), cells.X[:, 0] * 2, rtol=1e-6)


def test_controls(cells):
    np.testing.assert_array_equal(mt.get.controls(cells), cells.obs["Metadata_Control"].to_numpy())
    assert not mt.get.controls(cells, kind="poscon").any()
    with pytest.raises(ValueError, match="negcon"):
        mt.get.controls(cells, kind="nonsense")


def test_scanpy_reexports_are_available():
    assert callable(mt.get.obs_df) and callable(mt.get.var_df)


def test_importing_mantispy_does_not_import_scanpy():
    """The re-exports are fetched on first access because importing scanpy pulls in sklearn and matplotlib.

    A plain import that reached scanpy would pay that cost on every ``import mantispy``, which is what the lazy lookup exists to avoid.
    """
    import subprocess
    import sys

    probe = subprocess.run(
        [sys.executable, "-c", "import sys, mantispy; print('scanpy' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False", probe.stdout


def test_an_unknown_name_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="nope"):
        _ = mt.get.nope
