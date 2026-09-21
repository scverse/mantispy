"""Reading a CellProfiler ExportToSpreadsheet directory through read_profiles."""

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import validate


def test_reads_joins_and_validates(cellprofiler_dir):
    adata = mt.io.read_profiles(cellprofiler_dir)
    assert adata.n_obs == 24
    assert adata.X.dtype == np.float32
    assert validate(adata).ok, validate(adata).errors
    assert {"Cells_AreaShape_Area", "Nuclei_AreaShape_Area"} <= set(adata.var_names)
    assert adata.uns["mantispy"]["resolution"] == "cell"


@pytest.mark.parametrize("prefix", ["", "MyExpt_"], ids=["unprefixed", "prefixed"])
def test_the_file_name_prefix_of_a_run_is_found(tmp_path, make_cellprofiler_dir, prefix):
    """ExportToSpreadsheet puts the prefix a run was configured with in front of every file."""
    adata = mt.io.read_profiles(make_cellprofiler_dir(tmp_path / "run", prefix=prefix))
    assert adata.n_obs == 24
    assert "Nuclei_AreaShape_Area" in adata.var_names


@pytest.mark.parametrize("link_on", ["child", "primary"])
def test_parent_link_is_found_on_either_table(tmp_path, make_cellprofiler_dir, link_on):
    """CellProfiler writes Parent_Nuclei on the primary table when cells came from nuclei,
    and Parent_Cells on the child table otherwise."""
    directory = make_cellprofiler_dir(tmp_path / link_on, link_on=link_on)
    adata = mt.io.read_profiles(directory)
    assert "Nuclei_AreaShape_Area" in adata.var_names
    assert np.isfinite(adata[:, "Nuclei_AreaShape_Area"].X).all()


def test_objects_are_paired_through_the_link_not_the_object_number(tmp_path, make_cellprofiler_dir):
    """A link that does not follow the object numbers must still pair each cell with its own nucleus."""
    directory = make_cellprofiler_dir(tmp_path / "reversed", link_on="primary")
    cells = pd.read_csv(directory / "Cells.csv")
    cells["Parent_Nuclei"] = cells.groupby("ImageNumber")["ObjectNumber"].transform(lambda s: s.to_numpy()[::-1])
    cells.to_csv(directory / "Cells.csv", index=False)
    nuclei = pd.read_csv(directory / "Nuclei.csv").set_index(["ImageNumber", "ObjectNumber"])["AreaShape_Area"]

    adata = mt.io.read_profiles(directory)
    expected = nuclei.loc[pd.MultiIndex.from_arrays([cells["ImageNumber"], cells["Parent_Nuclei"]])].to_numpy()
    np.testing.assert_allclose(np.asarray(adata[:, "Nuclei_AreaShape_Area"].X).ravel(), expected, rtol=1e-6)


def test_missing_link_raises_rather_than_pairing_by_object_number(tmp_path, make_cellprofiler_dir):
    """Merging on ObjectNumber alone would pair unrelated objects with no error."""
    directory = make_cellprofiler_dir(tmp_path / "nolink", link_on="child")
    nuclei = pd.read_csv(directory / "Nuclei.csv").drop(columns=["Parent_Cells"])
    nuclei.to_csv(directory / "Nuclei.csv", index=False)
    with pytest.raises(ValueError, match="cannot link"):
        mt.io.read_profiles(directory)


def test_non_one_to_one_raises_and_can_be_waived(tmp_path, make_cellprofiler_dir):
    directory = make_cellprofiler_dir(tmp_path / "dup", one_to_one=False)
    with pytest.raises(ValueError, match="not one-to-one"):
        mt.io.read_profiles(directory)
    assert mt.io.read_profiles(directory, strict_one_to_one=False).n_obs == 24


def test_a_primary_object_without_a_child_is_detected(tmp_path, make_cellprofiler_dir):
    """Counting over the child frame misses parents that have zero children."""
    directory = make_cellprofiler_dir(tmp_path / "orphan")
    nuclei = pd.read_csv(directory / "Nuclei.csv")
    nuclei = nuclei[~((nuclei["ImageNumber"] == 1) & (nuclei["Parent_Cells"] == 1))]
    nuclei.to_csv(directory / "Nuclei.csv", index=False)
    with pytest.raises(ValueError, match="have no"):
        mt.io.read_profiles(directory)


def test_non_features_stay_out_of_x(cellprofiler_dir):
    adata = mt.io.read_profiles(cellprofiler_dir)
    for name in [
        "Cells_Location_Center_X",
        "Cells_Number_Object_Number",
        "Cells_Children_Nuclei_Count",
        "Nuclei_Parent_Cells",
    ]:
        assert name not in adata.var_names
    assert not any("ImageQuality" in name for name in adata.var_names)


def test_centroids_are_kept_in_obs(cellprofiler_dir):
    """qc_is_border needs them, so dropping them entirely would make it inert."""
    adata = mt.io.read_profiles(cellprofiler_dir)
    assert {"Metadata_Center_X", "Metadata_Center_Y"} <= set(adata.obs.columns)
    assert adata.obs["Metadata_Center_X"].between(0, 1024).all()


def test_a_cellprofiler_4_centroid_is_kept_in_obs_and_out_of_x(cellprofiler_dir):
    """CellProfiler 4 writes the centroid under AreaShape, where it was read as a morphology feature."""
    cells = pd.read_csv(cellprofiler_dir / "Cells.csv")
    cells.columns = [column.replace("Location_Center", "AreaShape_Center") for column in cells.columns]
    cells.to_csv(cellprofiler_dir / "Cells.csv", index=False)

    adata = mt.io.read_profiles(cellprofiler_dir)
    assert adata.obs["Metadata_Center_X"].between(0, 1024).all()
    assert not any("Center" in name for name in adata.var_names)


def test_metadata_and_well_normalization(tmp_path, make_cellprofiler_dir):
    directory = make_cellprofiler_dir(tmp_path / "wells")
    image = pd.read_csv(directory / "Image.csv")
    image["Metadata_Well"] = ["a1", "a1", "a2", "a2"]
    image.to_csv(directory / "Image.csv", index=False)
    adata = mt.io.read_profiles(directory)
    assert set(adata.obs["Metadata_Well"]) == {"A01", "A02"}
    assert adata.obs["Metadata_Plate"].unique().tolist() == ["P1"]
    assert {"Metadata_Site", "Metadata_ImageNumber"} <= set(adata.obs.columns)


def test_image_table_carries_plate_keys(cellprofiler_dir):
    """Without them pp.image_qc silently pools every plate instead of thresholding per plate."""
    table = mt.io.read_profiles(cellprofiler_dir).uns["mantispy"]["image_table"]
    assert isinstance(table, pd.DataFrame) and len(table) == 4
    assert any("ImageQuality" in column for column in table.columns)
    assert {"Metadata_Plate", "Metadata_Well"} <= set(table.columns)


def test_var_is_parsed_and_channels_inferred(cellprofiler_dir):
    adata = mt.io.read_profiles(cellprofiler_dir)
    row = adata.var.loc["Cells_Intensity_MeanIntensity_DNA"]
    assert (row["object"], row["feature_group"], row["channel"]) == ("Cells", "Intensity", "DNA")
    assert set(adata.uns["mantispy"]["channels"]) == {"DNA", "ER"}


def test_arbitrary_channel_names_work(tmp_path, make_cellprofiler_dir):
    directory = make_cellprofiler_dir(tmp_path / "alt", channels=["Hoechst", "GFP"])
    adata = mt.io.read_profiles(directory)
    assert set(adata.uns["mantispy"]["channels"]) == {"Hoechst", "GFP"}
    assert adata.var.loc["Cells_Intensity_MeanIntensity_Hoechst", "channel"] == "Hoechst"


def test_platemap_is_joined_by_well(cellprofiler_dir, platemap_path):
    adata = mt.io.read_profiles(cellprofiler_dir, platemap=platemap_path)
    mt.pp.annotate_controls(adata, negcon=("DMSO",))
    assert set(adata.obs["Metadata_Perturbation"]) == {"DMSO", "compound_a"}
    assert adata.obs["Metadata_Control"].sum() == 12
    assert adata.uns["mantispy"]["params"]["read_profiles"]["primary_object"] == "Cells"


def test_a_duplicated_platemap_row_is_refused(cellprofiler_dir, platemap_path):
    """Merging it would multiply the cells of that well and leave the object with more
    rows than the images produced."""
    frame = pd.read_csv(platemap_path)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="more than one row"):
        mt.io.read_profiles(cellprofiler_dir, platemap=doubled)


def test_metadata_on_both_the_object_and_image_tables_keeps_the_image_value(tmp_path, make_cellprofiler_dir):
    """CellProfiler can copy image metadata into the object tables, and the merge then
    suffixed the pair Metadata_Plate_x/_y, so the schema's plate column vanished silently."""
    directory = make_cellprofiler_dir(tmp_path / "collide")
    cells = pd.read_csv(directory / "Cells.csv")
    cells["Metadata_Plate"] = "from_the_object_table"
    cells.to_csv(directory / "Cells.csv", index=False)

    adata = mt.io.read_profiles(directory)

    assert not [column for column in adata.obs.columns if column.endswith(("_x", "_y"))]
    assert set(adata.obs["Metadata_Plate"]) == {"P1"}
    assert validate(adata).ok, validate(adata).errors


def test_several_directories_are_refused(tmp_path, make_cellprofiler_dir):
    """Both number their images from 1, and image_qc joins the image table on ImageNumber alone."""
    directories = [make_cellprofiler_dir(tmp_path / name) for name in ("a", "b")]
    with pytest.raises(ValueError, match="one directory at a time"):
        mt.io.read_profiles(directories)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [({"primary_object": "Nope"}, FileNotFoundError, "no Nope.csv"), ({"objects": ["Nope"]}, ValueError, "none of")],
)
def test_an_export_that_cannot_be_read_says_why(cellprofiler_dir, kwargs, error, match):
    with pytest.raises(error, match=match):
        mt.io.read_profiles(cellprofiler_dir, **kwargs)
