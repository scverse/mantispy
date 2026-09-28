"""Reading a CellProfiler ExportToSpreadsheet directory through read_profiles."""

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import validate


@pytest.mark.parametrize("link_on", ["child", "primary"])
def test_parent_link_is_found_on_either_table(tmp_path, make_cellprofiler_dir, link_on):
    """CellProfiler writes Parent_Nuclei on the primary table when cells came from nuclei, and Parent_Cells on the child table otherwise."""
    directory = make_cellprofiler_dir(tmp_path / link_on, link_on=link_on)
    adata = mt.io.read_profiles(directory)
    assert "Nuclei_AreaShape_Area" in adata.var_names
    assert np.isfinite(adata[:, "Nuclei_AreaShape_Area"].X).all()


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


def test_platemap_is_joined_by_well(cellprofiler_dir, platemap_path):
    adata = mt.io.read_profiles(cellprofiler_dir, platemap=platemap_path)
    mt.pp.annotate_controls(adata, negcon=("DMSO",))
    assert set(adata.obs["Metadata_Perturbation"]) == {"DMSO", "compound_a"}
    assert adata.obs["Metadata_Control"].sum() == 12
    assert adata.uns["mantispy"]["params"]["read_profiles"]["primary_object"] == "Cells"


def test_a_duplicated_platemap_row_is_refused(cellprofiler_dir, platemap_path):
    """Merging it would multiply the cells of that well and leave the object with more rows than the images produced."""
    frame = pd.read_csv(platemap_path)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="more than one row"):
        mt.io.read_profiles(cellprofiler_dir, platemap=doubled)


def test_metadata_on_both_the_object_and_image_tables_keeps_the_image_value(tmp_path, make_cellprofiler_dir):
    """CellProfiler can copy image metadata into the object tables, and the merge then suffixed the pair Metadata_Plate_x/_y, so the schema's plate column vanished silently."""
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
