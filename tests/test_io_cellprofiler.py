"""Reading a CellProfiler ExportToSpreadsheet directory through read_profiles."""

import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import validate

#: Slices of real exports of the CellProfiler tutorials and of PERISCOPE (cpg0021, plate CP257A, Well1 site 21):
#: the first images and a few linked objects of each table, with absolute paths scrubbed.
EXPORTS = Path(__file__).parent / "data" / "cellprofiler"


@pytest.mark.parametrize("link_on", ["child", "primary"])
def test_parent_link_is_found_on_either_table(tmp_path, make_cellprofiler_dir, link_on):
    """CellProfiler writes Parent_Nuclei on the primary table when cells came from nuclei, and Parent_Cells on the child table otherwise."""
    directory = make_cellprofiler_dir(tmp_path / link_on, link_on=link_on)
    adata = mt.io.read_profiles(directory)
    assert "Nuclei_AreaShape_Area" in adata.var_names
    assert np.isfinite(adata[:, "Nuclei_AreaShape_Area"].X).all()


def test_objects_linked_only_through_a_shared_child_are_joined(tmp_path, make_cellprofiler_dir):
    """JUMP gives Cytoplasm a Parent_Cells and a Parent_Nuclei and Cells no parent, so Cells and Nuclei are related only through it (#61)."""
    directory = make_cellprofiler_dir(tmp_path / "jump", link_on="child")
    cells = pd.read_csv(directory / "Cells.csv")
    nuclei = pd.read_csv(directory / "Nuclei.csv")
    n = cells["ObjectNumber"].max()
    # The nucleus of cell k is number n + 1 - k, so pairing by object number would be wrong.
    nuclei["ObjectNumber"] = n + 1 - nuclei["ObjectNumber"]
    nuclei.drop(columns=["Parent_Cells"]).to_csv(directory / "Nuclei.csv", index=False)
    pd.DataFrame(
        {
            "ImageNumber": cells["ImageNumber"],
            "ObjectNumber": cells["ObjectNumber"],
            "Parent_Cells": cells["ObjectNumber"],
            "Parent_Nuclei": n + 1 - cells["ObjectNumber"],
            "AreaShape_Area": 100.0,
        }
    ).to_csv(directory / "Cytoplasm.csv", index=False)

    adata = mt.io.read_profiles(directory)

    expected = nuclei.set_index(["ImageNumber", "ObjectNumber"])["AreaShape_Area"]
    keys = list(zip(cells["ImageNumber"], n + 1 - cells["ObjectNumber"], strict=True))
    np.testing.assert_allclose(adata[:, "Nuclei_AreaShape_Area"].X.ravel(), expected.loc[keys].to_numpy())


def test_two_children_pointing_at_the_same_object_are_refused(tmp_path, make_cellprofiler_dir):
    """With two shared children carrying Parent_Nuclei, picking one would be a guess (#61)."""
    directory = make_cellprofiler_dir(tmp_path / "ambiguous", link_on="child")
    cells = pd.read_csv(directory / "Cells.csv")
    nuclei = pd.read_csv(directory / "Nuclei.csv")
    nuclei.drop(columns=["Parent_Cells"]).to_csv(directory / "Nuclei.csv", index=False)
    for name in ("Cytoplasm", "Extra"):
        pd.DataFrame(
            {
                "ImageNumber": cells["ImageNumber"],
                "ObjectNumber": cells["ObjectNumber"],
                "Parent_Cells": cells["ObjectNumber"],
                "Parent_Nuclei": cells["ObjectNumber"],
                "AreaShape_Area": 100.0,
            }
        ).to_csv(directory / f"{name}.csv", index=False)

    with pytest.raises(ValueError, match=r"Cytoplasm_Parent_Nuclei.*Extra_Parent_Nuclei.*all point to"):
        mt.io.read_profiles(directory, objects=None)


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


def test_a_wide_export_reads_without_a_fragmentation_warning(tmp_path, make_cellprofiler_dir):
    """Joining a wide object table once trips pandas' fragmentation warning; the readers add columns via concat so it does not (reading a real CellProfiler export was noisy)."""
    directory = make_cellprofiler_dir(tmp_path / "wide", extra_object_columns=150)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        adata = mt.io.read_profiles(directory)
    assert "Nuclei_AreaShape_Area" in adata.var_names


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


# --- M2 object identity ---------------------------------------------------


def test_a_cp_object_is_identifiable_without_parsing_obs_names(cellprofiler_dir):
    """A primary object is identified by its own columns; the index carries no meaning."""
    adata = mt.io.read_profiles(cellprofiler_dir)
    assert {"Metadata_ImageID", "Metadata_ObjectType", "Metadata_ObjectNumber"} <= set(adata.obs.columns)
    assert set(adata.obs["Metadata_ObjectType"]) == {"Cells"}
    # The (image, object) pair is unique; scrambling the index does not change what the row is.
    assert not adata.obs[["Metadata_ImageID", "Metadata_ObjectNumber"]].duplicated().any()
    scrambled = adata.copy()
    scrambled.obs_names = [f"x{i}" for i in range(scrambled.n_obs)]
    assert validate(scrambled).ok, validate(scrambled).errors


def test_related_object_numbers_survive_a_cp_read(tmp_path, make_cellprofiler_dir):
    """Nuclei/cytoplasm instance numbers are kept as Metadata_<obj>ObjectNumber, not dropped (spec 5.4)."""
    directory = make_cellprofiler_dir(tmp_path / "related", link_on="child")
    adata = mt.io.read_profiles(directory)
    assert "Metadata_NucleiObjectNumber" in adata.obs.columns
    assert adata.obs["Metadata_NucleiObjectNumber"].notna().all()
    # The builder links each cell to the nucleus carrying the same object number, so the preserved
    # value equals the parent link rather than being lost or constant.
    assert (adata.obs["Metadata_NucleiObjectNumber"].to_numpy() == adata.obs["Metadata_ObjectNumber"].to_numpy()).all()


def test_image_table_joins_one_to_one_on_image_id(cellprofiler_dir):
    """Every object resolves to exactly one image-table row, keyed by the opaque Metadata_ImageID."""
    adata = mt.io.read_profiles(cellprofiler_dir)
    table = adata.uns["mantispy"]["image_table"]
    assert table.index.name == "Metadata_ImageID"
    assert set(adata.obs["Metadata_ImageID"]) <= set(table.index)
    assert validate(adata).ok, validate(adata).errors


# --- real tutorial exports --------------------------------------------------


def test_neighbour_object_numbers_stay_out_of_x():
    """MeasureObjectNeighbors' FirstClosestObjectNumber names another object; it was read as a measurement."""
    adata = mt.io.read_profiles(EXPORTS / "advanced")
    assert "Nuclei_Neighbors_FirstClosestDistance_Expanded" in adata.var_names
    assert not [name for name in adata.var_names if "ClosestObjectNumber" in name]


def test_per_parent_means_of_locations_and_ids_stay_out_of_x():
    """RelateObjects writes Mean_<Child>_<column> for every child column, centroids and object numbers included."""
    adata = mt.io.read_profiles(EXPORTS / "advanced")
    assert adata.var.loc["Nuclei_Mean_Nucleoli_AreaShape_Area", "object"] == "Nuclei"
    leaked = [
        name
        for name in adata.var_names
        if "_Mean_" in name
        and any(token in name for token in ("_Location_", "_Center_", "BoundingBoxM", "_Number_Object", "ObjectNumber"))
    ]
    assert not leaked


def test_calculate_math_features_keep_their_object_when_every_object_is_read():
    """objects=None filed Cytoplasm_Math_IntensityRatio as an object-less embedding."""
    adata = mt.io.read_profiles(EXPORTS / "translocation", objects=None)
    row = adata.var.loc["Cytoplasm_Math_IntensityRatio"]
    assert (row["object"], row["feature_group"], row["feature_kind"]) == ("Cytoplasm", "Math", "measurement")


def test_a_rowless_well_is_kept_and_round_trips(tmp_path, caplog):
    """PERISCOPE names its well Well1; the read raised on it, and a plate position must not be invented."""
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        adata = mt.io.read_profiles(EXPORTS / "periscope_site21")
    assert "Well1" in caplog.text
    assert set(adata.obs["Metadata_Well"]) == {"Well1"}
    assert not {"Metadata_Row", "Metadata_Col"} & set(adata.obs.columns)
    report = validate(adata)
    assert report.ok and report.warnings, report.errors
    mt.io.write(adata, tmp_path / "periscope.h5ad")
    assert set(mt.io.read(tmp_path / "periscope.h5ad").obs["Metadata_Well"]) == {"Well1"}


def test_a_platemap_joins_a_rowless_well():
    platemap = pd.DataFrame({"Metadata_Well": ["Well1"], "Metadata_Perturbation": ["library"]})
    adata = mt.io.read_profiles(EXPORTS / "periscope_site21", platemap=platemap)
    assert set(adata.obs["Metadata_Perturbation"]) == {"library"}


def test_an_imported_image_number_does_not_replace_cellprofilers():
    """The BBBC022 metadata file has an ImageNumber column, read as Metadata_ImageNumber beside CellProfiler's own."""
    adata = mt.io.read_profiles(EXPORTS / "advanced")
    image = pd.read_csv(EXPORTS / "advanced" / "MyExpt_Image.csv")
    assert set(adata.obs["Metadata_ImageNumber"]) == set(image["ImageNumber"])
    assert set(adata.obs["Metadata_ImportedImageNumber"]) == set(image["Metadata_ImageNumber"])


def test_a_tab_delimited_export_reads_like_the_csv_one():
    """A run set to tab delimiters writes Run1_Image.txt, which was not recognised as an export."""
    tab = mt.io.read_profiles(EXPORTS / "beginner_nulltab")
    csv = mt.io.read_profiles(EXPORTS / "beginner")
    assert list(tab.var_names) == list(csv.var_names)
    assert validate(tab).ok, validate(tab).errors


def test_a_repeated_header_with_different_values_is_refused(tmp_path):
    """Adding image metadata to the object tables writes Metadata_Well twice; the copies must agree to keep one."""
    directory = tmp_path / "nulltab"
    shutil.copytree(EXPORTS / "beginner_nulltab", directory)
    path = directory / "Run1_Cells.txt"
    lines = path.read_text().splitlines()
    second = [i for i, name in enumerate(lines[0].split("\t")) if name == "Metadata_Well"][1]
    row = lines[1].split("\t")
    row[second] = "Z99"
    lines[1] = "\t".join(row)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="'Metadata_Well' holding different values"):
        mt.io.read_profiles(directory)


def test_an_image_only_export_gives_one_row_per_image(tmp_path):
    """A QC pipeline identifies no objects; the read raised for want of a Cells table."""
    adata = mt.io.read_profiles(EXPORTS / "qc")
    image = pd.read_csv(EXPORTS / "qc" / "MyExpt_Image.csv")
    assert adata.n_obs == len(image)
    assert adata.uns["mantispy"]["resolution"] == "aggregate"
    assert list(adata.uns["mantispy"]["grouped_by"]) == ["Metadata_ImageID"]
    assert list(adata.obs_names) == list(adata.obs["Metadata_ImageID"].astype(str))
    assert set(adata.var["feature_group"]) == {"ImageQuality"}
    assert set(adata.uns["mantispy"]["image_table"].index) == set(adata.obs["Metadata_ImageID"])
    mt.io.write(adata, tmp_path / "qc.h5ad")
    assert mt.io.read(tmp_path / "qc.h5ad").n_obs == len(image)


def test_an_identity_column_left_empty_is_left_out(tmp_path):
    """The Metadata module writes Metadata_Well even when it extracted nothing, and no image id can be built from that."""
    directory = tmp_path / "qc"
    shutil.copytree(EXPORTS / "qc", directory)
    image = pd.read_csv(directory / "MyExpt_Image.csv")
    image["Metadata_Well"] = np.nan
    image.to_csv(directory / "MyExpt_Image.csv", index=False)
    adata = mt.io.read_profiles(directory)
    assert "Metadata_Well" not in adata.obs
    assert validate(adata).ok, validate(adata).errors


def test_an_export_of_image_metadata_only_says_there_is_nothing_to_profile():
    with pytest.raises(ValueError, match="only image metadata"):
        mt.io.read_profiles(EXPORTS / "im_2d_ex1")
