"""Reading CellProfiler exports: per-object tables, and the one profile read_cellprofiler joins from them."""

import logging
import re
import shutil
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import validate

#: Slices of real exports of the CellProfiler tutorials and of PERISCOPE (cpg0021, plate CP257A, Well1 site 21):
#: the first images and a few linked objects of each table, with absolute paths scrubbed.
#: The SQLite databases hold the same objects as the CSV slices of the same tutorial.
EXPORTS = Path(__file__).parent / "data" / "cellprofiler"
SQLITE = EXPORTS / "sqlite"


def _assert_same_tables(left: dict, right: dict) -> None:
    assert sorted(left) == sorted(right)
    for name in left:
        a, b = left[name], right[name]
        assert list(a.var_names) == list(b.var_names), name
        np.testing.assert_allclose(a.X, b.X, rtol=1e-12, err_msg=name)
        for column in ("Metadata_ImageID", "Metadata_ObjectNumber", "Metadata_Plate", "Metadata_Well", "Metadata_Site"):
            if column in a.obs:
                assert a.obs[column].dtype == b.obs[column].dtype, (name, column)
                assert list(a.obs[column]) == list(b.obs[column]), (name, column)
        assert a.uns["mantispy"]["links"] == b.uns["mantispy"]["links"], name


# --- the profile, on the synthetic export ------------------------------------


@pytest.mark.parametrize("link_on", ["child", "primary"])
def test_parent_link_is_found_on_either_table(tmp_path, make_cellprofiler_dir, link_on):
    """CellProfiler writes Parent_Nuclei on the primary table when cells came from nuclei, and Parent_Cells on the child table otherwise."""
    directory = make_cellprofiler_dir(tmp_path / link_on, link_on=link_on)
    adata = mt.io.read_cellprofiler(directory)
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

    adata = mt.io.read_cellprofiler(directory)

    expected = nuclei.set_index(["ImageNumber", "ObjectNumber"])["AreaShape_Area"]
    keys = list(zip(cells["ImageNumber"], n + 1 - cells["ObjectNumber"], strict=True))
    np.testing.assert_allclose(adata[:, "Nuclei_AreaShape_Area"].X.ravel(), expected.loc[keys].to_numpy())


def test_an_object_without_a_link_is_left_out_not_paired_by_number(tmp_path, make_cellprofiler_dir, caplog):
    """Merging on ObjectNumber alone would pair unrelated objects with no error."""
    directory = make_cellprofiler_dir(tmp_path / "nolink", link_on="child")
    pd.read_csv(directory / "Nuclei.csv").drop(columns=["Parent_Cells"]).to_csv(directory / "Nuclei.csv", index=False)
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        adata = mt.io.read_cellprofiler(directory)
    assert not [name for name in adata.var_names if name.startswith("Nuclei_")]
    assert "['Nuclei'], which are not linked to Cells" in caplog.text


def test_a_compartment_that_is_not_one_to_one_raises(tmp_path, make_cellprofiler_dir):
    directory = make_cellprofiler_dir(tmp_path / "dup", one_to_one=False)
    with pytest.raises(ValueError, match="more than one Nuclei"):
        mt.io.read_cellprofiler(directory)
    adata = mt.io.read_cellprofiler(directory, relations={"Nuclei": {"Cells": "aggregate"}}, aggregate="count")
    assert adata[:, "Nuclei_count"].X.sum() == 25


def test_a_link_nothing_explains_leaves_its_table_out_and_suggests_its_kind(tmp_path, make_cellprofiler_dir, caplog):
    """Without the pipeline, only the compartments with matching counts and RelateObjects' columns are read by name."""
    directory = make_cellprofiler_dir(tmp_path / "spots")
    image = pd.read_csv(directory / "Image.csv").assign(Count_Spots=2)
    image.to_csv(directory / "Image.csv", index=False)
    pd.DataFrame(
        {"ImageNumber": [1, 1], "ObjectNumber": [1, 2], "Parent_Cells": [1, 1], "Intensity_MeanIntensity_DNA": 0.5}
    ).to_csv(directory / "Spots.csv", index=False)
    suggestion = "relations={'Spots': {'Cells': 'aggregate'}}"
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        tables = mt.io.read_cellprofiler_csv(directory)
        adata = mt.io.read_cellprofiler(directory, aggregate="count")
    assert "Spots" not in tables
    assert "Spots_count" not in adata.var_names
    assert caplog.text.count("left out ['Spots']: cannot tell what the link(s) Spots -> Cells mean") == 2
    assert suggestion in caplog.text
    # The root's own links decide what a row is, so they are never guessed.
    with pytest.raises(ValueError, match=re.escape(suggestion)):
        mt.io.read_cellprofiler(directory, root="Spots")
    adata = mt.io.read_cellprofiler(directory, relations={"Spots": {"Cells": "aggregate"}}, aggregate="count")
    assert adata[:, "Spots_count"].X.ravel().tolist()[:2] == [2.0, 0.0]


def test_a_wide_export_reads_without_a_fragmentation_warning(tmp_path, make_cellprofiler_dir):
    """Joining a wide object table once trips pandas' fragmentation warning; the readers add columns via concat so it does not (reading a real CellProfiler export was noisy)."""
    directory = make_cellprofiler_dir(tmp_path / "wide", extra_object_columns=150)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        adata = mt.io.read_cellprofiler(directory)
    assert "Nuclei_AreaShape_Area" in adata.var_names


def test_platemap_is_joined_by_well(cellprofiler_dir, platemap_path):
    adata = mt.io.read_cellprofiler(cellprofiler_dir, platemap=platemap_path)
    mt.pp.annotate_controls(adata, negcon=("DMSO",))
    assert set(adata.obs["Metadata_Perturbation"]) == {"DMSO", "compound_a"}
    assert adata.obs["Metadata_Control"].sum() == 12
    assert adata.uns["mantispy"]["params"]["read_cellprofiler"]["root"] == "Cells"


def test_a_duplicated_platemap_row_is_refused(cellprofiler_dir, platemap_path):
    """Merging it would multiply the cells of that well and leave the object with more rows than the images produced."""
    frame = pd.read_csv(platemap_path)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="more than one row"):
        mt.io.read_cellprofiler(cellprofiler_dir, platemap=doubled)


def test_metadata_on_both_the_object_and_image_tables_keeps_the_image_value(tmp_path, make_cellprofiler_dir):
    """CellProfiler can copy image metadata into the object tables, and the merge then suffixed the pair Metadata_Plate_x/_y, so the schema's plate column vanished silently."""
    directory = make_cellprofiler_dir(tmp_path / "collide")
    cells = pd.read_csv(directory / "Cells.csv")
    cells["Metadata_Plate"] = "from_the_object_table"
    cells.to_csv(directory / "Cells.csv", index=False)

    adata = mt.io.read_cellprofiler(directory)

    assert not [column for column in adata.obs.columns if column.endswith(("_x", "_y"))]
    assert set(adata.obs["Metadata_Plate"]) == {"P1"}
    assert validate(adata).ok, validate(adata).errors


def test_a_missing_root_or_a_path_that_is_no_export_says_why(cellprofiler_dir, tmp_path):
    with pytest.raises(ValueError, match=r"no 'Nope' table; pass root= one of \['Cells', 'Nuclei'\]"):
        mt.io.read_cellprofiler(cellprofiler_dir, root="Nope")
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("")
    with pytest.raises(FileNotFoundError, match=r"not a CellProfiler export.*notes.txt"):
        mt.io.read_cellprofiler(tmp_path / "empty")


def test_a_missing_root_names_the_objects_that_could_be_one_on_either_route():
    """The bonus tutorials name their cells CellProfilerCells or RunCellpose; the two routes raised different errors."""
    objects = mt.io.read_cellprofiler_csv(EXPORTS / "translocation")
    for read in (
        lambda: mt.io.read_cellprofiler(EXPORTS / "translocation", root="CellposeCells"),
        lambda: mt.tl.aggregate_objects(objects, root="CellposeCells"),
    ):
        with pytest.raises(ValueError, match=r"pass root= one of \['Cells', 'Cytoplasm', 'Nuclei'\]"):
            read()
    unlinked = {name: table for name, table in objects.items() if name != "Cytoplasm"}
    for table in unlinked.values():
        table.uns["mantispy"]["links"] = {}
    unlinked["Nuclei"] = unlinked["Nuclei"][:, []]
    with pytest.raises(ValueError, match=r"pass root= one of \['Cells', 'Nuclei'\]"):
        mt.tl.aggregate_objects(unlinked, root="CellposeCells")


def test_a_3d_export_is_refused(tmp_path, make_cellprofiler_dir):
    directory = make_cellprofiler_dir(tmp_path / "3d")
    cells = pd.read_csv(directory / "Cells.csv").assign(Location_Center_Z=3.0)
    cells.to_csv(directory / "Cells.csv", index=False)
    with pytest.raises(ValueError, match="3D"):
        mt.io.read_cellprofiler(directory)


def test_a_cp_object_is_identifiable_without_parsing_obs_names(cellprofiler_dir):
    """A primary object is identified by its own columns; the index carries no meaning."""
    adata = mt.io.read_cellprofiler(cellprofiler_dir)
    assert {"Metadata_ImageID", "Metadata_ObjectType", "Metadata_ObjectNumber"} <= set(adata.obs.columns)
    assert set(adata.obs["Metadata_ObjectType"]) == {"Cells"}
    assert not adata.obs[["Metadata_ImageID", "Metadata_ObjectNumber"]].duplicated().any()
    scrambled = adata.copy()
    scrambled.obs_names = [f"x{i}" for i in range(scrambled.n_obs)]
    assert validate(scrambled).ok, validate(scrambled).errors


def test_related_object_numbers_survive_a_cp_read(tmp_path, make_cellprofiler_dir):
    """Nuclei/cytoplasm instance numbers are kept as Metadata_<obj>ObjectNumber, not dropped (spec 5.4)."""
    directory = make_cellprofiler_dir(tmp_path / "related", link_on="child")
    adata = mt.io.read_cellprofiler(directory)
    assert (adata.obs["Metadata_NucleiObjectNumber"].to_numpy() == adata.obs["Metadata_ObjectNumber"].to_numpy()).all()


def test_image_table_holds_every_image_keyed_by_image_id(cellprofiler_dir):
    adata = mt.io.read_cellprofiler(cellprofiler_dir)
    table = adata.uns["mantispy"]["image_table"]
    assert table.index.name == "Metadata_ImageID"
    assert set(adata.obs["Metadata_ImageID"]) == set(table.index)
    assert {"Count_Cells", "Metadata_Plate", "ImageQuality_FocusScore_DNA"} <= set(table.columns)
    assert validate(adata).ok, validate(adata).errors


def test_link_columns_to_objects_without_a_table_are_left_out(tmp_path, make_cellprofiler_dir, caplog):
    """JUMP links Cells to CellsIncludingEdges, which it counts but never exports."""
    directory = make_cellprofiler_dir(tmp_path / "edges")
    pd.read_csv(directory / "Cells.csv").assign(Parent_CellsIncludingEdges=1).to_csv(
        directory / "Cells.csv", index=False
    )
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        tables = mt.io.read_cellprofiler_csv(directory)
    assert "CellsIncludingEdges" in caplog.text
    assert "Metadata_CellsIncludingEdgesObjectNumber" not in tables["Cells"].obs


def test_a_compartment_without_a_root_is_kept_in_its_table_and_left_out_of_the_profile(
    tmp_path, make_cellprofiler_dir, caplog
):
    """PERISCOPE has a few cytoplasm objects with no cell (parent 0) on some sites."""
    directory = make_cellprofiler_dir(tmp_path / "orphan", link_on="child")
    nuclei = pd.read_csv(directory / "Nuclei.csv")
    nuclei.loc[0, "Parent_Cells"] = 0
    nuclei.to_csv(directory / "Nuclei.csv", index=False)

    tables = mt.io.read_cellprofiler_csv(directory)
    assert tables["Nuclei"].n_obs == len(nuclei)
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        adata = mt.io.read_cellprofiler(directory)
    assert "left out 1 of 24 Nuclei object(s), which link to no Cells; 1 Cells object(s) have no Nuclei" in caplog.text
    assert np.isnan(adata[:, "Nuclei_AreaShape_Area"].X).sum() == 1
    with pytest.raises(ValueError, match="strict=False"):
        mt.io.read_cellprofiler(directory, strict=True)


# --- per-object tables of the tutorial exports ------------------------------


def test_every_object_table_is_read_with_its_link_kinds():
    """Beginner: Nucleoli belong to nuclei many to one; NucleoliChildObjects, RelateObjects' saved copy, only carries the link."""
    tables = mt.io.read_cellprofiler_csv(EXPORTS / "beginner")
    assert {name: t.uns["mantispy"]["links"] for name, t in tables.items()} == {
        "Cells": {"Nuclei": "compartment"},
        "Cytoplasm": {"Cells": "compartment", "Nuclei": "compartment"},
        "Nuclei": {},
        "Nucleoli": {"Nuclei": "aggregate"},
    }
    nucleoli = pd.read_csv(EXPORTS / "beginner" / "MyExpt_Nucleoli.csv")
    assert tables["Nucleoli"].n_obs == len(nucleoli)
    assert tables["Nucleoli"].obs["Metadata_NucleiObjectNumber"].tolist() == nucleoli["Parent_Nuclei"].tolist()
    for table in tables.values():
        assert validate(table).ok, validate(table).errors


def test_a_table_of_locations_only_is_valid_and_round_trips(tmp_path):
    """Translocation measures nothing on its cells, which keep their locations and links."""
    cells = mt.io.read_cellprofiler_csv(EXPORTS / "translocation")["Cells"]
    assert cells.n_vars == 0
    assert {"Metadata_Center_X", "Cells_Location_Center_X"} <= set(cells.obs.columns)
    mt.io.write(cells, tmp_path / "cells.h5ad")
    assert mt.io.read(tmp_path / "cells.h5ad").uns["mantispy"]["links"] == {"Nuclei": "compartment"}


def test_a_tab_delimited_export_reads_like_the_csv_one():
    """A run set to tab delimiters writes Run1_Image.txt, which was not recognised as an export."""
    _assert_same_tables(
        mt.io.read_cellprofiler_csv(EXPORTS / "beginner_nulltab"), mt.io.read_cellprofiler_csv(EXPORTS / "beginner")
    )


def test_a_combined_table_splits_into_the_per_object_tables(caplog):
    """ExportToSpreadsheet can write every object into one file with two header rows, and no pipeline beside it."""
    beginner = mt.io.read_cellprofiler_csv(EXPORTS / "beginner")
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        _assert_same_tables(mt.io.read_cellprofiler_csv(EXPORTS / "beginner_combined"), beginner)
    assert "relations={'NucleoliChildObjects': {'Nucleoli': 'carrier'}}" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        combined = mt.io.read_cellprofiler_csv(
            EXPORTS / "beginner_combined", relations={"NucleoliChildObjects": {"Nucleoli": "carrier"}}
        )
    assert not caplog.text
    _assert_same_tables(combined, beginner)


@pytest.mark.parametrize("database", ["beginner_db_view.db", "beginner_db_single.db"])
def test_an_sqlite_export_reads_like_the_csv_one(database):
    """One table per object with a Per_Object view, or one Per_Object table that lines the objects up."""
    _assert_same_tables(
        mt.io.read_cellprofiler_sqlite(SQLITE / database), mt.io.read_cellprofiler_csv(EXPORTS / "beginner")
    )


def test_a_single_object_table_with_calculate_math_reads_like_the_csv_one():
    """Translocation ships ExportToDatabase with one Per_Object table."""
    _assert_same_tables(
        mt.io.read_cellprofiler_sqlite(SQLITE / "translocation_db.db"),
        mt.io.read_cellprofiler_csv(EXPORTS / "translocation"),
    )


def test_every_export_of_a_run_gives_the_same_obs_with_file_locations_in_the_image_table():
    """The tab export copied file names into obs as Cells_FileName_*, LoadData's as Metadata_Image_FileName_*."""
    beginner = [
        EXPORTS / "beginner",
        EXPORTS / "beginner_nulltab",
        SQLITE / "beginner_db_view.db",
        SQLITE / "beginner_db_single.db",
    ]
    profiles = [mt.io.read_cellprofiler(path) for path in beginner]
    profiles.append(
        mt.io.read_cellprofiler(
            EXPORTS / "beginner_combined", relations={"NucleoliChildObjects": {"Nucleoli": "carrier"}}
        )
    )
    expected = list(profiles[0].obs.columns)
    for adata in profiles:
        assert list(adata.obs.columns) == expected
        assert "FileName_OrigDNA" in adata.uns["mantispy"]["image_table"]
    assert not [c for c in expected if "FileName" in c or "PathName" in c]
    assert not {"Metadata_FileLocation", "Metadata_Frame", "Metadata_Series", "Metadata_ChannelNumber"} & set(expected)

    advanced = mt.io.read_cellprofiler(EXPORTS / "advanced")
    assert list(advanced.obs.columns) == list(mt.io.read_cellprofiler(SQLITE / "advanced_db.db").obs.columns)
    # LoadData read an earlier run's Image_Metadata_BROAD_ID back as Metadata_Image_Metadata_BROAD_ID.
    assert {"Metadata_BROAD_ID", "Metadata_Plate", "Metadata_Well", "Metadata_Site"} <= set(advanced.obs.columns)
    assert not [c for c in advanced.obs.columns if "FileName" in c or "PathName" in c]


def test_the_profile_leaves_cellprofilers_bookkeeping_in_the_per_object_tables():
    """Translocation measures nothing on its cells, whose links, counts and Z centroid filled the profile's obs."""
    bookkeeping = ["Cells_Children_Cytoplasm_Count", "Cells_Number_Object_Number", "Cells_Location_Center_Z"]
    cells = mt.io.read_cellprofiler_csv(EXPORTS / "translocation")["Cells"]
    assert set(bookkeeping) <= set(cells.obs.columns)
    adata = mt.io.read_cellprofiler(EXPORTS / "translocation")
    assert not set(bookkeeping) & set(adata.obs.columns)
    assert not [c for c in adata.obs.columns if c.startswith("Cells_Location_")]
    identity = ["Metadata_ImageID", "Metadata_ObjectNumber", "Metadata_ImageNumber", "Metadata_Plate", "Metadata_Well"]
    assert set(identity + ["Metadata_NucleiObjectNumber", "Metadata_Center_X", "Metadata_Dose"]) <= set(adata.obs)
    assert validate(adata).ok, validate(adata).errors


def test_column_names_exportodatabase_shortened_are_restored():
    """BBBC022 as shipped writes no table prefix and shortens names such as ..._FirstClosestObjectNumbr_Expndd."""
    with sqlite3.connect(SQLITE / "advanced_db.db") as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(Per_Cells)")]
    assert any(len(column) == 64 for column in columns)
    database = mt.io.read_cellprofiler_sqlite(SQLITE / "advanced_db.db")
    spreadsheet = mt.io.read_cellprofiler_csv(EXPORTS / "advanced")
    for name, table in spreadsheet.items():
        assert list(database[name].var_names) == list(table.var_names), name
        np.testing.assert_allclose(database[name].X, table.X, rtol=1e-12)
    assert set(database["Nuclei"].obs["Metadata_ImportedImageNumber"]) == set(
        spreadsheet["Nuclei"].obs["Metadata_ImportedImageNumber"]
    )


@pytest.mark.parametrize("source", [EXPORTS / "qc", SQLITE / "qc_db.db"], ids=["csv", "sqlite"])
def test_an_image_only_export_gives_one_row_per_image(tmp_path, source):
    """A QC pipeline identifies no objects; the read raised for want of a Cells table."""
    adata = mt.io.read_cellprofiler(source)
    assert adata.n_obs == 3
    assert adata.uns["mantispy"]["resolution"] == "aggregate"
    assert list(adata.uns["mantispy"]["grouped_by"]) == ["Metadata_ImageID"]
    assert list(adata.obs_names) == list(adata.obs["Metadata_ImageID"].astype(str))
    assert set(adata.var["feature_group"]) == {"ImageQuality"}
    mt.io.write(adata, tmp_path / "qc.h5ad")
    assert mt.io.read(tmp_path / "qc.h5ad").n_obs == 3


def test_an_export_of_image_metadata_only_reads_as_a_table_without_features(tmp_path, caplog):
    """Input-module pipelines measure nothing; their export is a list of images and their metadata."""
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        adata = mt.io.read_cellprofiler(EXPORTS / "im_2d_ex1")
    assert adata.n_vars == 0
    assert validate(adata).ok, validate(adata).errors
    assert "holds image metadata and no measurement" in caplog.text


def test_a_profile_of_locations_only_warns_and_names_the_per_object_reader(tmp_path, caplog):
    """The bonus tutorials segment with ilastik or Cellpose and measure nothing; the read returned 848 x 0 silently."""
    directory = tmp_path / "locations"
    shutil.copytree(EXPORTS / "translocation", directory)
    for name in ("Cytoplasm", "Nuclei"):
        table = pd.read_csv(directory / f"MyExpt_{name}.csv")
        kept = [
            c for c in table.columns if c in ("ImageNumber", "ObjectNumber") or c.startswith(("Location_", "Parent_"))
        ]
        table[kept].to_csv(directory / f"MyExpt_{name}.csv", index=False)
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        adata = mt.io.read_cellprofiler(directory)
    assert adata.n_vars == 0
    assert "the Cells profile holds no feature" in caplog.text
    assert "mt.io.read_cellprofiler_csv returns every object" in caplog.text


def test_an_identity_column_left_empty_is_left_out(tmp_path):
    """The Metadata module writes Metadata_Well even when it extracted nothing, and no image id can be built from that."""
    directory = tmp_path / "qc"
    shutil.copytree(EXPORTS / "qc", directory)
    image = pd.read_csv(directory / "MyExpt_Image.csv").assign(Metadata_Well=np.nan)
    image.to_csv(directory / "MyExpt_Image.csv", index=False)
    assert "Metadata_Well" not in mt.io.read_cellprofiler(directory).obs


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
        mt.io.read_cellprofiler_csv(directory)


def test_an_imported_image_number_does_not_replace_cellprofilers(caplog):
    """The BBBC022 metadata file has an ImageNumber column, read as Metadata_ImageNumber beside CellProfiler's own."""
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        adata = mt.io.read_cellprofiler(EXPORTS / "advanced")
    assert caplog.text.count("Metadata_ImportedImageNumber") == 1
    image = pd.read_csv(EXPORTS / "advanced" / "MyExpt_Image.csv")
    assert set(adata.obs["Metadata_ImageNumber"]) == set(image["ImageNumber"])
    assert set(adata.obs["Metadata_ImportedImageNumber"]) == set(image["Metadata_ImageNumber"])


def test_calculate_math_features_keep_their_object():
    row = mt.io.read_cellprofiler_csv(EXPORTS / "translocation")["Cytoplasm"].var.loc["Cytoplasm_Math_IntensityRatio"]
    assert (row["object"], row["feature_group"], row["feature_kind"]) == ("Cytoplasm", "Math", "measurement")


def test_periscope_keeps_measured_foci_as_tables_and_leaves_out_the_copies_that_carry_links():
    """Foci and BarcodeFoci carry barcode calls and intensities; the Resize copies and the pre-mask foci carry links only."""
    tables = mt.io.read_cellprofiler_csv(EXPORTS / "periscope_site21")
    assert sorted(tables) == ["BarcodeFoci", "Cells", "Cytoplasm", "Foci", "Foci_NonCellEdge", "Nuclei"]
    assert tables["BarcodeFoci"].uns["mantispy"]["links"] == {"Cells": "aggregate", "Foci": "subset"}
    assert tables["Foci"].uns["mantispy"]["links"] == {"Foci_PreMask": "carrier"}
    foci = tables["Foci"]
    assert foci.n_vars > 0
    assert "Foci_Barcode_MatchedTo_Score" in foci.obs
    assert not [name for name in foci.var_names if "Barcode_" in name]


def test_a_rowless_well_is_kept_and_round_trips(tmp_path, caplog):
    """PERISCOPE names its well Well1; the read raised on it, and a plate position must not be invented."""
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        adata = mt.io.read_cellprofiler(EXPORTS / "periscope_site21")
    assert "Well1" not in caplog.text
    assert set(adata.obs["Metadata_Well"]) == {"Well1"}
    assert not {"Metadata_Row", "Metadata_Col"} & set(adata.obs.columns)
    report = validate(adata)
    assert report.ok, report.errors
    mt.io.write(adata, tmp_path / "periscope.h5ad")
    assert set(mt.io.read(tmp_path / "periscope.h5ad").obs["Metadata_Well"]) == {"Well1"}


def test_a_platemap_joins_a_rowless_well():
    platemap = pd.DataFrame({"Metadata_Well": ["Well1"], "Metadata_Perturbation": ["library"]})
    adata = mt.io.read_cellprofiler(EXPORTS / "periscope_site21", platemap=platemap)
    assert set(adata.obs["Metadata_Perturbation"]) == {"library"}


# --- rolling children into their root ----------------------------------------


@pytest.fixture(scope="module")
def beginner():
    return mt.io.read_cellprofiler_csv(EXPORTS / "beginner")


def test_rolled_statistics_match_pandas(beginner):
    nucleoli = pd.read_csv(EXPORTS / "beginner" / "MyExpt_Nucleoli.csv")
    cells = pd.read_csv(EXPORTS / "beginner" / "MyExpt_Cells.csv")
    adata = mt.tl.aggregate_objects(beginner, root="Cells")

    # Cells reach nucleoli through their nucleus.
    keys = pd.MultiIndex.from_arrays([cells["ImageNumber"], cells["Parent_Nuclei"]])
    grouped = nucleoli.groupby(["ImageNumber", "Parent_Nuclei"])["AreaShape_Area"]
    expected = {
        "count": grouped.size().reindex(keys, fill_value=0),
        "mean": grouped.mean().reindex(keys),
        "median": grouped.median().reindex(keys),
        "std": grouped.std(ddof=1).reindex(keys),
    }
    count = adata[:, "Nucleoli_count"].X.ravel()
    np.testing.assert_array_equal(count, expected["count"].to_numpy())
    for stat in ("mean", "median", "std"):
        np.testing.assert_allclose(adata[:, f"Nucleoli_AreaShape_Area_{stat}"].X.ravel(), expected[stat].to_numpy())
    row = adata.var.loc["Nucleoli_AreaShape_Area_mean"]
    assert (row["object"], row["feature"], row["aggregation"], row["parent"]) == ("Nucleoli", "Area", "mean", "Cells")
    assert adata.var.loc["Nucleoli_count", "feature_kind"] == "derived"
    assert validate(adata).ok, validate(adata).errors


def test_rolled_means_match_cellprofilers_own(beginner):
    """RelateObjects writes Nuclei_Mean_Nucleoli_<column>; the roll-up replaces it with the same numbers."""
    adata = mt.tl.aggregate_objects(beginner, root="Nuclei", func="mean")
    nuclei = pd.read_csv(EXPORTS / "beginner" / "MyExpt_Nuclei.csv").set_index(["ImageNumber", "ObjectNumber"])
    rows = pd.MultiIndex.from_arrays([adata.obs["Metadata_ImageNumber"], adata.obs["Metadata_ObjectNumber"]])
    for column in ("AreaShape_Area", "Intensity_MeanIntensity_OrigDNA"):
        mine = adata[:, f"Nucleoli_{column}_mean"].X.ravel()
        theirs = nuclei.loc[rows, f"Mean_Nucleoli_{column}"].to_numpy()
        both = ~np.isnan(theirs)
        np.testing.assert_allclose(mine[both], theirs[both], rtol=1e-10)
    assert not [name for name in adata.var_names if name.startswith("Nuclei_Mean_Nucleoli_")]


def test_a_root_without_children_has_count_zero_and_nan_statistics(beginner):
    tables = dict(beginner)
    nucleoli = tables["Nucleoli"]
    tables["Nucleoli"] = nucleoli[nucleoli.obs["Metadata_NucleiObjectNumber"] != 1].copy()
    adata = mt.tl.aggregate_objects(tables, root="Nuclei", func=("count", "mean", "std"))
    first = (adata.obs["Metadata_ObjectNumber"] == 1).to_numpy()
    assert (adata[first, "Nucleoli_count"].X == 0).all()
    assert np.isnan(adata[first, "Nucleoli_AreaShape_Area_mean"].X).all()
    single = adata[:, "Nucleoli_count"].X.ravel() == 1
    assert np.isnan(adata[single, "Nucleoli_AreaShape_Area_std"].X).all()


def test_a_missing_value_of_a_child_is_skipped(beginner):
    tables = dict(beginner)
    nucleoli = tables["Nucleoli"].copy()
    column = nucleoli.var_names.get_loc("Nucleoli_AreaShape_Area")
    nucleoli.X[0, column] = np.nan
    tables["Nucleoli"] = nucleoli
    adata = mt.tl.aggregate_objects(tables, root="Nuclei", func=("count", "mean"))

    obs = nucleoli.obs
    parent = (obs["Metadata_ImageID"].iloc[0], obs["Metadata_NucleiObjectNumber"].iloc[0])
    siblings = ((obs["Metadata_ImageID"] == parent[0]) & (obs["Metadata_NucleiObjectNumber"] == parent[1])).to_numpy()
    row = ((adata.obs["Metadata_ImageID"] == parent[0]) & (adata.obs["Metadata_ObjectNumber"] == parent[1])).to_numpy()
    assert siblings.sum() > 1
    assert adata[row, "Nucleoli_count"].X.item() == siblings.sum()
    np.testing.assert_allclose(
        adata[row, "Nucleoli_AreaShape_Area_mean"].X.item(), np.nanmean(nucleoli.X[siblings, column])
    )


def test_without_aggregate_children_are_named_in_one_line(caplog):
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        adata = mt.io.read_cellprofiler(EXPORTS / "beginner")
    assert "['Nucleoli'], of which one Cells can have several; pass aggregate=" in caplog.text
    assert not [name for name in adata.var_names if name.startswith("Nucleoli_")]
    assert "aggregation" not in adata.var


def test_children_linking_to_no_root_are_counted_in_one_warning(caplog):
    """PERISCOPE: a sixth of the foci lie outside cells; three info lines per site were invisible at the default level."""
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.io.read_cellprofiler(EXPORTS / "periscope_site21", aggregate="count")
    lines = [record.getMessage() for record in caplog.records if "link to no Cells" in record.getMessage()]
    assert lines == [
        "left out 3 of 9 BarcodeFoci, 3 of 9 Foci, 3 of 3 Foci_NonCellEdge object(s), which link to no Cells; "
        "the per-object readers keep every row"
    ]


def test_aggregated_areas_are_not_scored_as_cell_areas_and_names_keep_the_statistic(beginner):
    adata = mt.tl.aggregate_objects(beginner, root="Cells", func="mean")
    from mantispy.pp._qc import _area_features

    assert "Nucleoli_AreaShape_Area_mean" not in _area_features(adata)
    renamed = mt.pp.standardize_feature_names(
        adata[:, ["Cells_AreaShape_Area", "Nucleoli_AreaShape_Area_mean"]], copy=True
    )
    assert list(renamed.var_names) == ["Cells_AreaShape_Area", "Nucleoli_AreaShape_Area_mean"]
