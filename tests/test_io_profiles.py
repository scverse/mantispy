import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.features import parse_feature_names
from mantispy._core.schema import SCHEMA_VERSION
from mantispy.io._profiles import from_dataframe


def _frame(n=4, prefix="Metadata_", well=("A01", "A02", "A03", "A04")):
    return pd.DataFrame(
        {
            f"{prefix}Plate": ["P1"] * n,
            f"{prefix}Well": list(well)[:n],
            "Cells_AreaShape_Area": np.linspace(1.0, 2.0, n),
            "Cells_Intensity_MeanIntensity_DNA": np.linspace(0.1, 0.2, n),
        }
    )


@pytest.mark.parametrize(
    ("published", "expected"),
    [({"Metadata_Count_Cells": 7, "Metadata_Object_Count": 9}, 7.0), ({"Metadata_Object_Count": 9}, 9.0)],
)
def test_pycytominer_counts_are_copied_to_the_names_mantispy_reads(published, expected):
    frame = _frame().assign(**published, Metadata_Site_Count=9)
    obs = from_dataframe(frame, channels=["DNA"]).obs
    assert (obs["Metadata_CellCount"] == expected).all()
    assert (obs["Metadata_SiteCount"] == 9).all()
    # Copied, not renamed: code reading the upstream columns keeps working.
    assert set(published) | {"Metadata_Site_Count"} <= set(obs.columns)


def test_sentinels_become_nan():
    frame = _frame()
    frame.loc[0, "Cells_AreaShape_Area"] = -999.0
    adata = from_dataframe(frame, channels=["DNA"], sentinels=-999.0)
    assert np.isnan(adata[0, "Cells_AreaShape_Area"].X).all()


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".tsv"])
def test_read_profiles_by_suffix(tmp_path, suffix):
    frame, path = _frame(), tmp_path / f"profiles{suffix}"
    if suffix == ".parquet":
        frame.to_parquet(path)
    else:
        frame.to_csv(path, sep="\t" if suffix == ".tsv" else ",", index=False)
    assert mt.io.read_profiles(path, channels=["DNA"]).shape == (4, 2)


def test_column_mismatch_raises_then_intersects(tmp_path):
    """Batches with no shared features occur (cpg0001), so intersecting them must be an explicit choice."""
    a, b = _frame(), _frame()
    b["Cells_AreaShape_Extra"] = 1.0
    (a_path, b_path) = (tmp_path / "a.csv", tmp_path / "b.csv")
    a.to_csv(a_path, index=False)
    b.to_csv(b_path, index=False)

    with pytest.raises(ValueError, match="disagree on columns"):
        mt.io.read_profiles([a_path, b_path], channels=["DNA"])
    adata = mt.io.read_profiles([a_path, b_path], channels=["DNA"], on_column_mismatch="intersect")
    assert adata.n_obs == 8 and adata.n_vars == 2


def test_empty_file_does_not_poison_dtypes(tmp_path):
    full, empty = tmp_path / "full.csv", tmp_path / "empty.csv"
    _frame().to_csv(full, index=False)
    _frame().iloc[:0].to_csv(empty, index=False)
    adata = mt.io.read_profiles([full, empty], channels=["DNA"])
    assert adata.n_obs == 4 and adata.X.dtype == np.float32


def test_path_columns_read_metadata_out_of_the_directory_name(tmp_path):
    directory = tmp_path / "Batch7" / "plateA"
    directory.mkdir(parents=True)
    path = directory / "profiles.csv"
    _frame().to_csv(path, index=False)
    adata = mt.io.read_profiles(path, channels=["DNA"], path_columns={"Metadata_Batch": 2})
    assert set(adata.obs["Metadata_Batch"]) == {"Batch7"}


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"], ids=["h5ad", "zarr"])
def test_write_read_round_trip(tmp_path, cells, suffix):
    path = tmp_path / f"profiles{suffix}"
    mt.io.write(cells, path)
    loaded = mt.io.read(path)
    np.testing.assert_array_equal(loaded.X, cells.X)
    pd.testing.assert_frame_equal(loaded.obs, cells.obs)
    assert loaded.uns["mantispy"]["schema_version"] == SCHEMA_VERSION
    assert isinstance(loaded.uns["mantispy"]["image_table"], pd.DataFrame)


def test_colliding_metadata_prefixes_are_refused():
    """Both normalise to Metadata_Plate, and obs would then hold a duplicate column whose every later lookup returns a frame instead of a series."""
    frame = pd.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1"],
            "Image_Metadata_Plate": ["P1", "P1"],
            "Metadata_Well": ["A01", "A02"],
            "Cells_AreaShape_Area": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="collapse onto the same column name"):
        from_dataframe(frame)


def test_dropping_a_majority_of_features_warns_but_a_minority_stays_quiet(capsys):
    """The default `objects=` filter applies without the caller asking for it and can remove most of a table.

    Losing the majority must reach the user at the default verbosity; losing a minority stays quiet.
    """
    previous = mt.settings.verbosity
    mt.settings.verbosity = 1
    try:
        majority = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1"],
                "Metadata_Well": ["A01", "A02"],
                "Cells_AreaShape_Area": [1.0, 2.0],
                "Image_Texture_Contrast_DNA_3_00_256": [5.0, 6.0],
                "Image_Granularity_1_DNA": [7.0, 8.0],
                "Image_AreaShape_Area": [9.0, 10.0],
            }
        )
        from_dataframe(majority)
        captured = capsys.readouterr().err
        assert "3 of 4 feature(s)" in captured
        assert "objects=None" in captured

        minority = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1"],
                "Metadata_Well": ["A01", "A02"],
                "Cells_AreaShape_Area": [1.0, 2.0],
                "Nuclei_Intensity_MeanIntensity_DNA": [3.0, 4.0],
                "Cytoplasm_AreaShape_Area": [5.0, 6.0],
                "Image_Texture_Contrast_DNA_3_00_256": [7.0, 8.0],
            }
        )
        from_dataframe(minority)
        assert capsys.readouterr().err == ""
    finally:
        mt.settings.verbosity = previous


def _cytotable_part(rows: range) -> pd.DataFrame:
    """A CytoTable-shaped frame: prefixed identifiers, compartment-prefixed features."""
    return pd.DataFrame(
        {
            "Metadata_ImageNumber": [1] * len(rows),
            "Metadata_ObjectNumber": list(rows),
            "Image_Metadata_Plate": ["P1"] * len(rows),
            "Image_Metadata_Well": ["A01"] * len(rows),
            "Cells_AreaShape_Area": np.arange(len(rows), dtype=float),
            "Nuclei_Intensity_MeanIntensity_DNA": np.arange(len(rows), dtype=float) * 2,
        }
    )


def test_a_directory_of_parquet_parts_reads_as_cells(tmp_path):
    """CytoTable writes a partitioned directory, and reading it needs no CytoTable."""
    (tmp_path / "sub").mkdir()
    _cytotable_part(range(3)).to_parquet(tmp_path / "part-0.parquet")
    _cytotable_part(range(3, 7)).to_parquet(tmp_path / "sub" / "part-1.parquet")

    adata = mt.io.read_profiles(tmp_path)
    assert adata.shape == (7, 2)
    assert adata.uns["mantispy"]["resolution"] == "object"
    assert {"Metadata_Plate", "Metadata_Well", "Metadata_ImageNumber"} <= set(adata.obs.columns)
    assert mt.io.validate(adata).ok, mt.io.validate(adata).errors


def test_a_directory_with_nothing_to_read_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither an ExportToSpreadsheet Image.csv nor .parquet parts"):
        mt.io.read_profiles(tmp_path)


def test_an_unknown_on_column_mismatch_is_refused(tmp_path):
    """A typo fell through to the intersect branch, which drops every column the files disagree on without saying so."""
    path = tmp_path / "a.csv"
    _frame().to_csv(path, index=False)

    with pytest.raises(ValueError, match="on_column_mismatch"):
        mt.io.read_profiles(path, channels=["DNA"], on_column_mismatch="intersct")


def test_a_file_with_a_header_and_no_rows_is_refused(tmp_path):
    """It read as a silent 0x0 object with every feature column misfiled into obs, because a column of no values has no dtype to recognise a feature by."""
    path = tmp_path / "header_only.csv"
    _frame().iloc[:0].to_csv(path, index=False)

    with pytest.raises(ValueError, match="no rows"):
        mt.io.read_profiles(path, channels=["DNA"])


def test_platemap_wells_that_match_nothing_are_reported(tmp_path, capsys):
    """They were left silently NaN, so a platemap naming the wrong wells looked like a successful read; io.read_jump warns for the identical join."""
    path = tmp_path / "profiles.csv"
    _frame().to_csv(path, index=False)
    platemap = pd.DataFrame({"Metadata_Well": ["A01"], "Metadata_Perturbation": ["DMSO"]})

    adata = mt.io.read_profiles(path, channels=["DNA"], platemap=platemap)

    assert int(adata.obs["Metadata_Perturbation"].isna().sum()) == 3
    assert "3 of 4" in capsys.readouterr().err


def test_index_columns_name_the_observations(tmp_path):
    path = tmp_path / "profile.csv"
    _frame().to_csv(path, index=False)

    adata = mt.io.read_profiles(path, index_columns=("Metadata_Plate", "Metadata_Well"))
    assert adata.obs_names.tolist() == ["P1:A01", "P1:A02", "P1:A03", "P1:A04"]
    with pytest.raises(ValueError, match="do not identify observations uniquely"):
        mt.io.read_profiles(path, index_columns=("Metadata_Plate",))
    with pytest.raises(KeyError, match="index columns not in metadata"):
        mt.io.read_profiles(path, index_columns=("Metadata_Nope",))


def test_stamp_refuses_what_the_resolution_needs_and_obs_lacks():
    """Stamping regardless would push the failure into whichever tool ran next."""
    adata = ad.AnnData(np.zeros((3, 2), dtype=np.float32), obs=pd.DataFrame(index=list("abc")))
    with pytest.raises(ValueError, match="no resolution to stamp"):
        mt.io.stamp(adata)  # no resolution, and the object records none
    with pytest.raises(ValueError, match=r"Metadata_Plate.*Metadata_Well"):
        mt.io.stamp(adata, resolution="well")
    with pytest.raises(ValueError, match="resolution must be one of"):
        mt.io.stamp(adata, resolution="plate")
    assert "mantispy" not in adata.uns


def test_stamp_leaves_an_annotation_that_is_already_there_alone():
    """Only the absent columns are supplied.

    Filling all ten unconditionally would overwrite a parsed annotation with blanks, so the test deletes one and checks the rest survived.
    """
    var = parse_feature_names(["Cells_AreaShape_Area", "Nuclei_Intensity_MeanIntensity_DNA"])
    parsed = var.drop(columns=["channel"]).copy()
    obj = ad.AnnData(
        np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=parsed,
    )

    mt.io.stamp(obj, resolution="well")

    assert obj.var["channel"].isna().all(), "the one that was absent is supplied empty"
    for column in parsed.columns:
        # .equals, not ==: a column the parser left empty holds NaN, which is not equal to itself.
        assert obj.var[column].equals(var[column]), f"{column} kept what the parser found"
