from pathlib import Path

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


def test_from_dataframe_records_an_aggregate_grouping():
    """An aggregate-resolution read carries grouped_by through to the stamp (needed for a per-barcode table)."""
    frame = _frame().assign(Metadata_Barcode=["b1", "b2", "b3", "b4"]).drop(columns=["Metadata_Plate", "Metadata_Well"])
    adata = from_dataframe(frame, channels=["DNA"], resolution="aggregate", grouped_by=["Metadata_Barcode"])
    assert adata.uns["mantispy"]["resolution"] == "aggregate"
    assert adata.uns["mantispy"]["grouped_by"] == ["Metadata_Barcode"]


def test_well_grid_is_filled_on_read():
    """A read fills 0-based Metadata_Row/Col so downstream code need not recompute well geometry."""
    obs = from_dataframe(_frame(well=("A01", "A02", "B01", "H12")), channels=["DNA"]).obs
    assert obs["Metadata_Row"].to_numpy().tolist() == [0, 0, 1, 7]
    assert obs["Metadata_Col"].to_numpy().tolist() == [0, 1, 0, 11]
    assert obs["Metadata_Row"].dtype.kind == "i"
    assert obs["Metadata_Col"].dtype.kind == "i"


def test_well_grid_is_filled_on_stamp():
    adata = ad.AnnData(
        np.ones((2, 2), dtype="float32"),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "B02"]}, index=["0", "1"]),
    )
    adata.var_names = ["Cells_AreaShape_Area", "Cells_AreaShape_Perimeter"]
    mt.io.stamp(adata, resolution="well", feature_kind="measurement")
    assert adata.obs["Metadata_Row"].to_numpy().tolist() == [0, 1]
    assert adata.obs["Metadata_Col"].to_numpy().tolist() == [0, 1]


def test_well_grid_is_filled_by_the_low_level_stamp():
    """A dataset builder that calls _core.schema.stamp directly (e.g. cp_posh) must still get the grid."""
    from mantispy._core.schema import stamp as core_stamp

    adata = ad.AnnData(
        np.ones((2, 2), dtype="float32"),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["B03", "C05"]}, index=["0", "1"]),
    )
    core_stamp(adata, resolution="object")
    assert adata.obs["Metadata_Row"].to_numpy().tolist() == [1, 2]
    assert adata.obs["Metadata_Col"].to_numpy().tolist() == [2, 4]


def test_well_grid_skipped_without_a_well_column():
    """Not every object is well-based, so an object with no Metadata_Well stamps without the columns and without error."""
    frame = _frame().assign(Metadata_Gene=["g1", "g2", "g3", "g4"]).drop(columns=["Metadata_Plate", "Metadata_Well"])
    obs = from_dataframe(frame, channels=["DNA"], resolution="aggregate", grouped_by=["Metadata_Gene"]).obs
    assert "Metadata_Row" not in obs.columns
    assert "Metadata_Col" not in obs.columns


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


def test_dropping_features_with_no_object_is_reported(capsys):
    """A column naming no object (``AreaShape_Perimeter``) parses with a missing object.

    Reporting the drop sorted that missing value alongside the object names and raised a TypeError on pandas 3.
    """
    previous = mt.settings.verbosity
    mt.settings.verbosity = 2
    try:
        frame = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1"],
                "Metadata_Well": ["A01", "A02"],
                "Cells_AreaShape_Area": [1.0, 2.0],
                "AreaShape_Perimeter": [3.0, 4.0],
            }
        )
        assert parse_feature_names(["AreaShape_Perimeter"])["object"].isna().all()
        adata = from_dataframe(frame, channels=["DAPI"])
        assert adata.var_names.tolist() == ["Cells_AreaShape_Area"]
        captured = capsys.readouterr().err
        assert "1 of 2 feature(s)" in captured
        assert "<no object>" in captured
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


def test_stamp_is_atomic_when_full_validation_fails():
    """A float64 matrix fails the schema; stamp must raise and leave the object unstamped, not half-written."""
    adata = ad.AnnData(
        np.ones((2, 3), dtype=np.float64),  # float64, which the 2.0 contract rejects
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
    )
    with pytest.raises(ValueError, match="float32"):
        mt.io.stamp(adata, resolution="well", feature_kind="measurement")
    assert "mantispy" not in adata.uns  # descriptors were not written
    assert "feature_kind" not in adata.var  # var was not touched


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


def test_stamp_mints_object_identity_for_an_external_object():
    """mt.io.stamp(resolution="object") must accept an externally built object, minting its identity."""
    obj = ad.AnnData(
        np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"Metadata_Plate": ["P1"] * 4, "Metadata_Well": ["A01", "A01", "A02", "A02"]},
            index=[str(i) for i in range(4)],
        ),
        var=parse_feature_names(["Cells_AreaShape_Area", "Cells_Intensity_MeanIntensity_DNA"], channels=["DNA"]),
    )
    mt.io.stamp(obj, resolution="object")
    assert {"Metadata_ImageID", "Metadata_ObjectType", "Metadata_ObjectNumber"} <= set(obj.obs.columns)
    assert not obj.obs[["Metadata_ImageID", "Metadata_ObjectNumber"]].duplicated().any()
    assert mt.io.validate(obj).ok, mt.io.validate(obj).errors


def test_stamp_object_type_names_the_object():
    obj = ad.AnnData(
        np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Site": ["s1", "s1", "s2", "s2"]}, index=[str(i) for i in range(4)]),
        var=parse_feature_names(["Cells_AreaShape_Area", "Cells_Intensity_MeanIntensity_DNA"], channels=["DNA"]),
    )
    mt.io.stamp(obj, resolution="object", object_type="cell")
    assert (obj.obs["Metadata_ObjectType"] == "cell").all()


def test_from_dataframe_is_public():
    assert mt.io.from_dataframe is not None


@pytest.mark.parametrize("resolution", ["aggregate", "perturbation"])
def test_an_aggregate_profile_table_can_be_read_with_its_grouping(resolution):
    """A consensus table has no plate or well; the reader takes the grouping, and the 1.0 name still reads."""
    frame = pd.DataFrame(
        {
            "Metadata_Perturbation": ["a", "b", "c"],
            "Cells_AreaShape_Area": [1.0, 2.0, 3.0],
            "Cells_Intensity_MeanIntensity_DNA": [0.1, 0.2, 0.3],
        }
    )
    adata = from_dataframe(frame, channels=["DNA"], resolution=resolution, grouped_by=["Metadata_Perturbation"])
    assert adata.uns["mantispy"]["resolution"] == "aggregate"
    assert list(adata.uns["mantispy"]["grouped_by"]) == ["Metadata_Perturbation"]
    assert mt.io.validate(adata).ok, mt.io.validate(adata).errors


def _a_1_0_well_object():
    adata = from_dataframe(_frame(), channels=["DNA"]).copy()
    adata.obs["Metadata_Control"] = [True, False, False, False]
    adata.obs = adata.obs.drop(columns="Metadata_Control_Type", errors="ignore")
    adata.uns["mantispy"] = {"schema_version": "1.0", "resolution": "well"}
    return adata


def test_write_migrates_an_older_object_instead_of_relabelling_it(tmp_path):
    adata = _a_1_0_well_object()
    mt.io.write(adata, tmp_path / "out.h5ad")
    back = mt.io.read(tmp_path / "out.h5ad")
    assert list(back.obs["Metadata_Control_Type"]) == ["negcon", "treatment", "treatment", "treatment"]


def test_a_failed_write_never_relabels_an_older_object_without_migrating_it(tmp_path):
    """A rejected write must not leave a 1.0 object stamped 2.0 with its 1.0 control model, which migrate then skips."""
    adata = _a_1_0_well_object()
    adata.X = adata.X.astype("float64")  # fails validation
    with pytest.raises(ValueError, match="float32"):
        mt.io.write(adata, tmp_path / "out.h5ad")
    if adata.uns["mantispy"]["schema_version"] == "2.0":
        assert list(adata.obs["Metadata_Control_Type"]) == ["negcon", "treatment", "treatment", "treatment"]


def test_a_failed_write_leaves_the_schema_stamp_untouched(tmp_path):
    adata = from_dataframe(_frame(), channels=["DNA"])
    adata.X = adata.X.astype("float64")  # fails validation
    before = dict(adata.uns["mantispy"])
    with pytest.raises(ValueError, match="float32"):
        mt.io.write(adata, tmp_path / "out.h5ad")
    assert adata.uns["mantispy"] == before


def _raise_midwrite(self, path, *a, **k):
    """Leave a partial output at ``path`` and fail, as an interrupted anndata/h5py write would."""
    path = Path(path)
    if path.suffix == ".zarr" or ".zarr" in path.name:
        path.mkdir()
    else:
        path.write_bytes(b"partial")
    raise RuntimeError("boom")


def test_write_h5ad_atomic_on_failure(tmp_path, cells, monkeypatch):
    monkeypatch.setattr(ad.AnnData, "write_h5ad", _raise_midwrite)
    path = tmp_path / "out.h5ad"
    with pytest.raises(RuntimeError, match="boom"):
        mt.io.write(cells, path)
    assert list(tmp_path.iterdir()) == []
    path.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="boom"):
        mt.io.write(cells, path)
    assert path.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["out.h5ad"]


def test_write_zarr_atomic_on_failure(tmp_path, cells, monkeypatch):
    monkeypatch.setattr(ad.AnnData, "write_zarr", _raise_midwrite)
    path = tmp_path / "out.zarr"
    with pytest.raises(RuntimeError, match="boom"):
        mt.io.write(cells, path)
    assert list(tmp_path.iterdir()) == []
    path.mkdir()
    (path / "marker").write_bytes(b"original")
    with pytest.raises(RuntimeError, match="boom"):
        mt.io.write(cells, path)
    assert (path / "marker").read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["out.zarr"]


def test_write_zarr_sweeps_orphaned_backup_from_a_crashed_run(tmp_path, cells):
    path = tmp_path / "out.zarr"
    mt.io.write(cells, path)
    orphan = tmp_path / ".out.zarr.old-99999"
    orphan.mkdir()
    (orphan / "marker").write_bytes(b"stale")
    mt.io.write(cells, path)
    assert not orphan.exists()
    assert mt.io.read(path).n_obs == cells.n_obs


def test_stamp_translates_the_control_model_of_a_1_0_object():
    """mt.io.stamp is the route migrate recommends for a 1.0 object it refuses, so it must not keep the old flag."""
    adata = _a_1_0_well_object()
    del adata.var["feature_kind"]
    mt.io.stamp(adata, feature_kind="embedding")
    assert list(adata.obs["Metadata_Control_Type"]) == ["negcon", "treatment", "treatment", "treatment"]
    assert adata.uns["mantispy"]["resolution"] == "well"


def _embedding_adata():
    adata = ad.AnnData(
        np.ones((2, 3), dtype="float32"),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
    )
    adata.var_names = ["emb_0", "emb_1", "emb_2"]
    mt.io.stamp(adata, resolution="well", feature_kind="embedding")
    return adata


def test_stamp_has_no_provenance_kwarg():
    adata = _embedding_adata()
    with pytest.raises(TypeError):
        mt.io.stamp(adata, resolution="well", feature_kind="embedding", provenance={"model": "x"})


def test_embedding_feature_kind_still_valid():
    adata = _embedding_adata()
    assert mt.io.validate(adata).ok
    assert "embedding" not in adata.uns["mantispy"]
    assert set(adata.var["feature_kind"][adata.var["is_feature"]]) == {"embedding"}


def test_embedding_uns_slot_is_not_a_registered_key():
    adata = _embedding_adata()
    adata.uns["mantispy"]["embedding"] = {"model": "m"}
    assert any("not a registered result key" in w for w in mt.io.validate(adata).warnings)


def test_stamp_records_the_perturbation_column():
    adata = _embedding_adata()
    assert "perturbation" not in adata.uns["mantispy"]
    adata.obs["Metadata_Perturbation"] = ["a", "b"]
    adata.obs["Metadata_Control"] = [True, False]
    adata.obs["Metadata_Control_Type"] = ["negcon", "treatment"]
    mt.io.stamp(adata, resolution="well", feature_kind="embedding")
    assert adata.uns["mantispy"]["perturbation"]["column"] == "Metadata_Perturbation"
    assert adata.uns["mantispy"]["perturbation"]["control_column"] == "Metadata_Control_Type"


def test_stamp_perturbation_override_and_validation():
    adata = _embedding_adata()
    adata.obs["gene"] = ["a", "b"]
    out = mt.io.stamp(adata, resolution="well", feature_kind="embedding", perturbation={"column": "gene"}, copy=True)
    assert out.uns["mantispy"]["perturbation"]["column"] == "gene"
    with pytest.raises(ValueError, match="no_such_column"):
        mt.io.stamp(adata, resolution="well", feature_kind="embedding", perturbation={"column": "no_such_column"})
    assert "perturbation" not in adata.uns["mantispy"]


def test_stamp_rejects_a_missing_control_column():
    adata = _embedding_adata()
    adata.obs["gene"] = ["a", "b"]
    with pytest.raises(ValueError, match="control_column"):
        mt.io.stamp(
            adata,
            resolution="well",
            feature_kind="embedding",
            perturbation={"column": "gene", "control_column": "typo"},
        )
    assert "perturbation" not in adata.uns["mantispy"]


def test_stamp_omits_absent_default_control_column():
    adata = _embedding_adata()
    adata.obs["Metadata_Perturbation"] = ["a", "b"]
    assert "Metadata_Control_Type" not in adata.obs
    mt.io.stamp(adata, resolution="well", feature_kind="embedding")
    record = adata.uns["mantispy"]["perturbation"]
    assert record["column"] == "Metadata_Perturbation"
    assert "control_column" not in record
    assert not [w for w in mt.io.validate(adata).warnings if "control_column" in w]
    gene = _embedding_adata()
    gene.obs["gene"] = ["a", "b"]
    mt.io.stamp(gene, resolution="well", feature_kind="embedding", perturbation={"column": "gene"})
    assert "control_column" not in gene.uns["mantispy"]["perturbation"]
    assert not [w for w in mt.io.validate(gene).warnings if "control_column" in w]


def test_perturbation_record_round_trips(tmp_path):
    adata = _embedding_adata()
    adata.obs["gene"] = ["a", "b"]
    mt.io.stamp(adata, resolution="well", feature_kind="embedding", perturbation={"column": "gene"})
    path = tmp_path / "p.h5ad"
    mt.io.write(adata, path)
    assert mt.io.read(path).uns["mantispy"]["perturbation"]["column"] == "gene"


def test_validate_suggests_recording_the_perturbation_column():
    adata = _embedding_adata()
    adata.obs["Metadata_Perturbation"] = ["a", "b"]
    report = mt.io.validate(adata)
    assert report.ok
    assert sum("perturbation=" in w for w in report.warnings) == 1
    mt.io.stamp(adata, resolution="well", feature_kind="embedding")
    assert not any("perturbation=" in w for w in mt.io.validate(adata).warnings)


def test_loaders_write_no_channels_or_dataset(tmp_path):
    path = tmp_path / "profiles.csv"
    _frame().to_csv(path, index=False)
    loaded = (
        from_dataframe(_frame(), channels=["DNA"]),
        mt.io.read_profiles(path, channels=["DNA"]),
        mt.ds.synthetic_plate(),
    )
    for adata in loaded:
        assert "channels" not in adata.uns["mantispy"]
        assert "dataset" not in adata.uns["mantispy"]


def test_unrecognised_numeric_column_kept_as_opaque_feature():
    df = pd.DataFrame({"Metadata_Site": ["s1", "s2"], "my_latent_0": [1.0, 2.0], "my_latent_1": [2.0, 3.0]})
    adata = from_dataframe(df, resolution="object")
    assert {"my_latent_0", "my_latent_1"} <= set(adata.var_names)
    assert (adata.var.loc[["my_latent_0", "my_latent_1"], "feature_kind"] == "embedding").all()
    assert adata.n_vars == 2


def test_recognised_non_features_still_excluded_beside_opaque_columns():
    df = pd.DataFrame(
        {
            "Metadata_Site": ["s1", "s2"],
            "Location_Center_X": [1.0, 2.0],
            "ObjectNumber": [1, 2],
            "Cells_AreaShape_Area": [10.0, 11.0],
            "my_latent_0": [0.5, 0.6],
        }
    )
    adata = from_dataframe(df, resolution="object")
    assert set(adata.var_names) == {"Cells_AreaShape_Area", "my_latent_0"}
    assert adata.var.loc["Cells_AreaShape_Area", "feature_kind"] != "embedding"


def test_keep_non_features_moves_only_recognised_non_features_to_obs():
    df = pd.DataFrame({"Metadata_Site": ["s1", "s2"], "Location_Center_X": [1.0, 2.0], "my_latent_0": [0.5, 0.6]})
    adata = from_dataframe(df, resolution="object", keep_non_features=True)
    assert list(adata.var_names) == ["my_latent_0"]
    assert "Location_Center_X" in adata.obs


def test_table_of_only_opaque_columns_builds():
    df = pd.DataFrame({"Metadata_Site": ["s1", "s2"], "dl_a": [1.0, 2.0], "dl_b": [3.0, 4.0], "dl_c": [5.0, 6.0]})
    adata = from_dataframe(df, resolution="object")
    assert list(adata.var_names) == ["dl_a", "dl_b", "dl_c"]
    assert (adata.var["feature_kind"] == "embedding").all()


def test_cellprofiler_frame_gains_no_columns_from_keep_by_default():
    df = _frame()
    df["Cells_Location_Center_X"] = 1.0
    df["Cells_Number_Object_Number"] = 1.0
    adata = from_dataframe(df)
    assert set(adata.var_names) == {"Cells_AreaShape_Area", "Cells_Intensity_MeanIntensity_DNA"}


def test_opaque_column_is_an_embedding_with_every_object_kept():
    df = pd.DataFrame({"Metadata_Site": ["s1", "s2"], "my_latent_0": [1.0, 2.0], "Cells_AreaShape_Area": [3.0, 4.0]})
    adata = from_dataframe(df, resolution="object", objects=None)
    assert adata.var.loc["my_latent_0", "feature_kind"] == "embedding"
    assert adata.var.loc["Cells_AreaShape_Area", "feature_kind"] == "measurement"


@pytest.mark.parametrize("name", ["Image_ImageNumber", "Image_TableNumber", "Cells_ImageNumber"])
def test_prefixed_image_and_table_numbers_are_excluded(name):
    df = pd.DataFrame({"Metadata_Site": ["s1", "s2"], name: [1.0, 2.0], "Cells_AreaShape_Area": [3.0, 4.0]})
    adata = from_dataframe(df, resolution="object", objects=None)
    assert list(adata.var_names) == ["Cells_AreaShape_Area"]


def test_mixed_measurement_and_embedding_round_trip(tmp_path):
    df = pd.DataFrame(
        {
            "Metadata_Site": ["s1", "s2", "s3"],
            "my_latent_0": [0.1, 0.2, 0.3],
            "Cells_AreaShape_Area": [10.0, 20.0, 30.0],
        }
    )
    adata = from_dataframe(df, resolution="object")
    assert mt.io.validate(adata).ok, mt.io.validate(adata).errors
    path = tmp_path / "mixed.h5ad"
    mt.io.write(adata, path)
    loaded = mt.io.read(path)
    assert loaded.var.loc["Cells_AreaShape_Area", "feature_kind"] == "measurement"
    assert loaded.var.loc["my_latent_0", "feature_kind"] == "embedding"
    got = pd.DataFrame(loaded.X, columns=loaded.var_names)
    np.testing.assert_allclose(got["Cells_AreaShape_Area"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(got["my_latent_0"], [0.1, 0.2, 0.3], rtol=1e-6)


def test_from_dataframe_cp_measure_builds_features_and_keeps_opaque():
    df = pd.DataFrame(
        {
            "Metadata_Site": ["s1", "s1"],
            "Intensity_MeanIntensity__DNA": [0.5, 0.6],
            "AreaShape_Area": [10.0, 11.0],
            "my_latent_0": [1.0, 2.0],
        }
    )
    adata = from_dataframe(df, convention="cp_measure", object_type="cell", resolution="object")
    var = adata.var
    assert list(adata.var_names) == ["Intensity_MeanIntensity__DNA", "AreaShape_Area", "my_latent_0"]
    assert var.loc["Intensity_MeanIntensity__DNA", "feature_kind"] == "measurement"
    assert var.loc["Intensity_MeanIntensity__DNA", "feature_group"] == "Intensity"
    assert var.loc["Intensity_MeanIntensity__DNA", "channel"] == "DNA"
    assert var.loc["AreaShape_Area", "feature_kind"] == "measurement"
    assert var.loc["my_latent_0", "feature_kind"] == "embedding"
    assert (adata.obs["Metadata_ObjectType"] == "cell").all()


def test_from_dataframe_cp_measure_keeps_colocalization_channel_pair():
    df = pd.DataFrame(
        {
            "Metadata_Site": ["s1", "s1"],
            "Correlation_Correlation__DNA__ER": [0.1, 0.2],
            "Intensity_MeanIntensity__DNA": [0.5, 0.6],
        }
    )
    adata = from_dataframe(df, convention="cp_measure", object_type="cell", resolution="object")
    assert adata.var.loc["Correlation_Correlation__DNA__ER", "channel"] == "DNA|ER"
    assert adata.var.loc["Intensity_MeanIntensity__DNA", "channel"] == "DNA"
    assert "channels" not in adata.uns["mantispy"]


def test_from_dataframe_reads_object_skeleton_as_a_measurement():
    df = pd.DataFrame(
        {
            "Metadata_Plate": ["P", "P"],
            "Metadata_Well": ["A01", "A02"],
            "Cells_ObjectSkeleton_NumberBranchEnds": [2.0, 3.0],
        }
    )
    row = parse_feature_names(["Cells_ObjectSkeleton_NumberBranchEnds"]).iloc[0]
    assert row["feature_group"] == "ObjectSkeleton"
    assert bool(row["is_feature"])
    adata = from_dataframe(df)
    assert adata.var.loc["Cells_ObjectSkeleton_NumberBranchEnds", "feature_kind"] == "measurement"


def test_from_dataframe_reads_an_unknown_cellprofiler_group_as_an_embedding():
    """A recognised object with a group outside the known set is kept, but as an opaque embedding feature."""
    df = pd.DataFrame(
        {
            "Metadata_Plate": ["P", "P"],
            "Metadata_Well": ["A01", "A02"],
            "Cells_AreaShape_Area": [3.0, 4.0],
            "Cells_Math_Value": [1.0, 2.0],
        }
    )
    adata = from_dataframe(df)
    assert adata.var.loc["Cells_AreaShape_Area", "feature_kind"] == "measurement"
    assert adata.var.loc["Cells_Math_Value", "feature_kind"] == "embedding"


def test_cp_measure_table_leaves_the_cellprofiler_default_alone():
    df = pd.DataFrame({"Metadata_Plate": ["P"], "Metadata_Well": ["A01"], "Cells_AreaShape_Area": [3.0]})
    adata = from_dataframe(df)
    assert list(adata.var_names) == ["Cells_AreaShape_Area"]
    assert adata.var.loc["Cells_AreaShape_Area", "object"] == "Cells"
