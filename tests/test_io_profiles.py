import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.features import canonical_channel
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


def test_from_dataframe_splits_features_and_metadata():
    adata = from_dataframe(_frame(), channels=["DNA"])
    assert adata.shape == (4, 2)
    assert adata.X.dtype == np.float32
    assert list(adata.obs.columns) == ["Metadata_Plate", "Metadata_Well"]
    assert adata.var.loc["Cells_Intensity_MeanIntensity_DNA", "channel"] == "DNA"
    assert adata.uns["mantispy"]["resolution"] == "well"


@pytest.mark.parametrize("prefix", ["Image_Metadata_", "Metadata_", "metadata_", "meta_"])
def test_every_real_world_metadata_prefix_is_recognised(prefix):
    """Accessions in the Cell Painting Gallery use all four spellings."""
    adata = from_dataframe(_frame(prefix=prefix), channels=["DNA"])
    assert "Metadata_Plate" in adata.obs
    assert adata.n_vars == 2


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


def test_read_profiles_stacks_several_files(tmp_path):
    paths = []
    for plate in ("P1", "P2"):
        frame = _frame()
        frame["Metadata_Plate"] = plate
        path = tmp_path / f"{plate}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    adata = mt.io.read_profiles(paths, channels=["DNA"])
    assert adata.n_obs == 8
    assert set(adata.obs["Metadata_Plate"]) == {"P1", "P2"}


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


def test_write_refuses_an_invalid_object(tmp_path, cells):
    cells.obs = cells.obs.drop(columns="Metadata_Plate")
    with pytest.raises(ValueError, match="Metadata_Plate"):
        mt.io.write(cells, tmp_path / "bad.h5ad")


def test_read_rejects_a_foreign_schema_version(tmp_path, cells):
    import anndata as ad

    path = tmp_path / "old.h5ad"
    mt.io.write(cells, path)
    stored = ad.read_h5ad(path)
    stored.uns["mantispy"]["schema_version"] = "99.0"
    stored.write_h5ad(path)
    with pytest.raises(ValueError, match="99.0"):
        mt.io.read(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Hoechst", "dna"), ("DAPI", "dna"), ("DNA|ER", "dna|er"), ("GFP", "gfp"), (None, None)],
)
def test_channel_aliases_make_vocabularies_comparable(raw, expected):
    """Datasets that name the nuclear channel differently should still compare."""
    assert canonical_channel(raw) == expected


def test_colliding_metadata_prefixes_are_refused():
    """Both normalise to Metadata_Plate, and obs would then hold a duplicate column whose
    every later lookup returns a frame instead of a series."""
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


def test_image_level_features_are_excluded_by_default_and_can_be_kept():
    """A JUMP plate carries 1077 whole-field Image_ features against 3634 per-cell ones.
    They are excluded by default, as in pycytominer's default compartments, and
    objects=None keeps them."""
    frame = pd.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1"],
            "Metadata_Well": ["A01", "A02"],
            "Cells_AreaShape_Area": [1.0, 2.0],
            "Nuclei_Intensity_MeanIntensity_DNA": [3.0, 4.0],
            "Image_Texture_Contrast_DNA_3_00_256": [5.0, 6.0],
            "Image_Granularity_1_DNA": [7.0, 8.0],
        }
    )

    default = from_dataframe(frame)
    assert set(default.var_names) == {"Cells_AreaShape_Area", "Nuclei_Intensity_MeanIntensity_DNA"}

    everything = from_dataframe(frame, objects=None)
    assert "Image_Texture_Contrast_DNA_3_00_256" in set(everything.var_names)
    assert set(everything.var["object"].astype(str)) == {"Cells", "Nuclei", "Image"}

    nuclei_only = from_dataframe(frame, objects=("Nuclei",))
    assert set(nuclei_only.var_names) == {"Nuclei_Intensity_MeanIntensity_DNA"}


def test_dropping_a_majority_of_features_warns_but_a_minority_stays_quiet(capsys):
    """The default `objects=` filter applies without the caller asking for it and can remove
    most of a table. Losing the majority must reach the user at the default verbosity;
    losing a minority stays quiet."""
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


def test_a_frame_whose_compartments_are_singular_is_refused_by_name():
    """pycytominer and many custom pipelines emit Cell_/Nucleus_ rather than
    Cells_/Nuclei_. The default filter would drop all of them and leave an (n, 0) AnnData
    that mt.io.validate() accepts."""
    frame = pd.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1"],
            "Metadata_Well": ["A01", "A02"],
            "Cell_AreaShape_Area": [1.0, 2.0],
            "Nucleus_Intensity_MeanIntensity_DNA": [3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="objects=None"):
        from_dataframe(frame)

    kept = from_dataframe(frame, objects=None)
    assert kept.n_vars == 2


def test_a_healthy_small_export_does_not_warn_about_its_image_columns(capsys):
    """Image_ columns removed by the default object filter do not trigger a warning.

    A four-image CellProfiler export routinely has more whole-field Image_ measurements
    than per-cell ones. Warning on every such read would teach users to ignore the warning
    for a feature set too thin to profile, which is covered above.
    """
    previous = mt.settings.verbosity
    mt.settings.verbosity = 1
    try:
        frame = pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]})
        for index in range(12):  # comfortably above THIN_FEATURE_SET
            frame[f"Cells_AreaShape_F{index}"] = [float(index), float(index) + 1]
        for index in range(30):  # and outnumbered by whole-field columns
            frame[f"Image_Texture_Contrast_DNA_{index}_00_256"] = [1.0, 2.0]

        adata = from_dataframe(frame)
        assert adata.n_vars == 12
        assert capsys.readouterr().err == "", "nothing reaches the default verbosity"

        mt.settings.verbosity = 2
        from_dataframe(frame)
        captured = capsys.readouterr().err
        assert "dropped 30 of 42" in captured, "it still says what it did, as bookkeeping"
        assert "WARNING" not in captured.upper()
    finally:
        mt.settings.verbosity = previous


def test_the_inferred_channel_vocabulary_travels_with_the_object():
    """A parse is only reproducible if what it was parsed with is recorded.

    The vocabulary is read off the column names handed in, so a subset of a plate can
    infer a smaller one than the whole plate and parse the same column differently:
    Cells_Correlation_Correlation_AGP_DNA is channel 'AGP|DNA' when AGP is in the
    vocabulary and channel 'DNA', feature 'Correlation_AGP', when it is not.
    """
    shared = ["Metadata_Plate", "Metadata_Well"]
    correlation = "Cells_Correlation_Correlation_AGP_DNA"
    full = pd.DataFrame(
        {
            "Metadata_Plate": ["P1"],
            "Metadata_Well": ["A01"],
            "Cells_Intensity_MeanIntensity_AGP": [1.0],
            "Cells_Intensity_MeanIntensity_DNA": [2.0],
            correlation: [0.5],
        }
    )
    subset = full[[*shared, correlation]]

    parsed_full = from_dataframe(full)
    parsed_subset = from_dataframe(subset)

    assert parsed_full.uns["mantispy"]["channels"] == ["AGP", "DNA"]
    assert parsed_subset.uns["mantispy"]["channels"] == ["DNA"]
    assert parsed_full.var.loc[correlation, "channel"] != parsed_subset.var.loc[correlation, "channel"]

    # Naming the vocabulary makes the subset parse the way the full plate did.
    pinned = from_dataframe(subset, channels=["AGP", "DNA"])
    assert pinned.var.loc[correlation, "channel"] == parsed_full.var.loc[correlation, "channel"]
    assert pinned.var.loc[correlation, "feature"] == parsed_full.var.loc[correlation, "feature"]


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
    assert adata.uns["mantispy"]["resolution"] == "cell"
    assert {"Metadata_Plate", "Metadata_Well", "Metadata_ImageNumber"} <= set(adata.obs.columns)
    assert mt.io.validate(adata).ok, mt.io.validate(adata).errors


def test_a_directory_with_nothing_to_read_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither an ExportToSpreadsheet Image.csv nor .parquet parts"):
        mt.io.read_profiles(tmp_path)


def test_index_columns_name_the_observations(tmp_path):
    path = tmp_path / "profile.csv"
    _frame().to_csv(path, index=False)

    adata = mt.io.read_profiles(path, index_columns=("Metadata_Plate", "Metadata_Well"))
    assert adata.obs_names.tolist() == ["P1:A01", "P1:A02", "P1:A03", "P1:A04"]
    with pytest.raises(ValueError, match="do not identify observations uniquely"):
        mt.io.read_profiles(path, index_columns=("Metadata_Plate",))
    with pytest.raises(KeyError, match="index columns not in metadata"):
        mt.io.read_profiles(path, index_columns=("Metadata_Nope",))
