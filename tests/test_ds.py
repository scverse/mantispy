from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import mantispy as mt
from mantispy.ds._datasets import _DATASETS


@pytest.mark.parametrize(("n_wells", "n_sites"), [(1, 1), (4, 2)])
def test_blobs_is_a_plate_with_everything_on_it(n_wells: int, n_sites: int) -> None:
    sdata = mt.ds.blobs(n_wells=n_wells, n_sites=n_sites)

    assert len(sdata.images) == n_wells * n_sites
    assert len(sdata.labels) == n_wells * n_sites * 3
    assert set(sdata.tables) == {"cells", "wells"}
    assert sdata.tables["wells"].n_obs == n_wells
    assert len(sdata.shapes["BLOBS01_wells"]) == 96


def test_blobs_looks_like_a_plate_that_was_read() -> None:
    sdata = mt.ds.blobs(n_wells=2, n_sites=2)
    cells = sdata["BLOBS01_A01_s1_cells"].to_numpy()
    nuclei = sdata["BLOBS01_A01_s1_nuclei"].to_numpy()
    var = sdata.tables["cells"].var

    assert {"BLOBS01", "BLOBS01_A01", "BLOBS01_A01_s1"} <= set(sdata.coordinate_systems)
    assert sd.get_extent(sdata["BLOBS01_A02_s1_image"], coordinate_system="BLOBS01")["x"][0] == pytest.approx(9000.0)
    assert set(np.unique(nuclei)) <= set(np.unique(cells))
    assert np.array_equal(sdata["BLOBS01_A01_s1_cytoplasm"].to_numpy(), np.where(nuclei > 0, 0, cells))
    assert var.loc["Cells_Intensity_MeanIntensity_DNA", "channel"] == "DNA"
    assert var.loc["Cells_Correlation_Correlation_DNA_RNA", "channel"] == "DNA|RNA"
    for table in sdata.tables.values():
        assert mt.io.validate(table).ok, mt.io.validate(table).errors


def test_the_same_seed_gives_the_same_data() -> None:
    def build(seed: int) -> np.ndarray:
        return np.asarray(mt.ds.blobs(seed=seed).tables["cells"].X)

    assert np.array_equal(build(1), build(1))
    assert not np.array_equal(build(1), build(2))


@pytest.mark.parametrize("name", sorted(_DATASETS))
def test_every_registered_dataset_has_a_loader_and_hashed_files(name: str) -> None:
    entry = _DATASETS[name]

    assert entry.files
    assert all(file.sha256 and (file.s3_key or file.url) for file in entry.files)
    # names starting with an underscore hold files another loader reads, such as JUMP's annotation tables
    assert name.startswith("_") or getattr(mt.ds, name)


@pytest.mark.network
@pytest.mark.slow
# Entries without a shape do not return an AnnData: jump_plate is a SpatialData and jump_export a directory.
@pytest.mark.parametrize("name", sorted(name for name, entry in _DATASETS.items() if "shape" in entry.metadata))
def test_the_downloads_read_back_at_the_shape_the_registry_claims(name: str) -> None:
    adata = getattr(mt.ds, name)()

    assert list(adata.shape) == _DATASETS[name].metadata["shape"]
    assert adata.obs_names.is_unique
    if name == "bbbc021":
        assert adata.obs["Metadata_MOA"].notna().all()
        assert adata.obs["Metadata_Control"].sum() == 330
    # Regression test for #63: every well-level dataset publishes an exact per-well count upstream.
    if name not in ("jump_cells", "pooled_rare"):
        assert (adata.obs["Metadata_CellCount"] > 0).all()
        # jump-profiling-recipe's count table, which jump_crispr reads, has no field count, and
        # JUMP-Lite publishes one count per well rather than per field.
        if name not in ("jump_crispr", "jump_lite"):
            assert adata.obs["Metadata_SiteCount"].between(1, 36).all()


@pytest.mark.network
@pytest.mark.slow
def test_jump_export_is_an_export_directory_that_reads() -> None:
    """The fixture tutorial 1 reads: a real ExportToSpreadsheet directory, untouched."""
    directory = mt.ds.jump_export()

    assert {path.name for path in directory.glob("*.csv")} == {
        "Cells.csv",
        "Cytoplasm.csv",
        "Experiment.csv",
        "Image.csv",
        "Nuclei.csv",
    }
    # JUMP gives Cytoplasm both parents and Cells none, so Cytoplasm is the primary that joins all three.
    adata = mt.io.read_profiles(directory, primary_object="Cytoplasm")
    assert str(adata.obs["Metadata_Well"].iloc[0]) == "J04"
    assert mt.io.validate(adata).ok


@pytest.mark.network
@pytest.mark.slow
def test_jump_cells_holds_controls_and_treatments_at_cell_resolution() -> None:
    adata = mt.ds.jump_cells()

    assert adata.uns["mantispy"]["resolution"] == "cell"
    assert adata.obs["Metadata_Well"].nunique() == 24
    assert adata.obs["Metadata_Site"].nunique() == 4
    assert bool(adata.obs["Metadata_Control"].any()) and not bool(adata.obs["Metadata_Control"].all())
    assert adata.obs_names.is_unique
    assert mt.io.validate(adata).ok


@pytest.mark.network
@pytest.mark.slow
def test_jump_plate_reads_the_fields_that_were_downloaded() -> None:
    sdata = mt.ds.jump_plate()

    assert len(sdata.images) == 2
    assert len(sdata.labels) == 2 * 3
    assert sdata.tables["cells"].n_obs > 0
    # The channel vocabulary comes from load_data, so no invented channel reaches var.
    channels = {part for value in sdata.tables["cells"].var["channel"].dropna() for part in str(value).split("|")}
    assert channels <= {"AGP", "Brightfield", "Brightfield_H", "Brightfield_L", "DNA", "ER", "Mito", "RNA"}


@pytest.mark.network
@pytest.mark.slow
def test_jump_cells_can_hand_back_only_the_selected_features() -> None:
    full = mt.ds.jump_cells()
    selected = mt.ds.jump_cells(selected=True)

    assert selected.n_obs == full.n_obs
    assert selected.n_vars < full.n_vars
    assert list(selected.var_names) == list(full.var_names[full.var["selected"].to_numpy()])


def test_selecting_without_the_annotation_is_refused() -> None:
    """The mask is computed against the negative controls, which only the annotation names."""
    with pytest.raises(KeyError, match="needs annotate"):
        mt.ds.jump_cells(annotate=False, selected=True)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "n_features"),
    [
        ("openphenom", 384),
        ("dinov2", 384),
        ("dinov2_random", 384),
        ("subcell", 1536),
        ("morphem", 1920),
        ("cp_measure", 2550),
    ],
)
def test_every_jump_lite_feature_set_reads_back_at_its_own_width(model: str, n_features: int) -> None:
    """The registry records one shape and the generic shape test only ever loads the default model, so the
    other five widths are asserted nowhere else. The rows are the same wells in all six."""
    adata = mt.ds.jump_lite(model=model, annotate=False)

    assert adata.shape == (1536, n_features)
    assert adata.obs["Metadata_CellCount"].notna().all()


def test_jump_lite_names_its_feature_sets():
    """The six feature sets cover the same wells, so a typo has to fail loudly rather than
    silently fall back to one of them."""
    with pytest.raises(ValueError, match="model must be one of"):
        mt.ds.jump_lite(model="openphenome")
    assert "cp_measure" in mt.ds.JUMP_LITE_MODELS


def test_jump_lite_returns_every_feature_set_in_one_row_order(tmp_path, monkeypatch):
    """Each feature set is distributed with the wells in its own order, so two of them stacked by position
    paired one well's features with another's. Read back, every set lists the wells in the same order."""
    from mantispy.ds import _datasets

    ids = ["P2_A01", "P1_B01", "P1_A01"]
    wells = pd.DataFrame(
        {
            "Metadata_Plate": ["P2", "P1", "P1"],
            "Metadata_Well": ["A01", "B01", "A01"],
            "Metadata_Source": ["source_4", "source_3", "source_3"],
            "Metadata_Batch": "b1",
            "Metadata_id": ids,
            "value": [2.0, 1.0, 0.0],
        }
    )
    wells.rename(columns={"value": "openphenom_0"}).to_parquet(tmp_path / "openphenom.parquet")
    shuffled = wells.iloc[[2, 0, 1]].reset_index(drop=True)
    shuffled.rename(columns={"value": "dinov2_0"}).to_parquet(tmp_path / "dinov2.parquet")
    pd.DataFrame({"Metadata_id": ids, "cell_count": [100, 110, 120]}).to_parquet(tmp_path / "cell_count.parquet")

    def files(name, cache_dir, select=None):
        return [path for path in sorted(tmp_path.glob("*.parquet")) if select is None or select(path.name)]

    monkeypatch.setattr(_datasets, "_files", files)

    first, second = (mt.ds.jump_lite(model=model, annotate=False) for model in ("openphenom", "dinov2"))

    assert list(first.obs_names) == list(second.obs_names)
    assert np.asarray(first.X)[:, 0].tolist() == np.asarray(second.X)[:, 0].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(("model", "parsed"), [("openphenom", False), ("cp_measure", True)])
def test_jump_lite_does_not_read_an_embedding_dimension_as_a_measurement(tmp_path, monkeypatch, model, parsed):
    """The parser finds structure in names that have none: it reads `openphenom_nahualX_17` as the
    `nahualX` feature group of an `openphenom` object, so the model's own tensor names became
    feature families and `scale` became the dimension index. cp_measure is real measurements and
    keeps its annotation."""
    from mantispy.ds import _datasets

    names = (
        [f"{model}_nahualX_{index}" for index in range(4)]
        if model == "openphenom"
        else ["cell_0/max/sizeshapeSolidity", "nuclei_3/max/intensityIntensity_MeanIntensity"]
    )
    wells = pd.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1"],
            "Metadata_Well": ["A01", "A02"],
            "Metadata_Source": ["source_3", "source_3"],
            "Metadata_Batch": ["b1", "b1"],
            "Metadata_id": ["P1_A01", "P1_A02"],
            **{name: [float(index), float(index) + 1] for index, name in enumerate(names)},
        }
    )
    wells.to_parquet(tmp_path / f"{model}.parquet")
    pd.DataFrame({"Metadata_id": ["P1_A01", "P1_A02"], "cell_count": [120, 130]}).to_parquet(
        tmp_path / "cell_count.parquet"
    )

    def files(name, cache_dir, select=None):
        return [path for path in sorted(tmp_path.glob("*.parquet")) if select is None or select(path.name)]

    monkeypatch.setattr(_datasets, "_files", files)

    adata = mt.ds.jump_lite(model=model, annotate=False)

    assert bool(adata.var["feature_group"].notna().any()) is parsed
    # The compartment and the channel are what an embedding cannot offer, so they are what cp_measure has to keep.
    if parsed:
        assert list(adata.var["object"]) == ["cell", "nuclei"]
        assert list(adata.var["feature_group"]) == ["sizeshape", "intensity"]
        assert list(adata.var["channel"]) == ["0", "3"]
    assert list(adata.obs["Metadata_CellCount"]) == [120, 130]
