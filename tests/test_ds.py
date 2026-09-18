from __future__ import annotations

import numpy as np
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
    assert adata.obs["Metadata_Well"].nunique() == 6
    assert adata.obs["Metadata_Site"].nunique() == 2
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
