from __future__ import annotations

from pathlib import Path

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
    assert var.loc["Cells_Intensity_MeanIntensity_DNA", "channel"] == "dna"
    assert var.loc["Cells_Correlation_Correlation_DNA_RNA", "n_channels"] == 2


@pytest.mark.parametrize("dataset", ["blobs", "blobs_profiles"])
def test_the_same_seed_gives_the_same_data(dataset: str) -> None:
    def build(seed: int) -> np.ndarray:
        made = getattr(mt.ds, dataset)(seed=seed)
        return np.asarray(made.tables["cells"].X if dataset == "blobs" else made.X)

    assert np.array_equal(build(1), build(1))
    assert not np.array_equal(build(1), build(2))


def test_blobs_profiles_carry_a_treatment_effect_and_a_plate_effect() -> None:
    adata = mt.ds.blobs_profiles(n_plates=3, n_wells=48)
    treated = np.asarray(adata[adata.obs["treatment"] == "compound_c"].X).mean()
    plate_means = [np.asarray(adata[adata.obs["Plate"] == p].X).mean() for p in adata.obs["Plate"].cat.categories]

    assert adata.shape == (144, 60)
    assert treated > np.asarray(adata[adata.obs["treatment"] == "DMSO"].X).mean()
    assert len(set(np.round(plate_means, 3))) == 3


@pytest.mark.parametrize("name", sorted(_DATASETS))
def test_every_registered_dataset_has_a_loader_and_hashed_files(name: str) -> None:
    entry = _DATASETS[name]

    assert entry.files
    assert all(file.sha256 and (file.s3_key or file.url) for file in entry.files)
    assert getattr(mt.ds, name)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_DATASETS))
def test_the_downloads_read_back_at_the_shape_the_registry_claims(name: str, tmp_path: Path) -> None:
    adata = getattr(mt.ds, name)(cache_dir=tmp_path)

    assert list(adata.shape) == _DATASETS[name].metadata["shape"]
    assert adata.obs_names.is_unique
    if name == "bbbc021":
        assert adata.obs["MOA"].notna().all()
        assert adata.obs["Control"].sum() == 330
