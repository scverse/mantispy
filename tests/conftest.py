"""Fixtures shared by more than one test module."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from _testdata import CHANNELS, OVERLAY_PLATE, PLATE, PLATES, build_export, write_cellprofiler_dir, write_plate

from mantispy._core.schema import stamp
from mantispy.ds import synthetic_plate

# re-exported: the test modules import these from here
__all__ = ["CHANNELS", "write_cellprofiler_dir"]


@pytest.fixture
def make_cellprofiler_dir():
    """Factory for tests that need a variant export (different link, channels, duplicates)."""
    return write_cellprofiler_dir


@pytest.fixture
def cellprofiler_dir(tmp_path):
    return write_cellprofiler_dir(tmp_path / "cp")


@pytest.fixture
def platemap_path(tmp_path):
    frame = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02"],
            "Metadata_Perturbation": ["DMSO", "compound_a"],
            "Metadata_Concentration": [0.0, 10.0],
        }
    )
    path = tmp_path / "platemap.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def cells():
    """A small single-cell plate."""
    return synthetic_plate(n_plates=2, n_wells=24, n_cells=15, n_features=20, seed=0)


@pytest.fixture
def wells(cells):
    """The same plate aggregated to wells."""
    import mantispy as mt

    return mt.tl.aggregate(cells, min_cells=0)


@pytest.fixture
def well_profiles():
    """Factory for well profiles nested in plates, half of each plate treated.

    Every well on a plate shares a plate shift. Without that shared structure a well-level
    test cannot be told apart from a naive one.
    """

    def build(n_plates=4, per_plate=8, n_features=200, plate_sd=0.8, effect=0.0, affected=0, seed=0):
        rng = np.random.default_rng(seed)
        plate_of_well = np.repeat(np.arange(n_plates), per_plate)
        values = rng.normal(size=(n_plates * per_plate, n_features))
        values += rng.normal(scale=plate_sd, size=(n_plates, n_features))[plate_of_well]

        treated = np.arange(values.shape[0]) % per_plate < per_plate // 2
        if affected:
            values[np.ix_(treated, np.arange(affected))] += effect

        obs = pd.DataFrame(
            {
                "Metadata_Plate": [f"P{plate}" for plate in plate_of_well],
                "Metadata_Well": [f"A{index + 1:02d}" for index in range(values.shape[0])],
                "Metadata_Perturbation": np.where(treated, "compound", "DMSO"),
                "Metadata_Control": ~treated,
            },
            index=[str(index) for index in range(values.shape[0])],
        )
        adata = ad.AnnData(X=values.astype(np.float32), obs=obs)
        adata.var_names = [f"Cells_AreaShape_F{index}" for index in range(n_features)]
        stamp(adata, resolution="well")
        return adata

    return build


@pytest.fixture
def pure_noise_screen():
    """A screen in which nothing happened, so every hit called on it is a false positive.

    Both nulls are biased by the ratio of features to reference rows. The bias shows on a
    wide, control-poor screen and hides on a narrow, control-rich one, so the defaults
    (120 controls, 80 features) are wide: a biased null calls most groups here, but almost
    none at 200 controls and 10 features.
    """

    def build(n_control=120, n_groups=12, per_group=12, n_features=80, seed=0):
        generator = np.random.default_rng(seed)
        values = generator.standard_normal((n_control + n_groups * per_group, n_features))
        labels = ["DMSO"] * n_control + [f"p{i:02d}" for i in range(n_groups) for _ in range(per_group)]
        obs = pd.DataFrame(
            {
                "Metadata_Perturbation": labels,
                "Metadata_Control": [label == "DMSO" for label in labels],
                "Metadata_Plate": "P1",
                "Metadata_Well": [f"A{i:04d}" for i in range(len(labels))],
            },
            index=[str(i) for i in range(len(labels))],
        )
        adata = ad.AnnData(
            values.astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=[f"Cells_AreaShape_f{i}" for i in range(n_features)]),
        )
        stamp(adata, resolution="well")
        return adata

    return build


@pytest.fixture
def gallery(tmp_path: Path) -> Path:
    """A minimal gallery source: two plates, two wells each, one well analysed, the second plate using overlays."""
    for plate in PLATES:
        write_plate(tmp_path, plate, overlay=plate == OVERLAY_PLATE, located=True)
    # a source that has been read once holds zarr stores, which carry a tables/ directory like an export does
    (tmp_path / f"{PLATE}.zarr" / "tables").mkdir(parents=True)
    (tmp_path / f"{PLATE}.zarr" / "tables" / "table.h5ad").write_bytes(b"")
    return tmp_path


@pytest.fixture
def unlocated_gallery(tmp_path: Path) -> Path:
    """A source that recorded no stage coordinates, so its fields cannot be laid out."""
    write_plate(tmp_path, PLATE, overlay=False, located=False)
    return tmp_path


@pytest.fixture
def make_export(tmp_path: Path) -> Callable[..., Path]:
    """Build one plate folder of an ExportForSpatialData run; see :func:`_testdata.build_export`."""
    return partial(build_export, tmp_path)


@pytest.fixture
def export(make_export: Callable[..., Path]) -> Path:
    return make_export()
