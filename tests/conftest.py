from collections.abc import Callable
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from _testdata import OVERLAY_PLATE, PLATE, PLATES, build_export, write_plate


@pytest.fixture
def adata():
    adata = ad.AnnData(X=np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]).astype(np.float32))
    adata.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32)

    return adata


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
    """Build one plate folder of a CellProfiler export; see :func:`_testdata.build_export`."""
    return partial(build_export, tmp_path)


@pytest.fixture
def export(make_export: Callable[..., Path]) -> Path:
    return make_export()
