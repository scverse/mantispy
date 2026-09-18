from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import dask.array as da
    import numpy.typing as npt
    from spatialdata import SpatialData

DATASET = "data"
MANIFEST = "cellprofiler_mapping"
ELEMENTS = "elements"
IMAGE_CHANNELS = "image_channels"
ELEMENT_IMAGE = "image"
ELEMENT_LABELS = "labels"
STATUS_OK = "ok"
REGION_KEY = "region_key"
INSTANCE_KEY = "label_id"

# CellProfiler narrows label arrays to the object count, so fields of one plate would otherwise disagree.
LABEL_DTYPE = np.uint32


def _table_path(plate: Path) -> Path:
    tables = sorted((plate / "tables").glob("*.h5ad"))
    if not tables:
        msg = f"no table under {plate / 'tables'}; is {plate} a plate folder of an export?"
        raise FileNotFoundError(msg)
    if len(tables) > 1:
        msg = f"expected one table under {plate / 'tables'}, found {[path.name for path in tables]}"
        raise ValueError(msg)
    return tables[0]


def _manifest(adata: ad.AnnData, plate: Path) -> tuple[pd.DataFrame, list[str]]:
    mapping = adata.uns.get(MANIFEST, {})
    if ELEMENTS not in mapping:
        msg = (
            f"{plate} has no uns['{MANIFEST}']['{ELEMENTS}'], so it holds no images or segmentations. "
            "ExportToAnnData writes a table alone; ExportForSpatialData writes the manifest this reader needs."
        )
        raise ValueError(msg)
    channels = mapping[IMAGE_CHANNELS].sort_values("stack_index")["channel"].astype(str).tolist()
    return mapping[ELEMENTS], channels


class _LazyDataset:
    """One HDF5 dataset that opens its file for each read instead of holding it open.

    A dask array over a live ``h5py`` dataset keeps one file descriptor for as long as the array is referenced.
    One plate of a 384-well export holds about ten thousand elements, which exhausts the process descriptor limit before the plate finishes reading.
    Reopening per read costs an open per chunk and keeps the descriptor count flat.
    """

    def __init__(self, path: Path) -> None:
        import h5py

        self.path = path
        with h5py.File(path, "r") as handle:
            dataset = handle[DATASET]
            self.shape: tuple[int, ...] = dataset.shape
            self.dtype = dataset.dtype
            #: The dataset's own chunking, which dask reads in as its chunks.
            self.hdf5_chunks: tuple[int, ...] | None = dataset.chunks
        self.ndim = len(self.shape)

    def __getitem__(self, key: Any) -> npt.NDArray:
        import h5py

        with h5py.File(self.path, "r") as handle:
            return handle[DATASET][key]


def _read_array(path: Path, *, lazy: bool) -> npt.NDArray | da.Array:
    import h5py

    if not lazy:
        with h5py.File(path, "r") as handle:
            return handle[DATASET][()]
    import dask.array as da

    dataset = _LazyDataset(path)
    # lock: HDF5 is not thread-safe and dask reads on several threads.
    # name: the path identifies the contents, and tokenizing the reader object would not.
    return da.from_array(dataset, chunks=dataset.hdf5_chunks or "auto", lock=True, name=f"mantispy-h5-{path}")


def read_cellprofiler_export(path: Path | str, *, lazy: bool = True) -> SpatialData:
    """Read one plate folder written by the ``ExportForSpatialData`` CellProfiler module.

    See :func:`mantispy.io.read_plate`, which dispatches here, for what the result holds.

    Args:
        path: A plate folder of an export, the directory holding ``images/``, ``labels/`` and ``tables/``.
        lazy: Read arrays through dask, one HDF5 dataset per element, instead of loading them into memory.
            Each read reopens the file, so a plate of many fields holds no descriptor per element.

    Returns:
        The plate as a :class:`~spatialdata.SpatialData` object, with the Images and Labels the manifest lists as written and a ``cells`` Table annotating them.

    Raises:
        FileNotFoundError: No table under ``path/tables``, or the manifest names an array that is not there.
        ValueError: Several tables under ``path/tables``, or the table carries no element manifest, or it has no ``region_key`` column to join its rows onto the label arrays.
    """
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Labels2DModel, TableModel
    from spatialdata.transformations import Identity

    plate = Path(path)
    adata = ad.read_h5ad(_table_path(plate))
    elements, channels = _manifest(adata, plate)
    written = elements[elements["status"] == STATUS_OK]

    images: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    for row in written.to_dict("records"):
        field = str(row["sample_key"])
        transformations = {field: Identity()}
        array = _read_array(plate / str(row["path"]), lazy=lazy)
        if row["element_type"] == ELEMENT_IMAGE:
            images[f"{field}_image"] = Image2DModel.parse(
                array, dims=("c", "y", "x"), c_coords=channels, transformations=transformations
            )
        elif row["element_type"] == ELEMENT_LABELS:
            labels[str(row["region_key_value"])] = Labels2DModel.parse(
                array.astype(LABEL_DTYPE), dims=("y", "x"), transformations=transformations
            )

    tables = {}
    if labels:
        if REGION_KEY not in adata.obs:
            msg = (
                f"the table in {plate} has no obs['{REGION_KEY}'] column, so its rows cannot be joined onto the "
                "label arrays. ExportForSpatialData adds it; a table from ExportToAnnData does not carry it."
            )
            raise ValueError(msg)
        named = np.asarray(adata.obs[REGION_KEY], dtype=str)
        regions = sorted(set(named.tolist()) & set(labels))
        table = adata[np.isin(named, regions)].copy()
        table.obs[REGION_KEY] = pd.Categorical(np.asarray(table.obs[REGION_KEY], dtype=str), categories=regions)
        # TableModel.parse refuses to run while ATTRS_KEY is still set.
        table.uns.pop(TableModel.ATTRS_KEY, None)
        tables["cells"] = TableModel.parse(table, region=regions, region_key=REGION_KEY, instance_key=INSTANCE_KEY)
    return SpatialData(images=images, labels=labels, tables=tables)


def is_export_plate_dir(path: Path) -> bool:
    """Whether `path` is one plate folder of an export.

    Args:
        path: The directory to test.

    Returns:
        Whether `path` is a directory holding an ``.h5ad`` table under ``tables/``.
        A SpatialData zarr store also holds a ``tables/`` directory, so the table itself has to be there.
    """
    return path.is_dir() and any((path / "tables").glob("*.h5ad"))


def export_plate_dirs(root: Path | str) -> list[Path]:
    """The plate folders of one export root, sorted by name.

    Args:
        root: The ``<prefix>_export`` directory the module wrote.

    Returns:
        Every immediate subdirectory that holds a ``tables/`` directory.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if is_export_plate_dir(path))
