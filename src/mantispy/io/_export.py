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
REGION_KEY = "region"
INSTANCE_KEY = "Metadata_ObjectNumber"

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


def _link_columns(obs: pd.DataFrame, attrs: Any, plate: Path) -> pd.DataFrame:
    """Bring the table's link onto the label instances to mantispy's ``region`` + ``Metadata_ObjectNumber``.

    ``ExportForSpatialData`` names its own link in ``uns["spatialdata_attrs"]``: ``region_key`` holds the label
    element of each row and ``label_id`` its instance, while ``region`` holds the object set (``"Cells"``). That
    layout is read as written, the object set moving to ``Metadata_ObjectType``; a table already in mantispy's
    layout passes through. The instance numbers are never invented from row order, since a table whose rows are
    not in label order, or that dropped an object, would then link rows to the wrong label instances.
    """
    attrs = attrs if isinstance(attrs, dict) else {}

    def _pick(declared: Any, candidates: tuple[str, ...]) -> str:
        # The table's own spatialdata attrs win when they name a column it has; otherwise the known layouts.
        if declared is not None and str(declared) in obs:
            return str(declared)
        return next((name for name in candidates if name in obs), candidates[-1])

    region_key = _pick(attrs.get("region_key"), ("region_key", REGION_KEY))
    # CellProfiler's ObjectNumber is the label value itself, so a raw table links through it as well.
    instance_key = _pick(attrs.get("instance_key"), ("label_id", "ObjectNumber", INSTANCE_KEY))
    if region_key not in obs:
        msg = (
            f"the table in {plate} has no obs['{region_key}'] column, so its rows cannot be joined onto the "
            "label arrays. ExportForSpatialData adds it; a table from ExportToAnnData does not carry it."
        )
        raise ValueError(msg)
    if instance_key not in obs:
        msg = (
            f"the table in {plate} has no obs['{instance_key}'] column naming each row's label instance, so its "
            "rows cannot be linked to the label arrays; numbering them by row order would link the wrong objects."
        )
        raise ValueError(msg)
    obs = obs.copy()
    if region_key != REGION_KEY:
        if REGION_KEY in obs and "Metadata_ObjectType" not in obs:
            obs["Metadata_ObjectType"] = np.asarray(obs[REGION_KEY], dtype=str)
        obs[REGION_KEY] = np.asarray(obs[region_key], dtype=str)
    if instance_key != INSTANCE_KEY:
        instances = pd.to_numeric(obs[instance_key]).to_numpy()
        if INSTANCE_KEY in obs and not np.array_equal(pd.to_numeric(obs[INSTANCE_KEY]).to_numpy(), instances):
            msg = (
                f"the table in {plate} carries both obs['{instance_key}'] and obs['{INSTANCE_KEY}'], and they disagree"
            )
            raise ValueError(msg)
        obs[INSTANCE_KEY] = instances
    if "Metadata_Plate" not in obs:
        # The module writes one folder per plate, named by it, so a table without the tag still has its plate.
        obs["Metadata_Plate"] = plate.name
    if "Metadata_Well" not in obs:
        msg = (
            f"the table in {plate} carries no Metadata_Well, which every object needs under the schema. Extract "
            "the well into the image metadata (the Metadata or NamesAndTypes module) so ExportForSpatialData writes it."
        )
        raise ValueError(msg)
    if "ImageNumber" in obs and "Metadata_ImageNumber" not in obs:
        # The field each row came from, so the image id separates fields that share plate, well and site tags.
        obs["Metadata_ImageNumber"] = obs["ImageNumber"].to_numpy()
    return obs


class _LazyDataset:
    """One HDF5 dataset that opens its file for each read instead of holding it open.

    A dask array over a live ``h5py`` dataset keeps one file descriptor for as long as the array is referenced.
    """

    def __init__(self, path: Path) -> None:
        import h5py

        self.path = path
        with h5py.File(path, "r") as handle:
            dataset = handle[DATASET]
            self.shape: tuple[int, ...] = dataset.shape
            self.dtype = dataset.dtype
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
    # lock: HDF5 is not thread-safe; name: the path identifies the contents, tokenizing the reader object would not.
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
        ValueError: Several tables under ``path/tables``, or the table carries no element manifest, or it names no label element or no label instance for its rows, or it does not satisfy the schema at object resolution.
    """
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Labels2DModel, TableModel
    from spatialdata.transformations import Identity

    from mantispy._core.frames import as_frame
    from mantispy._core.schema import make_image_id, validate
    from mantispy.io._profiles import stamp

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
        adata.obs = _link_columns(as_frame(adata.obs), adata.uns.get("spatialdata_attrs"), plate)
        if "Metadata_ImageID" not in adata.obs:
            # The id minted from plate, well and site tags is coarser than the field when the export carries no
            # site or image number; the manifest knows each label element's field, so it refines the id then.
            obs = as_frame(adata.obs)
            minted = make_image_id(obs)
            if pd.DataFrame({"image": minted, "number": obs[INSTANCE_KEY].to_numpy()}).duplicated().any():
                field_of = dict(
                    zip(written["region_key_value"].astype(str), written["sample_key"].astype(str), strict=False)
                )
                minted = minted + "|" + obs[REGION_KEY].astype(str).map(field_of).fillna("").to_numpy()
            adata.obs["Metadata_ImageID"] = minted
        named = np.asarray(adata.obs[REGION_KEY], dtype=str)
        regions = sorted(set(named.tolist()) & set(labels))
        table = adata[np.isin(named, regions)].copy()
        table.obs[REGION_KEY] = pd.Categorical(np.asarray(table.obs[REGION_KEY], dtype=str), categories=regions)
        # The export carries the identity columns (spec §18); stamp mints Metadata_ImageID from them and
        # tags the measurements. The table must satisfy the schema before it becomes a SpatialData table (§19).
        stamp(table, resolution="object", feature_kind="measurement")
        validate(table, raise_on_error=True)
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
