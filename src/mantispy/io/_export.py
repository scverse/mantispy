from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

Layout = Literal["gallery", "zarr"]


def _require_images(reader: str) -> None:
    """Raise a clear error when the ``images`` extra, which `reader` needs, is not installed."""
    try:
        import spatialdata  # noqa: F401
    except ImportError:
        msg = f"{reader} needs spatialdata; install it with pip install 'mantispy[images]'"
        raise ImportError(msg) from None


def _detect_layout(path: Path) -> Layout | None:
    """Tell a Cell Painting Gallery source from an ``ExportForSpatialData`` export by what sits under `path`.

    A gallery source is the one with a ``workspace/`` tree, and that is checked first: a source that has been read once also holds SpatialData zarr stores, which carry a ``tables/`` directory like an export does.
    An export plate folder has its table in ``tables/`` and is looked for at `path` and one level below it.

    Returns:
        ``"gallery"``, ``"zarr"``, or ``None`` when `path` holds neither.
    """
    if (path / "workspace").is_dir():
        return "gallery"
    if is_export_plate_dir(path) or export_plate_dirs(path):
        return "zarr"
    return None


def _export_plate_dir(root: Path, plate: str | None) -> Path:
    layout = _detect_layout(root)
    if layout == "gallery":
        msg = f"{root} is a Cell Painting Gallery source; read it with mantispy.io.read_cellprofiler_gallery"
        raise ValueError(msg)
    if layout is None:
        msg = f"{root} is not an ExportForSpatialData export, which holds a table under 'tables/' per plate"
        raise ValueError(msg)
    if is_export_plate_dir(root):
        if plate is not None and root.name != plate:
            msg = f"{root} is the plate folder of {root.name!r}, not of {plate!r}"
            raise ValueError(msg)
        return root
    folders = export_plate_dirs(root)
    if plate is not None:
        if (folder := root / plate) in folders:
            return folder
        msg = f"no plate {plate!r} under {root}; found {[path.name for path in folders]}"
        raise FileNotFoundError(msg)
    if len(folders) > 1:
        msg = f"{root} holds several plates; pass plate= to choose one of {[path.name for path in folders]}"
        raise ValueError(msg)
    return folders[0]


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


def read_cellprofiler_zarr(
    path: Path | str, *, plate: str | None = None, lazy: bool = True, image_root: Path | str | None = None
) -> SpatialData:
    """Read one plate written by the ``ExportForSpatialData`` CellProfiler module into a :class:`~spatialdata.SpatialData` object.

    The plate is read from the manifest in its table's ``uns``, so nothing is reconstructed and a folder that was moved still reads.
    Images and Labels are the arrays the manifest lists as written, and the ``cells`` Table annotates the Labels.
    Each field sits in its own coordinate system, named by the field.
    Label arrays are widened to ``uint32``, since CellProfiler narrows each one to fit its object count.
    See :doc:`the tutorial </tutorials/data/images>` for what the module writes.
    Needs the ``images`` extra.

    Args:
        path: An export root, the ``<prefix>_export`` directory the module wrote, or one of its plate folders.
        plate: Plate barcode, which picks one of the plate folders of an export root.
            May be left out when the root holds one plate.
        lazy: Read arrays through dask, one HDF5 dataset per element, instead of loading them into memory.
            Each read reopens the file, so a plate of many fields holds no descriptor per element.
        image_root: Directory the manifest's ``images/`` and ``labels/`` paths are resolved against, in place of the plate folder.
            For a table kept apart from its arrays.

    Returns:
        The plate, with the Images and Labels the manifest lists as written and a ``cells`` Table annotating them.

    Raises:
        ImportError: spatialdata is not installed.
        ValueError: `path` holds no export, or holds a gallery source, or holds several plates and `plate` is left out, or the plate folder carries several tables, no element manifest, no label element or no label instance for its rows, or a table that does not satisfy the schema at object resolution.
        FileNotFoundError: `plate` is not under `path`, the plate folder has no table under ``tables/``, or the manifest names an array that is not there.

    Examples:
        Read a plate straight out of a pipeline run:

        >>> import mantispy as mt
        >>> sdata = mt.io.read_cellprofiler_zarr("run_export", plate="Plate1")  # doctest: +SKIP
    """
    _require_images("read_cellprofiler_zarr")
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Labels2DModel, TableModel
    from spatialdata.transformations import BaseTransformation, Identity

    from mantispy._core.frames import as_frame
    from mantispy._core.schema import make_image_id, validate
    from mantispy.io._profiles import stamp

    folder = _export_plate_dir(Path(path), plate)
    arrays = folder if image_root is None else Path(image_root)
    adata = ad.read_h5ad(_table_path(folder))
    elements, channels = _manifest(adata, folder)
    written = elements[elements["status"] == STATUS_OK]

    images: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    for row in written.to_dict("records"):
        field = str(row["sample_key"])
        transformations: dict[str, BaseTransformation] = {field: Identity()}
        array = _read_array(arrays / str(row["path"]), lazy=lazy)
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
        adata.obs = _link_columns(as_frame(adata.obs), adata.uns.get("spatialdata_attrs"), folder)
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
