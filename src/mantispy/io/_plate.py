from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mantispy.io._cellprofiler import export_plate_dirs, is_export_plate_dir, read_cellprofiler_export
from mantispy.io._gallery import read_gallery_plate

if TYPE_CHECKING:
    from spatialdata import SpatialData

Layout = Literal["gallery", "cellprofiler"]


def _detect_layout(path: Path) -> Layout:
    """Tell a Cell Painting Gallery source from a CellProfiler export by what sits under `path`.

    A gallery source is the one with a ``workspace/`` tree, and that is checked first: a source that has been
    read once also holds SpatialData zarr stores, which carry a ``tables/`` directory like an export does.
    An export plate folder has its table in ``tables/`` and is looked for at `path` and one level below it.
    """
    if (path / "workspace").is_dir():
        return "gallery"
    if is_export_plate_dir(path) or export_plate_dirs(path):
        return "cellprofiler"
    msg = (
        f"{path} is neither a Cell Painting Gallery source, which holds a 'workspace/' tree, nor a "
        "CellProfiler export, which holds a table under 'tables/' per plate. Pass layout= to say which it is."
    )
    raise ValueError(msg)


def _export_plate_dir(root: Path, plate: str | None) -> Path:
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


def read_plate(
    path: Path | str,
    plate: str | None = None,
    *,
    batch: str | None = None,
    layout: Layout | None = None,
    wells: Sequence[str] | None = None,
    plane: int | None = None,
    profile: str | Path | None = "normalized_feature_select_negcon_batch",
    plate_format: int = 384,
    lazy: bool = True,
) -> SpatialData:
    """Read one plate of images, segmentations and measurements into a :class:`~spatialdata.SpatialData` object.

    Two layouts are read, told apart by what sits under `path` unless `layout` says which it is.

    A **Cell Painting Gallery** source (``layout="gallery"``, needs `batch` and `plate`) gives fields of view as Images, the CellProfiler Nuclei, Cells and Cytoplasm segmentations as Labels, the wells as Shapes, and the ``wells`` and ``cells`` Tables.
    The gallery publishes one-pixel outlines, not masks, so the Labels are reconstructed and only unambiguous objects survive.
    Element names carry the plate barcode, so two plates concatenate without renaming.

    A **CellProfiler export** (``layout="cellprofiler"``) is a folder from the ``ExportForSpatialData`` module and is read from the manifest in its table's ``uns``, so nothing is reconstructed and a folder that was moved still reads.

    Where a source recorded stage coordinates and a pixel size, every element sits in three coordinate systems, ``{plate}_{well}_s{site}``, ``{plate}_{well}`` and ``{plate}``.
    Where it did not, each field sits in its own frame.
    See :doc:`the tutorial </tutorials/reading_plates>` for what each layout publishes and what is dropped.

    Args:
        path: A Cell Painting Gallery source directory, or an export root or one of its plate folders.
        plate: Plate barcode.
            Required for a gallery source; for an export root it picks one of the plate folders, and may be left out when the root holds one.
        batch: Batch name, the directory below ``images/`` and ``workspace/analysis/``.
            Gallery sources only.
        layout: Which layout `path` holds, detected from `path` when left out.
        wells: Wells to read images and labels for.
            Defaults to every well whose images are present under `path`, so a partial download reads back as itself.
            The well table always covers the whole plate.
            Gallery sources only.
        plane: Which ``Metadata_PlaneID`` to read where a source imaged a z stack.
            Required in that case: no plane is preferable to another, and picking one silently would hide the rest.
            Gallery sources only.
        profile: Variant of the well-level profile, read as ``workspace/profiles/{batch}/{plate}/{plate}_{profile}.csv.gz``.
            Sources that publish the profile under another name, ``{plate}.parquet`` among them, take a :class:`~pathlib.Path` instead.
            Pass ``None`` to leave out the well table and the well shapes.
            Gallery sources only.
        plate_format: Number of wells on the plate, used to place the wells on their nominal grid.
            Gallery sources only.
        lazy: Read arrays through dask instead of loading them into memory.
            Exports only.

    Returns:
        The plate.
        A table or element group is left out when nothing it would hold was read.

    Raises:
        ValueError: The layout cannot be told from `path`, an argument does not apply to the layout that was read, the plate mixes pixel sizes, ``load_data.csv`` names its images in an unknown way, or an image does not sit in the gallery layout.
        FileNotFoundError: ``load_data.csv``, the requested profile, or the named plate folder is missing.

    Examples:
        Read one plate of a gallery source, images and all:

        >>> import mantispy as mt
        >>> sdata = mt.io.read_plate(  # doctest: +SKIP
        ...     "cpg0000-jump-pilot/source_4",
        ...     "BR00116991",
        ...     batch="2020_11_04_CPJUMP1",
        ... )

        Read one well of a plate a source published without a profile of the usual name:

        >>> sdata = mt.io.read_plate(  # doctest: +SKIP
        ...     "cpg0016-jump/source_1",
        ...     "UL001641",
        ...     batch="Batch1_20221004",
        ...     wells=["A01"],
        ...     profile=Path("UL001641.parquet"),
        ...     plate_format=1536,
        ... )

        Read a plate straight out of a pipeline run:

        >>> sdata = mt.io.read_plate("run_export/Plate1")  # doctest: +SKIP
    """
    path = Path(path)
    layout = layout or _detect_layout(path)
    if layout == "cellprofiler":
        if unsupported := [
            name for name, value in (("batch", batch), ("wells", wells), ("plane", plane)) if value is not None
        ]:
            msg = f"{unsupported} do not apply to a CellProfiler export"
            raise ValueError(msg)
        return read_cellprofiler_export(_export_plate_dir(path, plate), lazy=lazy)

    if batch is None or plate is None:
        msg = "a Cell Painting Gallery source needs both a plate and a batch"
        raise ValueError(msg)
    return read_gallery_plate(
        path,
        batch,
        plate,
        wells=wells,
        plane=plane,
        profile=profile,
        plate_format=plate_format,
    )
