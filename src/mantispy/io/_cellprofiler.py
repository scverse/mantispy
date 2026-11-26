"""Read a CellProfiler export into one table per object, or into one profile.

ExportToSpreadsheet writes an ``Image`` table beside one table per object, all behind the file-name prefix a run was configured with (``MyExpt_Image.csv``, ``MyExpt_Cells.csv``); ExportToDatabase writes the same tables into an SQLite database.
Inside an object table the columns do not carry the object's name, so they are prefixed with it.
Each object table keeps its links to other objects as ``Metadata_<Parent>ObjectNumber``, and records in ``uns["mantispy"]["links"]`` what each link means.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.features import parse_feature_names
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.plate import normalize_wells
from mantispy._core.provenance import record_params
from mantispy._core.schema import make_image_id
from mantispy.io._cp_export import SQLITE_SUFFIXES, RawExport, export_prefix, extract_csv, extract_sqlite
from mantispy.io._cp_links import Omissions, carriers, classify, unexplained_message

__all__ = ["export_prefix", "read_cellprofiler", "read_cellprofiler_csv", "read_cellprofiler_sqlite"]

_KEYS = ("ImageNumber", "ObjectNumber")
#: The columns that say where each image was loaded from; they belong to the image table, not to the objects.
FILE_PREFIXES = ("FileName_", "PathName_", "URL_")


def _is_file_column(column: str) -> bool:
    """Whether `column` locates an image file, as CellProfiler names it or as LoadData read it back as metadata."""
    return column.removeprefix("Metadata_Image_").startswith(FILE_PREFIXES)


#: Feature groups that identify an object rather than measure it, kept in ``obs`` of its table even where numeric.
IDENTITY_GROUPS = ("Barcode",)
#: Identity columns the Metadata module writes even when it extracted no value for them.
_IDENTITY = ("Metadata_Plate", "Metadata_Well", "Metadata_Site")


def infer_channels(image: pd.DataFrame, features: Sequence[str] = ()) -> list[str]:
    """Channel names, taken from the features and otherwise from the ``FileName_<channel>`` columns.

    CellProfiler writes one ``Intensity_MeanIntensity_<channel>`` measurement per channel it measured, under the name the features carry.
    The file names can differ from it: a pipeline loads ``OrigDNA``, corrects it with ``IllumDNA`` and saves ``CellOutlines``, and measures the corrected image as ``DNA``.

    Args:
        image: The ``Image`` table of an export.
        features: The feature columns of the export.

    Returns:
        The channels the features were measured in, sorted.
        Without an intensity feature, the names the ``FileName_`` or ``Image_FileName_`` columns carry, with the ``Orig`` and ``Illum`` prefixes stripped and saved outlines left out.
        An empty list when neither is present, in which case the parser infers the channels from the feature names instead.
    """
    measured = {
        name.split("_Intensity_MeanIntensity_", 1)[1] for name in features if "_Intensity_MeanIntensity_" in name
    }
    if measured:
        return sorted(measured)
    names = {
        column.removeprefix("Image_").removeprefix("FileName_")
        for column in image.columns
        if column.removeprefix("Image_").startswith("FileName_")
    }
    return sorted({name.removeprefix("Orig").removeprefix("Illum") for name in names if not name.endswith("Outlines")})


def _prefix(frame: pd.DataFrame, obj: str) -> pd.DataFrame:
    """Prefix an object's columns with its name, leaving the join keys and metadata alone."""
    keep = (f"{obj}_", "Metadata_")
    return frame.rename(columns={c: c if c in _KEYS or c.startswith(keep) else f"{obj}_{c}" for c in frame.columns})


def _imported_image_columns(image: pd.DataFrame) -> pd.DataFrame:
    """`image` with the ``Image_`` columns of an earlier export that LoadData read back as metadata under their own names.

    The Advanced tutorial loads the ``Image.csv`` of an earlier run, so its ``Image_Metadata_BROAD_ID`` and ``Image_FileName_OrigER`` come back as ``Metadata_Image_Metadata_BROAD_ID`` and ``Metadata_Image_FileName_OrigER``.
    A copy of a column the image table already holds with the same values is dropped, and one with other values is kept as it is.
    """
    renames: dict[str, str] = {}
    copies: list[str] = []
    for column in (c for c in image.columns if c.startswith("Metadata_Image_")):
        name = column.removeprefix("Metadata_Image_")
        if name not in image.columns and name not in renames.values():
            renames[column] = name
        elif name in image.columns and image[name].astype(str).equals(image[column].astype(str)):
            copies.append(column)
    return image.drop(columns=copies).rename(columns=renames)


def _image_metadata(image: pd.DataFrame) -> pd.DataFrame:
    """``ImageNumber`` and the image metadata, ``Image_Metadata_`` read as ``Metadata_``."""
    renames = {
        c: "Metadata_" + c.removeprefix("Image_Metadata_") for c in image.columns if c.startswith("Image_Metadata_")
    }
    renames |= {c: c for c in image.columns if c.startswith("Metadata_") and not _is_file_column(c)}
    return image[["ImageNumber", *renames]].rename(columns=renames)


def _image_ids(metadata: pd.DataFrame, run: str | None) -> np.ndarray:
    """``Metadata_ImageID`` of each image, with `run` in it when no plate, well or site tells the runs of several exports apart."""
    ids = make_image_id(metadata.rename(columns={"ImageNumber": "Metadata_ImageNumber"}))
    if run is not None and not any(column in metadata for column in _IDENTITY):
        ids = np.array([f"{run}|{image_id}" for image_id in ids], dtype=object)
    return ids


def _image_table(image: pd.DataFrame, metadata: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Per-image quality measurements, object counts and file locations, keyed by ``Metadata_ImageID`` and carrying plate/well/site.

    The plate and well columns let :func:`~mantispy.pp.image_qc` threshold per plate instead of pooling every plate together.
    Every image is a row, including those in which no object was found.
    """
    measured = [c for c in image.columns if "ImageQuality" in c or c.startswith("Count_") or _is_file_column(c)]
    table = image[measured].copy()
    # A location ExportToDatabase left empty reads as None, which h5ad cannot write in an object column.
    files = [c for c in measured if _is_file_column(c)]
    table[files] = table[files].astype("str")
    for column in _IDENTITY:
        if column in metadata:
            values = metadata[column].to_numpy()
            table[column] = normalize_wells(pd.Series(values)).to_numpy() if column == "Metadata_Well" else values
    table.index = pd.Index(ids.astype(str), name="Metadata_ImageID")
    return table


def _table(
    frame: pd.DataFrame,
    obj: str,
    metadata: pd.DataFrame,
    ids: pd.Series,
    *,
    measured: bool,
    channels: list[str] | None,
    sentinels: float | Collection[float] | None,
) -> AnnData:
    """One object table as an object-resolution AnnData, its links and centroid in ``obs``."""
    from mantispy.io._profiles import from_dataframe

    frame = frame.rename(
        columns={
            c: f"Metadata_{c.removeprefix(f'{obj}_Parent_')}ObjectNumber"
            for c in frame.columns
            if c.startswith(f"{obj}_Parent_")
        }
    )
    # ExportToSpreadsheet can also copy the image metadata into the object tables, and merging both copies would suffix them _x/_y.
    if shared := [c for c in metadata.columns if c != "ImageNumber" and c in frame.columns]:
        get_logger().debug("%s are on both the %s and Image tables; keeping the Image value", shared, obj)
        frame = frame.drop(columns=shared)
    frame = frame.merge(metadata, on="ImageNumber", how="left", validate="m:1")
    added = {"Metadata_ImageID": frame["ImageNumber"].map(ids).to_numpy(), "Metadata_ObjectType": obj}
    # CellProfiler 4 writes centroids under AreaShape, older versions under Location.
    for axis in ("X", "Y"):
        for source in (f"{obj}_Location_Center_{axis}", f"{obj}_AreaShape_Center_{axis}"):
            if source in frame.columns:
                added[f"Metadata_Center_{axis}"] = frame[source].to_numpy()
                break
    # Added in one concat: inserting one column at a time into a wide table trips pandas' fragmentation warning.
    frame = pd.concat([frame, pd.DataFrame(added, index=frame.index)], axis=1)
    frame = frame.rename(columns={key: f"Metadata_{key}" for key in _KEYS})
    numeric = frame.select_dtypes("number").columns
    identity = [c for c in numeric if c.startswith(tuple(f"{obj}_{group}_" for group in IDENTITY_GROUPS))]
    # A table of locations only, as an imported mask gives, keeps them in obs and has no feature.
    kept = identity if measured else [c for c in numeric if not c.startswith("Metadata_")]
    return from_dataframe(
        frame, metadata_columns=kept, channels=channels, sentinels=sentinels, objects=(obj,), resolution="object"
    )


def _finish(
    raw: RawExport,
    *,
    relations: Mapping[str, Mapping[str, str]] | None = None,
    channels: Sequence[str] | None = None,
    sentinels: float | Collection[float] | None = None,
    root: str | None = None,
    omissions: Omissions | None = None,
    run: str | None = None,
) -> dict[str, AnnData]:
    """Turn the tables of an export into one stamped AnnData per object.

    A table with a link nothing classifies is left out and added to `omissions`, or logged here without it.

    Raises:
        ValueError: The export measured objects in 3D, a link of `root` itself cannot be classified, or a link points to an object number the parent table lacks.
    """
    log = get_logger()
    if raw.renamed_image_number:
        log.info("kept the imported Metadata_ImageNumber as Metadata_ImportedImageNumber")
    image = _imported_image_columns(raw.image)
    if empty := [column for column in _IDENTITY if column in image and image[column].isna().all()]:
        log.info("%s are empty in every image and were left out", empty)
        image = image.drop(columns=empty)
    for name, frame in raw.objects.items():
        if "Location_Center_Z" in frame and (frame["Location_Center_Z"].fillna(0) != 0).any():
            raise ValueError(f"{name} was measured in 3D (Location_Center_Z is not 0); mantispy reads 2D exports only")

    # ExportToSpreadsheet can copy each image's file names into the object tables; the image table holds them.
    tables = {
        name: frame.drop(columns=[c for c in frame.columns if c.startswith(FILE_PREFIXES) and c in image.columns])
        for name, frame in raw.objects.items()
        if len(frame)
    }
    if empty_tables := sorted(set(raw.objects) - set(tables)):
        log.debug("left out %s, which hold no object", empty_tables)
    unexported: dict[str, list[str]] = {}
    for name, frame in tables.items():
        if dangling := [
            c for c in frame.columns if c.startswith("Parent_") and c.removeprefix("Parent_") not in tables
        ]:
            unexported[name] = [c.removeprefix("Parent_") for c in dangling]
            tables[name] = frame.drop(columns=dangling)
    if unexported:
        log.info("left out the links to objects that have no table: %s", unexported)

    prefixed = {name: _prefix(frame, name) for name, frame in tables.items()}
    columns = [c for frame in prefixed.values() for c in frame.columns]
    vocabulary = list(channels) if channels is not None else (infer_channels(image, columns) or None)
    measured = {
        name: bool(
            parse_feature_names(list(frame.select_dtypes("number").columns), channels=vocabulary)["is_feature"].any()
        )
        for name, frame in prefixed.items()
    }
    links, unexplained = classify(tables, measured, raw.pipeline, image, relations)
    if unexplained:
        if any(child == root for child, _ in unexplained):
            raise ValueError(unexplained_message(unexplained, raw.pipeline is not None))
        report = Omissions(root) if omissions is None else omissions
        report.unexplained |= unexplained
        report.pipeline = raw.pipeline is not None
        for child in {child for child, _ in unexplained}:
            del prefixed[child], links[child]
        if omissions is None:
            report.log()
    # A copy that only carries a link adds nothing a profile can use, so it is not returned.
    hidden = set(carriers(links, measured))

    metadata = _image_metadata(image)
    ids = _image_ids(metadata, run)
    image_table = _image_table(image, metadata, ids)
    by_number = pd.Series(ids, index=metadata["ImageNumber"].to_numpy())
    result = {}
    for name, frame in prefixed.items():
        if name in hidden:
            continue
        adata = _table(
            frame, name, metadata, by_number, measured=measured[name], channels=vocabulary, sentinels=sentinels
        )
        adata.uns["mantispy"]["links"] = dict(links[name])
        adata.uns["mantispy"]["image_table"] = image_table.copy()
        result[name] = adata
    return result


def _kind(path: Path) -> str:
    """``"csv"`` or ``"sqlite"``: which export `path` holds.

    Raises:
        ValueError: `path` is an ExportForSpatialData export or a Cell Painting Gallery source, which read_cellprofiler reads only alone.
        FileNotFoundError: `path` is no export.
    """
    from mantispy.io._export import _detect_layout

    if export_prefix(path) is not None:
        return "csv"
    if path.is_file():
        with path.open("rb") as handle:
            if handle.read(16) == b"SQLite format 3\x00":
                return "sqlite"
    if path.is_dir() and len([p for p in path.iterdir() if p.suffix in SQLITE_SUFFIXES]) == 1:
        return "sqlite"
    if path.is_dir() and (layout := _detect_layout(path)) is not None:
        alone = "on its own, without out=," if layout == "zarr" else "on its own,"
        raise ValueError(f"{path} is {_LAYOUTS[layout]}, which read_cellprofiler reads {alone} not in a list")
    found = sorted(p.name for p in path.iterdir())[:10] if path.is_dir() else []
    raise FileNotFoundError(
        f"{path} is not a CellProfiler export: no <prefix>Image.csv or Image.txt, and no SQLite database"
        + (f"; it holds {found}" if found else "")
    )


def _database(path: Path) -> Path:
    if path.is_file():
        return path
    (database,) = (p for p in path.iterdir() if p.suffix in SQLITE_SUFFIXES)
    return database


def read_cellprofiler_csv(
    path: str | Path,
    *,
    files: Mapping[str, str | Path] | None = None,
    relations: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, AnnData]:
    """Read a directory an ``ExportToSpreadsheet`` run wrote into one table per object.

    The objects are those the ``Image`` table counts (``Count_<Object>``), so the relationship and experiment tables are not read as objects.
    A file holding every object in one table, which ExportToSpreadsheet writes with two header rows, is split into its objects.
    Tab-delimited ``.txt`` exports read like ``.csv`` ones.

    Each table is one row per object, stamped at object resolution, with the object's measurements in ``X`` and, in ``obs``, ``Metadata_ImageID``, ``Metadata_ObjectNumber``, the centroid as ``Metadata_Center_X``/``Metadata_Center_Y`` and the number of each parent object as ``Metadata_<Parent>ObjectNumber``.
    No row is left out: an object without a parent keeps a parent number of 0.
    ``uns["mantispy"]["links"]`` says what each link means, ``"compartment"``, ``"subset"``, ``"aggregate"`` or ``"carrier"``, as the module that made the object decides it: the pipeline text the export stores is read for it.
    Without the pipeline, only RelateObjects' columns and the cell, cytoplasm and nucleus compartments with equal counts are read by name.
    A table with a link nothing explains is left out, with one warning naming the link and the `relations` that reads it as the tables suggest.
    ``uns["mantispy"]["image_table"]`` holds the image quality measurements, the object counts and the ``FileName_``, ``PathName_`` and ``URL_`` columns of every image, so the objects' ``obs`` does not repeat the file names.
    The image metadata is in ``obs`` as ``Metadata_<name>``, including the ``Image_Metadata_<name>`` of an earlier run that LoadData read back.
    A table that holds only locations, as an imported mask gives, has no feature and keeps its locations in ``obs``.
    A copy that only carries a link, such as the objects ResizeObjects writes or the children RelateObjects saves, is not returned; the link to it stays in ``obs`` and ``links`` of the tables that have one.
    An export of image measurements only, as a quality-control pipeline writes, gives one ``"Image"`` table of one row per image.

    Args:
        path: The export directory.
        files: Object tables under custom file names, by object name, for example ``{"Spots": "spots.csv"}``.
        relations: What a link means, by child and then by parent, for example ``{"Nucleoli": {"Nuclei": "aggregate"}}``.
            Needed for links the pipeline does not explain, and overrides it.

    Returns:
        One AnnData per object, keyed by the object's name.

    Raises:
        FileNotFoundError: `path` holds no ``Image.csv`` or ``Image.txt``.
        ValueError: A table repeats a column with different values, or the export was measured in 3D.

    Examples:
        >>> import mantispy as mt
        >>> objects = mt.io.read_cellprofiler_csv("analysis/")  # doctest: +SKIP
        >>> objects["Nucleoli"].uns["mantispy"]["links"]  # doctest: +SKIP
        {'Nuclei': 'aggregate'}
    """
    return _read_tables(Path(path), "csv", files=files, relations=relations)


def read_cellprofiler_sqlite(
    path: str | Path, *, relations: Mapping[str, Mapping[str, str]] | None = None
) -> dict[str, AnnData]:
    """Read an SQLite database an ``ExportToDatabase`` run wrote into one table per object.

    The run may have written one table per object, or one ``Per_Object`` table that lines up every object by its number, which is split into the objects; a ``Per_Object`` view is not read, because it joins the objects on their number and loses those without a partner.
    A table-name prefix is found from the ``Per_Image`` table.
    Column names ExportToDatabase shortened to fit its limit are restored where exactly one full name shortens to them.
    The tables are then the same as :func:`read_cellprofiler_csv` gives for the same run.

    Args:
        path: The database, or a directory holding one.
        relations: What a link means, by child and then by parent, as for :func:`read_cellprofiler_csv`.

    Returns:
        One AnnData per object, keyed by the object's name.

    Raises:
        FileNotFoundError: The database holds no ``Per_Image`` table.
        ValueError: The export was measured in 3D.
    """
    return _read_tables(Path(path), "sqlite", relations=relations)


def _read_tables(
    path: Path,
    kind: str,
    *,
    files: Mapping[str, str | Path] | None = None,
    relations: Mapping[str, Mapping[str, str]] | None = None,
    channels: Sequence[str] | None = None,
    sentinels: float | Collection[float] | None = None,
    root: str | None = None,
    omissions: Omissions | None = None,
    run: str | None = None,
) -> dict[str, AnnData]:
    raw = extract_csv(path, files) if kind == "csv" else extract_sqlite(_database(path))
    if not raw.objects:
        return {"Image": _image_only(raw, channels, run)}
    return _finish(
        raw, relations=relations, channels=channels, sentinels=sentinels, root=root, omissions=omissions, run=run
    )


def _image_only(raw: RawExport, channels: Sequence[str] | None, run: str | None = None) -> AnnData:
    """An export of image measurements only, as one row per image stamped as ``tl.aggregate(by="Metadata_ImageID")`` stamps it."""
    from mantispy.io._profiles import from_dataframe

    image = _imported_image_columns(raw.image)
    if empty := [column for column in _IDENTITY if column in image and image[column].isna().all()]:
        image = image.drop(columns=empty)
    metadata = _image_metadata(image)
    ids = _image_ids(metadata, run)
    dropped = [c for c in image.columns if _is_file_column(c) or (c in metadata and c != "ImageNumber")]
    frame = _prefix(image.drop(columns=dropped), "Image")
    identity = metadata.drop(columns="ImageNumber").assign(Metadata_ImageID=ids)
    frame = pd.concat([frame, identity], axis=1).rename(columns={"ImageNumber": "Metadata_ImageNumber"})
    numeric = [c for c in frame.select_dtypes("number").columns if not c.startswith("Metadata_")]
    vocabulary = list(channels) if channels is not None else (infer_channels(image) or None)
    features = parse_feature_names(numeric, channels=vocabulary)["is_feature"]
    adata = from_dataframe(
        frame,
        metadata_columns=numeric if not features.any() else (),
        channels=vocabulary,
        objects=("Image",),
        resolution="aggregate",
        grouped_by=["Metadata_ImageID"],
    )
    adata.obs_names = as_frame(adata.obs)["Metadata_ImageID"].astype(str).to_list()
    adata.uns["mantispy"]["links"] = {}
    adata.uns["mantispy"]["image_table"] = _image_table(image, metadata, ids)
    return adata


def _warn_featureless(root: str | None, kind: str | None) -> None:
    """Warn that a profile holds no feature, and say where the objects are."""
    if root is None:
        get_logger().warning("the export holds image metadata and no measurement; the metadata is in obs")
    else:
        get_logger().warning(
            "the %s profile holds no feature, as no object joined to it was measured; "
            "mt.io.read_cellprofiler_%s returns every object with its locations in obs",
            root,
            kind or "csv (or _sqlite)",
        )


def _profile(
    tables: dict[str, AnnData],
    *,
    root: str,
    func: Sequence[str] | None,
    strict: bool,
    platemap: str | Path | pd.DataFrame | None,
    index_columns: Sequence[str] | None,
    score: str | None = None,
    omissions: Omissions | None = None,
) -> AnnData:
    """One export's tables as one profile: the image-only table without bookkeeping, or every root with its objects joined and rolled up."""
    from mantispy.tl._aggregate_objects import combine, profile_obs

    if list(tables) == ["Image"]:
        adata = tables["Image"]
        adata.obs = profile_obs(as_frame(adata.obs), "Image")
    else:
        adata = combine(tables, root, func=func, strict=strict, omissions=omissions, score=score)
    return _index(adata, platemap=platemap, index_columns=index_columns)


def _index(
    adata: AnnData, *, platemap: str | Path | pd.DataFrame | None, index_columns: Sequence[str] | None
) -> AnnData:
    """Join `platemap` onto ``obs`` and index it by `index_columns`."""
    from mantispy.io._profiles import _join_platemap

    obs = as_frame(adata.obs)
    if platemap is not None:
        obs = _join_platemap(obs, platemap)
    if index_columns:
        if missing := [column for column in index_columns if column not in obs.columns]:
            raise KeyError(f"index columns not in metadata: {missing}")
        index = obs[list(index_columns)].astype(str).agg(":".join, axis=1)
        if index.duplicated().any():
            raise ValueError(f"{list(index_columns)} do not identify observations uniquely")
        obs.index = pd.Index(index.to_numpy())
    adata.obs = obs
    return adata


_LAYOUTS = {"zarr": "an ExportForSpatialData export", "gallery": "a Cell Painting Gallery source"}


def _gallery_plate(source: Path) -> tuple[str, str]:
    """The one batch and plate of a gallery source, from its ``workspace/load_data_csv`` tree."""
    found = sorted((p.parent.name, p.name) for p in (source / "workspace/load_data_csv").glob("*/*") if p.is_dir())
    if len(found) != 1:
        raise ValueError(
            f"{source} holds {len(found)} plates ({[f'{b}/{p}' for b, p in found][:10]}); "
            "read one with mt.io.read_cellprofiler_gallery(source, batch=..., plate=...)"
        )
    return found[0]


def _gallery_sites(source: Path) -> list[Path]:
    """The per-site ExportToSpreadsheet directories of the one plate of a gallery source.

    Raises:
        FileNotFoundError: The plate has no analysis directory, or none of its sites holds an ``Image.csv``.
    """
    batch, plate = _gallery_plate(source)
    analysis = source / "workspace/analysis" / batch / plate / "analysis"
    if not analysis.is_dir() or not (sites := sorted(p for p in analysis.iterdir() if p.is_dir())):
        raise FileNotFoundError(f"{source} holds no analysis of plate {plate} under {analysis}")
    if not any(export_prefix(site) is not None for site in sites):
        raise FileNotFoundError(
            f"the analysis directories of plate {plate} in {source} hold no Image.csv, so they are not whole "
            "ExportToSpreadsheet exports; read the cells from the images and outlines with "
            "mt.io.read_cellprofiler_gallery"
        )
    return sites


def _spatial_profile(path: Path, *, root: str, func: Sequence[str] | None) -> AnnData:
    """The cell table of an ExportForSpatialData export, as the profile."""
    from mantispy.io._export import export_plate_dirs, is_export_plate_dir, read_cellprofiler_zarr

    if root != "Cells":
        raise ValueError(f"{path} is {_LAYOUTS['zarr']}, which holds one table, of Cells; root={root!r} has none")
    if func is not None:
        raise ValueError(
            f"{path} is {_LAYOUTS['zarr']}, whose Cells table is already joined; aggregate= needs one table per object"
        )
    if not is_export_plate_dir(path) and len(plates := export_plate_dirs(path)) > 1:
        raise ValueError(f"{path} holds several plates; pass one plate folder, one of {[p.name for p in plates]}")
    sdata = read_cellprofiler_zarr(path)
    if "cells" not in sdata.tables:
        raise FileNotFoundError(f"no table for {root!r} in {path}: it holds no labelled cells")
    adata = sdata.tables["cells"].copy()
    adata.uns.pop("spatialdata_attrs", None)
    return adata


def read_cellprofiler(
    path: str | Path | Sequence[str | Path],
    *,
    root: str = "Cells",
    aggregate: str | Sequence[str] | None = None,
    platemap: str | Path | pd.DataFrame | None = None,
    index_columns: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    sentinels: float | Collection[float] | None = None,
    relations: Mapping[str, Mapping[str, str]] | None = None,
    files: Mapping[str, str | Path] | None = None,
    strict: bool = False,
    score: str | None = None,
    skip_non_exports: bool = False,
    out: str | Path | None = None,
) -> AnnData:
    """Read a CellProfiler export, or several, into one profile, one row per `root` object.

    `path` is a directory an ``ExportToSpreadsheet`` run wrote or an SQLite database an ``ExportToDatabase`` run wrote; :func:`read_cellprofiler_csv` and :func:`read_cellprofiler_sqlite` describe how each is read.
    It can also be a Cell Painting Gallery source of one plate, whose per-site ``ExportToSpreadsheet`` directories under ``workspace/analysis`` are read as a list of exports, without loading an image.
    Or it can be one plate of an ``ExportForSpatialData`` run, which needs the ``images`` extra: the profile is then the cell table :func:`read_cellprofiler_zarr` reads, already joined, so `root` is ``"Cells"`` and `aggregate` is not taken.
    The compartments of `root` (the objects one to one with it, such as the cytoplasm and nucleus of a cell) and its ancestors are joined onto it.
    Objects of which a root has several, such as the nucleoli of a nucleus, are rolled into it by :func:`~mantispy.tl.aggregate_objects` when `aggregate` is given, and otherwise left out with one warning naming them.
    Objects not linked to `root` at all, such as the objects of an imported mask, are left out with one warning; their counts per image stay in ``uns["mantispy"]["image_table"]``.
    So is an object with a link nothing explains, unless it is `root`, whose own links must be explained.
    A profile without a feature, as an export of locations or of image metadata only gives, is returned with a warning.

    Several exports, such as the one directory per site a pooled screen writes, are read one at a time and stacked.
    Each gives the same columns: a column one export lacks, such as the roll-up of an object it found none of, is a count of 0 there and missing otherwise.
    An export of image measurements only adds its images to the image table and no row.
    Without plate, well or site metadata the directory name enters ``Metadata_ImageID``, so the images of different runs never share one; two exports holding the same image raise.

    Args:
        path: The export directory or database, or a list of them.
        root: The object one row of the profile is.
        aggregate: The statistics to roll the objects a root has several of into it with, from ``"count"``, ``"mean"``, ``"median"`` and ``"std"``.
        platemap: A table, or a path to one, with a ``Metadata_Well`` column and optionally ``Metadata_Plate``, left-joined onto ``obs``.
        index_columns: ``obs`` columns joined with ``:`` into the observation index.
            Without them the index is the row number, after the directory name and ``:`` when `path` is a list.
        channels: Channel vocabulary passed to the feature-name parser.
            Read from the image table's ``FileName_`` columns and the intensity features when omitted.
        sentinels: Values in the feature matrix that stand for missing, replaced with NaN before any statistic is taken.
        relations: What a link means, by child and then by parent, as for :func:`read_cellprofiler_csv`.
        files: Object tables under custom file names, by object name, for a spreadsheet export.
        strict: Raise when a compartment row has no root or a root has no compartment, instead of warning with the count.
            A compartment row without a root is otherwise left out, and a root without a compartment gets NaN for its features.
        score: With `aggregate`, a numeric ``obs`` column of a child, such as ``"Foci_Barcode_MatchedTo_Score"``, averaged over the children that hold the most common value, as :func:`~mantispy.tl.aggregate_objects` does.
        skip_non_exports: Skip the paths of a list that are not CellProfiler exports, with one log line naming them, instead of raising.
        out: An ``.h5ad`` or ``.zarr`` path to write the profile to, one export at a time, for a screen that does not fit in memory.
            The profile is then read back from it lazily, as :func:`~mantispy.io.read` with ``lazy=True`` reads it.
            Needs dask, from ``mantispy[lazy]``.

    Returns:
        An object-resolution :class:`~anndata.AnnData` with `root`'s centroid as ``Metadata_Center_X``/``Metadata_Center_Y``, the number of each joined object as ``Metadata_<Object>ObjectNumber``, the image table under ``uns["mantispy"]["image_table"]`` and this call's parameters.
        CellProfiler's bookkeeping stays in the per-object tables and out of ``obs``: where in a file each image sat (``Metadata_FileLocation``, ``Metadata_Frame``, ``Metadata_Series``, ``Metadata_ChannelNumber``) and each object's own child counts, number, parent links and Z centroid.

    Raises:
        FileNotFoundError: `path` is not a CellProfiler export, a path of a list is not one and `skip_non_exports` is not set, or a gallery source holds no analysis of its plate.
        ImportError: `out` is given and dask is not installed, or `path` is an ``ExportForSpatialData`` export and spatialdata is not installed.
        ValueError: `path` is an ``ExportForSpatialData`` export given with `root` other than ``"Cells"``, with `aggregate`, with `out` or in a list, a gallery source of several plates or in a list, there is no table for `root`, a link of `root` cannot be classified, a compartment is not one to one with `root`, `strict` is set and an object has no partner, `aggregate` names an unknown statistic, or two exports hold the same image.

    Examples:
        >>> import mantispy as mt
        >>> cells = mt.io.read_cellprofiler("analysis/", platemap="platemap.csv")  # doctest: +SKIP
        >>> cells = mt.io.read_cellprofiler("analysis/", aggregate=("count", "mean"))  # doctest: +SKIP
        >>> sites = sorted(Path("analysis").glob("CP257A-Well1-*"))  # doctest: +SKIP
        >>> cells = mt.io.read_cellprofiler(
        ...     sites, aggregate="mean", skip_non_exports=True, out="well1.h5ad"
        ... )  # doctest: +SKIP
    """
    from mantispy.io._cp_directories import read_directories
    from mantispy.io._export import _detect_layout
    from mantispy.tl._aggregate_objects import statistics

    func = statistics(aggregate) if aggregate is not None else None
    if score is not None and func is None:
        raise ValueError("score= summarises the children rolled up with aggregate=; pass aggregate= as well")
    params = {"root": root, "aggregate": func, "channels": channels, "strict": strict, "score": score}
    read: dict[str, Any] = {"files": files, "relations": relations, "channels": channels, "sentinels": sentinels}
    profile: dict[str, Any] = {
        "root": root,
        "func": func,
        "strict": strict,
        "score": score,
        "platemap": platemap,
        "index_columns": index_columns,
    }
    if isinstance(path, str | Path) and Path(path).is_dir():
        layout = _detect_layout(Path(path))
        if layout == "gallery":
            path = _gallery_sites(Path(path))
        elif layout == "zarr" and out is None:
            adata = _spatial_profile(Path(path), root=root, func=func)
            adata = _index(adata, platemap=platemap, index_columns=index_columns)
            record_params(adata, "read_cellprofiler", {"path": str(path), **params})
            return adata
    if isinstance(path, str | Path) and out is None:
        path = Path(path)
        kind = _kind(path)
        tables = _read_tables(path, kind, root=root, **read)
        adata = _profile(tables, **profile)
        if not adata.n_vars:
            _warn_featureless(root if list(tables) != ["Image"] else None, kind)
        record_params(adata, "read_cellprofiler", {"path": str(path), **params})
        return adata
    paths = [Path(path)] if isinstance(path, str | Path) else [Path(p) for p in path]
    adata = read_directories(
        paths, read=read, profile=profile, params=params, skip_non_exports=skip_non_exports, out=out
    )
    if not adata.n_vars:
        _warn_featureless(root if adata.uns["mantispy"]["resolution"] == "object" else None, None)
    return adata
