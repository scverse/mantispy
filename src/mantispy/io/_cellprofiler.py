"""Join a CellProfiler ``ExportToSpreadsheet`` directory into one table.

The module writes ``Image.csv`` beside one CSV per object, all behind the file-name prefix a run was configured with (``MyExpt_Image.csv``, ``MyExpt_Cells.csv``).
A run set to tab delimiters writes ``.txt`` files instead.
Inside an object table the columns do not carry the object's name, so they are prefixed with it before the objects are joined.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mantispy._core.features import parse_feature_names
from mantispy._core.logging import get_logger
from mantispy._core.schema import make_image_id

_KEYS = ("ImageNumber", "ObjectNumber")
_FILENAME_RE = re.compile(r"^(?:Image_)?FileName_(.+)$")
_NOT_OBJECTS = ("Image", "Experiment")
#: The delimiter ExportToSpreadsheet writes behind each file suffix.
_DELIMITERS = {".csv": ",", ".txt": "\t"}
#: Identity columns the Metadata module writes even when it extracted no value for them.
_IDENTITY = ("Metadata_Plate", "Metadata_Well", "Metadata_Site")


def export_prefix(path: Path) -> str | None:
    """The file-name prefix an ``ExportToSpreadsheet`` run used, or ``None`` if `path` is not such a directory.

    Args:
        path: The directory to look in, which such a run writes ``<prefix>Image.csv`` or, with tab delimiters, ``<prefix>Image.txt`` into.

    Returns:
        The shortest prefix a ``*Image.csv`` or ``*Image.txt`` in `path` carries, which is ``""`` for a run configured without one, and ``None`` when `path` is not a directory or holds no such file.
    """
    if not path.is_dir():
        return None
    files = [file for suffix in _DELIMITERS for file in path.glob(f"*Image{suffix}")]
    prefixes = sorted((file.name.removesuffix(f"Image{file.suffix}") for file in files), key=len)
    return prefixes[0] if prefixes else None


def _suffix(directory: Path, prefix: str) -> str:
    """The suffix of the export's files, ``.csv`` unless the run wrote only a tab-delimited ``Image.txt``."""
    return ".csv" if (directory / f"{prefix}Image.csv").is_file() else ".txt"


def _read_table(path: Path) -> pd.DataFrame:
    """One table of an export, with a repeated header kept once and an imported ``Metadata_ImageNumber`` renamed.

    Asked to add image metadata, ExportToSpreadsheet copies it into each object table and writes a ``Metadata_`` column twice when the table already carried it.
    pandas would read the second copy as ``Metadata_Well.1``, so the copies are compared and one is kept.
    A metadata file imported with an ``ImageNumber`` column of its own arrives as ``Metadata_ImageNumber``, the name CellProfiler's ``ImageNumber`` is read under, so the imported one becomes ``Metadata_ImportedImageNumber``.

    Raises:
        ValueError: Two columns of the same name hold different values.
    """
    delimiter = _DELIMITERS[path.suffix]
    with path.open(newline="") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    frame = pd.read_csv(path, sep=delimiter)
    if len(header) == frame.shape[1]:
        frame.columns = pd.Index(header)
    for name in frame.columns[frame.columns.duplicated()].unique():
        copies = frame.loc[:, frame.columns == name]
        if not all(copies.iloc[:, 0].equals(copies.iloc[:, i]) for i in range(1, copies.shape[1])):
            raise ValueError(f"{path.name} has {copies.shape[1]} columns named {name!r} holding different values")
    frame = frame.loc[:, ~frame.columns.duplicated()]
    if "ImageNumber" in frame and "Metadata_ImageNumber" in frame:
        get_logger().info("%s: kept the imported Metadata_ImageNumber as Metadata_ImportedImageNumber", path.name)
        frame = frame.rename(columns={"Metadata_ImageNumber": "Metadata_ImportedImageNumber"})
    return frame


def _read_image(directory: Path, prefix: str, suffix: str) -> pd.DataFrame:
    """The export's ``Image`` table, without identity columns the Metadata module left empty in every row."""
    image = _read_table(directory / f"{prefix}Image{suffix}")
    if empty := [column for column in _IDENTITY if column in image and image[column].isna().all()]:
        get_logger().info("%s are empty in every image and were left out", empty)
        image = image.drop(columns=empty)
    return image


def object_tables(directory: Path) -> list[str]:
    """The objects an export directory holds a table for, by name; empty for an export of image measurements only."""
    prefix = export_prefix(directory) or ""
    suffix = _suffix(directory, prefix)
    return [
        name
        for path in sorted(directory.glob(f"{prefix}*{suffix}"))
        if (name := path.stem.removeprefix(prefix)) not in _NOT_OBJECTS
    ]


def infer_channels(image: pd.DataFrame, features: Sequence[str] = ()) -> list[str]:
    """Channel names, taken from the features and otherwise from the ``FileName_<channel>`` columns.

    CellProfiler writes one ``Intensity_MeanIntensity_<channel>`` measurement per channel it measured, under the name the features carry.
    The file names can differ from it: a pipeline loads ``OrigDNA``, corrects it with ``IllumDNA`` and saves ``CellOutlines``, and measures the corrected image as ``DNA``.

    Args:
        image: The ``Image.csv`` table of an export.
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
    names = {match.group(1) for match in map(_FILENAME_RE.match, image.columns) if match}
    return sorted({name.removeprefix("Orig").removeprefix("Illum") for name in names if not name.endswith("Outlines")})


def _prefix(frame: pd.DataFrame, obj: str) -> pd.DataFrame:
    """Prefix an object's columns with its name, leaving the join keys and metadata alone."""
    keep = (f"{obj}_", "Metadata_")
    return frame.rename(columns={c: c if c in _KEYS or c.startswith(keep) else f"{obj}_{c}" for c in frame.columns})


def _link_columns(
    primary_frame: pd.DataFrame, child_frame: pd.DataFrame, primary: str, obj: str
) -> tuple[pd.Series, pd.Series]:
    """Locate the parent/child link, which may be on either table.

    CellProfiler writes ``Cells_Parent_Nuclei`` on the primary table when cells were identified from nuclei, and ``Cytoplasm_Parent_Cells`` on the child table for a tertiary object.
    JUMP writes neither for ``Cells`` and ``Nuclei``: only ``Cytoplasm`` has a parent of each.
    The objects already joined onto `primary_frame` are searched for a ``<child>_Parent_<obj>`` column then, which links `obj` to the primary object through that shared child.
    Using the wrong column would silently pair unrelated objects that share an object number.
    """
    on_child = f"{obj}_Parent_{primary}"
    on_primary = f"{primary}_Parent_{obj}"
    if on_child in child_frame.columns:
        return primary_frame["ObjectNumber"], child_frame[on_child]
    if on_primary in primary_frame.columns:
        return primary_frame[on_primary], child_frame["ObjectNumber"]
    via = [c for c in primary_frame.columns if c.endswith(f"_Parent_{obj}")]
    if len(via) == 1:
        return primary_frame[via[0]], child_frame["ObjectNumber"]
    if len(via) > 1:
        raise ValueError(
            f"cannot link {obj!r} to {primary!r}: {via} all point to {obj!r}. Pass objects= to leave one out."
        )
    raise ValueError(
        f"cannot link {obj!r} to {primary!r}: neither {on_child!r} nor {on_primary!r} is present. "
        f"Pass objects= to read only the tables that are related, or set a different primary_object."
    )


def _join_child(merged: pd.DataFrame, child: pd.DataFrame, primary: str, obj: str, strict: bool) -> pd.DataFrame:
    primary_link, child_link = _link_columns(merged, child, primary, obj)

    left = pd.DataFrame({"ImageNumber": merged["ImageNumber"], "_link": primary_link.to_numpy()})
    # Keep the child's own object number as Metadata_<obj>ObjectNumber rather than dropping it: nuclei
    # and cytoplasm need not share the primary object's instance number (spec 5.4).
    # Add _link via concat, not .assign: a single insert onto a wide, many-block child table trips
    # pandas' fragmentation warning, while concat consolidates the blocks first.
    right = pd.concat([child, pd.DataFrame({"_link": child_link.to_numpy()}, index=child.index)], axis=1).rename(
        columns={"ObjectNumber": f"Metadata_{obj}ObjectNumber"}
    )

    counts = right.groupby(["ImageNumber", "_link"], dropna=False).size()
    wanted = pd.MultiIndex.from_frame(left)
    matches = counts.reindex(wanted, fill_value=0).to_numpy()

    if strict and not (matches == 1).all():
        without = int((matches == 0).sum())
        several = int((matches > 1).sum())
        raise ValueError(
            f"join to {obj!r} is not one-to-one with {primary!r}: "
            f"{without} {primary} object(s) have no {obj} and {several} have more than one. "
            "Pass strict_one_to_one=False to keep the first match and leave the rest missing."
        )

    right = right.drop_duplicates(subset=["ImageNumber", "_link"], keep="first")
    joined = left.merge(right, on=["ImageNumber", "_link"], how="left", validate="m:1")
    return joined.drop(columns=["ImageNumber", "_link"])


def read_export(
    directory: Path, primary_object: str, objects: Sequence[str] | None, strict_one_to_one: bool
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Join one export directory into a table with one row per `primary_object`.

    Args:
        directory: A directory an ``ExportToSpreadsheet`` run wrote.
        primary_object: The object one row of the table is.
        objects: Object tables to join onto it, defaulting to every one in the directory.
        strict_one_to_one: Raise when an object does not match the primary object exactly once.

    Returns:
        The joined table, with identifiers, image metadata and the primary object's centroid as ``Metadata_`` columns and every measurement prefixed by its object; the ``Image.csv`` table; and the channels its ``FileName_`` columns name.
        Where an object table and ``Image.csv`` both carry the same ``Metadata_`` column, the ``Image.csv`` value is the one kept.

    Raises:
        FileNotFoundError: There is no table for `primary_object`.
        ValueError: An object cannot be linked to the primary object, or does not match it one to one, or a table repeats a column with different values.
    """
    prefix = export_prefix(directory) or ""
    suffix = _suffix(directory, prefix)
    found = object_tables(directory)
    if primary_object not in found:
        raise FileNotFoundError(f"no {prefix}{primary_object}{suffix} in {directory}; found {found}")
    children = [name for name in found if name != primary_object and (objects is None or name in objects)]

    merged = _prefix(_read_table(directory / f"{prefix}{primary_object}{suffix}"), primary_object)
    n_primary = len(merged)
    for obj in children:
        child = _prefix(_read_table(directory / f"{prefix}{obj}{suffix}"), obj)
        merged = pd.concat([merged, _join_child(merged, child, primary_object, obj, strict_one_to_one)], axis=1)
    if len(merged) != n_primary:
        raise AssertionError("object join changed the row count")

    image = _read_image(directory, prefix, suffix)
    renames = {
        c: "Metadata_" + c.removeprefix("Image_Metadata_") for c in image.columns if c.startswith("Image_Metadata_")
    }
    renames |= {c: c for c in image.columns if c.startswith("Metadata_")}
    per_image = image[["ImageNumber", *renames]].rename(columns=renames)
    # ExportToSpreadsheet can also copy the image metadata into the object tables, and merging both copies would suffix them _x/_y.
    if shared := [c for c in per_image.columns if c != "ImageNumber" and c in merged.columns]:
        get_logger().debug("%s are on both the object tables and Image.csv; keeping the Image.csv value", shared)
        merged = merged.drop(columns=shared)
    table = merged.merge(per_image, on="ImageNumber", how="left", validate="m:1")
    # CellProfiler 4 writes centroids under AreaShape, older versions under Location.
    for axis in ("X", "Y"):
        for source in (f"{primary_object}_Location_Center_{axis}", f"{primary_object}_AreaShape_Center_{axis}"):
            if source in table.columns:
                table[f"Metadata_Center_{axis}"] = table[source].to_numpy()
                break
    table = table.rename(columns={key: f"Metadata_{key}" for key in _KEYS})
    table["Metadata_ObjectType"] = primary_object
    table["Metadata_ImageID"] = make_image_id(table)
    return table, image, infer_channels(image, list(table.columns))


def read_image_export(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read an export that holds image measurements and no object table into a table with one row per image.

    A quality-control pipeline measures whole images and identifies no objects.

    Args:
        directory: A directory an ``ExportToSpreadsheet`` run wrote.

    Returns:
        The image table with its measurements prefixed ``Image_``, ``ImageNumber`` as ``Metadata_ImageNumber`` and a ``Metadata_ImageID``; the table as read; and the channels its ``FileName_`` columns name.

    Raises:
        ValueError: The export holds no image measurement either, only the metadata an input-module pipeline writes.
    """
    prefix = export_prefix(directory) or ""
    image = _read_image(directory, prefix, _suffix(directory, prefix))
    table = _prefix(image, "Image").rename(columns={"ImageNumber": "Metadata_ImageNumber"})
    numeric = [column for column in table.select_dtypes("number").columns if not column.startswith("Metadata_")]
    if not parse_feature_names(numeric)["is_feature"].any():
        raise ValueError(
            f"{directory} holds only image metadata (file names, channels, extracted metadata) and no measurement, "
            "so there is nothing to profile. Read its Image table with pandas to inspect the metadata."
        )
    table["Metadata_ImageID"] = make_image_id(table)
    return table, image, infer_channels(image, list(table.columns))
