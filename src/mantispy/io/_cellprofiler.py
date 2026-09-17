"""Join a CellProfiler ``ExportToSpreadsheet`` directory into one table.

The module writes ``Image.csv`` beside one CSV per object, all behind the file-name prefix a run was configured with (``MyExpt_Image.csv``, ``MyExpt_Cells.csv``).
Inside an object table the columns do not carry the object's name, so they are prefixed with it before the objects are joined.
:func:`~mantispy.io.read_profiles` turns the table into AnnData.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mantispy._core.logging import get_logger

_KEYS = ("ImageNumber", "ObjectNumber")
_FILENAME_RE = re.compile(r"^(?:Image_)?FileName_(.+)$")
_NOT_OBJECTS = ("Image", "Experiment")


def export_prefix(path: Path) -> str | None:
    """The file-name prefix an ``ExportToSpreadsheet`` run used, or ``None`` if `path` is not such a directory.

    Args:
        path: The directory to look in, which such a run writes ``<prefix>Image.csv`` into.

    Returns:
        The shortest prefix a ``*Image.csv`` in `path` carries, which is ``""`` for a run configured without one, and ``None`` when `path` is not a directory or holds no such file.
    """
    if not path.is_dir():
        return None
    prefixes = sorted((file.name.removesuffix("Image.csv") for file in path.glob("*Image.csv")), key=len)
    return prefixes[0] if prefixes else None


def infer_channels(image: pd.DataFrame) -> list[str]:
    """Channel names, taken from the ``FileName_<channel>`` columns.

    Args:
        image: The ``Image.csv`` table of an export.

    Returns:
        The channel names its ``FileName_`` or ``Image_FileName_`` columns carry, sorted, and an empty list when it has none, in which case the parser infers the channels from the feature names instead.
        No channel vocabulary is assumed.
    """
    return sorted({match.group(1) for match in map(_FILENAME_RE.match, image.columns) if match})


def _prefix(frame: pd.DataFrame, obj: str) -> pd.DataFrame:
    """Prefix an object's columns with its name, leaving the join keys and metadata alone."""
    keep = (f"{obj}_", "Metadata_")
    return frame.rename(columns={c: c if c in _KEYS or c.startswith(keep) else f"{obj}_{c}" for c in frame.columns})


def _link_columns(
    primary_frame: pd.DataFrame, child_frame: pd.DataFrame, primary: str, obj: str
) -> tuple[pd.Series, pd.Series]:
    """Locate the parent/child link, which may be on either table.

    CellProfiler writes ``Cells_Parent_Nuclei`` on the primary table when cells were identified from nuclei, and ``Cytoplasm_Parent_Cells`` on the child table for a tertiary object.
    Both cases are handled; using the wrong column would silently pair unrelated objects that share an object number.
    """
    on_child = f"{obj}_Parent_{primary}"
    on_primary = f"{primary}_Parent_{obj}"
    if on_child in child_frame.columns:
        return primary_frame["ObjectNumber"], child_frame[on_child]
    if on_primary in primary_frame.columns:
        return primary_frame[on_primary], child_frame["ObjectNumber"]
    raise ValueError(
        f"cannot link {obj!r} to {primary!r}: neither {on_child!r} nor {on_primary!r} is present. "
        f"Pass objects= to read only the tables that are related, or set a different primary_object."
    )


def _join_child(merged: pd.DataFrame, child: pd.DataFrame, primary: str, obj: str, strict: bool) -> pd.DataFrame:
    """Attach one child object's columns to the primary table."""
    primary_link, child_link = _link_columns(merged, child, primary, obj)

    left = pd.DataFrame({"ImageNumber": merged["ImageNumber"], "_link": primary_link.to_numpy()})
    right = child.assign(_link=child_link.to_numpy()).drop(columns=["ObjectNumber"], errors="ignore")

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
        ValueError: An object cannot be linked to the primary object, or does not match it one to one.
    """
    prefix = export_prefix(directory) or ""
    found = [
        name
        for path in sorted(directory.glob(f"{prefix}*.csv"))
        if (name := path.stem.removeprefix(prefix)) not in _NOT_OBJECTS
    ]
    if primary_object not in found:
        raise FileNotFoundError(f"no {prefix}{primary_object}.csv in {directory}; found {found}")
    children = [name for name in found if name != primary_object and (objects is None or name in objects)]

    merged = _prefix(pd.read_csv(directory / f"{prefix}{primary_object}.csv"), primary_object)
    n_primary = len(merged)
    for obj in children:
        child = _prefix(pd.read_csv(directory / f"{prefix}{obj}.csv"), obj)
        merged = pd.concat([merged, _join_child(merged, child, primary_object, obj, strict_one_to_one)], axis=1)
    if len(merged) != n_primary:
        raise AssertionError("object join changed the row count")

    image = pd.read_csv(directory / f"{prefix}Image.csv")
    renames = {
        c: "Metadata_" + c.removeprefix("Image_Metadata_") for c in image.columns if c.startswith("Image_Metadata_")
    }
    renames |= {c: c for c in image.columns if c.startswith("Metadata_")}
    per_image = image[["ImageNumber", *renames]].rename(columns=renames)
    # ExportToSpreadsheet can copy the image metadata into the object tables as well.
    # Merging both copies would suffix the pair Metadata_Plate_x/_y and leave the schema's own columns missing, so the object table's copy goes and the per-image value stays.
    if shared := [c for c in per_image.columns if c != "ImageNumber" and c in merged.columns]:
        get_logger().info("%s are on both the object tables and Image.csv; keeping the Image.csv value", shared)
        merged = merged.drop(columns=shared)
    table = merged.merge(per_image, on="ImageNumber", how="left", validate="m:1")
    # Centroids are not profile features, but qc_is_border needs them.
    for axis in ("X", "Y"):
        if (source := f"{primary_object}_Location_Center_{axis}") in table.columns:
            table[f"Metadata_Center_{axis}"] = table[source].to_numpy()
    return table.rename(columns={key: f"Metadata_{key}" for key in _KEYS}), image, infer_channels(image)
