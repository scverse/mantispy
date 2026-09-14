from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

IMAGE = "Image"
EXPERIMENT = "Experiment"
KEYS = ("ImageNumber", "ObjectNumber")
QUALITY_PREFIX = "ImageQuality_"


def export_prefix(path: Path) -> str | None:
    """The file-name prefix an ``ExportToSpreadsheet`` run used, or ``None`` if `path` is not one.

    The module writes ``Image.csv`` beside one CSV per object, optionally behind a prefix the user chose,
    so a run configured with the usual ``MyExpt_`` writes ``MyExpt_Image.csv`` and ``MyExpt_Cells.csv``.
    """
    if not path.is_dir():
        return None
    prefixes = sorted((file.name[: -len(f"{IMAGE}.csv")] for file in path.glob(f"*{IMAGE}.csv")), key=len)
    return prefixes[0] if prefixes else None


def is_export_directory(path: Path) -> bool:
    """Whether `path` is a directory an ``ExportToSpreadsheet`` run wrote."""
    return export_prefix(path) is not None


def _object_files(directory: Path, prefix: str, objects: Sequence[str] | None) -> dict[str, Path]:
    found = {
        name: path
        for path in sorted(directory.glob(f"{prefix}*.csv"))
        if (name := path.stem.removeprefix(prefix)) not in (IMAGE, EXPERIMENT)
    }
    if objects is None:
        return found
    if missing := [name for name in objects if name not in found]:
        msg = f"no {missing} under {directory}; found {sorted(found)}"
        raise FileNotFoundError(msg)
    return {name: found[name] for name in objects}


def _prefixed(frame: pd.DataFrame, obj: str) -> pd.DataFrame:
    keep = [c for c in frame.columns if not c.startswith("Parent_")]
    return frame[keep].rename(columns=lambda c: c if c in KEYS else f"{obj}_{c}" if not c.startswith(f"{obj}_") else c)


def _join(
    primary: pd.DataFrame, links: pd.DataFrame, child: pd.DataFrame, name: str, other: str
) -> pd.DataFrame | None:
    """Join `child` onto `primary` through whichever of the two records the other as its parent."""
    if f"Parent_{other}" in links.columns:
        link = links[f"Parent_{other}"].to_numpy()
    elif f"Parent_{name}" in child.columns:
        link = primary["ObjectNumber"].to_numpy()
        child = child.assign(ObjectNumber=child[f"Parent_{name}"])
    else:
        return None
    right = _prefixed(child, other).rename(columns={"ObjectNumber": "_link"})
    right = right.drop_duplicates(subset=["ImageNumber", "_link"])
    keyed = pd.concat([primary, pd.Series(link, index=primary.index, name="_link")], axis=1)
    merged = keyed.merge(right, on=["ImageNumber", "_link"], how="left")
    return merged.drop(columns="_link")


def read_export_directory(
    directory: Path,
    *,
    primary_object: str = "Cells",
    objects: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one ``ExportToSpreadsheet`` directory into a per-object frame and an image-quality frame.

    Rows are the objects of `primary_object`.
    Every other object table is joined onto it through the ``Parent_`` column that links the two, and its
    columns are prefixed with the object name.
    ``Image.csv`` contributes its ``Metadata_`` columns, joined on ``ImageNumber``, and its ``ImageQuality_``
    columns are held back so that they never reach the feature matrix.

    Args:
        directory: A directory holding ``Image.csv`` and one CSV per object, behind the prefix the run used.
        primary_object: The object whose rows become observations.
        objects: Which object tables to read, defaulting to every one beside ``Image.csv``.

    Returns:
        The joined objects, and the image-quality measurements indexed by ``ImageNumber``.

    Raises:
        FileNotFoundError: `directory` is not an export, or a named object is not in it.
    """
    prefix = export_prefix(directory)
    if prefix is None:
        msg = f"{directory} holds no {IMAGE}.csv, so it is not an ExportToSpreadsheet directory"
        raise FileNotFoundError(msg)
    files = _object_files(directory, prefix, objects)
    if primary_object not in files:
        msg = f"no {prefix}{primary_object}.csv under {directory}; found {sorted(files)}"
        raise FileNotFoundError(msg)

    frames = {name: pd.read_csv(path) for name, path in files.items()}
    links = frames.pop(primary_object)
    joined = _prefixed(links, primary_object)
    for name, frame in frames.items():
        if (merged := _join(joined, links, frame, primary_object, name)) is not None:
            joined = merged

    image = pd.read_csv(directory / f"{prefix}{IMAGE}.csv")
    quality = [c for c in image.columns if c.startswith(QUALITY_PREFIX)]
    metadata = [c for c in image.columns if c.startswith("Metadata_")]
    joined = joined.merge(image[["ImageNumber", *metadata]], on="ImageNumber", how="left")
    joined = joined.rename(columns={key: f"Metadata_{key}" for key in KEYS})
    return joined.copy(), image[["ImageNumber", *quality]].set_index("ImageNumber")
