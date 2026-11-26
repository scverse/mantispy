"""Plate and well geometry.

Well names are normalized to the canonical ``A01`` form on read, so every downstream consumer can assume that shape.
Rows are 0-based and support the two-letter names used by 1536-well plates (``A`` -> 0, ``AA`` -> 26, ``AF`` -> 31).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

#: Standard plate formats, mapped to ``(n_rows, n_columns)``.
PLATE_FORMATS: dict[int, tuple[int, int]] = {96: (8, 12), 384: (16, 24), 1536: (32, 48)}

_WELL_RE = re.compile(r"^([A-Za-z]{1,2})[\s_-]?(\d{1,2})$")
#: ``row/column[/field]``, the spelling CZI's optical-pooled-screen releases use (``A/1/0``). The field is a
#: site index, not part of the well, so it is matched and discarded.
_WELL_SLASH_RE = re.compile(r"^([A-Za-z]{1,2})/(\d{1,2})(?:/\d+)?$")
#: A well named by its number alone, as PERISCOPE writes ``Well1``.
_NUMBERED_WELL_RE = re.compile(r"^Well\d+$", re.IGNORECASE)


def normalize_well(well: str) -> str:
    """Return ``well`` in canonical ``A01`` form.

    Args:
        well: A well name in any of the common spellings, e.g. ``A1``, ``a01``, ``B_3``, or the
            ``row/column/field`` form ``A/1/0`` (the field is dropped).

    Returns:
        The well name as an uppercase row label followed by a zero-padded column number.

    Raises:
        ValueError: If ``well`` does not look like a well name.
    """
    text = str(well).strip()
    match = _WELL_RE.match(text) or _WELL_SLASH_RE.match(text)
    if match is None:
        raise ValueError(f"cannot parse well name: {well!r}")
    row, column = match.group(1).upper(), int(match.group(2))
    if column < 1:
        raise ValueError(f"cannot parse well name: {well!r}")
    return f"{row}{column:02d}"


def unplaced_wells(wells: Iterable[str]) -> list[str]:
    """The sorted distinct names in `wells` that name no plate row and column, such as PERISCOPE's ``Well1``."""
    unplaced = set()
    for well in set(wells):
        try:
            normalize_well(well)
        except ValueError:
            unplaced.add(str(well))
    return sorted(unplaced)


def is_numbered_well(well: str) -> bool:
    """Whether `well` names a well by its number alone (``Well1``), which carries no plate position by design."""
    return _NUMBERED_WELL_RE.match(str(well)) is not None


def normalize_wells(wells: pd.Series) -> pd.Series:
    """Each well of `wells` in canonical ``A01`` form, keeping a well that has no row and column as given.

    Inventing a position for a well such as ``Well1`` would make plate-position corrections act on a layout the screen never had.

    Args:
        wells: Well names; missing values stay missing.

    Returns:
        The wells, normalized where they parse.
    """
    names = wells.dropna().astype(str).unique()
    kept = set(unplaced_wells(names))
    mapping = {name: name if name in kept else normalize_well(name) for name in names}
    return wells.astype(object).where(wells.isna(), wells.astype(str).map(mapping))


def well_positions(wells: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """0-based plate row and column of each well in `wells`.

    Raises:
        ValueError: A well names no plate row and column, such as PERISCOPE's ``Well1``; the message lists them.
    """
    wells = list(wells)
    if unplaced := unplaced_wells(wells):
        raise ValueError(
            f"the wells {unplaced[:5]} name no plate row and column, so they have no plate position. "
            "Map them onto plate wells such as 'A01' in obs['Metadata_Well'] first."
        )
    grid = {well: (well_row(well), well_col(well)) for well in set(wells)}
    rows = np.array([grid[well][0] for well in wells], dtype=int)
    columns = np.array([grid[well][1] for well in wells], dtype=int)
    return rows, columns


def well_row(well: str) -> int:
    """0-based row index of ``well`` (``A`` -> 0, ``AA`` -> 26)."""
    letters = normalize_well(well)[:-2]
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - 64)
    return value - 1


def well_col(well: str) -> int:
    """0-based column index of ``well``."""
    return int(normalize_well(well)[-2:]) - 1


def row_label(row: int) -> int | str:
    """Inverse of :func:`well_row`: the letter label for a 0-based row index.

    Rows past ``Z`` get the two-letter names CellProfiler and plate readers use (``AA``, ``AB``, ...), which is what 1536-well plates need.
    """
    if row < 0:
        raise ValueError(f"row index must be non-negative, got {row}")
    if row < 26:
        return chr(65 + row)
    return chr(65 + row // 26 - 1) + chr(65 + row % 26)


def well_name(row: int, column: int) -> str:
    """Canonical well name for a 0-based ``(row, column)`` pair."""
    return f"{row_label(row)}{column + 1:02d}"


def fill_well_grid(obs: pd.DataFrame) -> None:
    """Fill ``Metadata_Row`` and ``Metadata_Col`` (0-based) from ``Metadata_Well``, in place.

    A no-op when ``obs`` has no well column, when the coordinates are already present, or when a well does
    not parse: not every object is well-based, so an unparsable well is skipped rather than raised on.
    """
    if "Metadata_Well" not in obs.columns:
        return
    wells = obs["Metadata_Well"].astype(str)
    try:
        grid = {well: (well_row(well), well_col(well)) for well in wells.unique()}
    except ValueError:
        return
    if "Metadata_Row" not in obs.columns:
        obs["Metadata_Row"] = wells.map(lambda well: grid[well][0]).to_numpy(dtype=int)
    if "Metadata_Col" not in obs.columns:
        obs["Metadata_Col"] = wells.map(lambda well: grid[well][1]).to_numpy(dtype=int)


def detect_plate_format(wells: Iterable[str]) -> int:
    """Smallest standard plate format that contains every well in ``wells``.

    Raises:
        ValueError: If ``wells`` is empty, or if no standard format is large enough.
    """
    wells = list(wells)
    if not wells:
        raise ValueError("no wells given")
    n_rows = max(well_row(well) for well in wells) + 1
    n_cols = max(well_col(well) for well in wells) + 1
    for size in sorted(PLATE_FORMATS):
        max_rows, max_cols = PLATE_FORMATS[size]
        if n_rows <= max_rows and n_cols <= max_cols:
            return size
    raise ValueError(f"no standard plate format holds {n_rows} rows x {n_cols} columns")


def plate_grid(wells: Iterable[str]) -> tuple[int, int]:
    """Rows and columns of the smallest standard format these wells fit, so a plate is drawn on its own grid."""
    return PLATE_FORMATS[detect_plate_format(wells)]
