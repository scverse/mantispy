"""Plate and well geometry.

Well names are normalized to the canonical ``A01`` form on read, so every downstream consumer can assume that shape.
Rows are 0-based and support the two-letter names used by 1536-well plates (``A`` -> 0, ``AA`` -> 26, ``AF`` -> 31).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Standard plate formats, mapped to ``(n_rows, n_columns)``.
PLATE_FORMATS: dict[int, tuple[int, int]] = {96: (8, 12), 384: (16, 24), 1536: (32, 48)}

_WELL_RE = re.compile(r"^([A-Za-z]{1,2})[\s_-]?(\d{1,2})$")


def normalize_well(well: str) -> str:
    """Return ``well`` in canonical ``A01`` form.

    Args:
        well: A well name in any of the common spellings, e.g. ``A1``, ``a01``, ``B_3``.

    Returns:
        The well name as an uppercase row label followed by a zero-padded column number.

    Raises:
        ValueError: If ``well`` does not look like a well name.
    """
    match = _WELL_RE.match(str(well).strip())
    if match is None:
        raise ValueError(f"cannot parse well name: {well!r}")
    row, column = match.group(1).upper(), int(match.group(2))
    if column < 1:
        raise ValueError(f"cannot parse well name: {well!r}")
    return f"{row}{column:02d}"


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
