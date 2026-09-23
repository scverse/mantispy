"""Rename features to a canonical grammar."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.frames import as_frame
from mantispy._core.mutation import inplace_or_copy

TARGETS = ("cp_measure",)


def _canonical(row: dict[Hashable, Any]) -> str:
    """Rebuild a name from its parsed components, in cp_measure order."""
    parts = [row["object"], row["feature_group"], row["feature"], row["channel"]]
    tokens = [str(part) for part in parts if isinstance(part, str) and part]
    for key in ("params", "radial_bin"):
        if isinstance(row.get(key), str) and row[key]:
            tokens.append(row[key])
    return "_".join(tokens)


@inplace_or_copy()
def standardize_feature_names(adata: AnnData, target: str = "cp_measure", copy: bool = False) -> AnnData | None:
    """Rename features to ``target`` grammar, keeping the original in ``var``.

    CellProfiler feature names differ between versions, so renaming makes a dataset from an older pipeline comparable with one from a newer pipeline.

    Args:
        adata: Object to rename.
        target: Naming grammar. Only ``"cp_measure"`` is supported.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Rewrites ``var_names`` and writes ``var["original_name"]`` on the first call, never overwriting it afterwards, so the incoming names survive repeated calls.

    Raises:
        ValueError: If ``target`` is not supported, or renaming would give two features the same name (for example two Zernike orders).
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    if "original_name" not in adata.var:
        adata.var["original_name"] = adata.var_names.to_numpy()

    # `or name`: empty_annotation marks a column is_feature so the schema is satisfied while leaving
    # every descriptive column empty, and _canonical builds the name out of exactly those. A feature
    # whose annotation names nothing keeps the name it came with rather than becoming "".
    renamed = [
        (_canonical(row) or name) if row["is_feature"] else name
        for name, row in zip(adata.var_names, as_frame(adata.var).to_dict("records"), strict=True)
    ]
    values, counts = np.unique(renamed, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"renaming would make these names collide: {values[counts > 1][:5].tolist()}")

    adata.var_names = pd.Index(renamed)  # type: ignore[assignment]
    return None
